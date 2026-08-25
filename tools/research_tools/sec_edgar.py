import logging
import re
import html as _html
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

EDGAR_BASE   = "https://www.sec.gov"
EDGAR_DATA   = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
HEADERS      = {"User-Agent": "AlphaAgentNode research@alpha-agent.ai"}

_CIK_CACHE: dict[str, str] = {}
_MAX_QUERY_TERMS = 6


def _sanitize_query(query: str) -> str:
    """
    Trim an LLM-generated search query down to a small set of terms that
    SEC's full-text search backend can reliably handle.

    - Strips quote characters (already done by caller, kept here for safety).
    - Drops bare 4-digit year tokens (e.g. "2025", "2026") — EDGAR full-text
      search indexes filing dates separately; embedding years as keywords
      doesn't help relevance and just adds noise/term-count.
    - Caps the remaining terms to _MAX_QUERY_TERMS, preserving order so the
      most important (usually earliest) words from the LLM's query survive.
    """
    terms = [t for t in query.replace('"', '').split() if not re.fullmatch(r"(19|20)\d{2}", t)]
    if len(terms) > _MAX_QUERY_TERMS:
        logger.warning(
            "sec_edgar_search: query had %d terms after cleanup; truncating to %d. "
            "original=%r",
            len(terms), _MAX_QUERY_TERMS, query,
        )
        terms = terms[:_MAX_QUERY_TERMS]
    return " ".join(terms)


# ─────────────────────────────────────────────
# Tool 1: Search filings by keyword / ticker
# ─────────────────────────────────────────────
async def sec_edgar_search(
    query: str,
    ticker: Optional[str] = None,
    form_type: Optional[str] = None,   # "10-K" | "10-Q" | "8-K"
    max_results: int = 5,
) -> dict:
    """
    Full-text search across SEC EDGAR filings.
    Returns: list of filings with company, form_type, filed_at, accession_number, url.

    Gracefully degrades on upstream failure: returns
    ``{"query": query, "filings": [], "error": "..."}`` instead of raising,
    so a flaky/overloaded SEC endpoint never crashes the calling agent.
    """
    clean_query = _sanitize_query(query)
    params: dict = {"q": clean_query}
    if form_type:
        params["forms"] = form_type
    if ticker:
        try:
            cik = await _resolve_cik(ticker)
        except Exception as exc:
         
            logger.warning(
                "sec_edgar_search: CIK resolution failed for ticker=%r: %s",
                ticker, exc,
            )
            return {
                "query": query, "filings": [],
                "error": f"CIK resolution failed for {ticker!r}: {exc}",
            }
        if cik is None:
            logger.warning(
                "sec_edgar_search: no CIK found for ticker=%r — refusing to run "
                "an unscoped full-text search.", ticker,
            )
            return {
                "query": query, "filings": [],
                "error": f"CIK not found for ticker {ticker!r}; search aborted "
                         f"to avoid returning unrelated results from all filers.",
            }
        params["ciks"] = cik.zfill(10)

    try:
        async with httpx.AsyncClient(timeout=20.0, headers=HEADERS) as client:
            resp = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "sec_edgar_search: SEC EDGAR returned %s for query=%r (sanitized=%r): %s",
            exc.response.status_code, query, clean_query, exc,
        )
        return {"query": query, "filings": [], "error": f"SEC EDGAR {exc.response.status_code}: {exc}"}
    except httpx.HTTPError as exc:
        logger.warning("sec_edgar_search: network error for query=%r: %s", query, exc)
        return {"query": query, "filings": [], "error": f"network error: {exc}"}

    filings = []
    for hit in data.get("hits", {}).get("hits", [])[:max_results]:
        src = hit.get("_source", {})

        display_names = src.get("display_names") or []
        root_forms    = src.get("root_forms") or []

        filings.append({
            "company":          display_names[0] if display_names else "",
            "form_type":        root_forms[0] if root_forms else "",
            "filed_at":         src.get("file_date", ""),
            "accession_number": src.get("adsh", ""),
            "period":           src.get("period_ending", ""),
        })

    return {"query": query, "filings": filings}

_HEADING_PATTERN = re.compile(
    r"(?:(?<=[.?!]\s)|^)((?:[A-Z][A-Z&,'\u2019\-]{1,}\s+){1,7}[A-Z][A-Z&,'\u2019\-]{1,})(?=\s+[A-Z][a-z])"
)
_MAX_HEADINGS = 15


def _find_headings(body: str) -> list:
    seen = []
    for m in _HEADING_PATTERN.finditer(body):
        h = m.group(1).strip()
        if h.startswith("ITEM ") or h in seen or m.start() <= 25:
            continue
        seen.append(h)
        if len(seen) >= _MAX_HEADINGS:
            break
    return seen


def _query_snippet(body: str, query: str, window: int) -> dict:
    pos = body.lower().find(query.lower())
    if pos == -1:
        return {"found": False}
    start = max(0, pos - window // 2)
    end = min(len(body), pos + len(query) + window // 2)
    return {
        "found": True,
        "position": pos,
        "snippet": body[start:end],
        "snippet_range": [start, end],
    }


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_LEAD_SENTENCE_CHARS = 220
_MAX_LEADS = 10

_NUMERIC_HINT = re.compile(r"(\$\s?[\d,]+|\d+(?:\.\d+)?%|\bmillion\b|\bbillion\b)", re.I)
_MAX_EXCERPTS = 6
_EXCERPT_CHARS = 220


def _important_excerpts(body: str) -> list:
    out = []
    seen = set()
    for s in _SENTENCE_END.split(body):
        s = s.strip()
        if len(s) < 20 or not _NUMERIC_HINT.search(s):
            continue
        snippet = s[:_EXCERPT_CHARS]
        if snippet in seen:
            continue
        seen.add(snippet)
        out.append(snippet)
        if len(out) >= _MAX_EXCERPTS:
            break
    return out


def _heading_leads(body: str) -> list:
    out = []
    for h in _find_headings(body):
        pos = body.find(h)
        if pos == -1:
            continue
        after = body[pos + len(h):].strip()
        lead = _SENTENCE_END.split(after, maxsplit=1)[0][:_LEAD_SENTENCE_CHARS].strip()
        out.append({"heading": h, "lead": lead})
        if len(out) >= _MAX_LEADS:
            break
    return out

_KEY_FIGURE_LABELS = [
    "Total revenue",
    "Total cost of revenue",
    "Gross margin",
    "Research and development",
    "Sales and marketing",
    "General and administrative",
    "Operating income",
    "Income before income taxes",
    "Provision for income taxes",
    "Net income",
    "Net cash from operations",
    "Additions to property and equipment",
    "Net cash used in investing",
]
_NUM_TOKEN = r"\$?\s*\(?-?[\d,]+(?:\.\d+)?\s*\)?"


def _extract_key_figures(body: str) -> dict:
    out = {}
    for label in _KEY_FIGURE_LABELS:
        pat = re.compile(re.escape(label) + r"\s+((?:" + _NUM_TOKEN + r"\s*){1,4})")
        m = pat.search(body)
        if m:
            nums = re.findall(_NUM_TOKEN, m.group(1))
            out[label] = [n.strip() for n in nums]
    return out


# ─────────────────────────────────────────────
# Tool 2: Fetch and parse a specific filing
# ─────────────────────────────────────────────
async def sec_edgar_filing(
    ticker: str,
    form_type: Optional[str] = None,
    sections: list = None,             # e.g. ["risk_factors","mda"] -- REQUIRED, no silent "all"
    mode: str = "preview",             # "preview" | "full"
    preview_chars: int = 2000,
    query: str = None,                 # find this text inside each requested section
    query_window: int = 3000,          # chars of context around the query match
    offset: int = 0,                   # mode="full" only: start position within the section
    length: int = None,                # mode="full" only: chars to return (default: max_chars)
    max_chars: int = 25000,            # hard ceiling on how much of a section is ever parsed out
    accession_number: Optional[str] = None,
) -> dict:
    """
    Fetch an SEC filing (10-K / 10-Q) for a ticker and return one or more
    named sections: business, risk_factors, mda, financial_statements. By
    default the latest filing of ``form_type`` (10-K when omitted) is used.
    When ``accession_number`` is provided, that exact filing is resolved from
    the ticker's SEC submissions metadata instead.

    `sections` must be given explicitly and must name real sections -- there
    is no "give me everything" default. A single filing's sections can run
    to tens of thousands of tokens each; forcing an explicit, scoped request
    keeps a single tool call from flooding an agent's context.

    Two response shapes depending on `mode`:

    mode="preview" (default) -- for each section, returns `total_chars`,
    a short `preview` (first `preview_chars`), a best-effort list of
    `headings`, and one of:
      - `key_figures` (financial_statements only) -- standard income-
        statement line items (Total revenue, Net income, ...) pulled out
        character-for-character, never rounded or reparsed.
      - `digest` (all other sections) -- each detected sub-heading paired
        with the sentence right after it, a structural TL;DR of the whole
        section in a few hundred characters.
      - `important_excerpts` (all other sections) -- up to 6 sentences
        containing a dollar figure, a percentage, or "million"/"billion",
        pulled out verbatim. Headings' lead sentences often miss the
        actual number (it tends to land a sentence or two later); this
        catches it without re-reading the section.
    Use these to decide whether -- and where -- to read further, without
    paying for the full text up front.

    mode="full" -- returns the actual section text, sliced with
    `offset`/`length` (default: from 0 up to `max_chars`). Use this once
    you already know (from a preview, or from `query`) which section you
    need in full.

    `query`, if given, is a plain substring search (case-insensitive) run
    against each section's full text regardless of `mode`; returns whether
    it was found, its position, and a `query_window`-sized snippet around
    it. This is the intended way to jump to a specific topic ("cash flow
    outlook", "cybersecurity", ...) instead of guessing a char offset.
    """
    if not sections:
        return {
            "ticker": ticker,
            "error": (
                "`sections` is required -- name the specific section(s) you need "
                "(e.g. ['risk_factors'] or ['mda','financial_statements']). "
                "There's no default 'all' anymore: a single filing's sections "
                "can run to tens of thousands of tokens each, so an unscoped "
                "request isn't allowed. Valid names for a 10-K: business, "
                "risk_factors, mda, financial_statements. A 10-Q has no "
                "'business' section."
            ),
        }
    if mode not in ("preview", "full"):
        return {"ticker": ticker, "error": f"mode must be 'preview' or 'full', got {mode!r}"}

    normalized_accession = None
    if accession_number is not None:
        normalized_accession = _normalize_accession_number(accession_number)
        if normalized_accession is None:
            return {
                "ticker": ticker,
                "accession_number": accession_number,
                "error": (
                    f"Invalid SEC accession number {accession_number!r}; expected "
                    "18 digits, with or without dashes (for example "
                    "0001045810-25-000023)."
                ),
            }

    try:
        cik = await _resolve_cik(ticker)
    except Exception as exc:
        result = {"ticker": ticker, "error": f"CIK resolution failed for {ticker}: {exc}"}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result
    if not cik:
        result = {"ticker": ticker, "error": f"CIK not found for {ticker}"}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result

    if normalized_accession and normalized_accession[:10] != cik.zfill(10):
        return {
            "ticker": ticker,
            "accession_number": normalized_accession,
            "error": (
                f"Accession number {normalized_accession} does not belong to "
                f"ticker {ticker}."
            ),
        }

    try:
        submissions = await _get_submissions(cik)
    except Exception as exc:
        result = {"ticker": ticker, "error": f"Failed to fetch EDGAR submissions: {exc}"}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result
    if not submissions:
        result = {"ticker": ticker, "error": "Failed to fetch EDGAR submissions"}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result

    if normalized_accession:
        try:
            filing_meta = await _find_by_accession(submissions, normalized_accession)
        except Exception as exc:
            return {
                "ticker": ticker,
                "accession_number": normalized_accession,
                "error": f"Failed to resolve accession number from EDGAR submissions: {exc}",
            }
        if not filing_meta:
            return {
                "ticker": ticker,
                "accession_number": normalized_accession,
                "error": (
                    f"Accession number {normalized_accession} was not found in "
                    f"EDGAR submissions for ticker {ticker}."
                ),
            }
        actual_form_type = filing_meta.get("form", "")
        if not actual_form_type:
            return {
                "ticker": ticker,
                "accession_number": normalized_accession,
                "error": (
                    f"EDGAR submissions metadata for accession {normalized_accession} "
                    "does not include its form type."
                ),
            }
        if form_type and actual_form_type and form_type.upper() != actual_form_type.upper():
            return {
                "ticker": ticker,
                "accession_number": normalized_accession,
                "error": (
                    f"Requested form type {form_type} conflicts with accession "
                    f"{normalized_accession}, which is a {actual_form_type} filing."
                ),
            }
    else:
        actual_form_type = form_type or "10-K"
        filing_meta = _find_latest(submissions, actual_form_type)
        if not filing_meta:
            return {"ticker": ticker, "error": f"No {actual_form_type} found"}

    acc_dash   = filing_meta["accessionNumber"]          # e.g. 0000320193-26-000013
    acc_nodash = acc_dash.replace("-", "")               # e.g. 000032019326000013
    try:
        raw_text = await _fetch_text(cik, acc_dash, acc_nodash)
    except Exception as exc:
        return {
            "ticker": ticker,
            "accession_number": acc_dash,
            "error": f"Failed to fetch filing text for accession {acc_dash}: {exc}",
        }
    if not raw_text:
        return {
            "ticker": ticker,
            "accession_number": acc_dash,
            "error": f"Failed to fetch filing text for accession {acc_dash}.",
        }
    
    _EXTRACTION_SAFETY_CAP = 750_000  # generous ceiling against pathological input, not a response cap
    parsed = _extract_sections(raw_text, sections, _EXTRACTION_SAFETY_CAP, actual_form_type)

    sections_out = {}
    for name, body in parsed.items():
        entry = {"total_chars": len(body)}

        if mode == "preview":
            entry["preview"] = body[:preview_chars]
            entry["preview_chars_returned"] = min(preview_chars, len(body))
            entry["headings"] = _find_headings(body)
            if name == "financial_statements":
                
                entry["key_figures"] = _extract_key_figures(body)
            else:
                
                entry["digest"] = _heading_leads(body)
                
                entry["important_excerpts"] = _important_excerpts(body)
        else:  # mode == "full"
            take = length if length is not None else max_chars
            entry["text"] = body[offset:offset + take]
            entry["offset"] = offset
            entry["returned_chars"] = len(entry["text"])

        if query:
            entry["query_match"] = _query_snippet(body, query, query_window)

        sections_out[name] = entry

    return {
        "ticker":           ticker,
        "company":          submissions.get("name", ""),
        "form_type":        actual_form_type,
        "filed_at":         filing_meta.get("filingDate", ""),
        "accession_number": filing_meta.get("accessionNumber", ""),
        "mode":             mode,
        "sections":         sections_out,
    }



# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_TICKER_TO_CIK: dict[str, str] | None = None   # lazily loaded, shared across calls


async def _resolve_cik(ticker: str) -> Optional[str]:
    """
    Resolve a stock ticker to its zero-unpadded SEC CIK using the official
    ``company_tickers.json`` index (the same source used by
    financial_tools/sec_edgar.py's get_cik()), instead of scraping the
    ``browse-edgar`` HTML page with a regex.

    The previous implementation searched an atom/HTML page for the FIRST
    occurrence of ``CIK=(\\d+)``, which is fragile in two ways:
      1. It raised uncaught exceptions on network failure (no try/except),
         so a transient error propagated up as an unhandled exception.
      2. The page can contain multiple "CIK=" occurrences (pagination links,
         related-filer links), so the first match is not guaranteed to be
         the queried ticker's own CIK.
    Both failure modes result in ``_resolve_cik`` returning ``None`` or a
    wrong CIK, which upstream causes ``sec_edgar_search`` to silently drop
    the ``ciks`` filter and search the ENTIRE EDGAR corpus unscoped —
    exactly the behaviour that produced unrelated 2010–2013 filings in
    place of the requested ticker's filings.

    company_tickers.json is a simple, official, complete ticker→CIK map
    published by SEC itself, so a single exact-match lookup replaces the
    scraping heuristic entirely and can raise/return None explicitly on
    failure instead of failing silently.
    """
    global _TICKER_TO_CIK

    t = ticker.upper().strip()
    if t in _CIK_CACHE:
        return _CIK_CACHE[t]

    if _TICKER_TO_CIK is None:
        async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
            resp = await client.get(_TICKER_MAP_URL)
            resp.raise_for_status()
            raw = resp.json()
        # raw is a dict keyed by stringified row index: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        _TICKER_TO_CIK = {
            row["ticker"].upper(): str(row["cik_str"])
            for row in raw.values()
        }
        logger.info("sec_edgar: loaded %d tickers from company_tickers.json", len(_TICKER_TO_CIK))

    cik = _TICKER_TO_CIK.get(t)

    if cik is None and ("." in t or "-" in t):
        alt = t.replace(".", "-") if "." in t else t.replace("-", ".")
        cik = _TICKER_TO_CIK.get(alt)
    if cik is not None:
        _CIK_CACHE[t] = cik
    return cik


async def _get_submissions(cik: str) -> Optional[dict]:
    url = EDGAR_DATA.format(cik=cik.zfill(10))
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def _get_submission_file(name: str) -> dict:
    url = EDGAR_SUBMISSIONS.format(name=name)
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


def _normalize_accession_number(accession_number: str) -> Optional[str]:
    compact = str(accession_number).strip().replace("-", "")
    if not re.fullmatch(r"\d{18}", compact):
        return None
    return f"{compact[:10]}-{compact[10:12]}-{compact[12:]}"


def _find_accession_in_rows(rows: dict, accession_number: str) -> Optional[dict]:
    accessions = rows.get("accessionNumber", [])
    forms = rows.get("form", [])
    dates = rows.get("filingDate", [])
    for index, candidate in enumerate(accessions):
        if _normalize_accession_number(str(candidate)) == accession_number:
            return {
                "accessionNumber": accession_number,
                "filingDate": dates[index] if index < len(dates) else "",
                "form": forms[index] if index < len(forms) else "",
            }
    return None


async def _find_by_accession(
    submissions: dict,
    accession_number: str,
) -> Optional[dict]:
    filings = submissions.get("filings", {})
    filing_meta = _find_accession_in_rows(filings.get("recent", {}), accession_number)
    if filing_meta:
        return filing_meta

    for submission_file in filings.get("files", []):
        name = submission_file.get("name")
        if not name:
            continue
        historical_rows = await _get_submission_file(name)
        filing_meta = _find_accession_in_rows(historical_rows, accession_number)
        if filing_meta:
            return filing_meta
    return None


def _find_latest(submissions: dict, form_type: str) -> Optional[dict]:
    recent  = submissions.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    acc_nos = recent.get("accessionNumber", [])
    dates   = recent.get("filingDate", [])
    for i, f in enumerate(forms):
        if f == form_type:
            return {"accessionNumber": acc_nos[i], "filingDate": dates[i]}
    return None


async def _fetch_text(cik: str, acc_dash: str, acc_nodash: str) -> str:

    cik_int = str(int(cik))
    url = f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc_dash}.txt"
    async with httpx.AsyncClient(timeout=60.0, headers=HEADERS, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.text
    logger.warning("sec_edgar_filing: filing text fetch returned %s for %s",
                   resp.status_code, url)
    return ""


def _html_to_text(raw: str) -> str:
    """
    Convert a raw SEC submission (HTML/SGML) into reasonably clean plain text
    so the item-header regexes can match. Without this, tags and HTML entities
    sit between e.g. 'Item 2.' and 'Management', breaking contiguous matches.
    """
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)</?(p|div|tr|td|th|table|br|hr|h[1-6]|li|ul|ol)[^>]*>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", "", raw)          # drop remaining inline tags, no space
    raw = _html.unescape(raw)                      # &#160; &amp; &#8217; ...
    raw = raw.replace("\u00a0", " ")               # non-breaking space
    raw = re.sub(r"\s+", " ", raw)
    return raw

_SECTION_PATTERNS = {
    "10-K": {
        "business":             r"item\s+1[.\s]+business",
        "risk_factors":         r"item\s+1a[.\s]+risk\s+factors",
        "mda":                  r"item\s+7[.\s]+management",
        "financial_statements": r"item\s+8[.\s]+financial\s+statements",
    },
    "10-Q": {
        "financial_statements": r"item\s+1[.\s]+financial\s+statements",
        "mda":                  r"item\s+2[.\s]+management",
        "risk_factors":         r"item\s+1a[.\s]+risk\s+factors",
    },
}

MIN_SECTION_CHARS = 800

_TRAILING_PART_MARKER = re.compile(
    r"(?i)\s*part\s+[ivx]+\.?(?:\s+[a-z][a-z'\-]*){0,4}\s*$"
)

_ITEM_BOUNDARY_PATTERN = re.compile(r"\bITEM\s+\d{1,2}[A-Z]?\.")


def _extract_sections(text: str, sections: list, max_chars: int,
                      form_type: str = "10-K") -> dict:
    if not text:
        return {}

    patterns = _SECTION_PATTERNS.get(form_type.upper(), _SECTION_PATTERNS["10-K"])
    clean    = _html_to_text(text)
    text_low = clean.lower()
    want_all = "all" in sections

    all_matches: dict[str, list[int]] = {}
    for name, pat in patterns.items():
        starts = [m.start() for m in re.finditer(pat, text_low)]
        if starts:
            all_matches[name] = starts

    merged_boundaries = sorted(
        {pos for starts in all_matches.values() for pos in starts}
        | {m.start() for m in _ITEM_BOUNDARY_PATTERN.finditer(clean)}
    )

    def _end_boundary_after(start: int) -> int:
        """First merged boundary strictly after `start`, else end of text.
        This is intentionally computed against the FULL merged list (all
        occurrences of all headers), not just each header's own last
        occurrence — a candidate near the table of contents should be
        bounded by the very next heading-like text, wherever it's from."""
        for pos in merged_boundaries:
            if pos > start:
                return pos
        return len(clean)
    
    chosen_start: dict[str, int] = {}
    for name, starts in all_matches.items():
        if not want_all and name not in sections:
            continue
        best_start = starts[-1]  # fallback: old "last occurrence" behavior
        for candidate in reversed(starts):
            span_len = _end_boundary_after(candidate) - candidate
            if span_len >= MIN_SECTION_CHARS:
                best_start = candidate
                break
        chosen_start[name] = best_start

    sorted_pos = sorted(chosen_start.items(), key=lambda x: x[1])

    result = {}
    for idx, (name, start) in enumerate(sorted_pos):
        next_requested = sorted_pos[idx + 1][1] if idx + 1 < len(sorted_pos) else len(clean)
        end = min(next_requested, _end_boundary_after(start))
        body = clean[start:end].strip()
        body = _TRAILING_PART_MARKER.sub("", body).strip()
        result[name] = body[:max_chars]

    return result
