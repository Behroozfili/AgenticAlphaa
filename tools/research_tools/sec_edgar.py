import logging
import re
import html as _html
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

EDGAR_BASE   = "https://www.sec.gov"
EDGAR_DATA   = "https://data.sec.gov/submissions/CIK{cik}.json"
HEADERS      = {"User-Agent": "AlphaAgentNode research@alpha-agent.ai"}

_CIK_CACHE: dict[str, str] = {}

# SEC's full-text search backend (efts.sec.gov) becomes unreliable —
# and frequently returns 500 — once a query string accumulates too many
# ANDed terms (we've observed failures around 10+ words). Brain-generated
# queries tend to pile on synonyms/dates ("Apple AI capex spending Q4 2025
# earnings capital expenditure infrastructure investment June 2026"), so
# we defensively cap the term count before sending the request.
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
            # CIK resolution failing must NEVER silently widen the search to
            # the entire EDGAR corpus — log it and surface it to the caller
            # so an unfiltered, ticker-less query is never mistaken for a
            # ticker-scoped one.
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

        # SEC's efts.sec.gov full-text search index does NOT use the field
        # names this code previously assumed ("entity_name", "form_type",
        # "accession_no", "period_of_report"). Those keys don't exist in the
        # real response, so every filing silently came back with empty
        # strings for company/form_type/accession_number/period — the only
        # field that was ever populated was "filed_at" (file_date), which is
        # why prior traces showed filings with real dates but blank
        # everything else. The actual field names are:
        #   display_names      list[str]  e.g. ["APPLE INC (0000320193)"]
        #   root_forms         list[str]  e.g. ["10-Q"]
        #   adsh               str        e.g. "0000320193-26-000013"
        #   period_ending      str        e.g. "2026-03-28"
        #   file_date          str        e.g. "2026-05-01"
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


# ─────────────────────────────────────────────
# Tool 2: Fetch and parse a specific filing
# ─────────────────────────────────────────────
async def sec_edgar_filing(
    ticker: str,
    form_type: str = "10-K",
    sections: list = None,             # ["business","risk_factors","mda","all"]
    max_chars: int = 25000,
) -> dict:
    """
    Fetch the latest SEC filing (10-K / 10-Q) for a ticker.
    Parses and returns named sections: business, risk_factors, mda, financial_statements.
    """
    if sections is None:
        sections = ["all"]

    cik = await _resolve_cik(ticker)
    if not cik:
        return {"ticker": ticker, "error": f"CIK not found for {ticker}"}

    submissions = await _get_submissions(cik)
    if not submissions:
        return {"ticker": ticker, "error": "Failed to fetch EDGAR submissions"}

    filing_meta = _find_latest(submissions, form_type)
    if not filing_meta:
        return {"ticker": ticker, "error": f"No {form_type} found"}

    acc_dash   = filing_meta["accessionNumber"]          # e.g. 0000320193-26-000013
    acc_nodash = acc_dash.replace("-", "")               # e.g. 000032019326000013
    raw_text   = await _fetch_text(cik, acc_dash, acc_nodash)
    parsed     = _extract_sections(raw_text, sections, max_chars, form_type)

    return {
        "ticker":           ticker,
        "company":          submissions.get("name", ""),
        "form_type":        form_type,
        "filed_at":         filing_meta.get("filingDate", ""),
        "accession_number": filing_meta.get("accessionNumber", ""),
        "sections":         parsed,
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

    # Multi-class share tickers (e.g. Berkshire Hathaway) are sometimes typed
    # with a dot ("BRK.B") and sometimes stored/typed with a hyphen ("BRK-B").
    # An LLM-generated ticker directive could use either form, so try the
    # other separator before giving up — this is generic to ANY dotted/
    # hyphenated ticker, not special-cased to one company.
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
    # EDGAR's Archives path uses the un-padded integer CIK, an accession-number
    # *directory* WITHOUT dashes, and a full-submission *file* WITH dashes.
    # Getting either wrong yields a 404 and silently empty section output.
    #
    # No truncation on the response text. The full-submission .txt file
    # concatenates the main document with exhibits, the XBRL instance
    # document, and other attachments — for a large filer (e.g. MSFT) this
    # exceeds even 5 MB. Two earlier fixed caps (2M, then 5M chars) both
    # cut the raw text off partway through the document's own table of
    # contents / cover matter, before the real Item 1/2/1A narrative body
    # was ever reached — confirmed via a direct diagnostic fetch each time
    # (raw text hit the cap exactly, and the "mda" section extracted
    # afterward was provably just the table of contents, not the actual
    # MD&A body). This does NOT increase LLM token cost: every returned
    # section is separately capped by max_chars (default 25000 chars) in
    # _extract_sections below before it ever reaches an LLM — the only
    # real cost of fetching more here is a bit more network/memory for
    # this one request, which is negligible for a single filing document.
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
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)        # drop all remaining tags
    raw = _html.unescape(raw)                      # &#160; &amp; &#8217; ...
    raw = raw.replace("\u00a0", " ")               # non-breaking space
    raw = re.sub(r"[ \t]+", " ", raw)
    return raw


# Item-header numbering differs by form type. A 10-Q's MD&A is "Item 2" and
# its financial statements are "Item 1" (and it has no standalone "Business"
# section), whereas a 10-K uses Item 1 / 1A / 7 / 8.
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


# A real section body is always at least this long in practice (SEC's
# shortest genuine MD&A/risk-factors sections still run several thousand
# characters). A candidate match whose span falls short of this is almost
# certainly a table-of-contents entry or a cross-reference, not the actual
# section body — see MIN_SECTION_CHARS usage in _extract_sections below.
MIN_SECTION_CHARS = 800


def _extract_sections(text: str, sections: list, max_chars: int,
                      form_type: str = "10-K") -> dict:
    if not text:
        return {}

    patterns = _SECTION_PATTERNS.get(form_type.upper(), _SECTION_PATTERNS["10-K"])
    clean    = _html_to_text(text)
    text_low = clean.lower()
    want_all = "all" in sections

    # Collect ALL occurrences of every header (not just the last one) so we
    # can search backward for one that actually yields a substantive span,
    # and build one merged, sorted list of boundary points (from every
    # pattern) to compute "where does this candidate section actually end".
    all_matches: dict[str, list[int]] = {}
    for name, pat in patterns.items():
        starts = [m.start() for m in re.finditer(pat, text_low)]
        if starts:
            all_matches[name] = starts

    merged_boundaries = sorted(
        {pos for starts in all_matches.values() for pos in starts}
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

    # For each requested section, walk its own occurrences from LAST to
    # FIRST and take the first one whose span is long enough to plausibly
    # be the real body — not a table-of-contents stub. Falls back to the
    # last occurrence (old behavior) if NONE meet the length bar, since a
    # too-short real answer is still better than silently returning nothing.
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
        end = sorted_pos[idx + 1][1] if idx + 1 < len(sorted_pos) else len(clean)
        result[name] = clean[start:end].strip()[:max_chars]

    return result