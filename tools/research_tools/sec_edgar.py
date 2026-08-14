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


# A rough "sub-heading" inside a section body: a short run of ALL-CAPS
# words immediately followed by a normal sentence (e.g. "STRATEGIC AND
# COMPETITIVE RISKS We face intense competition..."). These come from SEC's
# own bolded sub-headers within Risk Factors / MD&A and are the cheapest
# possible signal an agent can use to jump to the right part of a huge
# section without reading the whole thing. Heuristic, not perfect: misses
# sub-headers that don't happen to be followed directly by a capitalized
# sentence, and can occasionally catch a stray all-caps phrase that isn't
# really a heading. Good enough as a navigation aid, not a source of truth.
_HEADING_PATTERN = re.compile(
    r"(?:(?<=[.?!]\s)|^)((?:[A-Z][A-Z&,'\u2019\-]{1,}\s+){1,7}[A-Z][A-Z&,'\u2019\-]{1,})(?=\s+[A-Z][a-z])"
)
_MAX_HEADINGS = 15


def _find_headings(body: str) -> list:
    seen = []
    for m in _HEADING_PATTERN.finditer(body):
        h = m.group(1).strip()
        # Skip the section's own title line re-matching (e.g. "RISK FACTORS"
        # right at the start, from "ITEM 1A. RISK FACTORS ...") and
        # near-duplicates -- these aren't useful sub-headings.
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


# ── Deterministic compression, no LLM ──────────────────────────────
#
# Preview mode's raw `preview` (first N chars) and `headings` (bare list of
# sub-header titles) are still useful, but neither is actually a *summary*
# -- one is just the beginning of the document, the other has no content.
# These two helpers turn the section into something genuinely short but
# information-dense, using only string/regex work (no model call, no
# risk of paraphrasing a number wrong):
#
#   - `_heading_leads`: for prose sections (risk_factors, mda, business),
#     pairs each detected sub-heading with the first sentence that follows
#     it -- a structural TL;DR the agent can scan in one screen.
#   - `_extract_key_figures`: for financial_statements, which is a table
#     of numbers rather than prose, pulls the standard income-statement
#     line items out verbatim (exact strings, never rounded/reformatted)
#     instead of returning any prose at all.

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_LEAD_SENTENCE_CHARS = 220
_MAX_LEADS = 10

# Sentences containing a dollar figure, a percentage, or "million"/"billion"
# are disproportionately likely to be the actual analyst-relevant content
# in a prose section (revenue growth, tax exposure, cost changes, ...).
# Pulling these out separately from _heading_leads catches numeric detail
# that a heading's *first* sentence often doesn't include (the number
# usually lands 2-3 sentences in, after the topic is introduced).
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


# Standard 10-K/10-Q financial-statement line items this looks for, in
# roughly the order they appear across Income Statement, then Cash Flow
# Statement. Each is followed by 1-4 numeric tokens (current/prior period,
# quarter/YTD -- the exact count varies by filing), which this pulls out
# character-for-character (with $, commas, and parens intact) rather than
# re-parsing into floats -- avoids ever silently mangling a number.
#
# The cash-flow items (Additions to property and equipment, Net cash from
# operations, Net cash used in investing) matter as much as the income
# statement ones: "Additions to property and equipment" IS the capex
# figure -- for a query like "AI infrastructure spending", this is often
# THE number that answers the question, and previously wasn't extracted
# at all (income-statement-only list), forcing the agent to fall back to
# weaker secondary sources (news articles that only mention capex in
# passing) instead of the filing's own authoritative figure.
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

# Matches a number, optionally $-prefixed, optionally wrapped in
# parentheses (SEC's convention for a negative/outflow figure). The `\s*`
# immediately before the closing paren is required: real filings routinely
# render negatives as "(30,876 )" with a space before ")", and without
# tolerating that space here, the match silently stops short (e.g. capturing
# just "(30,876" with no closing paren and dropping the rest of the row) --
# this went unnoticed for the income-statement labels above because SEC
# income-statement rows are rarely negative, but cash-flow outflow lines
# (capex, investing activities) are negative far more often and would have
# been silently mis-extracted without this fix.
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
    form_type: str = "10-K",
    sections: list = None,             # e.g. ["risk_factors","mda"] -- REQUIRED, no silent "all"
    mode: str = "preview",             # "preview" | "full"
    preview_chars: int = 2000,
    query: str = None,                 # find this text inside each requested section
    query_window: int = 3000,          # chars of context around the query match
    offset: int = 0,                   # mode="full" only: start position within the section
    length: int = None,                # mode="full" only: chars to return (default: max_chars)
    max_chars: int = 25000,            # hard ceiling on how much of a section is ever parsed out
) -> dict:
    """
    Fetch the latest SEC filing (10-K / 10-Q) for a ticker and return one or
    more named sections: business, risk_factors, mda, financial_statements.

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
    # Extract with a generous internal cap -- NOT max_chars. Preview mode's
    # total_chars/headings/query need the section's real full size; if we
    # truncated at max_chars (which defaults to 25000, a "full mode response
    # size" cap) before computing those, preview would report a fake,
    # artificially-small total instead of the document's actual length.
    # max_chars/length are applied below, only when slicing for output.
    _EXTRACTION_SAFETY_CAP = 750_000  # generous ceiling against pathological input, not a response cap
    parsed = _extract_sections(raw_text, sections, _EXTRACTION_SAFETY_CAP, form_type)

    sections_out = {}
    for name, body in parsed.items():
        entry = {"total_chars": len(body)}

        if mode == "preview":
            entry["preview"] = body[:preview_chars]
            entry["preview_chars_returned"] = min(preview_chars, len(body))
            entry["headings"] = _find_headings(body)
            if name == "financial_statements":
                # Numbers table, not prose -- pull the standard line items
                # out verbatim instead of a heading+sentence digest.
                entry["key_figures"] = _extract_key_figures(body)
            else:
                # Structural TL;DR: each sub-heading + the sentence right
                # after it. Usually all the signal an agent needs without
                # reading the full section.
                entry["digest"] = _heading_leads(body)
                # Numeric-signal sentences the heading+lead approach can
                # miss (the number is often a few sentences into a
                # subsection, not in its opening line).
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
        "form_type":        form_type,
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
    # Block/structural tags genuinely separate content -- replace with a space.
    # Everything else (inline wrapper tags like <span>, <font>, <ix:nonFraction>,
    # <a>, etc.) is frequently inserted MID-WORD by SEC's Inline XBRL tagging
    # (e.g. "FINANC<ix:nonFraction ...>IAL</ix:nonFraction>" must become
    # "FINANCIAL", not "FINANC IAL"). Stripping those with no space avoids
    # splitting words apart and breaking the "item N. <heading>" regexes.
    raw = re.sub(r"(?is)</?(p|div|tr|td|th|table|br|hr|h[1-6]|li|ul|ol)[^>]*>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", "", raw)          # drop remaining inline tags, no space
    raw = _html.unescape(raw)                      # &#160; &amp; &#8217; ...
    raw = raw.replace("\u00a0", " ")               # non-breaking space
    raw = re.sub(r"\s+", " ", raw)
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

# A bare "PART I" / "PART II" trailer left dangling at the very END of an
# extracted section (immediately before the next real header starts) is
# table-of-contents/section-break noise, not content -- strip it.
#
# NOTE: this must be a tail-only cleanup, NOT a general boundary used to cut
# section spans. Real SEC filings repeat "PART I. FINANCIAL INFORMATION" /
# "PART II. OTHER INFORMATION" as a running header on EVERY page. Treating
# every occurrence as a hard boundary (an earlier version of this function
# did) truncates real sections after their first page -- e.g. MD&A cut off
# mid-sentence after ~700 chars instead of running its full multi-page
# length. Only strip it when it's trailing noise, never use it to bound
# extraction.
_TRAILING_PART_MARKER = re.compile(
    r"(?i)\s*part\s+[ivx]+\.?(?:\s+[a-z][a-z'\-]*){0,4}\s*$"
)

# Generic boundary for ANY numbered item header ("ITEM 3.", "ITEM 4.",
# "ITEM 1.", etc.) -- not just the 4 named sections we extract. Without
# this, content from untracked items (e.g. Item 3 Quantitative Disclosures,
# Item 4 Controls and Procedures, Part II Item 1 Legal Proceedings) gets
# silently swallowed into the tail of whichever named section precedes it,
# because only business/risk_factors/mda/financial_statements were ever
# treated as boundaries.
#
# Deliberately CASE-SENSITIVE and requires the period right after the
# number/letter. Real section headers appear as "ITEM 3. QUANTITATIVE...".
# Real SEC filings ALSO repeat a short running header on every page, e.g.
# "PART I Item 2" or "PART II Item 1A" (mixed case, no period) -- matching
# those too would re-truncate sections after their first page, which is
# exactly the regression that occurred with the earlier PART-only boundary.
# Requiring "ITEM" in all caps + "." excludes those running-header stubs
# while still catching every genuine header.
_ITEM_BOUNDARY_PATTERN = re.compile(r"\bITEM\s+\d{1,2}[A-Z]?\.")


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
        next_requested = sorted_pos[idx + 1][1] if idx + 1 < len(sorted_pos) else len(clean)
        # Cap at whichever comes first: the next requested section's start,
        # or the nearest ITEM-header boundary (tracked or not). Without the
        # second cap, an untracked item (Item 3, Item 4, Part II Item 1
        # Legal Proceedings, ...) sitting between this section and the next
        # *requested* section leaks into this section's tail.
        end = min(next_requested, _end_boundary_after(start))
        body = clean[start:end].strip()
        # Strip a dangling "PART II" (etc.) that sits right before the next
        # real header -- tail-only, so it never affects where real content
        # gets cut.
        body = _TRAILING_PART_MARKER.sub("", body).strip()
        result[name] = body[:max_chars]

    return result