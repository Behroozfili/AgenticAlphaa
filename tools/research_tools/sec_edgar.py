import logging
import re
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
        cik = await _resolve_cik(ticker)
        if cik:
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
        filings.append({
            "company":          src.get("entity_name", ""),
            "form_type":        src.get("form_type", ""),
            "filed_at":         src.get("file_date", ""),
            "accession_number": src.get("accession_no", ""),
            "period":           src.get("period_of_report", ""),
        })

    return {"query": query, "filings": filings}


# ─────────────────────────────────────────────
# Tool 2: Fetch and parse a specific filing
# ─────────────────────────────────────────────
async def sec_edgar_filing(
    ticker: str,
    form_type: str = "10-K",
    sections: list = None,             # ["business","risk_factors","mda","all"]
    max_chars: int = 8000,
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

    acc_no     = filing_meta["accessionNumber"].replace("-", "")
    cik_padded = cik.zfill(10)
    raw_text   = await _fetch_text(cik_padded, acc_no)
    parsed     = _extract_sections(raw_text, sections, max_chars)

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
async def _resolve_cik(ticker: str) -> Optional[str]:
    t = ticker.upper()
    if t in _CIK_CACHE:
        return _CIK_CACHE[t]
    url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?CIK={t}&action=getcompany&output=atom"
    async with httpx.AsyncClient(timeout=15.0, headers=HEADERS) as client:
        resp = await client.get(url)
        match = re.search(r"CIK=(\d+)", resp.text)
        if match:
            _CIK_CACHE[t] = match.group(1)
            return match.group(1)
    return None


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


async def _fetch_text(cik_padded: str, acc_no: str) -> str:
    url = f"{EDGAR_BASE}/Archives/edgar/data/{cik_padded}/{acc_no}/{acc_no}.txt"
    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.text[:500_000]
    return ""


def _extract_sections(text: str, sections: list, max_chars: int) -> dict:
    PATTERNS = {
        "business":             r"item\s+1[.\s]+business",
        "risk_factors":         r"item\s+1a[.\s]+risk\s+factors",
        "mda":                  r"item\s+7[.\s]+management",
        "financial_statements": r"item\s+8[.\s]+financial\s+statements",
    }
    want_all = "all" in sections
    text_low = text.lower()

    positions = {
        name: m.start()
        for name, pat in PATTERNS.items()
        if (m := re.search(pat, text_low))
    }
    sorted_pos = sorted(positions.items(), key=lambda x: x[1])

    result = {}
    for idx, (name, start) in enumerate(sorted_pos):
        if not want_all and name not in sections:
            continue
        end   = sorted_pos[idx + 1][1] if idx + 1 < len(sorted_pos) else len(text)
        result[name] = text[start:end].strip()[:max_chars]

    return result