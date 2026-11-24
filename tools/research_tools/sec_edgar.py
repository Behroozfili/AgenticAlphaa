import logging
import re
import html as _html
import httpx
from typing import Any, Mapping, Optional

from core.forecast_contract import AnalysisMode, ForecastContext
from tools.research_tools import (
    context_metadata,
    eligible_evidence,
    historical_unavailable,
    observed_research_operation,
    require_research_context,
)

logger = logging.getLogger(__name__)

EDGAR_BASE   = "https://www.sec.gov"
EDGAR_DATA   = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/{name}"
HEADERS      = {"User-Agent": "AlphaAgentNode research@alpha-agent.ai"}

_CIK_CACHE: dict[str, str] = {}
_MAX_QUERY_TERMS = 6
_DATE_QUERY_TERMS = {
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "q1", "q2", "q3", "q4",
}


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
    terms = [
        t for t in query.replace('"', '').split()
        if not re.fullmatch(r"(19|20)\d{2}", t)
        and t.strip(".,;:()[]").lower() not in _DATE_QUERY_TERMS
    ]
    if len(terms) > _MAX_QUERY_TERMS:
        logger.warning(
            "SEC_QUERY_TRUNCATED term_count=%d max_terms=%d",
            len(terms), _MAX_QUERY_TERMS,
        )
        terms = terms[:_MAX_QUERY_TERMS]
    return " ".join(terms)


# ─────────────────────────────────────────────
# Tool 1: Search filings by keyword / ticker
# ─────────────────────────────────────────────
@observed_research_operation("sec_edgar_search")
async def sec_edgar_search(
    query: str,
    ticker: Optional[str] = None,
    form_type: Optional[str] = None,   # "10-K" | "10-Q" | "8-K"
    max_results: int = 5,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    archived_records: list[Mapping[str, Any]] | None = None,
) -> dict:
    """
    Full-text search across SEC EDGAR filings.
    Returns: list of filings with company, form_type, filed_at, accession_number, url.

    Gracefully degrades on upstream failure: returns
    A failed provider call returns a stable ``error_code`` and no exception
    text, so a flaky/overloaded SEC endpoint never crashes or leaks to callers.
    """
    context, run_identity = require_research_context(forecast_context, run_id)
    if ticker and ticker.strip().upper() != context.ticker:
        raise ValueError("ticker must match ForecastContext.ticker")
    ticker = context.ticker
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        if archived_records is None:
            return historical_unavailable(
                context, run_identity, source="sec_edgar_search",
                reason="Historical SEC research requires retained versioned filing metadata; no local SEC archive was supplied.",
            ) | {"query": query, "filings": []}
        evidence, rejected = eligible_evidence(
            archived_records, context=context, run_id=run_identity,
            source="sec_edgar_search", historical=True,
        )
        return {
            **context_metadata(context, run_identity), "query": query,
            "filings": evidence, "evidence": evidence, "quarantined": rejected,
            "status": "OK" if evidence else "DATA_UNAVAILABLE",
            "missing": not evidence, "degraded": bool(rejected) or not evidence,
        }

    clean_query = _sanitize_query(query)
    params: dict = {
        "q": clean_query,
        "startdt": (context.as_of.date() - __import__("datetime").timedelta(days=context.lookback_days)).isoformat(),
        "enddt": context.as_of.date().isoformat(),
    }
    if form_type:
        params["forms"] = form_type
    if ticker:
        try:
            cik = await _resolve_cik(ticker)
        except Exception:
         
            logger.warning(
                "SEC_CIK_RESOLUTION_FAILED",
            )
            return {
                "filings": [], "error_code": "SEC_CIK_RESOLUTION_FAILED",
                "status": "FAIL_CLOSED", "missing": True, "degraded": True,
            }
        if cik is None:
            logger.warning(
                "sec_edgar_search: no CIK found for ticker=%r — refusing to run "
                "an unscoped full-text search.", ticker,
            )
            return {
                "filings": [], "error_code": "SEC_CIK_NOT_FOUND",
                "status": "DATA_UNAVAILABLE", "missing": True, "degraded": True,
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
        logger.warning("SEC_HTTP_STATUS_FAILED status_code=%s", exc.response.status_code)
        return {"filings": [], "error_code": "SEC_HTTP_STATUS_FAILED", "status": "FAIL_CLOSED", "missing": True, "degraded": True}
    except httpx.HTTPError:
        logger.warning("SEC_NETWORK_FAILED")
        return {"filings": [], "error_code": "SEC_NETWORK_FAILED", "status": "FAIL_CLOSED", "missing": True, "degraded": True}

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
            "ticker":           context.ticker,
            "dataset_version":  context.dataset_version,
            "provider":         "sec_live",
        })

    evidence, rejected = eligible_evidence(
        filings, context=context, run_id=run_identity,
        source="sec_edgar_search", historical=False,
    )
    eligible_accessions = {
        str(item.get("citation", {}).get("accession_number")): item for item in evidence
        if isinstance(item.get("citation"), dict)
    }
    filings = [
        {**filing, **eligible_accessions[str(filing.get("accession_number"))]}
        for filing in filings if str(filing.get("accession_number")) in eligible_accessions
    ]
    return {
        **context_metadata(context, run_identity), "query": query,
        "filings": filings, "evidence": evidence, "quarantined": rejected,
        "status": "OK" if evidence else "DATA_UNAVAILABLE",
        "missing": not evidence, "degraded": bool(rejected) or not evidence,
    }

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
@observed_research_operation("sec_edgar_filing")
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
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    archived_filing: Mapping[str, Any] | None = None,
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
    context, run_identity = require_research_context(forecast_context, run_id)
    if ticker.strip().upper() != context.ticker:
        raise ValueError("ticker must match ForecastContext.ticker")
    ticker = context.ticker
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        if archived_filing is None:
            return historical_unavailable(
                context, run_identity, source="sec_edgar_filing",
                reason="Historical filing text must come from a retained versioned SEC archive.",
            ) | {"ticker": ticker, "sections": {}}
        evidence, rejected = eligible_evidence(
            [archived_filing], context=context, run_id=run_identity,
            source="sec_edgar_filing", historical=True,
        )
        if not evidence:
            return {
                **historical_unavailable(
                    context, run_identity, source="sec_edgar_filing",
                    reason="Archived filing failed ticker/version/filed_at eligibility.",
                ),
                "ticker": ticker, "sections": {}, "quarantined": rejected,
            }
        return {
            **dict(archived_filing), **context_metadata(context, run_identity),
            "ticker": ticker, "evidence": evidence, "quarantined": rejected,
            "status": "OK", "missing": False, "degraded": False,
        }

    if not sections:
        return {
            "ticker": ticker,
            "error_code": "SEC_SECTIONS_REQUIRED", "status": "FAIL_CLOSED",
            "missing": True, "degraded": True,
        }
    if mode not in ("preview", "full"):
        return {"ticker": ticker, "error_code": "SEC_MODE_INVALID", "status": "FAIL_CLOSED", "missing": True, "degraded": True}

    normalized_accession = None
    if accession_number is not None:
        normalized_accession = _normalize_accession_number(accession_number)
        if normalized_accession is None:
            return {
                "ticker": ticker,
                "error_code": "SEC_ACCESSION_INVALID", "status": "FAIL_CLOSED",
                "missing": True, "degraded": True,
            }

    try:
        cik = await _resolve_cik(ticker)
    except Exception:
        result = {"ticker": ticker, "error_code": "SEC_CIK_RESOLUTION_FAILED", "status": "FAIL_CLOSED", "missing": True, "degraded": True}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result
    if not cik:
        result = {"ticker": ticker, "error_code": "SEC_CIK_NOT_FOUND", "status": "DATA_UNAVAILABLE", "missing": True, "degraded": True}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result

    if normalized_accession and normalized_accession[:10] != cik.zfill(10):
        return {
            "ticker": ticker,
            "error_code": "SEC_ACCESSION_TICKER_MISMATCH", "status": "FAIL_CLOSED",
            "missing": True, "degraded": True,
        }

    try:
        submissions = await _get_submissions(cik)
    except Exception:
        result = {"ticker": ticker, "error_code": "SEC_SUBMISSIONS_FETCH_FAILED", "status": "FAIL_CLOSED", "missing": True, "degraded": True}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result
    if not submissions:
        result = {"ticker": ticker, "error_code": "SEC_SUBMISSIONS_FETCH_FAILED", "status": "FAIL_CLOSED", "missing": True, "degraded": True}
        if normalized_accession:
            result["accession_number"] = normalized_accession
        return result

    if normalized_accession:
        try:
            filing_meta = await _find_by_accession(submissions, normalized_accession)
        except Exception:
            return {
                "ticker": ticker,
                "accession_number": normalized_accession,
                "error_code": "SEC_ACCESSION_RESOLUTION_FAILED", "status": "FAIL_CLOSED",
                "missing": True, "degraded": True,
            }
        if not filing_meta:
            return {
                "ticker": ticker,
                "error_code": "SEC_ACCESSION_NOT_FOUND", "status": "DATA_UNAVAILABLE",
                "missing": True, "degraded": True,
            }
        actual_form_type = filing_meta.get("form", "")
        if not actual_form_type:
            return {
                "ticker": ticker,
                "error_code": "SEC_FORM_TYPE_MISSING", "status": "DATA_UNAVAILABLE",
                "missing": True, "degraded": True,
            }
        if form_type and actual_form_type and form_type.upper() != actual_form_type.upper():
            return {
                "ticker": ticker,
                "error_code": "SEC_FORM_TYPE_CONFLICT", "status": "FAIL_CLOSED",
                "missing": True, "degraded": True,
            }
        if str(filing_meta.get("filingDate") or "") > context.as_of.date().isoformat():
            return {
                **context_metadata(context, run_identity),
                "ticker": ticker, "accession_number": normalized_accession,
                "sections": {}, "evidence": [],
                "status": "DATA_UNAVAILABLE", "missing": True, "degraded": True,
                "error_code": "FUTURE_EVIDENCE",
            }
    else:
        actual_form_type = form_type or "10-K"
        filing_meta = _find_latest(
            submissions, actual_form_type, as_of=context.as_of.date().isoformat()
        )
        if not filing_meta:
            return {"ticker": ticker, "error_code": "SEC_FILING_NOT_FOUND", "status": "DATA_UNAVAILABLE", "missing": True, "degraded": True}

    acc_dash   = filing_meta["accessionNumber"]          # e.g. 0000320193-26-000013
    acc_nodash = acc_dash.replace("-", "")               # e.g. 000032019326000013
    try:
        raw_text = await _fetch_text(cik, acc_dash, acc_nodash)
    except Exception:
        return {
            "ticker": ticker,
            "accession_number": acc_dash,
            "error_code": "SEC_FILING_FETCH_FAILED", "status": "FAIL_CLOSED",
            "missing": True, "degraded": True,
        }
    if not raw_text:
        return {
            "ticker": ticker,
            "error_code": "SEC_FILING_FETCH_FAILED", "status": "FAIL_CLOSED",
            "missing": True, "degraded": True,
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

    result = {
        **context_metadata(context, run_identity),
        "ticker":           ticker,
        "company":          submissions.get("name", ""),
        "form_type":        actual_form_type,
        "filed_at":         filing_meta.get("filingDate", ""),
        "accession_number": filing_meta.get("accessionNumber", ""),
        "mode":             mode,
        "sections":         sections_out,
    }
    evidence, rejected = eligible_evidence(
        [{**result, "provider": "sec_live", "title": f"{ticker} {actual_form_type}"}],
        context=context, run_id=run_identity, source="sec_edgar_filing", historical=False,
    )
    if not evidence:
        reason_codes = {
            item.get("reason_code") for item in rejected if isinstance(item, Mapping)
        }
        return {
            **context_metadata(context, run_identity), "ticker": ticker,
            "sections": {}, "evidence": [], "quarantined": rejected,
            "status": "DATA_UNAVAILABLE", "missing": True, "degraded": True,
            "error_code": (
                "FUTURE_EVIDENCE" if "FUTURE_EVIDENCE" in reason_codes
                else "SEC_EVIDENCE_OUTSIDE_WINDOW"
            ),
        }
    return {
        **result, "evidence": evidence, "quarantined": rejected,
        "status": "OK", "missing": False, "degraded": False,
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


def _find_latest(
    submissions: dict, form_type: str, *, as_of: str | None = None,
) -> Optional[dict]:
    recent  = submissions.get("filings", {}).get("recent", {})
    forms   = recent.get("form", [])
    acc_nos = recent.get("accessionNumber", [])
    dates   = recent.get("filingDate", [])
    for i, f in enumerate(forms):
        if f == form_type and (as_of is None or str(dates[i]) <= as_of):
            return {"accessionNumber": acc_nos[i], "filingDate": dates[i]}
    return None


async def _fetch_text(cik: str, acc_dash: str, acc_nodash: str) -> str:

    cik_int = str(int(cik))
    url = f"{EDGAR_BASE}/Archives/edgar/data/{cik_int}/{acc_nodash}/{acc_dash}.txt"
    async with httpx.AsyncClient(timeout=60.0, headers=HEADERS, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            return resp.text
    logger.warning("SEC_FILING_FETCH_FAILED status_code=%s", resp.status_code)
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
