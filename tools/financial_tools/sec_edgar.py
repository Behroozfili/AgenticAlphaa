"""
sec_edgar.py
------------
Tool for retrieving data from the U.S. Securities and Exchange Commission
(SEC) EDGAR system. No API key is required, but the SEC mandates a
descriptive ``User-Agent`` header on every request — requests without it
receive HTTP 403.

This module exposes four functions consumed by the Financial Analyst Agent's
MCP server (``financial_server.py``):

    get_cik(ticker)                       -> resolve ticker  -> CIK
    list_filings(ticker, form_type, limit)-> recent filings metadata
    get_filing_text(accession_number, cik)-> clean plain text of a filing
    get_xbrl_financials(ticker)           -> structured XBRL company facts

All functions return flat dictionaries with an ``error`` key (``None`` on
success) so they can be serialised straight back over the MCP channel and
never raise across the tool boundary.

Endpoints used
--------------
- https://www.sec.gov/files/company_tickers.json          (ticker -> CIK map)
- https://data.sec.gov/submissions/CIK##########.json     (filing history)
- https://www.sec.gov/Archives/edgar/data/...             (filing documents)
- https://data.sec.gov/api/xbrl/companyfacts/CIK#####.json (XBRL facts)

Configuration
-------------
Set ``SEC_EDGAR_USER_AGENT`` in the environment to your own
``"Name contact@example.com"`` string. A safe default is provided so the
module works out of the box, but using your own contact is the SEC's
documented expectation for sustained use.
"""

from __future__ import annotations

import os
import re
import time
import html as _html
import logging
from typing import Any

import requests

log = logging.getLogger("sec_edgar")

# ---------------------------------------------------------------------------
# Constants & shared session
# ---------------------------------------------------------------------------

# SEC requires a descriptive User-Agent. Override via env var for your org.
_USER_AGENT = os.environ.get(
    "SEC_EDGAR_USER_AGENT",
    "Financial Analyst Agent admin@example.com",
)

_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": None,  # set per-request below; placeholder kept for clarity
}

_TIMEOUT = 30  # seconds
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.5  # seconds, multiplied by attempt number

# Reuse one session for connection pooling.
_session = requests.Session()
_session.headers.update({"User-Agent": _USER_AGENT,
                         "Accept-Encoding": "gzip, deflate"})

# Cache the (large) ticker->CIK map for the lifetime of the process.
_ticker_map_cache: dict[str, dict] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(url: str, *, host: str, as_json: bool = True) -> Any:
    """
    Perform a GET request against an SEC endpoint with the mandatory headers,
    basic retry/backoff, and clear error propagation.

    Parameters
    ----------
    url : str
        Fully-qualified URL to fetch.
    host : str
        The ``Host`` header value (e.g. "www.sec.gov" or "data.sec.gov").
        SEC's CDN is sensitive to a correct Host header.
    as_json : bool
        If True, parse and return JSON; otherwise return raw text.

    Returns
    -------
    Any
        Parsed JSON (dict/list) or raw text.

    Raises
    ------
    RuntimeError
        On a non-recoverable HTTP error after exhausting retries.
    """
    headers = {"Host": host}
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = _session.get(url, headers=headers, timeout=_TIMEOUT)

            # 403 almost always means a missing/blocked User-Agent.
            if resp.status_code == 403:
                raise RuntimeError(
                    "SEC returned 403 Forbidden. Set a valid SEC_EDGAR_USER_AGENT "
                    "environment variable (format: 'Name contact@example.com')."
                )
            # 429 = rate limited; back off and retry.
            if resp.status_code == 429:
                wait = _RETRY_BACKOFF * attempt
                log.warning("SEC rate limit (429); retrying in %.1fs", wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json() if as_json else resp.text

        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * attempt
                log.warning("SEC request failed (%s); retry %d/%d in %.1fs",
                            exc, attempt, _MAX_RETRIES, wait)
                time.sleep(wait)
            else:
                break

    raise RuntimeError(f"SEC request failed for {url}: {last_exc}")


def _pad_cik(cik: str | int) -> str:
    """Return a 10-digit zero-padded CIK string."""
    return str(int(cik)).zfill(10)


def _load_ticker_map() -> dict[str, dict]:
    """
    Load (and cache) the SEC ticker->CIK map.

    Returns a dict keyed by upper-case ticker:
        {"AAPL": {"cik": "0000320193", "title": "Apple Inc."}, ...}
    """
    global _ticker_map_cache
    if _ticker_map_cache is not None:
        return _ticker_map_cache

    data = _get("https://www.sec.gov/files/company_tickers.json",
                host="www.sec.gov")

    mapping: dict[str, dict] = {}
    # The payload is { "0": {cik_str, ticker, title}, "1": {...}, ... }
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        mapping[ticker] = {
            "cik":   _pad_cik(entry["cik_str"]),
            "title": entry.get("title", "N/A"),
        }

    _ticker_map_cache = mapping
    return mapping


def _strip_html(raw: str) -> str:
    """
    Convert an HTML/XBRL filing document into reasonably clean plain text
    without external dependencies.

    Removes script/style blocks and all tags, unescapes HTML entities,
    and collapses excessive whitespace.
    """
    # Drop script & style content entirely.
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    # Treat block-level closings as line breaks for readability.
    raw = re.sub(r"(?i)</(p|div|tr|li|h[1-6]|table)>", "\n", raw)
    raw = re.sub(r"(?i)<br\s*/?>", "\n", raw)
    # Remove all remaining tags.
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    # Unescape entities (&amp; &nbsp; &#160; …).
    raw = _html.unescape(raw)
    # Normalise whitespace.
    raw = re.sub(r"[ \t\u00a0]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> dict:
    """
    Resolve a stock ticker symbol to its SEC Central Index Key (CIK).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (case-insensitive), e.g. "NVDA".

    Returns
    -------
    dict
        - "ticker"        (str)        Normalised (upper-case) ticker.
        - "cik"           (str)        10-digit zero-padded CIK.
        - "company_name"  (str)        Official company name in EDGAR.
        - "error"         (str | None)

    Examples
    --------
    >>> get_cik("NVDA")["cik"]
    '0001045810'
    """
    t = (ticker or "").upper().strip()
    try:
        mapping = _load_ticker_map()
        entry = mapping.get(t)
        if entry is None:
            return {
                "ticker": t, "cik": None, "company_name": None,
                "error": f"Ticker '{t}' not found in SEC EDGAR ticker index.",
            }
        return {
            "ticker":       t,
            "cik":          entry["cik"],
            "company_name": entry["title"],
            "error":        None,
        }
    except Exception as exc:
        return {"ticker": t, "cik": None, "company_name": None, "error": str(exc)}


def list_filings(ticker: str, form_type: str = "10-K", limit: int = 5) -> dict:
    """
    List the most recent SEC EDGAR filings of a given form type for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    form_type : str
        SEC form type to filter by, e.g. "10-K", "10-Q", "8-K". Default "10-K".
    limit : int
        Maximum number of filings to return. Default 5.

    Returns
    -------
    dict
        - "ticker"    (str)
        - "cik"       (str)
        - "form_type" (str)
        - "filings"   (list[dict]) Each: {accession_number, filing_date,
                                          report_date, document_url}
        - "error"     (str | None)
    """
    t = (ticker or "").upper().strip()
    form_type = (form_type or "10-K").strip()

    cik_info = get_cik(t)
    if cik_info.get("error"):
        return {"ticker": t, "cik": None, "form_type": form_type,
                "filings": [], "error": cik_info["error"]}

    cik = cik_info["cik"]
    try:
        data = _get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                    host="data.sec.gov")

        recent = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        fil_dates  = recent.get("filingDate", [])
        rep_dates  = recent.get("reportDate", [])
        primaries  = recent.get("primaryDocument", [])

        cik_int = str(int(cik))  # un-padded for Archives URLs
        filings: list[dict] = []

        for i, form in enumerate(forms):
            if form != form_type:
                continue

            accession = accessions[i] if i < len(accessions) else None
            if not accession:
                continue
            accession_nodash = accession.replace("-", "")

            # Index page for the filing.
            document_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{accession_nodash}/{accession}-index.htm"
            )

            filings.append({
                "accession_number": accession,
                "filing_date":      fil_dates[i] if i < len(fil_dates) else None,
                "report_date":      rep_dates[i] if i < len(rep_dates) else None,
                "document_url":     document_url,
            })

            if len(filings) >= limit:
                break

        return {
            "ticker":    t,
            "cik":       cik,
            "form_type": form_type,
            "filings":   filings,
            "error":     None,
        }

    except Exception as exc:
        return {"ticker": t, "cik": cik, "form_type": form_type,
                "filings": [], "error": str(exc)}


def get_filing_text(accession_number: str, cik: str) -> dict:
    """
    Download and extract the plain-text content of a specific SEC filing.

    Fetches the filing's primary document, strips HTML/XBRL markup, and
    returns clean text suitable for chunking and embedding.

    Parameters
    ----------
    accession_number : str
        EDGAR accession number with dashes, e.g. "0001045810-24-000010".
        Obtain from :func:`list_filings`.
    cik : str
        CIK of the filer (zero-padded or not). Obtain from :func:`get_cik`.

    Returns
    -------
    dict
        - "accession_number" (str)
        - "cik"              (str)
        - "text"             (str | None)
        - "word_count"       (int)
        - "error"            (str | None)
    """
    acc = (accession_number or "").strip()
    try:
        cik_int = str(int(cik))           # un-padded for Archives path
        cik_padded = _pad_cik(cik)
        accession_nodash = acc.replace("-", "")
        base = (f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{accession_nodash}")

        # 1) Use the filing's index.json to locate the primary document.
        index = _get(f"{base}/index.json", host="www.sec.gov")
        items = index.get("directory", {}).get("item", [])

        # Prefer the main HTML document; skip XBRL/exhibit/graphic files.
        candidate = None
        htm_files = [it["name"] for it in items
                     if it.get("name", "").lower().endswith((".htm", ".html"))]
        # Heuristic: the primary doc usually contains the form id or is the
        # largest .htm that is not an exhibit ("ex" / "R" XBRL viewer pages).
        preferred = [n for n in htm_files
                     if not re.match(r"(?i)^(ex|r\d|.*-index)", n)]
        if preferred:
            candidate = preferred[0]
        elif htm_files:
            candidate = htm_files[0]
        else:
            # Fall back to the full submission .txt file.
            candidate = f"{acc}.txt"

        doc_url = f"{base}/{candidate}"
        raw = _get(doc_url, host="www.sec.gov", as_json=False)

        text = _strip_html(raw)
        word_count = len(text.split())

        return {
            "accession_number": acc,
            "cik":              cik_padded,
            "text":             text,
            "word_count":       word_count,
            "error":            None,
        }

    except Exception as exc:
        return {"accession_number": acc, "cik": str(cik),
                "text": None, "word_count": 0, "error": str(exc)}


# Candidate XBRL concept tags, in priority order, per metric. Companies
# tag the same economic figure differently, so we try several.
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
_NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
_ASSETS_TAGS = ["Assets"]
_LIABILITIES_TAGS = ["Liabilities"]


def _extract_annual(facts: dict, tags: list[str]) -> list[dict]:
    """
    From a companyfacts payload, extract annual (10-K, full-year) USD values
    for the first tag in *tags* that has data.

    Returns a list of {period_end, value, unit} dicts, deduplicated by
    period end and limited to the 10 most recent periods (newest first).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    for tag in tags:
        concept = us_gaap.get(tag)
        if not concept:
            continue

        units = concept.get("units", {})
        # Prefer USD; fall back to the first available unit.
        unit_key = "USD" if "USD" in units else next(iter(units), None)
        if unit_key is None:
            continue

        rows = units[unit_key]
        by_period: dict[str, dict] = {}

        for row in rows:
            # Keep annual figures filed on 10-K with a full fiscal-year frame.
            form = row.get("form")
            fp = row.get("fp")
            end = row.get("end")
            val = row.get("val")
            if end is None or val is None:
                continue
            # Annual balance-sheet items (Assets/Liabilities) have no fp=FY
            # distinction, so accept either FY income items or 10-K snapshots.
            is_annual = (fp == "FY") or (form == "10-K")
            if not is_annual:
                continue

            # Last write wins → favour the most recently filed restatement.
            by_period[end] = {
                "period_end": end,
                "value":      float(val),
                "unit":       unit_key,
            }

        if by_period:
            ordered = sorted(by_period.values(),
                             key=lambda r: r["period_end"], reverse=True)
            return ordered[:10]

    return []


def get_xbrl_financials(ticker: str) -> dict:
    """
    Retrieve structured financial statement data from the SEC EDGAR XBRL API.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        - "ticker"             (str)
        - "cik"                (str)
        - "revenue_annual"     (list[dict]) [{period_end, value, unit}]
        - "net_income_annual"  (list[dict])
        - "total_assets"       (list[dict])
        - "total_liabilities"  (list[dict])
        - "error"              (str | None)

        Values are in USD. Up to the 10 most recent annual data points per
        metric are returned (newest first).
    """
    t = (ticker or "").upper().strip()

    cik_info = get_cik(t)
    if cik_info.get("error"):
        return {"ticker": t, "cik": None, "revenue_annual": [],
                "net_income_annual": [], "total_assets": [],
                "total_liabilities": [], "error": cik_info["error"]}

    cik = cik_info["cik"]
    try:
        facts = _get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            host="data.sec.gov",
        )

        return {
            "ticker":            t,
            "cik":               cik,
            "revenue_annual":    _extract_annual(facts, _REVENUE_TAGS),
            "net_income_annual": _extract_annual(facts, _NET_INCOME_TAGS),
            "total_assets":      _extract_annual(facts, _ASSETS_TAGS),
            "total_liabilities": _extract_annual(facts, _LIABILITIES_TAGS),
            "error":             None,
        }

    except Exception as exc:
        return {"ticker": t, "cik": cik, "revenue_annual": [],
                "net_income_annual": [], "total_assets": [],
                "total_liabilities": [], "error": str(exc)}