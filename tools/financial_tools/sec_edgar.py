"""
sec_edgar.py
------------
Tool for accessing, downloading, and parsing official SEC EDGAR filings.

Supports:
- 10-K  (annual reports)
- 10-Q  (quarterly reports)
- 8-K   (current/material events)

Uses the SEC EDGAR full-text search API (no API key required) and the
official EDGAR data APIs to locate CIK numbers, list filings, and
retrieve filing documents in JSON or plain-text format.

References
----------
- EDGAR Full-Text Search : https://efts.sec.gov/LATEST/search-index?q=...
- EDGAR Filing API       : https://data.sec.gov/submissions/CIK{cik}.json
- EDGAR XBRL API         : https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
"""

import re
import time
import requests
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EDGAR_BASE_URL      = "https://www.sec.gov"
EDGAR_DATA_API      = "https://data.sec.gov"
EDGAR_SEARCH_URL    = "https://efts.sec.gov/LATEST/search-index"

# SEC requires a descriptive User-Agent header; update the contact email.
HEADERS = {
    "User-Agent": "AlphaAgentNode/1.0 contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# Polite delay between requests to respect SEC rate-limit guidelines (10 req/s max).
REQUEST_DELAY_SECONDS = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, base: str = EDGAR_DATA_API) -> dict | None:
    """
    Perform a GET request with SEC-required headers and basic error handling.

    Parameters
    ----------
    url : str
        Absolute URL to request.
    params : dict | None
        Optional query parameters.
    base : str
        The base host to set in the Host header (switches between data.sec.gov
        and efts.sec.gov as needed).

    Returns
    -------
    dict | None
        Parsed JSON response, or None on failure.
    """
    time.sleep(REQUEST_DELAY_SECONDS)
    headers = dict(HEADERS)
    headers["Host"] = base.replace("https://", "").split("/")[0]
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _pad_cik(cik: int | str) -> str:
    """
    Zero-pad a CIK number to the 10-digit format required by EDGAR APIs.

    Parameters
    ----------
    cik : int | str
        Raw CIK number.

    Returns
    -------
    str
        10-digit zero-padded CIK string.
    """
    return str(cik).zfill(10)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> dict:
    """
    Resolve a stock ticker symbol to its SEC CIK (Central Index Key) number.

    Uses the EDGAR company-tickers JSON endpoint, which maps all exchange-listed
    ticker symbols to their corresponding CIK values.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (case-insensitive, e.g. "NVDA", "aapl").

    Returns
    -------
    dict
        A dictionary with:
        - "ticker"       (str)       : Normalised ticker (upper-case).
        - "cik"          (str)       : 10-digit zero-padded CIK.
        - "company_name" (str)       : Official company name from EDGAR.
        - "error"        (str | None): Error message if lookup failed.

    Examples
    --------
    >>> result = get_cik("NVDA")
    >>> result["cik"]
    '0001045810'
    """
    try:
        url  = f"{EDGAR_DATA_API}/submissions/company_tickers.json"
        data = _get(url)
        if data is None:
            return {"ticker": ticker, "cik": None, "company_name": None,
                    "error": "Failed to fetch company tickers index."}

        ticker_upper = ticker.upper()
        for _, entry in data.items():
            if entry.get("ticker", "").upper() == ticker_upper:
                cik = _pad_cik(entry["cik_str"])
                return {
                    "ticker":       ticker_upper,
                    "cik":          cik,
                    "company_name": entry.get("title", "N/A"),
                    "error":        None,
                }

        return {"ticker": ticker_upper, "cik": None, "company_name": None,
                "error": f"Ticker '{ticker_upper}' not found in EDGAR index."}

    except Exception as exc:
        return {"ticker": ticker, "cik": None, "company_name": None, "error": str(exc)}


def list_filings(ticker: str, form_type: str = "10-K", limit: int = 5) -> dict:
    """
    List the most recent SEC filings of a given type for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    form_type : str
        SEC form type to filter by. Common values: "10-K", "10-Q", "8-K".
        Defaults to "10-K".
    limit : int
        Maximum number of filings to return. Defaults to 5.

    Returns
    -------
    dict
        A dictionary with:
        - "ticker"    (str)        : Normalised ticker symbol.
        - "cik"       (str)        : 10-digit CIK.
        - "form_type" (str)        : The requested form type.
        - "filings"   (list[dict]) : Each dict contains:
            - "accession_number" (str) : EDGAR accession number (dashes included).
            - "filing_date"      (str) : Date filed (YYYY-MM-DD).
            - "report_date"      (str) : Period of report (YYYY-MM-DD).
            - "document_url"     (str) : URL to the filing index page.
        - "error"     (str | None)

    Examples
    --------
    >>> filings = list_filings("AAPL", form_type="10-K", limit=3)
    >>> filings["filings"][0]["filing_date"]
    '2024-11-01'
    """
    try:
        cik_info = get_cik(ticker)
        if cik_info["error"]:
            return {"ticker": ticker, "cik": None, "form_type": form_type,
                    "filings": [], "error": cik_info["error"]}

        cik = cik_info["cik"]
        url = f"{EDGAR_DATA_API}/submissions/CIK{cik}.json"
        data = _get(url)
        if data is None:
            return {"ticker": ticker, "cik": cik, "form_type": form_type,
                    "filings": [], "error": "Failed to fetch submission data from EDGAR."}

        recent = data.get("filings", {}).get("recent", {})
        forms       = recent.get("form",           [])
        acc_numbers = recent.get("accessionNumber", [])
        filed_dates = recent.get("filingDate",     [])
        report_dates= recent.get("reportDate",     [])

        filings = []
        for form, acc, filed, report in zip(forms, acc_numbers, filed_dates, report_dates):
            if form == form_type:
                acc_clean = acc.replace("-", "")
                doc_url   = (f"{EDGAR_BASE_URL}/Archives/edgar/data/"
                             f"{int(cik)}/{acc_clean}/{acc}-index.htm")
                filings.append({
                    "accession_number": acc,
                    "filing_date":      filed,
                    "report_date":      report,
                    "document_url":     doc_url,
                })
                if len(filings) >= limit:
                    break

        return {
            "ticker":    ticker.upper(),
            "cik":       cik,
            "form_type": form_type,
            "filings":   filings,
            "error":     None,
        }

    except Exception as exc:
        return {"ticker": ticker, "cik": None, "form_type": form_type,
                "filings": [], "error": str(exc)}


def get_filing_text(accession_number: str, cik: str) -> dict:
    """
    Download the plain-text content of a specific SEC filing document.

    Retrieves the primary document from the filing index and returns its
    raw text content, suitable for chunking and embedding into a vector store.

    Parameters
    ----------
    accession_number : str
        The EDGAR accession number with dashes (e.g. "0001045810-24-000010").
    cik : str
        The 10-digit zero-padded CIK number for the filer.

    Returns
    -------
    dict
        A dictionary with:
        - "accession_number" (str)        : The requested accession number.
        - "cik"              (str)        : The filer's CIK.
        - "text"             (str | None) : Raw filing text content.
        - "word_count"       (int)        : Approximate word count of the text.
        - "error"            (str | None)

    Notes
    -----
    SEC filings can be very large (100k–500k words for 10-K). Callers should
    chunk the returned text before embedding. Returns None for text if the
    primary document cannot be located or is in an unsupported format.
    """
    try:
        acc_clean = accession_number.replace("-", "")
        cik_int   = int(cik)

        # Fetch the filing index JSON
        index_url = (f"{EDGAR_DATA_API}/submissions/CIK{cik}.json")
        idx_url   = (f"{EDGAR_BASE_URL}/Archives/edgar/data/"
                     f"{cik_int}/{acc_clean}/{accession_number}-index.json")

        headers = {
            "User-Agent": HEADERS["User-Agent"],
            "Host": "www.sec.gov",
        }
        time.sleep(REQUEST_DELAY_SECONDS)
        idx_resp = requests.get(idx_url, headers=headers, timeout=15)

        if idx_resp.status_code != 200:
            # Fall back: try plain .htm index
            return {"accession_number": accession_number, "cik": cik,
                    "text": None, "word_count": 0,
                    "error": f"Index fetch failed (HTTP {idx_resp.status_code})."}

        idx_data    = idx_resp.json()
        documents   = idx_data.get("documents", [])
        primary_doc = None

        # Prefer the document labelled as the 10-K/10-Q complete submission
        for doc in documents:
            if doc.get("type") in ("10-K", "10-Q", "8-K") and doc.get("documentUrl"):
                primary_doc = doc
                break
        if primary_doc is None and documents:
            primary_doc = documents[0]

        if primary_doc is None:
            return {"accession_number": accession_number, "cik": cik,
                    "text": None, "word_count": 0,
                    "error": "No primary document found in filing index."}

        doc_url = f"{EDGAR_BASE_URL}{primary_doc['documentUrl']}"
        time.sleep(REQUEST_DELAY_SECONDS)
        doc_resp = requests.get(doc_url, headers=headers, timeout=30)
        doc_resp.raise_for_status()

        raw_text = doc_resp.text
        # Strip HTML/XBRL tags for plain text
        clean_text = re.sub(r"<[^>]+>", " ", raw_text)
        clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()

        return {
            "accession_number": accession_number,
            "cik":              cik,
            "text":             clean_text,
            "word_count":       len(clean_text.split()),
            "error":            None,
        }

    except Exception as exc:
        return {"accession_number": accession_number, "cik": cik,
                "text": None, "word_count": 0, "error": str(exc)}


def get_xbrl_financials(ticker: str) -> dict:
    """
    Retrieve structured financial data from SEC EDGAR's XBRL company facts API.

    The XBRL API provides machine-readable financial statement data including
    revenue, net income, assets, and liabilities across all reported periods.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        A dictionary with:
        - "ticker"          (str)
        - "cik"             (str)
        - "revenue_annual"  (list[dict]) : [{period_end, value, unit}]
        - "net_income_annual" (list[dict])
        - "total_assets"    (list[dict])
        - "total_liabilities" (list[dict])
        - "error"           (str | None)

    Notes
    -----
    Values are in USD unless otherwise specified in the "unit" field.
    Only the 10 most recent annual data points are returned per metric.
    """
    try:
        cik_info = get_cik(ticker)
        if cik_info["error"]:
            return {"ticker": ticker, "cik": None, "error": cik_info["error"]}

        cik  = cik_info["cik"]
        url  = f"{EDGAR_DATA_API}/api/xbrl/companyfacts/CIK{cik}.json"
        data = _get(url)
        if data is None:
            return {"ticker": ticker, "cik": cik, "error": "XBRL data fetch failed."}

        facts_us_gaap = data.get("facts", {}).get("us-gaap", {})

        def _extract(concept: str, form: str = "10-K", n: int = 10) -> list[dict]:
            """Extract the last *n* annual values for a given XBRL concept."""
            concept_data = facts_us_gaap.get(concept, {})
            units        = concept_data.get("units", {})
            usd_entries  = units.get("USD", [])
            # Filter to annual filings only
            annual = [e for e in usd_entries if e.get("form") == form and e.get("end")]
            # Sort by period end descending
            annual.sort(key=lambda x: x["end"], reverse=True)
            return [
                {"period_end": e["end"], "value": e["val"], "unit": "USD"}
                for e in annual[:n]
            ]

        return {
            "ticker":               ticker.upper(),
            "cik":                  cik,
            "revenue_annual":       _extract("Revenues") or _extract("RevenueFromContractWithCustomerExcludingAssessedTax"),
            "net_income_annual":    _extract("NetIncomeLoss"),
            "total_assets":         _extract("Assets"),
            "total_liabilities":    _extract("Liabilities"),
            "error":                None,
        }

    except Exception as exc:
        return {"ticker": ticker, "cik": None, "error": str(exc)}
