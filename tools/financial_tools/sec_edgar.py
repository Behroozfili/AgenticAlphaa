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
from datetime import date
import html as _html
import logging
from collections.abc import Mapping
from typing import Any

import requests

from core.forecast_contract import ForecastContext
from tools.financial_tools import (
    deterministic_financial_reference, financial_lifecycle_event,
    require_eligible_filed_at,
    require_financial_context,
)

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
_companyfacts_cache: dict[str, dict[str, Any]] = {}
_submissions_cache: dict[str, dict[str, Any]] = {}


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

        except requests.RequestException:
            if attempt < _MAX_RETRIES:
                wait = _RETRY_BACKOFF * attempt
                log.warning(
                    "SEC_REQUEST_RETRY attempt=%d max_attempts=%d wait_seconds=%.1f",
                    attempt, _MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                break

    raise RuntimeError("SEC_REQUEST_FAILED")


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

def get_cik(
    ticker: str,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
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
    context, identity = require_financial_context(forecast_context, run_id, ticker=t)
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
            "dataset_version": context.dataset_version, "run_id": identity,
            "error":        None,
        }
    except Exception:
        return {"ticker": t, "cik": None, "company_name": None, "error_code": "SEC_CIK_UNAVAILABLE", "error": "SEC CIK data is unavailable."}


def list_filings(
    ticker: str, form_type: str = "10-K", limit: int = 5,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
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

    context, identity = require_financial_context(forecast_context, run_id, ticker=t)
    started_at = time.perf_counter()
    cik_info = get_cik(t, context, identity)
    if cik_info.get("error"):
        financial_lifecycle_event(
            context=context, run_id=identity, boundary="sec_submissions",
            started_at=started_at, outcome="unavailable",
            counts={"result_count": 0, "missing_count": 1},
            missing=True, degraded=True, error_code="SEC_SUBMISSIONS_UNAVAILABLE",
        )
        return {"ticker": t, "cik": None, "form_type": form_type,
                "filings": [], "error": cik_info["error"]}

    cik = cik_info["cik"]
    try:
        data = _submissions_cache.get(cik)
        if data is None:
            data = _get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                        host="data.sec.gov")
            _submissions_cache[cik] = data

        recent = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        fil_dates  = recent.get("filingDate", [])
        rep_dates  = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        cik_int = str(int(cik))  # un-padded for Archives URLs
        filings: list[dict] = []

        for i, form in enumerate(forms):
            if form not in {form_type, f"{form_type}/A"}:
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

            filed_at = fil_dates[i] if i < len(fil_dates) else None
            try:
                filed_at = require_eligible_filed_at(filed_at, context)
            except Exception:
                continue
            report_date = rep_dates[i] if i < len(rep_dates) else None
            filings.append({
                "accession_number": accession,
                "filing_date":      filed_at,
                "filed_at":         filed_at,
                "report_date":      report_date,
                "period_end":       report_date,
                "filing_type":      form,
                "primary_document": primary_docs[i] if i < len(primary_docs) else None,
                "document_url":     document_url,
                "dataset_version":  context.dataset_version,
                "evidence_id": deterministic_financial_reference(
                    provider="sec_submissions", ticker=t,
                    accession_number=accession, period_end=report_date,
                    filed_at=filed_at, dataset_version=context.dataset_version,
                ),
            })

        # A future amendment is invisible. For duplicate reports/forms select
        # the newest version actually filed by the cutoff.
        selected: dict[tuple[str | None, str], dict[str, Any]] = {}
        for filing in filings:
            base_form = str(filing.get("filing_type") or "").removesuffix("/A")
            key = (filing.get("period_end"), base_form)
            current = selected.get(key)
            if current is None or (
                str(filing.get("filed_at") or ""), str(filing.get("accession_number") or "")
            ) > (
                str(current.get("filed_at") or ""), str(current.get("accession_number") or "")
            ):
                selected[key] = filing
        filings = sorted(
            selected.values(),
            key=lambda item: (str(item.get("filed_at") or ""), str(item.get("accession_number") or "")),
            reverse=True,
        )[:max(0, int(limit))]

        result = {
            "ticker":    t,
            "cik":       cik,
            "form_type": form_type,
            "filings":   filings,
            "dataset_version": context.dataset_version, "run_id": identity,
            "error":     None,
        }
        financial_lifecycle_event(
            context=context, run_id=identity, boundary="sec_submissions",
            started_at=started_at, outcome="success",
            counts={
                "candidate_count": len(forms), "eligible_count": len(filings),
                "result_count": len(filings),
                "rejected_count": max(0, len(forms) - len(filings)),
            },
        )
        return result

    except Exception:
        financial_lifecycle_event(
            context=context, run_id=identity, boundary="sec_submissions",
            started_at=started_at, outcome="fail_closed",
            counts={"result_count": 0, "missing_count": 1},
            missing=True, degraded=True, error_code="SEC_SUBMISSIONS_UNAVAILABLE",
        )
        return {"ticker": t, "cik": cik, "form_type": form_type,
                "filings": [], "error_code": "SEC_SUBMISSIONS_UNAVAILABLE", "error": "SEC submissions are unavailable."}


_FILING_MIN_WORDS = {
    "10-K": 500, "10-K/A": 500,
    "10-Q": 250, "10-Q/A": 250,
    "8-K": 100, "8-K/A": 100,
}


def _looks_like_exhibit(filename: str) -> bool:
    """Identify common exhibit, XBRL resource, and consent filenames."""
    stem = filename.lower().rsplit("/", 1)[-1]
    return bool(
        re.match(r"^(ex(?:hibit)?[-_\d.]|r\d+|.*-index)", stem)
        or any(token in stem for token in (
            "auditorconsent", "consentletter", "exhibit", "graphic",
        ))
    )


def _filing_document_candidates(
    items: list[dict[str, Any]],
    *,
    primary_document: str | None,
    form_type: str | None,
    accession_number: str,
) -> list[str]:
    """Rank document candidates using filing metadata, filename, and size."""
    html_items = [
        item for item in items
        if str(item.get("name", "")).lower().endswith((".htm", ".html"))
    ]
    available = {str(item.get("name")): item for item in html_items}
    candidates: list[str] = []

    primary = (primary_document or "").strip().rsplit("/", 1)[-1]
    if primary and primary in available and not _looks_like_exhibit(primary):
        candidates.append(primary)

    form_token = re.sub(r"[^a-z0-9]", "", (form_type or "").lower())

    def score(item: dict[str, Any]) -> tuple[int, int, str]:
        name = str(item.get("name", ""))
        compact_name = re.sub(r"[^a-z0-9]", "", name.lower())
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        return (int(bool(form_token and form_token in compact_name)), size, name)

    ranked = sorted(
        (item for item in html_items if not _looks_like_exhibit(str(item.get("name", "")))),
        key=score,
        reverse=True,
    )
    for item in ranked:
        name = str(item.get("name", ""))
        if name and name not in candidates:
            candidates.append(name)

    # The complete submission is a final recovery candidate, not an
    # automatic success: it must pass the same content validation.
    candidates.append(f"{accession_number}.txt")
    return candidates


def _validate_filing_text(text: str, form_type: str | None) -> tuple[bool, list[str]]:
    """Check that a candidate is substantial and resembles the requested form."""
    normalized_form = (form_type or "").upper().strip()
    word_count = len(text.split())
    minimum = _FILING_MIN_WORDS.get(normalized_form, 100)
    reasons: list[str] = []
    if word_count < minimum:
        reasons.append(f"word_count_below_{minimum}")

    lowered = re.sub(r"\s+", " ", text.lower())
    markers: tuple[str, ...] = ()
    if normalized_form.startswith("10-K"):
        markers = ("form 10-k", "annual report", "item 1", "business")
    elif normalized_form.startswith("10-Q"):
        markers = ("form 10-q", "quarterly report", "financial statements", "item 1")
    elif normalized_form.startswith("8-K"):
        markers = ("form 8-k", "current report", "item 1.01", "item 2.02")
    if markers and not any(marker in lowered for marker in markers):
        reasons.append("missing_expected_form_markers")
    return not reasons, reasons


def get_filing_text(
    accession_number: str,
    cik: str,
    primary_document: str | None = None,
    form_type: str | None = None,
    filed_at: str | None = None,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
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
    context, identity = require_financial_context(forecast_context, run_id)
    started_at = time.perf_counter()
    eligible_filed_at = require_eligible_filed_at(filed_at, context)
    try:
        cik_int = str(int(cik))           # un-padded for Archives path
        cik_padded = _pad_cik(cik)
        accession_nodash = acc.replace("-", "")
        base = (f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{accession_nodash}")

        # 1) Use the filing's index.json to locate the primary document.
        index = _get(f"{base}/index.json", host="www.sec.gov")
        items = index.get("directory", {}).get("item", [])

        candidates = _filing_document_candidates(
            items,
            primary_document=primary_document,
            form_type=form_type,
            accession_number=acc,
        )
        attempts: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                raw = _get(f"{base}/{candidate}", host="www.sec.gov", as_json=False)
                text = _strip_html(raw)
                valid, reasons = _validate_filing_text(text, form_type)
                attempts.append({
                    "document": candidate,
                    "word_count": len(text.split()),
                    "valid": valid,
                    "reasons": reasons,
                })
                if not valid:
                    continue
                result = {
                    "accession_number": acc,
                    "cik": cik_padded,
                    "text": text,
                    "word_count": len(text.split()),
                    "selected_document": candidate,
                    "validation_status": "VALID",
                    "attempted_documents": attempts,
                    "filed_at": eligible_filed_at,
                    "dataset_version": context.dataset_version,
                    "run_id": identity,
                    "evidence_id": deterministic_financial_reference(
                        provider="sec_filing", ticker=context.ticker,
                        accession_number=acc, filed_at=eligible_filed_at,
                        dataset_version=context.dataset_version,
                    ),
                    "error": None,
                }
                financial_lifecycle_event(
                    context=context, run_id=identity,
                    boundary="sec_filing_selection_and_text",
                    started_at=started_at, outcome="success",
                    counts={
                        "candidate_count": len(candidates),
                        "result_count": 1, "eligible_count": 1,
                        "rejected_count": max(0, len(attempts) - 1),
                        "accession_count": 1,
                    },
                )
                return result
            except Exception:
                attempts.append({
                    "document": candidate,
                    "valid": False,
                    "reasons": ["FETCH_FAILED"],
                })

        result = {
            "accession_number": acc,
            "cik": cik_padded,
            "text": None,
            "word_count": 0,
            "selected_document": None,
            "validation_status": "INVALID",
            "attempted_documents": attempts,
            "error": "No candidate document passed filing validation.",
        }
        financial_lifecycle_event(
            context=context, run_id=identity,
            boundary="sec_filing_selection_and_text",
            started_at=started_at, outcome="rejected",
            counts={
                "candidate_count": len(candidates), "result_count": 0,
                "eligible_count": 0, "rejected_count": len(attempts),
                "missing_count": 1,
            },
            missing=True, degraded=True,
        )
        return result

    except Exception:
        financial_lifecycle_event(
            context=context, run_id=identity,
            boundary="sec_filing_selection_and_text",
            started_at=started_at, outcome="fail_closed",
            counts={"result_count": 0, "missing_count": 1},
            missing=True, degraded=True, error_code="SEC_FILING_UNAVAILABLE",
        )
        return {"accession_number": acc, "cik": str(cik),
                "text": None, "word_count": 0,
                "validation_status": "INVALID", "error_code": "SEC_FILING_UNAVAILABLE", "error": "SEC filing is unavailable."}


# Candidate XBRL concept tags, in priority order, per metric. Companies
# tag the same economic figure differently, so we try several.
_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
_NET_INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
_OPERATING_INCOME_TAGS = ["OperatingIncomeLoss"]
# Cost of revenue — required to compute gross margin. Companies tag this
# under several concepts depending on whether they sell goods, services, or
# both; _extract_annual tries each in order and uses the first with data.
_COST_OF_REVENUE_TAGS = [
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfServices",
]
_PRETAX_INCOME_TAGS = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments", "IncomeLossFromContinuingOperationsBeforeIncomeTaxes"]
_INCOME_TAX_TAGS = ["IncomeTaxExpenseBenefit"]
_OTHER_INCOME_TAGS = ["NonoperatingIncomeExpense", "OtherNonoperatingIncomeExpense", "InterestAndOtherNet"]
_UNUSUAL_ITEMS_TAGS = ["RestructuringCharges", "ImpairmentOfLongLivedAssetsHeldForUse", "GainLossOnSaleOfPropertyPlantEquipment", "LitigationSettlementAmountAwardedToOtherParty"]
_ASSETS_TAGS = ["Assets"]
_LIABILITIES_TAGS = ["Liabilities"]

# Cash-flow-statement tags — shared foundation for both Capital Allocation
# analysis (buybacks/dividends/capex) and DCF's Free Cash Flow input
# (FCF = operating cash flow − capex). Multiple candidate tags per metric
# because companies tag the same line item differently; _extract_annual
# already tries each in order and uses the first with data, same pattern
# as the existing revenue/net-income tags above.
_OPERATING_CASH_FLOW_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
_CAPEX_TAGS = [
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquireProductiveAssets",
]
_DIVIDENDS_PAID_TAGS = [
    "PaymentsOfDividends",
    "PaymentsOfDividendsCommonStock",
]
_BUYBACKS_TAGS = [
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsForRepurchaseOfEquity",
]


def _extract_annual_legacy(facts: dict, tags: list[str]) -> list[dict]:
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
            start = row.get("start")
            duration_months = None
            if start:
                try:
                    duration_months = round((date.fromisoformat(end) - date.fromisoformat(start)).days / 30.4375)
                except ValueError:
                    duration_months = None
            by_period[end] = {
                "period_end": end,
                "period_start": start,
                "duration_months": duration_months,
                "value":      float(val),
                "unit":       unit_key,
                "fiscal_year": row.get("fy"),
                "fiscal_quarter": row.get("fp"),
                "filing_type": form,
                "accession_number": row.get("accn"),
                "filed_at": row.get("filed"),
                "metric_scope": "duration" if start else "instant",
                "gaap_status": "GAAP",
                "xbrl_concept": tag,
            }

        if by_period:
            ordered = sorted(by_period.values(),
                             key=lambda r: r["period_end"], reverse=True)
            return ordered[:10]

    return []


def _extract_annual(
    facts: dict,
    tags: list[str],
    metric_scope: str = "duration",
    *,
    context: ForecastContext,
) -> list[dict]:
    """Select the strongest annual facts after evaluating every candidate concept."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    candidates: list[dict[str, Any]] = []

    for concept_priority, tag in enumerate(tags):
        concept = us_gaap.get(tag)
        if not concept:
            continue
        for unit_key, rows in concept.get("units", {}).items():
            for row in rows:
                form = str(row.get("form") or "").upper()
                end = row.get("end")
                start = row.get("start")
                val = row.get("val")
                if form not in {"10-K", "10-K/A"} or end is None or val is None:
                    continue
                try:
                    filed_at = require_eligible_filed_at(row.get("filed"), context)
                except Exception:
                    continue
                if row.get("segment") or row.get("dimensions"):
                    continue

                duration_days: int | None = None
                try:
                    end_date = date.fromisoformat(str(end))
                    if start:
                        duration_days = (end_date - date.fromisoformat(str(start))).days
                except (TypeError, ValueError):
                    continue

                if metric_scope == "duration":
                    # 10-K documents contain quarterly comparative facts too.
                    if duration_days is None or not 300 <= duration_days <= 430:
                        continue
                elif metric_scope == "instant":
                    if start is not None:
                        continue
                else:
                    raise ValueError(f"Unsupported metric_scope: {metric_scope}")

                candidates.append({
                    "period_end": str(end),
                    "period_start": str(start) if start is not None else None,
                    "duration_months": (
                        round(duration_days / 30.4375) if duration_days is not None else None
                    ),
                    "value": float(val),
                    "unit": unit_key,
                    "fiscal_year": row.get("fy"),
                    "fiscal_quarter": row.get("fp"),
                    "filing_type": form,
                    "accession_number": row.get("accn"),
                    "filed_at": filed_at,
                    "metric_scope": metric_scope,
                    "gaap_status": "GAAP",
                    "xbrl_concept": tag,
                    "validation_status": "VALID",
                    "_concept_priority": concept_priority,
                })

    def candidate_rank(row: dict[str, Any]) -> tuple[int, int, str, int, str, int]:
        return (
            int(row.get("unit") == "USD"),
            int(row.get("fiscal_quarter") == "FY"),
            str(row.get("filed_at") or ""),
            int(row.get("filing_type") == "10-K/A"),
            str(row.get("accession_number") or ""),
            -int(row.get("_concept_priority", 0)),
        )

    # Select per exact period across all concepts. Recent periods are sorted
    # only after semantic/unit/restatement ranking has chosen each fact.
    by_period: dict[tuple[str | None, str], dict[str, Any]] = {}
    for row in candidates:
        key = (row.get("period_start"), row["period_end"])
        current = by_period.get(key)
        if current is None or candidate_rank(row) > candidate_rank(current):
            by_period[key] = row

    ordered = sorted(
        by_period.values(),
        key=lambda row: (row["period_end"], candidate_rank(row)),
        reverse=True,
    )
    for row in ordered:
        concept_priority = int(row.get("_concept_priority", 0))
        row["validation_status"] = "VALID_PRIMARY" if concept_priority == 0 else "FALLBACK"
        row["source"] = "sec_xbrl"
        row["selected_source"] = f"sec_xbrl:{row.get('xbrl_concept')}"
        row["exact_period"] = {
            "basis": metric_scope,
            "period_start": row.get("period_start"),
            "period_end": row.get("period_end"),
        }
        row["fallback_reason"] = (
            None if concept_priority == 0
            else f"Preferred concept {tags[0]} had no eligible fact for this exact period; selected {row.get('xbrl_concept')}."
        )
        row["confidence"] = "high" if concept_priority == 0 else "medium"
        row["is_approximation"] = False
        row["eligible_as_of"] = context.to_dict()["as_of"]
        row["dataset_version"] = context.dataset_version
        row["evidence_id"] = deterministic_financial_reference(
            provider="sec_companyfacts", ticker=context.ticker,
            accession_number=row.get("accession_number"),
            metric=row.get("xbrl_concept"), period_end=row.get("period_end"),
            filed_at=row.get("filed_at"), dataset_version=context.dataset_version,
        )
        row.pop("_concept_priority", None)
    return ordered[:10]


def _extract_quarterly(
    facts: dict, tags: list[str], metric_scope: str = "duration", *,
    context: ForecastContext,
) -> list[dict]:
    """Select eligible 10-Q facts, newest eligible amendment per exact period."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    candidates: list[dict[str, Any]] = []
    for concept_priority, tag in enumerate(tags):
        concept = us_gaap.get(tag) or {}
        for unit_key, rows in concept.get("units", {}).items():
            for source_row in rows:
                row = dict(source_row)
                form = str(row.get("form") or "").upper()
                if form not in {"10-Q", "10-Q/A"} or row.get("end") is None or row.get("val") is None:
                    continue
                if row.get("segment") or row.get("dimensions"):
                    continue
                try:
                    filed_at = require_eligible_filed_at(row.get("filed"), context)
                    end = date.fromisoformat(str(row["end"]))
                    start = date.fromisoformat(str(row["start"])) if row.get("start") else None
                except Exception:
                    continue
                duration_days = (end - start).days if start else None
                if metric_scope == "duration" and (duration_days is None or not 70 <= duration_days <= 120):
                    continue
                if metric_scope == "instant" and start is not None:
                    continue
                candidates.append({
                    "period_end": end.isoformat(),
                    "period_start": start.isoformat() if start else None,
                    "duration_months": round(duration_days / 30.4375) if duration_days is not None else None,
                    "value": float(row["val"]), "unit": unit_key,
                    "fiscal_year": row.get("fy"), "fiscal_quarter": row.get("fp"),
                    "filing_type": form, "accession_number": row.get("accn"),
                    "filed_at": filed_at, "metric_scope": metric_scope,
                    "gaap_status": "GAAP", "xbrl_concept": tag,
                    "_concept_priority": concept_priority,
                })
    by_period: dict[tuple[str | None, str], dict[str, Any]] = {}
    for row in candidates:
        key = (row.get("period_start"), row["period_end"])
        current = by_period.get(key)
        rank = (str(row["filed_at"]), int(row["filing_type"] == "10-Q/A"), -int(row["_concept_priority"]), str(row.get("accession_number") or ""))
        current_rank = (str(current["filed_at"]), int(current["filing_type"] == "10-Q/A"), -int(current["_concept_priority"]), str(current.get("accession_number") or "")) if current else None
        if current is None or rank > current_rank:
            by_period[key] = row
    ordered = sorted(by_period.values(), key=lambda row: (row["period_end"], row["filed_at"]), reverse=True)
    for row in ordered:
        row.update({
            "validation_status": "VALID_PRIMARY" if row["_concept_priority"] == 0 else "FALLBACK",
            "source": "sec_xbrl", "selected_source": f"sec_xbrl:{row['xbrl_concept']}",
            "exact_period": {"basis": metric_scope, "period_start": row.get("period_start"), "period_end": row["period_end"]},
            "confidence": "high" if row["_concept_priority"] == 0 else "medium",
            "is_approximation": False, "eligible_as_of": context.to_dict()["as_of"],
            "dataset_version": context.dataset_version,
            "evidence_id": deterministic_financial_reference(
                provider="sec_companyfacts", ticker=context.ticker,
                accession_number=row.get("accession_number"), metric=row.get("xbrl_concept"),
                period_end=row.get("period_end"), filed_at=row.get("filed_at"),
                dataset_version=context.dataset_version,
            ),
        })
        row.pop("_concept_priority", None)
    return ordered[:20]


def get_xbrl_financials(
    ticker: str,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Retrieve structured financial statement data from the SEC EDGAR XBRL API.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        - "ticker"                    (str)
        - "cik"                       (str)
        - "revenue_annual"            (list[dict]) [{period_end, value, unit}]
        - "net_income_annual"         (list[dict])
        - "total_assets"              (list[dict])
        - "total_liabilities"         (list[dict])
        - "operating_cash_flow_annual" (list[dict])  Cash from operations —
                                        used for Capital Allocation and as
                                        the starting point for DCF's FCF.
        - "capex_annual"              (list[dict])  Capital expenditures
                                        (payments to acquire PP&E).
                                        FCF = operating_cash_flow − capex.
        - "dividends_paid_annual"     (list[dict])  Cash dividends paid.
        - "buybacks_annual"           (list[dict])  Common stock repurchases.
        - "error"                     (str | None)

        Values are in USD. Up to the 10 most recent annual data points per
        metric are returned (newest first). Any series may come back empty
        if the company doesn't tag that concept (e.g. no dividends paid) —
        an empty list means "none reported", not a fetch failure.
    """
    t = (ticker or "").upper().strip()

    empty_series = {
        "revenue_annual": [], "net_income_annual": [],
        "operating_income_annual": [], "cost_of_revenue_annual": [],
        "pretax_income_annual": [],
        "income_tax_annual": [], "other_income_annual": [], "unusual_items_annual": [],
        "total_assets": [], "total_liabilities": [],
        "operating_cash_flow_annual": [], "capex_annual": [],
        "dividends_paid_annual": [], "buybacks_annual": [],
        "revenue_quarterly": [], "net_income_quarterly": [],
        "operating_income_quarterly": [], "cost_of_revenue_quarterly": [],
        "operating_cash_flow_quarterly": [], "capex_quarterly": [],
        "total_assets_quarterly": [], "total_liabilities_quarterly": [],
    }

    context, identity = require_financial_context(forecast_context, run_id, ticker=t)
    started_at = time.perf_counter()

    cik_info = get_cik(t, context, identity)
    if cik_info.get("error"):
        for boundary in ("sec_companyfacts", "sec_restatement_gate"):
            financial_lifecycle_event(
                context=context, run_id=identity, boundary=boundary,
                started_at=started_at, outcome="unavailable",
                counts={"result_count": 0, "missing_count": 1},
                missing=True, degraded=True, error_code="SEC_COMPANYFACTS_UNAVAILABLE",
            )
        return {"ticker": t, "cik": None, **empty_series, "error": cik_info["error"]}

    cik = cik_info["cik"]
    try:
        facts = _companyfacts_cache.get(cik)
        if facts is None:
            facts = _get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                host="data.sec.gov",
            )
            _companyfacts_cache[cik] = facts

        audited_tags = set().union(
            _REVENUE_TAGS, _NET_INCOME_TAGS, _OPERATING_INCOME_TAGS,
            _COST_OF_REVENUE_TAGS, _PRETAX_INCOME_TAGS, _INCOME_TAX_TAGS,
            _OTHER_INCOME_TAGS, _UNUSUAL_ITEMS_TAGS, _ASSETS_TAGS,
            _LIABILITIES_TAGS, _OPERATING_CASH_FLOW_TAGS, _CAPEX_TAGS,
            _DIVIDENDS_PAID_TAGS, _BUYBACKS_TAGS,
        )
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        candidate_count = sum(
            len(rows)
            for tag in audited_tags
            for rows in (
                us_gaap.get(tag, {}).get("units", {}).values()
                if isinstance(us_gaap.get(tag), dict) else []
            )
            if isinstance(rows, list)
        )
        result = {
            "ticker":                     t,
            "cik":                        cik,
            "revenue_annual":             _extract_annual(facts, _REVENUE_TAGS, context=context),
            "net_income_annual":          _extract_annual(facts, _NET_INCOME_TAGS, context=context),
            "operating_income_annual":    _extract_annual(facts, _OPERATING_INCOME_TAGS, context=context),
            "cost_of_revenue_annual":     _extract_annual(facts, _COST_OF_REVENUE_TAGS, context=context),
            "pretax_income_annual":       _extract_annual(facts, _PRETAX_INCOME_TAGS, context=context),
            "income_tax_annual":          _extract_annual(facts, _INCOME_TAX_TAGS, context=context),
            "other_income_annual":        _extract_annual(facts, _OTHER_INCOME_TAGS, context=context),
            "unusual_items_annual":       _extract_annual(facts, _UNUSUAL_ITEMS_TAGS, context=context),
            "total_assets":               _extract_annual(facts, _ASSETS_TAGS, "instant", context=context),
            "total_liabilities":          _extract_annual(facts, _LIABILITIES_TAGS, "instant", context=context),
            "operating_cash_flow_annual": _extract_annual(facts, _OPERATING_CASH_FLOW_TAGS, context=context),
            "capex_annual":               _extract_annual(facts, _CAPEX_TAGS, context=context),
            "dividends_paid_annual":      _extract_annual(facts, _DIVIDENDS_PAID_TAGS, context=context),
            "buybacks_annual":            _extract_annual(facts, _BUYBACKS_TAGS, context=context),
            "revenue_quarterly":          _extract_quarterly(facts, _REVENUE_TAGS, context=context),
            "net_income_quarterly":       _extract_quarterly(facts, _NET_INCOME_TAGS, context=context),
            "operating_income_quarterly": _extract_quarterly(facts, _OPERATING_INCOME_TAGS, context=context),
            "cost_of_revenue_quarterly":  _extract_quarterly(facts, _COST_OF_REVENUE_TAGS, context=context),
            "operating_cash_flow_quarterly": _extract_quarterly(facts, _OPERATING_CASH_FLOW_TAGS, context=context),
            "capex_quarterly":            _extract_quarterly(facts, _CAPEX_TAGS, context=context),
            "total_assets_quarterly":     _extract_quarterly(facts, _ASSETS_TAGS, "instant", context=context),
            "total_liabilities_quarterly": _extract_quarterly(facts, _LIABILITIES_TAGS, "instant", context=context),
            "as_of": context.to_dict()["as_of"], "dataset_version": context.dataset_version,
            "run_id": identity, "provider": "sec_companyfacts",
            "error":                      None,
        }
        selected_count = sum(
            len(value) for value in result.values() if isinstance(value, list)
        )
        financial_lifecycle_event(
            context=context, run_id=identity, boundary="sec_companyfacts",
            started_at=started_at, outcome="success",
            counts={"candidate_count": candidate_count, "result_count": selected_count},
        )
        financial_lifecycle_event(
            context=context, run_id=identity, boundary="sec_restatement_gate",
            started_at=started_at, outcome="success",
            counts={
                "candidate_count": candidate_count, "eligible_count": selected_count,
                "rejected_count": max(0, candidate_count - selected_count),
                "result_count": selected_count,
            },
        )
        return result

    except Exception:
        for boundary in ("sec_companyfacts", "sec_restatement_gate"):
            financial_lifecycle_event(
                context=context, run_id=identity, boundary=boundary,
                started_at=started_at, outcome="fail_closed",
                counts={"result_count": 0, "missing_count": 1},
                missing=True, degraded=True, error_code="SEC_COMPANYFACTS_UNAVAILABLE",
            )
        return {"ticker": t, "cik": cik, **empty_series, "error_code": "SEC_COMPANYFACTS_UNAVAILABLE", "error": "SEC CompanyFacts are unavailable."}
