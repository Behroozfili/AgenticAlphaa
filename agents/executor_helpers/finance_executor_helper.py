from __future__ import annotations
import json
import re
from typing import Any
import logging
import math

logger = logging.getLogger(__name__)

INDUSTRY_PEERS: dict[str, list[str]] = {
    "Semiconductors":                       ["NVDA", "INTC", "QCOM", "AVGO"],
    "Consumer Electronics":                 ["MSFT", "GOOGL", "SSNLF"],
    "Software—Infrastructure":              ["MSFT", "ORCL", "CRM"],
    "Software—Application":                 ["CRM", "ADBE", "INTU"],
    "Internet Content & Information":       ["GOOGL", "META", "MSFT"],
    "Internet Retail":                      ["WMT", "TGT", "EBAY"],
    "Auto Manufacturers":                   ["GM", "F", "TM"],
    "Banks—Diversified":                    ["BAC", "WFC", "C"],
    "Aerospace & Defense":                  ["LMT", "NOC", "RTX"],
    "Beverages—Non-Alcoholic":              ["PEP", "KDP"],
    "Discount Stores":                      ["WMT", "COST", "TGT"],
    "Telecom Services":                     ["VZ", "TMUS"],
}


def infer_peers(sector: str, industry: str, ticker: str) -> list[str]:
    """Look up static peers by industry, excluding the ticker itself."""
    peers = INDUSTRY_PEERS.get(industry, [])
    return [p for p in peers if p.upper() != ticker.upper()][:5]


# Common English / finance tokens that look like tickers but aren't, used to
# filter the regex fallback in extract_ticker().
_TICKER_STOP_WORDS: frozenset[str] = frozenset({
    "A", "I", "THE", "AND", "OR", "IN", "OF", "FOR", "AI", "Q1",
    "Q2", "Q3", "Q4", "US", "SEC", "PE", "YOY", "TTM", "EPS",
    "RAG", "NLP", "ETF", "IPO", "GDP", "CEO", "CFO", "MCP",
})


def extract_ticker(
    task_query: str,
    directives: dict[str, Any],
    financial_summary: dict[str, Any] | None = None,
) -> str | None:
    
    if directives.get("ticker"):
        return str(directives["ticker"]).upper()

    if financial_summary and financial_summary.get("ticker"):
        return str(financial_summary["ticker"]).upper()

  
    candidates = re.findall(r"\b([A-Z]{1,5})\b", task_query)
    for candidate in candidates:
        if candidate not in _TICKER_STOP_WORDS:
            logger.info("Inferred ticker from task_query: %s", candidate)
            return candidate

    logger.warning("Could not infer ticker from task_query: '%s'", task_query)
    return None


def confidence_for(period_tag: str | None, is_approximation: bool = False) -> str:
   
    if is_approximation:
        return "low"
    if not period_tag:
        return "unknown"
    if period_tag.startswith("MIXED(") or period_tag == "mixed":
        return "medium"
    return "high"


def annotate_confidence(calculated_ratios: dict[str, Any]) -> None:
    state_confidence = {
        "VALID_PRIMARY": "high",
        "FALLBACK": "medium",
        "DEGRADED": "low",
        "UNAVAILABLE": "none",
    }
    for ratio in calculated_ratios.values():
        if isinstance(ratio, dict) and "_period" in ratio:
            ratio.setdefault(
                "confidence",
                state_confidence.get(
                    str(ratio.get("validation_status")),
                    confidence_for(ratio.get("_period"), ratio.get("is_approximation", False)),
                ),
            )


def pct_change(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100.0


def direction(pct: float | None, flat_band_pct: float = 2.0) -> str:
    """Classify a % change into 'up' / 'down' / 'flat' / 'insufficient_data'."""
    if pct is None:
        return "insufficient_data"
    if pct > flat_band_pct:
        return "up"
    if pct < -flat_band_pct:
        return "down"
    return "flat"


def build_consistency_checks(raw: dict[str, Any], calc: dict[str, Any]) -> dict[str, Any]:
  
    checks: dict[str, dict[str, str]] = {}


    annual_rev = raw.get("revenue_growth", {}).get("annual_revenue", [])
    if len(annual_rev) >= 2:
        pct = pct_change(annual_rev[1]["revenue"], annual_rev[0]["revenue"])
        checks["revenue_growth"] = {
            "status": direction(pct),
            "detail": f"{annual_rev[1]['year']}→{annual_rev[0]['year']}: {pct:+.1f}%" if pct is not None else "unavailable",
        }
    else:
        checks["revenue_growth"] = {"status": "insufficient_data", "detail": "fewer than 2 years of annual revenue"}

    annual_ni = raw.get("revenue_growth", {}).get("annual_net_income", [])
    if len(annual_ni) >= 2:
        pct = pct_change(annual_ni[1]["net_income"], annual_ni[0]["net_income"])
        checks["net_income_growth"] = {
            "status": direction(pct),
            "detail": f"{annual_ni[1]['year']}→{annual_ni[0]['year']}: {pct:+.1f}%" if pct is not None else "unavailable",
        }
    else:
        checks["net_income_growth"] = {"status": "insufficient_data", "detail": "fewer than 2 years of annual net income"}

    ocf_series = raw.get("xbrl_financials", {}).get("operating_cash_flow_annual", [])
    capex_series = raw.get("xbrl_financials", {}).get("capex_annual", [])
    if len(ocf_series) >= 2 and len(capex_series) >= 2:
        fcf_new = ocf_series[0]["value"] - capex_series[0]["value"]
        fcf_old = ocf_series[1]["value"] - capex_series[1]["value"]
        pct = pct_change(fcf_old, fcf_new)
        checks["fcf_growth"] = {
            "status": direction(pct),
            "detail": f"{pct:+.1f}%" if pct is not None else "unavailable",
        }
    else:
        checks["fcf_growth"] = {"status": "insufficient_data", "detail": "fewer than 2 years of XBRL operating cash flow / capex"}

    if (len(annual_rev) >= 2 and len(annual_ni) >= 2
            and annual_rev[0]["year"] == annual_ni[0]["year"]
            and annual_rev[1]["year"] == annual_ni[1]["year"]
            and annual_rev[0]["revenue"] and annual_rev[1]["revenue"]):
        margin_new = annual_ni[0]["net_income"] / annual_rev[0]["revenue"] * 100
        margin_old = annual_ni[1]["net_income"] / annual_rev[1]["revenue"] * 100
        delta = margin_new - margin_old
        if delta > 1.0:
            status = "expanding"
        elif delta < -1.0:
            status = "compressing"
        else:
            status = "stable"
        checks["margin_trend"] = {
            "status": status,
            "detail": f"{margin_old:.1f}% → {margin_new:.1f}% ({delta:+.1f}pp)",
        }
    else:
        checks["margin_trend"] = {
            "status": "insufficient_data",
            "detail": "revenue/net_income annual series don't share aligned fiscal years, or too few years",
        }

    ndte = calc.get("net_debt_to_ebitda", {})
    if ndte.get("interpretation") and ndte["interpretation"] != "unavailable":
        checks["leverage_snapshot"] = {
            "status": ndte["interpretation"],
            "detail": f"Net Debt/EBITDA = {ndte.get('net_debt_to_ebitda')} (current snapshot, not a trend -- no historical debt series available)",
        }
    else:
        checks["leverage_snapshot"] = {"status": "insufficient_data", "detail": "net_debt_to_ebitda unavailable"}

    checks["valuation_vs_history"] = {
        "status": "insufficient_data",
        "detail": "no historical P/E (or other multiple) series available in this pipeline -- only the current snapshot",
    }

    return checks


def build_data_quality_score(raw: dict[str, Any], calc: dict[str, Any]) -> dict[str, Any]:
   
    sources_available = {
        "yahoo_finance": bool(raw.get("yahoo_ratios")) and not raw.get("yahoo_ratios", {}).get("error"),
        "sec_xbrl":       bool(raw.get("xbrl_financials")) and not raw.get("xbrl_financials", {}).get("error"),
        "balance_sheet_snapshot": bool(raw.get("balance_sheet_snapshot")) and not raw.get("balance_sheet_snapshot", {}).get("error"),
        "peer_comparison": bool(raw.get("peer_comparison")) and not raw.get("peer_comparison", {}).get("error"),
    }

    extraction_error_count = len(raw.get("extraction_errors", []))
    ratio_dicts = [v for v in calc.values() if isinstance(v, dict)]
    unavailable_count = sum(
        1 for v in ratio_dicts
        if v.get("interpretation") == "unavailable" or v.get("validation_status") == "UNAVAILABLE"
    )
    fallback_count = sum(1 for v in ratio_dicts if v.get("validation_status") == "FALLBACK")
    degraded_count = sum(1 for v in ratio_dicts if v.get("validation_status") == "DEGRADED")
    low_confidence_count = sum(1 for v in ratio_dicts if v.get("confidence") == "low")

    score = 100
    score -= extraction_error_count * 15
    score -= unavailable_count * 5
    score -= fallback_count * 1
    score -= degraded_count * 3
    score -= low_confidence_count * 3
    score -= sum(1 for ok in sources_available.values() if not ok) * 10
    score = max(0, min(100, score))

    return {
        "sources_available":          sources_available,
        "extraction_error_count":     extraction_error_count,
        "unavailable_metric_count":   unavailable_count,
        "fallback_metric_count":      fallback_count,
        "degraded_metric_count":      degraded_count,
        "low_confidence_metric_count": low_confidence_count,
        "total_metrics_computed":     len(ratio_dicts),
        "reliability_score_pct":      score,
        "formula": (
            "100 - 15*extraction_errors - 5*unavailable_metrics "
            "- 1*fallback_metrics - 3*degraded_metrics "
            "- 3*low_confidence_metrics - 10*unavailable_sources, floored at 0"
        ),
    }







def sanitize_nans(value: Any) -> Any:
   
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize_nans(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_nans(v) for v in value]
    return value
