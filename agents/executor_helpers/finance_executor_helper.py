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
    """
    Resolve the target ticker symbol from multiple context sources.

    Resolution order (highest priority first):
        1. ``manager_directives["ticker"]``            — explicit Manager override.
        2. ``financial_metrics_summary["ticker"]``     — already resolved upstream
           by the FinancialAnalystAgent (only checked when a summary is passed).
        3. Regex scan of ``task_query`` for 1-5 uppercase-letter tokens,
           filtering known stop-words.

    Shared by FinancialAnalystAgent and SentimentAgent. The Financial Agent
    calls it without ``financial_summary`` (that source doesn't exist yet at
    its stage); the Sentiment Agent passes the summary so an already-resolved
    ticker is reused rather than re-inferred.

    Returns the uppercase ticker, or ``None`` if none can be inferred.
    """
    # Priority 1: explicit directive
    if directives.get("ticker"):
        return str(directives["ticker"]).upper()

    # Priority 2: already resolved by FinancialAnalystAgent (if provided)
    if financial_summary and financial_summary.get("ticker"):
        return str(financial_summary["ticker"]).upper()

    # Priority 3: heuristic regex scan of task_query
    candidates = re.findall(r"\b([A-Z]{1,5})\b", task_query)
    for candidate in candidates:
        if candidate not in _TICKER_STOP_WORDS:
            logger.info("Inferred ticker from task_query: %s", candidate)
            return candidate

    logger.warning("Could not infer ticker from task_query: '%s'", task_query)
    return None


# ---------------------------------------------------------------------------
# Deterministic analysis layer -- confidence, consistency, data quality.
#
# All three functions below are pure rule-based post-processing over
# already-computed ratios; none of them call an LLM or make a judgment call
# about what a number MEANS for the business (that's the Manager Agent
# finalizer's job, deliberately kept separate -- see the call site in run()
# for the full rationale). They only: (1) formalize the approximation/period
# signals this pipeline already tracks into a simple confidence label,
# (2) compare numbers this pipeline already has to flag directional
# agreement/disagreement, and (3) tally errors/approximations into one
# score. Nothing here invents a number that wasn't already computed.
# ---------------------------------------------------------------------------

def confidence_for(period_tag: str | None, is_approximation: bool = False) -> str:
    """
    Classify a single ratio's reliability from signals this pipeline
    already produces (its "_period" tag and "is_approximation" flag) --
    NOT a new judgment, just formalizing what was already being computed
    into one label so the Manager/finalizer doesn't have to re-derive it
    from parsing "_period" strings itself.

    Returns "high" | "medium" | "low" | "unknown".
    """
    if is_approximation:
        return "low"
    if not period_tag:
        return "unknown"
    if period_tag.startswith("MIXED(") or period_tag == "mixed":
        return "medium"
    return "high"


def annotate_confidence(calculated_ratios: dict[str, Any]) -> None:
    """
    Mutate calculated_ratios IN PLACE: add a "confidence" key to every
    ratio dict that carries a "_period" tag. Safe/additive -- every
    existing consumer (financial_metrics_summary construction above,
    manager_agent.py's fin_summary extraction) reads specific named keys
    and ignores keys it doesn't ask for, so this can't break anything
    downstream; it just makes "confidence" available wherever a caller
    wants it.
    """
    for ratio in calculated_ratios.values():
        if isinstance(ratio, dict) and "_period" in ratio:
            ratio["confidence"] = confidence_for(
                ratio.get("_period"), ratio.get("is_approximation", False)
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
    """
    Rule-based cross-checks over data this pipeline ALREADY has -- each
    check either runs on real multi-period data or is explicitly marked
    "insufficient_data" with a reason. Never fabricates a trend from a
    single data point, and never claims a metric this pipeline has no
    historical series for (e.g. there's no historical debt series here,
    only a current snapshot -- so leverage is reported as a status, not a
    trend, rather than mislabeled as "Debt Stable").

    Returns a dict of {check_name: {"status": ..., "detail": ...}}.
    """
    checks: dict[str, dict[str, str]] = {}

    # Revenue trend -- from raw.revenue_growth.annual_revenue (newest-first)
    annual_rev = raw.get("revenue_growth", {}).get("annual_revenue", [])
    if len(annual_rev) >= 2:
        pct = pct_change(annual_rev[1]["revenue"], annual_rev[0]["revenue"])
        checks["revenue_growth"] = {
            "status": direction(pct),
            "detail": f"{annual_rev[1]['year']}→{annual_rev[0]['year']}: {pct:+.1f}%" if pct is not None else "unavailable",
        }
    else:
        checks["revenue_growth"] = {"status": "insufficient_data", "detail": "fewer than 2 years of annual revenue"}

    # Net income trend -- proxy for earnings growth. Labeled "net_income",
    # NOT "eps", since this pipeline has no historical shares-outstanding
    # series to compute actual historical EPS.
    annual_ni = raw.get("revenue_growth", {}).get("annual_net_income", [])
    if len(annual_ni) >= 2:
        pct = pct_change(annual_ni[1]["net_income"], annual_ni[0]["net_income"])
        checks["net_income_growth"] = {
            "status": direction(pct),
            "detail": f"{annual_ni[1]['year']}→{annual_ni[0]['year']}: {pct:+.1f}%" if pct is not None else "unavailable",
        }
    else:
        checks["net_income_growth"] = {"status": "insufficient_data", "detail": "fewer than 2 years of annual net income"}

    # FCF trend -- from XBRL operating_cash_flow_annual − capex_annual,
    # index-aligned (both series come from the same XBRL source/ordering).
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

    # Margin trend -- only computed when the two annual series actually
    # share the same fiscal years at the same index positions; otherwise
    # explicitly flagged rather than silently comparing mismatched periods
    # (the exact class of bug this pipeline's period-fidelity rules exist
    # to prevent).
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

    # Leverage -- a SNAPSHOT, not a trend (no historical debt series
    # available here) -- deliberately named/worded to not imply "stable".
    ndte = calc.get("net_debt_to_ebitda", {})
    if ndte.get("interpretation") and ndte["interpretation"] != "unavailable":
        checks["leverage_snapshot"] = {
            "status": ndte["interpretation"],
            "detail": f"Net Debt/EBITDA = {ndte.get('net_debt_to_ebitda')} (current snapshot, not a trend -- no historical debt series available)",
        }
    else:
        checks["leverage_snapshot"] = {"status": "insufficient_data", "detail": "net_debt_to_ebitda unavailable"}

    # Valuation vs. historical average -- explicitly NOT computed. This
    # pipeline has no historical P/E series (only the current snapshot),
    # so a "valuation vs. history" claim would have to be fabricated.
    checks["valuation_vs_history"] = {
        "status": "insufficient_data",
        "detail": "no historical P/E (or other multiple) series available in this pipeline -- only the current snapshot",
    }

    return checks


def build_data_quality_score(raw: dict[str, Any], calc: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate this run's own error/approximation signals into one
    reliability score -- a tally of what's ALREADY known (extraction
    errors, unavailable ratios, low-confidence ratios), not a new
    assessment. Downstream agents (Manager, Sentiment) can use this to
    decide how much weight to put on this run's numbers without having to
    re-derive it from scanning every individual ratio themselves.
    """
    sources_available = {
        "yahoo_finance": bool(raw.get("yahoo_ratios")) and not raw.get("yahoo_ratios", {}).get("error"),
        "sec_xbrl":       bool(raw.get("xbrl_financials")) and not raw.get("xbrl_financials", {}).get("error"),
        "balance_sheet_snapshot": bool(raw.get("balance_sheet_snapshot")) and not raw.get("balance_sheet_snapshot", {}).get("error"),
        "peer_comparison": bool(raw.get("peer_comparison")) and not raw.get("peer_comparison", {}).get("error"),
    }

    extraction_error_count = len(raw.get("extraction_errors", []))
    ratio_dicts = [v for v in calc.values() if isinstance(v, dict)]
    unavailable_count = sum(1 for v in ratio_dicts if v.get("interpretation") == "unavailable")
    low_confidence_count = sum(1 for v in ratio_dicts if v.get("confidence") == "low")

    # Simple, documented, deterministic formula -- not a statistical model,
    # just a readable penalty tally so the number's meaning is inspectable.
    score = 100
    score -= extraction_error_count * 15
    score -= unavailable_count * 5
    score -= low_confidence_count * 3
    score -= sum(1 for ok in sources_available.values() if not ok) * 10
    score = max(0, min(100, score))

    return {
        "sources_available":          sources_available,
        "extraction_error_count":     extraction_error_count,
        "unavailable_metric_count":   unavailable_count,
        "low_confidence_metric_count": low_confidence_count,
        "total_metrics_computed":     len(ratio_dicts),
        "reliability_score_pct":      score,
        "formula": (
            "100 − 15×extraction_errors − 5×unavailable_metrics "
            "− 3×low_confidence_metrics − 10×unavailable_sources, floored at 0"
        ),
    }







def sanitize_nans(value: Any) -> Any:
    """
    Recursively replace any float NaN/Infinity in a nested dict/list
    structure with None, so the result is always strict-JSON-safe
    (RFC 8259 has no NaN/Infinity token; Python's json module allows it
    by default, but downstream persistence clients — e.g. Supabase's
    postgrest client — reject it with
    "Out of range float values are not JSON compliant: nan").

    This is a defensive last resort, not a substitute for fixing NaN at
    its source (see _safe_float() in yahoo_finance.py and the
    _is_valid_number() guard in the CAGR block above) — but it guarantees
    financial_metrics_summary can never crash persistence regardless of
    which upstream data source introduces a stray NaN in the future.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize_nans(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_nans(v) for v in value]
    return value