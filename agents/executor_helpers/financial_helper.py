"""
agents/financial_helper.py
"""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple

from agents.executor_helpers.finance_executor_helper import infer_peers
from agents.state import FinancialAgentState
from evaluation.financial_sanity import build_valuation_input_assessments


INTENTS: frozenset[str] = frozenset({
    "valuation", "financial_health", "profitability", "growth",
    "liquidity", "cash_flow", "peer_relative", "overview", "full_report",
})


BASELINE_EXTRACTION: tuple[str, ...] = (
    "tool_get_financial_ratios",
    "tool_get_xbrl_financials",
)
BASELINE_COMPUTE: tuple[str, ...] = ("tool_calc_composite_score",)


STATUTORY_TAX_RATE_PCT = 21.0

# Monte-Carlo sample counts per analysis depth. 'shallow' never reaches a
# Monte-Carlo tool (_resolve_tools drops MONTE_CARLO_TOOLS at that depth),
# so it is listed only to keep the mapping total.
MC_SIMULATIONS_BY_DEPTH: dict[str, int] = {
    "shallow": 250,
    "standard": 500,
    "deep": 2000,
}
DEFAULT_MC_SIMULATIONS: int = MC_SIMULATIONS_BY_DEPTH["standard"]

RECOVERY_STATES: frozenset[str] = frozenset({
    "VALID_PRIMARY", "FALLBACK", "DEGRADED", "UNAVAILABLE",
})


def recovery_metadata(
    *,
    state: str,
    source: str,
    exact_period: Any,
    formula: str | None,
    fallback_reason: str | None = None,
    confidence: str | None = None,
    is_approximation: bool = False,
    semantic_name: str | None = None,
    normalization: str | None = None,
    filed_at: str | None = None,
    accession_references: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    eligible_as_of: str | None = None,
) -> dict[str, Any]:
    """Build the additive metadata contract shared by recovered metrics."""
    if state not in RECOVERY_STATES:
        raise ValueError(f"Unsupported recovery state: {state}")
    if confidence is None:
        confidence = {
            "VALID_PRIMARY": "high",
            "FALLBACK": "medium",
            "DEGRADED": "low",
            "UNAVAILABLE": "none",
        }[state]
    return {
        "validation_status": state,
        "source": source,
        "exact_period": exact_period,
        "formula": formula,
        "fallback_reason": fallback_reason,
        "confidence": confidence,
        "is_approximation": is_approximation,
        **({"semantic_name": semantic_name} if semantic_name else {}),
        **({"normalization": normalization} if normalization else {}),
        "filed_at": filed_at,
        "accession_references": sorted(set(accession_references or [])),
        "evidence_ids": sorted(set(evidence_ids or [])),
        "eligible_as_of": eligible_as_of,
    }


ANALYSIS_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "valuation": {
        "extraction": (
            "tool_get_financial_ratios", "tool_get_xbrl_financials",
            "tool_get_revenue_growth", "tool_get_balance_sheet_snapshot",
        ),
        "compute": (
            "tool_calc_pe", "tool_calc_pb", "tool_calc_ev_ebitda", "tool_calc_peg",
            "tool_calc_cagr", "tool_build_forecast", "tool_calc_dcf_scenarios",
            "tool_calc_reverse_dcf", "tool_calc_dcf_multi_factor_monte_carlo",
            "tool_calc_dcf_monte_carlo",
        ),
    },
    "financial_health": {
        "extraction": (
            "tool_get_financial_ratios", "tool_get_xbrl_financials",
            "tool_get_balance_sheet_snapshot", "tool_get_revenue_growth",
        ),
        "compute": (
            "tool_calc_roa", "tool_calc_roe", "tool_calc_debt_to_equity",
            "tool_calc_debt_ratio", "tool_calc_equity_ratio",
            "tool_calc_interest_coverage", "tool_calc_net_debt_to_ebitda",
            "tool_calc_quick_ratio", "tool_calc_current_ratio",
            "tool_calc_altman_z_score",
        ),
    },
    "profitability": {
        "extraction": (
            "tool_get_financial_ratios", "tool_get_xbrl_financials",
            "tool_get_revenue_growth", "tool_get_balance_sheet_snapshot",
        ),
        "compute": (
            "tool_calc_gross_margin", "tool_calc_operating_margin",
            "tool_calc_net_margin", "tool_calc_roe", "tool_calc_roa",
            "tool_calc_roic", "tool_calc_roce",
        ),
    },
    "growth": {
        "extraction": ("tool_get_revenue_growth", "tool_get_xbrl_financials"),
        "compute": (
            "tool_calc_cagr", "tool_calc_revenue_cagr_from_growth",
            "tool_build_forecast",
        ),
    },
    "liquidity": {
        "extraction": ("tool_get_balance_sheet_snapshot", "tool_get_xbrl_financials"),
        "compute": (
            "tool_calc_current_ratio", "tool_calc_quick_ratio", "tool_calc_ocf_ratio",
        ),
    },
    "cash_flow": {
        "extraction": (
            "tool_get_financial_ratios", "tool_get_xbrl_financials",
            "tool_get_revenue_growth", "tool_get_balance_sheet_snapshot",
        ),
        "compute": (
            "tool_calc_fcf_margin", "tool_calc_fcf_yield", "tool_calc_ocf_ratio",
        ),
    },
    "peer_relative": {
        
        "extraction": ("tool_get_peer_comparison",),
        "compute": (
            "tool_calc_pe", "tool_calc_pb", "tool_calc_ev_ebitda", "tool_calc_peg",
        ),
    },
    "overview": {
        "extraction": (
            "tool_get_financial_ratios", "tool_get_revenue_growth",
            "tool_get_xbrl_financials", "tool_get_balance_sheet_snapshot",
        ),
        "compute": (
            "tool_calc_pe", "tool_calc_roe", "tool_calc_net_margin",
            "tool_calc_debt_to_equity", "tool_calc_cagr",
            "tool_calc_composite_score",
        ),
    },
    
}


FULL_REPORT_PROFILES: tuple[str, ...] = (
    "valuation", "profitability", "growth", "financial_health",
    "liquidity", "cash_flow", "peer_relative",
)


MONTE_CARLO_TOOLS: frozenset[str] = frozenset({
    "tool_calc_dcf_monte_carlo", "tool_calc_dcf_multi_factor_monte_carlo",
})


INTENT_CHECK_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "valuation": (
        ("pe", "pb", "ev_ebitda", "peg"),
        ("dcf", "reverse_dcf"),
    ),
    "financial_health": (
        ("roe", "roa"),
        ("de_ratio", "debt_ratio", "equity_ratio"),
        ("interest_coverage", "net_debt_to_ebitda", "altman_z_score"),
    ),
    "profitability": (
        ("operating_margin", "net_margin", "gross_margin"),
        ("roe", "roa", "roic", "roce"),
    ),
    "growth": (
        ("cagr", "revenue_cagr_from_growth"),
        ("forecast",),
    ),
    "liquidity": (
        ("current_ratio_calc", "quick_ratio"),
        ("ocf_ratio",),
    ),
    "cash_flow": (
        ("fcf_margin", "fcf_yield"),
        ("ocf_ratio",),
    ),
    "peer_relative": (("peer_comparison",),),
    "overview": (
        ("pe",), ("roe",), ("net_margin",), ("de_ratio",), ("cagr",),
    ),
}


def result_available(value: Any) -> bool:
    """Return True when a metric result contains data, regardless of validity."""
    if not isinstance(value, dict) or value.get("error"):
        return False
    if value.get("interpretation") in {"unavailable", "insufficient_data", "insufficient_history"}:
        return False
    for key, item in value.items():
        if key.startswith("_") or key in {"interpretation", "formula", "note", "assumptions"}:
            continue
        if is_valid_number(item):
            return True
        if isinstance(item, (dict, list)) and item:
            return True
    return False


def result_valid(value: Any) -> bool:
    """Return True only when an available result passed deterministic gates."""
    if not result_available(value):
        return False
    if value.get("validation_status") in {
        "INVALID", "INCOMPATIBLE_PERIOD", "UNAVAILABLE",
    }:
        return False
    period = str(value.get("_period") or "")
    return not period.upper().startswith("MIXED")


def checker_requirements(state: FinancialAgentState) -> dict[str, Any]:
    """Resolve and evaluate the deterministic completeness contract."""
    intents = list(state.get("intents") or ["overview"])
    raw = state.get("raw_numerical_data", {})
    is_full_report = "full_report" in intents
    expanded = list(FULL_REPORT_PROFILES) if is_full_report else intents
    groups: list[tuple[str, ...]] = []
    for intent in expanded:
       
        if intent == "peer_relative" and is_full_report and not raw.get("peer_comparison", {}).get("peers"):
            continue
        for group in INTENT_CHECK_REQUIREMENTS.get(intent, ()):
            if group not in groups:
                groups.append(group)

    calc = state.get("calculated_ratios", {})
    availability = {
        name: (
            bool(raw.get("peer_comparison", {}).get("peers"))
            if name == "peer_comparison" else result_available(calc.get(name, {}))
        )
        for group in groups for name in group
    }
    validity = {
        name: (
            availability[name]
            if name == "peer_comparison" else result_valid(calc.get(name, {}))
        )
        for group in groups for name in group
    }
    missing_groups = [list(group) for group in groups if not any(validity[name] for name in group)]
    return {
        "intents": intents,
        "required_any_of_groups": [list(group) for group in groups],
        "availability": availability,
        "validity": validity,
        "missing_groups": missing_groups,
        "passed": not missing_groups,
    }




COMPUTE_ORDER: tuple[str, ...] = (
    "tool_calc_pe", "tool_calc_pb", "tool_calc_ev_ebitda", "tool_calc_peg",
    "tool_calc_gross_margin", "tool_calc_operating_margin", "tool_calc_net_margin",
    "tool_calc_roe", "tool_calc_roa", "tool_calc_roic", "tool_calc_roce",
    "tool_calc_debt_to_equity", "tool_calc_debt_ratio", "tool_calc_equity_ratio",
    "tool_calc_interest_coverage", "tool_calc_net_debt_to_ebitda",
    "tool_calc_current_ratio", "tool_calc_quick_ratio", "tool_calc_ocf_ratio",
    "tool_calc_altman_z_score",
    "tool_calc_cagr", "tool_calc_revenue_cagr_from_growth",
    "tool_calc_fcf_margin", "tool_calc_fcf_yield",
    "tool_build_forecast", "tool_calc_dcf_scenarios",
    "tool_calc_dcf_multi_factor_monte_carlo", "tool_calc_dcf_monte_carlo",
    "tool_calc_reverse_dcf",
    "tool_calc_composite_score",
)


def normalize_depth(directives: dict[str, Any]) -> str:
   
    raw = ""
    for key in ("depth", "analysis_depth", "detail_level", "level"):
        if directives.get(key):
            raw = str(directives[key]).strip().lower()
            break
    if raw in {"shallow", "quick", "brief", "fast", "light", "low"}:
        return "shallow"
    if raw in {"deep", "thorough", "comprehensive", "full", "exhaustive", "high"}:
        return "deep"
    return "standard"


def build_args(**kw: Any) -> dict[str, Any] | None:
   
    return None if any(v is None for v in kw.values()) else kw




def extract_ticker_args(ticker: str, raw: dict[str, Any]) -> dict[str, Any] | None:
    return {"ticker": ticker}


def extract_peer_args(ticker: str, raw: dict[str, Any]) -> dict[str, Any] | None:
  
    industry = raw.get("yahoo_ratios", {}).get("industry", "")
    sector = raw.get("yahoo_ratios", {}).get("sector", "")
    peers = infer_peers(sector, industry, ticker)
    return {"ticker": ticker, "peers": peers} if peers else None


EXTRACTION_REGISTRY: dict[str, tuple[Any, str]] = {
    "tool_get_financial_ratios":       (extract_ticker_args, "yahoo_ratios"),
    "tool_get_peer_comparison":        (extract_peer_args,   "peer_comparison"),
    "tool_get_revenue_growth":         (extract_ticker_args, "revenue_growth"),
    "tool_get_xbrl_financials":        (extract_ticker_args, "xbrl_financials"),
    "tool_get_balance_sheet_snapshot": (extract_ticker_args, "balance_sheet_snapshot"),
}


EXTRACTION_ORDER: tuple[str, ...] = (
    "tool_get_financial_ratios",
    "tool_get_peer_comparison",
    "tool_get_revenue_growth",
    "tool_get_xbrl_financials",
    "tool_get_balance_sheet_snapshot",
)



def is_valid_number(x: Any) -> bool:
    if x is None or isinstance(x, (bool, str, bytes)):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError, OverflowError):
        return False


def newest(series: list[dict] | None, key: str = "value") -> Any:
    """Most recent entry's value from an XBRL-style newest-first series."""
    if not series:
        return None
    try:
        return series[0][key]
    except (KeyError, TypeError, IndexError):
        return None


def _fact_period_key(
    fact: dict[str, Any],
    *,
    scope: str,
) -> tuple[str | None, str, str] | None:
    """Build an exact, unit-aware period key for a validated XBRL fact."""
    if not isinstance(fact, dict) or not is_valid_number(fact.get("value")):
        return None
    if fact.get("validation_status") in {"INVALID", "INCOMPATIBLE_PERIOD", "UNAVAILABLE"}:
        return None
    if fact.get("metric_scope") not in {None, scope}:
        return None
    end = fact.get("period_end")
    start = fact.get("period_start")
    unit = fact.get("unit")
    if not end or not unit:
        return None
    if scope == "duration" and not start:
        return None
    if scope == "instant" and start is not None:
        return None
    return (str(start) if start is not None else None, str(end), str(unit))


def _matching_facts(
    *series: list[dict[str, Any]] | None,
    scope: str = "duration",
) -> tuple[dict[str, Any], ...] | None:
    """Return the newest facts sharing one exact period, scope, and unit."""
    if not series or any(not items for items in series):
        return None
    indexed: list[dict[tuple[str | None, str, str], dict[str, Any]]] = []
    for items in series:
        by_key: dict[tuple[str | None, str, str], dict[str, Any]] = {}
        for fact in items or []:
            key = _fact_period_key(fact, scope=scope)
            if key is not None and key not in by_key:
                by_key[key] = fact
        indexed.append(by_key)
    common = set(indexed[0])
    for by_key in indexed[1:]:
        common &= set(by_key)
    if not common:
        return None
    selected_key = max(common, key=lambda key: (key[1], key[0] or "", key[2]))
    return tuple(by_key[selected_key] for by_key in indexed)


def _period_label(fact: dict[str, Any] | None) -> str | None:
    if not fact:
        return None
    start = fact.get("period_start")
    end = fact.get("period_end")
    return f"annual:{start}/{end}" if start and end else None


def _combined_fact_metadata(
    facts: tuple[dict[str, Any], ...] | None,
    *,
    formula: str,
    semantic_name: str,
) -> dict[str, Any]:
    if not facts:
        return recovery_metadata(
            state="UNAVAILABLE", source="unavailable", exact_period=None,
            formula=formula, semantic_name=semantic_name,
            fallback_reason="No compatible exact-period fact bundle was available.",
        )
    fallback_facts = [f for f in facts if f.get("validation_status") == "FALLBACK"]
    state = "FALLBACK" if fallback_facts else "VALID_PRIMARY"
    sources = [
        str(f.get("selected_source") or f"sec_xbrl:{f.get('xbrl_concept')}")
        for f in facts
    ]
    reasons = [str(f.get("fallback_reason")) for f in fallback_facts if f.get("fallback_reason")]
    filed_values = sorted(str(f.get("filed_at")) for f in facts if f.get("filed_at"))
    return recovery_metadata(
        state=state,
        source=" + ".join(sources),
        exact_period=facts[0].get("exact_period") or {
            "basis": facts[0].get("metric_scope"),
            "period_start": facts[0].get("period_start"),
            "period_end": facts[0].get("period_end"),
        },
        formula=formula,
        fallback_reason="; ".join(reasons) or None,
        semantic_name=semantic_name,
        filed_at=filed_values[-1] if filed_values else None,
        accession_references=[str(f.get("accession_number")) for f in facts if f.get("accession_number")],
        evidence_ids=[str(f.get("evidence_id")) for f in facts if f.get("evidence_id")],
        eligible_as_of=next((str(f.get("eligible_as_of")) for f in facts if f.get("eligible_as_of")), None),
    )


def derive_compute_facts(
    raw: dict[str, Any],
    calc: dict[str, Any],
    depth: str = "standard",
) -> dict[str, Any]:
    """Flatten the raw extraction payloads into one dict of compute inputs.

    ``depth`` is the normalized analysis depth (see normalize_depth); it only
    controls Monte-Carlo sample counts. It defaults to "standard" so existing
    callers and tests keep working unchanged.
    """
    ratios_src = raw.get("yahoo_ratios", {})
    growth_src = raw.get("revenue_growth", {})
    xbrl_src   = raw.get("xbrl_financials", {})
    bs_snapshot = raw.get("balance_sheet_snapshot", {})

    d: dict[str, Any] = {
        "calc": calc,
        "ratios_src": ratios_src, "growth_src": growth_src,
        "xbrl_src": xbrl_src, "bs_snapshot": bs_snapshot,
        "depth": depth,
        "n_simulations": MC_SIMULATIONS_BY_DEPTH.get(depth, DEFAULT_MC_SIMULATIONS),
    }

    # -- P/E inputs --------------------------------------------------------
    d["price"] = ratios_src.get("current_price")
    d["eps"]   = ratios_src.get("eps_trailing")
    d["pe_ratio_yahoo"] = ratios_src.get("pe_ratio")

    # -- Return on Equity --------------------------------------------------
    xbrl_assets      = xbrl_src.get("total_assets", [])
    xbrl_liabilities = xbrl_src.get("total_liabilities", [])
    matched_balance_facts = _matching_facts(
        xbrl_assets, xbrl_liabilities, scope="instant",
    )
    annual_ni        = growth_src.get("annual_net_income", [])
    d["xbrl_assets"] = xbrl_assets
    d["xbrl_liabilities"] = xbrl_liabilities
    d["annual_ni"] = annual_ni

    net_income      = annual_ni[0]["net_income"] if annual_ni else None
    net_income_year = annual_ni[0].get("year") if annual_ni else None
    
    # Equity source priority:
    #   1. balance_sheet_snapshot.total_equity -- the reported equity line,
    #      which already excludes minority interest.
    #   2. XBRL Assets - Liabilities -- a fallback that INCLUDES minority
    #      interest and therefore overstates equity for companies with
    #      partially-owned subsidiaries, depressing ROE/ROIC.
    # Feeds roe, roic, equity_ratio and de_ratio, so the difference between
    # the two propagates into four reported metrics.
    shareholders_equity: float | None = None
    equity_period_end: str | None = None
    equity_source: str = "unavailable"

    snapshot_equity = bs_snapshot.get("total_equity")
    if is_valid_number(snapshot_equity) and snapshot_equity:
        shareholders_equity = snapshot_equity
        equity_period_end = bs_snapshot.get("period_end")
        equity_source = "balance_sheet_snapshot.total_equity"
    elif matched_balance_facts:
        try:
            assets_fact, liabilities_fact = matched_balance_facts
            shareholders_equity = assets_fact["value"] - liabilities_fact["value"]
            equity_period_end = assets_fact.get("period_end")
            equity_source = "xbrl_assets_minus_liabilities"
        except (KeyError, TypeError, IndexError):
            shareholders_equity = None
    d["equity_source"] = equity_source

    
    years_aligned = (
        net_income_year is not None
        and equity_period_end is not None
        and str(net_income_year) in str(equity_period_end)
    )
    if net_income_year is None:
        roe_period = None
    elif equity_period_end is not None and not years_aligned:
        roe_period = f"MIXED(net_income=annual:{net_income_year}, equity=annual:{equity_period_end})"
    else:
        roe_period = f"annual:{net_income_year}"

    d["net_income"] = net_income
    d["net_income_year"] = net_income_year
    d["shareholders_equity"] = shareholders_equity or None  # 0 equity is unusable
    d["equity_period_end"] = equity_period_end
    d["roe_period"] = roe_period

    
    quarterly_rev = growth_src.get("quarterly_revenue", [])
    quarterly_ni  = growth_src.get("quarterly_net_income", [])
    annual_rev    = growth_src.get("annual_revenue", [])
    revenue_year  = annual_rev[0].get("year") if annual_rev else None
    d["annual_rev"] = annual_rev
    d["revenue_year"] = revenue_year

    use_quarterly = (
        quarterly_rev and quarterly_ni
        and quarterly_rev[0].get("quarter") == quarterly_ni[0].get("quarter")
        and quarterly_rev[0].get("revenue") is not None
        and quarterly_ni[0].get("net_income") is not None
    )
    if use_quarterly:
        d["nm_net_income"] = quarterly_ni[0]["net_income"]
        d["nm_revenue"]    = quarterly_rev[0]["revenue"]
        d["net_margin_period"] = f"quarterly:{quarterly_rev[0]['quarter']}"
    else:
       
        d["nm_net_income"] = net_income
        d["nm_revenue"]    = annual_rev[0]["revenue"] if annual_rev else None
        d["net_margin_period"] = (
            f"MIXED(net_income=annual:{net_income_year}, revenue=annual:{revenue_year})"
            if net_income_year is not None and revenue_year is not None and net_income_year != revenue_year
            else f"annual:{revenue_year}" if revenue_year is not None
            else None
        )

   
    d["de_ratio_yahoo"] = (
        ratios_src.get("de_ratio")
        if ratios_src.get("de_ratio") is not None else ratios_src.get("debtToEquity")
    )
    d["yahoo_period_end"] = ratios_src.get("financial_data_period_end")
    d["yahoo_fcf_ttm"] = ratios_src.get("free_cash_flow")
    d["yahoo_ocf_ttm"] = ratios_src.get("operating_cash_flow")
    d["yahoo_revenue_ttm"] = ratios_src.get("revenue_ttm")
    d["total_liabilities_val"] = (
        matched_balance_facts[1]["value"] if matched_balance_facts else None
    )
    d["liabilities_period_end"] = (
        matched_balance_facts[1].get("period_end") if matched_balance_facts else None
    )

   
    valid_rev = [
        e for e in annual_rev
        if is_valid_number(e.get("revenue")) and e.get("year") is not None
    ]
    valid_rev.sort(key=lambda e: e["year"])  # oldest -> newest
    d["valid_rev"] = valid_rev
    d["cagr_start"] = d["cagr_end"] = d["cagr_years"] = None
    d["cagr_period"] = None
    if len(valid_rev) >= 2:
        # Named oldest_rev/newest_rev, not oldest/newest: a local called
        # `newest` would shadow the module-level newest() helper for the
        # whole function scope and make the earlier calls raise
        # UnboundLocalError.
        oldest_rev, newest_rev = valid_rev[0]["revenue"], valid_rev[-1]["revenue"]
        years = valid_rev[-1]["year"] - valid_rev[0]["year"]
        if newest_rev and oldest_rev and oldest_rev > 0 and years > 0:
            d["cagr_start"], d["cagr_end"], d["cagr_years"] = oldest_rev, newest_rev, float(years)
            d["cagr_period"] = f"annual:{valid_rev[0]['year']}-{valid_rev[-1]['year']}"

    
    operating_cf_annual = xbrl_src.get("operating_cash_flow_annual", [])
    capex_annual        = xbrl_src.get("capex_annual", [])
    d["operating_cf_annual"] = operating_cf_annual
    d["capex_annual"] = capex_annual
    d["beta"]          = ratios_src.get("beta")
    d["market_cap"]    = ratios_src.get("market_cap")
    d["current_price"] = ratios_src.get("current_price")
    d["enterprise_value"] = ratios_src.get("enterprise_value")
    d["ebitda_ttm"]    = ratios_src.get("ebitda")

    observed_diluted_shares = ratios_src.get("diluted_shares")
    d["observed_diluted_shares"] = observed_diluted_shares
    d["shares_outstanding"] = observed_diluted_shares or (
        d["market_cap"] / d["current_price"]
        if d["market_cap"] and d["current_price"] else None
    )
    d["total_debt_ttm"]        = ratios_src.get("total_debt")
    d["total_cash_ttm"]        = ratios_src.get("total_cash")
    d["lease_liabilities_ttm"] = ratios_src.get("lease_liabilities")
    d["minority_interest_ttm"] = ratios_src.get("minority_interest")
    d["interest_expense_ttm"]  = ratios_src.get("interest_expense")

    # Effective tax rate, in priority order:
    #   1. XBRL income_tax_annual / pretax_income_annual for the newest year
    #      -- the company's actual reported rate, straight from the filing.
    #   2. Yahoo's effective_tax_rate field.
    #   3. The US statutory rate as a last resort.
    # This single number feeds forecast, dcf, reverse_dcf and roic, so a
    # placeholder here biases four models in the same direction at once.
    matched_tax_facts = _matching_facts(
        xbrl_src.get("income_tax_annual", []),
        xbrl_src.get("pretax_income_annual", []),
        scope="duration",
    )
    xbrl_tax = matched_tax_facts[0]["value"] if matched_tax_facts else None
    xbrl_pretax = matched_tax_facts[1]["value"] if matched_tax_facts else None
    tax_rate_raw = ratios_src.get("effective_tax_rate")

    derived_tax_pct: float | None = None
    if is_valid_number(xbrl_tax) and is_valid_number(xbrl_pretax) and xbrl_pretax:
        candidate = float(xbrl_tax) / float(xbrl_pretax) * 100.0
        # Reject nonsense: loss years, one-off benefits and restatements can
        # produce negative or absurd rates that would poison the DCF.
        if 0.0 <= candidate <= 50.0:
            derived_tax_pct = candidate

    if derived_tax_pct is not None:
        d["effective_tax_rate_pct"] = derived_tax_pct
        d["effective_tax_rate_source"] = "xbrl_income_tax_over_pretax_income"
    elif tax_rate_raw is not None:
        d["effective_tax_rate_pct"] = (
            float(tax_rate_raw) * 100 if abs(float(tax_rate_raw)) <= 1
            else float(tax_rate_raw)
        )
        d["effective_tax_rate_source"] = "yahoo_effective_tax_rate"
    else:
        d["effective_tax_rate_pct"] = STATUTORY_TAX_RATE_PCT
        d["effective_tax_rate_source"] = "us_statutory_placeholder"
    d["pre_tax_cost_of_debt_pct"] = (
        abs(float(d["interest_expense_ttm"])) / d["total_debt_ttm"] * 100
        if d["interest_expense_ttm"] is not None and d["total_debt_ttm"] and d["total_debt_ttm"] > 0
        else 5.0
    )
    d["valuation_input_assessments"] = build_valuation_input_assessments(
        market_cap=d["market_cap"], total_debt=d["total_debt_ttm"], cash=d["total_cash_ttm"],
        marketable_securities=None, lease_liabilities=d["lease_liabilities_ttm"],
        minority_interest=d["minority_interest_ttm"], diluted_shares=d["shares_outstanding"],
        diluted_shares_observed=observed_diluted_shares is not None,
        cost_of_debt_observed=(
            d["interest_expense_ttm"] is not None
            and bool(d["total_debt_ttm"] and d["total_debt_ttm"] > 0)
        ),
        tax_rate_observed=d["effective_tax_rate_source"] != "us_statutory_placeholder",
        pre_tax_cost_of_debt_pct=d["pre_tax_cost_of_debt_pct"],
        effective_tax_rate_pct=d["effective_tax_rate_pct"],
        sector=ratios_src.get("sector"),
    )


    revenue_xbrl = xbrl_src.get("revenue_annual", [])
    matched_fcf_facts = _matching_facts(
        operating_cf_annual, capex_annual, scope="duration",
    )
    matched_valuation_facts = _matching_facts(
        operating_cf_annual, capex_annual, revenue_xbrl, scope="duration",
    )
    fcf_formula = "Operating Cash Flow - Capital Expenditures"
    fcf_metadata = _combined_fact_metadata(
        matched_fcf_facts,
        formula=fcf_formula,
        semantic_name="free_cash_flow",
    )
    valuation_revenue_fact = matched_valuation_facts[2] if matched_valuation_facts else None

    if matched_fcf_facts:
        fcf_base = matched_fcf_facts[0]["value"] - matched_fcf_facts[1]["value"]
        fcf_period = _period_label(matched_fcf_facts[0])
        revenue_base = valuation_revenue_fact["value"] if valuation_revenue_fact else None
        valuation_period = _period_label(valuation_revenue_fact)
        capex_pct_revenue = (
            matched_valuation_facts[1]["value"] / revenue_base * 100.0
            if matched_valuation_facts and revenue_base else None
        )
        revenue_base_year = (
            valuation_revenue_fact.get("fiscal_year")
            or str(valuation_revenue_fact.get("period_end", ""))[:4]
            if valuation_revenue_fact else None
        )
    elif is_valid_number(d["yahoo_fcf_ttm"]):
        fcf_base = float(d["yahoo_fcf_ttm"])
        yahoo_period_end = d.get("yahoo_period_end")
        fcf_period = f"ttm:{yahoo_period_end or 'as_of_latest'}"
        revenue_base = (
            float(d["yahoo_revenue_ttm"])
            if is_valid_number(d["yahoo_revenue_ttm"]) else None
        )
        valuation_period = fcf_period if revenue_base is not None else None
        capex_pct_revenue = None
        revenue_base_year = None
        fcf_metadata = recovery_metadata(
            state="FALLBACK",
            source="yahoo_finance.freeCashflow",
            exact_period={"basis": "TTM", "period_start": None, "period_end": yahoo_period_end},
            formula="Yahoo direct Free Cash Flow (TTM)",
            fallback_reason=(
                "No compatible exact-period SEC XBRL operating-cash-flow/capex bundle; "
                "used Yahoo's direct TTM free-cash-flow metric."
            ),
            semantic_name="free_cash_flow",
            normalization="Yahoo TTM values are kept as a complete TTM basis and are not mixed with annual operands.",
        )
    else:
        fcf_base = None
        fcf_period = None
        revenue_base = None
        valuation_period = None
        capex_pct_revenue = None
        revenue_base_year = None

    d["fcf_base"] = fcf_base
    d["dcf_fcf_base"] = fcf_base
    d["fcf_period"] = fcf_period
    d["fcf_metadata"] = fcf_metadata
    d["revenue_base"] = revenue_base
    d["revenue_base_year"] = revenue_base_year
    d["valuation_base_period"] = valuation_period

    trajectory_raw = growth_src.get("revenue_growth_ttm")
    if trajectory_raw is None and len(valid_rev) >= 2 and valid_rev[-2]["revenue"]:
        trajectory_raw = valid_rev[-1]["revenue"] / valid_rev[-2]["revenue"] - 1.0
    d["current_growth_pct"] = (
        float(trajectory_raw) * 100.0
        if trajectory_raw is not None and abs(float(trajectory_raw)) <= 2.0
        else float(trajectory_raw) if trajectory_raw is not None else None
    )
    d["capex_pct_revenue"] = capex_pct_revenue

    
    d["target_ev"] = (
        d["market_cap"] + d["total_debt_ttm"] - d["total_cash_ttm"]
        if d["market_cap"] is not None and d["total_debt_ttm"] is not None
        and d["total_cash_ttm"] is not None else None
    )

    # -- Balance-sheet snapshot / Phase-2 inputs ---------------------------
    d["total_assets_val"]        = (
        matched_balance_facts[0]["value"] if matched_balance_facts else None
    )
    d["ebit_snap"]               = bs_snapshot.get("ebit")
    d["retained_earnings_snap"]  = bs_snapshot.get("retained_earnings")
    d["working_capital_snap"]    = bs_snapshot.get("working_capital")
    d["current_assets_snap"]     = bs_snapshot.get("current_assets")
    d["current_liabilities_snap"] = bs_snapshot.get("current_liabilities")
    d["total_equity_snap"]       = bs_snapshot.get("total_equity")
    d["bs_period_end"]           = bs_snapshot.get("period_end", "unknown")
    d["fcf_margin_revenue"]      = d["revenue_base"]
    d["altman_revenue"]          = annual_rev[0]["revenue"] if annual_rev else None
    d["current_ratio_val"]       = ratios_src.get("current_ratio")

    bs_period_end = bs_snapshot.get("period_end")
    aligned_ocf_fact = next(
        (
            fact for fact in operating_cf_annual
            if isinstance(fact, dict)
            and fact.get("period_end") == bs_period_end
            and _fact_period_key(fact, scope="duration") is not None
        ),
        None,
    )
    if aligned_ocf_fact is not None:
        d["ocf_annual"] = aligned_ocf_fact["value"]
        d["ocf_coverage_period"] = _period_label(aligned_ocf_fact)
        d["ocf_coverage_metadata"] = recovery_metadata(
            state=(
                "FALLBACK" if aligned_ocf_fact.get("validation_status") == "FALLBACK"
                else "VALID_PRIMARY"
            ),
            source=str(aligned_ocf_fact.get("selected_source") or "sec_xbrl"),
            exact_period=aligned_ocf_fact.get("exact_period"),
            formula="Operating Cash Flow / Current Liabilities",
            fallback_reason=aligned_ocf_fact.get("fallback_reason"),
            semantic_name="operating_cash_flow_coverage",
        )
    elif (
        is_valid_number(d["yahoo_ocf_ttm"])
        and bs_period_end
        and d.get("yahoo_period_end") == bs_period_end
    ):
        d["ocf_annual"] = float(d["yahoo_ocf_ttm"])
        d["ocf_coverage_period"] = f"ttm:{bs_period_end}"
        d["ocf_coverage_metadata"] = recovery_metadata(
            state="FALLBACK",
            source="yahoo_finance.operatingCashflow",
            exact_period={"basis": "TTM", "period_start": None, "period_end": bs_period_end},
            formula="Operating Cash Flow / Current Liabilities",
            fallback_reason="No SEC annual OCF ended on the balance-sheet date; used Yahoo TTM OCF with the same period end.",
            semantic_name="operating_cash_flow_coverage",
            normalization="TTM flow divided by liabilities at the matching TTM period end.",
        )
    else:
        d["ocf_annual"] = None
        d["ocf_coverage_period"] = None
        d["ocf_coverage_metadata"] = recovery_metadata(
            state="UNAVAILABLE", source="unavailable", exact_period=None,
            formula="Operating Cash Flow / Current Liabilities",
            fallback_reason="No operating-cash-flow value shared the balance-sheet period end.",
            semantic_name="operating_cash_flow_coverage",
        )

    
    d["book_value_per_share"] = (
        d["total_equity_snap"] / d["observed_diluted_shares"]
        if d["total_equity_snap"] is not None and d["observed_diluted_shares"] else None
    )
  
    d["earnings_growth_pct"] = None
    if len(annual_ni) >= 2:
        prev = annual_ni[1].get("net_income")
        curr = annual_ni[0].get("net_income")
        if is_valid_number(prev) and is_valid_number(curr) and prev and prev > 0:
            d["earnings_growth_pct"] = (curr - prev) / prev * 100.0
    matched_operating_margin = _matching_facts(
        revenue_xbrl, xbrl_src.get("operating_income_annual", []), scope="duration",
    )
    matched_gross_margin = _matching_facts(
        revenue_xbrl, xbrl_src.get("cost_of_revenue_annual", []), scope="duration",
    )
    d["xbrl_revenue_annual"] = (
        matched_gross_margin[0]["value"] if matched_gross_margin else None
    )
    d["cost_of_revenue_annual"] = (
        matched_gross_margin[1]["value"] if matched_gross_margin else None
    )
    d["gross_margin_period"] = (
        _period_label(matched_gross_margin[0]) if matched_gross_margin else None
    )
    d["gross_margin_metadata"] = _combined_fact_metadata(
        matched_gross_margin,
        formula="(Revenue - Cost of Revenue) / Revenue * 100",
        semantic_name="gross_margin",
    )
    d["operating_margin_revenue"] = (
        matched_operating_margin[0]["value"] if matched_operating_margin else None
    )
    d["operating_income_annual"] = (
        matched_operating_margin[1]["value"] if matched_operating_margin else None
    )
    d["operating_margin_period"] = (
        _period_label(matched_operating_margin[0]) if matched_operating_margin else None
    )
    d["operating_margin_metadata"] = _combined_fact_metadata(
        matched_operating_margin,
        formula="Operating Income / Revenue * 100",
        semantic_name="operating_margin",
    )
    d["quick_ratio_yahoo"]       = ratios_src.get("quick_ratio")
    return d


class ComputeSpec(NamedTuple):
    """One row of COMPUTE_REGISTRY. See the module comment above for the
    mapper's three possible return shapes."""
    key:         str                                   
    mapper:      Callable[[dict[str, Any]], Any]
    unavailable: dict[str, Any]                        
    post:        Callable[[dict[str, Any], dict[str, Any]], None] | None = None





def post_de_ratio(result: dict[str, Any], d: dict[str, Any]) -> None:
  
    result["formula"] = "Total Liabilities / Shareholders' Equity"
    result["interpretation"] = "approximation_not_comparable_to_standard_thresholds"
    result["is_approximation"] = True
    result["semantic_name"] = "total_liabilities_to_equity_approximation"
    result["display_name"] = "Total Liabilities / Equity (approximation; not standard D/E)"
    result["note"] = (
        "tool_get_xbrl_financials only exposes total_liabilities, not "
        "interest-bearing debt separately, so this uses total liabilities "
        "as a rough stand-in. This is NOT a true debt-to-equity ratio and "
        "should not be compared against standard D/E leverage thresholds."
    )


def post_roic(result: dict[str, Any], d: dict[str, Any]) -> None:
    """Flag NOPAT as approximate only when the tax rate really was a guess."""
    source = d.get("effective_tax_rate_source", "us_statutory_placeholder")
    rate = d.get("effective_tax_rate_pct")
    result["tax_rate_source"] = source
    if source == "us_statutory_placeholder":
        result["is_approximation"] = True
        result["note"] = (
            f"tax_rate_pct fell back to the US statutory rate ({STATUTORY_TAX_RATE_PCT}%): "
            "neither the XBRL income-tax/pretax-income series nor Yahoo's effective tax "
            "rate was usable -- NOPAT here is an approximation."
        )
    else:
        result["is_approximation"] = False
        result["note"] = (
            f"tax_rate_pct = {rate:.1f}% taken from {source}, not a statutory placeholder."
        )


def map_pe(d):
    args = build_args(price=d["price"], eps=d["eps"])
    if args and d["eps"] != 0:
        return args, "ttm"
  
    if not is_valid_number(d["pe_ratio_yahoo"]):
        return None
    return {
        "pe_ratio": d["pe_ratio_yahoo"],
        "interpretation": "sourced_from_yahoo",
        "note": "Used pre-computed Yahoo Finance P/E (price or EPS unavailable for direct calc).",
        "source": "yahoo_finance.trailingPE",
        "exact_period": {"basis": "TTM", "period_start": None, "period_end": d.get("yahoo_period_end")},
        "formula": "Current Price / Trailing EPS",
        "fallback_reason": "Price/EPS components were unavailable; used Yahoo's direct trailing P/E.",
        "confidence": "medium",
        "is_approximation": False,
        "semantic_name": "price_to_earnings",
        "validation_status": "FALLBACK",
        "_inputs": {"pe_ratio": d["pe_ratio_yahoo"]},
        "_period": "ttm",
    }


def map_pb(d):
    args = build_args(price=d["price"], book_value_per_share=d["book_value_per_share"])
    return (args, f"MIXED(price=ttm, book_value={d['bs_period_end']})") if args else None


def map_ev_ebitda(d):
    args = build_args(enterprise_value=d["enterprise_value"], ebitda=d["ebitda_ttm"])
    return (args, "ttm") if args and d["ebitda_ttm"] != 0 else None


def map_peg(d):
    pe = d["calc"].get("pe", {}).get("pe_ratio") or d["pe_ratio_yahoo"]
    args = build_args(pe=pe, earnings_growth_rate_pct=d["earnings_growth_pct"])
    return (args, "MIXED(pe=ttm, earnings_growth=annual_yoy)") if args else None


def map_gross_margin(d):
    """Gross margin from the XBRL cost-of-revenue series.

    Both inputs come from the same annual XBRL extraction, so the periods
    are aligned by construction -- no MIXED() tag needed.
    """
    args = build_args(
        revenue=d["xbrl_revenue_annual"], cogs=d["cost_of_revenue_annual"],
    )
    return (
        args, d["gross_margin_period"], d["gross_margin_metadata"]
    ) if args and d["xbrl_revenue_annual"] and d["gross_margin_period"] else None


def map_operating_margin(d):
    args = build_args(
        operating_income=d["operating_income_annual"],
        revenue=d["operating_margin_revenue"],
    )
    return (
        args, d["operating_margin_period"], d["operating_margin_metadata"]
    ) if args and d["operating_margin_revenue"] and d["operating_margin_period"] else None


def map_net_margin(d):
    args = build_args(net_income=d["nm_net_income"], revenue=d["nm_revenue"])
    compatible = not str(d.get("net_margin_period") or "").upper().startswith("MIXED")
    return (
        args, d["net_margin_period"]
    ) if args and d["nm_revenue"] != 0 and compatible else None


def map_roe(d):
    args = build_args(
        net_income=d["net_income"], shareholders_equity=d["shareholders_equity"],
    )
    return (args, d["roe_period"]) if args and d["net_income"] else None


def map_roa(d):
    args = build_args(net_income=d["net_income"], total_assets=d["total_assets_val"])
    return (
        args,
        f"MIXED(net_income=annual:{d['net_income_year']}, total_assets=annual:{d['equity_period_end']})",
    ) if args and d["total_assets_val"] else None


def map_roic(d):
    args = build_args(
        ebit=d["ebit_snap"], tax_rate_pct=d["effective_tax_rate_pct"],
        total_debt=d["total_debt_ttm"], total_equity=d["shareholders_equity"],
        cash=d["total_cash_ttm"],
    )
    return (args, (
        f"MIXED(ebit={d['bs_period_end']}, equity=annual:{d['equity_period_end']}, debt/cash=ttm)"
    )) if args else None


def map_roce(d):
    args = build_args(
        ebit=d["ebit_snap"], total_assets=d["total_assets_val"],
        current_liabilities=d["current_liabilities_snap"],
    )
    return (args, (
        f"MIXED(ebit={d['bs_period_end']}, total_assets=annual:{d['equity_period_end']})"
    )) if args and d["total_assets_val"] else None


def map_debt_to_equity(d):
    
    if d["de_ratio_yahoo"] is not None:
        return {
            "de_ratio": d["de_ratio_yahoo"],
            "interpretation": "sourced_from_yahoo",
            "semantic_name": "debt_to_equity",
            "display_name": "Debt-to-Equity Ratio",
            "formula": "Total Interest-Bearing Debt / Shareholders' Equity",
            "source": "yahoo_finance.debtToEquity",
            "exact_period": {
                "basis": "snapshot", "period_start": None,
                "period_end": d.get("yahoo_period_end"),
            },
            "fallback_reason": "Interest-bearing debt components were not available as an exact-period primary bundle; used Yahoo's direct D/E metric.",
            "confidence": "medium",
            "is_approximation": False,
            "validation_status": "FALLBACK",
            "_inputs": {"de_ratio_yahoo": d["de_ratio_yahoo"]},
            "_period": f"snapshot:{d.get('yahoo_period_end') or 'mrq'}",
        }
    args = build_args(
        total_debt=d["total_liabilities_val"], shareholders_equity=d["shareholders_equity"],
    )
    return (
        args,
        f"snapshot:{d['equity_period_end']}" if d["equity_period_end"] else None,
        recovery_metadata(
            state="DEGRADED",
            source="sec_xbrl.total_liabilities + balance_sheet_equity",
            exact_period={
                "basis": "snapshot", "period_start": None,
                "period_end": d.get("equity_period_end"),
            },
            formula="Total Liabilities / Shareholders' Equity",
            fallback_reason="Interest-bearing debt was unavailable; total liabilities are used only as a labeled leverage approximation.",
            is_approximation=True,
            semantic_name="total_liabilities_to_equity_approximation",
        ),
    ) if (
        args and d["total_liabilities_val"]
        and d.get("liabilities_period_end") == d.get("equity_period_end")
    ) else None


def map_debt_ratio(d):
    
    args = build_args(
        total_liabilities=d["total_liabilities_val"], total_assets=d["total_assets_val"],
    )
    return (
        args, f"annual:{d['equity_period_end']}" if d["equity_period_end"] else "annual",
    ) if args and d["total_assets_val"] else None


def map_equity_ratio(d):
    args = build_args(total_equity=d["shareholders_equity"], total_assets=d["total_assets_val"])
    return (
        args, f"annual:{d['equity_period_end']}" if d["equity_period_end"] else "annual",
    ) if args and d["total_assets_val"] else None


def map_interest_coverage(d):
    args = build_args(ebit=d["ebit_snap"], interest_expense=d["interest_expense_ttm"])
    return (args, f"MIXED(ebit={d['bs_period_end']}, interest_expense=ttm)") if args else None


def map_net_debt_to_ebitda(d):
    args = build_args(
        total_debt=d["total_debt_ttm"], total_cash=d["total_cash_ttm"], ebitda=d["ebitda_ttm"],
    )
    return (args, "ttm") if args and d["ebitda_ttm"] else None


def map_current_ratio(d):
    args = build_args(
        current_assets=d["current_assets_snap"], current_liabilities=d["current_liabilities_snap"],
    )
    if args and d.get("bs_period_end") not in {None, "unknown"}:
        return (
            args,
            f"snapshot:{d['bs_period_end']}",
            recovery_metadata(
                state="VALID_PRIMARY",
                source="yahoo_finance.balance_sheet",
                exact_period={
                    "basis": "snapshot", "period_start": None,
                    "period_end": d["bs_period_end"],
                },
                formula="Current Assets / Current Liabilities",
                semantic_name="current_ratio",
            ),
        )
    if is_valid_number(d.get("current_ratio_val")):
        return {
            "current_ratio": d["current_ratio_val"],
            "interpretation": "sourced_from_yahoo",
            "semantic_name": "current_ratio",
            "display_name": "Current Ratio",
            "formula": "Current Assets / Current Liabilities",
            "source": "yahoo_finance.currentRatio",
            "exact_period": {
                "basis": "snapshot", "period_start": None,
                "period_end": d.get("yahoo_period_end"),
            },
            "fallback_reason": "Current-asset/current-liability components were unavailable; used Yahoo's direct Current Ratio.",
            "confidence": "medium",
            "is_approximation": False,
            "validation_status": "FALLBACK",
            "_inputs": {"current_ratio": d["current_ratio_val"]},
            "_period": f"snapshot:{d.get('yahoo_period_end') or 'mrq'}",
        }
    return None


def map_quick_ratio(d):
    """Quick ratio, sourced from Yahoo rather than computed.

    Computing it directly needs cash + short-term investments + receivables
    split out, which get_balance_sheet_snapshot() doesn't expose. Yahoo
    already publishes the finished ratio, so this follows the same
    pre-computed-fallback pattern as map_pe rather than reporting
    "unavailable" for a number we actually have.
    """
    if d["quick_ratio_yahoo"] is None:
        return None
    return {
        "quick_ratio": d["quick_ratio_yahoo"],
        "interpretation": "sourced_from_yahoo",
        "note": (
            "Pre-computed Yahoo Finance quick ratio; the balance-sheet snapshot "
            "doesn't expose receivables/short-term investments separately, so this "
            "was not recomputed from components."
        ),
        "source": "yahoo_finance.quickRatio",
        "exact_period": {"basis": "snapshot", "period_start": None, "period_end": d.get("yahoo_period_end")},
        "formula": "(Cash + Short-term Investments + Receivables) / Current Liabilities",
        "fallback_reason": "Quick-ratio components were unavailable; used Yahoo's direct Quick Ratio.",
        "confidence": "medium",
        "is_approximation": False,
        "semantic_name": "quick_ratio",
        "validation_status": "FALLBACK",
        "_inputs": {"quick_ratio": d["quick_ratio_yahoo"]},
        "_period": "mrq",
    }


def map_ocf_ratio(d):
    args = build_args(
        operating_cash_flow=d["ocf_annual"], current_liabilities=d["current_liabilities_snap"],
    )
    return (
        args, d["ocf_coverage_period"], d["ocf_coverage_metadata"],
    ) if args and d["ocf_coverage_period"] else None


def map_altman_z_score(d):
    args = build_args(
        working_capital=d["working_capital_snap"], retained_earnings=d["retained_earnings_snap"],
        ebit=d["ebit_snap"], total_assets=d["total_assets_val"], market_cap=d["market_cap"],
        total_liabilities=d["total_liabilities_val"], revenue=d["altman_revenue"],
    )
    return (args, "mixed") if args and d["total_assets_val"] else None


def map_cagr(d):
    args = build_args(
        start_value=d["cagr_start"], end_value=d["cagr_end"], years=d["cagr_years"],
    )
    return (args, d["cagr_period"]) if args else None


def map_revenue_cagr_from_growth(d):
    
    return (
        ({"annual_revenue": d["annual_rev"]}, "annual")
        if len(d["annual_rev"]) >= 2 else None
    )


def map_fcf_margin(d):
  
    args = build_args(free_cash_flow=d["fcf_base"], revenue=d["fcf_margin_revenue"])
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Free Cash Flow / Revenue * 100",
        "semantic_name": "free_cash_flow_margin",
        "exact_period": d["fcf_metadata"].get("exact_period"),
    })
    return (
        args, d["valuation_base_period"], metadata,
    ) if args and d["fcf_margin_revenue"] and d["valuation_base_period"] else None


def map_fcf_yield(d):
    args = build_args(free_cash_flow=d["fcf_base"], market_cap=d["market_cap"])
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Free Cash Flow / Current Market Capitalization * 100",
        "semantic_name": "free_cash_flow_yield",
        "normalization": "A 12-month FCF flow is divided by current market capitalization, the standard FCF-yield basis.",
    })
    return (
        args, f"normalized:{d['fcf_period']}+current_market_cap", metadata,
    ) if args and d["market_cap"] and d["fcf_period"] else None


def map_build_forecast(d):
   
    historical_cagr_pct = d["calc"].get("cagr", {}).get("cagr_pct")
    if d["dcf_fcf_base"] is None or d["beta"] is None or not d["revenue_base"]:
        return None
    if historical_cagr_pct is None:
       
        return None
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Explicit revenue/FCF scenario forecast from a coherent 12-month base",
        "semantic_name": "financial_forecast",
        "exact_period": d["fcf_metadata"].get("exact_period"),
    })
    return {
        "revenue_base": d["revenue_base"],
        "fcf_base": d["dcf_fcf_base"],
        "historical_revenue_cagr_pct": historical_cagr_pct,
        "current_revenue_growth_pct": d["current_growth_pct"],
        "capex_pct_revenue": d["capex_pct_revenue"],
        "projection_years": 5,
        "source_period": d["valuation_base_period"],
    }, f"projection_from:{d['valuation_base_period']}", metadata


def map_dcf_scenarios(d):
   
    forecast = d["calc"].get("forecast", {})
    if d["dcf_fcf_base"] is None or d["beta"] is None or not forecast.get("scenarios"):
        return None
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Present value of explicit forecast FCF plus terminal value",
        "semantic_name": "discounted_cash_flow_valuation",
    })
    return {
        "fcf_base": d["dcf_fcf_base"],
        "beta": d["beta"],
        "shares_outstanding": d["shares_outstanding"],
        "diluted_shares": d["shares_outstanding"],
        "market_cap": d["market_cap"],
        "total_debt": d["total_debt_ttm"],
        "cash": d["total_cash_ttm"],
        "marketable_securities": None,
        "lease_liabilities": d["lease_liabilities_ttm"],
        "minority_interest": d["minority_interest_ttm"],
        "pre_tax_cost_of_debt_pct": d["pre_tax_cost_of_debt_pct"],
        "effective_tax_rate_pct": d["effective_tax_rate_pct"],
        "input_assessments": d["valuation_input_assessments"],
        "forecast_scenarios": forecast.get("scenarios", {}),
    }, f"projection_from:{d['valuation_base_period']}", metadata


def map_dcf_multi_factor_monte_carlo(d):
  
    base_dcf = d["calc"].get("dcf", {}).get("base", {})
    base_forecast = d["calc"].get("forecast", {}).get("scenarios", {}).get("base", {})
    if base_dcf.get("error") is not None or not base_forecast.get("fcf_by_year"):
        return None
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Monte Carlo valuation over explicit multi-factor FCF paths",
        "semantic_name": "dcf_monte_carlo_multi_factor",
    })
    return {
        "fcf_by_year": base_forecast["fcf_by_year"],
        "wacc_pct": base_dcf.get("wacc_pct"),
        "terminal_growth_rate_pct": base_dcf.get("assumptions", {}).get("terminal_growth_rate_pct", 2.5),
        "cash": d["total_cash_ttm"] or 0.0,
        "total_debt": d["total_debt_ttm"] or 0.0,
        "lease_liabilities": d["lease_liabilities_ttm"] or 0.0,
        "minority_interest": d["minority_interest_ttm"] or 0.0,
        "diluted_shares": d["shares_outstanding"],
        "n_simulations": d.get("n_simulations", DEFAULT_MC_SIMULATIONS),
    }, f"projection_from:{d['fcf_period']}", metadata


def map_dcf_monte_carlo(d):
    
    args = build_args(
        fcf_base=d["dcf_fcf_base"], beta=d["beta"],
        base_growth_rate_pct=d["calc"].get("cagr", {}).get("cagr_pct"),
    )
    if args is None:
        return None
    args["shares_outstanding"] = d["shares_outstanding"]
    args["n_simulations"] = d.get("n_simulations", DEFAULT_MC_SIMULATIONS)
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Monte Carlo DCF from a coherent 12-month FCF base",
        "semantic_name": "dcf_monte_carlo_growth",
    })
    return args, f"projection_from:{d['fcf_period']}", metadata


def map_reverse_dcf(d):
    args = build_args(
        fcf_base=d["dcf_fcf_base"], beta=d["beta"], target_enterprise_value=d["target_ev"],
        market_cap=d["market_cap"], total_debt=d["total_debt_ttm"],
    )
    if args is None or d["target_ev"] <= 0:
        return None
    args["pre_tax_cost_of_debt_pct"] = d["pre_tax_cost_of_debt_pct"]
    args["effective_tax_rate_pct"] = d["effective_tax_rate_pct"]
    metadata = dict(d["fcf_metadata"])
    metadata.update({
        "formula": "Implied FCF growth that reconciles enterprise value with a DCF",
        "semantic_name": "reverse_dcf",
        "normalization": "A 12-month FCF base is reconciled to current enterprise value.",
    })
    return args, f"normalized:{d['fcf_period']}+current_enterprise_value", metadata


def map_composite_score(d):
   
    cr = d["calc"]
    return {
        "pe":                cr.get("pe", {}).get("pe_ratio"),
        "roe_pct":           cr.get("roe", {}).get("roe_pct"),
        "net_margin_pct":    cr.get("net_margin", {}).get("net_margin_pct"),
        "de_ratio":          cr.get("de_ratio", {}).get("de_ratio"),
        "revenue_cagr_pct":  cr.get("cagr", {}).get("cagr_pct"),
        "current_ratio_val": d["current_ratio_val"],
    }, "mixed"


COMPUTE_REGISTRY: dict[str, ComputeSpec] = {
    "tool_calc_pe":               ComputeSpec("pe", map_pe, {"pe_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_pb":               ComputeSpec("pb", map_pb, {"pb_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_ev_ebitda":        ComputeSpec("ev_ebitda", map_ev_ebitda, {"ev_ebitda": None, "interpretation": "unavailable"}),
    "tool_calc_peg":              ComputeSpec("peg", map_peg, {"peg_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_gross_margin":     ComputeSpec("gross_margin", map_gross_margin, {"gross_margin_pct": None, "interpretation": "unavailable"}),
    "tool_calc_operating_margin": ComputeSpec("operating_margin", map_operating_margin, {"operating_margin_pct": None, "interpretation": "unavailable"}),
    "tool_calc_net_margin":       ComputeSpec("net_margin", map_net_margin, {"net_margin_pct": None, "interpretation": "insufficient_data"}),
    "tool_calc_roe":              ComputeSpec("roe", map_roe, {"roe_pct": None, "interpretation": "insufficient_data"}),
    "tool_calc_roa":              ComputeSpec("roa", map_roa, {"roa_pct": None, "interpretation": "unavailable"}),
    "tool_calc_roic":             ComputeSpec("roic", map_roic, {"roic_pct": None, "interpretation": "unavailable"}, post_roic),
    "tool_calc_roce":             ComputeSpec("roce", map_roce, {"roce_pct": None, "interpretation": "unavailable"}),
    "tool_calc_debt_to_equity":   ComputeSpec("de_ratio", map_debt_to_equity, {"de_ratio": None, "interpretation": "unavailable"}, post_de_ratio),
    "tool_calc_debt_ratio":       ComputeSpec("debt_ratio", map_debt_ratio, {"debt_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_equity_ratio":     ComputeSpec("equity_ratio", map_equity_ratio, {"equity_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_interest_coverage": ComputeSpec("interest_coverage", map_interest_coverage, {"interest_coverage": None, "interpretation": "unavailable"}),
    "tool_calc_net_debt_to_ebitda": ComputeSpec("net_debt_to_ebitda", map_net_debt_to_ebitda, {"net_debt_to_ebitda": None, "interpretation": "unavailable"}),
    "tool_calc_current_ratio":    ComputeSpec("current_ratio_calc", map_current_ratio, {"current_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_quick_ratio":      ComputeSpec("quick_ratio", map_quick_ratio, {"quick_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_ocf_ratio":        ComputeSpec("ocf_ratio", map_ocf_ratio, {"ocf_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_altman_z_score":   ComputeSpec("altman_z_score", map_altman_z_score, {"z_score": None, "interpretation": "unavailable"}),
    "tool_calc_cagr":             ComputeSpec("cagr", map_cagr, {"cagr_pct": None, "interpretation": "insufficient_history"}),
    "tool_calc_revenue_cagr_from_growth": ComputeSpec("revenue_cagr_from_growth", map_revenue_cagr_from_growth, {"cagr_pct": None, "interpretation": "insufficient_history"}),
    "tool_calc_fcf_margin":       ComputeSpec("fcf_margin", map_fcf_margin, {"fcf_margin_pct": None, "interpretation": "unavailable"}),
    "tool_calc_fcf_yield":        ComputeSpec("fcf_yield", map_fcf_yield, {"fcf_yield_pct": None, "interpretation": "unavailable"}),
    "tool_build_forecast":        ComputeSpec("forecast", map_build_forecast, {"interpretation": "unavailable", "error": "Missing fcf_base/beta/revenue_base, or no revenue_cagr available to anchor the base-case FCF growth assumption."}),
    "tool_calc_dcf_scenarios":    ComputeSpec("dcf", map_dcf_scenarios, {"enterprise_value": None, "interpretation": "unavailable", "error": "Missing operating_cash_flow_annual/capex_annual from XBRL (fcf_base), beta from Yahoo ratios, or a base forecast — cannot run DCF."}),
    "tool_calc_dcf_multi_factor_monte_carlo": ComputeSpec("dcf_monte_carlo", map_dcf_multi_factor_monte_carlo, {"interpretation": "unavailable", "model_classification": "MULTI_FACTOR", "error": "A valid explicit base forecast and WACC are required."}),
    "tool_calc_dcf_monte_carlo":  ComputeSpec("dcf_monte_carlo_growth", map_dcf_monte_carlo, {"interpretation": "unavailable", "model_classification": "SINGLE_FACTOR", "error": "Missing fcf_base, beta, or a revenue CAGR to anchor the growth distribution."}),
    "tool_calc_reverse_dcf":      ComputeSpec("reverse_dcf", map_reverse_dcf, {"implied_growth_rate_pct": None, "interpretation": "unavailable", "error": "Missing fcf_base, beta, or a valid target enterprise value (market_cap + total_debt − total_cash)."}),
    "tool_calc_composite_score":  ComputeSpec("composite_score", map_composite_score, {"score": None, "interpretation": "unavailable"}),
}

SUMMARY_CALCULATED_PASSTHROUGH: tuple[str, ...] = tuple(
    dict.fromkeys(spec.key for spec in COMPUTE_REGISTRY.values())
)


def surface_calculated_ratios(summary: dict[str, Any], calc: dict[str, Any]) -> None:
    """Add every registered compute result without replacing legacy fields."""
    for key in SUMMARY_CALCULATED_PASSTHROUGH:
        summary.setdefault(key, calc.get(key, {}))


def select_sec_filing_type(state: FinancialAgentState) -> str | None:
    """Choose a filing only when the task actually calls for filing prose."""
    query = state.get("task_query", "").lower()
    intents = set(state.get("intents") or [])
    if any(term in query for term in ("8-k", "material event", "current event", "recent event")):
        return "8-K"
    if any(term in query for term in ("10-q", "quarterly", "latest quarter", "recent quarter")):
        return "10-Q"
    if "full_report" in intents or any(
        term in query for term in ("10-k", "annual report", "fundamental", "sec filing")
    ):
        return "10-K"
    return None
