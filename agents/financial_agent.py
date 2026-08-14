"""
agents/financial_agent.py
--------------------------
Production-grade FinancialAnalystAgent for the Alpha-Agent Node platform.

Architecture
------------
The agent follows a three-tiered internal lifecycle:

    ┌──────────────────────────────────────────────────────────┐
    │                    FinancialAnalystAgent                 │
    │                                                          │
    │  LAYER 1 — EXECUTORS (MCP Protocol Interface)            │
    │    _execute_data_extraction()  → raw_numerical_data      │
    │    _execute_ratio_computation() → calculated_ratios      │
    │                                                          │
    │  LAYER 2 — CHECKER / CRITIC (Validation & QC)           │
    │    _check_data_quality()  → {is_complete, feedback}      │
    │                                                          │
    │  LAYER 3 — BRAIN / ORCHESTRATOR (Lifecycle Gateway)     │
    │    _brain()  → structured tool instructions              │
    │    run()     → entry point; drives the full loop         │
    └──────────────────────────────────────────────────────────┘

MCP Tool Servers consumed
-------------------------
    YahooFinanceClient   : tool_get_financial_ratios, tool_get_revenue_growth
    SecEdgarParser       : tool_get_xbrl_financials
    FinancialRatioCalc   : tool_calc_pe, tool_calc_roe, tool_calc_net_margin,
                           tool_calc_debt_to_equity, tool_calc_cagr,
                           tool_calc_composite_score

State contract
--------------
    Input  : SharedManagerState  (task_query, manager_directives)
    Output : SharedManagerState  (+ financial_metrics_summary populated)

Dependencies
------------
    pip install anthropic "mcp[cli]"
"""

from __future__ import annotations

import json
import logging
import math
import asyncio
from typing import Any, Callable, NamedTuple

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import END, StateGraph
import os
from agents.prompts.promt_finance_checker import CHECKER_SYSTEM_PROMPT
from evaluation.financial_sanity import evaluate_financial_sanity, build_valuation_input_assessments
from evaluation.earnings_quality import normalize_earnings_quality
from evaluation.claim_contract import typed_financial_claim
from agents.executor_helpers.finance_executor_helper import (infer_peers, extract_ticker, annotate_confidence, build_consistency_checks, build_data_quality_score, sanitize_nans)

# ---------------------------------------------------------------------------
# State contract imports
# ---------------------------------------------------------------------------
# Both SharedManagerState and FinancialAgentState are declared in
# agents/state.py — the single source of truth for all state TypedDicts
# across the Alpha-Agent Node platform.
from agents.state import FinancialAgentState, SharedManagerState
from langsmith import traceable
from core.observability import sentry_enabled
from core.progress_bus import publish as _publish_progress, session_from_directives

# ---------------------------------------------------------------------------
# Logging — stderr only; stdout is reserved for MCP JSON-RPC
# NOTE: logging.basicConfig() must NOT be called in library/agent code.
# Configuration belongs exclusively in entry points (api/main.py or __main__).
# ---------------------------------------------------------------------------
log = logging.getLogger("financial-analyst-agent")
_MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[os.path.join(os.path.dirname(__file__), "..","tools","financial_tools" ,"financial_server.py")],
    env=None,   # inherits current environment (API keys etc.)
)

# ---------------------------------------------------------------------------
# Peer map for relative valuation (tool_get_peer_comparison)
# ---------------------------------------------------------------------------
# get_peer_comparison() in rag_finance/yahoo_finance.py claims to
# auto-infer peers from the ticker's industry when none are given, but
# that path is broken: it fetches yf.Ticker(ticker).recommendations into
# a local variable and then immediately discards it, always falling back
# to peers=[] regardless of what recommendations returned. Rather than
# depend on a live "similar stocks" API (rate-limited, inconsistent
# across tickers), a small static map of common industries to typical
# public-market peers is used instead — good enough for a relative-
# valuation sanity check, not meant to be exhaustive. Falls back to no
# peer comparison (rather than guessing) for industries not listed here.


# ---------------------------------------------------------------------------
# INTENT TAXONOMY — what the Brain is allowed to choose
# ---------------------------------------------------------------------------
# The Brain used to emit freeform "priority_tools" strings, which (a) were
# discarded by _node_brain anyway and (b) would have let an LLM name tools
# this pipeline cannot supply arguments for. It now picks from this CLOSED
# enum instead; code alone maps an intent to a curated tool bundle, and code
# alone fills every numeric argument. Anything outside the enum is dropped.
INTENTS: frozenset[str] = frozenset({
    "valuation", "financial_health", "profitability", "growth",
    "liquidity", "cash_flow", "peer_relative", "overview", "full_report",
})

# Always run, regardless of intent, so nothing downstream starves: the
# Checker, the composite score and financial_metrics_summary all assume a
# yahoo_ratios + xbrl_financials floor exists.
BASELINE_EXTRACTION: tuple[str, ...] = (
    "tool_get_financial_ratios",
    "tool_get_xbrl_financials",
)
BASELINE_COMPUTE: tuple[str, ...] = ("tool_calc_composite_score",)

# Intent → curated bundle. Names here are the EXACT names registered in
# tools/financial_tools/financial_server.py — verified, not recalled.
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
        # Gated at call time, not here: peers come from the static
        # INDUSTRY_PEERS map keyed off the industry, which is only known
        # AFTER tool_get_financial_ratios returns. See _extract_peer_args.
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
    # NOTE: "full_report" is deliberately absent — it is a MACRO-intent
    # resolved as the union of _FULL_REPORT_PROFILES below, so it can never
    # drift out of sync with the narrow profiles it is meant to cover.
}

# Union expanded for the "full_report" macro-intent (see _resolve_tools).
_FULL_REPORT_PROFILES: tuple[str, ...] = (
    "valuation", "profitability", "growth", "financial_health",
    "liquidity", "cash_flow", "peer_relative",
)

# Heavy probabilistic valuation — dropped on depth=shallow. dcf_scenarios and
# reverse_dcf survive at every depth (they are single-pass, not 2000-draw
# simulations) so a shallow valuation still yields a fair-value range.
_MONTE_CARLO_TOOLS: frozenset[str] = frozenset({
    "tool_calc_dcf_monte_carlo", "tool_calc_dcf_multi_factor_monte_carlo",
})

# Deterministic execution order for compute tools. Two of the resolved tools
# read UPSTREAM results out of calculated_ratios rather than raw data
# (dcf_scenarios needs tool_build_forecast's scenarios; the multi-factor MC
# needs both; composite_score needs pe/roe/net_margin/de_ratio/cagr), so the
# dependency graph is expressed as an ordering here instead of a scheduler.
# Anything not listed runs, in registry order, before composite_score.
_COMPUTE_ORDER: tuple[str, ...] = (
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


def _normalize_depth(directives: dict[str, Any]) -> str:
    """
    Map whatever depth-ish key/value the Manager actually sent onto one of
    'shallow' | 'standard' | 'deep'. Deliberately tolerant: the request
    shape is not owned by this agent, so an unrecognised key or value
    degrades to 'standard' rather than silently dropping heavy tools.
    """
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


def _args(**kw: Any) -> dict[str, Any] | None:
    """
    Build a tool-argument dict, or return None if ANY value is missing.

    This is the single gate that keeps numbers deterministic: a mapper that
    cannot source every argument from raw_numerical_data returns None and the
    tool is skipped outright. No partial calls, no zero-filling, and — most
    importantly — no LLM-supplied numerics anywhere in the pipeline.
    """
    return None if any(v is None for v in kw.values()) else kw


# ---------------------------------------------------------------------------
# EXTRACTION REGISTRY — tool → (args_builder(ticker, raw) → dict | None, store_key)
# ---------------------------------------------------------------------------
# Every extraction tool in this pipeline is keyed off the ticker alone, with
# ONE exception: tool_get_peer_comparison also needs a peer list, which is
# derived from the industry that tool_get_financial_ratios just returned.
# An args_builder returning None means "skip, inputs unavailable".

def _extract_ticker_args(ticker: str, raw: dict[str, Any]) -> dict[str, Any] | None:
    return {"ticker": ticker}


def _extract_peer_args(ticker: str, raw: dict[str, Any]) -> dict[str, Any] | None:
    """
    Gate for tool_get_peer_comparison — UNCHANGED logic, just relocated.

    get_peer_comparison()'s own auto-inference is broken (it fetches
    recommendations into a local and discards them, always falling back to
    peers=[]), so peers come from the static INDUSTRY_PEERS map keyed off the
    industry/sector fetched by tool_get_financial_ratios. Industries absent
    from the map yield no peers and the call is skipped rather than guessed.
    """
    industry = raw.get("yahoo_ratios", {}).get("industry", "")
    sector = raw.get("yahoo_ratios", {}).get("sector", "")
    peers = infer_peers(sector, industry, ticker)
    return {"ticker": ticker, "peers": peers} if peers else None


EXTRACTION_REGISTRY: dict[str, tuple[Any, str]] = {
    "tool_get_financial_ratios":       (_extract_ticker_args, "yahoo_ratios"),
    "tool_get_peer_comparison":        (_extract_peer_args,   "peer_comparison"),
    "tool_get_revenue_growth":         (_extract_ticker_args, "revenue_growth"),
    "tool_get_xbrl_financials":        (_extract_ticker_args, "xbrl_financials"),
    "tool_get_balance_sheet_snapshot": (_extract_ticker_args, "balance_sheet_snapshot"),
}

# Extraction runs in this order so tool_get_financial_ratios (which supplies
# the industry) always precedes tool_get_peer_comparison (which needs it).
_EXTRACTION_ORDER: tuple[str, ...] = (
    "tool_get_financial_ratios",
    "tool_get_peer_comparison",
    "tool_get_revenue_growth",
    "tool_get_xbrl_financials",
    "tool_get_balance_sheet_snapshot",
)


# ---------------------------------------------------------------------------
# COMPUTE REGISTRY — tool → mapper(derived_facts) → args, entirely code-filled
# ---------------------------------------------------------------------------
# The mappers below are the ONLY place ratio arguments are constructed, and
# every value they read comes from raw_numerical_data (or from an upstream
# ratio already in calculated_ratios). The LLM never supplies a number: it
# chooses intents, and intents choose which of these mappers get to run.
#
# A mapper returns one of three things:
#   (args, period) → call the tool through _call() with that provenance label
#   dict           → a finished result needing no MCP round-trip (e.g. a value
#                    Yahoo already computed), stored verbatim
#   None           → inputs unavailable; the tool is SKIPPED and the spec's
#                    `unavailable` placeholder is stored instead

def _is_valid_number(x: Any) -> bool:
    """
    True for a real, JSON-safe number.

    DEFENSE-IN-DEPTH: excludes NaN, not just None. yfinance/pandas represents
    a missing cell in an otherwise-present row as `numpy.nan`, and
    `nan is not None` is True — so a None-only check let a NaN through as if
    it were valid data, silently voiding CAGR and breaking JSON persistence
    downstream ("NaN is not JSON compliant").
    """
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _newest(series: list[dict] | None, key: str = "value") -> Any:
    """Most recent entry's value from an XBRL-style newest-first series."""
    if not series:
        return None
    try:
        return series[0][key]
    except (KeyError, TypeError, IndexError):
        return None


def _derive_compute_facts(raw: dict[str, Any], calc: dict[str, Any]) -> dict[str, Any]:
    """
    Compute every deterministic intermediate the ratio mappers need, once.

    This is the arg-construction logic that used to live inline in
    _execute_ratio_computation, lifted out VERBATIM (same guards, same
    period-provenance labels, same approximation caveats) so the mappers stay
    one-liners and the numbers stay identical. Nothing here calls a tool or an
    LLM — it is pure arithmetic over already-extracted raw data.

    ``d["calc"]`` is a LIVE reference to state["calculated_ratios"], which is
    how the few order-dependent tools (dcf_scenarios needs build_forecast's
    output; composite_score needs pe/roe/net_margin/de_ratio/cagr) read their
    upstream results. _COMPUTE_ORDER guarantees the producer runs first.
    """
    ratios_src = raw.get("yahoo_ratios", {})
    growth_src = raw.get("revenue_growth", {})
    xbrl_src   = raw.get("xbrl_financials", {})
    bs_snapshot = raw.get("balance_sheet_snapshot", {})

    d: dict[str, Any] = {
        "calc": calc,
        "ratios_src": ratios_src, "growth_src": growth_src,
        "xbrl_src": xbrl_src, "bs_snapshot": bs_snapshot,
    }

    # -- P/E inputs --------------------------------------------------------
    d["price"] = ratios_src.get("current_price")
    d["eps"]   = ratios_src.get("eps_trailing")
    d["pe_ratio_yahoo"] = ratios_src.get("pe_ratio")

    # -- Return on Equity --------------------------------------------------
    xbrl_assets      = xbrl_src.get("total_assets", [])
    xbrl_liabilities = xbrl_src.get("total_liabilities", [])
    annual_ni        = growth_src.get("annual_net_income", [])
    d["xbrl_assets"] = xbrl_assets
    d["xbrl_liabilities"] = xbrl_liabilities
    d["annual_ni"] = annual_ni

    net_income      = annual_ni[0]["net_income"] if annual_ni else None
    net_income_year = annual_ni[0].get("year") if annual_ni else None
    # Approximate equity = assets - liabilities (most recent period).
    # NOTE: xbrl_assets/xbrl_liabilities come from sec_edgar.py's
    # _extract_annual(), which only pulls 10-K/FY entries — so this is an
    # ANNUAL balance-sheet snapshot, not a quarterly one, even though it's
    # paired here with the *current* net_income figure above.
    shareholders_equity: float | None = None
    equity_period_end: str | None = None
    if xbrl_assets and xbrl_liabilities:
        try:
            shareholders_equity = xbrl_assets[0]["value"] - xbrl_liabilities[0]["value"]
            equity_period_end = xbrl_assets[0].get("period_end")
        except (KeyError, TypeError, IndexError):
            shareholders_equity = None

    # ROE mixes an annual (10-K) net_income with an annual (10-K) balance
    # sheet — same source family, but don't just assume index [0] on both
    # lists lines up to the same fiscal year; check it.
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

    # -- Net Margin --------------------------------------------------------
    # quarterly_net_income mirrors quarterly_revenue (see yahoo_finance.py).
    # Prefer the most recent MATCHING quarter for both so net_margin
    # describes the SAME quarter as the gross/operating margins the report
    # writer pulls from the 10-Q's quarterly MD&A text. Only fall back to
    # annual figures — clearly tagged — when quarterly data isn't available.
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
        # Fallback: most recent ANNUAL (10-K/FY) figures, flagged MIXED if the
        # two annual arrays' index-[0] years don't actually agree, so a caller
        # narrating this next to quarterly text can catch the period
        # difference instead of silently mismatching them.
        d["nm_net_income"] = net_income
        d["nm_revenue"]    = annual_rev[0]["revenue"] if annual_rev else None
        d["net_margin_period"] = (
            f"MIXED(net_income=annual:{net_income_year}, revenue=annual:{revenue_year})"
            if net_income_year is not None and revenue_year is not None and net_income_year != revenue_year
            else f"annual:{revenue_year}" if revenue_year is not None
            else None
        )

    # -- Debt-to-Equity ----------------------------------------------------
    d["de_ratio_yahoo"] = ratios_src.get("de_ratio") or ratios_src.get("debtToEquity")
    d["total_liabilities_val"] = _newest(xbrl_liabilities)

    # -- Revenue CAGR ------------------------------------------------------
    # annual_rev may contain entries whose "revenue" is None (the oldest
    # fiscal year often has no value from yfinance). Using annual_rev[-1]
    # blindly picks that null and voids the whole CAGR even when 3-4 valid
    # years are present — filter to valid points first.
    valid_rev = [
        e for e in annual_rev
        if _is_valid_number(e.get("revenue")) and e.get("year") is not None
    ]
    valid_rev.sort(key=lambda e: e["year"])  # oldest -> newest
    d["valid_rev"] = valid_rev
    d["cagr_start"] = d["cagr_end"] = d["cagr_years"] = None
    d["cagr_period"] = None
    if len(valid_rev) >= 2:
        oldest, newest = valid_rev[0]["revenue"], valid_rev[-1]["revenue"]
        years = valid_rev[-1]["year"] - valid_rev[0]["year"]
        if newest and oldest and oldest > 0 and years > 0:
            d["cagr_start"], d["cagr_end"], d["cagr_years"] = oldest, newest, float(years)
            d["cagr_period"] = f"annual:{valid_rev[0]['year']}-{valid_rev[-1]['year']}"

    # -- Valuation / DCF inputs -------------------------------------------
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

    tax_rate_raw = ratios_src.get("effective_tax_rate")
    d["effective_tax_rate_pct"] = (
        float(tax_rate_raw) * 100 if tax_rate_raw is not None and abs(float(tax_rate_raw)) <= 1
        else float(tax_rate_raw) if tax_rate_raw is not None else 21.0
    )
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
        tax_rate_observed=tax_rate_raw is not None,
        pre_tax_cost_of_debt_pct=d["pre_tax_cost_of_debt_pct"],
        effective_tax_rate_pct=d["effective_tax_rate_pct"],
        sector=ratios_src.get("sector"),
    )

    # FCF = operating cash flow − capex (most recent year, both from
    # tool_get_xbrl_financials, sorted newest-first by _extract_annual).
    # capex is reported as a positive outflow in XBRL's
    # PaymentsToAcquirePropertyPlantAndEquipment tag (a "payments" concept,
    # not a signed reduction), so it is subtracted rather than added.
    fcf_base = None
    if operating_cf_annual and capex_annual:
        try:
            fcf_base = operating_cf_annual[0]["value"] - capex_annual[0]["value"]
        except (KeyError, TypeError, IndexError):
            fcf_base = None
    d["fcf_base"] = fcf_base
    d["revenue_base"] = valid_rev[-1]["revenue"] if valid_rev else None
    d["revenue_base_year"] = valid_rev[-1]["year"] if valid_rev else None

    trajectory_raw = growth_src.get("revenue_growth_ttm")
    if trajectory_raw is None and len(valid_rev) >= 2 and valid_rev[-2]["revenue"]:
        trajectory_raw = valid_rev[-1]["revenue"] / valid_rev[-2]["revenue"] - 1.0
    d["current_growth_pct"] = (
        float(trajectory_raw) * 100.0
        if trajectory_raw is not None and abs(float(trajectory_raw)) <= 2.0
        else float(trajectory_raw) if trajectory_raw is not None else None
    )
    d["capex_pct_revenue"] = (
        capex_annual[0]["value"] / d["revenue_base"] * 100.0
        if capex_annual and d["revenue_base"] else None
    )

    # Reverse DCF target EV ≈ market_cap + total_debt − total_cash (all TTM);
    # see reverse_dcf's own docstring for why raw market_cap alone would bias
    # the result.
    d["target_ev"] = (
        d["market_cap"] + d["total_debt_ttm"] - d["total_cash_ttm"]
        if d["market_cap"] is not None and d["total_debt_ttm"] is not None
        and d["total_cash_ttm"] is not None else None
    )

    # -- Balance-sheet snapshot / Phase-2 inputs ---------------------------
    d["total_assets_val"]        = _newest(xbrl_assets)
    d["ebit_snap"]               = bs_snapshot.get("ebit")
    d["retained_earnings_snap"]  = bs_snapshot.get("retained_earnings")
    d["working_capital_snap"]    = bs_snapshot.get("working_capital")
    d["current_assets_snap"]     = bs_snapshot.get("current_assets")
    d["current_liabilities_snap"] = bs_snapshot.get("current_liabilities")
    d["total_equity_snap"]       = bs_snapshot.get("total_equity")
    d["bs_period_end"]           = bs_snapshot.get("period_end", "unknown")
    d["fcf_margin_revenue"]      = annual_rev[0]["revenue"] if annual_rev else None
    d["altman_revenue"]          = annual_rev[0]["revenue"] if annual_rev else None
    d["ocf_annual"]              = _newest(operating_cf_annual)
    d["current_ratio_val"]       = ratios_src.get("current_ratio")

    # -- Newly-reachable inputs (previously no mapper existed) -------------
    # Book value per share: total_equity is the yfinance balance-sheet
    # snapshot; diluted_shares is yfinance .info sharesOutstanding. Both are
    # observed fields — nothing is imputed.
    d["book_value_per_share"] = (
        d["total_equity_snap"] / d["observed_diluted_shares"]
        if d["total_equity_snap"] is not None and d["observed_diluted_shares"] else None
    )
    # PEG's growth leg uses NET INCOME growth (not revenue growth) because
    # PEG is by definition earnings-relative. Sourced from the two most
    # recent annual_net_income entries returned by tool_get_revenue_growth.
    d["earnings_growth_pct"] = None
    if len(annual_ni) >= 2:
        prev = annual_ni[1].get("net_income")
        curr = annual_ni[0].get("net_income")
        if _is_valid_number(prev) and _is_valid_number(curr) and prev and prev > 0:
            d["earnings_growth_pct"] = (curr - prev) / prev * 100.0
    d["operating_income_annual"] = _newest(xbrl_src.get("operating_income_annual"))
    d["xbrl_revenue_annual"]     = _newest(xbrl_src.get("revenue_annual"))
    return d


class ComputeSpec(NamedTuple):
    """One row of COMPUTE_REGISTRY. See the module comment above for the
    mapper's three possible return shapes."""
    key:         str                                   # calculated_ratios key
    mapper:      Callable[[dict[str, Any]], Any]
    unavailable: dict[str, Any]                        # stored when mapper → None
    post:        Callable[[dict[str, Any], dict[str, Any]], None] | None = None


# US federal statutory rate, used as a DOCUMENTED placeholder for ROIC's tax
# leg — this pipeline has no pretax-income/tax-provision series to derive the
# company's real effective rate from. Flagged on the result, never hidden.
_STATUTORY_TAX_RATE_PCT = 21.0


def _post_de_ratio(result: dict[str, Any], d: dict[str, Any]) -> None:
    """
    tool_get_xbrl_financials only exposes total_liabilities, not
    interest-bearing debt separately — so the XBRL D/E path includes
    non-interest-bearing liabilities (payables, deferred revenue) that a true
    D/E excludes. The tool's threshold interpretation is calibrated for TRUE
    D/E, so override it rather than let the mismatch hide.
    """
    result["formula"] = "total_liabilities / shareholders_equity (approximation)"
    result["interpretation"] = "approximation_not_comparable_to_standard_thresholds"
    result["is_approximation"] = True
    result["note"] = (
        "tool_get_xbrl_financials only exposes total_liabilities, not "
        "interest-bearing debt separately, so this uses total liabilities "
        "as a rough stand-in. This is NOT a true debt-to-equity ratio and "
        "should not be compared against standard D/E leverage thresholds."
    )


def _post_roic(result: dict[str, Any], d: dict[str, Any]) -> None:
    result["is_approximation"] = True
    result["note"] = (
        f"tax_rate_pct used the US statutory rate ({_STATUTORY_TAX_RATE_PCT}%) as a "
        "placeholder, not this company's actual effective tax rate (no pretax-income/"
        "tax-provision series available in this pipeline) -- NOPAT here is an approximation."
    )


def _map_pe(d):
    args = _args(price=d["price"], eps=d["eps"])
    if args and d["eps"] != 0:
        return args, "ttm"
    # Yahoo already published a trailing P/E — use it rather than skip the
    # metric entirely, but label the provenance so it is never mistaken for
    # a locally-computed figure.
    return {
        "pe_ratio": d["pe_ratio_yahoo"],
        "interpretation": "sourced_from_yahoo",
        "note": "Used pre-computed Yahoo Finance P/E (price or EPS unavailable for direct calc).",
        "_inputs": {"pe_ratio": d["pe_ratio_yahoo"]},
        "_period": "ttm",
    }


def _map_pb(d):
    args = _args(price=d["price"], book_value_per_share=d["book_value_per_share"])
    return (args, f"MIXED(price=ttm, book_value={d['bs_period_end']})") if args else None


def _map_ev_ebitda(d):
    args = _args(enterprise_value=d["enterprise_value"], ebitda=d["ebitda_ttm"])
    return (args, "ttm") if args and d["ebitda_ttm"] != 0 else None


def _map_peg(d):
    pe = d["calc"].get("pe", {}).get("pe_ratio") or d["pe_ratio_yahoo"]
    args = _args(pe=pe, earnings_growth_rate_pct=d["earnings_growth_pct"])
    return (args, "MIXED(pe=ttm, earnings_growth=annual_yoy)") if args else None


def _map_gross_margin(d):
    # No extraction tool in this pipeline supplies COGS / cost of revenue:
    # sec_edgar.get_xbrl_financials() tags revenue, net/operating/pretax
    # income, tax, assets, liabilities and cash flow — but not cost of
    # revenue — and yfinance .info has no COGS field either. Always skipped
    # rather than derived from a margin the tool is meant to produce.
    # TODO: add a _COST_OF_REVENUE_TAGS series to sec_edgar.py to enable this.
    return None


def _map_operating_margin(d):
    args = _args(
        operating_income=d["operating_income_annual"], revenue=d["xbrl_revenue_annual"],
    )
    return (args, "annual") if args and d["xbrl_revenue_annual"] != 0 else None


def _map_net_margin(d):
    args = _args(net_income=d["nm_net_income"], revenue=d["nm_revenue"])
    return (args, d["net_margin_period"]) if args and d["nm_revenue"] != 0 else None


def _map_roe(d):
    args = _args(
        net_income=d["net_income"], shareholders_equity=d["shareholders_equity"],
    )
    return (args, d["roe_period"]) if args and d["net_income"] else None


def _map_roa(d):
    args = _args(net_income=d["net_income"], total_assets=d["total_assets_val"])
    return (
        args,
        f"MIXED(net_income=annual:{d['net_income_year']}, total_assets=annual:{d['equity_period_end']})",
    ) if args and d["total_assets_val"] else None


def _map_roic(d):
    args = _args(
        ebit=d["ebit_snap"], tax_rate_pct=_STATUTORY_TAX_RATE_PCT,
        total_debt=d["total_debt_ttm"], total_equity=d["shareholders_equity"],
        cash=d["total_cash_ttm"],
    )
    return (args, (
        f"MIXED(ebit={d['bs_period_end']}, equity=annual:{d['equity_period_end']}, debt/cash=ttm)"
    )) if args else None


def _map_roce(d):
    args = _args(
        ebit=d["ebit_snap"], total_assets=d["total_assets_val"],
        current_liabilities=d["current_liabilities_snap"],
    )
    return (args, (
        f"MIXED(ebit={d['bs_period_end']}, total_assets=annual:{d['equity_period_end']})"
    )) if args and d["total_assets_val"] else None


def _map_debt_to_equity(d):
    # Yahoo's .info["debtToEquity"] is a most-recent-quarter balance-sheet
    # snapshot and takes priority — no MCP round-trip needed for it.
    if d["de_ratio_yahoo"] is not None:
        return {
            "de_ratio": d["de_ratio_yahoo"],
            "interpretation": "sourced_from_yahoo",
            "_inputs": {"de_ratio_yahoo": d["de_ratio_yahoo"]},
            "_period": "mrq",
        }
    args = _args(
        total_debt=d["total_liabilities_val"], shareholders_equity=d["shareholders_equity"],
    )
    return (
        args,
        f"annual:{d['equity_period_end']}" if d["equity_period_end"] else None,
    ) if args and d["total_liabilities_val"] else None


def _map_debt_ratio(d):
    # total_liabilities and total_assets are BOTH from the same XBRL annual
    # snapshot (index [0] of each series) — clean annual tag, not mixed.
    args = _args(
        total_liabilities=d["total_liabilities_val"], total_assets=d["total_assets_val"],
    )
    return (
        args, f"annual:{d['equity_period_end']}" if d["equity_period_end"] else "annual",
    ) if args and d["total_assets_val"] else None


def _map_equity_ratio(d):
    args = _args(total_equity=d["shareholders_equity"], total_assets=d["total_assets_val"])
    return (
        args, f"annual:{d['equity_period_end']}" if d["equity_period_end"] else "annual",
    ) if args and d["total_assets_val"] else None


def _map_interest_coverage(d):
    args = _args(ebit=d["ebit_snap"], interest_expense=d["interest_expense_ttm"])
    return (args, f"MIXED(ebit={d['bs_period_end']}, interest_expense=ttm)") if args else None


def _map_net_debt_to_ebitda(d):
    args = _args(
        total_debt=d["total_debt_ttm"], total_cash=d["total_cash_ttm"], ebitda=d["ebitda_ttm"],
    )
    return (args, "ttm") if args and d["ebitda_ttm"] else None


def _map_current_ratio(d):
    args = _args(
        current_assets=d["current_assets_snap"], current_liabilities=d["current_liabilities_snap"],
    )
    return (args, f"snapshot:{d['bs_period_end']}") if args else None


def _map_quick_ratio(d):
    # tool_calc_quick_ratio needs cash + short_term_investments + receivables
    # separately. get_balance_sheet_snapshot() returns total/current assets
    # and liabilities but neither receivables nor short-term investments, and
    # yfinance .info's totalCash is a COMBINED cash + short-term-investments
    # bucket that cannot be split back apart. Skipped rather than guessed —
    # yahoo_ratios["quick_ratio"] is still carried through raw data untouched.
    # TODO: add receivables + short_term_investments to
    #       yahoo_finance.get_balance_sheet_snapshot() to enable this.
    return None


def _map_ocf_ratio(d):
    args = _args(
        operating_cash_flow=d["ocf_annual"], current_liabilities=d["current_liabilities_snap"],
    )
    return (args, f"MIXED(ocf=annual, current_liabilities={d['bs_period_end']})") if args else None


def _map_altman_z_score(d):
    args = _args(
        working_capital=d["working_capital_snap"], retained_earnings=d["retained_earnings_snap"],
        ebit=d["ebit_snap"], total_assets=d["total_assets_val"], market_cap=d["market_cap"],
        total_liabilities=d["total_liabilities_val"], revenue=d["altman_revenue"],
    )
    return (args, "mixed") if args and d["total_assets_val"] else None


def _map_cagr(d):
    args = _args(
        start_value=d["cagr_start"], end_value=d["cagr_end"], years=d["cagr_years"],
    )
    return (args, d["cagr_period"]) if args else None


def _map_revenue_cagr_from_growth(d):
    # Passes the annual_revenue list straight through — the tool sorts by
    # year and picks its own endpoints, so this needs no local arithmetic.
    return (
        ({"annual_revenue": d["annual_rev"]}, "annual")
        if len(d["annual_rev"]) >= 2 else None
    )


def _map_fcf_margin(d):
    # fcf_base is annual (XBRL); revenue is the most recent ANNUAL figure to
    # stay in the same source family, but the two annual series (XBRL cash
    # flow vs. yfinance income statement) aren't guaranteed to be the exact
    # same fiscal year — tag MIXED rather than assume alignment.
    args = _args(free_cash_flow=d["fcf_base"], revenue=d["fcf_margin_revenue"])
    return (
        args, f"MIXED(fcf=annual_derived, revenue=annual:{d['revenue_year']})",
    ) if args and d["fcf_margin_revenue"] else None


def _map_fcf_yield(d):
    args = _args(free_cash_flow=d["fcf_base"], market_cap=d["market_cap"])
    return (args, "MIXED(fcf=annual_derived, market_cap=ttm)") if args and d["market_cap"] else None


def _map_build_forecast(d):
    # Ground the base-case FCF growth assumption in the company's own revenue
    # CAGR (computed upstream by tool_calc_cagr — see _COMPUTE_ORDER).
    historical_cagr_pct = d["calc"].get("cagr", {}).get("cagr_pct")
    if d["fcf_base"] is None or d["beta"] is None or not d["revenue_base"]:
        return None
    if historical_cagr_pct is None:
        # No CAGR to anchor the assumption — skip rather than guessing a base
        # growth rate out of thin air.
        return None
    return {
        "revenue_base": d["revenue_base"],
        "fcf_base": d["fcf_base"],
        "historical_revenue_cagr_pct": historical_cagr_pct,
        "current_revenue_growth_pct": d["current_growth_pct"],
        "capex_pct_revenue": d["capex_pct_revenue"],
        "projection_years": 5,
        "source_period": f"annual:{d['revenue_base_year']}",
    }, "derived:annual_projection"


def _map_dcf_scenarios(d):
    """
    Uses the scenario-range tool (not the single-point tool_calc_dcf)
    deliberately: a single DCF using trailing growth routinely lands far
    below market price for a high-growth stock, which reads as "the model is
    broken" unless growth-rate sensitivity is shown explicitly.
    """
    forecast = d["calc"].get("forecast", {})
    if d["fcf_base"] is None or d["beta"] is None or not forecast.get("scenarios"):
        return None
    return {
        "fcf_base": d["fcf_base"],
        "beta": d["beta"],
        "shares_outstanding": d["shares_outstanding"],
        "diluted_shares": d["shares_outstanding"],
        "market_cap": d["market_cap"],
        "total_debt": d["total_debt_ttm"],
        "cash": d["total_cash_ttm"],
        # Kept None rather than silently converted to zero. The assessment
        # marks it NOT_APPLICABLE because Yahoo totalCash is the combined
        # cash/short-term-investment bucket.
        "marketable_securities": None,
        "lease_liabilities": d["lease_liabilities_ttm"],
        "minority_interest": d["minority_interest_ttm"],
        "pre_tax_cost_of_debt_pct": d["pre_tax_cost_of_debt_pct"],
        "effective_tax_rate_pct": d["effective_tax_rate_pct"],
        "input_assessments": d["valuation_input_assessments"],
        "forecast_scenarios": forecast.get("scenarios", {}),
    }, "derived:annual_projection"


def _map_dcf_multi_factor_monte_carlo(d):
    """
    Runs alongside the three named scenarios: bear/base/bull are easier to
    narrate, while P10/P50/P90 percentiles give a continuous picture of
    valuation uncertainty rather than three discrete points.
    """
    base_dcf = d["calc"].get("dcf", {}).get("base", {})
    base_forecast = d["calc"].get("forecast", {}).get("scenarios", {}).get("base", {})
    if base_dcf.get("error") is not None or not base_forecast.get("fcf_by_year"):
        return None
    return {
        "fcf_by_year": base_forecast["fcf_by_year"],
        "wacc_pct": base_dcf.get("wacc_pct"),
        "terminal_growth_rate_pct": base_dcf.get("assumptions", {}).get("terminal_growth_rate_pct", 2.5),
        "cash": d["total_cash_ttm"] or 0.0,
        "total_debt": d["total_debt_ttm"] or 0.0,
        "lease_liabilities": d["lease_liabilities_ttm"] or 0.0,
        "minority_interest": d["minority_interest_ttm"] or 0.0,
        "diluted_shares": d["shares_outstanding"],
        "n_simulations": 2000,
    }, "derived:annual_projection"


def _map_dcf_monte_carlo(d):
    """
    Single-factor MC: randomises the FCF growth rate only, anchored on the
    company's own revenue CAGR. Complements the multi-factor variant.
    """
    args = _args(
        fcf_base=d["fcf_base"], beta=d["beta"],
        base_growth_rate_pct=d["calc"].get("cagr", {}).get("cagr_pct"),
    )
    if args is None:
        return None
    args["shares_outstanding"] = d["shares_outstanding"]
    args["n_simulations"] = 2000
    return args, "derived:annual_projection"


def _map_reverse_dcf(d):
    args = _args(
        fcf_base=d["fcf_base"], beta=d["beta"], target_enterprise_value=d["target_ev"],
        market_cap=d["market_cap"], total_debt=d["total_debt_ttm"],
    )
    if args is None or d["target_ev"] <= 0:
        return None
    args["pre_tax_cost_of_debt_pct"] = d["pre_tax_cost_of_debt_pct"]
    args["effective_tax_rate_pct"] = d["effective_tax_rate_pct"]
    return args, "MIXED(fcf=annual_derived, market_cap/debt/cash=ttm)"


def _map_composite_score(d):
    """
    DELIBERATELY built from ratios spanning several periods (ttm P/E, annual
    ROE/net_margin, mrq current/D-E ratio, multi-year CAGR) — fine for a
    single composite health score, but tagged "mixed" explicitly so
    validate_period_consistency.py doesn't flag it the way it flags an
    *accidental* mix. Unlike every other mapper this does NOT gate on None:
    the tool is designed to score partial inputs and reports which were
    missing, so passing the Nones through is the informative behaviour.
    """
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
    "tool_calc_pe":               ComputeSpec("pe", _map_pe, {"pe_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_pb":               ComputeSpec("pb", _map_pb, {"pb_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_ev_ebitda":        ComputeSpec("ev_ebitda", _map_ev_ebitda, {"ev_ebitda": None, "interpretation": "unavailable"}),
    "tool_calc_peg":              ComputeSpec("peg", _map_peg, {"peg_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_gross_margin":     ComputeSpec("gross_margin", _map_gross_margin, {"gross_margin_pct": None, "interpretation": "unavailable"}),
    "tool_calc_operating_margin": ComputeSpec("operating_margin", _map_operating_margin, {"operating_margin_pct": None, "interpretation": "unavailable"}),
    "tool_calc_net_margin":       ComputeSpec("net_margin", _map_net_margin, {"net_margin_pct": None, "interpretation": "insufficient_data"}),
    "tool_calc_roe":              ComputeSpec("roe", _map_roe, {"roe_pct": None, "interpretation": "insufficient_data"}),
    "tool_calc_roa":              ComputeSpec("roa", _map_roa, {"roa_pct": None, "interpretation": "unavailable"}),
    "tool_calc_roic":             ComputeSpec("roic", _map_roic, {"roic_pct": None, "interpretation": "unavailable"}, _post_roic),
    "tool_calc_roce":             ComputeSpec("roce", _map_roce, {"roce_pct": None, "interpretation": "unavailable"}),
    "tool_calc_debt_to_equity":   ComputeSpec("de_ratio", _map_debt_to_equity, {"de_ratio": None, "interpretation": "unavailable"}, _post_de_ratio),
    "tool_calc_debt_ratio":       ComputeSpec("debt_ratio", _map_debt_ratio, {"debt_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_equity_ratio":     ComputeSpec("equity_ratio", _map_equity_ratio, {"equity_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_interest_coverage": ComputeSpec("interest_coverage", _map_interest_coverage, {"interest_coverage": None, "interpretation": "unavailable"}),
    "tool_calc_net_debt_to_ebitda": ComputeSpec("net_debt_to_ebitda", _map_net_debt_to_ebitda, {"net_debt_to_ebitda": None, "interpretation": "unavailable"}),
    "tool_calc_current_ratio":    ComputeSpec("current_ratio_calc", _map_current_ratio, {"current_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_quick_ratio":      ComputeSpec("quick_ratio", _map_quick_ratio, {"quick_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_ocf_ratio":        ComputeSpec("ocf_ratio", _map_ocf_ratio, {"ocf_ratio": None, "interpretation": "unavailable"}),
    "tool_calc_altman_z_score":   ComputeSpec("altman_z_score", _map_altman_z_score, {"z_score": None, "interpretation": "unavailable"}),
    "tool_calc_cagr":             ComputeSpec("cagr", _map_cagr, {"cagr_pct": None, "interpretation": "insufficient_history"}),
    "tool_calc_revenue_cagr_from_growth": ComputeSpec("revenue_cagr_from_growth", _map_revenue_cagr_from_growth, {"cagr_pct": None, "interpretation": "insufficient_history"}),
    "tool_calc_fcf_margin":       ComputeSpec("fcf_margin", _map_fcf_margin, {"fcf_margin_pct": None, "interpretation": "unavailable"}),
    "tool_calc_fcf_yield":        ComputeSpec("fcf_yield", _map_fcf_yield, {"fcf_yield_pct": None, "interpretation": "unavailable"}),
    "tool_build_forecast":        ComputeSpec("forecast", _map_build_forecast, {"interpretation": "unavailable", "error": "Missing fcf_base/beta/revenue_base, or no revenue_cagr available to anchor the base-case FCF growth assumption."}),
    "tool_calc_dcf_scenarios":    ComputeSpec("dcf", _map_dcf_scenarios, {"enterprise_value": None, "interpretation": "unavailable", "error": "Missing operating_cash_flow_annual/capex_annual from XBRL (fcf_base), beta from Yahoo ratios, or a base forecast — cannot run DCF."}),
    "tool_calc_dcf_multi_factor_monte_carlo": ComputeSpec("dcf_monte_carlo", _map_dcf_multi_factor_monte_carlo, {"interpretation": "unavailable", "model_classification": "MULTI_FACTOR", "error": "A valid explicit base forecast and WACC are required."}),
    "tool_calc_dcf_monte_carlo":  ComputeSpec("dcf_monte_carlo_growth", _map_dcf_monte_carlo, {"interpretation": "unavailable", "model_classification": "SINGLE_FACTOR", "error": "Missing fcf_base, beta, or a revenue CAGR to anchor the growth distribution."}),
    "tool_calc_reverse_dcf":      ComputeSpec("reverse_dcf", _map_reverse_dcf, {"implied_growth_rate_pct": None, "interpretation": "unavailable", "error": "Missing fcf_base, beta, or a valid target enterprise value (market_cap + total_debt − total_cash)."}),
    "tool_calc_composite_score":  ComputeSpec("composite_score", _map_composite_score, {"score": None, "interpretation": "unavailable"}),
}


# ══════════════════════════════════════════════════════════════════════════════
# FinancialAnalystAgent
# ══════════════════════════════════════════════════════════════════════════════

class FinancialAnalystAgent:
    """
    Specialist Financial Analyst Agent for the Alpha-Agent Node platform.

    Responsibilities
    ----------------
    1. Receive a research task from the Manager Agent via ``SharedManagerState``.
    2. Extract the target ticker symbol from the natural-language ``task_query``.
    3. Retrieve raw market data and SEC financials from MCP-hosted tool servers.
    4. Compute verified financial ratios (P/E, ROE, D/E, CAGR, composite score)
       by calling the FinancialRatioCalculator MCP server.
    5. Validate the completeness and mathematical consistency of all gathered data.
    6. Commit a structured ``financial_metrics_summary`` back into the shared
       state for consumption by downstream agents (Sentiment Agent, Report Writer).

    Parameters
    ----------
    server_script_path : str
        Absolute path to ``server.py`` — the FastMCP stdio server script that
        exposes Yahoo Finance, SEC EDGAR, and ratio-calculator tools.
    model : str
        Anthropic model identifier. Defaults to ``"claude-haiku-4-5"``.
    max_loops : int
        Default safety guardrail for the internal extraction loop.
        Can be overridden by ``manager_directives["max_loops"]``. Defaults to 3.

    Attributes
    ----------
    _llm : anthropic.Anthropic
        Anthropic SDK client used for Brain planning calls.
    _server_params : StdioServerParameters
        Connection parameters for launching the MCP server subprocess.
    _model : str
        Model string forwarded to every Anthropic API call.
    _default_max_loops : int
        Fallback loop limit when the Manager does not specify one.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_loops: int = 3,
        mcp_server_params: StdioServerParameters | None = None,
        llm_client: anthropic.Anthropic | None = None,
    ) -> None:
        # Accept an injected client so tests can pass a mock without
        # making real API calls.
        self._llm    = llm_client or anthropic.Anthropic()
        self._model  = model
        self._default_max_loops = max_loops
        self._server_params = mcp_server_params or _MCP_SERVER_PARAMS

        # MCP session is opened once per run() (see run()) and shared across
        # all graph nodes via this slot. Nodes receive only `state`, so they
        # read the live session from here rather than as a parameter.
        self._session: ClientSession | None = None

        # Compile the internal LangGraph state machine once at construction.
        self._graph = self._build_graph()

        log.info(
            "FinancialAnalystAgent initialised — model=%s, server_script=%s",
            model,
            self._server_params.args[-1] if self._server_params.args else "<none>",
        )

    # =========================================================================
    # INTERNAL UTILITY — ticker extraction
    # =========================================================================

    # =========================================================================

    # =========================================================================
    # LAYER 1 — EXECUTORS (MCP Protocol Interface)
    # =========================================================================

    @traceable(name="financial.extract", run_type="tool")
    async def _execute_data_extraction(
        self,
        session: ClientSession,
        state:   FinancialAgentState,
    ) -> None:
        """
        EXECUTOR — Raw Data Extraction via MCP Tool Calls.

        Parses ``task_query`` to determine the target ticker, then dispatches
        asynchronous MCP tool calls to the YahooFinanceClient and SecEdgarParser
        servers. All retrieved data is written into ``state["raw_numerical_data"]``.

        INTENT-DRIVEN: the set of tools called is no longer fixed. It comes from
        ``_resolve_tools(state)``, which expands the Brain's classified intents
        into a curated bundle (always including BASELINE_EXTRACTION). Each tool
        is dispatched through the single loop below via EXTRACTION_REGISTRY —
        the five copy-pasted call blocks this replaced were identical apart from
        the tool name and the store key.

        MCP Tools Available (selected per intent)
        ----------------------------------------
        - ``tool_get_financial_ratios``  → P/E, P/B, EPS, market cap, beta, etc.
        - ``tool_get_revenue_growth``    → Annual/quarterly revenue & net income trends.
        - ``tool_get_xbrl_financials``   → SEC EDGAR XBRL structured balance-sheet data.
        - ``tool_get_balance_sheet_snapshot`` → EBIT, current assets/liabilities,
          retained earnings (yfinance) -- feeds ROIC/ROCE/debt_ratio/
          equity_ratio/altman_z_score in _execute_ratio_computation.
        - ``tool_get_peer_comparison``   → Relative valuation vs. static peers
          (skipped when no peers are mapped for the ticker's industry).

        Parameters
        ----------
        session : ClientSession
            Active MCP client session connected to the financial server subprocess.
        state : FinancialAgentState
            Mutable agent-local state. This method appends data under:
            ``state["raw_numerical_data"]["yahoo_ratios"]``
            ``state["raw_numerical_data"]["revenue_growth"]``
            ``state["raw_numerical_data"]["xbrl_financials"]``
            ``state["raw_numerical_data"]["balance_sheet_snapshot"]``
            ``state["raw_numerical_data"]["peer_comparison"]``

        Returns
        -------
        None
            Results are written directly into ``state``. Errors are recorded
            under ``state["raw_numerical_data"]["extraction_errors"]``.
        """
        directives = state["manager_directives"]
        task_query = state["task_query"]
        ticker     = extract_ticker(task_query, directives)

        if not ticker:
            state["raw_numerical_data"]["extraction_errors"] = (
                "Could not determine ticker symbol from task_query or manager_directives."
            )
            log.error("Executor: ticker extraction failed — aborting data pull.")
            return

        state["raw_numerical_data"]["ticker"] = ticker
        raw = state["raw_numerical_data"]
        errors: list[str] = []
        session_id = session_from_directives(state["manager_directives"])
        _publish_progress(
            session_id, "agent_brain", agent="financial",
            message=f"Financial Agent: starting raw data extraction for {ticker}",
            detail={"ticker": ticker},
        )

        extraction_tools, _ = self._resolve_tools(state)
        log.info("Executor: intent-resolved extraction tools = %s", extraction_tools)

        for idx, tool in enumerate(extraction_tools):
            args_builder, store_key = EXTRACTION_REGISTRY[tool]
            args = args_builder(ticker, raw)

            # A None args dict means the tool's inputs cannot be sourced from
            # this ticker — skip it rather than call with fabricated values.
            # Only tool_get_peer_comparison currently gates this way; its
            # "no peers mapped" note is preserved verbatim.
            if args is None:
                industry = raw.get("yahoo_ratios", {}).get("industry", "")
                log.info(
                    "Executor: skipping %s — inputs unavailable for ticker=%s "
                    "(industry=%r).", tool, ticker, industry,
                )
                if tool == "tool_get_peer_comparison":
                    raw["peer_comparison"] = {
                        "note": f"No peer mapping available for industry={industry!r}.",
                    }
                continue

            log.info("Executor: calling %s for %s", tool, ticker)
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: calling tool '{tool}'...",
                detail={"tool": tool, **({"peers": args["peers"]} if "peers" in args else {})},
            )
            try:
                if sentry_enabled():
                    import sentry_sdk
                    sentry_sdk.add_breadcrumb(
                        category="mcp.financial",
                        message=f"Calling {tool}",
                        data={"tool": tool, "ticker": ticker},
                        level="info",
                    )
                result = await session.call_tool(tool, arguments=args)
                data = json.loads(result.content[0].text)
                if data.get("error"):
                    errors.append(f"{tool} error: {data['error']}")
                raw[store_key] = data
            except Exception as exc:
                errors.append(f"{tool} exception: {exc}")
                raw[store_key] = {}
                log.exception("Executor: %s failed", tool)
                if sentry_enabled():
                    import sentry_sdk
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("tool", tool)
                        scope.set_tag("component", "mcp.financial")
                        sentry_sdk.capture_exception(exc)

            is_last = idx == len(extraction_tools) - 1
            _publish_progress(
                session_id, "agent_tool_result", agent="financial",
                message=(
                    f"Financial Agent: '{tool}' completed — data extraction done"
                    if is_last else f"Financial Agent: '{tool}' completed"
                ),
                detail=(
                    {"tool": tool, "errors": len(errors)} if is_last else {"tool": tool}
                ),
            )

        if errors:
            raw["extraction_errors"] = errors
        log.info(
            "Executor: data extraction complete — %d error(s) recorded.", len(errors)
        )

    @traceable(name="financial.compute", run_type="tool")
    async def _execute_ratio_computation(
        self,
        session: ClientSession,
        state:   FinancialAgentState,
    ) -> None:
        """
        EXECUTOR — Financial Ratio Computation via MCP Tool Calls.

        Reads validated numeric inputs from ``state["raw_numerical_data"]`` and
        dispatches MCP tool calls to the FinancialRatioCalculator server to
        compute standardised ratios. Results are stored in
        ``state["calculated_ratios"]``.

        INTENT-DRIVEN: which ratios get computed comes from
        ``_resolve_tools(state)`` — the Brain's classified intents expanded into
        a curated bundle (always including BASELINE_COMPUTE). Every argument is
        still built by code, in COMPUTE_REGISTRY's mappers, from
        ``raw_numerical_data`` only. The LLM supplies no numbers, ever.

        Order matters and is fixed by ``_COMPUTE_ORDER``: tool_build_forecast
        feeds tool_calc_dcf_scenarios, which feeds the multi-factor Monte
        Carlo, and tool_calc_composite_score consumes pe/roe/net_margin/
        de_ratio/cagr. Mappers read those upstream results through the live
        ``calculated_ratios`` reference in the derived-facts dict.

        Parameters
        ----------
        session : ClientSession
            Active MCP client session.
        state : FinancialAgentState
            Mutable agent-local state. This method writes into:
            ``state["calculated_ratios"]``

        Returns
        -------
        None
            All ratio results are written into ``state["calculated_ratios"]``.
            Individual tool failures are logged but do not abort the full run.
        """
        raw        = state["raw_numerical_data"]
        xbrl_src   = raw.get("xbrl_financials", {})
        session_id = session_from_directives(state["manager_directives"])

        # -- Helper: safe MCP tool call ---------------------------------------
        #
        # `period` is a provenance label for whichever reporting period the
        # `args` values actually came from (e.g. "annual:2026",
        # "quarterly:2026-04-26", "ttm", "mrq", or "mixed" for
        # composites/derived values spanning several periods). It is NOT sent
        # to the tool — it's stored on the result so that downstream
        # consumers (report writer, validate_period_consistency.py) can tell
        # whether two ratios narrated side-by-side actually describe the same
        # period. Every ratio in this pipeline currently resolves to either
        # "annual" (sec_edgar.py's XBRL extractor and yfinance's annual
        # income-statement columns are both 10-K/FY only — there is no
        # quarterly income-statement source in this pipeline) or "ttm"/"mrq"
        # (yfinance's live `.info` snapshot fields). Get this wrong and a
        # ratio silently gets narrated as if it described the same quarter as
        # the 10-Q text the report also quotes — see the NVDA net_margin bug.
        async def _call(tool: str, args: dict, period: str | None = None) -> dict:
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: computing '{tool}'...",
                detail={"tool": tool},
            )
            try:
                if sentry_enabled():
                    import sentry_sdk
                    sentry_sdk.add_breadcrumb(
                        category="mcp.financial",
                        message=f"Calling {tool}",
                        data={"tool": tool},
                        level="info",
                    )
                result = await session.call_tool(tool, arguments=args)
                parsed = json.loads(result.content[0].text)
                parsed["_inputs"] = dict(args)
                parsed["_period"] = period
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' computed",
                    detail={"tool": tool, "outcome": "success"},
                )
                return parsed
            except Exception as exc:
                log.warning("Ratio computation tool '%s' failed: %s", tool, exc)
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' failed",
                    detail={"tool": tool, "outcome": "error", "error": str(exc)},
                )
                if sentry_enabled():
                    import sentry_sdk
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("tool", tool)
                        scope.set_tag("component", "mcp.financial")
                        sentry_sdk.capture_exception(exc)
                return {"error": str(exc), "_inputs": dict(args), "_period": period}

        # Phase 6/P1.1: same-period SEC earnings-quality normalization. This
        # remains deterministic and never invents missing tax or operating
        # data, and is unconditional — it is a pure local transform of already
        # extracted XBRL, not an MCP tool, so no intent gates it.
        state["calculated_ratios"]["earnings_quality"] = normalize_earnings_quality(xbrl_src)

        _, compute_tools = self._resolve_tools(state)
        log.info("Executor: intent-resolved compute tools = %s", compute_tools)

        d = _derive_compute_facts(raw, state["calculated_ratios"])

        for tool in compute_tools:
            spec = COMPUTE_REGISTRY[tool]
            plan = spec.mapper(d)

            if plan is None:
                # Inputs unavailable — skip the call entirely (no partial or
                # zero-filled arguments) and record the documented placeholder
                # so data_quality still counts it as an unavailable metric.
                log.info("Compute: skipping %s — required inputs missing.", tool)
                state["calculated_ratios"][spec.key] = dict(spec.unavailable)
                continue

            if isinstance(plan, dict):
                # A finished result needing no MCP round-trip (e.g. a ratio
                # Yahoo already published). Stored verbatim, provenance intact.
                state["calculated_ratios"][spec.key] = plan
                continue

            args, period = plan
            log.info("Computing %s (period=%s)", tool, period)
            result = await _call(tool, args, period=period)
            if spec.post is not None:
                spec.post(result, d)
            state["calculated_ratios"][spec.key] = result

        log.info("Executor: ratio computation complete.")

    @traceable(name="financial.checker", run_type="llm")
    async def _check_data_quality(self, state: FinancialAgentState) -> dict[str, Any]:
        """
        CHECKER — Claude-Powered Financial Data Critic.

        Delegates the full audit to Claude (``self._llm``) using
        ``_CHECKER_SYSTEM_PROMPT``, which instructs the model to evaluate
        7 financial data quality criteria and return a structured JSON verdict.

        This replaces the previous rule-based implementation with an LLM critic
        that can detect semantic issues beyond simple null-checks — such as
        valuation implausibility, internal ratio inconsistencies, and whether
        the dataset is genuinely sufficient for Manager Agent decision-making.

        Audit Criteria (evaluated by Claude)
        -------------------------------------
        1. DATA PRESENCE       : Yahoo Finance payload non-empty and error-free.
        2. REVENUE HISTORY     : >= 2 years of annual revenue for valid CAGR.
        3. CORE RATIO COVERAGE : <= 1 of 5 core ratios (pe, roe, net_margin,
                                 de_ratio, cagr) may be null.
        4. VALUATION SANITY    : P/E present, positive, and sector-plausible.
        5. COMPOSITE SCORE     : Weighted health score (0-100) is computable.
        6. INTERNAL CONSISTENCY: Ratios mathematically consistent with raw data.
        7. MANAGER READINESS   : Dataset rich enough for actionable conclusions.

        Hard-coded pre-flight guard
        ---------------------------
        Before calling Claude, a lightweight pre-flight check verifies that
        at least some raw data exists. If both ``raw_numerical_data`` and
        ``calculated_ratios`` are empty (i.e. the Executor produced nothing),
        the method returns immediately with ``is_complete=False`` to avoid
        wasting an API call on an obviously empty state.

        Parameters
        ----------
        state : FinancialAgentState
            Current agent-local state. Read fields:
            - ``state["raw_numerical_data"]``  : Yahoo + SEC EDGAR extractions.
            - ``state["calculated_ratios"]``   : MCP-computed ratio results.
            - ``state["task_query"]``  : For task_query context.
            - ``state["loop_counter"]``        : For audit context logging.

        Returns
        -------
        dict[str, Any]
            Structured verdict with keys:
            - ``"is_complete"`` (bool)       : True if all 7 criteria pass.
            - ``"score"``       (int)        : Claude's data quality score 0-100.
            - ``"passed"``      (list[str])  : Criteria names that passed.
            - ``"failed"``      (list[str])  : Criteria names that failed.
            - ``"issues"``      (list[str])  : Specific problem descriptions.
            - ``"feedback"``    (str)        : Actionable Brain re-plan instructions.
                                               Empty string when is_complete=True.
        """
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]
        loop = state["loop_counter"]
        task_query = state["task_query"]

        log.info("[Checker] Auditing iteration %d data with Claude...", loop)

        # -- Pre-flight guard: skip Claude call if state is obviously empty ---
        if not raw and not calc:
            log.warning("[Checker] Pre-flight: both raw_numerical_data and "
                        "calculated_ratios are empty — skipping Claude call.")
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["DATA PRESENCE", "REVENUE HISTORY",
                                "CORE RATIO COVERAGE", "VALUATION SANITY",
                                "COMPOSITE SCORE", "INTERNAL CONSISTENCY",
                                "MANAGER READINESS"],
                "issues":      ["No data was extracted. The Executor produced "
                                "empty raw_numerical_data and calculated_ratios."],
                "feedback":    (
                    "No data was retrieved in this iteration. Re-call "
                    "tool_get_financial_ratios, tool_get_revenue_growth, and "
                    "tool_get_xbrl_financials with the correct ticker symbol."
                ),
            }

        # -- Assemble the audit payload for Claude ----------------------------
        yahoo   = raw.get("yahoo_ratios", {})
        growth  = raw.get("revenue_growth", {})
        xbrl    = raw.get("xbrl_financials", {})

        raw_summary = {
            "ticker":            raw.get("ticker"),
            "yahoo_error":       yahoo.get("error"),
            "current_price":     yahoo.get("current_price"),
            "eps_trailing":      yahoo.get("eps_trailing"),
            "pe_ratio_yahoo":    yahoo.get("pe_ratio"),
            "market_cap":        yahoo.get("market_cap"),
            "sector":            yahoo.get("sector"),
            "annual_revenue_count": len(growth.get("annual_revenue", [])),
            "annual_revenue_sample": growth.get("annual_revenue", [])[:3],
            "revenue_growth_ttm":   growth.get("revenue_growth_ttm"),
            "xbrl_assets_available":      bool(xbrl.get("total_assets")),
            "xbrl_liabilities_available": bool(xbrl.get("total_liabilities")),
            "extraction_errors": raw.get("extraction_errors", []),
        }

        ratio_summary = {
            "pe":             calc.get("pe", {}),
            "roe":            calc.get("roe", {}),
            "net_margin":     calc.get("net_margin", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
            "de_ratio":       calc.get("de_ratio", {}),
            "cagr":           calc.get("cagr", {}),
            "composite_score": calc.get("composite_score", {}),
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "valuation_input_assessments": (
                calc.get("dcf", {}).get("base", {}).get("input_assessments", {})
            ),
            "reverse_dcf": calc.get("reverse_dcf", {}),
        }

        deterministic_validation = evaluate_financial_sanity({
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "reverse_dcf": calc.get("reverse_dcf", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
        })
        if not deterministic_validation["passed"]:
            issues = [item["message"] for item in deterministic_validation["critical_failures"]]
            return {
                "is_complete": False,
                "score": deterministic_validation["score"],
                "passed": [],
                "failed": [item["rule"] for item in deterministic_validation["critical_failures"]],
                "issues": issues,
                "feedback": "Correct the deterministic financial-model failures before rerunning valuation.",
                "deterministic_validation": deterministic_validation,
            }

        user_content = (
            f"TASK QUERY:\n{task_query}\n\n"
            f"LOOP ITERATION: {loop}\n\n"
            f"RAW DATA SUMMARY:\n{json.dumps(raw_summary, indent=2)}\n\n"
            f"CALCULATED RATIOS:\n{json.dumps(ratio_summary, indent=2)}\n\n"
            f"DETERMINISTIC FINANCIAL VALIDATION:\n{json.dumps(deterministic_validation, indent=2)}\n\n"
            "Audit the above and return your JSON verdict."
        )

        # -- Call Claude as the Financial Critic ------------------------------
        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=768,
                system=CHECKER_SYSTEM_PROMPT,
                temperature=0.0,
                messages=[{"role": "user", "content": user_content}],
            )
            verdict_text = response.content[0].text.strip()

            # Strip markdown fences if Claude wraps output
            verdict_text = verdict_text.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(verdict_text)

        except json.JSONDecodeError as exc:
            log.warning("[Checker] Claude returned non-JSON verdict: %s | raw=%s",
                        exc, verdict_text[:300])
            # Treat JSON parse failure as incomplete — loop will retry
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["CHECKER PARSE ERROR"],
                "issues":      [f"Claude Checker returned malformed JSON: {exc}"],
                "feedback":    "Checker could not parse its own verdict. "
                               "Re-run full extraction with tool_get_financial_ratios, "
                               "tool_get_revenue_growth, and tool_get_xbrl_financials.",
            }

        except Exception as exc:
            log.exception("[Checker] Claude API call failed: %s", exc)
            # Treat API failure as incomplete — loop will retry with Brain replanning
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["CHECKER API ERROR"],
                "issues":      [f"Claude Checker API call failed: {exc}"],
                "feedback":    "Checker API call failed. Retry full data extraction "
                               "on the next iteration.",
            }

        # -- Parse and log the verdict ----------------------------------------
        is_complete = bool(verdict.get("is_complete", False))
        score       = int(verdict.get("score", 0))
        passed      = verdict.get("passed", [])
        failed      = verdict.get("failed", [])
        issues      = verdict.get("issues", [])
        feedback    = str(verdict.get("feedback", "")) if not is_complete else ""

        if is_complete:
            log.info(
                "[Checker] PASS — score=%d | criteria passed=%d | iteration=%d",
                score, len(passed), loop,
            )
        else:
            log.warning(
                "[Checker] FAIL — score=%d | failed=%s | issues=%s",
                score, failed, issues,
            )

        _publish_progress(
            session_from_directives(state["manager_directives"]), "agent_checker", agent="financial",
            message=(
                f"Financial Agent: checking data quality — "
                f"{'sufficient ✓' if is_complete else 'needs more data'} (score={score})"
            ),
            detail={"is_complete": is_complete, "score": score, "failed": failed},
        )

        return {
            "is_complete": is_complete,
            "score":       score,
            "passed":      passed,
            "failed":      failed,
            "issues":      issues,
            "feedback":    feedback,
            "deterministic_validation": deterministic_validation,
        }

    # =========================================================================
    # LAYER 3 — BRAIN / ORCHESTRATOR (Lifecycle & Loop Gateway)
    # =========================================================================

    @traceable(name="financial.brain", run_type="llm")
    async def _brain(self, state: FinancialAgentState) -> dict[str, Any]:
        """
        BRAIN — Intent Classifier for the Internal Execution Loop.

        Consults Claude to classify the user's query into one or more analysis
        INTENTS, digesting any incoming ``validation_feedback`` from the
        Checker. The intents (not raw tool names, and never numbers) are what
        drive tool selection: ``_resolve_tools()`` expands them into curated
        extraction/compute bundles for the Executor nodes.

        This replaced a freeform ``priority_tools`` list, which let the model
        name tools this pipeline cannot supply arguments for. The only escape
        hatch left is ``extra_tools``, allow-listed against the real MCP
        registry and silently dropped when unrecognised.

        Parameters
        ----------
        state : FinancialAgentState
            Current mutable agent-local state. Read fields:
            - ``state["messages"]``            : Full conversation history.
            - ``state["loop_counter"]``        : Current iteration index.
            - ``state["validation_feedback"]`` : Last Checker critique (may be "").
            - ``state["task_query"]`` / ``state["manager_directives"]``.

        Returns
        -------
        dict[str, Any]
            - ``"intents"``     (list[str]) : Subset of the INTENTS enum; never
                                              empty (defaults to ``["overview"]``).
            - ``"extra_tools"`` (list[str]) : Validated exact MCP tool names.
            - ``"rationale"``   (str)       : One-sentence justification.
            - ``"raw_plan"``    (str)       : Full model response text.
        """
        iteration   = state["loop_counter"]
        feedback    = state["validation_feedback"]
        task_query  = state["task_query"]
        directives  = state["manager_directives"]

        system_prompt = (
            "You are the Brain of a FinancialAnalystAgent. You do NOT choose numbers, "
            "formulas, or tool arguments — code does all of that deterministically. "
            "Your ONLY job is to classify the analyst's query into analysis intents.\n\n"
            "Respond with valid JSON and exactly three keys:\n"
            '  "intents"     — a list drawn ONLY from this closed set: '
            "valuation, financial_health, profitability, growth, liquidity, "
            "cash_flow, peer_relative, overview, full_report\n"
            '  "rationale"   — one sentence explaining the choice\n'
            '  "extra_tools" — usually [] ; only exact MCP tool names the query '
            "explicitly demands beyond the intent bundles\n\n"
            "Intent guidance:\n"
            "- valuation: fair value, intrinsic value, over/undervalued, price target, DCF.\n"
            "- financial_health: solvency, leverage, balance-sheet strength, bankruptcy risk.\n"
            "- profitability: margins, returns on equity/assets/capital.\n"
            "- growth: revenue/earnings growth trajectory, CAGR, forecasts.\n"
            "- liquidity: short-term obligations, current/quick ratio, working capital.\n"
            "- cash_flow: free cash flow, cash conversion, FCF yield.\n"
            "- peer_relative: comparison against competitors or sector peers.\n"
            "- overview: generic, ambiguous, or 'tell me about X' queries.\n"
            "- full_report: the query asks for a COMPREHENSIVE write-up — an "
            "investment analysis report, a full research note, or a complete "
            "assessment of a company (e.g. 'INVESTMENT ANALYSIS REPORT: <COMPANY>'). "
            "Return intents = [\"full_report\"], optionally plus any narrower "
            "intent the query explicitly emphasises.\n\n"
            "Pick multiple intents when the query genuinely spans several. "
            "Do not include any explanation outside the JSON object."
        )

        user_message = (
            f"ITERATION: {iteration}\n"
            f"TASK QUERY: {task_query}\n"
            f"MANAGER DIRECTIVES: {json.dumps(directives)}\n"
            f"CHECKER FEEDBACK: {feedback if feedback else 'None — first iteration or prior run passed.'}\n\n"
            "Classify the analysis intents for this iteration."
        )

        # Append user message to the running conversation log
        state["messages"].append({"role": "user", "content": user_message})

        log.info("Brain: invoking Claude for iteration %d intent classification...", iteration)
        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                system=system_prompt,
                messages=state["messages"],
            )
            raw_plan = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw_plan})

            clean = raw_plan.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            result = self._validate_brain_output(parsed)
            result["raw_plan"] = raw_plan
            log.info(
                "Brain intents for iteration %d: %s (%s)",
                iteration, result["intents"], result["rationale"],
            )
            return result

        except json.JSONDecodeError:
            # Unparseable classification is not worth guessing at — fall back
            # to the overview bundle, which is designed to yield a coherent
            # minimal report on its own.
            log.warning("Brain: Claude response was not valid JSON — defaulting to overview.")
            return {
                "intents": ["overview"], "extra_tools": [],
                "rationale": "Brain response was not valid JSON; defaulted to overview.",
                "raw_plan": raw_plan,
            }

        except Exception as exc:
            log.exception("Brain: Claude API call failed — defaulting to overview.")
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "financial.brain")
                    scope.set_tag("iteration", str(iteration))
                    sentry_sdk.capture_exception(exc)
            default_plan = "Default plan: overview bundle (extraction + core ratios)."
            state["messages"].append({"role": "assistant", "content": default_plan})
            return {
                "intents": ["overview"], "extra_tools": [],
                "rationale": "Brain API call failed; defaulted to overview.",
                "raw_plan": default_plan,
            }

    @staticmethod
    def _validate_brain_output(parsed: dict[str, Any]) -> dict[str, Any]:
        """
        Filter the Brain's raw JSON down to what this agent will actually act
        on. Unrecognised intents and unknown tool names are DROPPED, not
        errored on — a mis-classification should degrade to a thinner report,
        never crash the run or reach the MCP server as a bad tool name.
        """
        raw_intents = parsed.get("intents") or []
        if not isinstance(raw_intents, list):
            raw_intents = []
        intents = [i for i in raw_intents if isinstance(i, str) and i in INTENTS]

        raw_extras = parsed.get("extra_tools") or []
        if not isinstance(raw_extras, list):
            raw_extras = []
        known_tools = set(EXTRACTION_REGISTRY) | set(COMPUTE_REGISTRY)
        extra_tools = [t for t in raw_extras if isinstance(t, str) and t in known_tools]

        if not intents:
            intents = ["overview"]

        return {
            "intents": intents,
            "extra_tools": extra_tools,
            "rationale": str(parsed.get("rationale", "")),
        }

    # =========================================================================
    # INTENT → TOOL RESOLUTION
    # =========================================================================

    def _resolve_tools(
        self, state: FinancialAgentState,
    ) -> tuple[list[str], list[str]]:
        """
        Expand the Brain's intents into the concrete tool bundles to execute.

        Returns ``(extraction_tools, compute_tools)``, both deduped and in a
        deterministic order (_EXTRACTION_ORDER / _COMPUTE_ORDER) so the DCF
        chain's dependencies resolve and traces stay comparable across runs.

        Resolution
        ----------
        1. ``full_report`` expands to the UNION of _FULL_REPORT_PROFILES, so it
           can never drift out of sync with the narrow profiles.
        2. baseline ∪ every chosen profile ∪ allow-listed ``extra_tools``.
        3. Depth gate: on ``shallow`` the Monte-Carlo variants are dropped
           (dcf_scenarios + reverse_dcf survive); on ``deep`` they are kept.
           ``full_report`` implies deep unless the request said shallow.

        Note that ``tool_get_peer_comparison`` is NOT gated here: its peer list
        depends on the industry, which is only known after
        tool_get_financial_ratios returns. That gate lives in
        _extract_peer_args, which skips the call when no peers are mapped.
        """
        chosen = list(state.get("intents") or []) or ["overview"]

        if "full_report" in chosen:
            chosen = [i for i in chosen if i != "full_report"] + list(_FULL_REPORT_PROFILES)

        extraction: set[str] = set(BASELINE_EXTRACTION)
        compute:    set[str] = set(BASELINE_COMPUTE)
        for intent in chosen:
            profile = ANALYSIS_PROFILES.get(intent)
            if profile is None:
                continue
            extraction.update(profile["extraction"])
            compute.update(profile["compute"])

        for tool in state.get("extra_tools") or []:
            if tool in EXTRACTION_REGISTRY:
                extraction.add(tool)
            elif tool in COMPUTE_REGISTRY:
                compute.add(tool)

        depth = _normalize_depth(state["manager_directives"])
        if "full_report" in (state.get("intents") or []) and depth != "shallow":
            depth = "deep"
        if depth == "shallow":
            compute -= _MONTE_CARLO_TOOLS

        # TODO: the chained SEC tools (tool_get_cik → tool_list_filings →
        # tool_get_filing_text) stay out of every bundle. tool_get_filing_text
        # needs accession_number + cik, which are not derivable from a ticker
        # in one step, and this agent has no place to put unstructured filing
        # prose. Wiring them needs a real multi-step extraction plan, not a
        # fabricated argument.
        return (
            [t for t in _EXTRACTION_ORDER if t in extraction],
            [t for t in _COMPUTE_ORDER if t in compute],
        )

    # =========================================================================
    # LANGGRAPH NODES — thin wrappers over the layer methods above
    # =========================================================================
    # Each node returns a partial-state dict (LangGraph merges it). The heavy
    # logic stays in _brain / _execute_data_extraction / _execute_ratio_
    # computation / _check_data_quality — unchanged. The MCP session is read
    # from self._session, set once per run() before the graph is invoked.

    async def _node_brain(self, state: FinancialAgentState) -> dict[str, Any]:
        # loop_counter is incremented here, once per cycle, exactly as the
        # old while-loop did at the top of each iteration.
        new_counter = state["loop_counter"] + 1
        log.info("--- Loop iteration %d ---", new_counter)
        brain_output = await self._brain(state)
        log.info(
            "Brain intents: %s — %s",
            brain_output["intents"], brain_output["rationale"],
        )
        # This node used to return ONLY loop_counter, silently discarding the
        # plan — so the Brain could never influence the Executors and tool
        # selection was effectively hardcoded. Persist the classification so
        # _resolve_tools() (called by both Executors) can act on it.
        return {
            "loop_counter": new_counter,
            "intents":      brain_output["intents"],
            "extra_tools":  brain_output["extra_tools"],
            "rationale":    brain_output["rationale"],
        }

    async def _node_extract(self, state: FinancialAgentState) -> dict[str, Any]:
        await self._execute_data_extraction(self._session, state)
        # _execute_data_extraction mutates state["raw_numerical_data"] in place;
        # return it so the graph state reflects the update explicitly.
        return {"raw_numerical_data": state["raw_numerical_data"]}

    async def _node_compute(self, state: FinancialAgentState) -> dict[str, Any]:
        await self._execute_ratio_computation(self._session, state)
        return {"calculated_ratios": state["calculated_ratios"]}

    async def _node_checker(self, state: FinancialAgentState) -> dict[str, Any]:
        check_result = await self._check_data_quality(state)
        is_complete = check_result["is_complete"]
        if is_complete:
            log.info("[Checker] PASSED (score=%s) after %d iteration(s).",
                     check_result.get("score"), state["loop_counter"])
        else:
            log.warning("[Checker] FAILED iteration %d (score=%s). Feedback: %s",
                        state["loop_counter"], check_result.get("score"),
                        str(check_result.get("feedback", ""))[:120])
        return {
            "is_complete":         is_complete,
            "validation_feedback": check_result["feedback"],
        }

    def _should_continue(self, state: FinancialAgentState):
        """
        Conditional edge after the Checker. Mirrors the old while-loop exit
        conditions exactly:
          - loop_counter >= max_loops  → END (safety guardrail)
          - is_complete == True        → END (normal completion)
          - otherwise                  → back to brain (another cycle)
        """
        directives = state["manager_directives"]
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        if state["loop_counter"] >= max_loops:
            log.warning("[Router] loop_counter=%d >= max_loops=%d → END.",
                        state["loop_counter"], max_loops)
            return "__end__"
        if state["is_complete"]:
            return "__end__"
        return "brain"

    def _build_graph(self):
        """
        Compile the internal state machine:
            START → brain → extract → compute → checker
                  → (conditional) → brain | END
        """
        builder = StateGraph(FinancialAgentState)

        builder.add_node("brain",    self._node_brain)
        builder.add_node("extract",  self._node_extract)
        builder.add_node("compute",  self._node_compute)
        builder.add_node("checker",  self._node_checker)

        builder.set_entry_point("brain")
        builder.add_edge("brain",   "extract")
        builder.add_edge("extract", "compute")
        builder.add_edge("compute", "checker")
        builder.add_conditional_edges(
            "checker",
            self._should_continue,
            {"brain": "brain", "__end__": END},
        )

        return builder.compile()

    # =========================================================================
    # ENTRY GATEWAY — run()
    # =========================================================================

    @traceable(name="FinancialAnalystAgent.run", run_type="chain")
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        """
        PRIMARY ENTRY GATEWAY — Drives the Full Lifecycle Loop.

        Called by the Manager Agent to initiate a financial analysis task.
        Manages the complete MCP client session lifecycle, spawns the
        agent-local state, drives the three-tiered internal loop, enforces
        the loop safety guardrail, and commits verified results into the
        shared state before returning.

        Lifecycle
        ---------
        1. Open an async MCP client session via ``stdio_client``.
        2. Initialise ``FinancialAgentState`` from ``shared_state``.
        3. Enter the loop:
            a. Increment ``loop_counter``; enforce ``max_loops`` guardrail.
            b. Call ``_brain()`` to plan the current iteration.
            c. Call ``_execute_data_extraction()`` to pull raw market data.
            d. Call ``_execute_ratio_computation()`` to compute ratios.
            e. Call ``_check_data_quality()`` to audit results.
            f. If ``is_complete`` → break.  Else → continue with new feedback.
        4. Assemble ``financial_metrics_summary`` from verified ``calculated_ratios``
           and ``raw_numerical_data``, and write it into ``shared_state``.
        5. Return the mutated ``shared_state`` to the Manager Agent.

        Parameters
        ----------
        shared_state : SharedManagerState
            The public state contract received from the Manager Agent.
            Expected keys: ``task_query``, ``manager_directives`` (optional).

        Returns
        -------
        SharedManagerState
            The same dict mutated in-place to include:
            ``shared_state["financial_metrics_summary"]`` — a dict containing
            all verified ratios, scores, ticker info, and loop metadata.

        Raises
        ------
        RuntimeError
            If the MCP server subprocess cannot be started.
        """
        directives = shared_state.get("manager_directives", {})
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        log.info(
            "FinancialAnalystAgent.run() started — task_query='%s', max_loops=%d",
            shared_state.get("task_query", ""),
            max_loops,
        )

        # -- Initialise agent-local state -------------------------------------
        state: FinancialAgentState = {
            "messages":            [],
            "raw_numerical_data":  {},
            "calculated_ratios":   {},
            "loop_counter":        0,
            "validation_feedback": "",
            "is_complete":         False,
            "task_query":          shared_state["task_query"],
            "manager_directives":  dict(shared_state.get("manager_directives", {})),
            # Populated by _node_brain each iteration; empty here means
            # "unclassified", which _resolve_tools() reads as ["overview"].
            "intents":             [],
            "extra_tools":         [],
            "rationale":           "",
        }

        # -- Open MCP client session ------------------------------------------
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                # Handshake: initialise the MCP protocol connection
                await session.initialize()
                log.info("MCP session initialised successfully.")

                # Expose the live session to graph nodes (see __init__).
                self._session = session

                # -- Drive the compiled graph ---------------------------------
                # Topology: brain → extract → compute → checker →
                #           (conditional) → brain | END
                # recursion_limit mirrors the old while-loop guardrail: each
                # loop is 4 nodes, +1 slack, same effective cap as max_loops.
                try:
                    state = await self._graph.ainvoke(
                        state,
                        config={"recursion_limit": (max_loops + 1) * 4},
                    )
                finally:
                    self._session = None

                if not state["is_complete"]:
                    log.warning(
                        "Safety guardrail hit: exiting after %d loops without full validation.",
                        state["loop_counter"],
                    )

        # -- Assemble financial_metrics_summary --------------------------------
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]

        # Deterministic, rule-based analysis layer -- NO LLM. See module
        # docstring (_annotate_confidence / _build_consistency_checks /
        # _build_data_quality_score) for why this stays rule-based: any
        # interpretive judgment ("what does this mean for the business")
        # belongs to the Manager Agent's finalizer LLM, which is the ONLY
        # place in this pipeline meant to narrate numbers -- keeping this
        # agent's own layer mechanical is what makes its output safely
        # re-usable/re-auditable rather than one more place a number could
        # get silently reinterpreted.
        annotate_confidence(calc)
        consistency_checks = build_consistency_checks(raw, calc)
        data_quality = build_data_quality_score(raw, calc)
        financial_sanity = evaluate_financial_sanity({
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "reverse_dcf": calc.get("reverse_dcf", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
        })
        financial_facts = []
        for series_name, entries in raw.get("xbrl_financials", {}).items():
            if not isinstance(entries, list):
                continue
            metric = series_name.removesuffix("_annual")
            for entry in entries:
                if not isinstance(entry, dict) or "value" not in entry:
                    continue
                financial_facts.append(typed_financial_claim({
                    "metric": metric, "value": entry.get("value"), "unit": entry.get("unit"),
                    "ticker": raw.get("ticker"), "fiscal_year": entry.get("fiscal_year"),
                    "fiscal_quarter": entry.get("fiscal_quarter"),
                    "period_start": entry.get("period_start"), "period_end": entry.get("period_end"),
                    "duration_months": entry.get("duration_months"),
                    "filing_type": entry.get("filing_type"),
                    "accession_number": entry.get("accession_number"),
                    "metric_scope": entry.get("metric_scope"),
                    "gaap_status": entry.get("gaap_status"),
                    "provenance_status": "COMPLETE" if entry.get("accession_number") else "INCOMPLETE",
                    # Distinct historical facts are retained, not merged into one claim.
                    "comparison": True,
                }))

        financial_metrics_summary: dict[str, Any] = {
            # Identification
            "ticker":        raw.get("ticker"),
            "company_name":  raw.get("yahoo_ratios", {}).get("company_name"),
            "sector":        raw.get("yahoo_ratios", {}).get("sector"),
            "industry":      raw.get("yahoo_ratios", {}).get("industry"),

            # Market data
            "current_price": raw.get("yahoo_ratios", {}).get("current_price"),
            "market_cap":    raw.get("yahoo_ratios", {}).get("market_cap"),
            "beta":          raw.get("yahoo_ratios", {}).get("beta"),
            "52w_high":      raw.get("yahoo_ratios", {}).get("52w_high"),
            "52w_low":       raw.get("yahoo_ratios", {}).get("52w_low"),

            # Calculated ratios (verified)
            "pe_ratio":      calc.get("pe", {}),
            "forward_pe":    raw.get("yahoo_ratios", {}).get("forward_pe"),
            "peg_ratio":     raw.get("yahoo_ratios", {}).get("peg_ratio"),
            "roe":           calc.get("roe", {}),
            "net_margin":    calc.get("net_margin", {}),
            "de_ratio":      calc.get("de_ratio", {}),
            "revenue_cagr":  calc.get("cagr", {}),
            "forecast":      calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "valuation_input_assessments": (
                calc.get("dcf", {}).get("base", {}).get("input_assessments", {})
            ),
            "dcf_monte_carlo": calc.get("dcf_monte_carlo", {}),
            "current_ratio": raw.get("yahoo_ratios", {}).get("current_ratio"),
            "composite_score": calc.get("composite_score", {}),

            # Phase 1/2 ratios (net_debt_to_ebitda, fcf_margin, fcf_yield,
            # ocf_ratio, reverse_dcf, roic, roce, debt_ratio, equity_ratio,
            # altman_z_score) -- same pass-through pattern as roe/net_margin/
            # de_ratio above: the full dict (value + interpretation +
            # formula + _inputs + _period), not just the bare number, so
            # manager_agent.py can extract period tags the same way it does
            # for every other ratio.
            "net_debt_to_ebitda": calc.get("net_debt_to_ebitda", {}),
            "fcf_margin":         calc.get("fcf_margin", {}),
            "fcf_yield":          calc.get("fcf_yield", {}),
            "ocf_ratio":          calc.get("ocf_ratio", {}),
            "reverse_dcf":        calc.get("reverse_dcf", {}),
            "roic":               calc.get("roic", {}),
            "roce":               calc.get("roce", {}),
            "debt_ratio":         calc.get("debt_ratio", {}),
            "equity_ratio":       calc.get("equity_ratio", {}),
            "altman_z_score":     calc.get("altman_z_score", {}),

            # Relative valuation vs. sector peers
            "peer_comparison": raw.get("peer_comparison", {}),

            # Raw revenue history (for downstream charting / narration)
            "annual_revenue_history": raw.get("revenue_growth", {}).get("annual_revenue", []),
            "revenue_growth_ttm":     raw.get("revenue_growth", {}).get("revenue_growth_ttm"),

            # SEC EDGAR
            "xbrl_total_assets":      (
                raw.get("xbrl_financials", {}).get("total_assets", [{}])[:1]
            ),
            "xbrl_total_liabilities": (
                raw.get("xbrl_financials", {}).get("total_liabilities", [{}])[:1]
            ),

            # Capital allocation (most recent fiscal year, from XBRL —
            # see sec_edgar.py's get_xbrl_financials). Empty lists mean
            # "not reported this year" (e.g. no dividends paid), not a
            # fetch failure — check raw_numerical_data.extraction_errors
            # for actual failures.
            "capital_allocation": {
                "operating_cash_flow": (
                    raw.get("xbrl_financials", {}).get("operating_cash_flow_annual", [{}])[:1]
                ),
                "capex": (
                    raw.get("xbrl_financials", {}).get("capex_annual", [{}])[:1]
                ),
                "dividends_paid": (
                    raw.get("xbrl_financials", {}).get("dividends_paid_annual", [{}])[:1]
                ),
                "buybacks": (
                    raw.get("xbrl_financials", {}).get("buybacks_annual", [{}])[:1]
                ),
            },

            # Deterministic analysis layer (rule-based, no LLM -- see note above)
            "consistency_checks": consistency_checks,
            "data_quality":       data_quality,
            "financial_sanity":   financial_sanity,
            "financial_facts":    financial_facts,

            # Execution metadata
            "loop_iterations_used":  state["loop_counter"],
            "validation_passed":     state["is_complete"],
            "extraction_errors":     raw.get("extraction_errors", []),
        }

        # -- Final safety net: strip any NaN/Inf that slipped through ---------
        # This is deliberately the LAST step before the summary leaves the
        # agent. Root cause: pandas represents a missing cell in an
        # otherwise-present row as `numpy.nan`, which `float()` happily
        # converts to a real (but JSON-illegal) `nan` — and that value can
        # pass ordinary `is not None` checks anywhere upstream. A single
        # surviving NaN anywhere in this dict crashes downstream JSON
        # persistence to Supabase with "Out of range float values are not
        # JSON compliant: nan". Rather than rely on every upstream call site
        # remembering to guard against NaN individually (yahoo_finance.py's
        # _safe_float() and this module's CAGR filter both do, but a future
        # data source might not), recursively sanitize the whole summary
        # once, here, so this class of crash cannot recur regardless of
        # where a NaN originates.
        financial_metrics_summary = sanitize_nans(financial_metrics_summary)

        # Commit to the shared state contract
        shared_state["financial_metrics_summary"] = financial_metrics_summary  # type: ignore[literal-required]

        log.info(
            "FinancialAnalystAgent.run() complete — ticker=%s, score=%s, grade=%s",
            financial_metrics_summary["ticker"],
            calc.get("composite_score", {}).get("score"),
            calc.get("composite_score", {}).get("grade"),
        )

        return shared_state
