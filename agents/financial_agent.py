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
import re
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import END, StateGraph
import os
from agents.prompts.promt_finance_checker import CHECKER_SYSTEM_PROMPT
from evaluation.financial_sanity import evaluate_financial_sanity, build_valuation_input_assessments
from evaluation.earnings_quality import normalize_earnings_quality
from evaluation.claim_contract import typed_financial_claim
from agents.executor_helpers.finance_executor_helper import (infer_peers, extract_ticker, confidence_for, annotate_confidence, pct_change,
                                                      direction, build_consistency_checks, build_data_quality_score, sanitize_nans)

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

        MCP Tools Called
        ----------------
        - ``tool_get_financial_ratios``  → P/E, P/B, EPS, market cap, beta, etc.
        - ``tool_get_revenue_growth``    → Annual/quarterly revenue & net income trends.
        - ``tool_get_xbrl_financials``   → SEC EDGAR XBRL structured balance-sheet data.
        - ``tool_get_balance_sheet_snapshot`` → EBIT, current assets/liabilities,
          retained earnings (yfinance) -- feeds ROIC/ROCE/debt_ratio/
          equity_ratio/altman_z_score in _execute_ratio_computation.

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
        errors: list[str] = []
        session_id = session_from_directives(state["manager_directives"])
        _publish_progress(
            session_id, "agent_brain", agent="financial",
            message=f"Financial Agent: starting raw data extraction for {ticker}",
            detail={"ticker": ticker},
        )

        # -- Yahoo Finance: key ratios ----------------------------------------
        log.info("Executor: calling tool_get_financial_ratios for %s", ticker)
        _publish_progress(
            session_id, "agent_tool_call", agent="financial",
            message="Financial Agent: calling tool 'tool_get_financial_ratios'...",
            detail={"tool": "tool_get_financial_ratios"},
        )
        try:
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.add_breadcrumb(
                    category="mcp.financial",
                    message="Calling tool_get_financial_ratios",
                    data={"tool": "tool_get_financial_ratios", "ticker": ticker},
                    level="info",
                )
            ratios_result = await session.call_tool(
                "tool_get_financial_ratios",
                arguments={"ticker": ticker},
            )
            ratios_data = json.loads(ratios_result.content[0].text)
            if ratios_data.get("error"):
                errors.append(f"tool_get_financial_ratios error: {ratios_data['error']}")
            state["raw_numerical_data"]["yahoo_ratios"] = ratios_data
        except Exception as exc:
            errors.append(f"tool_get_financial_ratios exception: {exc}")
            state["raw_numerical_data"]["yahoo_ratios"] = {}
            log.exception("Executor: tool_get_financial_ratios failed")
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("tool", "tool_get_financial_ratios")
                    scope.set_tag("component", "mcp.financial")
                    sentry_sdk.capture_exception(exc)

        # -- Yahoo Finance: peer comparison (relative valuation) --------------
        # Deterministic step (not LLM-planned) — peers come from the static
        # _INDUSTRY_PEERS map keyed off the industry/sector just fetched
        # above, since get_peer_comparison's own auto-inference is broken
        # (see _infer_peers docstring). Never blocks the pipeline: if the
        # industry isn't in the map, peers=[] and this call is skipped.
        industry = state["raw_numerical_data"].get("yahoo_ratios", {}).get("industry", "")
        sector   = state["raw_numerical_data"].get("yahoo_ratios", {}).get("sector", "")
        peers    = infer_peers(sector, industry, ticker)

        if peers:
            log.info("Executor: calling tool_get_peer_comparison for %s vs %s", ticker, peers)
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message="Financial Agent: calling tool 'tool_get_peer_comparison'...",
                detail={"tool": "tool_get_peer_comparison", "peers": peers},
            )
            try:
                peer_result = await session.call_tool(
                    "tool_get_peer_comparison",
                    arguments={"ticker": ticker, "peers": peers},
                )
                peer_data = json.loads(peer_result.content[0].text)
                if peer_data.get("error"):
                    errors.append(f"tool_get_peer_comparison error: {peer_data['error']}")
                state["raw_numerical_data"]["peer_comparison"] = peer_data
            except Exception as exc:
                errors.append(f"tool_get_peer_comparison exception: {exc}")
                state["raw_numerical_data"]["peer_comparison"] = {}
                log.exception("Executor: tool_get_peer_comparison failed")
                if sentry_enabled():
                    import sentry_sdk
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("tool", "tool_get_peer_comparison")
                        scope.set_tag("component", "mcp.financial")
                        sentry_sdk.capture_exception(exc)
        else:
            log.info(
                "Executor: skipping tool_get_peer_comparison — no static "
                "peers mapped for industry=%r (ticker=%s).", industry, ticker,
            )
            state["raw_numerical_data"]["peer_comparison"] = {
                "note": f"No peer mapping available for industry={industry!r}.",
            }

        # -- Yahoo Finance: revenue growth ------------------------------------
        log.info("Executor: calling tool_get_revenue_growth for %s", ticker)
        _publish_progress(
            session_id, "agent_tool_result", agent="financial",
            message="Financial Agent: 'tool_get_financial_ratios' completed",
            detail={"tool": "tool_get_financial_ratios"},
        )
        _publish_progress(
            session_id, "agent_tool_call", agent="financial",
            message="Financial Agent: calling tool 'tool_get_revenue_growth'...",
            detail={"tool": "tool_get_revenue_growth"},
        )
        try:
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.add_breadcrumb(
                    category="mcp.financial",
                    message="Calling tool_get_revenue_growth",
                    data={"tool": "tool_get_revenue_growth", "ticker": ticker},
                    level="info",
                )
            growth_result = await session.call_tool(
                "tool_get_revenue_growth",
                arguments={"ticker": ticker},
            )
            growth_data = json.loads(growth_result.content[0].text)
            if growth_data.get("error"):
                errors.append(f"tool_get_revenue_growth error: {growth_data['error']}")
            state["raw_numerical_data"]["revenue_growth"] = growth_data
        except Exception as exc:
            errors.append(f"tool_get_revenue_growth exception: {exc}")
            state["raw_numerical_data"]["revenue_growth"] = {}
            log.exception("Executor: tool_get_revenue_growth failed")
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("tool", "tool_get_revenue_growth")
                    scope.set_tag("component", "mcp.financial")
                    sentry_sdk.capture_exception(exc)

        # -- SEC EDGAR: XBRL structured financials ----------------------------
        log.info("Executor: calling tool_get_xbrl_financials for %s", ticker)
        _publish_progress(
            session_id, "agent_tool_result", agent="financial",
            message="Financial Agent: 'tool_get_revenue_growth' completed",
            detail={"tool": "tool_get_revenue_growth"},
        )
        _publish_progress(
            session_id, "agent_tool_call", agent="financial",
            message="Financial Agent: calling tool 'tool_get_xbrl_financials'...",
            detail={"tool": "tool_get_xbrl_financials"},
        )
        try:
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.add_breadcrumb(
                    category="mcp.financial",
                    message="Calling tool_get_xbrl_financials",
                    data={"tool": "tool_get_xbrl_financials", "ticker": ticker},
                    level="info",
                )
            xbrl_result = await session.call_tool(
                "tool_get_xbrl_financials",
                arguments={"ticker": ticker},
            )
            xbrl_data = json.loads(xbrl_result.content[0].text)
            if xbrl_data.get("error"):
                errors.append(f"tool_get_xbrl_financials error: {xbrl_data['error']}")
            state["raw_numerical_data"]["xbrl_financials"] = xbrl_data
        except Exception as exc:
            errors.append(f"tool_get_xbrl_financials exception: {exc}")
            state["raw_numerical_data"]["xbrl_financials"] = {}
            log.exception("Executor: tool_get_xbrl_financials failed")
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("tool", "tool_get_xbrl_financials")
                    scope.set_tag("component", "mcp.financial")
                    sentry_sdk.capture_exception(exc)

        # -- Yahoo Finance: balance-sheet snapshot (EBIT, current assets/
        #    liabilities, retained earnings -- not in tool_get_financial_ratios'
        #    flat .info-based output, needed for ROIC/ROCE/debt_ratio/
        #    equity_ratio/altman_z_score in _execute_ratio_computation) -------
        _publish_progress(
            session_id, "agent_tool_result", agent="financial",
            message="Financial Agent: 'tool_get_xbrl_financials' completed",
            detail={"tool": "tool_get_xbrl_financials"},
        )
        log.info("Executor: calling tool_get_balance_sheet_snapshot for %s", ticker)
        _publish_progress(
            session_id, "agent_tool_call", agent="financial",
            message="Financial Agent: calling tool 'tool_get_balance_sheet_snapshot'...",
            detail={"tool": "tool_get_balance_sheet_snapshot"},
        )
        try:
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.add_breadcrumb(
                    category="mcp.financial",
                    message="Calling tool_get_balance_sheet_snapshot",
                    data={"tool": "tool_get_balance_sheet_snapshot", "ticker": ticker},
                    level="info",
                )
            bs_result = await session.call_tool(
                "tool_get_balance_sheet_snapshot",
                arguments={"ticker": ticker},
            )
            bs_data = json.loads(bs_result.content[0].text)
            if bs_data.get("error"):
                errors.append(f"tool_get_balance_sheet_snapshot error: {bs_data['error']}")
            state["raw_numerical_data"]["balance_sheet_snapshot"] = bs_data
        except Exception as exc:
            errors.append(f"tool_get_balance_sheet_snapshot exception: {exc}")
            state["raw_numerical_data"]["balance_sheet_snapshot"] = {}
            log.exception("Executor: tool_get_balance_sheet_snapshot failed")
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("tool", "tool_get_balance_sheet_snapshot")
                    scope.set_tag("component", "mcp.financial")
                    sentry_sdk.capture_exception(exc)

        if errors:
            state["raw_numerical_data"]["extraction_errors"] = errors
        log.info(
            "Executor: data extraction complete — %d error(s) recorded.", len(errors)
        )
        _publish_progress(
            session_id, "agent_tool_result", agent="financial",
            message="Financial Agent: 'tool_get_balance_sheet_snapshot' completed — data extraction done",
            detail={"tool": "tool_get_balance_sheet_snapshot", "errors": len(errors)},
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

        MCP Tools Called
        ----------------
        - ``tool_calc_pe``               → Price-to-Earnings with interpretation.
        - ``tool_calc_roe``              → Return on Equity (%).
        - ``tool_calc_net_margin``       → Net Profit Margin (%).
        - ``tool_calc_debt_to_equity``   → Debt-to-Equity leverage ratio.
        - ``tool_calc_cagr``             → Revenue CAGR over available history.
        - ``tool_calc_composite_score``  → Weighted composite health score (0-100).

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
        ratios_src = raw.get("yahoo_ratios", {})
        growth_src = raw.get("revenue_growth", {})
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
        # remains deterministic and never invents missing tax or operating data.
        state["calculated_ratios"]["earnings_quality"] = normalize_earnings_quality(xbrl_src)

        # -- P/E Ratio --------------------------------------------------------
        price = ratios_src.get("current_price")
        eps   = ratios_src.get("eps_trailing")
        if price is not None and eps is not None and eps != 0:
            log.info("Computing P/E: price=%.2f eps=%.4f", price, eps)
            state["calculated_ratios"]["pe"] = await _call(
                "tool_calc_pe", {"price": price, "eps": eps}, period="ttm",
            )
        else:
            state["calculated_ratios"]["pe"] = {
                "pe_ratio": ratios_src.get("pe_ratio"),
                "interpretation": "sourced_from_yahoo",
                "note": "Used pre-computed Yahoo Finance P/E (price or EPS unavailable for direct calc).",
                "_inputs": {"pe_ratio": ratios_src.get("pe_ratio")},
                "_period": "ttm",
            }

        # -- Return on Equity -------------------------------------------------
        xbrl_assets      = xbrl_src.get("total_assets", [])
        xbrl_liabilities = xbrl_src.get("total_liabilities", [])
        annual_ni        = growth_src.get("annual_net_income", [])

        net_income = annual_ni[0]["net_income"] if annual_ni else None
        net_income_year = annual_ni[0].get("year") if annual_ni else None
        # Approximate equity = assets - liabilities (most recent period)
        # NOTE: xbrl_assets/xbrl_liabilities come from sec_edgar.py's
        # _extract_annual(), which only pulls 10-K/FY entries — so this is
        # an ANNUAL balance-sheet snapshot, not a quarterly one, even though
        # it's paired here with the *current* net_income figure above.
        shareholders_equity: float | None = None
        equity_period_end: str | None = None
        if xbrl_assets and xbrl_liabilities:
            try:
                shareholders_equity = (
                    xbrl_assets[0]["value"] - xbrl_liabilities[0]["value"]
                )
                equity_period_end = xbrl_assets[0].get("period_end")
            except (KeyError, TypeError, IndexError):
                shareholders_equity = None

        # ROE mixes an annual (10-K) net_income with an annual (10-K)
        # balance sheet — same source family, but don't just assume index
        # [0] on both lists lines up to the same fiscal year; check it.
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

        if net_income and shareholders_equity and shareholders_equity != 0:
            log.info("Computing ROE: net_income=%.0f equity=%.0f", net_income, shareholders_equity)
            state["calculated_ratios"]["roe"] = await _call(
                "tool_calc_roe",
                {"net_income": net_income, "shareholders_equity": shareholders_equity},
                period=roe_period,
            )
        else:
            state["calculated_ratios"]["roe"] = {"roe_pct": None, "interpretation": "insufficient_data"}

        # -- Net Margin -------------------------------------------------------
        # REAL FIX (previously worked around by tagging annual as annual):
        # quarterly_net_income now exists (see yahoo_finance.py), mirroring
        # quarterly_revenue. Prefer the most recent matching quarter for
        # both net_income and revenue so net_margin describes the SAME
        # quarter as gross_margin/operating_margin (which the report writer
        # pulls straight from the 10-Q's quarterly MD&A text). Only fall
        # back to annual figures — clearly tagged as such — when quarterly
        # data isn't available (e.g. some tickers/data providers don't
        # expose quarterly_financials).
        quarterly_rev  = growth_src.get("quarterly_revenue", [])
        quarterly_ni   = growth_src.get("quarterly_net_income", [])
        annual_rev     = growth_src.get("annual_revenue", [])
        revenue_year   = annual_rev[0].get("year") if annual_rev else None

        use_quarterly = (
            quarterly_rev and quarterly_ni
            and quarterly_rev[0].get("quarter") == quarterly_ni[0].get("quarter")
            and quarterly_rev[0].get("revenue") is not None
            and quarterly_ni[0].get("net_income") is not None
        )

        if use_quarterly:
            nm_net_income = quarterly_ni[0]["net_income"]
            nm_revenue    = quarterly_rev[0]["revenue"]
            net_margin_period = f"quarterly:{quarterly_rev[0]['quarter']}"
        else:
            # Fallback: most recent ANNUAL (10-K/FY) figures. Still tagged
            # accurately (and flagged MIXED if the two annual arrays'
            # index-[0] years don't actually agree) so a caller narrating
            # this next to quarterly text can catch the period difference
            # instead of silently mismatching them — see the NVDA bug this
            # was built to catch.
            nm_net_income = net_income
            nm_revenue    = annual_rev[0]["revenue"] if annual_rev else None
            net_margin_period = (
                f"MIXED(net_income=annual:{net_income_year}, revenue=annual:{revenue_year})"
                if net_income_year is not None and revenue_year is not None and net_income_year != revenue_year
                else f"annual:{revenue_year}" if revenue_year is not None
                else None
            )

        if nm_net_income is not None and nm_revenue is not None and nm_revenue != 0:
            log.info("Computing Net Margin: net_income=%.0f revenue=%.0f (period=%s, quarterly=%s)",
                      nm_net_income, nm_revenue, net_margin_period, use_quarterly)
            state["calculated_ratios"]["net_margin"] = await _call(
                "tool_calc_net_margin",
                {"net_income": nm_net_income, "revenue": nm_revenue},
                period=net_margin_period,
            )
        else:
            state["calculated_ratios"]["net_margin"] = {"net_margin_pct": None, "interpretation": "insufficient_data"}

        # -- Debt-to-Equity ---------------------------------------------------
        # D/E from Yahoo Finance info if available; otherwise note as unavailable
        de_ratio_yahoo = ratios_src.get("de_ratio") or ratios_src.get("debtToEquity")
        if de_ratio_yahoo is not None:
            # Yahoo already provides D/E; store directly without an MCP round-trip.
            # yfinance's `.info["debtToEquity"]` is a most-recent-quarter (MRQ)
            # balance-sheet snapshot, not a full-quarter income-statement figure
            # and not the annual 10-K figure used elsewhere in this function.
            state["calculated_ratios"]["de_ratio"] = {
                "de_ratio": de_ratio_yahoo,
                "interpretation": "sourced_from_yahoo",
                "_inputs": {"de_ratio_yahoo": de_ratio_yahoo},
                "_period": "mrq",
            }
        elif shareholders_equity and shareholders_equity != 0:
            # Attempt approximation from XBRL (annual 10-K balance sheet —
            # see the shareholders_equity/equity_period_end computation above)
            total_liabilities_val = xbrl_liabilities[0]["value"] if xbrl_liabilities else None
            if total_liabilities_val:
                log.info("Computing D/E from XBRL liabilities/equity")
                de_result = await _call(
                    "tool_calc_debt_to_equity",
                    {
                        "total_debt": total_liabilities_val,
                        "shareholders_equity": shareholders_equity,
                    },
                    period=f"annual:{equity_period_end}" if equity_period_end else None,
                )
                # tool_get_xbrl_financials only exposes total_liabilities,
                # not interest-bearing debt separately — so this ratio
                # includes non-interest-bearing liabilities (accounts
                # payable, deferred revenue, etc.) that a true D/E ratio
                # excludes. The tool's own threshold-based interpretation
                # (e.g. "moderate_leverage") is calibrated for TRUE D/E and
                # is NOT comparable to standard thresholds when computed
                # this way — override it so that's never silently hidden.
                de_result["formula"] = "total_liabilities / shareholders_equity (approximation)"
                de_result["interpretation"] = "approximation_not_comparable_to_standard_thresholds"
                de_result["is_approximation"] = True
                de_result["note"] = (
                    "tool_get_xbrl_financials only exposes total_liabilities, not "
                    "interest-bearing debt separately, so this uses total liabilities "
                    "as a rough stand-in. This is NOT a true debt-to-equity ratio and "
                    "should not be compared against standard D/E leverage thresholds."
                )
                state["calculated_ratios"]["de_ratio"] = de_result
            else:
                state["calculated_ratios"]["de_ratio"] = {"de_ratio": None, "interpretation": "unavailable"}
        else:
            state["calculated_ratios"]["de_ratio"] = {"de_ratio": None, "interpretation": "unavailable"}

        # -- Revenue CAGR (using oldest & newest annual revenue points) --------
        # IMPORTANT: annual_rev may contain entries whose "revenue" is None
        # (e.g. the oldest fiscal year often has no value from yfinance). Using
        # annual_rev[-1] blindly picks that null and voids the whole CAGR, even
        # when 3-4 valid years are present. Filter to valid points first.
        #
        # DEFENSE-IN-DEPTH: this also excludes NaN, not just None. yfinance/
        # pandas represents a missing cell in an otherwise-present row as
        # `numpy.nan`, and `nan is not None` is True — so a None-only check
        # let a NaN "revenue" through as if it were valid data. That NaN then
        # silently failed the `oldest > 0` guard below (NaN fails every
        # ordering comparison) and made every CAGR computation report
        # "unavailable" even with 4 good years of history, and separately
        # broke JSON persistence downstream ("NaN is not JSON compliant").
        # yahoo_finance.py's get_revenue_growth() now converts NaN cells to
        # None at the source (_safe_float()); this check stays as a second
        # line of defense in case any other upstream source ever supplies a
        # raw NaN here.
        def _is_valid_number(x: Any) -> bool:
            return x is not None and not (isinstance(x, float) and math.isnan(x))

        valid_rev = [
            e for e in annual_rev
            if _is_valid_number(e.get("revenue")) and e.get("year") is not None
        ]
        valid_rev.sort(key=lambda e: e["year"])  # oldest -> newest
        if len(valid_rev) >= 2:
            oldest = valid_rev[0]["revenue"]
            newest = valid_rev[-1]["revenue"]
            years  = valid_rev[-1]["year"] - valid_rev[0]["year"]
            if newest and oldest and oldest > 0 and years > 0:
                log.info("Computing Revenue CAGR: start=%.0f end=%.0f years=%d", oldest, newest, years)
                cagr_period = f"annual:{valid_rev[0]['year']}-{valid_rev[-1]['year']}"
                state["calculated_ratios"]["cagr"] = await _call(
                    "tool_calc_cagr",
                    {"start_value": oldest, "end_value": newest, "years": float(years)},
                    period=cagr_period,
                )
            else:
                state["calculated_ratios"]["cagr"] = {"cagr_pct": None, "interpretation": "unavailable"}
        else:
            state["calculated_ratios"]["cagr"] = {"cagr_pct": None, "interpretation": "insufficient_history"}

        # -- DCF Valuation ------------------------------------------------------
        # FCF = operating cash flow − capex (most recent year, matched by
        # taking the newest entry from each series — both come from
        # tool_get_xbrl_financials, sorted newest-first by _extract_annual).
        operating_cf_annual = xbrl_src.get("operating_cash_flow_annual", [])
        capex_annual         = xbrl_src.get("capex_annual", [])
        beta                 = ratios_src.get("beta")
        market_cap           = ratios_src.get("market_cap")
        current_price        = ratios_src.get("current_price")
        observed_diluted_shares = ratios_src.get("diluted_shares")
        shares_outstanding   = observed_diluted_shares or (
            market_cap / current_price
            if market_cap and current_price else None
        )
        total_debt_ttm = ratios_src.get("total_debt")
        total_cash_ttm = ratios_src.get("total_cash")
        lease_liabilities_ttm = ratios_src.get("lease_liabilities")
        minority_interest_ttm = ratios_src.get("minority_interest")
        interest_expense_ttm = ratios_src.get("interest_expense")
        tax_rate_raw = ratios_src.get("effective_tax_rate")
        tax_rate_observed = tax_rate_raw is not None
        effective_tax_rate_pct = (
            float(tax_rate_raw) * 100 if tax_rate_raw is not None and abs(float(tax_rate_raw)) <= 1
            else float(tax_rate_raw) if tax_rate_raw is not None else 21.0
        )
        cost_of_debt_observed = interest_expense_ttm is not None and bool(total_debt_ttm and total_debt_ttm > 0)
        pre_tax_cost_of_debt_pct = (
            abs(float(interest_expense_ttm)) / total_debt_ttm * 100
            if interest_expense_ttm is not None and total_debt_ttm and total_debt_ttm > 0
            else 5.0
        )
        valuation_input_assessments = build_valuation_input_assessments(
            market_cap=market_cap, total_debt=total_debt_ttm, cash=total_cash_ttm,
            marketable_securities=None, lease_liabilities=lease_liabilities_ttm,
            minority_interest=minority_interest_ttm, diluted_shares=shares_outstanding,
            diluted_shares_observed=observed_diluted_shares is not None,
            cost_of_debt_observed=cost_of_debt_observed,
            tax_rate_observed=tax_rate_observed,
            pre_tax_cost_of_debt_pct=pre_tax_cost_of_debt_pct,
            effective_tax_rate_pct=effective_tax_rate_pct,
            sector=ratios_src.get("sector"),
        )

        fcf_base = None
        if operating_cf_annual and capex_annual:
            try:
                # capex is typically reported as a positive outflow figure
                # in XBRL's PaymentsToAcquirePropertyPlantAndEquipment tag
                # (a "payments" concept, not a signed reduction), so it's
                # subtracted here rather than added.
                fcf_base = operating_cf_annual[0]["value"] - capex_annual[0]["value"]
            except (KeyError, TypeError, IndexError):
                fcf_base = None

        revenue_base = valid_rev[-1]["revenue"] if valid_rev else None
        if fcf_base is not None and beta is not None and revenue_base:
            # Ground the base-case FCF growth assumption in the company's
            # own revenue CAGR (already computed above) — bear/bull default
            # to ~0.5x/~1.75x of this inside dcf_scenario_range unless
            # overridden. Using the scenario-range tool (not the single-
            # point one) deliberately: a single DCF number using trailing
            # growth routinely lands far below market price for a
            # high-growth stock, which reads as "the model is broken"
            # unless the growth-rate sensitivity is shown explicitly.
            cagr_result = state["calculated_ratios"].get("cagr", {})
            historical_cagr_pct = cagr_result.get("cagr_pct")
            if historical_cagr_pct is None:
                # No CAGR available to ground the assumption — skip rather
                # than guessing a base growth rate out of thin air.
                state["calculated_ratios"]["dcf"] = {
                    "interpretation": "unavailable",
                    "error": "No revenue_cagr available to anchor the base-case FCF growth assumption.",
                }
            else:
                trajectory_raw = growth_src.get("revenue_growth_ttm")
                if trajectory_raw is None and len(valid_rev) >= 2:
                    trajectory_raw = valid_rev[-1]["revenue"] / valid_rev[-2]["revenue"] - 1.0
                current_growth_pct = (
                    float(trajectory_raw) * 100.0
                    if trajectory_raw is not None and abs(float(trajectory_raw)) <= 2.0
                    else float(trajectory_raw) if trajectory_raw is not None else None
                )
                capex_pct_revenue = capex_annual[0]["value"] / revenue_base * 100.0 if capex_annual else None
                forecast = await _call(
                    "tool_build_forecast",
                    {
                        "revenue_base": revenue_base,
                        "fcf_base": fcf_base,
                        "historical_revenue_cagr_pct": historical_cagr_pct,
                        "current_revenue_growth_pct": current_growth_pct,
                        "capex_pct_revenue": capex_pct_revenue,
                        "projection_years": 5,
                        "source_period": f"annual:{valid_rev[-1]['year']}",
                    },
                    period="derived:annual_projection",
                )
                state["calculated_ratios"]["forecast"] = forecast
                # Derived/projected — built from the most recent ANNUAL
                # operating cash flow & capex (both XBRL 10-K, same
                # source/period family as net_margin/roe above), grounded on
                # the annual revenue CAGR. Not comparable to a quarterly
                # actual; label it as such rather than silently "annual".
                state["calculated_ratios"]["dcf"] = await _call(
                    "tool_calc_dcf_scenarios",
                    {
                        "fcf_base": fcf_base,
                        "beta": beta,
                        "shares_outstanding": shares_outstanding,
                        "diluted_shares": shares_outstanding,
                        "market_cap": market_cap,
                        "total_debt": total_debt_ttm,
                        "cash": total_cash_ttm,
                        # Kept None rather than silently converted to zero. The
                        # assessment marks it NOT_APPLICABLE because Yahoo totalCash
                        # is the combined cash/short-term-investment bucket.
                        "marketable_securities": None,
                        "lease_liabilities": lease_liabilities_ttm,
                        "minority_interest": minority_interest_ttm,
                        "pre_tax_cost_of_debt_pct": pre_tax_cost_of_debt_pct,
                        "effective_tax_rate_pct": effective_tax_rate_pct,
                        "input_assessments": valuation_input_assessments,
                        "forecast_scenarios": forecast.get("scenarios", {}),
                    },
                    period="derived:annual_projection",
                )
                # Also run the Monte Carlo version alongside the three
                # named scenarios — bear/base/bull are easier to narrate,
                # while the P10/P50/P90 percentiles give a continuous
                # picture of valuation uncertainty rather than three
                # discrete points. Both are kept since they serve
                # different reading purposes in the final report.
                base_dcf = state["calculated_ratios"]["dcf"].get("base", {})
                base_forecast = forecast.get("scenarios", {}).get("base", {})
                if base_dcf.get("error") is None and base_forecast.get("fcf_by_year"):
                    state["calculated_ratios"]["dcf_monte_carlo"] = await _call(
                        "tool_calc_dcf_multi_factor_monte_carlo",
                        {
                            "fcf_by_year": base_forecast["fcf_by_year"],
                            "wacc_pct": base_dcf.get("wacc_pct"),
                            "terminal_growth_rate_pct": base_dcf.get("assumptions", {}).get("terminal_growth_rate_pct", 2.5),
                            "cash": total_cash_ttm or 0.0,
                            "total_debt": total_debt_ttm or 0.0,
                            "lease_liabilities": lease_liabilities_ttm or 0.0,
                            "minority_interest": minority_interest_ttm or 0.0,
                            "diluted_shares": shares_outstanding,
                            "n_simulations": 2000,
                        },
                        period="derived:annual_projection",
                    )
                else:
                    state["calculated_ratios"]["dcf_monte_carlo"] = {
                        "interpretation": "unavailable", "model_classification": "MULTI_FACTOR",
                        "error": "A valid explicit base forecast and WACC are required.",
                    }
        else:
            state["calculated_ratios"]["dcf"] = {
                "enterprise_value": None,
                "interpretation": "unavailable",
                "error": (
                    "Missing operating_cash_flow_annual/capex_annual from "
                    "XBRL (fcf_base) or beta from Yahoo ratios — cannot run DCF."
                ),
            }

        # -- Phase 1: Cash Flow Ratios & Reverse DCF ---------------------------
        # total_debt/total_cash/ebitda are TTM (Yahoo .info, added to
        # yahoo_finance.get_financial_ratios for exactly these ratios).
        ebitda_ttm     = ratios_src.get("ebitda")

        if total_debt_ttm is not None and total_cash_ttm is not None and ebitda_ttm:
            log.info("Computing Net Debt/EBITDA: debt=%.0f cash=%.0f ebitda=%.0f",
                      total_debt_ttm, total_cash_ttm, ebitda_ttm)
            state["calculated_ratios"]["net_debt_to_ebitda"] = await _call(
                "tool_calc_net_debt_to_ebitda",
                {"total_debt": total_debt_ttm, "total_cash": total_cash_ttm, "ebitda": ebitda_ttm},
                period="ttm",
            )
        else:
            state["calculated_ratios"]["net_debt_to_ebitda"] = {"net_debt_to_ebitda": None, "interpretation": "unavailable"}

        # fcf_base is annual (XBRL, computed above for DCF); revenue here
        # uses the most recent ANNUAL figure to stay in the same source
        # family, but the two annual series (XBRL cash flow vs. yfinance
        # income statement) aren't guaranteed to be the exact same fiscal
        # year — tag MIXED rather than assume alignment, same convention
        # as net_margin's fallback path above.
        fcf_margin_revenue = annual_rev[0]["revenue"] if annual_rev else None
        if fcf_base is not None and fcf_margin_revenue:
            log.info("Computing FCF Margin: fcf=%.0f revenue=%.0f", fcf_base, fcf_margin_revenue)
            state["calculated_ratios"]["fcf_margin"] = await _call(
                "tool_calc_fcf_margin",
                {"free_cash_flow": fcf_base, "revenue": fcf_margin_revenue},
                period=f"MIXED(fcf=annual_derived, revenue=annual:{revenue_year})",
            )
        else:
            state["calculated_ratios"]["fcf_margin"] = {"fcf_margin_pct": None, "interpretation": "unavailable"}

        if fcf_base is not None and market_cap:
            log.info("Computing FCF Yield: fcf=%.0f market_cap=%.0f", fcf_base, market_cap)
            state["calculated_ratios"]["fcf_yield"] = await _call(
                "tool_calc_fcf_yield",
                {"free_cash_flow": fcf_base, "market_cap": market_cap},
                period="MIXED(fcf=annual_derived, market_cap=ttm)",
            )
        else:
            state["calculated_ratios"]["fcf_yield"] = {"fcf_yield_pct": None, "interpretation": "unavailable"}

        bs_snapshot = raw.get("balance_sheet_snapshot", {})
        current_liabilities_snap = bs_snapshot.get("current_liabilities")

        if operating_cf_annual and current_liabilities_snap:
            ocf_annual = operating_cf_annual[0]["value"]
            log.info("Computing OCF Ratio: ocf=%.0f current_liab=%.0f", ocf_annual, current_liabilities_snap)
            state["calculated_ratios"]["ocf_ratio"] = await _call(
                "tool_calc_ocf_ratio",
                {"operating_cash_flow": ocf_annual, "current_liabilities": current_liabilities_snap},
                period=f"MIXED(ocf=annual, current_liabilities={bs_snapshot.get('period_end', 'unknown')})",
            )
        else:
            state["calculated_ratios"]["ocf_ratio"] = {"ocf_ratio": None, "interpretation": "unavailable"}

        # Reverse DCF: target enterprise value ≈ market_cap + total_debt −
        # total_cash (all TTM) -- see reverse_dcf's own docstring for why
        # raw market_cap alone would bias the result.
        target_ev = None
        if market_cap is not None and total_debt_ttm is not None and total_cash_ttm is not None:
            target_ev = market_cap + total_debt_ttm - total_cash_ttm

        if fcf_base is not None and beta is not None and target_ev is not None and target_ev > 0:
            log.info("Computing Reverse DCF: fcf_base=%.0f beta=%.3f target_ev=%.0f",
                      fcf_base, beta, target_ev)
            state["calculated_ratios"]["reverse_dcf"] = await _call(
                "tool_calc_reverse_dcf",
                {
                    "fcf_base": fcf_base, "beta": beta,
                    "target_enterprise_value": target_ev,
                    "market_cap": market_cap, "total_debt": total_debt_ttm,
                    "pre_tax_cost_of_debt_pct": pre_tax_cost_of_debt_pct,
                    "effective_tax_rate_pct": effective_tax_rate_pct,
                },
                period="MIXED(fcf=annual_derived, market_cap/debt/cash=ttm)",
            )
        else:
            state["calculated_ratios"]["reverse_dcf"] = {
                "implied_growth_rate_pct": None, "interpretation": "unavailable",
                "error": "Missing fcf_base, beta, or a valid target enterprise value (market_cap + total_debt − total_cash).",
            }

        # -- Phase 2: Advanced Profitability, Leverage & Bankruptcy Risk ------
        total_assets_val      = xbrl_assets[0]["value"] if xbrl_assets else None
        total_liabilities_val_p2 = xbrl_liabilities[0]["value"] if xbrl_liabilities else None
        ebit_snap    = bs_snapshot.get("ebit")
        retained_earnings_snap = bs_snapshot.get("retained_earnings")
        working_capital_snap   = bs_snapshot.get("working_capital")

        # ROIC needs an effective tax rate this pipeline doesn't have a
        # direct source for (no pretax-income/tax-provision series
        # available here) — use the US federal statutory rate (21%) as a
        # documented placeholder, NOT the company's real effective rate.
        # Flagged explicitly in the result, same "is_approximation" pattern
        # used for the D/E-from-XBRL approximation above.
        _STATUTORY_TAX_RATE_PCT = 21.0
        if ebit_snap is not None and total_debt_ttm is not None and shareholders_equity and total_cash_ttm is not None:
            log.info("Computing ROIC: ebit=%.0f equity=%.0f", ebit_snap, shareholders_equity)
            roic_result = await _call(
                "tool_calc_roic",
                {
                    "ebit": ebit_snap, "tax_rate_pct": _STATUTORY_TAX_RATE_PCT,
                    "total_debt": total_debt_ttm, "total_equity": shareholders_equity,
                    "cash": total_cash_ttm,
                },
                period=f"MIXED(ebit={bs_snapshot.get('period_end','unknown')}, equity=annual:{equity_period_end}, debt/cash=ttm)",
            )
            roic_result["is_approximation"] = True
            roic_result["note"] = (
                f"tax_rate_pct used the US statutory rate ({_STATUTORY_TAX_RATE_PCT}%) as a "
                "placeholder, not this company's actual effective tax rate (no pretax-income/"
                "tax-provision series available in this pipeline) -- NOPAT here is an approximation."
            )
            state["calculated_ratios"]["roic"] = roic_result
        else:
            state["calculated_ratios"]["roic"] = {"roic_pct": None, "interpretation": "unavailable"}

        if ebit_snap is not None and total_assets_val and current_liabilities_snap is not None:
            log.info("Computing ROCE: ebit=%.0f total_assets=%.0f", ebit_snap, total_assets_val)
            state["calculated_ratios"]["roce"] = await _call(
                "tool_calc_roce",
                {"ebit": ebit_snap, "total_assets": total_assets_val, "current_liabilities": current_liabilities_snap},
                period=f"MIXED(ebit={bs_snapshot.get('period_end','unknown')}, total_assets=annual:{equity_period_end})",
            )
        else:
            state["calculated_ratios"]["roce"] = {"roce_pct": None, "interpretation": "unavailable"}

        # debt_ratio / equity_ratio: total_liabilities and total_assets are
        # BOTH from the same XBRL annual snapshot (index [0] of each series,
        # same source/period family as ROE/D-E above) -- clean annual tag,
        # not mixed.
        if total_liabilities_val_p2 is not None and total_assets_val:
            state["calculated_ratios"]["debt_ratio"] = await _call(
                "tool_calc_debt_ratio",
                {"total_liabilities": total_liabilities_val_p2, "total_assets": total_assets_val},
                period=f"annual:{equity_period_end}" if equity_period_end else "annual",
            )
        else:
            state["calculated_ratios"]["debt_ratio"] = {"debt_ratio": None, "interpretation": "unavailable"}

        if shareholders_equity and total_assets_val:
            state["calculated_ratios"]["equity_ratio"] = await _call(
                "tool_calc_equity_ratio",
                {"total_equity": shareholders_equity, "total_assets": total_assets_val},
                period=f"annual:{equity_period_end}" if equity_period_end else "annual",
            )
        else:
            state["calculated_ratios"]["equity_ratio"] = {"equity_ratio": None, "interpretation": "unavailable"}

        altman_revenue = annual_rev[0]["revenue"] if annual_rev else None
        if (working_capital_snap is not None and retained_earnings_snap is not None
                and ebit_snap is not None and total_assets_val and market_cap is not None
                and total_liabilities_val_p2 is not None and altman_revenue is not None):
            log.info("Computing Altman Z-Score")
            state["calculated_ratios"]["altman_z_score"] = await _call(
                "tool_calc_altman_z_score",
                {
                    "working_capital": working_capital_snap, "retained_earnings": retained_earnings_snap,
                    "ebit": ebit_snap, "total_assets": total_assets_val, "market_cap": market_cap,
                    "total_liabilities": total_liabilities_val_p2, "revenue": altman_revenue,
                },
                period="mixed",
            )
        else:
            state["calculated_ratios"]["altman_z_score"] = {"z_score": None, "interpretation": "unavailable"}

        # -- Composite Financial Score ----------------------------------------
        log.info("Computing composite financial score")
        cr   = state["calculated_ratios"]
        # current_ratio now comes through from tool_get_financial_ratios
        # (yfinance .info "currentRatio"). Without passing it here the composite
        # score always flagged current_ratio as a missing input.
        current_ratio_val = ratios_src.get("current_ratio")
        # This composite is DELIBERATELY built from ratios spanning several
        # periods (ttm P/E, annual ROE/net_margin, mrq current/D-E ratio,
        # multi-year CAGR) — that's fine for a single composite health score,
        # but tag it "mixed" explicitly so validate_period_consistency.py
        # doesn't (correctly, but unhelpfully) flag it the way it flags an
        # *accidental* mix like the net_margin bug. See each component's own
        # `_period` tag (in cr[...]["_period"]) for its specific provenance.
        state["calculated_ratios"]["composite_score"] = await _call(
            "tool_calc_composite_score",
            {
                "pe":                cr.get("pe", {}).get("pe_ratio"),
                "roe_pct":           cr.get("roe", {}).get("roe_pct"),
                "net_margin_pct":    cr.get("net_margin", {}).get("net_margin_pct"),
                "de_ratio":          cr.get("de_ratio", {}).get("de_ratio"),
                "revenue_cagr_pct":  cr.get("cagr", {}).get("cagr_pct"),
                "current_ratio_val": current_ratio_val,
            },
            period="mixed",
        )

        log.info("Executor: ratio computation complete.")

    # =========================================================================
    # LAYER 2 — CHECKER / CRITIC (Validation & Quality Control)
    # =========================================================================

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
        BRAIN — Planning Node for the Internal Execution Loop.

        Consults Claude to review the current state of the extraction loop,
        digest any incoming ``validation_feedback`` from the Checker, and produce
        structured internal instructions for the Executor nodes in the next cycle.

        This method is called once per loop iteration before the Executor nodes
        run. Its output is advisory: the Executor nodes use the Brain's plan
        to prioritise which MCP tools to invoke.

        Parameters
        ----------
        state : FinancialAgentState
            Current mutable agent-local state. Read fields:
            - ``state["messages"]``           : Full conversation history.
            - ``state["loop_counter"]``        : Current iteration index.
            - ``state["validation_feedback"]`` : Last Checker critique (may be "").
            - ``state["task_query"]`` / ``state["manager_directives"]``  : Original Manager task.

        Returns
        -------
        dict[str, Any]
            Structured planning output with keys:
            - ``"plan"``        (str)          : Summary of intended actions.
            - ``"priority_tools"`` (list[str]) : Ordered list of MCP tools to call.
            - ``"raw_plan"``    (str)          : Full model response text.
        """
        iteration   = state["loop_counter"]
        feedback    = state["validation_feedback"]
        task_query  = state["task_query"]
        directives  = state["manager_directives"]

        system_prompt = (
            "You are the Brain of a FinancialAnalystAgent. "
            "Your role is to produce concise, structured execution plans for financial data extraction loops. "
            "Always respond in valid JSON with exactly two keys: "
            "'plan' (a 1-3 sentence summary of what to do) and "
            "'priority_tools' (an ordered list of MCP tool name strings to invoke). "
            "Do not include any explanation outside the JSON object."
        )

        user_message = (
            f"ITERATION: {iteration}\n"
            f"TASK QUERY: {task_query}\n"
            f"MANAGER DIRECTIVES: {json.dumps(directives)}\n"
            f"CHECKER FEEDBACK: {feedback if feedback else 'None — first iteration or prior run passed.'}\n\n"
            "Based on the above, produce the execution plan for this iteration."
        )

        # Append user message to the running conversation log
        state["messages"].append({"role": "user", "content": user_message})

        log.info("Brain: invoking Claude for iteration %d planning...", iteration)
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

            # Attempt to parse as JSON; fall back gracefully
            try:
                clean = raw_plan.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean)
                plan           = parsed.get("plan", "Execute standard extraction.")
                priority_tools = parsed.get("priority_tools", [])
            except json.JSONDecodeError:
                log.warning("Brain: Claude response was not valid JSON — using defaults.")
                plan           = raw_plan
                priority_tools = [
                    "tool_get_financial_ratios",
                    "tool_get_revenue_growth",
                    "tool_get_xbrl_financials",
                ]

            log.info("Brain plan for iteration %d: %s", iteration, plan)
            return {"plan": plan, "priority_tools": priority_tools, "raw_plan": raw_plan}

        except Exception as exc:
            log.exception("Brain: Claude API call failed — using default plan.")
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "financial.brain")
                    scope.set_tag("iteration", str(iteration))
                    sentry_sdk.capture_exception(exc)
            default_plan = "Default plan: run full data extraction and ratio computation."
            state["messages"].append({"role": "assistant", "content": default_plan})
            return {
                "plan": default_plan,
                "priority_tools": [
                    "tool_get_financial_ratios",
                    "tool_get_revenue_growth",
                    "tool_get_xbrl_financials",
                ],
                "raw_plan": default_plan,
            }

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
        log.info("Brain plan: %s", brain_output.get("plan"))
        return {"loop_counter": new_counter}

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
