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
import sys
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

# ---------------------------------------------------------------------------
# State contract imports
# ---------------------------------------------------------------------------
# Both SharedManagerState and FinancialAgentState are declared in
# agents/state.py — the single source of truth for all state TypedDicts
# across the Alpha-Agent Node platform.
from agents.state import FinancialAgentState, SharedManagerState
from langsmith import traceable
from core.observability import sentry_enabled
from core.progress_bus import publish as _publish_progress, session_from_shared

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
# Checker system prompt — Claude-powered financial data critic
# ---------------------------------------------------------------------------
_CHECKER_SYSTEM_PROMPT = """You are the Financial Data Critic for the Alpha-Agent Node platform.

Your role: Perform a rigorous audit of the extracted financial data and \
computed ratios to decide whether the dataset is complete, consistent, and \
sufficient for the Manager Agent to make high-quality investment decisions.

You will receive:
  - TASK QUERY       : The original research objective.
  - RAW DATA SUMMARY : Key fields extracted from Yahoo Finance and SEC EDGAR.
  - CALCULATED RATIOS: Computed metrics (P/E, ROE, Net Margin, D/E, CAGR, Score).

Audit criteria (ALL must pass for is_complete = true):
  1. DATA PRESENCE      : Yahoo Finance ratios payload is non-empty and error-free.
  2. REVENUE HISTORY    : At least 2 years of annual revenue data exist for CAGR.
  3. CORE RATIO COVERAGE: No more than 1 of the 5 core ratios (pe, roe,
                          net_margin, de_ratio, cagr) may be null or missing.
  4. VALUATION SANITY   : P/E ratio is present and numerically plausible
                          (positive, below 1000) for the company's sector.
  5. COMPOSITE SCORE    : Weighted composite health score (0-100) is computable
                          and not None.
  6. INTERNAL CONSISTENCY: Computed ratios are mathematically consistent with
                          the raw data (e.g. net_margin aligns with net_income /
                          revenue, CAGR direction matches revenue trend).
  7. MANAGER READINESS  : The dataset as a whole is rich enough for the Manager
                          Agent to draw actionable conclusions about the company's
                          financial health, valuation, and growth trajectory.

Output format (strict JSON — no markdown fences, no preamble):
{
  "is_complete"  : true | false,
  "score"        : <int 0-100, overall data quality score>,
  "passed"       : ["<criterion name>", ...],
  "failed"       : ["<criterion name>", ...],
  "issues"       : ["<specific problem description>", ...],
  "feedback"     : "<actionable re-extraction instructions for the Brain, or empty string if is_complete is true>"
}

Rules:
- Output ONLY valid JSON. No explanation outside the JSON object.
- Be strict: partial or inconsistent data must NOT pass.
- If is_complete is true, feedback MUST be an empty string "".
- feedback must be specific and actionable — name the exact tools to re-call
  and what data is missing, so the Brain can plan the next iteration precisely.
"""


def _sanitize_nans(value: Any) -> Any:
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
        return {k: _sanitize_nans(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nans(v) for v in value]
    return value


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
        log.info(
            "FinancialAnalystAgent initialised — model=%s, server_script=%s",
            model,
            self._server_params.args[-1] if self._server_params.args else "<none>",
        )

    # =========================================================================
    # INTERNAL UTILITY — ticker extraction
    # =========================================================================

    def _extract_ticker(self, task_query: str, directives: dict[str, Any]) -> str | None:
        """
        Extract the target ticker symbol from manager directives or task query.

        Resolution order:
            1. ``manager_directives["ticker"]`` (explicit override, highest priority).
            2. Regex scan of ``task_query`` for 1-5 uppercase letter tokens.
            3. Returns ``None`` if no ticker can be inferred.

        Parameters
        ----------
        task_query : str
            Natural-language research question from the Manager Agent.
        directives : dict[str, Any]
            Manager directives dictionary (may be empty).

        Returns
        -------
        str | None
            Uppercase ticker symbol, or ``None`` if extraction fails.
        """
        if directives.get("ticker"):
            return str(directives["ticker"]).upper()

        # Heuristic: match isolated uppercase tokens of 1–5 chars
        candidates = re.findall(r"\b([A-Z]{1,5})\b", task_query)
        # Filter out common English stop-words that look like tickers
        stop_words = {"A", "I", "THE", "AND", "OR", "IN", "OF", "FOR", "AI", "Q1",
                      "Q2", "Q3", "Q4", "US", "SEC", "PE", "YOY", "TTM", "EPS"}
        for candidate in candidates:
            if candidate not in stop_words:
                log.info("Inferred ticker from task_query: %s", candidate)
                return candidate

        log.warning("Could not infer ticker from task_query: '%s'", task_query)
        return None

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

        Parameters
        ----------
        session : ClientSession
            Active MCP client session connected to the financial server subprocess.
        state : FinancialAgentState
            Mutable agent-local state. This method appends data under:
            ``state["raw_numerical_data"]["yahoo_ratios"]``
            ``state["raw_numerical_data"]["revenue_growth"]``
            ``state["raw_numerical_data"]["xbrl_financials"]``

        Returns
        -------
        None
            Results are written directly into ``state``. Errors are recorded
            under ``state["raw_numerical_data"]["extraction_errors"]``.
        """
        directives = state["shared_manager_ref"].get("manager_directives", {})
        task_query = state["shared_manager_ref"].get("task_query", "")
        ticker     = self._extract_ticker(task_query, directives)

        if not ticker:
            state["raw_numerical_data"]["extraction_errors"] = (
                "Could not determine ticker symbol from task_query or manager_directives."
            )
            log.error("Executor: ticker extraction failed — aborting data pull.")
            return

        state["raw_numerical_data"]["ticker"] = ticker
        errors: list[str] = []
        session_id = session_from_shared(state["shared_manager_ref"])
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

        if errors:
            state["raw_numerical_data"]["extraction_errors"] = errors
        log.info(
            "Executor: data extraction complete — %d error(s) recorded.", len(errors)
        )
        _publish_progress(
            session_id, "agent_tool_result", agent="financial",
            message="Financial Agent: 'tool_get_xbrl_financials' completed — data extraction done",
            detail={"tool": "tool_get_xbrl_financials", "errors": len(errors)},
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
        session_id = session_from_shared(state["shared_manager_ref"])

        # -- Helper: safe MCP tool call ---------------------------------------
        async def _call(tool: str, args: dict) -> dict:
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
                return {"error": str(exc)}

        # -- P/E Ratio --------------------------------------------------------
        price = ratios_src.get("current_price")
        eps   = ratios_src.get("eps_trailing")
        if price is not None and eps is not None and eps != 0:
            log.info("Computing P/E: price=%.2f eps=%.4f", price, eps)
            state["calculated_ratios"]["pe"] = await _call(
                "tool_calc_pe", {"price": price, "eps": eps}
            )
        else:
            state["calculated_ratios"]["pe"] = {
                "pe_ratio": ratios_src.get("pe_ratio"),
                "interpretation": "sourced_from_yahoo",
                "note": "Used pre-computed Yahoo Finance P/E (price or EPS unavailable for direct calc).",
            }

        # -- Return on Equity -------------------------------------------------
        xbrl_assets      = xbrl_src.get("total_assets", [])
        xbrl_liabilities = xbrl_src.get("total_liabilities", [])
        annual_ni        = growth_src.get("annual_net_income", [])

        net_income = annual_ni[0]["net_income"] if annual_ni else None
        # Approximate equity = assets - liabilities (most recent period)
        shareholders_equity: float | None = None
        if xbrl_assets and xbrl_liabilities:
            try:
                shareholders_equity = (
                    xbrl_assets[0]["value"] - xbrl_liabilities[0]["value"]
                )
            except (KeyError, TypeError, IndexError):
                shareholders_equity = None

        if net_income and shareholders_equity and shareholders_equity != 0:
            log.info("Computing ROE: net_income=%.0f equity=%.0f", net_income, shareholders_equity)
            state["calculated_ratios"]["roe"] = await _call(
                "tool_calc_roe",
                {"net_income": net_income, "shareholders_equity": shareholders_equity},
            )
        else:
            state["calculated_ratios"]["roe"] = {"roe_pct": None, "interpretation": "insufficient_data"}

        # -- Net Margin -------------------------------------------------------
        annual_rev = growth_src.get("annual_revenue", [])
        revenue = annual_rev[0]["revenue"] if annual_rev else None

        if net_income is not None and revenue is not None and revenue != 0:
            log.info("Computing Net Margin: net_income=%.0f revenue=%.0f", net_income, revenue)
            state["calculated_ratios"]["net_margin"] = await _call(
                "tool_calc_net_margin",
                {"net_income": net_income, "revenue": revenue},
            )
        else:
            state["calculated_ratios"]["net_margin"] = {"net_margin_pct": None, "interpretation": "insufficient_data"}

        # -- Debt-to-Equity ---------------------------------------------------
        # D/E from Yahoo Finance info if available; otherwise note as unavailable
        de_ratio_yahoo = ratios_src.get("de_ratio") or ratios_src.get("debtToEquity")
        if de_ratio_yahoo is not None:
            # Yahoo already provides D/E; store directly without an MCP round-trip
            state["calculated_ratios"]["de_ratio"] = {
                "de_ratio": de_ratio_yahoo,
                "interpretation": "sourced_from_yahoo",
            }
        elif shareholders_equity and shareholders_equity != 0:
            # Attempt approximation from XBRL
            total_liabilities_val = xbrl_liabilities[0]["value"] if xbrl_liabilities else None
            if total_liabilities_val:
                log.info("Computing D/E from XBRL liabilities/equity")
                state["calculated_ratios"]["de_ratio"] = await _call(
                    "tool_calc_debt_to_equity",
                    {
                        "total_debt": total_liabilities_val,
                        "shareholders_equity": shareholders_equity,
                    },
                )
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
                state["calculated_ratios"]["cagr"] = await _call(
                    "tool_calc_cagr",
                    {"start_value": oldest, "end_value": newest, "years": float(years)},
                )
            else:
                state["calculated_ratios"]["cagr"] = {"cagr_pct": None, "interpretation": "unavailable"}
        else:
            state["calculated_ratios"]["cagr"] = {"cagr_pct": None, "interpretation": "insufficient_history"}

        # -- Composite Financial Score ----------------------------------------
        log.info("Computing composite financial score")
        cr   = state["calculated_ratios"]
        # current_ratio now comes through from tool_get_financial_ratios
        # (yfinance .info "currentRatio"). Without passing it here the composite
        # score always flagged current_ratio as a missing input.
        current_ratio_val = ratios_src.get("current_ratio")
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
            - ``state["shared_manager_ref"]``  : For task_query context.
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
        task_query = state["shared_manager_ref"].get("task_query", "")

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
            "de_ratio":       calc.get("de_ratio", {}),
            "cagr":           calc.get("cagr", {}),
            "composite_score": calc.get("composite_score", {}),
        }

        user_content = (
            f"TASK QUERY:\n{task_query}\n\n"
            f"LOOP ITERATION: {loop}\n\n"
            f"RAW DATA SUMMARY:\n{json.dumps(raw_summary, indent=2)}\n\n"
            f"CALCULATED RATIOS:\n{json.dumps(ratio_summary, indent=2)}\n\n"
            "Audit the above and return your JSON verdict."
        )

        # -- Call Claude as the Financial Critic ------------------------------
        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=768,
                system=_CHECKER_SYSTEM_PROMPT,
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
            session_from_shared(state["shared_manager_ref"]), "agent_checker", agent="financial",
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
            - ``state["shared_manager_ref"]``  : Original Manager task.

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
        task_query  = state["shared_manager_ref"].get("task_query", "")
        directives  = state["shared_manager_ref"].get("manager_directives", {})

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
                system=system_prompt,
                messages=state["messages"],
            )
            raw_plan = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw_plan})

            # Attempt to parse as JSON; fall back gracefully
            try:
                parsed = json.loads(raw_plan)
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
            "shared_manager_ref":  shared_state,
        }

        # -- Open MCP client session ------------------------------------------
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                # Handshake: initialise the MCP protocol connection
                await session.initialize()
                log.info("MCP session initialised successfully.")

                # -- Main lifecycle loop --------------------------------------
                while state["loop_counter"] < max_loops:
                    state["loop_counter"] += 1
                    log.info(
                        "--- Loop iteration %d / %d ---",
                        state["loop_counter"],
                        max_loops,
                    )

                    # Layer 3: Brain plans this iteration
                    brain_output = await self._brain(state)
                    log.info("Brain plan: %s", brain_output["plan"])

                    # Layer 1a: Extract raw market and SEC data
                    await self._execute_data_extraction(session, state)

                    # Layer 1b: Compute financial ratios
                    await self._execute_ratio_computation(session, state)

                    # Layer 2: Claude-powered Checker audits the extracted data
                    check_result = await self._check_data_quality(state)
                    state["is_complete"]         = check_result["is_complete"]
                    state["validation_feedback"] = check_result["feedback"]

                    if state["is_complete"]:
                        log.info(
                            "[Loop] Checker PASSED (score=%s) — exiting after %d iteration(s).",
                            check_result.get("score"),
                            state["loop_counter"],
                        )
                        break
                    else:
                        log.warning(
                            "[Loop] Checker FAILED on iteration %d "
                            "(score=%s, failed=%s). Feedback: %s",
                            state["loop_counter"],
                            check_result.get("score"),
                            check_result.get("failed", []),
                            state["validation_feedback"][:120],
                        )

                if not state["is_complete"]:
                    log.warning(
                        "Safety guardrail hit: exiting after %d loops without full validation.",
                        state["loop_counter"],
                    )

        # -- Assemble financial_metrics_summary --------------------------------
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]

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
            "roe":           calc.get("roe", {}),
            "net_margin":    calc.get("net_margin", {}),
            "de_ratio":      calc.get("de_ratio", {}),
            "revenue_cagr":  calc.get("cagr", {}),
            "current_ratio": raw.get("yahoo_ratios", {}).get("current_ratio"),
            "composite_score": calc.get("composite_score", {}),

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
        financial_metrics_summary = _sanitize_nans(financial_metrics_summary)

        # Commit to the shared state contract
        shared_state["financial_metrics_summary"] = financial_metrics_summary  # type: ignore[literal-required]

        log.info(
            "FinancialAnalystAgent.run() complete — ticker=%s, score=%s, grade=%s",
            financial_metrics_summary["ticker"],
            calc.get("composite_score", {}).get("score"),
            calc.get("composite_score", {}).get("grade"),
        )

        return shared_state