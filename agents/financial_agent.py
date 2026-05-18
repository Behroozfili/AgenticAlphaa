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
import re
import sys
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# State contract imports
# ---------------------------------------------------------------------------
# Both SharedManagerState and FinancialAgentState are declared in
# agents/state.py — the single source of truth for all state TypedDicts
# across the Alpha-Agent Node platform.
from agents.state import FinancialAgentState, SharedManagerState

# ---------------------------------------------------------------------------
# Logging — stderr only; stdout is reserved for MCP JSON-RPC
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("financial-analyst-agent")




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
        Anthropic model identifier. Defaults to ``"claude-sonnet-4-20250514"``.
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
        server_script_path: str,
        model: str = "claude-sonnet-4-20250514",
        max_loops: int = 3,
    ) -> None:
        self._llm    = anthropic.Anthropic()
        self._model  = model
        self._default_max_loops = max_loops
        self._server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None,  # inherits the current process environment
        )
        log.info(
            "FinancialAnalystAgent initialised — model=%s, server=%s",
            model,
            server_script_path,
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

        # -- Yahoo Finance: key ratios ----------------------------------------
        log.info("Executor: calling tool_get_financial_ratios for %s", ticker)
        try:
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

        # -- Yahoo Finance: revenue growth ------------------------------------
        log.info("Executor: calling tool_get_revenue_growth for %s", ticker)
        try:
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

        # -- SEC EDGAR: XBRL structured financials ----------------------------
        log.info("Executor: calling tool_get_xbrl_financials for %s", ticker)
        try:
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

        if errors:
            state["raw_numerical_data"]["extraction_errors"] = errors
        log.info(
            "Executor: data extraction complete — %d error(s) recorded.", len(errors)
        )

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

        # -- Helper: safe MCP tool call ---------------------------------------
        async def _call(tool: str, args: dict) -> dict:
            try:
                result = await session.call_tool(tool, arguments=args)
                return json.loads(result.content[0].text)
            except Exception as exc:
                log.warning("Ratio computation tool '%s' failed: %s", tool, exc)
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
        if len(annual_rev) >= 2:
            newest = annual_rev[0]["revenue"]
            oldest = annual_rev[-1]["revenue"]
            years  = len(annual_rev) - 1
            if newest and oldest and oldest > 0:
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
        state["calculated_ratios"]["composite_score"] = await _call(
            "tool_calc_composite_score",
            {
                "pe":               cr.get("pe", {}).get("pe_ratio"),
                "roe_pct":          cr.get("roe", {}).get("roe_pct"),
                "net_margin_pct":   cr.get("net_margin", {}).get("net_margin_pct"),
                "de_ratio":         cr.get("de_ratio", {}).get("de_ratio"),
                "revenue_cagr_pct": cr.get("cagr", {}).get("cagr_pct"),
            },
        )

        log.info("Executor: ratio computation complete.")

    # =========================================================================
    # LAYER 2 — CHECKER / CRITIC (Validation & Quality Control)
    # =========================================================================

    def _check_data_quality(self, state: FinancialAgentState) -> dict[str, Any]:
        """
        CHECKER — Data Integrity Audit and Quality Control.

        Acts as the Financial Critic: audits the contents of
        ``state["raw_numerical_data"]`` and ``state["calculated_ratios"]``
        against a set of data integrity constraints.

        Integrity Constraints Enforced
        --------------------------------
        1. ``raw_numerical_data["yahoo_ratios"]`` must be non-empty and error-free.
        2. ``raw_numerical_data["revenue_growth"]["annual_revenue"]`` must contain
           at least 2 data points (required for CAGR calculation).
        3. ``calculated_ratios["pe"]["pe_ratio"]`` must be a finite non-None number.
        4. ``calculated_ratios["composite_score"]["score"]`` must be non-None.
        5. No more than 2 of the 5 core ratios (pe, roe, net_margin, de_ratio,
           cagr) may have a ``None`` value.

        Parameters
        ----------
        state : FinancialAgentState
            Current mutable agent-local state (read-only in this method).

        Returns
        -------
        dict[str, Any]
            Structured validation result with keys:
            - ``"is_complete"`` (bool)  : True if all constraints pass.
            - ``"feedback"``    (str)   : Actionable critique for the Brain node.
                                          Empty string when ``is_complete`` is True.
            - ``"missing"``     (list)  : List of failed constraint descriptions.
        """
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]
        issues: list[str] = []

        # Constraint 1: Yahoo ratios must be present and error-free
        yahoo = raw.get("yahoo_ratios", {})
        if not yahoo or yahoo.get("error"):
            issues.append(
                f"Yahoo Finance ratios are missing or errored: {yahoo.get('error', 'empty response')}."
            )

        # Constraint 2: At least 2 annual revenue data points for CAGR
        rev_history = raw.get("revenue_growth", {}).get("annual_revenue", [])
        if len(rev_history) < 2:
            issues.append(
                f"Insufficient revenue history for CAGR: {len(rev_history)} year(s) found, minimum 2 required."
            )

        # Constraint 3: P/E ratio must be a finite number
        pe_val = calc.get("pe", {}).get("pe_ratio")
        if pe_val is None:
            issues.append("P/E ratio is None — price or EPS data could not be resolved.")

        # Constraint 4: Composite score must be computable
        score_val = calc.get("composite_score", {}).get("score")
        if score_val is None:
            issues.append(
                "Composite financial score is None — insufficient sub-metrics available."
            )

        # Constraint 5: No more than 2 of the 5 core ratio fields may be None
        core_ratios = {
            "pe":          calc.get("pe", {}).get("pe_ratio"),
            "roe":         calc.get("roe", {}).get("roe_pct"),
            "net_margin":  calc.get("net_margin", {}).get("net_margin_pct"),
            "de_ratio":    calc.get("de_ratio", {}).get("de_ratio"),
            "cagr":        calc.get("cagr", {}).get("cagr_pct"),
        }
        null_count = sum(1 for v in core_ratios.values() if v is None)
        if null_count > 2:
            null_fields = [k for k, v in core_ratios.items() if v is None]
            issues.append(
                f"Too many null core ratios ({null_count}/5). Null fields: {null_fields}. "
                "Re-extraction or alternative data sources required."
            )

        if issues:
            feedback = (
                "DATA QUALITY ISSUES DETECTED — re-extraction required:\n"
                + "\n".join(f"  [{i+1}] {issue}" for i, issue in enumerate(issues))
            )
            log.warning("Checker: %d quality issue(s) found.", len(issues))
            return {"is_complete": False, "feedback": feedback, "missing": issues}

        log.info("Checker: all data quality constraints passed — loop can exit.")
        return {"is_complete": True, "feedback": "", "missing": []}

    # =========================================================================
    # LAYER 3 — BRAIN / ORCHESTRATOR (Lifecycle & Loop Gateway)
    # =========================================================================

    def _brain(self, state: FinancialAgentState) -> dict[str, Any]:
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
            response = self._llm.messages.create(
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
                    brain_output = self._brain(state)
                    log.info("Brain plan: %s", brain_output["plan"])

                    # Layer 1a: Extract raw market and SEC data
                    await self._execute_data_extraction(session, state)

                    # Layer 1b: Compute financial ratios
                    await self._execute_ratio_computation(session, state)

                    # Layer 2: Check data quality
                    check_result = self._check_data_quality(state)
                    state["is_complete"]         = check_result["is_complete"]
                    state["validation_feedback"] = check_result["feedback"]

                    if state["is_complete"]:
                        log.info(
                            "Checker passed — exiting loop after %d iteration(s).",
                            state["loop_counter"],
                        )
                        break
                    else:
                        log.warning(
                            "Checker failed on iteration %d. Feedback: %s",
                            state["loop_counter"],
                            state["validation_feedback"],
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

        # Commit to the shared state contract
        shared_state["financial_metrics_summary"] = financial_metrics_summary  # type: ignore[literal-required]

        log.info(
            "FinancialAnalystAgent.run() complete — ticker=%s, score=%s, grade=%s",
            financial_metrics_summary["ticker"],
            calc.get("composite_score", {}).get("score"),
            calc.get("composite_score", {}).get("grade"),
        )

        return shared_state
