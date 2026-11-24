"""
agents/financial_agent.py

"""

from __future__ import annotations

import json
import logging
import asyncio
import time
from types import SimpleNamespace
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import END, StateGraph
import os
from agents.prompts.prompt_finance_checker import CHECKER_SYSTEM_PROMPT, system_prompt
from evaluation.financial_sanity import evaluate_financial_sanity
from evaluation.earnings_quality import normalize_earnings_quality
from agents.executor_helpers.finance_executor_helper import (
    annotate_confidence, build_consistency_checks, build_data_quality_score,
    extract_ticker, sanitize_nans,
)
from agents.state import FinancialAgentState, SharedManagerState
from langsmith import traceable
from core.forecast_contract import ForecastContext
from core.observability import elapsed_latency_ms
from core.error_handler import add_breadcrumb
from core.progress_bus import publish as _publish_progress, session_from_directives
from agents.executor_helpers.financial_helper import (INTENTS,  BASELINE_EXTRACTION,
    BASELINE_COMPUTE, ANALYSIS_PROFILES, FULL_REPORT_PROFILES,  MONTE_CARLO_TOOLS,
    EXTRACTION_REGISTRY, EXTRACTION_ORDER, COMPUTE_ORDER,
    COMPUTE_REGISTRY, SUMMARY_CALCULATED_PASSTHROUGH, checker_requirements,
    normalize_depth, derive_compute_facts, select_sec_filing_type,

)
from agents.executor_helpers.monthly_financial_features import build_monthly_financial_features
from tools.financial_tools import (
    financial_implementation_identity, financial_lifecycle_event,
    financial_trace_metadata, require_financial_context,
)


# Backward-compatible private aliases retained for the existing test and
# extension contract while implementation lives in financial_helper.
_FULL_REPORT_PROFILES = FULL_REPORT_PROFILES
_MONTE_CARLO_TOOLS = MONTE_CARLO_TOOLS
_checker_requirements = checker_requirements
_select_sec_filing_type = select_sec_filing_type




log = logging.getLogger("financial-analyst-agent")

_TRACE_IDENTITY = financial_implementation_identity()
_FINANCIAL_MCP_TOOLS = frozenset(
    set(EXTRACTION_REGISTRY) | set(COMPUTE_REGISTRY) | {
        "tool_get_cik", "tool_list_filings", "tool_get_filing_text",
    }
)
_MCP_ERROR_CODES = frozenset({
    "CURRENT_ONLY_PROVIDER_REJECTED", "FINANCIAL_CALCULATION_FAIL_CLOSED",
    "FINANCIAL_TOOL_DATA_UNAVAILABLE", "FINANCIAL_TOOL_FAIL_CLOSED",
    "SEC_CIK_UNAVAILABLE", "SEC_COMPANYFACTS_UNAVAILABLE",
    "SEC_FILING_UNAVAILABLE", "SEC_SUBMISSIONS_UNAVAILABLE",
    "YAHOO_ADJUSTED_HISTORY_UNAVAILABLE",
})


def _financial_identity(state: Any) -> tuple[ForecastContext, str]:
    payload = state if isinstance(state, dict) else {}
    directives = payload.get("manager_directives") if isinstance(payload.get("manager_directives"), dict) else {}
    return require_financial_context(
        payload.get("forecast_context") or directives.get("forecast_context"),
        directives.get("run_id") or payload.get("run_id"),
        ticker=directives.get("ticker"),
    )


def _trace_inputs(operation: str):
    def process(inputs: dict[str, Any]) -> dict[str, Any]:
        state = inputs.get("shared_state") or inputs.get("state") or {}
        try:
            context, run_id = _financial_identity(state)
            metadata = financial_trace_metadata(context, run_id)
        except (TypeError, ValueError):
            metadata = dict(_TRACE_IDENTITY)
        return {
            "operation": operation, "metadata": metadata,
            "loop_counter": state.get("loop_counter", 0) if isinstance(state, dict) else 0,
            "resume": "not_applicable",
        }
    return process


def _trace_outputs(operation: str):
    def process(outputs: Any) -> dict[str, Any]:
        payload = outputs if isinstance(outputs, dict) else {}
        features = payload.get("financial_features") if isinstance(payload.get("financial_features"), dict) else {}
        evidence = features.get("evidence_ids") if isinstance(features.get("evidence_ids"), list) else []
        return {
            "operation": operation, "metadata": dict(_TRACE_IDENTITY),
            "evidence_count": len(evidence), "missing": bool(features.get("missing", False)),
            "degraded": bool(features.get("degraded", False)), "resume": "not_applicable",
        }
    return process


def _safe_tool_name(value: Any) -> str:
    tool = str(value or "").strip()
    if tool not in _FINANCIAL_MCP_TOOLS:
        raise ValueError("unknown Stage05 financial MCP tool")
    return tool


def _bounded_score(value: Any) -> int:
    try:
        return max(0, min(int(value), 100))
    except (TypeError, ValueError, OverflowError):
        return 0


class _SafeFinancialMCPResult:
    """Proxy the MCP result while exposing only compact trace metadata."""

    __slots__ = ("_raw", "safe_trace")

    def __init__(self, raw: Any, safe_trace: dict[str, Any]) -> None:
        self._raw = raw
        self.safe_trace = safe_trace

    @property
    def content(self) -> Any:
        return self._raw.content


def _financial_mcp_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    context = inputs.get("context")
    run_id = inputs.get("run_id")
    try:
        metadata = financial_trace_metadata(context, run_id)
        tool = _safe_tool_name(inputs.get("tool"))
    except (TypeError, ValueError):
        metadata = dict(_TRACE_IDENTITY)
        tool = "financial_tool_unknown"
    arguments = inputs.get("arguments")
    return {
        **metadata, "operation": "mcp_dispatch", "tool": tool,
        "input_count": min(len(arguments), 1000) if isinstance(arguments, dict) else 0,
    }


def _financial_mcp_trace_outputs(outputs: Any) -> dict[str, Any]:
    event = outputs.safe_trace if isinstance(outputs, _SafeFinancialMCPResult) else {}
    allowed = {
        "stage", "stage_revision", "attempt", "agent", "implementation_version",
        "run_id", "ticker", "as_of", "target", "horizon", "analysis_mode",
        "dataset_version", "feature_version", "model_version", "schema_version",
        "boundary", "provider", "operation", "latency_ms", "outcome", "missing",
        "degraded", "input_count", "result_count", "error_code", "tool",
    }
    return {key: value for key, value in event.items() if key in allowed}


@traceable(
    name="financial.mcp_dispatch", run_type="tool", metadata=_TRACE_IDENTITY,
    process_inputs=_financial_mcp_trace_inputs,
    process_outputs=_financial_mcp_trace_outputs,
)
async def _financial_call_tool(
    session: ClientSession, tool: str, arguments: dict[str, Any], *,
    context: ForecastContext, run_id: str,
) -> _SafeFinancialMCPResult:
    """Call MCP without permitting arguments, results or exceptions into traces."""
    validated_context, identity = require_financial_context(context, run_id)
    safe_tool = _safe_tool_name(tool)
    started_at = time.perf_counter()
    outcome = "success"
    error_code: str | None = None
    missing = False
    degraded = False
    try:
        raw = await session.call_tool(safe_tool, arguments)
        try:
            compact_probe = json.loads(raw.content[0].text)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
            compact_probe = {}
        if isinstance(compact_probe, dict) and (
            compact_probe.get("error") or compact_probe.get("status") == "DATA_UNAVAILABLE"
        ):
            outcome = "unavailable"
            missing = True
            degraded = True
            candidate = str(compact_probe.get("error_code") or "")
            error_code = candidate if candidate in _MCP_ERROR_CODES else "FINANCIAL_TOOL_DATA_UNAVAILABLE"
    except Exception:
        outcome = "fail_closed"
        missing = True
        degraded = True
        error_code = "FINANCIAL_TOOL_FAIL_CLOSED"
        raw = SimpleNamespace(content=[SimpleNamespace(text=json.dumps({
            "error": "Financial tool failed closed.",
            "error_code": error_code,
        }))])
    safe_trace = financial_lifecycle_event(
        context=validated_context, run_id=identity, boundary="financial_mcp_ratio_dispatcher",
        started_at=started_at, outcome=outcome,
        counts={
            "input_count": len(arguments),
            "result_count": len(getattr(raw, "content", []) or []),
        },
        missing=missing, degraded=degraded, error_code=error_code,
    )
    safe_trace["tool"] = safe_tool
    return _SafeFinancialMCPResult(raw, safe_trace)

_MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[os.path.join(os.path.dirname(__file__), "..","tools","financial_tools" ,"financial_server.py")],
    env=None,   
)


class FinancialAnalystAgent:
    _extract_ticker = staticmethod(extract_ticker)
    
    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        max_loops: int = 3,
        mcp_server_params: StdioServerParameters | None = None,
        llm_client: anthropic.Anthropic | None = None,
    ) -> None:
        
        self._llm    = llm_client or anthropic.Anthropic()
        self._model  = model
        self._default_max_loops = max_loops
        self._server_params = mcp_server_params or _MCP_SERVER_PARAMS

        
        self._session: ClientSession | None = None

       
        self._graph = self._build_graph()

        log.info("FINANCIAL_AGENT_INITIALIZED")

   

    async def _execute_sec_filing_chain(
        self, session: ClientSession, state: FinancialAgentState,
        ticker: str, form_type: str, errors: list[str],
    ) -> None:
        """Run CIK -> filing list -> filing text using only returned identifiers."""
        raw = state["raw_numerical_data"]
        context, run_id = _financial_identity(state)
        session_id = session_from_directives(state["manager_directives"])
        workflow: dict[str, Any] = {"form_type": form_type, "status": "started"}
        raw["sec_filing"] = workflow

        async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
            arguments = dict(arguments)
            arguments.update(forecast_context=context.to_dict(), run_id=run_id)
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: calling tool '{tool}'...",
                detail={"tool": tool, "form_type": form_type},
            )
            try:
                add_breadcrumb(
                    "mcp.financial", f"Calling {tool}",
                    {"tool": tool, "ticker": ticker},
                )
                result = await _financial_call_tool(
                    session, tool, arguments, context=context, run_id=run_id,
                )
                data = json.loads(result.content[0].text)
                if data.get("error"):
                    errors.append("FINANCIAL_TOOL_DATA_UNAVAILABLE")
                    return None
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' completed",
                    detail={"tool": tool},
                )
                return data
            except Exception:
                errors.append("FINANCIAL_TOOL_FAIL_CLOSED")
                log.error("FINANCIAL_TOOL_FAIL_CLOSED tool=%s", tool)
                return None

        cik_data = await call("tool_get_cik", {"ticker": ticker})
        cik = cik_data.get("cik") if cik_data else None
        if not cik:
            workflow.update(status="unavailable", error="CIK was not returned; downstream SEC steps skipped.")
            return
        workflow["cik"] = cik

        filings_data = await call(
            "tool_list_filings", {"ticker": ticker, "form_type": form_type, "limit": 5},
        )
        filings = filings_data.get("filings", []) if filings_data else []
        filing_candidates = [
            item for item in filings
            if isinstance(item, dict) and item.get("accession_number")
        ]
        if not filing_candidates:
            workflow.update(status="unavailable", error="No real accession number was returned; filing text skipped.")
            return

        workflow["attempted_filings"] = []
        for filing in filing_candidates:
            accession = filing["accession_number"]
            workflow["attempted_filings"].append(accession)
            text_data = await call(
                "tool_get_filing_text",
                {
                    "accession_number": accession,
                    "cik": cik,
                    "primary_document": filing.get("primary_document"),
                    "form_type": form_type,
                    "filed_at": filing.get("filed_at") or filing.get("filing_date"),
                },
            )
            if text_data is None or text_data.get("validation_status") != "VALID":
                continue
            workflow.update(
                status="complete",
                selected_filing=dict(filing),
                filing_text=text_data,
                validation_status="VALID",
            )
            return

        workflow.update(
            status="unavailable",
            validation_status="INVALID",
            error="No filing candidate passed primary-document validation.",
        )

    @traceable(name="financial.extract", run_type="tool", metadata=_TRACE_IDENTITY,
               process_inputs=_trace_inputs("financial_extract"),
               process_outputs=_trace_outputs("financial_extract"))
    async def _execute_data_extraction(
        self,
        session: ClientSession,
        state:   FinancialAgentState,
    ) -> None:
        
        directives = state["manager_directives"]
        context, run_id = _financial_identity(state)
        task_query = state["task_query"]
        ticker     = extract_ticker(task_query, directives)

        if not ticker:
            state["raw_numerical_data"]["extraction_errors"] = [
                "Could not determine ticker symbol from task_query or manager_directives."
            ]
            log.error("FINANCIAL_TICKER_EXTRACTION_FAILED")
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
        log.info("FINANCIAL_EXTRACTION_PLAN tool_count=%d", len(extraction_tools))
        feedback = (state.get("validation_feedback") or "").lower()
        pending: list[str] = []
        for tool in extraction_tools:
            _, store_key = EXTRACTION_REGISTRY[tool]
            cached = raw.get(store_key)
            already_good = isinstance(cached, dict) and bool(cached) and not cached.get("error")
            if already_good and tool.lower() not in feedback:
                log.info("Executor: reusing cached %s from a previous loop.", tool)
                continue
            pending.append(tool)

        if not pending:
            log.info("Executor: all extraction tools served from cache — no calls made.")

        async def run_tool(tool: str) -> None:
            """Call one extraction tool and store its payload under raw[store_key]."""
            args_builder, store_key = EXTRACTION_REGISTRY[tool]
            args = args_builder(ticker, raw)

            if args is None:
                industry = raw.get("yahoo_ratios", {}).get("industry", "")
                log.info("FINANCIAL_EXTRACTION_SKIPPED tool=%s missing=True", tool)
                if tool == "tool_get_peer_comparison":
                    raw["peer_comparison"] = {
                        "note": f"No peer mapping available for industry={industry!r}.",
                    }
                return

            args = dict(args)
            args.update(forecast_context=context.to_dict(), run_id=run_id)

            log.info("Executor: calling %s for %s", tool, ticker)
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: calling tool '{tool}'...",
                detail={"tool": tool, **({"peer_count": len(args["peers"])} if "peers" in args else {})},
            )
            try:
                add_breadcrumb(
                    "mcp.financial", f"Calling {tool}",
                    {"tool": tool, "ticker": ticker},
                )
                result = await _financial_call_tool(
                    session, tool, args, context=context, run_id=run_id,
                )
                data = json.loads(result.content[0].text)
                if data.get("error") or data.get("status") == "DATA_UNAVAILABLE":
                    errors.append(str(data.get("error_code") or "FINANCIAL_TOOL_DATA_UNAVAILABLE"))
                raw[store_key] = (
                    {} if data.get("error_code") == "FINANCIAL_TOOL_FAIL_CLOSED"
                    else data
                )
            except Exception:
                errors.append("FINANCIAL_TOOL_FAIL_CLOSED")
                raw[store_key] = {}
                log.error("FINANCIAL_TOOL_FAIL_CLOSED tool=%s", tool)

            _publish_progress(
                session_id, "agent_tool_result", agent="financial",
                message=f"Financial Agent: '{tool}' completed",
                detail={"tool": tool},
            )

        peer_tool = "tool_get_peer_comparison"
        wave_one = [t for t in pending if t != peer_tool]
        if wave_one:
            await asyncio.gather(*(run_tool(t) for t in wave_one))
        if peer_tool in pending:
            await run_tool(peer_tool)

        if pending:
            _publish_progress(
                session_id, "agent_tool_result", agent="financial",
                message="Financial Agent: data extraction done",
                detail={"tools": len(pending), "errors": len(errors)},
            )

        form_type = select_sec_filing_type(state)
        prior_filing = raw.get("sec_filing", {})
        already_filed = (
            prior_filing.get("status") == "complete"
            and prior_filing.get("form_type") == form_type
        )
        if form_type is not None and not already_filed:
            await self._execute_sec_filing_chain(session, state, ticker, form_type, errors)
        elif already_filed:
            log.info("Executor: reusing %s filing chain result from a previous loop.", form_type)

        raw["extraction_errors"] = errors
        log.info(
            "Executor: data extraction complete — %d error(s) recorded.", len(errors)
        )

    @traceable(name="financial.compute", run_type="tool", metadata=_TRACE_IDENTITY,
               process_inputs=_trace_inputs("financial_compute"),
               process_outputs=_trace_outputs("financial_compute"))
    async def _execute_ratio_computation(
        self,
        session: ClientSession,
        state:   FinancialAgentState,
    ) -> None:
       
        raw        = state["raw_numerical_data"]
        xbrl_src   = raw.get("xbrl_financials", {})
        context, run_id = _financial_identity(state)
        session_id = session_from_directives(state["manager_directives"])

        
        async def _call(
            tool: str,
            args: dict,
            period: str | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> dict:
            _publish_progress(
                session_id, "agent_tool_call", agent="financial",
                message=f"Financial Agent: computing '{tool}'...",
                detail={"tool": tool},
            )
            try:
                add_breadcrumb("mcp.financial", f"Calling {tool}", {"tool": tool})
                result = await _financial_call_tool(
                    session, tool, args, context=context, run_id=run_id,
                )
                parsed = json.loads(result.content[0].text)
                parsed["_inputs"] = dict(args)
                parsed["_period"] = period
                default_state = (
                    "UNAVAILABLE" if parsed.get("error")
                    else "DEGRADED" if str(period or "").upper().startswith("MIXED")
                    else "VALID_PRIMARY"
                )
                parsed.setdefault("validation_status", default_state)
                parsed.setdefault("source", "calculated_from_validated_inputs")
                parsed.setdefault("exact_period", period)
                parsed.setdefault("fallback_reason", None)
                parsed.setdefault("confidence", {
                    "VALID_PRIMARY": "high",
                    "FALLBACK": "medium",
                    "DEGRADED": "low",
                    "UNAVAILABLE": "none",
                }[default_state])
                parsed.setdefault("is_approximation", False)
                if metadata:
                    parsed.update(metadata)
                if parsed.get("error"):
                    parsed["validation_status"] = "UNAVAILABLE"
                    parsed["confidence"] = "none"
                    parsed["fallback_reason"] = (
                        f"Metric calculation failed after compatible inputs were selected: "
                        f"{parsed.get('error')}"
                    )
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' computed",
                    detail={"tool": tool, "outcome": "success"},
                )
                return parsed
            except Exception:
                log.warning("FINANCIAL_CALCULATION_FAIL_CLOSED tool=%s", tool)
                _publish_progress(
                    session_id, "agent_tool_result", agent="financial",
                    message=f"Financial Agent: '{tool}' failed",
                    detail={"tool": tool, "outcome": "FAIL_CLOSED", "error_code": "FINANCIAL_CALCULATION_FAIL_CLOSED"},
                )
                failure = {
                    "error": "Financial calculation failed closed.",
                    "error_code": "FINANCIAL_CALCULATION_FAIL_CLOSED",
                    "_inputs": {},
                    "_period": period,
                    "validation_status": "UNAVAILABLE",
                    "source": "calculation_error",
                    "exact_period": period,
                    "formula": None,
                    "fallback_reason": "Financial calculation failed closed.",
                    "confidence": "none",
                    "is_approximation": False,
                }
                if metadata:
                    for key in ("source", "exact_period", "formula", "semantic_name"):
                        if key in metadata:
                            failure[key] = metadata[key]
                return failure

       
        state["calculated_ratios"]["earnings_quality"] = normalize_earnings_quality(xbrl_src)

        _, compute_tools = self._resolve_tools(state)
        log.info("FINANCIAL_COMPUTE_PLAN tool_count=%d", len(compute_tools))

        d = derive_compute_facts(
            raw, state["calculated_ratios"], depth=self._resolve_depth(state),
        )

        for tool in compute_tools:
            spec = COMPUTE_REGISTRY[tool]
            plan = spec.mapper(d)

            if plan is None:
                
                log.info("Compute: skipping %s — required inputs missing.", tool)
                unavailable = dict(spec.unavailable)
                unavailable.update({
                    "validation_status": "UNAVAILABLE",
                    "source": "unavailable",
                    "exact_period": None,
                    "formula": unavailable.get("formula"),
                    "fallback_reason": unavailable.get("error") or "Required compatible inputs were unavailable.",
                    "confidence": "none",
                    "is_approximation": False,
                })
                state["calculated_ratios"][spec.key] = unavailable
                continue

            if isinstance(plan, dict):
                plan.setdefault(
                    "validation_status",
                    "DEGRADED"
                    if str(plan.get("_period") or "").upper().startswith("MIXED")
                    else "VALID_PRIMARY",
                )
                plan.setdefault("source", "direct_structured_metric")
                plan.setdefault("exact_period", plan.get("_period"))
                plan.setdefault("formula", None)
                plan.setdefault("fallback_reason", None)
                plan.setdefault("confidence", "low" if plan["validation_status"] == "DEGRADED" else "high")
                plan.setdefault("is_approximation", False)
                state["calculated_ratios"][spec.key] = plan
                continue

            if len(plan) == 3:
                args, period, metadata = plan
            else:
                args, period = plan
                metadata = None
            log.info("FINANCIAL_CALCULATION_STARTED tool=%s", tool)
            result = await _call(tool, args, period=period, metadata=metadata)
            if spec.post is not None:
                spec.post(result, d)
            state["calculated_ratios"][spec.key] = result

        log.info("Executor: ratio computation complete.")

    @traceable(name="financial.checker", run_type="llm", metadata=_TRACE_IDENTITY,
               process_inputs=_trace_inputs("financial_checker"),
               process_outputs=_trace_outputs("financial_checker"))
    async def _check_data_quality(self, state: FinancialAgentState) -> dict[str, Any]:
        
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
        for metric_key in SUMMARY_CALCULATED_PASSTHROUGH:
            ratio_summary.setdefault(metric_key, calc.get(metric_key, {}))
        ratio_summary["peer_comparison"] = raw.get("peer_comparison", {})
        intent_requirements = checker_requirements(state)

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
                "intent_requirements": intent_requirements,
            }
        if not intent_requirements["passed"]:
            missing = intent_requirements["missing_groups"]
            log.warning(
                "FINANCIAL_CHECKER_COVERAGE_FAILED missing_count=%d", len(missing),
            )
            _publish_progress(
                session_from_directives(state["manager_directives"]),
                "agent_checker", agent="financial",
                message="Financial Agent: checking data quality — needs more data",
                detail={"is_complete": False, "missing_count": len(missing)},
            )
            return {
                "is_complete": False,
                "score": deterministic_validation["score"],
                "passed": [],
                "failed": ["INTENT-SPECIFIC COVERAGE"],
                "issues": [f"Missing required metric groups for active intents: {missing}"],
                "feedback": (
                    "Re-run the tools for these missing intent-specific metric groups: "
                    f"{missing}. Do not fetch unrelated analysis bundles."
                ),
                "deterministic_validation": deterministic_validation,
                "intent_requirements": intent_requirements,
            }

        user_content = (
            f"TASK QUERY:\n{task_query}\n\n"
            f"LOOP ITERATION: {loop}\n\n"
            f"RAW DATA SUMMARY:\n{json.dumps(raw_summary, indent=2)}\n\n"
            f"CALCULATED RATIOS:\n{json.dumps(ratio_summary, indent=2)}\n\n"
            f"ACTIVE INTENT COMPLETENESS CONTRACT:\n"
            f"{json.dumps(intent_requirements, indent=2)}\n\n"
            f"DETERMINISTIC FINANCIAL VALIDATION:\n{json.dumps(deterministic_validation, indent=2)}\n\n"
            "Audit only the active intent contract for completeness. Missing metrics "
            "outside that contract are irrelevant; every required any-of group must pass. "
            "Return your JSON verdict."
        )

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

            verdict_text = verdict_text.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(verdict_text)

        except json.JSONDecodeError:
            log.warning("FINANCIAL_CHECKER_PARSE_ERROR")
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["CHECKER PARSE ERROR"],
                "issues":      ["FINANCIAL_CHECKER_PARSE_ERROR"],
                "feedback":    "Checker could not parse its own verdict. "
                               "Re-run full extraction with tool_get_financial_ratios, "
                               "tool_get_revenue_growth, and tool_get_xbrl_financials.",
            }

        except Exception:
            log.error("FINANCIAL_CHECKER_API_ERROR")
            
            return {
                "is_complete": False,
                "score":       0,
                "passed":      [],
                "failed":      ["CHECKER API ERROR"],
                "issues":      ["FINANCIAL_CHECKER_API_ERROR"],
                "feedback":    "Checker API call failed. Retry full data extraction "
                               "on the next iteration.",
            }

       
        is_complete = bool(verdict.get("is_complete", False))
        score       = _bounded_score(verdict.get("score", 0))
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
            log.warning("FINANCIAL_CHECKER_FAIL score=%d failed_count=%d issue_count=%d", score, len(failed), len(issues))

        _publish_progress(
            session_from_directives(state["manager_directives"]), "agent_checker", agent="financial",
            message=(
                f"Financial Agent: checking data quality — "
                f"{'sufficient ✓' if is_complete else 'needs more data'}"
            ),
            detail={"is_complete": is_complete, "score": score, "failed_count": len(failed)},
        )

        return {
            "is_complete": is_complete,
            "score":       score,
            "passed":      passed,
            "failed":      failed,
            "issues":      issues,
            "feedback":    feedback,
            "deterministic_validation": deterministic_validation,
            "intent_requirements": intent_requirements,
        }

    # =========================================================================
    # LAYER 3 — BRAIN / ORCHESTRATOR (Lifecycle & Loop Gateway)
    # =========================================================================

    @traceable(name="financial.brain", run_type="llm", metadata=_TRACE_IDENTITY,
               process_inputs=_trace_inputs("financial_brain"),
               process_outputs=_trace_outputs("financial_brain"))
    async def _brain(self, state: FinancialAgentState) -> dict[str, Any]:
        
        iteration   = state["loop_counter"]
        feedback    = state["validation_feedback"]
        task_query  = state["task_query"]
        directives  = state["manager_directives"]

        

        user_message = (
            f"ITERATION: {iteration}\n"
            f"TASK QUERY: {task_query}\n"
            f"MANAGER DIRECTIVES: {json.dumps(directives)}\n"
            f"CHECKER FEEDBACK: {feedback if feedback else 'None — first iteration or prior run passed.'}\n\n"
            "Classify the analysis intents for this iteration."
        )

        
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
            log.info("FINANCIAL_BRAIN_COMPLETE iteration=%d intent_count=%d", iteration, len(result["intents"]))
            return result

        except json.JSONDecodeError:
            
            log.warning("Brain: Claude response was not valid JSON — defaulting to overview.")
            return {
                "intents": ["overview"], "extra_tools": [],
                "rationale": "Brain response was not valid JSON; defaulted to overview.",
                "raw_plan": raw_plan,
            }

        except Exception:
            log.error("FINANCIAL_BRAIN_API_ERROR iteration=%d", iteration)
            default_plan = "Default plan: overview bundle (extraction + core ratios)."
            state["messages"].append({"role": "assistant", "content": default_plan})
            return {
                "intents": ["overview"], "extra_tools": [],
                "rationale": "Brain API call failed; defaulted to overview.",
                "raw_plan": default_plan,
            }

    @staticmethod
    def _validate_brain_output(parsed: dict[str, Any]) -> dict[str, Any]:
        
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
        
     
        chosen = list(state.get("intents") or []) or ["overview"]

        if "full_report" in chosen:
            chosen = [i for i in chosen if i != "full_report"] + list(FULL_REPORT_PROFILES)

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

        if self._resolve_depth(state) == "shallow":
            compute -= MONTE_CARLO_TOOLS

        return (
            [t for t in EXTRACTION_ORDER if t in extraction],
            [t for t in COMPUTE_ORDER if t in compute],
        )

    @staticmethod
    def _resolve_depth(state: FinancialAgentState) -> str:
        """Normalized analysis depth for this iteration.

        Single source of truth for two consumers that must not diverge:
        _resolve_tools() drops the Monte-Carlo tools at "shallow", and
        derive_compute_facts() sizes the Monte-Carlo sample count from it.
        """
        depth = normalize_depth(state["manager_directives"])
        if "full_report" in (state.get("intents") or []) and depth != "shallow":
            return "deep"
        return depth

    # =========================================================================
    # LANGGRAPH NODES — thin wrappers over the layer methods above
    # =========================================================================
   

    async def _node_brain(self, state: FinancialAgentState) -> dict[str, Any]:
     
        new_counter = state["loop_counter"] + 1
        log.info("--- Loop iteration %d ---", new_counter)
        brain_output = await self._brain(state)
        intents = list(brain_output["intents"])
        if state.get("manager_directives", {}).get("monthly_feature_profile"):
            # The monthly model has a fixed feature contract. Intent routing
            # may add tools, but cannot omit required numerical families.
            for intent in (
                "valuation", "profitability", "growth",
                "financial_health", "liquidity", "cash_flow",
            ):
                if intent not in intents:
                    intents.append(intent)
        log.info("FINANCIAL_BRAIN_ROUTED intent_count=%d", len(intents))
       
        return {
            "loop_counter": new_counter,
            "intents":      intents,
            "extra_tools":  brain_output["extra_tools"],
            "rationale":    brain_output["rationale"],
        }

    async def _node_extract(self, state: FinancialAgentState) -> dict[str, Any]:
        started_at = time.perf_counter()
        await self._execute_data_extraction(self._session, state)
        context, run_id = _financial_identity(state)
        raw = state["raw_numerical_data"]
        errors = raw.get("extraction_errors") if isinstance(raw.get("extraction_errors"), list) else []
        financial_lifecycle_event(
            context=context, run_id=run_id, boundary="agent_extraction",
            started_at=started_at, outcome="degraded" if errors else "success",
            counts={"tool_count": len(raw), "error_count": len(errors)},
            missing=not bool(raw), degraded=bool(errors),
            error_code="FINANCIAL_TOOL_DATA_UNAVAILABLE" if errors else None,
        )
        return {"raw_numerical_data": state["raw_numerical_data"]}

    async def _node_compute(self, state: FinancialAgentState) -> dict[str, Any]:
        started_at = time.perf_counter()
        await self._execute_ratio_computation(self._session, state)
        context, run_id = _financial_identity(state)
        calculated = state["calculated_ratios"]
        missing_count = sum(
            1 for value in calculated.values()
            if isinstance(value, dict) and value.get("validation_status") == "UNAVAILABLE"
        )
        financial_lifecycle_event(
            context=context, run_id=run_id, boundary="agent_calculation",
            started_at=started_at, outcome="degraded" if missing_count else "success",
            counts={"result_count": len(calculated), "missing_count": missing_count},
            missing=bool(missing_count), degraded=bool(missing_count),
        )
        return {"calculated_ratios": state["calculated_ratios"]}

    async def _node_checker(self, state: FinancialAgentState) -> dict[str, Any]:
        started_at = time.perf_counter()
        check_result = await self._check_data_quality(state)
        is_complete = check_result["is_complete"]
        if is_complete:
            log.info("FINANCIAL_CHECKER_PASSED loop_count=%d", state["loop_counter"])
        else:
            log.warning("FINANCIAL_CHECKER_RETRY loop_count=%d", state["loop_counter"])
        context, run_id = _financial_identity(state)
        failed = check_result.get("failed") if isinstance(check_result.get("failed"), list) else []
        financial_lifecycle_event(
            context=context, run_id=run_id, boundary="agent_checker",
            started_at=started_at, outcome="success" if is_complete else "degraded",
            counts={
                "result_count": 1, "missing_count": len(failed),
                "loop_count": state["loop_counter"],
            },
            missing=bool(failed), degraded=not is_complete,
        )
        # Cached so run() doesn't recompute evaluate_financial_sanity() over
        # the same inputs after the graph finishes.
        return {
            "is_complete":         is_complete,
            "validation_feedback": check_result["feedback"],
            "financial_sanity":    check_result.get("deterministic_validation", {}),
        }

    def _should_continue(self, state: FinancialAgentState):
       
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

    @traceable(
        name="FinancialAnalystAgent.run", run_type="chain", metadata=_TRACE_IDENTITY,
        process_inputs=_trace_inputs("financial_run"),
        process_outputs=_trace_outputs("financial_run"),
    )
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        
        directives = dict(shared_state.get("manager_directives", {}))
        context, run_id = require_financial_context(
            shared_state.get("forecast_context"),
            directives.get("run_id") or shared_state.get("run_id"),
        )
        directive_ticker = str(directives.get("ticker") or context.ticker).upper()
        if directive_ticker != context.ticker:
            raise ValueError("manager ticker must match ForecastContext.ticker")
        directives.update(
            ticker=context.ticker, forecast_context=context.to_dict(), run_id=run_id,
            monthly_feature_profile=True,
        )
        run_started_at = time.perf_counter()
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        log.info("FINANCIAL_RUN_STARTED max_loops=%d", max_loops)

        # -- Initialise agent-local state -------------------------------------
        state: FinancialAgentState = {
            "messages":            [],
            "raw_numerical_data":  {},
            "calculated_ratios":   {},
            "loop_counter":        0,
            "validation_feedback": "",
            "is_complete":         False,
            "task_query":          shared_state["task_query"],
            "manager_directives":  directives,
            
            "intents":             [],
            "extra_tools":         [],
            "rationale":           "",
            "financial_sanity":    {},
        }

   
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                await session.initialize()
                log.info("MCP session initialised successfully.")

                
                self._session = session

              
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

        
        raw  = state["raw_numerical_data"]
        calc = state["calculated_ratios"]

       
        pre_state_started_at = time.perf_counter()
        annotate_confidence(calc)
        features = build_monthly_financial_features(
            raw=raw, calculated=calc, forecast_context=context, run_id=run_id,
        )
        feature_slot = features.to_state_slot()

        financial_lifecycle_event(
            context=context, run_id=run_id, boundary="pre_shared_state_gate",
            started_at=pre_state_started_at,
            outcome="degraded" if feature_slot["degraded"] else "success",
            counts={
                "result_count": len(feature_slot["values"]),
                "evidence_count": len(feature_slot["evidence_ids"]),
                "missing_count": len(feature_slot["missing_flags"]),
            },
            missing=feature_slot["missing"], degraded=feature_slot["degraded"],
        )

        # Shared State excludes raw Yahoo/SEC payloads, filing text, checker
        # prose and attempt history. It retains a compact typed slot plus the
        # calculated compatibility view consumed by legacy Manager paths.
        yahoo = raw.get("yahoo_ratios", {})
        xbrl = raw.get("xbrl_financials", {})
        financial_sanity = state.get("financial_sanity") or evaluate_financial_sanity({
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "reverse_dcf": calc.get("reverse_dcf", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
        })
        current_ratio_metric = calc.get("current_ratio_calc", {})
        if not current_ratio_metric and yahoo.get("current_ratio") is not None:
            current_ratio_metric = {
                "current_ratio": yahoo.get("current_ratio"),
                "validation_status": "FALLBACK",
                "source": "yahoo_finance.currentRatio",
                "confidence": "medium",
                "is_approximation": False,
            }

        financial_summary = {
            "ticker": context.ticker,
            "financial_features": feature_slot["values"],
            "evidence_ids": feature_slot["evidence_ids"],
            "missing": feature_slot["missing"], "degraded": feature_slot["degraded"],
            "missing_flags": feature_slot["missing_flags"],
            "degraded_flags": feature_slot["degraded_flags"],
            "validation_passed": bool(state["is_complete"]),
            "loop_iterations_used": int(state["loop_counter"]),
            "analysis_intents": sorted(set(state.get("intents") or [])),
            "implementation": _TRACE_IDENTITY,
            "latency_ms": elapsed_latency_ms(run_started_at),
            "company_name": yahoo.get("company_name"),
            "sector": yahoo.get("sector"),
            "industry": yahoo.get("industry"),
            "current_price": yahoo.get("current_price"),
            "market_cap": yahoo.get("market_cap"),
            "beta": yahoo.get("beta"),
            "52w_high": yahoo.get("52w_high"),
            "52w_low": yahoo.get("52w_low"),
            "forward_pe": yahoo.get("forward_pe"),
            "peg_ratio": yahoo.get("peg_ratio"),
            "pe_ratio": calc.get("pe", {}),
            "roe": calc.get("roe", {}),
            "net_margin": calc.get("net_margin", {}),
            "de_ratio": calc.get("de_ratio", {}),
            "revenue_cagr": calc.get("cagr", {}),
            "forecast": calc.get("forecast", {}),
            "dcf_valuation": calc.get("dcf", {}),
            "valuation_input_assessments": (
                calc.get("dcf", {}).get("base", {}).get("input_assessments", {})
            ),
            "dcf_monte_carlo": calc.get("dcf_monte_carlo", {}),
            "current_ratio": current_ratio_metric,
            "composite_score": calc.get("composite_score", {}),
            "earnings_quality": calc.get("earnings_quality", {}),
            "net_debt_to_ebitda": calc.get("net_debt_to_ebitda", {}),
            "fcf_margin": calc.get("fcf_margin", {}),
            "fcf_yield": calc.get("fcf_yield", {}),
            "ocf_ratio": calc.get("ocf_ratio", {}),
            "reverse_dcf": calc.get("reverse_dcf", {}),
            "roic": calc.get("roic", {}),
            "roce": calc.get("roce", {}),
            "debt_ratio": calc.get("debt_ratio", {}),
            "equity_ratio": calc.get("equity_ratio", {}),
            "altman_z_score": calc.get("altman_z_score", {}),
            "peer_comparison": raw.get("peer_comparison", {}),
            "capital_allocation": {
                "operating_cash_flow": xbrl.get("operating_cash_flow_annual", [])[:1],
                "capex": xbrl.get("capex_annual", [])[:1],
                "dividends_paid": xbrl.get("dividends_paid_annual", [])[:1],
                "buybacks": xbrl.get("buybacks_annual", [])[:1],
            },
            "consistency_checks": build_consistency_checks(raw, calc),
            "data_quality": build_data_quality_score(raw, calc),
            "financial_sanity": financial_sanity,
            "financial_facts": [],
            "extraction_errors": list(raw.get("extraction_errors", [])),
        }


        shared_state["financial_features"] = feature_slot  # type: ignore[literal-required]
        shared_state["financial_metrics_summary"] = sanitize_nans(  # type: ignore[literal-required]
            financial_summary
        )

        log.info(
            "FINANCIAL_RUN_COMPLETE evidence_count=%d missing=%s degraded=%s",
            len(feature_slot["evidence_ids"]), feature_slot["missing"], feature_slot["degraded"],
        )

        return shared_state
