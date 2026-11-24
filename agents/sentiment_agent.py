from __future__ import annotations

import json
import logging
import asyncio
import time
from types import SimpleNamespace
from urllib.parse import urlparse
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.prompts.prompt_sentiment import (BRAIN_PLAN_SYSTEM_PROMPT, BRAIN_ANALYZE_SYSTEM_PROMPT)
from evaluation.aspect_sentiment import extract_aspect_observations, supported_causal_aspects
from agents.state import SentimentAgentState, SharedManagerState
from agents.executor_helpers.finance_executor_helper import extract_ticker
from langsmith import traceable
from core.forecast_contract import ForecastContext
from core.observability import forecast_trace_metadata
from core.progress_bus import (
    publish as _publish_progress,
    publish_forecast as _publish_forecast,
    session_from_directives,
)
from core.error_handler import capture_exception, add_breadcrumb
from agents.executor_helpers.sentiment_helper import (
    build_monthly_sentiment_features,
    canonical_sentiment_model_status,
    canonical_sentiment_polarity,
    causal_audit,
    compact_sentiment_summary,
    attempt_rank,
    require_sentiment_context,
    SENTIMENT_IMPLEMENTATION_IDENTITY,
    sentiment_lifecycle_event,
    valid_composite_score,
)
from tools.sentiment_tools import (
    FEAR_GREED_MODEL_VERSION,
    FINBERT_MODEL_VERSION,
    VADER_MODEL_VERSION,
)

log = logging.getLogger("sentiment-agent")

_TRACE_IDENTITY = dict(SENTIMENT_IMPLEMENTATION_IDENTITY)
_SENTIMENT_TOOLS = frozenset({
    "retrieve_social_data", "analyze_finbert", "score_vader", "calculate_fear_greed",
})
_TOOL_BOUNDARIES = {
    "retrieve_social_data": "point_in_time_retrieval",
    "analyze_finbert": "finbert_inference",
    "score_vader": "vader_scoring",
    "calculate_fear_greed": "fear_greed_fusion",
}
_TOOL_ERROR_CODES = frozenset({
    "SENTIMENT_TOOL_FAIL_CLOSED", "SENTIMENT_TOOL_DATA_UNAVAILABLE",
})


def _safe_error_type(exc: BaseException) -> str:
    candidate = type(exc).__name__
    return candidate if candidate in {
        "RuntimeError", "TimeoutError", "ConnectionError", "ValueError",
        "LookupError", "JSONDecodeError",
    } else "Exception"


def _semantic_result_count(tool: str, payload: dict[str, Any]) -> int:
    def count(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(result, 1_000_000))

    if tool == "retrieve_social_data":
        return count(
            payload.get("unique_document_count")
            or len(payload.get("chunks", []) or [])
        )
    if tool in {"analyze_finbert", "score_vader"}:
        return count(payload.get("total_chunks"))
    if tool == "calculate_fear_greed":
        return sum(payload.get(key) is not None for key in ("finbert_score", "vader_score"))
    return 0


def _canonicalize_provider_payload(
    tool: str, payload: dict[str, Any],
) -> dict[str, Any]:
    """Remove untrusted model status/polarity before any executor State assignment."""
    safe = dict(payload)
    if tool not in {"analyze_finbert", "score_vader", "calculate_fear_greed"}:
        return safe
    if tool in {"analyze_finbert", "score_vader"}:
        available = _semantic_result_count(tool, safe) > 0
    else:
        available = valid_composite_score(safe.get("score"))
    status = canonical_sentiment_model_status(safe, available=available)
    safe["status"] = status
    if tool == "calculate_fear_greed":
        polarity, polarity_invalid = canonical_sentiment_polarity(safe.get("polarity"))
        safe["polarity"] = polarity
        safe["polarity_invalid"] = polarity_invalid
    if status == "UNKNOWN":
        if tool in {"analyze_finbert", "score_vader"}:
            safe["chunk_scores"] = []
            safe["total_chunks"] = 0
        else:
            safe.update({
                "score": None, "confidence": 0.0, "label": None,
                "polarity": None,
            })
    return safe


def _publish_lifecycle_progress(
    session_id: str | None, *, context: ForecastContext, run_id: str,
    boundary: str, started_at: float, outcome: str,
    evidence_count: int = 0, degraded_flags: list[str] | None = None,
) -> None:
    """Map internal sentiment boundaries to UI events without false completion."""
    tool_by_boundary = {value: key for key, value in _TOOL_BOUNDARIES.items()}
    safe_detail = {
        "outcome": outcome,
        "evidence_count": max(0, int(evidence_count)),
        "degraded_flags": list(degraded_flags or []),
    }
    if boundary in {"brain_plan", "brain_analyze"}:
        _publish_progress(
            session_id, "agent_brain", agent="sentiment",
            message=f"Sentiment Agent: {boundary.replace('_', ' ')} {outcome}",
            detail={"boundary": boundary, **safe_detail}, run_id=run_id,
        )
        return
    if boundary in tool_by_boundary:
        tool = tool_by_boundary[boundary]
        _publish_progress(
            session_id, "agent_tool_result", agent="sentiment",
            message=f"Sentiment Agent: '{tool}' {outcome}",
            detail={"tool": tool, **safe_detail}, run_id=run_id,
        )
        return
    if boundary == "whole_sentiment_run":
        _publish_progress(
            session_id, "dispatch_end", agent="sentiment",
            message=f"Sentiment Agent: run {outcome}",
            detail=safe_detail, run_id=run_id,
        )
        return
    event_type = (
        "feature_complete" if outcome == "success"
        else "forecast_error" if outcome == "fail_closed"
        else "forecast_degraded"
    )
    _publish_forecast(
        session_id, event_type, run_id=run_id, context=context,
        stage=str(_TRACE_IDENTITY["stage"]),
        stage_revision=int(_TRACE_IDENTITY["stage_revision"]),
        attempt=int(_TRACE_IDENTITY["attempt"]),
        agent=str(_TRACE_IDENTITY["agent"]),
        message=f"SENTIMENT_{boundary.upper()}", started_at=started_at,
        evidence_count=max(0, int(evidence_count)),
        degraded_flags=degraded_flags or [],
    )


def _trace_inputs(operation: str):
    def process(inputs: dict[str, Any]) -> dict[str, Any]:
        state = inputs.get("shared_state") or inputs.get("state") or {}
        directives = state.get("manager_directives", {}) if isinstance(state, dict) else {}
        try:
            context, run_id = require_sentiment_context(
                state.get("forecast_context") or directives.get("forecast_context"),
                state.get("run_id") or directives.get("run_id"),
            )
            metadata = forecast_trace_metadata(
                context, run_id=run_id,
                stage=str(_TRACE_IDENTITY["stage"]),
                stage_revision=int(_TRACE_IDENTITY["stage_revision"]),
                attempt=int(_TRACE_IDENTITY["attempt"]),
                agent=str(_TRACE_IDENTITY["agent"]),
            )
        except (TypeError, ValueError):
            metadata = dict(_TRACE_IDENTITY)
        return {
            "operation": operation, "metadata": metadata,
            "loop_counter": state.get("loop_counter", 0) if isinstance(state, dict) else 0,
        }
    return process


def _trace_outputs(operation: str):
    def process(outputs: Any) -> dict[str, Any]:
        payload = outputs if isinstance(outputs, dict) else {}
        features = payload.get("sentiment_features") if isinstance(payload.get("sentiment_features"), dict) else {}
        evidence = features.get("evidence_ids") if isinstance(features.get("evidence_ids"), list) else []
        return {
            "operation": operation, "metadata": dict(_TRACE_IDENTITY),
            "evidence_count": len(evidence), "missing": bool(features.get("missing", False)),
            "degraded": bool(features.get("degraded", False)),
        }
    return process


class _SafeSentimentMCPResult:
    __slots__ = ("_raw", "safe_trace")

    def __init__(self, raw: Any, safe_trace: dict[str, Any]) -> None:
        self._raw = raw
        self.safe_trace = safe_trace

    @property
    def content(self) -> Any:
        return self._raw.content


def _mcp_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    context = inputs.get("context")
    run_id = inputs.get("run_id")
    tool = str(inputs.get("tool") or "")
    metadata = forecast_trace_metadata(
        context, run_id=run_id,
        stage=str(_TRACE_IDENTITY["stage"]),
        stage_revision=int(_TRACE_IDENTITY["stage_revision"]),
        attempt=int(_TRACE_IDENTITY["attempt"]),
        agent=str(_TRACE_IDENTITY["agent"]),
    )
    arguments = inputs.get("arguments")
    return {
        **metadata, "operation": "mcp_dispatch",
        "tool": tool if tool in _SENTIMENT_TOOLS else "sentiment_tool_unknown",
        "input_count": len(arguments) if isinstance(arguments, dict) else 0,
    }


def _mcp_trace_outputs(outputs: Any) -> dict[str, Any]:
    event = outputs.safe_trace if isinstance(outputs, _SafeSentimentMCPResult) else {}
    allowed = {
        "stage", "stage_revision", "attempt", "agent", "implementation_version",
        "run_id", "ticker", "as_of", "target", "horizon", "analysis_mode",
        "dataset_version", "feature_version", "model_version", "schema_version",
        "boundary", "provider", "operation", "latency_ms", "outcome", "missing",
        "degraded", "input_count", "result_count", "evidence_count", "error_code",
        "error_type", "tool", "degraded_flags", "sentiment_model_versions",
    }
    return {key: value for key, value in event.items() if key in allowed}


@traceable(
    name="sentiment.mcp_dispatch", run_type="tool", metadata=_TRACE_IDENTITY,
    process_inputs=_mcp_trace_inputs, process_outputs=_mcp_trace_outputs,
)
async def _sentiment_call_tool(
    session: ClientSession, tool: str, arguments: dict[str, Any], *,
    context: ForecastContext, run_id: str,
) -> _SafeSentimentMCPResult:
    if tool not in _SENTIMENT_TOOLS:
        raise ValueError("unknown Stage 06 sentiment tool")
    validated_context, identity = require_sentiment_context(context, run_id)
    started_at = time.perf_counter()
    outcome = "success"
    error_code: str | None = None
    error_type: str | None = None
    missing = False
    degraded = False
    try:
        raw = await session.call_tool(tool, arguments)
        try:
            payload = json.loads(raw.content[0].text)
            if not isinstance(payload, dict):
                raise ValueError("tool payload must be an object")
        except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            error_type = _safe_error_type(exc)
            outcome, missing, degraded = "fail_closed", True, True
            error_code = "SENTIMENT_TOOL_FAIL_CLOSED"
            payload = {"error": "Sentiment tool failed closed.", "error_code": error_code}
            raw = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])
        else:
            payload = _canonicalize_provider_payload(tool, payload)
            raw = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])
            if payload.get("error") or str(payload.get("status") or "").upper() in {
                "ERROR", "FAILED", "NO_DATA", "DATA_UNAVAILABLE",
                "UNAVAILABLE", "MISSING", "UNKNOWN",
            }:
                outcome, missing, degraded = "unavailable", True, True
                candidate = str(payload.get("error_code") or "")
                error_code = candidate if candidate in _TOOL_ERROR_CODES else "SENTIMENT_TOOL_DATA_UNAVAILABLE"
    except Exception as exc:
        error_type = _safe_error_type(exc)
        outcome, missing, degraded = "fail_closed", True, True
        error_code = "SENTIMENT_TOOL_FAIL_CLOSED"
        payload = {"error": "Sentiment tool failed closed.", "error_code": error_code}
        raw = SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])
    semantic_count = _semantic_result_count(tool, payload)
    model_versions: dict[str, str] = {}
    if outcome == "success" and tool == "analyze_finbert":
        model_versions["finbert"] = FINBERT_MODEL_VERSION
    elif outcome == "success" and tool == "score_vader":
        model_versions["vader"] = VADER_MODEL_VERSION
    elif outcome == "success" and tool == "calculate_fear_greed":
        model_versions["fusion"] = FEAR_GREED_MODEL_VERSION
    safe_trace = sentiment_lifecycle_event(
        forecast_context=validated_context, run_id=identity,
        boundary=_TOOL_BOUNDARIES[tool], started_at=started_at, outcome=outcome,
        counts={
            "input_count": len(arguments), "result_count": semantic_count,
            "evidence_count": semantic_count,
        },
        missing=missing, degraded=degraded,
        degraded_flags=["tool_unavailable"] if degraded else [],
        error_code=error_code, error_type=error_type,
        model_versions=model_versions,
    )
    safe_trace["tool"] = tool
    return _SafeSentimentMCPResult(raw, safe_trace)


# ══════════════════════════════════════════════════════════════════════════════
# SentimentAgent
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAgent:

    _MIN_UNIQUE_DOCUMENTS = 3
    _MIN_UNIQUE_DOMAINS = 2

    @staticmethod
    def _extract_ticker(
        task_query: str,
        directives: dict[str, Any],
        financial_summary: dict[str, Any],
    ) -> str | None:
        """Backward-compatible wrapper around the shared ticker resolver."""
        return extract_ticker(task_query, directives, financial_summary)

    @staticmethod
    def _causal_audit(
        analysis: dict[str, Any],
        supported: list[str],
    ) -> dict[str, Any]:
        """Backward-compatible wrapper around the sentiment audit helper."""
        return causal_audit(analysis, supported)

    @classmethod
    def _select_best_attempt(cls, state: SentimentAgentState) -> dict[str, Any] | None:
        attempts = state.get("attempts", [])
        if not attempts:
            return None
        selected = max(attempts, key=attempt_rank)
        retrieval = selected.get("retrieval", {}) or {}
        outputs = selected.get("model_outputs", {}) or {}
        state["retrieved_chunks"] = list(retrieval.get("chunks", []))
        state["sources_metadata"] = list(retrieval.get("sources_metadata", []))
        state["finbert_result"] = dict(outputs.get("finbert", {}))
        state["vader_result"] = dict(outputs.get("vader", {}))
        state["fear_greed_result"] = dict(outputs.get("fear_greed", {}))
        state["extraction_errors"] = list(selected.get("errors", []))
        state["sentiment_status"] = str(selected.get("status", "NO_DATA"))
        state["sentiment_polarity"] = selected.get("polarity")
        state["fear_greed_result"]["retrieval_stats"] = dict(
            selected.get("retrieval_stats", {})
        )
        state["fear_greed_result"]["selected_attempt_number"] = selected.get(
            "attempt_number"
        )
        return selected

    def __init__(
        self,
        server_script_path: str,
        model: str = "claude-haiku-4-5",
        max_loops: int = 2,
        llm_client: anthropic.Anthropic | None = None,
        mcp_server_params: StdioServerParameters | None = None,
    ) -> None:
        self._llm               = llm_client or anthropic.Anthropic()
        self._model             = model
        self._default_max_loops = max_loops
        self._server_params     = mcp_server_params or StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None,
        )

        self._session: ClientSession | None = None
        self._graph = self._build_graph()

        log.info(
            "SentimentAgent initialised — model=%s, server=%s",
            model, server_script_path,
        )

    # =========================================================================
    # LAYER 2 — BRAIN (Cognitive / LLM Interpretation Layer)
    # =========================================================================

    @traceable(name="sentiment.brain_plan", run_type="llm",
               process_inputs=_trace_inputs("brain_plan"), process_outputs=_trace_outputs("brain_plan"))
    async def _brain_plan(self, state: SentimentAgentState) -> dict[str, Any]:

        started_at = time.perf_counter()
        task_query = state.get("task_query", "")
        directives = state.get("manager_directives",  {})
        fin_summary = state.get("financial_metrics_summary",{})
        research_ctx = state.get("aggregated_research_context", [])
        forecast_context = state.get("forecast_context", {})
        context, run_id = require_sentiment_context(
            forecast_context or directives.get("forecast_context"),
            state.get("run_id") or directives.get("run_id"),
        )
        session_id = session_from_directives(directives)
        ticker = extract_ticker(task_query, directives, fin_summary)

        fin_ctx_snippet = ""
        if fin_summary:
            fin_ctx_snippet = (
                f"Ticker: {fin_summary.get('ticker')} | "
                f"Sector: {fin_summary.get('sector')} | "
                f"Composite score: {fin_summary.get('composite_score', {}).get('score')} | "
                f"Grade: {fin_summary.get('composite_score', {}).get('grade')}"
            )

        research_sample = " ".join(research_ctx)[:800] if research_ctx else "None"

        user_content = (
            f"LOOP ITERATION: {state['loop_counter']}\n"
            f"TASK QUERY: {task_query}\n"
            f"TICKER: {ticker or 'unknown'}\n"
            f"MANAGER DIRECTIVES: {json.dumps(directives)}\n"
            f"FORECAST CONTEXT: {json.dumps(forecast_context)}\n"
            f"FINANCIAL CONTEXT: {fin_ctx_snippet or 'Not available'}\n"
            f"RESEARCH CONTEXT SAMPLE: {research_sample}\n\n"
            "Produce the retrieval plan for this iteration."
        )

        state["messages"].append({"role": "user", "content": user_content})
        log.info("[Brain-Plan] Invoking Claude for retrieval planning (iteration %d)...",
                 state["loop_counter"])

        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=768,
                temperature=0,
                system=BRAIN_PLAN_SYSTEM_PROMPT,
                messages=state["messages"],
            )
            raw = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw})

            raw = raw.replace("```json", "").replace("```", "").strip()
            plan = json.loads(raw)
            plan["days_back"] = 30
            log.info("[Brain-Plan] SENTIMENT_PLAN_READY ticker=%s days_back=%s",
                     context.ticker, plan.get("days_back"))
            sentiment_lifecycle_event(
                forecast_context=context, run_id=run_id, boundary="brain_plan",
                started_at=started_at, outcome="success",
                counts={"result_count": 1, "evidence_count": 0},
            )
            _publish_lifecycle_progress(
                session_id, context=context, run_id=run_id, boundary="brain_plan",
                started_at=started_at, outcome="success",
            )
            return plan

        except Exception as exc:
            log.warning(
                "[Brain-Plan] planning failed; using defaults error_type=%s",
                type(exc).__name__,
            )
            capture_exception(
                "sentiment.brain_plan",
                RuntimeError("SENTIMENT_BRAIN_PLAN_FAILED"),
                tags={"iteration": str(state["loop_counter"])},
            )
            sentiment_lifecycle_event(
                forecast_context=context, run_id=run_id, boundary="brain_plan",
                started_at=started_at, outcome="degraded",
                counts={"result_count": 1, "evidence_count": 0},
                degraded=True, degraded_flags=["planning_degraded"],
                error_code="SENTIMENT_BRAIN_PLAN_FAILED",
                error_type=_safe_error_type(exc),
            )
            _publish_lifecycle_progress(
                session_id, context=context, run_id=run_id, boundary="brain_plan",
                started_at=started_at, outcome="degraded",
                degraded_flags=["planning_degraded"],
            )
            default_plan = {
                "retrieval_query": f"{ticker or 'market'} earnings sentiment investor reaction",
                "ticker":          ticker,
                "days_back":       30,
                "reasoning":       "Default plan: Brain planning failed; using heuristic query.",
            }
            state["messages"].append({"role": "assistant","content": json.dumps(default_plan)})
            return default_plan

    @traceable(name="sentiment.brain_analyze", run_type="llm",
               process_inputs=_trace_inputs("brain_analyze"), process_outputs=_trace_outputs("brain_analyze"))
    async def _brain_analyze(self, state: SentimentAgentState) -> str:

        started_at = time.perf_counter()
        directives = state.get("manager_directives", {})
        context, run_id = require_sentiment_context(
            state.get("forecast_context") or directives.get("forecast_context"),
            state.get("run_id") or directives.get("run_id"),
        )
        session_id = session_from_directives(directives)
        task_query  = state.get("task_query","")
        fin_summary = state.get("financial_metrics_summary", {})
        fg          = state["fear_greed_result"]
        fb          = state["finbert_result"]
        vd          = state["vader_result"]
        status      = state.get("sentiment_status", "NO_DATA")

        if status in {"NO_DATA", "ERROR"}:
            sentiment_lifecycle_event(
                forecast_context=context, run_id=run_id, boundary="brain_analyze",
                started_at=started_at, outcome="unavailable",
                counts={"evidence_count": len(state.get("retrieved_chunks", [])), "result_count": 0},
                missing=True, degraded=True,
                degraded_flags=["analysis_unavailable"],
            )
            _publish_lifecycle_progress(
                session_id, context=context, run_id=run_id, boundary="brain_analyze",
                started_at=started_at, outcome="unavailable",
                evidence_count=len(state.get("retrieved_chunks", [])),
                degraded_flags=["analysis_unavailable"],
            )
            return json.dumps({
                "status": status, "overall_sentiment": None,
                "conviction_level": "Low", "key_signals": [],
                "model_agreement": "Unavailable",
                "narrative": "Sentiment interpretation unavailable because no valid evidence was selected.",
                "risk_flags": ["No valid sentiment evidence"],
                "data_quality_note": "Missing input is not neutral sentiment.",
            })

        fin_ctx = ""
        if fin_summary:
            fin_ctx = json.dumps({
                "ticker":          fin_summary.get("ticker"),
                "sector":          fin_summary.get("sector"),
                "pe_ratio":        fin_summary.get("pe_ratio", {}).get("pe_ratio"),
                "composite_score": fin_summary.get("composite_score", {}).get("score"),
                "grade":           fin_summary.get("composite_score", {}).get("grade"),
                "revenue_cagr":    fin_summary.get("revenue_cagr", {}).get("cagr_pct"),
            }, indent=2)

        user_content = (
            f"TASK QUERY: {task_query}\n"
            f"TICKER: {fg.get('diagnostics', {}).get('ticker') or fin_summary.get('ticker', 'unknown')}\n\n"
            f"FEAR/GREED RESULT:\n{json.dumps(fg, indent=2)}\n\n"
            f"FINBERT RESULT:\n{json.dumps({k: v for k, v in fb.items() if k != 'chunk_scores'}, indent=2)}\n\n"
            f"VADER RESULT:\n{json.dumps({k: v for k, v in vd.items() if k != 'chunk_scores'}, indent=2)}\n\n"
            f"CHUNK COUNT ANALYSED: {len(state['retrieved_chunks'])}\n\n"
            f"ASPECT OBSERVATIONS:\n{json.dumps(state.get('aspect_observations', []), indent=2)}\n\n"
            "CAUSAL EVIDENCE CONTRACT: Present an aspect as a cause of sentiment ONLY when "
            "that exact observation has causal_language_supported=True. An observation with "
            "causal_language_supported=False or a missing flag may only be described as "
            "'mentioned in evidence'; it must not be described as driving, causing, weighing "
            "on, supporting, or explaining sentiment. List every aspect used causally in the "
            "causal_aspects_used output field.\n\n"
            f"FINANCIAL CONTEXT:\n{fin_ctx or 'Not available'}\n\n"
            "Produce your sentiment interpretation JSON verdict."
        )

        state["messages"].append({"role": "user", "content": user_content})
        log.info("[Brain-Analyze] Invoking Claude for sentiment interpretation...")

        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=768,
                temperature=0,
                system=BRAIN_ANALYZE_SYSTEM_PROMPT,
                messages=state["messages"],
            )
            raw = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw})
            raw = raw.replace("```json", "").replace("```", "").strip()
            log.info("[Brain-Analyze] Interpretation complete.")
            evidence_count = len(state.get("retrieved_chunks", []))
            sentiment_lifecycle_event(
                forecast_context=context, run_id=run_id, boundary="brain_analyze",
                started_at=started_at, outcome="success",
                counts={"evidence_count": evidence_count, "result_count": 1},
            )
            _publish_lifecycle_progress(
                session_id, context=context, run_id=run_id, boundary="brain_analyze",
                started_at=started_at, outcome="success", evidence_count=evidence_count,
            )
            return raw

        except Exception as exc:
            log.error(
                "[Brain-Analyze] analysis failed error_type=%s",
                type(exc).__name__,
            )
            capture_exception(
                "sentiment.brain_analyze",
                RuntimeError("SENTIMENT_BRAIN_ANALYZE_FAILED"),
            )
            evidence_count = len(state.get("retrieved_chunks", []))
            sentiment_lifecycle_event(
                forecast_context=context, run_id=run_id, boundary="brain_analyze",
                started_at=started_at, outcome="fail_closed",
                counts={"evidence_count": evidence_count, "result_count": 0},
                missing=True, degraded=True,
                degraded_flags=["analysis_unavailable"],
                error_code="SENTIMENT_BRAIN_ANALYZE_FAILED",
                error_type=_safe_error_type(exc),
            )
            _publish_lifecycle_progress(
                session_id, context=context, run_id=run_id, boundary="brain_analyze",
                started_at=started_at, outcome="fail_closed", evidence_count=evidence_count,
                degraded_flags=["analysis_unavailable"],
            )
            fallback = json.dumps({
                "status":             "ERROR",
                "overall_sentiment":  None,
                "conviction_level":   "Low",
                "key_signals":        [],
                "model_agreement":    "Unavailable",
                "narrative":          "Sentiment interpretation unavailable because the analysis service failed.",
                "risk_flags":         ["Brain API call failed"],
                "data_quality_note":  "Interpretation unavailable due to API error.",
            })
            state["messages"].append({"role": "assistant", "content": fallback})
            return fallback

    # =========================================================================
    # LAYER 1 — EXECUTOR (MCP Protocol Interface)
    # =========================================================================

    @traceable(name="sentiment.executor", run_type="tool",
               process_inputs=_trace_inputs("executor"), process_outputs=_trace_outputs("executor"))
    async def _execute_sentiment_pipeline(
        self,
        session:  ClientSession,
        state:    SentimentAgentState,
        plan:     dict[str, Any],
    ) -> None:

        attempts = state.setdefault("attempts", [])
        prior_attempt_errors = [
            error for attempt in attempts for error in attempt.get("errors", [])
        ]
        state.update({
            "retrieved_chunks": [], "sources_metadata": [],
            "finbert_result": {}, "vader_result": {}, "fear_greed_result": {},
            "extraction_errors": [], "prior_attempt_errors": prior_attempt_errors,
            "sentiment_status": "NO_DATA", "sentiment_polarity": None,
        })

        directives_for_attempt = state.get("manager_directives", {})
        context, run_id = require_sentiment_context(
            state.get("forecast_context") or directives_for_attempt.get("forecast_context"),
            state.get("run_id") or directives_for_attempt.get("run_id"),
            ticker=directives_for_attempt.get("ticker"),
        )

        session_id = session_from_directives(directives_for_attempt)

        def _record_unavailable(
            boundary: str, flag: str, model_versions: dict[str, str],
        ) -> None:
            unavailable_started_at = time.perf_counter()
            sentiment_lifecycle_event(
                forecast_context=context, run_id=run_id, boundary=boundary,
                started_at=unavailable_started_at, outcome="unavailable",
                counts={"input_count": 0, "result_count": 0, "evidence_count": 0},
                missing=True, degraded=True, degraded_flags=[flag],
                model_versions=model_versions,
            )
            _publish_lifecycle_progress(
                session_id, context=context, run_id=run_id, boundary=boundary,
                started_at=unavailable_started_at, outcome="unavailable",
                degraded_flags=[flag],
            )

        async def _call(tool: str, args: dict) -> dict:
            call_started_at = time.perf_counter()
            try:
                add_breadcrumb("mcp.sentiment", f"Calling {tool}", {"tool": tool})
                _publish_progress(
                    session_id, "agent_tool_call", agent="sentiment",
                    message=f"Sentiment Agent: calling tool '{tool}'...",
                    detail={"tool": tool}, run_id=run_id,
                )
                call_args = dict(args)
                call_args.update(forecast_context=context.to_dict(), run_id=run_id)
                result = await _sentiment_call_tool(
                    session, tool, call_args, context=context, run_id=run_id,
                )
                payload = json.loads(result.content[0].text)
                safe_trace = result.safe_trace
                semantic_count = int(safe_trace.get("evidence_count") or 0)
                outcome = str(safe_trace.get("outcome") or "fail_closed")
                flags = list(safe_trace.get("degraded_flags") or [])
                _publish_lifecycle_progress(
                    session_id, context=context, run_id=run_id,
                    boundary=_TOOL_BOUNDARIES[tool], started_at=call_started_at,
                    outcome=outcome, evidence_count=semantic_count,
                    degraded_flags=flags,
                )
                if payload.get("error") or outcome != "success":
                    error_code = str(safe_trace.get("error_code") or "SENTIMENT_TOOL_DATA_UNAVAILABLE")
                    if error_code not in _TOOL_ERROR_CODES:
                        error_code = "SENTIMENT_TOOL_DATA_UNAVAILABLE"
                    state["extraction_errors"].append(f"{tool}: {error_code}")
                    log.warning(
                        "[Executor] tool=%s outcome=%s error_code=%s",
                        tool, outcome, error_code,
                    )
                    if outcome == "fail_closed":
                        capture_exception(
                            "mcp.sentiment", RuntimeError("SENTIMENT_TOOL_FAIL_CLOSED"),
                            tags={
                                "tool": tool,
                                "error_type": str(safe_trace.get("error_type") or "Exception"),
                            },
                        )
                    safe_unavailable = {
                        "error": "Sentiment tool unavailable.",
                        "error_code": error_code,
                        "status": (
                            str(payload.get("status") or "UNKNOWN").upper()
                            if str(payload.get("status") or "").upper() in {"ERROR", "UNKNOWN"}
                            else "ERROR"
                            if outcome == "fail_closed"
                            else "NO_DATA"
                        ),
                    }
                    if payload.get("polarity_invalid") is True:
                        safe_unavailable["polarity_invalid"] = True
                    return safe_unavailable
                log.info("[Executor] tool=%s outcome=success", tool)
                return payload
            except Exception as exc:
                state["extraction_errors"].append(
                    f"{tool} exception: SENTIMENT_TOOL_CALL_FAILED"
                )
                log.error(
                    "[Executor] tool=%s outcome=error error_type=%s",
                    tool, _safe_error_type(exc),
                )
                _publish_lifecycle_progress(
                    session_id, context=context, run_id=run_id,
                    boundary=_TOOL_BOUNDARIES.get(tool, "point_in_time_retrieval"),
                    started_at=call_started_at, outcome="fail_closed",
                    degraded_flags=["tool_unavailable"],
                )
                capture_exception(
                    "mcp.sentiment",
                    RuntimeError("SENTIMENT_TOOL_CALL_FAILED"),
                    tags={"tool": tool, "error_type": _safe_error_type(exc)},
                )
                return {}

        # -- Step 1: retrieve_social_data ------------------------------------
        retrieval_query = plan.get("retrieval_query", "market sentiment")
        ticker          = context.ticker
        days_back       = 30

        log.info(
            "[Executor] operation=retrieve_social_data ticker=%s days=%d",
            ticker, days_back,
        )

        retrieval_args: dict[str, Any] = {"query": retrieval_query, "days_back": days_back}
        if ticker:
            retrieval_args["ticker"] = ticker

        social_data = await _call("retrieve_social_data", retrieval_args)

        log.info("[Executor] SENTIMENT_RETRIEVAL_RESULT chunks=%d",
                 len(social_data.get("chunks", [])))

        state["retrieved_chunks"] = social_data.get("chunks", [])
        state["sources_metadata"] = social_data.get("sources_metadata", social_data.get("sources", []))
        log.info("[Executor] Retrieved %d chunks.", len(state["retrieved_chunks"]))

        urls = [
            str(item.get("url", "")).strip()
            for item in state["sources_metadata"]
            if isinstance(item, dict)
        ]
        inferred_documents = len({url for url in urls if url}) or len({
            str(chunk).strip()
            for chunk in state["retrieved_chunks"]
            if str(chunk).strip()
        })
        inferred_domains = len({
            urlparse(url).netloc.lower()
            for url in urls
            if urlparse(url).netloc
        })
        unique_documents = int(
            social_data.get("unique_document_count", inferred_documents) or 0
        )
        unique_domains = int(
            social_data.get("unique_domain_count", inferred_domains) or 0
        )
        raw_chunk_count = int(
            social_data.get("raw_chunk_count", len(state["retrieved_chunks"])) or 0
        )
        evidence_independence = social_data.get("evidence_independence")
        if evidence_independence is None and unique_documents:
            evidence_independence = round(unique_domains / unique_documents, 4)
        retrieval_stats = {
            "raw_chunk_count": raw_chunk_count,
            "unique_documents": unique_documents,
            "unique_domains": unique_domains,
            "chunks_collapsed_to_documents": int(
                social_data.get(
                    "chunks_collapsed_to_documents",
                    max(0, raw_chunk_count - unique_documents),
                ) or 0
            ),
            "duplicate_text_chunks_removed": int(
                social_data.get("duplicate_text_chunks_removed", 0) or 0
            ),
            "evidence_independence": evidence_independence,
        }

        if state["retrieved_chunks"]:
            log.info("[Executor] Step 2 — analyze_finbert (%d chunks).",
                     len(state["retrieved_chunks"]))
            state["finbert_result"] = await _call(
                "analyze_finbert",
                {"texts": state["retrieved_chunks"]},
            )
        else:
            log.warning("[Executor] Step 2 — skipping analyze_finbert (no chunks).")
            state["finbert_result"] = {
                "bullish_prob": None, "bearish_prob": None, "neutral_prob": None,
                "label": None, "polarity": None, "status": "NO_DATA",
                "total_chunks": 0, "skipped_chunks": 0,
                "model_version": FINBERT_MODEL_VERSION,
            }
            _record_unavailable(
                "finbert_inference", "missing_finbert",
                {"finbert": FINBERT_MODEL_VERSION},
            )

        if state["retrieved_chunks"]:
            log.info("[Executor] Step 3 — score_vader (%d chunks).",
                     len(state["retrieved_chunks"]))
            state["vader_result"] = await _call(
                "score_vader",
                {"texts": state["retrieved_chunks"]},
            )
        else:
            log.warning("[Executor] Step 3 — skipping score_vader (no chunks).")
            state["vader_result"] = {
                "compound": None, "positive_mean": None, "negative_mean": None,
                "neutral_mean": None, "label": None, "polarity": None,
                "status": "NO_DATA", "total_chunks": 0, "skipped_chunks": 0,
                "model_version": VADER_MODEL_VERSION,
            }
            _record_unavailable(
                "vader_scoring", "missing_vader", {"vader": VADER_MODEL_VERSION},
            )

        fb_has_data = (
            state["finbert_result"].get("total_chunks", 0) > 0
            and state["finbert_result"].get("status") in {"VALID", "LOW_COVERAGE"}
        )
        vd_has_data = (
            state["vader_result"].get("total_chunks", 0) > 0
            and state["vader_result"].get("status") in {"VALID", "LOW_COVERAGE"}
        )

        if fb_has_data or vd_has_data:
            log.info("[Executor] Step 4 — calculate_fear_greed.")
            directives = directives_for_attempt
            fear_greed_args: dict[str, Any] = {
                "finbert_result": state["finbert_result"],
                "vader_result":   state["vader_result"],
            }
            if "finbert_weight" in directives:
                fear_greed_args["finbert_weight"] = float(directives["finbert_weight"])
            if "vader_weight" in directives:
                fear_greed_args["vader_weight"] = float(directives["vader_weight"])
            if social_data.get("evidence_independence") is not None:
                fear_greed_args["evidence_independence"] = float(
                    social_data["evidence_independence"]
                )

            state["fear_greed_result"] = await _call(
                "calculate_fear_greed", fear_greed_args
            )
            if evidence_independence is not None:
                raw_confidence = state["fear_greed_result"].get("confidence")
                if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
                    state["fear_greed_result"]["confidence"] = round(
                        min(max(float(raw_confidence), 0.0), float(evidence_independence)), 4
                    )
        else:
            log.warning("[Executor] Step 4 — skipping calculate_fear_greed (no model data).")
            state["fear_greed_result"] = {
                "score": None, "label": None, "polarity": None, "status": "NO_DATA",
                "finbert_score": None, "vader_score": None,
                "weights": {}, "confidence": 0.0, "diagnostics": {},
                "model_version": FEAR_GREED_MODEL_VERSION,
            }
            _record_unavailable(
                "fear_greed_fusion", "missing_fusion",
                {"fusion": FEAR_GREED_MODEL_VERSION},
            )

        model_has_data = fb_has_data or vd_has_data
        score = state["fear_greed_result"].get("score")
        is_valid_composite_score = valid_composite_score(score)
        fusion_status = canonical_sentiment_model_status(
            state["fear_greed_result"], available=is_valid_composite_score,
        )
        fusion_polarity, fusion_polarity_invalid = canonical_sentiment_polarity(
            state["fear_greed_result"].get("polarity")
        )
        fusion_polarity_invalid = (
            fusion_polarity_invalid
            or state["fear_greed_result"].get("polarity_invalid") is True
        )
        state["fear_greed_result"].update({
            "status": fusion_status,
            "polarity": fusion_polarity,
            "polarity_invalid": fusion_polarity_invalid,
        })
        if fusion_polarity_invalid:
            state["extraction_errors"].append(
                "fear_greed returned invalid polarity"
            )
        for output_name, output in (
            ("finbert", state["finbert_result"]),
            ("vader", state["vader_result"]),
            ("fear_greed", state["fear_greed_result"]),
        ):
            if str(output.get("status") or "").upper() in {"ERROR", "FAILED", "UNKNOWN"}:
                state["extraction_errors"].append(
                    f"{output_name} returned critical model status"
                )
        critical_errors = list(state["extraction_errors"])
        quality_failures: list[str] = []
        if not state["retrieved_chunks"]:
            quality_failures.append("no sentiment documents retrieved")
        if not model_has_data:
            quality_failures.append("no valid sentiment model output")
        if not is_valid_composite_score:
            quality_failures.append("missing or invalid composite sentiment score")
        if unique_documents < self._MIN_UNIQUE_DOCUMENTS:
            quality_failures.append(
                f"unique document coverage {unique_documents} is below {self._MIN_UNIQUE_DOCUMENTS}"
            )
        if unique_domains < self._MIN_UNIQUE_DOMAINS:
            quality_failures.append(
                f"unique source-domain coverage {unique_domains} is below {self._MIN_UNIQUE_DOMAINS}"
            )
        if critical_errors:
            quality_failures.append("critical extraction or scoring errors were recorded")

        if critical_errors:
            status = "ERROR"
        elif not state["retrieved_chunks"] or not model_has_data:
            status = "NO_DATA"
        elif not is_valid_composite_score:
            status = "ERROR"
        elif (
            unique_documents < self._MIN_UNIQUE_DOCUMENTS
            or unique_domains < self._MIN_UNIQUE_DOMAINS
        ):
            status = "LOW_COVERAGE"
        else:
            status = "VALID"
        polarity = fusion_polarity
        if (
            polarity is None
            and not fusion_polarity_invalid
            and is_valid_composite_score
            and fusion_status in {"VALID", "LOW_COVERAGE"}
        ):
            score = float(score)
            polarity = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
        state["fear_greed_result"].update({
            "status": fusion_status, "attempt_status": status, "polarity": polarity,
            "coverage": {"unique_documents": unique_documents, "unique_domains": unique_domains,
                         "evidence_independence": evidence_independence},
        })
        state["sentiment_status"] = status
        state["sentiment_polarity"] = polarity
        execution_success = not critical_errors
        quality_passed = status == "VALID" and not quality_failures
        attempts.append({
            "attempt_number": len(attempts) + 1, "plan": dict(plan), "status": status,
            "execution_success": execution_success,
            "quality_passed": quality_passed,
            "quality_failures": quality_failures,
            "polarity": polarity,
            "retrieval_stats": dict(retrieval_stats),
            "retrieval": {"chunks": list(state["retrieved_chunks"]),
                          "sources_metadata": list(state["sources_metadata"])},
            "model_outputs": {"finbert": dict(state["finbert_result"]),
                              "vader": dict(state["vader_result"]),
                              "fear_greed": dict(state["fear_greed_result"])},
            "errors": list(state["extraction_errors"]),
        })

        log.info(
            "[Executor] Pipeline complete — chunks=%d finbert=%s vader=%s fg_score=%.4f fg_label=%s",
            len(state["retrieved_chunks"]),
            state["finbert_result"].get("label", "N/A"),
            state["vader_result"].get("label", "N/A"),
            state["fear_greed_result"].get("score") or 0.0,
            state["fear_greed_result"].get("label", "N/A"),
        )

    # =========================================================================
    # LANGGRAPH NODES — thin wrappers over the layer methods above
    # =========================================================================

    async def _node_brain_plan(self, state: SentimentAgentState) -> dict[str, Any]:
        new_counter = state["loop_counter"] + 1
        log.info("[Run] Iteration %d", new_counter)
        plan = await self._brain_plan(state)
        return {"loop_counter": new_counter, "current_plan": plan}

    async def _node_execute(self, state: SentimentAgentState) -> dict[str, Any]:
        await self._execute_sentiment_pipeline(
            self._session, state, state.get("current_plan", {})
        )
        return {
            "retrieved_chunks":  state["retrieved_chunks"],
            "sources_metadata":  state["sources_metadata"],
            "finbert_result":    state["finbert_result"],
            "vader_result":      state["vader_result"],
            "fear_greed_result": state["fear_greed_result"],
            "extraction_errors": state["extraction_errors"],
            "prior_attempt_errors": state.get("prior_attempt_errors", []),
            "attempts":           state.get("attempts", []),
            "sentiment_status":   state.get("sentiment_status", "NO_DATA"),
            "sentiment_polarity": state.get("sentiment_polarity"),
        }

    async def _node_analyze(self, state: SentimentAgentState) -> dict[str, Any]:
        self._select_best_attempt(state)
        state["aspect_observations"] = extract_aspect_observations(
            state["retrieved_chunks"], state["sources_metadata"],
            state.get("finbert_result", {}).get("chunk_scores", []),
        )
        raw_analysis = await self._brain_analyze(state)
        return {
            "brain_reasoning": raw_analysis,
            "aspect_observations": state["aspect_observations"],
            "retrieved_chunks": state["retrieved_chunks"],
            "sources_metadata": state["sources_metadata"],
            "finbert_result": state["finbert_result"],
            "vader_result": state["vader_result"],
            "fear_greed_result": state["fear_greed_result"],
            "extraction_errors": state["extraction_errors"],
            "sentiment_status": state["sentiment_status"],
            "sentiment_polarity": state["sentiment_polarity"],
        }

    def _should_continue(self, state: SentimentAgentState):
        directives = state["manager_directives"]
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        if state.get("sentiment_status") == "VALID":
            log.info("[Run] Sentiment attempt passed deterministic quality checks.")
            return "analyze"
        if state["loop_counter"] >= max_loops:
            log.warning("[Run] loop_counter=%d >= max_loops=%d — exiting to analyze.",
                        state["loop_counter"], max_loops)
            return "analyze"
        log.info(
            "[Run] Sentiment status=%s; retrying for stronger evidence coverage.",
            state.get("sentiment_status", "NO_DATA"),
        )
        return "brain_plan"

    def _build_graph(self):

        builder = StateGraph(SentimentAgentState)

        builder.add_node("brain_plan", self._node_brain_plan)
        builder.add_node("execute",    self._node_execute)
        builder.add_node("analyze",    self._node_analyze)

        builder.set_entry_point("brain_plan")
        builder.add_edge("brain_plan", "execute")
        builder.add_conditional_edges(
            "execute",
            self._should_continue,
            {"brain_plan": "brain_plan", "analyze": "analyze"},
        )
        builder.add_edge("analyze", END)

        return builder.compile()

    # =========================================================================
    # ENTRY GATEWAY — run()
    # =========================================================================

    @traceable(name="SentimentAgent.run", run_type="chain",
               process_inputs=_trace_inputs("run"), process_outputs=_trace_outputs("run"))
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        run_started_at = time.perf_counter()
        directives = shared_state.get("manager_directives", {})
        context, run_id = require_sentiment_context(
            shared_state.get("forecast_context") or directives.get("forecast_context"),
            directives.get("run_id") or shared_state.get("run_id"),
            ticker=directives.get("ticker"),
        )
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        log.info("SentimentAgent.run() started max_loops=%d", max_loops)

        internal_directives = dict(shared_state.get("manager_directives", {}))
        internal_directives.update(
            forecast_context=context.to_dict(), run_id=run_id, ticker=context.ticker,
        )
        state: SentimentAgentState = {
            "messages":           [],
            "retrieved_chunks":   [],
            "sources_metadata":   [],
            "finbert_result":     {},
            "vader_result":       {},
            "fear_greed_result":  {},
            "brain_reasoning":    "",
            "loop_counter":       0,
            "extraction_errors":  [],
            "prior_attempt_errors": [],
            "attempts":           [],
            "sentiment_status":   "NO_DATA",
            "sentiment_polarity": None,
            "task_query":                  shared_state["task_query"],
            "manager_directives":          internal_directives,
            "financial_metrics_summary":   shared_state.get("financial_metrics_summary", {}),
            "aggregated_research_context": shared_state.get("aggregated_research_context", []),
            "current_plan":       {},
            "forecast_context":   context.to_dict(),
            "run_id":             run_id,
        }

        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                await session.initialize()
                log.info("[Run] MCP session initialised.")

                self._session = session
                try:
                    state = await self._graph.ainvoke(
                        state,
                        config={"recursion_limit": (max_loops + 1) * 3},
                    )
                finally:
                    self._session = None

        raw_analysis = state["brain_reasoning"]
        analysis_parse_error = False
        try:
            analysis = json.loads(raw_analysis)
            if not isinstance(analysis, dict):
                raise ValueError("sentiment analysis must be a JSON object")
        except (json.JSONDecodeError, ValueError):
            analysis_parse_error = True
            log.warning("[Run] Brain analysis JSON malformed; output withheld.")
            analysis = {
                "status":            "ERROR",
                "overall_sentiment": None,
                "conviction_level":  "Low",
                "key_signals":       [],
                "model_agreement":   "Unavailable",
                "narrative":         "Sentiment interpretation unavailable because the model response was malformed.",
                "risk_flags":        ["Brain analysis JSON parse failed"],
                "data_quality_note": "Malformed model output was withheld.",
            }

        fg = state["fear_greed_result"]
        fb = state["finbert_result"]
        vd = state["vader_result"]
        retrieval_stats = fg.get("retrieval_stats", {}) or fg.get("coverage", {}) or {}
        selected_attempt_number = fg.get("selected_attempt_number")
        causal_aspects = supported_causal_aspects(state.get("aspect_observations", []))
        causal_audit_result = causal_audit(analysis, causal_aspects)
        if causal_audit_result["causal_mismatch_detected"]:
            analysis["narrative"] = (
                "Sentiment narrative withheld because its causal attribution was not "
                "supported by an observation with causal_language_supported=True."
            )
            analysis["key_signals"] = []
            analysis["causal_aspects_used"] = []
            existing_risk_flags = analysis.get("risk_flags", [])
            if not isinstance(existing_risk_flags, list):
                existing_risk_flags = [str(existing_risk_flags)]
            analysis["risk_flags"] = existing_risk_flags + [
                "Unsupported causal sentiment attribution"
            ]
        selected_attempt = next(
            (
                attempt for attempt in state.get("attempts", [])
                if attempt.get("attempt_number") == selected_attempt_number
            ),
            None,
        ) or {}
        quality_failures = list(selected_attempt.get("quality_failures", []))
        if causal_audit_result["causal_mismatch_detected"]:
            quality_failures.append(
                "analysis claimed causal aspects without causal_language_supported=True"
            )
        analysis_risk_flags = analysis.get("risk_flags", [])
        brain_execution_error = analysis_parse_error or any(
            "brain api call failed" in str(flag).lower()
            for flag in analysis_risk_flags
        )
        if brain_execution_error:
            quality_failures.append("sentiment interpretation failed or was malformed")
        quality_failures = list(dict.fromkeys(quality_failures))
        quality_passed = bool(
            selected_attempt.get("quality_passed")
            and not causal_audit_result["causal_mismatch_detected"]
            and not brain_execution_error
        )
        attempt_errors = [
            error
            for attempt in state.get("attempts", [])
            for error in attempt.get("errors", [])
        ]

        sentiment_analysis_summary: dict[str, Any] = {
            "status":               state.get("sentiment_status", "NO_DATA"),
            "polarity":             state.get("sentiment_polarity"),
            "execution_success":    True,
            "pipeline_execution_success": bool(selected_attempt.get("execution_success")),
            "quality_passed":       quality_passed,
            "quality_failures":     quality_failures,
            # Core Fear/Greed index
            "ticker":               extract_ticker(
                                        shared_state.get("task_query", ""),
                                        directives,
                                        shared_state.get("financial_metrics_summary", {}),
                                    ),
            "fear_greed_score":     fg.get("score"),
            "fear_greed_label":     fg.get("label"),
            "fear_greed_confidence": fg.get("confidence"),
            "direction_strength":   fg.get("direction_strength"),
            "model_agreement_score": fg.get("model_agreement_score"),
            "evidence_independence": fg.get(
                "evidence_independence",
                retrieval_stats.get("evidence_independence"),
            ),
            "confidence_components": {
                "confidence": fg.get("confidence"),
                "direction_strength": fg.get("direction_strength"),
                "model_agreement_score": fg.get("model_agreement_score"),
                "evidence_independence": fg.get(
                    "evidence_independence",
                    retrieval_stats.get("evidence_independence"),
                ),
                "model_coverage_score": (fg.get("diagnostics", {}) or {}).get(
                    "model_coverage_score"
                ),
            },

            # FinBERT signals
            "finbert_label":        fb.get("label"),
            "finbert_bullish_prob": fb.get("bullish_prob"),
            "finbert_bearish_prob": fb.get("bearish_prob"),
            "finbert_neutral_prob": fb.get("neutral_prob"),

            # VADER signals
            "vader_compound":       vd.get("compound"),
            "vader_label":          vd.get("label"),

            # Data coverage
            "total_chunks_analyzed": len(state["retrieved_chunks"]),
            "raw_chunk_count":       retrieval_stats.get("raw_chunk_count"),
            "unique_document_count": retrieval_stats.get("unique_documents", 0),
            "unique_domain_count":   retrieval_stats.get("unique_domains", 0),
            "chunks_collapsed_to_documents": retrieval_stats.get(
                "chunks_collapsed_to_documents", 0
            ),
            "duplicate_text_chunks_removed": retrieval_stats.get(
                "duplicate_text_chunks_removed", 0
            ),
            "source_document_counts": dict(retrieval_stats),
            "sources_metadata":      state["sources_metadata"],
            "aspect_observations":   state.get("aspect_observations", []),
            "causal_aspects_supported": causal_aspects,
            "causal_aspects_used":   causal_audit_result["claimed_causal_aspects"],
            "causal_audit":          causal_audit_result,
            "causal_mismatch_detected": causal_audit_result["causal_mismatch_detected"],

            # Brain interpretation
            "overall_sentiment":    analysis.get("overall_sentiment"),
            "conviction_level":     analysis.get("conviction_level"),
            "model_agreement":      analysis.get("model_agreement"),
            "key_signals":          analysis.get("key_signals", []),
            "narrative":            analysis.get("narrative", ""),
            "risk_flags":           analysis.get("risk_flags", []),
            "data_quality_note":    analysis.get("data_quality_note", ""),


            "brain_reasoning":       raw_analysis,
            "fear_greed_diagnostics": fg.get("diagnostics", {}),

            "loop_iterations_used": state["loop_counter"],
            "selected_attempt_number": selected_attempt_number,
            "extraction_errors":    state["extraction_errors"],
            "prior_attempt_errors": state.get("prior_attempt_errors", []),
            "attempt_extraction_errors": attempt_errors,
            "attempts":             state.get("attempts", []),
        }

        if sentiment_analysis_summary["status"] in {"NO_DATA", "ERROR"}:
            sentiment_analysis_summary.update({
                "fear_greed_score": None, "fear_greed_label": None,
                "fear_greed_confidence": 0.0, "finbert_label": None,
                "finbert_bullish_prob": None, "finbert_bearish_prob": None,
                "finbert_neutral_prob": None, "vader_compound": None,
                "vader_label": None, "overall_sentiment": None,
            })

        feature_started_at = time.perf_counter()
        features = build_monthly_sentiment_features(
            texts=state["retrieved_chunks"],
            sources_metadata=state["sources_metadata"],
            finbert_result=fb,
            vader_result=vd,
            fear_greed_result=fg,
            selected_attempt=selected_attempt,
            forecast_context=context,
            run_id=run_id,
        )
        feature_slot = features.to_state_slot()
        _publish_lifecycle_progress(
            session_from_directives(directives), context=context, run_id=run_id,
            boundary="feature_aggregation_and_pre_shared_state",
            started_at=feature_started_at,
            outcome="degraded" if features.degraded_flags else "success",
            evidence_count=len(features.evidence_ids),
            degraded_flags=["feature_degraded"] if features.degraded_flags else [],
        )
        sentiment_analysis_summary.update({
            "evidence_ids": list(features.evidence_ids),
            "model_versions": dict(features.model_versions),
            "missing_flags": list(features.missing_flags),
            "degraded_flags": list(features.degraded_flags),
        })

        # Commit compact, reference-only payloads to shared state. Raw chunks,
        # full attempts, duplicated metadata and brain reasoning remain private.
        shared_state["sentiment_analysis_summary"] = compact_sentiment_summary(  # type: ignore[literal-required]
            sentiment_analysis_summary
        )
        shared_state["sentiment_features"] = feature_slot

        log.info(
            "SentimentAgent.run() complete ticker=%s status=%s evidence_count=%d",
            context.ticker,
            state.get("sentiment_status", "NO_DATA"),
            len(features.evidence_ids),
        )

        run_degraded = bool(features.degraded_flags) or not quality_passed
        run_flags = ["low_coverage"] if not quality_passed else []
        sentiment_lifecycle_event(
            forecast_context=context, run_id=run_id, boundary="whole_sentiment_run",
            started_at=run_started_at,
            outcome="degraded" if run_degraded else "success",
            counts={
                "evidence_count": len(features.evidence_ids),
                "result_count": 1, "loop_count": state["loop_counter"],
                "missing_count": len(features.missing_flags),
            },
            missing=bool(features.missing_flags), degraded=run_degraded,
            degraded_flags=run_flags,
            model_versions=features.model_versions,
        )
        _publish_lifecycle_progress(
            session_from_directives(directives), context=context, run_id=run_id,
            boundary="whole_sentiment_run", started_at=run_started_at,
            outcome="degraded" if run_degraded else "success",
            evidence_count=len(features.evidence_ids), degraded_flags=run_flags,
        )

        return shared_state
