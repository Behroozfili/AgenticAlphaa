"""
agents/manager_agent.py
------------------------

"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Union
from agents.executor_helpers.manager_executor_helper import (
    format_news_chunk_fairly,
    extract_chunk_text,
    render_section_entry,
    format_filing_chunk_fairly,
    feedback_to_snapshot,
    infer_result_keys,
    maybe_await,
    run_monthly_specialists_concurrently,
    controlled_monthly_join,
)

import anthropic

from langgraph.graph import END, StateGraph
from langsmith import traceable

from agents.financial_agent import FinancialAnalystAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from agents.monthly_forecast_agent import MonthlyForecastAgent
from agents.executor_helpers.monthly_forecast_helper import STAGE08_IMPLEMENTATION_IDENTITY
from agents.state import (
    EvaluationSnapshot,
    ManagerGraphState,
    SharedManagerState,
)
from memory.manager_memory import EvaluationFeedback, ManagerMemory
from core.error_handler import add_breadcrumb, capture_exception, capture_message
from core.progress_bus import publish as _publish_progress, publish_forecast as _publish_forecast
from core.observability import forecast_trace_metadata
from core.forecast_contract import ForecastContext
from tools.market_tools import validate_run_id
from agents.prompts.prompt_manager_agent import (ROUTER_SYSTEM_PROMPT ,EVALUATOR_SYSTEM_PROMPT,FINALISER_SYSTEM_PROMPT,VALID_ACTIONS,ACTION_DESCRIPTIONS)
from evaluation.validate_period_consistency import (
    check_narration_vs_period, detect_material_period_conflicts,
)
from evaluation.recommendation_gate import build_recommendation_gate, contains_actionable_recommendation
from evaluation.canonical_synthesis import build_canonical_synthesis_payload
from evaluation.publication_policy import (
    assemble_publication,
    execution_status_from_publication,
)
from evaluation.final_publication import (
    normalize_validation_aware_narrative,
    validate_final_publication,
)
from evaluation.financial_sanity import evaluate_financial_sanity


# Backward-compatible public aliases retained for existing consumers/tests.
_extract_chunk_text = extract_chunk_text
_feedback_to_snapshot = feedback_to_snapshot
_infer_result_keys = infer_result_keys


log = logging.getLogger("manager-agent")


_DEFAULT_MODEL = "claude-haiku-4-5"
_DEFAULT_MAX_ROUTING_LOOPS = 8

_TRANSIENT_ERROR_MARKERS = (
    "timeout", "timed out", "rate limit", "429", "connection reset",
    "connection refused", "temporarily unavailable", "temporary failure",
    "service unavailable", "gateway timeout", "502", "503", "504",
)


def _stage08_trace_metadata(context: Any = None, run_id: Any = None, **measurements: Any) -> dict[str, Any]:
    metadata = dict(STAGE08_IMPLEMENTATION_IDENTITY)
    if context is None:
        return metadata
    try:
        safe = forecast_trace_metadata(
            context, run_id=str(run_id) if run_id else None,
            stage=STAGE08_IMPLEMENTATION_IDENTITY["stage"],
            stage_revision=STAGE08_IMPLEMENTATION_IDENTITY["stage_revision"],
            attempt=STAGE08_IMPLEMENTATION_IDENTITY["attempt"],
            agent=STAGE08_IMPLEMENTATION_IDENTITY["agent"],
            **measurements,
        )
    except (TypeError, ValueError):
        return metadata
    return {**safe, "implementation_version": STAGE08_IMPLEMENTATION_IDENTITY["implementation_version"]}


def _manager_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Allowlist Manager entry metadata; never trace task/directive payloads."""
    directives = inputs.get("manager_directives")
    directives = directives if isinstance(directives, dict) else {}
    context = directives.get("forecast_context")
    return {
        "operation": "manager_run",
        "metadata": _stage08_trace_metadata(context, directives.get("run_id")),
        "monthly_mode": context is not None,
        "has_client_session": bool(inputs.get("client_session_id")),
    }


def _manager_trace_outputs(outputs: Any) -> dict[str, Any]:
    """Allowlist result status/counts; never trace complete Shared State."""
    state = outputs if isinstance(outputs, dict) else {}
    directives = state.get("manager_directives")
    directives = directives if isinstance(directives, dict) else {}
    forecast = state.get("monthly_forecast")
    forecast = forecast if isinstance(forecast, dict) else {}
    evidence = forecast.get("evidence_ids")
    warnings = forecast.get("warnings")
    raw_status = forecast.get("status") or state.get("execution_status")
    status = raw_status if raw_status in {
        "READY", "DEGRADED", "INSUFFICIENT_DATA", "completed", "failed",
    } else "UNKNOWN"
    return {
        "operation": "manager_run",
        "metadata": _stage08_trace_metadata(
            state.get("forecast_context") or directives.get("forecast_context"),
            forecast.get("run_id") or directives.get("run_id"),
            evidence_count=len(evidence) if isinstance(evidence, list) else 0,
            degraded_flags=["monthly_degraded"] if forecast.get("degraded") else [],
        ),
        "status": status,
        "missing": bool(forecast.get("missing")),
        "degraded": bool(forecast.get("degraded")),
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
        "history_count": len(state.get("agent_execution_history", [])) if isinstance(state.get("agent_execution_history"), list) else 0,
    }


def _publish_stage08_monthly(
    session_id: str | None, event_type: str, *, context: ForecastContext,
    run_id: str, started_at: float, message: str,
    evidence_count: int | None = None, degraded_flags: tuple[str, ...] = (),
) -> None:
    _publish_forecast(
        session_id, event_type, run_id=run_id, context=context,
        stage=STAGE08_IMPLEMENTATION_IDENTITY["stage"],
        stage_revision=STAGE08_IMPLEMENTATION_IDENTITY["stage_revision"],
        attempt=STAGE08_IMPLEMENTATION_IDENTITY["attempt"],
        agent=STAGE08_IMPLEMENTATION_IDENTITY["agent"],
        message=message, started_at=started_at, evidence_count=evidence_count,
        degraded_flags=degraded_flags,
    )


def sentiment_result_quality_passed(summary: dict[str, Any]) -> bool:
    """Validate sentiment quality independently of agent execution success."""
    if isinstance(summary.get("fear_greed_score"), bool):
        return False
    try:
        score = float(summary.get("fear_greed_score"))
        unique_documents = int(summary.get("unique_document_count") or 0)
        unique_domains = int(summary.get("unique_domain_count") or 0)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        summary.get("status") == "VALID"
        and summary.get("quality_passed") is True
        and math.isfinite(score)
        and -1.0 <= score <= 1.0
        and unique_documents >= SentimentAgent._MIN_UNIQUE_DOCUMENTS
        and unique_domains >= SentimentAgent._MIN_UNIQUE_DOMAINS
        and not summary.get("extraction_errors")
        and not summary.get("causal_mismatch_detected")
    )


def is_transient_failure(value: Any) -> bool:
    """Return true only for retryable transport/service failures."""
    if isinstance(value, dict):
        return any(is_transient_failure(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(is_transient_failure(item) for item in value)
    text = str(value or "").lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


def agent_result_is_valid(agent_key: str, state: dict[str, Any]) -> bool:
    """Determine whether a previously produced agent result can be reused."""
    if agent_key == "research":
        return len(state.get("aggregated_research_context", [])) >= 3
    if agent_key == "financial":
        fm = state.get("financial_metrics_summary", {}) or {}
        sanity = fm.get("financial_sanity", {}) or {}
        features = state.get("financial_features")
        if isinstance(features, dict):
            values = features.get("values")
            evidence_ids = features.get("evidence_ids")
            return (
                bool(fm)
                and fm.get("validation_passed") is True
                and sanity.get("passed") is True
                and isinstance(values, dict) and bool(values)
                and isinstance(evidence_ids, list) and bool(evidence_ids)
                and features.get("missing") is False
            )
        return (
            bool(fm)
            and fm.get("validation_passed") is True
            and sanity.get("passed") is True
        )
    if agent_key == "sentiment":
        sm = state.get("sentiment_analysis_summary", {}) or {}
        return sentiment_result_quality_passed(sm)
    return False


def _agent_was_attempted(agent_key: str, state: dict[str, Any]) -> bool:
    class_name = {
        "research": "ResearchAgent",
        "financial": "FinancialAnalystAgent",
        "sentiment": "SentimentAgent",
    }.get(agent_key)
    return any(
        isinstance(row, dict) and row.get("agent_name") == class_name
        for row in state.get("agent_execution_history", [])
    )


def _agent_failure_context(agent_key: str, state: dict[str, Any]) -> list[Any]:
    context: list[Any] = []
    class_name = {
        "research": "ResearchAgent",
        "financial": "FinancialAnalystAgent",
        "sentiment": "SentimentAgent",
    }.get(agent_key)
    context.extend(
        row.get("error_message") for row in state.get("agent_execution_history", [])
        if isinstance(row, dict) and row.get("agent_name") == class_name
    )
    if agent_key == "financial":
        context.extend((state.get("financial_metrics_summary", {}) or {}).get("extraction_errors", []))
    elif agent_key == "sentiment":
        sm = state.get("sentiment_analysis_summary", {}) or {}
        context.extend(sm.get("extraction_errors", []))
        context.extend(sm.get("prior_attempt_errors", []))
    return context


def next_pipeline_action(agent_key: str, state: dict[str, Any]) -> str:
    """Advance without rerunning an already attempted non-transient failure."""
    if agent_key == "research":
        return "run_financial"
    if agent_key == "financial":
        return (
            "run_sentiment"
            if state.get("manager_directives", {}).get("include_sentiment", True)
            else "finalise"
        )
    return "finalise"


def normalize_route_action(action: str, state: dict[str, Any]) -> str:
    if not (action.startswith("run_") or action.startswith("rerun_")):
        return action
    agent_key = action.removeprefix("rerun_").removeprefix("run_")
    if agent_result_is_valid(agent_key, state):
        return next_pipeline_action(agent_key, state)
    if _agent_was_attempted(agent_key, state) and not is_transient_failure(
        _agent_failure_context(agent_key, state)
    ):
        return next_pipeline_action(agent_key, state)
    return action




class ManagerAgent:
   

    def __init__(
        self,
        research_agent:    ResearchAgent,
        financial_agent:   FinancialAnalystAgent,
        sentiment_agent:   SentimentAgent,
        memory:            ManagerMemory,
        model:             str = _DEFAULT_MODEL,
        max_routing_loops: int = _DEFAULT_MAX_ROUTING_LOOPS,
        llm_client:        anthropic.AsyncAnthropic | None = None,
        monthly_forecast_agent: MonthlyForecastAgent | None = None,
        market_forecast_service: Any | None = None,
        market_data_provider: Any | None = None,
    ) -> None:
        
        self._llm               = llm_client or anthropic.AsyncAnthropic()
        self._model             = model
        self._max_routing_loops = max_routing_loops
        self._memory            = memory
        self._monthly_forecast_agent = monthly_forecast_agent or MonthlyForecastAgent(
            llm_client=self._llm, model=model,
        )
        self._market_forecast_service = market_forecast_service
        self._market_data_provider = market_data_provider

        self._agents: dict[str, Any] = {
            "research":  research_agent,
            "financial": financial_agent,
            "sentiment": sentiment_agent,
        }

        self._graph = self._build_graph()

        log.info(
            "ManagerAgent initialised — model=%s max_loops=%d graph=compiled",
            model, max_routing_loops,
        )

    # =========================================================================
    # STEP 1 — INGEST & HYDRATE
    # =========================================================================

    def _hydrate_state(
        self,
        task_query:         str,
        manager_directives: dict[str, Any],
    ) -> SharedManagerState:
        
        state: SharedManagerState = {
            "task_query":                  task_query,
            "manager_directives":          manager_directives,
            "agent_execution_history":     [],
            "orchestrator_logs":           [],
            "final_report":                "",
            "aggregated_research_context": [],
            "financial_metrics_summary":   {},
            "sentiment_analysis_summary":  {},
        }
        log.info("[Hydrate] State initialised for task: '%s'", task_query[:80])
        return state

    # =========================================================================
    # STEP 2 — RECALL
    # =========================================================================

    def _recall(self, ticker: str | None) -> dict[str, Any]:
       
        recall = self._memory.recall(ticker=ticker)
        log.info(
            "[Recall] agents_run=%s heuristics=%d ticker_cached=%s",
            recall["short_term"].get("agents_dispatched", []),
            len(recall["long_term"].get("heuristics", {})),
            bool(recall["long_term"].get("ticker_insight")),
        )
        return recall

    # =========================================================================
    # STEP 3 — BRAIN  (all three passes are now async)
    # =========================================================================

    async def _brain_route(
        self,
        state:         SharedManagerState,
        memory_recall: dict[str, Any],
        loop_counter:  int,
    ) -> dict[str, Any]:
      
        research_chunks = len(state.get("aggregated_research_context", []))
        fin_score       = state.get("financial_metrics_summary", {}).get(
                              "composite_score", {}).get("score")
        fin_passed      = state.get("financial_metrics_summary", {}).get("validation_passed")
        sent_score      = state.get("sentiment_analysis_summary", {}).get("fear_greed_score")
        sent_label      = state.get("sentiment_analysis_summary", {}).get("fear_greed_label")

        state_summary = {
            "task_query":          state.get("task_query", ""),
            "loop_counter":        loop_counter,
            "max_routing_loops":   self._max_routing_loops,
            "manager_directives":  state.get("manager_directives", {}),
            "research_chunks":     research_chunks,
            "financial_score":     fin_score,
            "financial_passed":    fin_passed,
            "sentiment_fg_score":  sent_score,
            "sentiment_fg_label":  sent_label,
            "agents_run":          self._memory.agents_run(),
            "last_evaluation":     memory_recall["short_term"].get("last_evaluation"),
            "orchestrator_logs":   state.get("orchestrator_logs", [])[-5:],
        }

        user_content = (
            f"STATE SUMMARY:\n{json.dumps(state_summary, indent=2)}\n\n"
            f"MEMORY CONTEXT:\n{json.dumps(memory_recall, indent=2)}\n\n"
            "Decide the next routing action."
        )

        self._memory.add_message(role="user", content=user_content)
        log.info("[Brain-Route] Querying Claude for routing decision (loop=%d)...", loop_counter)

        try:
            response = await self._llm.messages.create(
                model=self._model,
                max_tokens=768,
                temperature=0.0,
                system=ROUTER_SYSTEM_PROMPT,
                messages=self._memory.get_messages(),
            )
            if response.stop_reason == "max_tokens":
                log.warning("[Brain-Route] Response truncated by max_tokens — JSON may be invalid.")
            raw = response.content[0].text.strip()
            self._memory.add_message(role="assistant", content=raw)
            raw = raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw)

            action = decision.get("action", "")
            if action not in VALID_ACTIONS:
                log.warning("[Brain-Route] Invalid action '%s' — defaulting to abort.", action)
                decision["action"] = "abort"
            else:
                normalized_action = normalize_route_action(action, state)
                if normalized_action != action:
                    decision["action"] = normalized_action
                    decision["reasoning"] = (
                        f"{decision.get('reasoning', '')} Reused an existing valid "
                        "result or avoided a non-transient retry."
                    ).strip()

            log.info(
                "[Brain-Route] Decision: action=%s reasoning='%s'",
                decision["action"],
                decision.get("reasoning", "")[:80],
            )
            return decision

        except Exception as exc:
            log.exception("[Brain-Route] Claude call failed: %s", exc)
            capture_exception("brain_route", exc)
            agents_run = self._memory.agents_run()
            if "ResearchAgent" not in agents_run:
                fallback_action = "run_research"
            elif "FinancialAnalystAgent" not in agents_run:
                fallback_action = "run_financial"
            elif "SentimentAgent" not in agents_run:
                fallback_action = "run_sentiment"
            else:
                fallback_action = "finalise"
            fallback_action = normalize_route_action(fallback_action, state)
            log.warning("[Brain-Route] Falling back to action='%s'.", fallback_action)
            return {
                "action":            fallback_action,
                "reasoning":         f"Fallback: Brain API call failed ({exc}).",
                "directive_updates": {},
            }

    async def _brain_evaluate(
        self,
        agent_name:  str,
        state:       SharedManagerState,
        memory_ctx:  dict[str, Any],
    ) -> EvaluationFeedback:
    
        if agent_name == "ResearchAgent":
            output_summary = {
                "chunks_count": len(state.get("aggregated_research_context", [])),
                "sample": [
                    extract_chunk_text(c)[:200]
                    for c in state.get("aggregated_research_context", [])[:2]
                ],
            }
            if output_summary["chunks_count"] < 3:
                issue = "Research output contains fewer than 3 validated chunks."
                return EvaluationFeedback(
                    step="research", timestamp=time.time(), passed=False,
                    score=30, issues=[issue], next_action="rerun_research",
                    raw_verdict=json.dumps({"passed": False, "issues": [issue]}),
                )
        elif agent_name == "FinancialAnalystAgent":
            fm = state.get("financial_metrics_summary", {})
            deterministic_validation = fm.get("financial_sanity") or evaluate_financial_sanity(fm)
            output_summary = {
                "ticker":            fm.get("ticker"),
                "composite_score":   fm.get("composite_score", {}).get("score"),
                "grade":             fm.get("composite_score", {}).get("grade"),
                "validation_passed": fm.get("validation_passed"),
                "extraction_errors": fm.get("extraction_errors", []),
                "pe_ratio":          fm.get("pe_ratio", {}).get("pe_ratio"),
                "forecast":          fm.get("forecast", {}),
                "dcf_valuation":     fm.get("dcf_valuation", {}),
                "reverse_dcf":       fm.get("reverse_dcf", {}),
                "deterministic_financial_validation": deterministic_validation,
            }
            if not deterministic_validation["passed"]:
                issues = [item["message"] for item in deterministic_validation["critical_failures"]]
                verdict = {
                    "passed": False,
                    "score": deterministic_validation["score"],
                    "issues": issues,
                    "next_action": "rerun_financial",
                    "reasoning": "Deterministic critical financial checks failed before LLM evaluation.",
                    "deterministic_validation": deterministic_validation,
                }
                return EvaluationFeedback(
                    step="financial", timestamp=time.time(), passed=False,
                    score=verdict["score"], issues=issues,
                    next_action="rerun_financial", raw_verdict=json.dumps(verdict),
                )
        else:
            sm = state.get("sentiment_analysis_summary", {})
            sentiment_status = sm.get("status", "ERROR")
            deterministic_quality_passed = sentiment_result_quality_passed(sm)
            output_summary = {
                "status":            sentiment_status,
                "execution_success": sm.get("execution_success"),
                "quality_passed":    deterministic_quality_passed,
                "quality_failures":  sm.get("quality_failures", []),
                "polarity":          sm.get("polarity"),
                "fear_greed_score":  sm.get("fear_greed_score"),
                "fear_greed_label":  sm.get("fear_greed_label"),
                "overall_sentiment": sm.get("overall_sentiment"),
                "conviction_level":  sm.get("conviction_level"),
                "total_chunks":      sm.get("total_chunks_analyzed"),
                "extraction_errors": sm.get("extraction_errors", []),
                "prior_attempt_errors": sm.get("prior_attempt_errors", []),
                "attempts":          sm.get("attempts", []),
            }
            if not deterministic_quality_passed:
                detail = "; ".join(str(item) for item in sm.get("quality_failures", []))
                issue = (
                    f"Sentiment quality failed (status={sentiment_status})"
                    + (f": {detail}" if detail else ".")
                )
                verdict = {
                    "passed": False, "score": 40 if sentiment_status == "LOW_COVERAGE" else 0,
                    "issues": [issue], "next_action": "rerun_sentiment",
                    "reasoning": (
                        "Execution completion is separate from quality: missing score, "
                        "critical errors, causal mismatch, or insufficient independent "
                        "sources cannot pass sentiment quality."
                    ),
                }
                return EvaluationFeedback(
                    step="sentiment", timestamp=time.time(), passed=False,
                    score=verdict["score"], issues=[issue], next_action="rerun_sentiment",
                    raw_verdict=json.dumps(verdict),
                )

        user_content = (
            f"AGENT NAME: {agent_name}\n\n"
            f"OUTPUT SUMMARY:\n{json.dumps(output_summary, indent=2)}\n\n"
            f"SESSION HISTORY:\n{json.dumps(memory_ctx['short_term'], indent=2)}\n\n"
            "Evaluate this agent's output and recommend the next action."
        )

        self._memory.add_message(role="user", content=user_content)
        log.info("[Brain-Evaluate] Evaluating %s output...", agent_name)

        try:
            response = await self._llm.messages.create(
                model=self._model,
                temperature=0.0,
                max_tokens=768,
                system=EVALUATOR_SYSTEM_PROMPT,
                messages=self._memory.get_messages(),
            )
            if response.stop_reason == "max_tokens":
                log.warning("[Brain-Evaluate] Response truncated by max_tokens — JSON may be invalid.")
            raw = response.content[0].text.strip()
            self._memory.add_message(role="assistant", content=raw)
            raw = raw.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(raw)

        except Exception as exc:
            log.exception("[Brain-Evaluate] Evaluation failed: %s", exc)
            capture_exception("brain_evaluate", exc, tags={"agent_name": agent_name})
            step_key = {
                "ResearchAgent": "research",
                "FinancialAnalystAgent": "financial",
                "SentimentAgent": "sentiment",
            }.get(agent_name, "")
            reusable = agent_result_is_valid(step_key, state)
            verdict = {
                "passed":      reusable,
                "score":       50 if reusable else 0,
                "issues":      [f"Evaluation API call failed: {exc}"],
                "next_action": (
                    next_pipeline_action(step_key, state)
                    if reusable else f"rerun_{step_key}"
                ),
                "reasoning": (
                    "The evaluator failed, but the existing deterministic result contract is valid and was reused."
                    if reusable else "The evaluator failed and no valid reusable result exists."
                ),
            }

        step_map = {
            "ResearchAgent":         "research",
            "FinancialAnalystAgent": "financial",
            "SentimentAgent":        "sentiment",
        }
        feedback = EvaluationFeedback(
            step=        step_map.get(agent_name, agent_name.lower()),
            timestamp=   time.time(),
            passed=      bool(verdict.get("passed", False)),
            score=       int(verdict.get("score", 0)),
            issues=      verdict.get("issues", []),
            next_action= verdict.get("next_action", "finalise"),
            raw_verdict= json.dumps(verdict),
        )
        log.info(
            "[Brain-Evaluate] %s verdict: passed=%s score=%d next=%s",
            agent_name, feedback.passed, feedback.score, feedback.next_action,
        )
        return feedback

    async def _brain_finalise(self, state: SharedManagerState, regeneration_context: list[dict] | None = None) -> str:
        
        all_chunks = state.get("aggregated_research_context", [])
        
        synthesis_chunks = [
            c for c in all_chunks
            if "[SYNTHESIZED RESEARCH SUMMARY" in extract_chunk_text(c)[:60]
        ]

        if synthesis_chunks:
            log.info(
                "_brain_finalise: using SYNTHESIZED summary for research_lines "
                "(%d chunk(s), skipping raw filing/news fair-budgeting entirely)",
                len(synthesis_chunks),
            )
            research_lines = "\n".join(
                f"  - {extract_chunk_text(c)[:6000]}" for c in synthesis_chunks
            )
        else:
            log.info(
                "_brain_finalise: no synthesis chunk found — falling back to "
                "raw filing/news chunk fair-budgeting"
            )
            
            filing_chunks = [
                c for c in all_chunks
                if "[TOOL: sec_edgar_filing]" in extract_chunk_text(c)[:150]
            ]
            news_chunks = [
                c for c in all_chunks
                if "[TOOL: news_search]" in extract_chunk_text(c)[:150]
            ]
            other_chunks = [c for c in all_chunks if c not in filing_chunks and c not in news_chunks]
            research_sample = filing_chunks[:2] + news_chunks[:2] + other_chunks[:3]
            research_lines = "\n".join(
                f"  - {format_filing_chunk_fairly(extract_chunk_text(c))}" if c in filing_chunks
                else f"  - {format_news_chunk_fairly(extract_chunk_text(c))}" if c in news_chunks
                else f"  - {extract_chunk_text(c)[:200]}"
                for c in research_sample
            )

        fm = state.get("financial_metrics_summary", {})
        sm = state.get("sentiment_analysis_summary", {})

        current_ratio_metric = fm.get("current_ratio", {})
        if not isinstance(current_ratio_metric, dict):
            current_ratio_metric = {"current_ratio": current_ratio_metric}
        de_ratio_metric = fm.get("de_ratio", {})
        if not isinstance(de_ratio_metric, dict):
            de_ratio_metric = {"de_ratio": de_ratio_metric}
        ocf_coverage_metric = fm.get("ocf_ratio", {})
        if not isinstance(ocf_coverage_metric, dict):
            ocf_coverage_metric = {"ocf_ratio": ocf_coverage_metric}

        fin_summary = {
            "ticker":            fm.get("ticker"),
            "company_name":      fm.get("company_name"),
            "sector":            fm.get("sector"),
            "current_price":     fm.get("current_price"),
            "market_cap":        fm.get("market_cap"),
            "composite_score":   fm.get("composite_score", {}),
            "pe_ratio":          fm.get("pe_ratio", {}).get("pe_ratio"),
            "pe_ratio_period":   fm.get("pe_ratio", {}).get("_period"),
            "roe":               fm.get("roe", {}).get("roe_pct"),
            "roe_period":        fm.get("roe", {}).get("_period"),
            "net_margin":        fm.get("net_margin", {}).get("net_margin_pct"),
            "net_margin_period": fm.get("net_margin", {}).get("_period"),
            # Preserve legacy scalars while forwarding the complete additive
            # recovery contract for newer consumers.
            "current_ratio":     current_ratio_metric.get("current_ratio"),
            "current_ratio_metric": current_ratio_metric,
            "de_ratio":          de_ratio_metric.get("de_ratio"),
            "de_ratio_period":   de_ratio_metric.get("_period"),
            "de_ratio_metric":   de_ratio_metric,
            "revenue_cagr":      fm.get("revenue_cagr", {}).get("cagr_pct"),
            "revenue_cagr_period": fm.get("revenue_cagr", {}).get("_period"),
            "forward_pe":        fm.get("forward_pe"),
            "peg_ratio":         fm.get("peg_ratio"),
            "peer_comparison":   fm.get("peer_comparison", {}),
            "dcf_valuation":     fm.get("dcf_valuation", {}),
            "dcf_monte_carlo":   fm.get("dcf_monte_carlo", {}),
            "capital_allocation": fm.get("capital_allocation", {}),
            "earnings_quality":   fm.get("earnings_quality", {}),
            "forecast":           fm.get("forecast", {}),
            "financial_facts":    fm.get("financial_facts", []),
            "financial_sanity":   fm.get("financial_sanity", {}),
            "validation_passed": fm.get("validation_passed"),
            "net_debt_to_ebitda":        fm.get("net_debt_to_ebitda", {}).get("net_debt_to_ebitda"),
            "net_debt_to_ebitda_period": fm.get("net_debt_to_ebitda", {}).get("_period"),
            "fcf_margin":                fm.get("fcf_margin", {}).get("fcf_margin_pct"),
            "fcf_margin_period":         fm.get("fcf_margin", {}).get("_period"),
            "fcf_yield":                 fm.get("fcf_yield", {}).get("fcf_yield_pct"),
            "fcf_yield_period":          fm.get("fcf_yield", {}).get("_period"),
            "ocf_ratio":                 ocf_coverage_metric.get("ocf_ratio"),
            "ocf_ratio_period":          ocf_coverage_metric.get("_period"),
            "ocf_coverage_metric":       ocf_coverage_metric,
            "reverse_dcf":               fm.get("reverse_dcf", {}),
            "roic":                      fm.get("roic", {}).get("roic_pct"),
            "roic_period":               fm.get("roic", {}).get("_period"),
            "roic_is_approximation":     fm.get("roic", {}).get("is_approximation"),
            "roic_note":                 fm.get("roic", {}).get("note"),
            "roce":                      fm.get("roce", {}).get("roce_pct"),
            "roce_period":               fm.get("roce", {}).get("_period"),
            "debt_ratio":                fm.get("debt_ratio", {}).get("debt_ratio"),
            "debt_ratio_period":         fm.get("debt_ratio", {}).get("_period"),
            "equity_ratio":              fm.get("equity_ratio", {}).get("equity_ratio"),
            "equity_ratio_period":       fm.get("equity_ratio", {}).get("_period"),
            "altman_z_score":            fm.get("altman_z_score", {}),

            "consistency_checks": fm.get("consistency_checks", {}),
            "data_quality":       fm.get("data_quality", {}),
        }
        sent_summary = {
            "fear_greed_score":  sm.get("fear_greed_score"),
            "fear_greed_label":  sm.get("fear_greed_label"),
            "overall_sentiment": sm.get("overall_sentiment"),
            "conviction_level":  sm.get("conviction_level"),
            "model_agreement":   sm.get("model_agreement"),
            "narrative":         sm.get("narrative"),
            "risk_flags":        sm.get("risk_flags", []),
            "status":            sm.get("status"),
            "aspect_observations": sm.get("aspect_observations", []),
            "causal_aspects_supported": sm.get("causal_aspects_supported", []),
        }

        canonical_payload = build_canonical_synthesis_payload(
            state,
            research_evidence=research_lines,
        )
       
        state["canonical_synthesis_payload"] = canonical_payload

        user_content = (
            f"TASK QUERY:\n{state.get('task_query', '')}\n\n"
            "CANONICAL SYNTHESIS PAYLOAD (SOLE FACTUAL AUTHORITY):\n"
            f"{json.dumps(canonical_payload, indent=2)}\n\n"
            "SOURCE RULE: obey source_precedence; never let excluded legacy text "
            "override structured values. Preserve canonical conflicts without guessing.\n\n"
            + (f"MANDATORY REGENERATION FINDINGS:\n{json.dumps(regeneration_context, indent=2)}\nCorrect every finding; do not repeat the invalid wording.\n\n" if regeneration_context else "")
            + "EXECUTION HISTORY:\n"
            + json.dumps(state.get("agent_execution_history", []), indent=2)
            + "\n\nSynthesize the final investment analysis report."
        )

        log.info("[Brain-Finalise] Synthesising final report...")
        try:
            response = await self._llm.messages.create(
                model=self._model,
                max_tokens=6144,
                temperature=0.0,
                system=FINALISER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            report = response.content[0].text.strip()
            if response.stop_reason == "max_tokens":
                log.warning("[Brain-Finalise] Report may be truncated by max_tokens (%d chars).", len(report))
                report += "\n\n[REPORT TRUNCATED BEFORE VALIDATED COMPLETION]"
            log.info("[Brain-Finalise] Report generated (%d chars).", len(report))
            return report

        except Exception as exc:
            log.exception("[Brain-Finalise] Synthesis failed: %s", exc)
            capture_exception("brain_finalise", exc)
            return (
                f"[REPORT GENERATION FAILED — {exc}]\n\n"
                f"Raw Financial Score: {fm.get('composite_score', {}).get('score')}\n"
                f"Raw Sentiment: {sm.get('fear_greed_label')}\n"
            )

    # =========================================================================
    # STEP 4 — EXECUTION (Agent Dispatch)
    # =========================================================================

    async def _dispatch(
        self,
        action: str,
        state:  SharedManagerState,
    ) -> SharedManagerState:
       
        agent_key  = action.removeprefix("rerun_").removeprefix("run_")
        agent      = self._agents.get(agent_key)

        if agent is None:
            log.error("[Dispatch] Unknown agent key '%s' from action '%s'.", agent_key, action)
            return state

        agent_class = type(agent).__name__
        directives  = state.get("manager_directives", {})

        record = self._memory.log_dispatch(
            agent_name=agent_class,
            directives=directives,
        )

        ts = datetime.now(timezone.utc).isoformat()
        state["orchestrator_logs"].append(
            f"[{ts}] [DISPATCH] → {agent_class} (action={action})"
        )

        log.info("[Dispatch] Dispatching %s (action=%s)...", agent_class, action)
        t0 = time.time()

        session_id = directives.get("_progress_session_id")
        friendly = {
            "research": "Research Agent",
            "financial": "Financial Analyst Agent",
            "sentiment": "Sentiment Agent",
        }.get(agent_key, agent_class)
        _publish_progress(
            session_id, "dispatch_start", agent=agent_key,
            message=f"Manager is dispatching {friendly}...",
            detail={"agent_class": agent_class, "action": action},
        )

        try:
            add_breadcrumb("manager_agent.dispatch", f"Dispatching {agent_class}",
                           {"agent_class": agent_class, "action": action})
            state   = await agent.run(state)
            elapsed = round(time.time() - t0, 2)

            result_keys = infer_result_keys(agent_key, state)
            quality_passed = (
                sentiment_result_quality_passed(
                    state.get("sentiment_analysis_summary", {}) or {}
                )
                if agent_key == "sentiment"
                else None
            )

            record.outcome     = "success"
            record.duration_s  = elapsed
            record.result_keys = result_keys

            state["agent_execution_history"].append({
                "agent_name":    agent_class,
                "dispatched_at": record.dispatched_at,
                "outcome":       "success",
                "execution_success": True,
                "quality_passed": quality_passed,
                "duration_s":    elapsed,
                "result_keys":   result_keys,
                "error_message": None,
            })
            state["orchestrator_logs"].append(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"[SUCCESS] {agent_class} completed in {elapsed}s — keys={result_keys}"
            )
            log.info("[Dispatch] %s completed in %.2fs.", agent_class, elapsed)
            _publish_progress(
                session_id, "dispatch_end", agent=agent_key,
                message=f"{friendly} completed successfully ({elapsed}s) — returning to Manager",
                detail={
                    "outcome": "success",
                    "execution_success": True,
                    "quality_passed": quality_passed,
                    "duration_s": elapsed,
                    "result_keys": result_keys,
                },
            )

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            record.outcome       = "error"
            record.duration_s    = elapsed
            record.error_message = str(exc)

            capture_exception("manager_agent.dispatch", exc,
                              tags={"agent_name": agent_class, "action": action})

            state["agent_execution_history"].append({
                "agent_name":    agent_class,
                "dispatched_at": record.dispatched_at,
                "outcome":       "error",
                "execution_success": False,
                "quality_passed": False if agent_key == "sentiment" else None,
                "duration_s":    elapsed,
                "result_keys":   [],
                "error_message": str(exc),
            })
            state["orchestrator_logs"].append(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"[ERROR] {agent_class} failed: {exc}"
            )
            log.exception("[Dispatch] %s raised an exception.", agent_class)
            _publish_progress(
                session_id, "dispatch_end", agent=agent_key,
                message=f"{friendly} failed — returning to Manager",
                detail={
                    "outcome": "error",
                    "execution_success": False,
                    "quality_passed": False if agent_key == "sentiment" else None,
                    "duration_s": elapsed,
                    "error": str(exc),
                },
            )

        return state

    # =========================================================================
    # STEP 5 — PERSIST
    # =========================================================================

    def _persist(
        self,
        agent_key:  str,
        state:      SharedManagerState,
        evaluation: EvaluationFeedback,
    ) -> None:
       
        directives = state.get("manager_directives", {})
        ticker     = directives.get("ticker")

        if agent_key == "research" and ticker:
            self._memory.store_heuristic(
                f"{ticker}_research_chunks",
                len(state.get("aggregated_research_context", [])),
            )

        elif agent_key == "financial" and ticker:
            fm = state.get("financial_metrics_summary", {})
            self._memory.store_ticker_insight(ticker, {
                "last_composite_score": fm.get("composite_score", {}).get("score"),
                "last_grade":           fm.get("composite_score", {}).get("grade"),
                "sector":               fm.get("sector"),
                "validation_passed":    fm.get("validation_passed"),
            })
            self._memory.store_heuristic(
                f"{ticker}_financial_score",
                fm.get("composite_score", {}).get("score"),
            )

        elif agent_key == "sentiment" and ticker:
            sm = state.get("sentiment_analysis_summary", {})
            self._memory.store_ticker_insight(ticker, {
                "last_fear_greed_score": sm.get("fear_greed_score"),
                "last_fear_greed_label": sm.get("fear_greed_label"),
                "last_sentiment":        sm.get("overall_sentiment"),
                "last_conviction":       sm.get("conviction_level"),
            })

        log.info("[Persist] Memory updated after %s step.", agent_key)

    # =========================================================================
    # LANGGRAPH NODES
    # =========================================================================

    @traceable(name="hydrate", run_type="chain")
    async def _node_hydrate(self, g: ManagerGraphState) -> dict:
    
        shared     = g["shared_state"]
        directives = dict(shared.get("manager_directives", {}))

        if "search_depth" not in directives:
            saved_depth = self._memory.get_preference("search_depth")
            if saved_depth:
                directives["search_depth"] = saved_depth
                shared["manager_directives"] = directives

        ticker = directives.get("ticker")
        log.info("[Node:Hydrate] session=%s ticker=%s", g["session_id"], ticker)
        _publish_progress(
            g["session_id"], "hydrate", agent="manager",
            message=f"Initial state prepared for {ticker or 'analysis'}",
            detail={"ticker": ticker},
        )
        return {"shared_state": shared, "ticker": ticker}

    @traceable(name="brain_route", run_type="chain")
    async def _node_brain_route(self, g: ManagerGraphState) -> dict:
        
        new_counter = g["loop_counter"] + 1
        shared      = g["shared_state"]
        ticker      = g["ticker"]

        log.info("[Node:BrainRoute] iteration %d / %d", new_counter, self._max_routing_loops)

        memory_ctx = self._recall(ticker=ticker)
        decision   = await self._brain_route(
            state=shared,
            memory_recall=memory_ctx,
            loop_counter=new_counter,
        )
        action            = decision["action"]
        directive_updates = decision.get("directive_updates", {})

        if directive_updates:
            shared["manager_directives"].update(directive_updates)
            log.info("[Node:BrainRoute] Directive updates: %s", directive_updates)

        ts = datetime.now(timezone.utc).isoformat()
        shared["orchestrator_logs"].append(
            f"[{ts}] [ROUTE] loop={new_counter} action={action} "
            f"reasoning={decision.get('reasoning', '')[:80]}"
        )

        updated_ticker = shared["manager_directives"].get("ticker", ticker)
        _publish_progress(
            g["session_id"], "route", agent="manager",
            message=f"Manager decided: {action}",
            detail={"action": action, "reasoning": decision.get("reasoning", ""), "loop": new_counter},
        )
        return {
            "loop_counter": new_counter,
            "last_action":  action,
            "shared_state": shared,
            "ticker":       updated_ticker,
        }

    @traceable(name="dispatch", run_type="chain")
    async def _node_dispatch(self, g: ManagerGraphState) -> dict:
      
        action    = g["last_action"]
        shared    = g["shared_state"]
        agent_key = action.removeprefix("rerun_").removeprefix("run_")

        log.info("[Node:Dispatch] action=%s agent_key=%s", action, agent_key)
        if agent_result_is_valid(agent_key, shared):
            shared["orchestrator_logs"].append(
                f"[{datetime.now(timezone.utc).isoformat()}] [REUSE] "
                f"Skipped {agent_key}; a valid result is already present."
            )
            return {"shared_state": shared, "last_agent_key": agent_key}
        shared = await self._dispatch(action=action, state=shared)

        return {"shared_state": shared, "last_agent_key": agent_key}

    @traceable(name="evaluate", run_type="chain")
    async def _node_evaluate(self, g: ManagerGraphState) -> dict:
      
        agent_key  = g["last_agent_key"]
        shared     = g["shared_state"]
        ticker     = g["ticker"]

        agent_class_map = {
            "research":  "ResearchAgent",
            "financial": "FinancialAnalystAgent",
            "sentiment": "SentimentAgent",
        }
        agent_class = agent_class_map.get(agent_key, agent_key)
        memory_ctx  = self._recall(ticker=ticker)

        evaluation = await self._brain_evaluate(
            agent_name=agent_class,
            state=shared,
            memory_ctx=memory_ctx,
        )

        # Single write point — _persist() must NOT call add_evaluation() again
        self._memory.add_evaluation(evaluation)

        snapshot = feedback_to_snapshot(evaluation)

        log.info(
            "[Node:Evaluate] %s → passed=%s score=%d next=%s",
            agent_class, evaluation.passed, evaluation.score, evaluation.next_action,
        )
        _publish_progress(
            g["session_id"], "evaluate", agent="manager",
            message=(
                f"Evaluating {agent_class} output: "
                f"{'passed ✓' if evaluation.passed else 'needs revision'} (score={evaluation.score})"
            ),
            detail={
                "target_agent": agent_key,
                "passed": evaluation.passed,
                "score": evaluation.score,
                "next_action": evaluation.next_action,
            },
        )
        return {
            "evaluation_passed": evaluation.passed,
            "last_evaluation":   snapshot,
        }

    @traceable(name="persist", run_type="chain")
    async def _node_persist(self, g: ManagerGraphState) -> dict:
       
        agent_key = g["last_agent_key"]
        shared    = g["shared_state"]

        snapshot: EvaluationSnapshot | None = g.get("last_evaluation")

        if snapshot is not None:
            
            fb = EvaluationFeedback(
                step=        snapshot["step"],
                timestamp=   time.time(),
                passed=      snapshot["passed"],
                score=       snapshot["score"],
                issues=      snapshot.get("issues", []),
                next_action= snapshot["next_action"],
                raw_verdict= "{}",
            )
            self._persist(agent_key=agent_key, state=shared, evaluation=fb)

        # Determine updated action for routing
        updated_action = g["last_action"]
        if not g["evaluation_passed"] and snapshot is not None:
            next_action    = snapshot["next_action"]
            next_agent_key = next_action.removeprefix("rerun_").removeprefix("run_")

            if next_agent_key == agent_key:
            
                updated_action = "rerun_" + next_agent_key
                log.info(
                    "[Node:Persist] Eval failed — retrying same agent '%s', "
                    "action='%s'", agent_key, updated_action,
                )
            else:
             
                updated_action = next_action
                log.info(
                    "[Node:Persist] Eval failed on '%s' — evaluator recommends "
                    "advancing to a different agent, action='%s' (label "
                    "preserved, not forced to rerun_*)",
                    agent_key, updated_action,
                )

        if updated_action.startswith("rerun_") and not is_transient_failure([
            snapshot.get("issues", []) if snapshot else [],
            _agent_failure_context(agent_key, shared),
        ]):
            updated_action = next_pipeline_action(agent_key, shared)
            log.info(
                "[Node:Persist] Suppressed non-transient retry for '%s'; advancing as '%s'.",
                agent_key, updated_action,
            )

        log.info("[Node:Persist] Memory persisted for agent_key=%s", agent_key)
        return {"last_action": updated_action}

    @traceable(name="finalise", run_type="chain")
    async def _node_finalise(self, g: ManagerGraphState) -> dict:
    
        shared     = g["shared_state"]
        ticker     = g["ticker"]
        session_id = g["session_id"]

        _publish_progress(
            session_id, "finalise", agent="manager",
            message="Manager is writing the final investment report...",
        )
        structured_findings = detect_material_period_conflicts(
            shared.get("financial_metrics_summary", {}).get("financial_facts", [])
        )
        if any(item.get("blocking") for item in structured_findings):
            shared["qa_period_findings"] = structured_findings
            shared["recommendation_gate"] = build_recommendation_gate(shared)
            critical = [{"check":"period_consistency","message":str(item)} for item in structured_findings if item.get("blocking")]
            validation = {"passed":False,"critical_findings":critical,"warnings":[],"required_regeneration_context":critical,"publication_status":"REGENERATION_REQUIRED","checks":{}}
            shared["final_publication_validation"] = validation
            outcome = assemble_publication("", validation, shared)
            shared.update(outcome)
            shared["execution_status"] = execution_status_from_publication(
                shared.get("publication_status"), validation_passed=False
            )
            log.error("[Node:Finalise] Material period conflict limited publication: %s", structured_findings)
            return {"shared_state": shared}

        shared["recommendation_gate"] = build_recommendation_gate(shared)
        shared["enforce_report_sections"] = True
        final_report = await self._brain_finalise(shared)
        final_report = normalize_validation_aware_narrative(
            final_report, shared.get("financial_metrics_summary", {})
        )
        validation = validate_final_publication(final_report, shared)
        shared["final_publication_validation"] = validation
        shared["publication_validation_attempts"] = [{"attempt": 1, "validation": validation}]
        if not validation["passed"]:
            first_candidate = final_report
            final_report = await self._brain_finalise(shared, validation["required_regeneration_context"])
            final_report = normalize_validation_aware_narrative(
                final_report, shared.get("financial_metrics_summary", {})
            )
            second = validate_final_publication(final_report, shared)
            shared["final_publication_validation"] = second
            shared["publication_validation_attempts"].append({"attempt": 2, "validation": second})
            if not second["passed"]:
                shared["blocked_final_report"] = final_report
                shared["first_blocked_final_report"] = first_candidate
                shared.update(assemble_publication(final_report, second, shared))
                shared["execution_status"] = execution_status_from_publication(
                    shared.get("publication_status"), validation_passed=False
                )
                return {"shared_state": shared}

     
        try:
            period_findings = check_narration_vs_period(
                final_report,
                shared.get("financial_metrics_summary", {}),
            )
        except Exception as exc:
            period_findings = []
            log.warning(
                "[Node:Finalise] period-consistency check itself failed "
                "(non-fatal, report still returned): %s", exc,
            )
            capture_exception("qa.period_consistency", exc)

        shared["qa_period_findings"] = structured_findings + period_findings
        if period_findings:
            log.warning(
                "[Node:Finalise] %d period-consistency issue(s) found in "
                "%s report: %s",
                len(period_findings), ticker or "?", period_findings,
            )
            capture_message(
                "qa.period_consistency",
                f"Period-consistency issue(s) in {ticker or '?'} report "
                f"({len(period_findings)} finding(s))",
                level="warning",
            )
        blocking_narration_findings = [
            finding for finding in period_findings if finding.get("severity") == "HIGH"
        ]
        if blocking_narration_findings:
            shared["blocked_final_report"] = final_report
            extra=[{"check":"period_consistency","message":str(x)} for x in blocking_narration_findings]
            validation={**shared.get("final_publication_validation",{}),"passed":False}
            validation["publication_status"] = "REGENERATION_REQUIRED"
            validation["critical_findings"]=[*validation.get("critical_findings",[]),*extra]
            validation["required_regeneration_context"]=validation["critical_findings"]
            shared["final_publication_validation"]=validation
            log.error("[Node:Finalise] Material period conflict limited report publication.")
        shared.update(assemble_publication(final_report, shared.get("final_publication_validation",validation), shared))
        shared["execution_status"] = execution_status_from_publication(
            shared.get("publication_status"),
            validation_passed=shared.get("final_publication_validation", {}).get("passed"),
        )
        # -- end QA: period-consistency check ----------------------------------

        if ticker:
            self._memory.store_ticker_insight(ticker, {
                "last_task_query":          shared.get("task_query", ""),
                "last_final_report_length": len(final_report),
            })
        self._memory.store_heuristic(
            f"session_{session_id}_loops_used", g["loop_counter"]
        )
        try:
            self._memory.persist_long_term()
        except Exception as exc:
            log.warning("[Node:Finalise] persist_long_term() failed (non-fatal): %s", exc)
            capture_exception("memory.persist_long_term", exc)

        log.info(
            "[Node:Finalise] Report generated (%d chars). Long-term memory persisted.",
            len(final_report),
        )
        return {"shared_state": shared}

    @traceable(name="abort", run_type="chain")
    async def _node_abort(self, g: ManagerGraphState) -> dict:
        """
        NODE: abort — Guardrail / Error Exit.

        Reached when Brain returns action="abort" or loop_counter exceeds
        max_routing_loops. Logs the abort and persists long-term memory.
        """
        shared = g["shared_state"]
        loop   = g["loop_counter"]

        ts = datetime.now(timezone.utc).isoformat()
        shared["orchestrator_logs"].append(
            f"[{ts}] [ABORT] Orchestration aborted at loop {loop} "
            f"(max_routing_loops={self._max_routing_loops})."
        )
        try:
            self._memory.persist_long_term()
        except Exception as exc:
            log.warning("[Node:Abort] persist_long_term() failed (non-fatal): %s", exc)
            capture_exception("memory.persist_long_term", exc)

        log.warning(
            "[Node:Abort] Orchestration aborted at loop %d / %d.",
            loop, self._max_routing_loops,
        )
        validation = {
            "passed": False,
            "critical_findings": [{
                "check": "orchestration_status",
                "message": "The orchestration aborted before successful publication.",
            }],
            "warnings": [],
            "required_regeneration_context": [],
            "publication_status": "ABORTED",
            "checks": {},
        }
        shared["final_publication_validation"] = validation
        shared.update(assemble_publication(shared.get("final_report", ""), validation, shared))
        shared["execution_status"] = execution_status_from_publication(
            shared.get("publication_status"), validation_passed=False, aborted=True
        )
        return {"shared_state": shared}

    # =========================================================================
    # CONDITIONAL EDGE ROUTERS
    # =========================================================================

    def _should_route(self, g: ManagerGraphState) -> str:
        """
        CONDITIONAL EDGE — Route after brain_route node.

        Returns
        -------
        "dispatch"  for run_* / rerun_* actions
        "finalise"  when Brain decides pipeline is complete
        "abort"     on guardrail hit or unknown action
        """
        action  = g["last_action"]
        counter = g["loop_counter"]

        if counter >= self._max_routing_loops:
            log.warning(
                "[Router] Guardrail hit: loop=%d >= max=%d → abort.",
                counter, self._max_routing_loops,
            )
            return "abort"

        if action == "finalise":
            return "finalise"

        if action == "abort":
            return "abort"

        if action.startswith("run_") or action.startswith("rerun_"):
            return "dispatch"

        log.warning("[Router] Unknown action '%s' → abort.", action)
        return "abort"

    def _should_continue_after_persist(self, g: ManagerGraphState) -> str:
        
        if g["loop_counter"] >= self._max_routing_loops:
            return "abort"

        if not g["evaluation_passed"]:
            action = g["last_action"]
            if action.startswith("rerun_"):
                log.info(
                    "[Router-Persist] Eval failed → rerun via dispatch (action=%s).", action
                )
                return "dispatch"

        return "brain_route"

    # =========================================================================
    # GRAPH BUILDER
    # =========================================================================

    def _build_graph(self):
        
        builder = StateGraph(ManagerGraphState)

        builder.add_node("hydrate",     self._node_hydrate)
        builder.add_node("brain_route", self._node_brain_route)
        builder.add_node("dispatch",    self._node_dispatch)
        builder.add_node("evaluate",    self._node_evaluate)
        builder.add_node("persist",     self._node_persist)
        builder.add_node("finalise",    self._node_finalise)
        builder.add_node("abort",       self._node_abort)

        builder.set_entry_point("hydrate")
        builder.add_edge("hydrate",  "brain_route")
        builder.add_edge("dispatch", "evaluate")
        builder.add_edge("evaluate", "persist")
        builder.add_edge("finalise", END)
        builder.add_edge("abort",    END)

        builder.add_conditional_edges(
            "brain_route",
            self._should_route,
            {
                "dispatch": "dispatch",
                "finalise": "finalise",
                "abort":    "abort",
            },
        )

        builder.add_conditional_edges(
            "persist",
            self._should_continue_after_persist,
            {
                "brain_route": "brain_route",
                "dispatch":    "dispatch",
                "abort":       "abort",
            },
        )

        log.info("[Graph] ManagerAgent StateGraph compiled — 7 nodes, 2 conditional edges.")
        return builder.compile()

    # =========================================================================
    # ENTRY POINT — run()
    # =========================================================================

    async def _run_monthly(
        self,
        *,
        task_query: str,
        directives: dict[str, Any],
        session_id: str,
    ) -> SharedManagerState:
        """Domain integration path; request lifecycle remains Stage 10A scope."""
        run_started_at = time.perf_counter()
        raw_context = directives.get("forecast_context")
        context = (
            raw_context if isinstance(raw_context, ForecastContext)
            else ForecastContext.from_dict(dict(raw_context or {}))
        )
        requested_ticker = str(directives.get("ticker") or context.ticker).upper()
        if requested_ticker != context.ticker:
            raise ValueError("manager ticker must match ForecastContext.ticker")
        run_id = validate_run_id(str(directives.get("run_id") or uuid.uuid4()))
        try:
            _publish_stage08_monthly(
                session_id, "forecast_context", context=context, run_id=run_id,
                started_at=run_started_at, message="STAGE08_CONTEXT_ACCEPTED",
            )
            monthly_directives = {
                **directives,
                "ticker": context.ticker,
                "forecast_context": context.to_dict(),
                "run_id": run_id,
                "_progress_session_id": session_id,
            }
            shared = self._hydrate_state(task_query, monthly_directives)
            shared["forecast_context"] = context.to_dict()
            shared["run_id"] = run_id

            specialist_started_at = time.perf_counter()
            _publish_stage08_monthly(
                session_id, "feature_start", context=context, run_id=run_id,
                started_at=specialist_started_at, message="STAGE08_SPECIALISTS_STARTED",
            )
            specialist_results = await run_monthly_specialists_concurrently(
                self._agents, shared,
            )
            shared, node_order = controlled_monthly_join(shared, specialist_results)
            specialist_slots = [
                shared.get(name) for name in (
                    "research_features", "financial_features", "sentiment_features",
                )
            ]
            specialist_evidence_count = sum(
                len(slot.get("evidence_ids", []))
                for slot in specialist_slots if isinstance(slot, dict)
                and isinstance(slot.get("evidence_ids"), list)
            )
            specialist_degraded = any(
                bool(slot.get("degraded")) or bool(slot.get("missing"))
                for slot in specialist_slots if isinstance(slot, dict)
            )
            _publish_stage08_monthly(
                session_id, "feature_complete", context=context, run_id=run_id,
                started_at=specialist_started_at,
                message="STAGE08_SPECIALISTS_JOINED",
                evidence_count=specialist_evidence_count,
                degraded_flags=("specialist_features_degraded",) if specialist_degraded else (),
            )

            market_started_at = time.perf_counter()
            _publish_stage08_monthly(
                session_id, "feature_start", context=context, run_id=run_id,
                started_at=market_started_at, message="STAGE08_MARKET_QUANT_STARTED",
            )
            if self._market_forecast_service is None or self._market_data_provider is None:
                shared["market_features"] = {
                    "version": "unavailable", "values": {}, "evidence_ids": [],
                    "missing": True, "degraded": True,
                    "errors": ["QUANT_SERVICE_NOT_CONFIGURED"],
                }
            else:
                market = await maybe_await(self._market_data_provider(context))
                market_features = await maybe_await(
                    self._market_forecast_service.build_features(
                        context=context, market=market,
                        research_features=shared.get("research_features"),
                        financial_features=shared.get("financial_features"),
                        sentiment_features=shared.get("sentiment_features"),
                    )
                )
                prediction = await maybe_await(
                    self._market_forecast_service.forecast(
                        context=context, market_features=market_features, run_id=run_id,
                    )
                )
                if not hasattr(market_features, "to_state_slot"):
                    raise TypeError("Quant market_features must provide to_state_slot()")
                market_slot = market_features.to_state_slot()
                market_slot = {
                    **market_slot,
                    "values": {
                        **dict(market_slot.get("values", {})),
                        "run_id": run_id,
                        "quant_prediction": dict(prediction),
                    },
                }
                shared["market_features"] = market_slot
            node_order.append("market_features")
            market_slot = shared["market_features"]
            market_evidence_count = len(market_slot.get("evidence_ids", []))
            market_degraded = bool(market_slot.get("missing")) or bool(market_slot.get("degraded"))
            _publish_stage08_monthly(
                session_id, "feature_complete", context=context, run_id=run_id,
                started_at=market_started_at, message="STAGE08_MARKET_QUANT_COMPLETED",
                evidence_count=market_evidence_count,
                degraded_flags=("market_features_degraded",) if market_degraded else (),
            )

            forecast_started_at = time.perf_counter()
            _publish_stage08_monthly(
                session_id, "monthly_forecast_start", context=context, run_id=run_id,
                started_at=forecast_started_at, message="STAGE08_FORECAST_STARTED",
            )
            shared = await self._monthly_forecast_agent.run(shared)
            node_order.extend(["monthly_forecast", "canonical_synthesis", "publication"])
            forecast = dict(shared.get("monthly_forecast", {}))
            shared["monthly_node_order"] = node_order
            shared["canonical_synthesis"] = {
                "forecast": {
                    key: forecast.get(key) for key in (
                        "status", "ticker", "as_of", "target_date",
                        "horizon_trading_days", "cutoff_policy", "benchmark",
                        "quality_gate_passed", "quality_gate_failures", "probability_positive",
                        "expected_return_21d", "expected_excess_return",
                        "confidence_interval", "confidence", "action", "versions",
                        "specialist_contributions", "decision_drivers", "narrative",
                    )
                },
                "dcf_semantics": "LONG_TERM_VALUATION_FEATURE_NOT_21_DAY_FORECAST",
            }
            shared["final_report"] = str(forecast.get("narrative") or "")
            # Request-local recall of this result; durable storage is owned by
            # the API. Prior forecasts never become Quant training features.
            try:
                self._memory.store_monthly_insight(
                    ticker=context.ticker, as_of=context.as_of,
                    horizon=context.horizon_trading_days,
                    analysis_mode=context.analysis_mode, schema_version=context.schema_version,
                    insight=forecast,
                )
            except Exception:
                log.warning("monthly_memory_cache_failed")
            insufficient = forecast.get("status") == "INSUFFICIENT_DATA"
            quality_blocked = forecast.get("quality_gate_passed") is not True
            if insufficient:
                shared["publication_status"] = "DATA_INSUFFICIENT"
                shared["execution_status"] = "failed"
            elif quality_blocked:
                shared["publication_status"] = "DATA_QUALITY_BLOCKED"
                shared["execution_status"] = "failed"
            elif forecast.get("degraded"):
                shared["publication_status"] = "PUBLISHED_WITH_WARNINGS"
                shared["execution_status"] = "completed"
            else:
                shared["publication_status"] = "PUBLISHED"
                shared["execution_status"] = "completed"
            evidence = forecast.get("evidence_ids")
            evidence_count = len(evidence) if isinstance(evidence, list) else 0
            degraded = bool(forecast.get("degraded"))
            _publish_stage08_monthly(
                session_id, "monthly_forecast_complete", context=context, run_id=run_id,
                started_at=run_started_at, message="STAGE08_FORECAST_COMPLETED",
                evidence_count=evidence_count,
                degraded_flags=("monthly_degraded",) if degraded else (),
            )
            if degraded:
                _publish_stage08_monthly(
                    session_id, "forecast_degraded", context=context, run_id=run_id,
                    started_at=run_started_at, message="STAGE08_FORECAST_DEGRADED",
                    evidence_count=evidence_count, degraded_flags=("monthly_degraded",),
                )
            return shared
        except Exception:
            _publish_stage08_monthly(
                session_id, "forecast_error", context=context, run_id=run_id,
                started_at=run_started_at, message="STAGE08_MONTHLY_FAIL_CLOSED",
                degraded_flags=("monthly_fail_closed",),
            )
            raise

    @traceable(
        name="ManagerAgent.run", run_type="chain",
        metadata=STAGE08_IMPLEMENTATION_IDENTITY,
        process_inputs=_manager_trace_inputs,
        process_outputs=_manager_trace_outputs,
    )
    async def run(
        self,
        task_query:         str,
        manager_directives: dict[str, Any] | None = None,
        user_preferences:   dict[str, Any] | None = None,
        client_session_id:  str | None = None,
    ) -> SharedManagerState:
        
        session_id  = client_session_id or str(uuid.uuid4())[:8]
        directives  = dict(manager_directives or {})
        preferences = dict(user_preferences or {})

        directives["_progress_session_id"] = session_id

        if "forecast_context" in directives:
            return await self._run_monthly(
                task_query=task_query, directives=directives, session_id=session_id,
            )

        self._memory.new_session(session_id=session_id, task_query=task_query)
        for k, v in preferences.items():
            self._memory.store_preference(k, v)

        log.info(
            "ManagerAgent.run() started — session=%s task='%s'",
            session_id, task_query[:80],
        )
        _publish_progress(
            session_id, "pipeline_start", agent="manager",
            message="Analysis started — Manager is preparing",
            detail={"task_query": task_query, "ticker": directives.get("ticker")},
        )

        shared_state = self._hydrate_state(
            task_query=task_query,
            manager_directives=directives,
        )

        initial: ManagerGraphState = {
            "shared_state":      shared_state,
            "loop_counter":      0,
            "last_action":       "",
            "last_agent_key":    "",
            "evaluation_passed": False,
            "last_evaluation":   None,
            "ticker":            directives.get("ticker"),
            "session_id":        session_id,
        }

        try:
            final: ManagerGraphState = await self._graph.ainvoke(
                initial,
                config={"recursion_limit": (self._max_routing_loops + 2) * 4},
            )
        except Exception as exc:
            log.exception("ManagerAgent graph execution failed: %s", exc)
            _publish_progress(
                session_id, "pipeline_error", agent="manager",
                message=f"Internal orchestration graph error: {exc}",
            )
            raise RuntimeError(f"ManagerAgent internal graph failed: {exc}") from exc

        final_shared = final["shared_state"]
        log.info(
            "ManagerAgent.run() complete — session=%s loops=%d report_chars=%d",
            session_id,
            final["loop_counter"],
            len(final_shared.get("final_report", "")),
        )
        _publish_progress(
            session_id, "pipeline_complete", agent="manager",
            message="Analysis complete — final report ready",
            detail={"loops": final["loop_counter"]},
        )
        return final_shared
