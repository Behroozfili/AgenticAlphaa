"""agents/research_agent.py — ResearchAgent
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import replace
from typing import Any, Literal

import anthropic
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.state import ResearchAgentState, SharedManagerState
from dotenv import load_dotenv
from langsmith import traceable
from core.forecast_contract import AnalysisMode, ForecastContext
from core.observability import elapsed_latency_ms, sentry_enabled
from core.progress_bus import publish as _publish_progress, session_from_directives
from tools.research_tools import (
    build_research_features, citation_reference_id, compact_lifecycle,
    eligible_evidence, rejection_counts, research_implementation_identity,
    research_trace_metadata,
)
from tools.research_tools.context_synthesizer import synthesize_research_context
from agents.prompts.prompt_research_agent import BRAIN_SYSTEM_PROMPT, CHECKER_SYSTEM_PROMPT
from agents.executor_helpers.search_executor_helpers import (RESEARCH_TOOL_NAMES,
                                                             TICKER_ARG_BY_TOOL,
                                                             ensure_ticker_argument, 
                                                             inject_execution_identity,
                                                             require_execution_identity,
                                                             tool_call_signature, 
                                                             sec_filing_doc_key, 
                                                             sec_filing_sections, 
                                                             build_sec_filing_coverage, 
                                                             format_skip_notice)

load_dotenv()

logger = logging.getLogger(__name__)

_TRACE_IDENTITY = research_implementation_identity()
_SAFE_AGENT_ERROR_CODES = frozenset({
    "RESEARCH_AGENT_FAIL_CLOSED", "RESEARCH_TOOL_FAIL_CLOSED",
    "RESEARCH_MCP_UNAVAILABLE", "RESEARCH_PLAN_INVALID",
    "RESEARCH_TOOL_NOT_ALLOWLISTED", "RESEARCH_EVIDENCE_UNAVAILABLE",
})


def _trace_identity_from_state(value: Any) -> dict[str, Any]:
    state = value if isinstance(value, dict) else {}
    directives = state.get("manager_directives") if isinstance(state.get("manager_directives"), dict) else {}
    context = state.get("forecast_context") or directives.get("forecast_context") or {}
    run_id = directives.get("run_id") or state.get("run_id")
    try:
        return research_trace_metadata(context, run_id)
    except (TypeError, ValueError):
        return dict(_TRACE_IDENTITY)


def _trace_inputs(operation: str):
    def process(inputs: dict[str, Any]) -> dict[str, Any]:
        state = inputs.get("shared_state") or inputs.get("state") or {}
        return {
            "operation": operation,
            "metadata": _trace_identity_from_state(state),
            "message_count": len(state.get("messages", [])) if isinstance(state, dict) else 0,
            "context_reference_count": len(state.get("context_chunks", [])) if isinstance(state, dict) else 0,
            "loop_counter": state.get("loop_counter", 0) if isinstance(state, dict) else 0,
            "resume": "not_applicable",
        }
    return process


def _trace_outputs(operation: str):
    def process(outputs: Any) -> dict[str, Any]:
        payload = outputs if isinstance(outputs, dict) else {}
        state = payload
        features = payload.get("research_features") if isinstance(payload.get("research_features"), dict) else {}
        evidence_ids = features.get("evidence_ids") if isinstance(features.get("evidence_ids"), list) else []
        chunks = payload.get("context_chunks") if isinstance(payload.get("context_chunks"), list) else []
        errors = features.get("errors") if isinstance(features.get("errors"), list) else []
        return {
            "operation": operation,
            "metadata": _trace_identity_from_state(state),
            "message_count": len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0,
            "context_reference_count": len(chunks),
            "evidence_count": len(evidence_ids) or sum(
                1 for item in chunks if isinstance(item, dict) and item.get("evidence_id")
            ),
            "error_code_count": sum(1 for code in errors if code in _SAFE_AGENT_ERROR_CODES),
            "missing": bool(features.get("missing", False)),
            "degraded": bool(features.get("degraded", False)),
            "resume": "not_applicable",
        }
    return process


_DEFAULT_MAX_LOOPS: int = 3

def _research_window_context(
    context: ForecastContext, directives: dict[str, Any],
) -> ForecastContext:
    """Narrow Research evidence without shrinking the market/finance history."""
    requested = directives.get("days_back")
    if requested is None:
        return context
    if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= 365:
        raise ValueError("manager days_back must be an integer within 1..365")
    return replace(context, lookback_days=min(context.lookback_days, requested))

_MCP_SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=[os.path.join(os.path.dirname(__file__), "..","tools","research_tools" ,"research_server.py")],
    env=None,   # inherits current environment (API keys etc.)
)


# ══════════════════════════════════════════════════════════════════
# ResearchAgent
# ══════════════════════════════════════════════════════════════════

class ResearchAgent:
   

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        model: str = "claude-haiku-4-5",
        max_loops: int = _DEFAULT_MAX_LOOPS,
        mcp_server_params: StdioServerParameters | None = None,
        llm_client: anthropic.Anthropic | None = None,
    ) -> None:
   
        self._llm = llm_client or anthropic.Anthropic(
            api_key=anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self._model    = model
        self._max_loops = max_loops
        self._mcp_params = mcp_server_params or _MCP_SERVER_PARAMS

        
        self._graph = self._build_graph()
        logger.info(
            "ResearchAgent initialised (model=%s, max_loops=%d).",
            model, max_loops,
        )

    # ══════════════════════════════════════════════════════════════
    # Public gateway
    # ══════════════════════════════════════════════════════════════

    @traceable(
        name="ResearchAgent.run", run_type="chain", metadata=_TRACE_IDENTITY,
        process_inputs=_trace_inputs("research_run"),
        process_outputs=_trace_outputs("research_run"),
    )
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
       
        task_query = shared_state.get("task_query", "").strip()
        if not task_query:
            raise ValueError(
                "SharedManagerState.task_query must be a non-empty string."
            )

        directives = dict(shared_state.get("manager_directives", {}))
        context, run_id = require_execution_identity(
            shared_state.get("forecast_context"),
            directives.get("run_id") or shared_state.get("run_id"),
        )
        directive_ticker = str(directives.get("ticker") or context.ticker).upper()
        if directive_ticker != context.ticker:
            raise ValueError("manager ticker must match ForecastContext.ticker")
        context = _research_window_context(context, directives)
        directives.update(
            ticker=context.ticker,
            forecast_context=context.to_dict(),
            run_id=run_id,
        )

        run_started_at = time.perf_counter()
        logger.info("RESEARCH_RUN_STARTED ticker=%s run_id=%s", context.ticker, run_id)

        internal_state: ResearchAgentState = {
            "messages":           [],
            "context_chunks":     [],
            "called_tool_signatures": [],
            "loop_counter":       0,
            "validation_feedback": "",
            "is_complete":        False,
            "task_query":         shared_state["task_query"],
            "manager_directives": directives,
        }

       
        try:
            final_state: ResearchAgentState = await self._graph.ainvoke(
                internal_state,
                config={"recursion_limit": (self._max_loops + 1) * 3},
            )
        except Exception:
            logger.error("RESEARCH_AGENT_FAIL_CLOSED")
            raise RuntimeError("RESEARCH_AGENT_FAIL_CLOSED") from None

        
        evidence, final_rejected = eligible_evidence(
            [c for c in final_state["context_chunks"] if isinstance(c, dict)],
            context=context, run_id=run_id, source="research_agent_final",
            historical=False,
        )
        synthesis = await synthesize_research_context(
            chunks=evidence,
            task_query=task_query,
            llm_client=self._llm,
            model=self._model,
            forecast_context=context,
            run_id=run_id,
        )

        features = build_research_features(evidence, context=context, run_id=run_id)

        existing = shared_state.get("aggregated_research_context") or []
        safe_existing = [
            {
                key: item[key] for key in (
                    "evidence_id", "citation_reference_id", "source", "provider",
                    "timestamp", "dataset_version", "run_id",
                ) if key in item
            }
            for item in existing if isinstance(item, dict) and item.get("evidence_id")
        ]
        references = [
            {
                "evidence_id": item["evidence_id"],
                "citation_reference_id": citation_reference_id(item.get("citation")),
                "source": item.get("source"), "provider": item.get("provider"),
                "timestamp": item.get("timestamp"),
                "dataset_version": item.get("dataset_version"), "run_id": run_id,
            }
            for item in evidence
        ]
        summary_chunk = None
        if isinstance(synthesis, dict) and synthesis.get('summary_generated'):
            summary = synthesis.get('summary')
            synthesis_ids = synthesis.get('evidence_ids')
            accepted_ids = {item['evidence_id'] for item in evidence}
            if (
                isinstance(summary, str)
                and summary.startswith('[SYNTHESIZED RESEARCH SUMMARY]')
                and isinstance(synthesis_ids, list)
                and set(synthesis_ids).issubset(accepted_ids)
            ):
                summary_chunk = summary[:6_000]
        rejected_counts = rejection_counts(final_rejected)
        final_errors = set(rejected_counts)
        if not evidence:
            final_errors.add("RESEARCH_EVIDENCE_UNAVAILABLE")
        lifecycle = compact_lifecycle(
            context=context, run_id=run_id, operation="pre_state_leakage_gate",
            latency_ms=elapsed_latency_ms(run_started_at),
            payload={"evidence": evidence, "quarantined": final_rejected,
                     "degraded": bool(final_rejected) or not evidence},
            input_count=len(final_state["context_chunks"]),
        )
        updated_shared: SharedManagerState = {
            **shared_state,
            "aggregated_research_context": (
                safe_existing + references + ([summary_chunk] if summary_chunk else [])
            ),
            "research_features": {
                **features.to_state_slot(),
                "values": {
                    **features.to_state_slot()["values"],
                    "rejection_counts": rejected_counts,
                    "lifecycle": lifecycle,
                },
                "errors": sorted(final_errors),
            },
        }

        logger.info(
            "RESEARCH_RUN_COMPLETED evidence_count=%d loop_count=%d degraded=%s",
            len(evidence), final_state["loop_counter"], features.degraded,
        )
        return updated_shared

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Brain (Planner)
    # ══════════════════════════════════════════════════════════════

    @traceable(
        name="research.brain", run_type="llm", metadata=_TRACE_IDENTITY,
        process_inputs=_trace_inputs("research_brain"),
        process_outputs=_trace_outputs("research_brain"),
    )
    async def _brain_node(
        self, state: ResearchAgentState
    ) -> dict[str, Any]:
        
        task_query   = state["task_query"]
        directives   = state["manager_directives"]
        feedback     = state["validation_feedback"]
        loop_num     = state["loop_counter"]

        logger.info("RESEARCH_BRAIN_STARTED loop=%d has_feedback=%s", loop_num, bool(feedback))

        
        directive_str = json.dumps(directives, indent=2) if directives else "None"
        feedback_block = (
            f"\n\nVALIDATOR FEEDBACK (address this in your plan):\n{feedback}"
            if feedback else ""
        )

        user_content = (
            f"RESEARCH TASK:\n{task_query}\n\n"
            f"MANAGER DIRECTIVES:\n{directive_str}"
            f"{feedback_block}\n\n"
            f"ITERATION: {loop_num + 1} of {self._max_loops}\n\n"
            "Produce the JSON action plan now."
        )

       
        history = state["messages"]
        messages_for_api = history + [{"role": "user", "content": user_content}]

        deterministic_fallback = False
        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=1024,
                temperature=0,
                system=BRAIN_SYSTEM_PROMPT,
                messages=messages_for_api,
            )
            plan_text = response.content[0].text.strip()
        except Exception:
            # Planning is advisory; the execution identity and evidence gates
            # remain deterministic. If the planner provider is unavailable,
            # run a bounded diversified plan instead of deleting the entire
            # Research feature family.
            deterministic_fallback = True
            ticker = str(directives.get("ticker") or "").strip().upper()
            keyword_query = f"{ticker} earnings guidance".strip()
            plan_text = json.dumps({
                "reasoning": "Deterministic diversified plan after planner provider failure.",
                "actions": [
                    {
                        "tool": "rag_hybrid_query",
                        "arguments": {
                            "query": task_query,
                            "entity": ticker,
                            "top_k": 10,
                        },
                    },
                    {
                        "tool": "tavily_search",
                        "arguments": {
                            "query": keyword_query,
                            "max_results": 5,
                            "search_depth": "advanced",
                            "topic": "news",
                        },
                    },
                    {
                        "tool": "news_search",
                        "arguments": {"query": keyword_query},
                    },
                    {
                        "tool": "sec_edgar_filing",
                        "arguments": {
                            "ticker": ticker,
                            "sections": ["mda", "risk_factors"],
                        },
                    },
                ],
            })
            logger.warning(
                "RESEARCH_BRAIN_PROVIDER_UNAVAILABLE deterministic_plan=true loop=%d",
                loop_num,
            )

        _publish_progress(
            session_from_directives(directives), "agent_brain", agent="research",
            message=f"Research Agent: planning research (iteration {loop_num + 1})",
            detail={
                **research_trace_metadata(
                    directives["forecast_context"], directives["run_id"]
                ),
                "loop": loop_num + 1,
                "deterministic_fallback": deterministic_fallback,
            },
        )

        return {
            "messages": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": plan_text},
            ]
        }

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Executor (Tool Caller)
    # ══════════════════════════════════════════════════════════════

    @traceable(
        name="research.executor", run_type="tool", metadata=_TRACE_IDENTITY,
        process_inputs=_trace_inputs("research_executor"),
        process_outputs=_trace_outputs("research_executor"),
    )
    async def _executor_node(
        self, state: ResearchAgentState
    ) -> dict[str, Any]:
       
        new_counter = state["loop_counter"] + 1
        logger.info("[Executor] loop_counter → %d", new_counter)
        session_id = session_from_directives(state["manager_directives"])

       
        plan_text = state["messages"][-1]["content"]
        actions   = self._parse_plan(plan_text)

        if not actions:
            logger.warning("RESEARCH_PLAN_INVALID")
            return {
                "loop_counter":  new_counter,
                "context_chunks": ["[EXECUTOR WARNING] Brain produced no valid action plan."],
                "called_tool_signatures": [],
            }

        
        new_chunks: list[Any] = []
        new_signatures: list[str] = []
        seen_signatures: set[str] = set(state.get("called_tool_signatures", []))
        sec_filing_coverage = build_sec_filing_coverage(seen_signatures)

        try:
            from contextlib import AsyncExitStack
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(
                    stdio_client(self._mcp_params)
                )
                session: ClientSession = await stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()

                for action in actions:
                    tool_name = action.get("tool", "")
                    arguments = action.get("arguments", {})

                    if not tool_name:
                        logger.warning("[Executor] Skipping action with no tool name.")
                        continue
                    if tool_name not in RESEARCH_TOOL_NAMES:
                        logger.warning("RESEARCH_TOOL_NOT_ALLOWLISTED")
                        new_chunks.append({
                            "source": "research_mcp",
                            "error_code": "RESEARCH_TOOL_NOT_ALLOWLISTED",
                        })
                        continue

                    arguments = ensure_ticker_argument(
                        tool_name, arguments,
                        state["manager_directives"],
                    )
                    context, run_id = require_execution_identity(
                        state["manager_directives"].get("forecast_context"),
                        state["manager_directives"].get("run_id"),
                    )
                    arguments = inject_execution_identity(
                        tool_name, arguments,
                        forecast_context=context, run_id=run_id,
                    )

                    query_hint = arguments.get("query") or arguments.get("entity") or arguments.get("ticker") or "N/A"
                    signature = tool_call_signature(tool_name, arguments)

                    if signature in seen_signatures:
                        logger.info("RESEARCH_TOOL_SKIPPED_DUPLICATE tool=%s", tool_name)
                        new_chunks.append(format_skip_notice(
                            tool_name, query_hint, "duplicate",
                            "This exact call (same tool, same arguments) was already made earlier "
                            "in this session -- its result is already present in the context above. "
                            "No new data was fetched. If you need more, change the arguments "
                            "(different sections/mode/query), don't repeat the same call."
                        ))
                        continue

                    doc_key = sec_filing_doc_key(arguments) if tool_name == "sec_edgar_filing" else None
                    requested_sections = sec_filing_sections(arguments) if doc_key else set()
                    already_covered = sec_filing_coverage.get(doc_key, set()) if doc_key else set()
                    if doc_key and requested_sections and requested_sections <= already_covered:
                        logger.info("RESEARCH_TOOL_SKIPPED_REDUNDANT tool=sec_edgar_filing")
                        new_chunks.append(format_skip_notice(
                            "sec_edgar_filing", query_hint, "redundant subset",
                            f"Section(s) {sorted(requested_sections)} for this filing "
                            f"(ticker={arguments.get('ticker')}, form_type={arguments.get('form_type', '10-K')}, "
                            f"mode={arguments.get('mode', 'preview')}) were already fetched in an earlier "
                            f"call -- see that sec_edgar_filing result above. No new data was fetched. "
                            f"Request a DIFFERENT section, or use mode='full'/query=... on this section "
                            f"for more depth instead of re-requesting it."
                        ))
                        continue

                    logger.info("RESEARCH_TOOL_STARTED tool=%s", tool_name)
                    _publish_progress(
                        session_id, "agent_tool_call", agent="research",
                        message=f"Research Agent: calling tool '{tool_name}'...",
                        detail={
                            "tool": tool_name, "provider": tool_name,
                            **research_trace_metadata(context, run_id),
                            "ticker": context.ticker, "run_id": run_id,
                            "as_of": context.as_of.isoformat(),
                            "dataset_version": context.dataset_version,
                            "analysis_mode": context.analysis_mode.value,
                            "outcome": "STARTED",
                        },
                    )

                    try:
                        tool_started = time.perf_counter()
                        if sentry_enabled():
                            import sentry_sdk
                            sentry_sdk.add_breadcrumb(
                                category="mcp.research",
                                message=f"Calling tool: {tool_name}",
                                data={
                                    **research_trace_metadata(context, run_id),
                                    "tool_name": tool_name,
                                    "ticker": context.ticker,
                                    "run_id": run_id,
                                    "argument_keys": sorted(arguments),
                                },
                                level="info",
                            )
                        result = await session.call_tool(tool_name, arguments)
                        raw_text = (
                            result.content[0].text
                            if result.content and result.content[0].type == "text"
                            else ""
                        )
                        extracted = self._extract_evidence(tool_name, raw_text, context, run_id)
                        new_chunks.extend(extracted)
                        safe_tool_telemetry = self._safe_tool_telemetry(raw_text)
                        new_signatures.append(signature)
                        seen_signatures.add(signature)
                        if doc_key:
                            sec_filing_coverage.setdefault(doc_key, set()).update(requested_sections)
                        logger.info("RESEARCH_TOOL_COMPLETED tool=%s evidence_count=%d", tool_name, len(extracted))
                        _publish_progress(
                            session_id, "agent_tool_result", agent="research",
                            message=f"Research Agent: tool '{tool_name}' succeeded",
                            detail={
                                "tool": tool_name,
                                **research_trace_metadata(context, run_id),
                                "provider": safe_tool_telemetry.get("provider", tool_name),
                                "outcome": safe_tool_telemetry.get("outcome", "SUCCESS"),
                                "latency_ms": round((time.perf_counter() - tool_started) * 1000, 3),
                                "ticker": context.ticker, "run_id": run_id,
                                "dataset_version": context.dataset_version,
                                "as_of": context.as_of.isoformat(),
                                "analysis_mode": context.analysis_mode.value,
                                "evidence_count": len(extracted),
                                "citation_count": safe_tool_telemetry.get("citation_count", 0),
                                "quarantine_count": safe_tool_telemetry.get("quarantine_count", 0),
                                "rejection_counts": safe_tool_telemetry.get("rejection_counts", {}),
                                "degraded": bool(safe_tool_telemetry.get("degraded", False)),
                            },
                        )

                    except Exception:
                        error_code = "RESEARCH_TOOL_FAIL_CLOSED"
                        logger.warning("%s tool=%s", error_code, tool_name)
                        if sentry_enabled():
                            import sentry_sdk
                            sentry_sdk.capture_message(error_code, level="error")
                        new_chunks.append({"source": tool_name, "error_code": error_code})
                        _publish_progress(
                            session_id, "agent_tool_result", agent="research",
                            message=f"Research Agent: tool '{tool_name}' failed",
                            detail={
                                "tool": tool_name, "outcome": "FAIL_CLOSED",
                                **research_trace_metadata(context, run_id),
                                "error_code": error_code,
                                "ticker": context.ticker, "run_id": run_id,
                                "provider": tool_name,
                                "as_of": context.as_of.isoformat(),
                                "dataset_version": context.dataset_version,
                                "analysis_mode": context.analysis_mode.value,
                                "latency_ms": round((time.perf_counter() - tool_started) * 1000, 3),
                                "evidence_count": 0, "citation_count": 0,
                                "quarantine_count": 0, "degraded": True,
                            },
                        )

        except Exception:
            error_code = "RESEARCH_MCP_UNAVAILABLE"
            logger.error(error_code)
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.capture_message(error_code, level="error")
            new_chunks.append({"source": "research_mcp", "error_code": error_code})

        return {
            "loop_counter":   new_counter,
            "context_chunks": new_chunks,
            "called_tool_signatures": new_signatures,
        }

    # ══════════════════════════════════════════════════════════════
    # LangGraph node: Checker (Validator / Critic)
    # ══════════════════════════════════════════════════════════════

    @traceable(
        name="research.checker", run_type="llm", metadata=_TRACE_IDENTITY,
        process_inputs=_trace_inputs("research_checker"),
        process_outputs=_trace_outputs("research_checker"),
    )
    async def _checker_node(
        self, state: ResearchAgentState
    ) -> dict[str, Any]:
        
        task_query    = state["task_query"]
        all_chunks    = state["context_chunks"]
        loop_counter  = state["loop_counter"]

        logger.info(
            "[Checker] auditing %d chunks (loop=%d)", len(all_chunks), loop_counter
        )

        if not all_chunks:
            logger.warning("[Checker] No context chunks to audit.")
            return {
                "is_complete":        False,
                "validation_feedback": "No data was retrieved. Retry with different tools or broader queries.",
            }

        
        per_chunk_budget = max(800, 12_000 // max(len(all_chunks), 1))

        def _budget_chunk(c: Any) -> str:
            if not isinstance(c, str):
                c = json.dumps(c, sort_keys=True, default=str)
            if len(c) <= per_chunk_budget:
                return c
            
            return c[:per_chunk_budget] + "\n[truncated for audit]"

        combined = "\n\n---\n\n".join(_budget_chunk(c) for c in all_chunks)

        user_content = (
            f"ORIGINAL RESEARCH QUERY:\n{task_query}\n\n"
            f"GATHERED CONTEXT ({len(all_chunks)} chunks, loop {loop_counter}):\n\n"
            f"{combined}\n\n"
            "Audit the context and return your JSON verdict."
        )

        deterministic_fallback = False
        try:
            response = await asyncio.to_thread(
                self._llm.messages.create,
                model=self._model,
                max_tokens=512,
                temperature=0,
                system=CHECKER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            verdict_text = response.content[0].text.strip()
        except Exception:
            # A checker-provider outage must not discard point-in-time evidence
            # already validated by the deterministic evidence contract. This is
            # still fail-closed: sparse or single-source evidence remains
            # incomplete, and downstream specialist/quality gates still run.
            deterministic_fallback = True
            context, run_id = require_execution_identity(
                state["manager_directives"].get("forecast_context"),
                state["manager_directives"].get("run_id"),
            )
            eligible, _ = eligible_evidence(
                [item for item in all_chunks if isinstance(item, dict)],
                context=context, run_id=run_id,
                source="research_checker_fallback", historical=False,
            )
            providers = {
                str(item.get("provider") or item.get("source") or "").strip()
                for item in eligible
                if str(item.get("provider") or item.get("source") or "").strip()
            }
            is_complete = len(eligible) >= 3 and len(providers) >= 2
            feedback = (
                "" if is_complete else
                "Checker provider unavailable and deterministic evidence coverage is insufficient."
            )
            logger.warning(
                "RESEARCH_CHECKER_PROVIDER_UNAVAILABLE deterministic_complete=%s "
                "evidence_count=%d provider_count=%d",
                is_complete, len(eligible), len(providers),
            )
        else:
            try:
                # Accept the first complete JSON object even when the model wraps it
                # in a fence or appends a prose rationale.
                object_start = verdict_text.index("{")
                verdict, _ = json.JSONDecoder().raw_decode(verdict_text[object_start:])
            except (json.JSONDecodeError, ValueError):
                logger.warning("RESEARCH_CHECKER_OUTPUT_INVALID")
                verdict = {"is_complete": False, "feedback": "Checker JSON parse error; retry."}

            is_complete = bool(verdict.get("is_complete", False))
            feedback = str(verdict.get("feedback", ""))

        logger.info("RESEARCH_CHECKER_COMPLETED is_complete=%s", is_complete)
        _publish_progress(
            session_from_directives(state["manager_directives"]), "agent_checker", agent="research",
            message=(
                f"Research Agent: checking data quality — "
                f"{'sufficient ✓' if is_complete else 'needs more data'}"
            ),
            detail={
                **research_trace_metadata(
                    state["manager_directives"]["forecast_context"],
                    state["manager_directives"]["run_id"],
                ),
                "is_complete": is_complete,
                "deterministic_fallback": deterministic_fallback,
            },
        )

        return {
            "is_complete":        is_complete,
            "validation_feedback": "" if is_complete else feedback,
        }

    # ══════════════════════════════════════════════════════════════
    # Conditional routing edge
    # ══════════════════════════════════════════════════════════════

    def _should_continue(
        self, state: ResearchAgentState
    ) -> Literal["brain", "__end__"]:
        
        loop_counter = state["loop_counter"]
        is_complete  = state["is_complete"]
        max_loops    = state["manager_directives"].get("max_loops", self._max_loops)

        # Guardrail: hard cap on iterations
        if loop_counter >= max_loops:
            logger.warning(
                "[Router] loop_counter=%d >= max_loops=%d → forcing END.",
                loop_counter, max_loops,
            )
            return "__end__"

        
        if is_complete:
            logger.info("[Router] is_complete=True → routing to END.")
            return "__end__"

        
        logger.info(
            "[Router] is_complete=False, loop=%d/%d → routing to Brain.",
            loop_counter, max_loops,
        )
        return "brain"

    # ══════════════════════════════════════════════════════════════
    # Graph builder
    # ══════════════════════════════════════════════════════════════

    def _build_graph(self) -> StateGraph:
        
        builder = StateGraph(ResearchAgentState)

        # ── Register nodes ───────────────────────────────────────
        builder.add_node("brain",    self._brain_node)
        builder.add_node("executor", self._executor_node)
        builder.add_node("checker",  self._checker_node)

        # ── Wire edges ───────────────────────────────────────────
        builder.set_entry_point("brain")
        builder.add_edge("brain",    "executor")
        builder.add_edge("executor", "checker")

       
        builder.add_conditional_edges(
            "checker",
            self._should_continue,
            {
                "brain":    "brain",
                "__end__":  END,
            },
        )

        return builder.compile()

    # ══════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_tool_telemetry(raw_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return {}
        telemetry = payload.get("telemetry") if isinstance(payload, dict) else None
        if not isinstance(telemetry, dict):
            return {}
        safe: dict[str, Any] = {}
        if telemetry.get("provider") in {
            "tavily_search", "news_search", "sec_edgar_search", "sec_edgar_filing",
            "rag_vector_search", "rag_graph_traverse", "rag_hybrid_query",
            "comprehensive_analysis", "research_synthesis",
        }:
            safe["provider"] = telemetry["provider"]
        if telemetry.get("outcome") in {"SUCCESS", "DEGRADED", "DATA_UNAVAILABLE", "FAIL_CLOSED"}:
            safe["outcome"] = telemetry["outcome"]
        for key in ("citation_count", "quarantine_count"):
            value = telemetry.get(key)
            safe[key] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
        counts = telemetry.get("rejection_counts")
        if isinstance(counts, dict):
            safe["rejection_counts"] = {
                key: value for key, value in counts.items()
                if key in {
                    "FUTURE_EVIDENCE", "EVIDENCE_DATA_UNAVAILABLE",
                    "EVIDENCE_TYPE_INVALID", "EVIDENCE_VALUE_INVALID",
                    "HISTORICAL_PROVIDER_REJECTED",
                } and isinstance(value, int) and value >= 0
            }
        safe["degraded"] = bool(telemetry.get("degraded", False))
        return safe

    @staticmethod
    def _extract_evidence(
        tool_name: str,
        raw_text: str,
        context: ForecastContext,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Extract only compact point-in-time references from a tool payload."""
        try:
            payload = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(payload, dict) or payload.get("error"):
            return []

        candidates: list[dict[str, Any]] = []
        nested = [payload]
        if tool_name == "comprehensive_analysis":
            nested.extend(
                value for key in ("news", "filing")
                if isinstance((value := payload.get(key)), dict)
            )
        for item_payload in nested:
            evidence = item_payload.get("evidence")
            if isinstance(evidence, list):
                candidates.extend(dict(item) for item in evidence if isinstance(item, dict))
                continue
            branch_items = (
                item_payload.get("results")
                or item_payload.get("nodes")
                or item_payload.get("citations")
                or []
            )
            for item in branch_items:
                if not isinstance(item, dict):
                    continue
                citation = item.get("citation") if isinstance(item.get("citation"), dict) else {}
                candidate = {**item, **citation}
                candidate["citation"] = citation or {
                    "source": tool_name,
                    "url": candidate.get("url") or candidate.get("source_url"),
                    "document_id": candidate.get("document_id"),
                    "chunk_id": candidate.get("chunk_id"),
                }
                candidate.setdefault("ticker", context.ticker)
                candidate.setdefault("dataset_version", context.dataset_version)
                candidate["provider"] = tool_name
                candidates.append(candidate)

        eligible, _ = eligible_evidence(
            candidates, context=context, run_id=run_id, source=tool_name,
            historical=context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST,
        )
        return eligible

    def _parse_plan(self, plan_text: str) -> list[dict[str, Any]]:
       
        try:
            clean = plan_text.replace("```json", "").replace("```", "").strip()
            plan  = json.loads(clean)
            actions = plan.get("actions", [])
            if not isinstance(actions, list):
                logger.warning("RESEARCH_PLAN_ACTIONS_INVALID")
                return []
            return actions
        except (json.JSONDecodeError, AttributeError):
            logger.warning("RESEARCH_PLAN_INVALID")
            return []

    @staticmethod
    def _format_tool_result(
        tool_name: str,
        arguments: dict[str, Any],
        raw_text: str,
    ) -> str:
        
        query_hint = arguments.get("query") or arguments.get("entity") or arguments.get("ticker") or "N/A"
        separator  = "─" * 60
        return (
            f"{separator}\n"
            f"[TOOL: {tool_name}] | [QUERY: {query_hint}]\n"
            f"{separator}\n"
            f"{raw_text}\n"
        )

    _TOOL_TAG_RE = re.compile(r"\[TOOL:\s*([^\]]+)\]")

    @classmethod
    def _tag_chunk_with_source(cls, chunk: str) -> dict:
        
        m = cls._TOOL_TAG_RE.search(chunk)
        return {"source": m.group(1).strip() if m else None, "text": chunk}
