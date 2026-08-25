from __future__ import annotations

import json
import logging
import asyncio
from urllib.parse import urlparse
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.prompts.promt_sentinemt import (BRAIN_PLAN_SYSTEM_PROMPT, BRAIN_ANALYZE_SYSTEM_PROMPT)
from evaluation.aspect_sentiment import extract_aspect_observations, supported_causal_aspects
from agents.state import SentimentAgentState, SharedManagerState
from agents.executor_helpers.finance_executor_helper import extract_ticker
from langsmith import traceable
from core.observability import traced_call_tool
from core.progress_bus import publish as _publish_progress, session_from_directives
from core.error_handler import capture_exception, add_breadcrumb
from agents.executor_helpers.sentiment_helper import (
    valid_composite_score,
    causal_audit,
    attempt_rank,
)

log = logging.getLogger("sentiment-agent")


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

    @traceable(name="sentiment.brain_plan", run_type="llm")
    async def _brain_plan(self, state: SentimentAgentState) -> dict[str, Any]:

        task_query = state.get("task_query", "")
        directives = state.get("manager_directives",  {})
        fin_summary = state.get("financial_metrics_summary",{})
        research_ctx = state.get("aggregated_research_context", [])
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
            log.info("[Brain-Plan] Plan: query='%s' ticker=%s days_back=%s",
                     plan.get("retrieval_query"), plan.get("ticker"), plan.get("days_back"))
            _publish_progress(
                session_from_directives(directives), "agent_brain", agent="sentiment",
                message="Sentiment Agent: planning market sentiment data collection",
                detail={"query": plan.get("retrieval_query"), "ticker": plan.get("ticker")},
            )
            return plan

        except (json.JSONDecodeError, Exception) as exc:
            log.warning("[Brain-Plan] Claude returned invalid plan (%s) — using defaults.", exc)
            capture_exception(
                "sentiment.brain_plan", exc,
                tags={"iteration": str(state["loop_counter"])},
            )
            default_plan = {
                "retrieval_query": f"{ticker or 'market'} earnings sentiment investor reaction",
                "ticker":          ticker,
                "days_back":       int(directives.get("days_back", 30)),
                "reasoning":       "Default plan: Brain planning failed; using heuristic query.",
            }
            state["messages"].append({"role": "assistant","content": json.dumps(default_plan)})
            return default_plan

    @traceable(name="sentiment.brain_analyze", run_type="llm")
    async def _brain_analyze(self, state: SentimentAgentState) -> str:

        task_query  = state.get("task_query","")
        fin_summary = state.get("financial_metrics_summary", {})
        fg          = state["fear_greed_result"]
        fb          = state["finbert_result"]
        vd          = state["vader_result"]
        status      = state.get("sentiment_status", "NO_DATA")

        if status in {"NO_DATA", "ERROR"}:
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
            _publish_progress(
                session_from_directives(state["manager_directives"]), "agent_checker", agent="sentiment",
                message="Sentiment Agent: final sentiment interpretation ready",
            )
            return raw

        except Exception as exc:
            log.exception("[Brain-Analyze] Claude API call failed: %s", exc)
            capture_exception("sentiment.brain_analyze", exc)
            fallback = json.dumps({
                "overall_sentiment":  "Neutral",
                "conviction_level":   "Low",
                "key_signals":        [],
                "model_agreement":    "Weak",
                "narrative":          f"Brain analysis failed: {exc}. Raw signals stored.",
                "risk_flags":         ["Brain API call failed"],
                "data_quality_note":  "Interpretation unavailable due to API error.",
            })
            state["messages"].append({"role": "assistant", "content": fallback})
            return fallback

    # =========================================================================
    # LAYER 1 — EXECUTOR (MCP Protocol Interface)
    # =========================================================================

    @traceable(name="sentiment.executor", run_type="tool")
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

        directives_for_attempt = state.get(
            "manager_directives", {})

        session_id = session_from_directives(directives_for_attempt)

        async def _call(tool: str, args: dict) -> dict:
            _publish_progress(
                session_id, "agent_tool_call", agent="sentiment",
                message=f"Sentiment Agent: running '{tool}'...",
                detail={"tool": tool},
            )
            try:
                add_breadcrumb("mcp.sentiment", f"Calling {tool}", {"tool": tool})
                result = await traced_call_tool(session, tool, args)
                payload = json.loads(result.content[0].text)
                if payload.get("error"):
                    state["extraction_errors"].append(
                        f"{tool} error: {payload['error']}"
                    )
                    log.warning("[Executor] %s returned error: %s", tool, payload["error"])
                    _publish_progress(
                        session_id, "agent_tool_result", agent="sentiment",
                        message=f"Sentiment Agent: '{tool}' returned an error",
                        detail={"tool": tool, "outcome": "error", "error": payload["error"]},
                    )
                else:
                    _publish_progress(
                        session_id, "agent_tool_result", agent="sentiment",
                        message=f"Sentiment Agent: '{tool}' completed",
                        detail={"tool": tool, "outcome": "success"},
                    )
                return payload
            except Exception as exc:
                state["extraction_errors"].append(f"{tool} exception: {exc}")
                log.exception("[Executor] %s call failed: %s", tool, exc)
                _publish_progress(
                    session_id, "agent_tool_result", agent="sentiment",
                    message=f"Sentiment Agent: '{tool}' failed",
                    detail={"tool": tool, "outcome": "error", "error": str(exc)},
                )
                capture_exception("mcp.sentiment", exc, tags={"tool": tool})
                return {}

        # -- Step 1: retrieve_social_data ------------------------------------
        retrieval_query = plan.get("retrieval_query", "market sentiment")
        ticker          = plan.get("ticker")
        days_back       = int(plan.get("days_back", 30))

        log.info("[Executor] Step 1 — retrieve_social_data: query='%s' ticker=%s days=%d",
                 retrieval_query, ticker, days_back)

        retrieval_args: dict[str, Any] = {"query": retrieval_query, "days_back": days_back}
        if ticker:
            retrieval_args["ticker"] = ticker

        social_data = await _call("retrieve_social_data", retrieval_args)

        log.info("[DEBUG] social_data keys=%s chunks=%d",
                 list(social_data.keys()), len(social_data.get("chunks", [])))

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
            }

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
            }

        fb_has_data = state["finbert_result"].get("total_chunks", 0) > 0
        vd_has_data = state["vader_result"].get("total_chunks", 0) > 0

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
        else:
            log.warning("[Executor] Step 4 — skipping calculate_fear_greed (no model data).")
            state["fear_greed_result"] = {
                "score": None, "label": None, "polarity": None, "status": "NO_DATA",
                "finbert_score": None, "vader_score": None,
                "weights": {}, "confidence": 0.0, "diagnostics": {},
            }

        model_has_data = fb_has_data or vd_has_data
        score = state["fear_greed_result"].get("score")
        is_valid_composite_score = valid_composite_score(score)
        for output_name, output in (
            ("finbert", state["finbert_result"]),
            ("vader", state["vader_result"]),
            ("fear_greed", state["fear_greed_result"]),
        ):
            if str(output.get("status") or "").upper() == "ERROR":
                state["extraction_errors"].append(
                    f"{output_name} returned critical status ERROR"
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
        polarity = state["fear_greed_result"].get("polarity")
        if polarity is None and is_valid_composite_score:
            score = float(score)
            polarity = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
        state["fear_greed_result"].update({
            "status": status, "polarity": polarity,
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

    @traceable(name="SentimentAgent.run", run_type="chain")
    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        directives = shared_state.get("manager_directives", {})
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        log.info(
            "SentimentAgent.run() started — task_query='%s' max_loops=%d",
            shared_state.get("task_query", ""),
            max_loops,
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
            "manager_directives":          dict(shared_state.get("manager_directives", {})),
            "financial_metrics_summary":   shared_state.get("financial_metrics_summary", {}),
            "aggregated_research_context": shared_state.get("aggregated_research_context", []),
            "current_plan":       {},
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
            log.warning("[Run] Brain analysis JSON malformed — using raw text.")
            analysis = {
                "overall_sentiment": "Neutral",
                "conviction_level":  "Low",
                "key_signals":       [],
                "model_agreement":   "Weak",
                "narrative":         raw_analysis,
                "risk_flags":        ["Brain analysis JSON parse failed"],
                "data_quality_note": "",
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

        # Commit to the shared state contract
        shared_state["sentiment_analysis_summary"] = sentiment_analysis_summary  # type: ignore[literal-required]

        log.info(
            "SentimentAgent.run() complete — ticker=%s fg_score=%s fg_label=%s "
            "sentiment=%s conviction=%s chunks=%d",
            sentiment_analysis_summary["ticker"],
            sentiment_analysis_summary["fear_greed_score"],
            sentiment_analysis_summary["fear_greed_label"],
            sentiment_analysis_summary["overall_sentiment"],
            sentiment_analysis_summary["conviction_level"],
            sentiment_analysis_summary["total_chunks_analyzed"],
        )

        return shared_state
