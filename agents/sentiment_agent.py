"""
agents/sentiment_agent.py
--------------------------
Production-grade SentimentAgent for the Alpha-Agent Node platform.

Architecture
------------
The agent follows a two-tiered internal lifecycle:

    ┌──────────────────────────────────────────────────────────────┐
    │                       SentimentAgent                        │
    │                                                              │
    │  LAYER 1 — EXECUTOR (MCP Protocol Interface)                 │
    │    _execute_sentiment_pipeline()                             │
    │        ├─ retrieve_social_data   → retrieved_chunks          │
    │        ├─ analyze_finbert        → finbert_result            │
    │        ├─ score_vader            → vader_result              │
    │        └─ calculate_fear_greed   → fear_greed_result         │
    │                                                              │
    │  LAYER 2 — BRAIN (Cognitive / LLM Interpretation Layer)      │
    │    _brain_plan()    → structured retrieval plan              │
    │    _brain_analyze() → narrative interpretation + verdict     │
    │    run()            → entry point; drives the full loop      │
    └──────────────────────────────────────────────────────────────┘

Design notes
------------
  - No Checker layer: the Brain performs both pre-execution planning
    and post-execution semantic interpretation of sentiment signals.
  - The Brain calls Claude twice per loop iteration:
      Pass 1 (_brain_plan)    — before the Executor runs, to decide
                                 what query/ticker/days_back to use.
      Pass 2 (_brain_analyze) — after the Executor runs, to interpret
                                 FinBERT + VADER + Fear/Greed signals
                                 in the context of the research task.
  - MCP tools run sequentially: retrieve → finbert → vader → fear_greed.
  - The loop runs at most ``max_loops`` times (default: 2). A second
    pass only occurs if the Executor produced zero chunks on the first.

MCP Tool Servers consumed
-------------------------
    SentimentServer: retrieve_social_data, analyze_finbert,
                     score_vader, calculate_fear_greed

State contract
--------------
    Input  : SharedManagerState  (task_query, manager_directives,
                                  aggregated_research_context,
                                  financial_metrics_summary)
    Output : SharedManagerState  (+ sentiment_analysis_summary populated)

Dependencies
------------
    pip install anthropic "mcp[cli]"
"""

from __future__ import annotations

import json
import logging
import asyncio
import re
from urllib.parse import urlparse
from typing import Any

import anthropic
from langgraph.graph import END, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.prompts.promt_sentinemt import (BRAIN_PLAN_SYSTEM_PROMPT,BRAIN_ANALYZE_SYSTEM_PROMPT)
from evaluation.aspect_sentiment import extract_aspect_observations, supported_causal_aspects
# ---------------------------------------------------------------------------
# State contract imports
# ---------------------------------------------------------------------------
# SharedManagerState and SentimentAgentState are declared in agents/state.py
# — the single source of truth for all state TypedDicts across the platform.
from agents.state import SentimentAgentState, SharedManagerState
from agents.executor_helpers.finance_executor_helper import extract_ticker
from langsmith import traceable
from core.observability import sentry_enabled
from core.progress_bus import publish as _publish_progress, session_from_directives

# ---------------------------------------------------------------------------
# Logging — stderr only; stdout is reserved for MCP JSON-RPC
# NOTE: logging.basicConfig() must NOT be called in library/agent code.
# Configuration belongs exclusively in entry points (api/main.py or __main__).
# ---------------------------------------------------------------------------
log = logging.getLogger("sentiment-agent")





# ══════════════════════════════════════════════════════════════════════════════
# SentimentAgent
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAgent:
    """
    Specialist Sentiment Analysis Agent for the Alpha-Agent Node platform.

    Responsibilities
    ----------------
    1. Receive a research task from the Manager Agent via ``SharedManagerState``.
    2. Brain Pass 1 — plan an optimal RAG retrieval query for social/news data.
    3. Executor — run the four-step MCP sentiment pipeline:
           retrieve_social_data → analyze_finbert → score_vader
           → calculate_fear_greed
    4. Brain Pass 2 — interpret and synthesise all sentiment signals into
       an actionable narrative verdict.
    5. Commit a structured ``sentiment_analysis_summary`` into ``SharedManagerState``
       for consumption by the Report Writer Agent and Manager Agent.

    Two-Tier Architecture
    ---------------------
    Unlike the FinancialAnalystAgent (Brain + Executor + Checker) and the
    ResearchAgent (LangGraph Brain + Executor + Checker), the SentimentAgent
    uses a lighter Brain → Executor two-tier design. This is appropriate
    because:
      - The sentiment pipeline has a fixed, deterministic tool sequence.
      - The Brain's second pass (post-analysis) replaces the Checker by
        semantically evaluating data quality and flagging risks inline.
      - A second loop iteration only triggers if zero chunks were retrieved.

    Parameters
    ----------
    server_script_path : str
        Absolute path to ``sentiment_server.py`` — the FastMCP stdio server
        that exposes the four sentiment tools.
    model : str
        Anthropic model identifier. Defaults to ``"claude-haiku-4-5"``.
    max_loops : int
        Safety guardrail for the internal execution loop. Default: 2.
        Iteration 2 only runs if the Executor retrieved zero chunks on
        iteration 1 (e.g. query was too narrow).

    Attributes
    ----------
    _llm : anthropic.Anthropic
        Anthropic SDK client for Brain planning and analysis calls.
    _server_params : StdioServerParameters
        MCP server subprocess connection parameters.
    _model : str
        Model string forwarded to every Anthropic API call.
    _default_max_loops : int
        Fallback loop limit when Manager does not specify one.
    """

    @staticmethod
    def _extract_ticker(task_query: str, directives: dict[str, Any], financial_summary: dict[str, Any]) -> str | None:
        """Backward-compatible wrapper around the shared ticker resolver."""
        return extract_ticker(
            task_query, directives, financial_summary,
        )

    def __init__(
        self,
        server_script_path: str,
        model: str = "claude-haiku-4-5",
        max_loops: int = 2,
        llm_client: anthropic.Anthropic | None = None,
        mcp_server_params: StdioServerParameters | None = None,
    ) -> None:
        # Accept an injected client so tests can pass a mock without
        # making real API calls.
        self._llm               = llm_client or anthropic.Anthropic()
        self._model             = model
        self._default_max_loops = max_loops
        self._server_params     = mcp_server_params or StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None,
        )

        # MCP session is opened once per run() and shared with graph nodes via
        # this slot (nodes receive only `state`, so they read the live session
        # here). brain_analyze doesn't use it — it's a pure LLM call — but the
        # slot being set during its node is harmless.
        self._session: ClientSession | None = None

        # Compile the internal LangGraph state machine once at construction.
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
        """
        BRAIN PASS 1 — Pre-Execution Retrieval Planning.

        Consults Claude before the Executor runs to produce an optimal
        RAG retrieval plan: query string, ticker, and time window.

        This pass is called once per loop iteration, immediately before
        ``_execute_sentiment_pipeline()``. Its output directly configures
        the ``retrieve_social_data`` MCP tool call.

        Parameters
        ----------
        state : SentimentAgentState
            Current agent-local state. Read fields:
            - ``state["loop_counter"]``        : Current iteration index.
            - ``state["task_query"]`` / ``state["manager_directives"]`` /
              ``state["financial_metrics_summary"]`` /
              ``state["aggregated_research_context"]``
                                                  financial_metrics_summary,
                                                  aggregated_research_context.

        Returns
        -------
        dict[str, Any]
            Structured retrieval plan with keys:
            - ``"retrieval_query"`` (str)        : Semantic query for RAG.
            - ``"ticker"``          (str | None) : Ticker to filter by.
            - ``"days_back"``       (int)        : Recency window in days.
            - ``"reasoning"``       (str)        : Claude's brief rationale.
        """
        shared = state.get("shared_manager_ref", {})
        task_query = state.get("task_query", shared.get("task_query", ""))
        directives = state.get("manager_directives", shared.get("manager_directives", {}))
        fin_summary = state.get("financial_metrics_summary", shared.get("financial_metrics_summary", {}))
        research_ctx = state.get("aggregated_research_context", shared.get("aggregated_research_context", []))

        ticker = extract_ticker(task_query, directives, fin_summary)

        # Build financial context snippet for Claude
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

            # Strip markdown fences if present
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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "sentiment.brain_plan")
                    scope.set_tag("iteration", str(state["loop_counter"]))
                    sentry_sdk.capture_exception(exc)
            default_plan = {
                "retrieval_query": f"{ticker or 'market'} earnings sentiment investor reaction",
                "ticker":          ticker,
                "days_back":       int(directives.get("days_back", 7)),
                "reasoning":       "Default plan: Brain planning failed; using heuristic query.",
            }
            state["messages"].append({"role": "assistant",
                                       "content": json.dumps(default_plan)})
            return default_plan

    @traceable(name="sentiment.brain_analyze", run_type="llm")
    async def _brain_analyze(self, state: SentimentAgentState) -> str:
        """
        BRAIN PASS 2 — Post-Execution Sentiment Interpretation.

        Called after the Executor has populated ``finbert_result``,
        ``vader_result``, and ``fear_greed_result``. Consults Claude to
        synthesise all three signals into an actionable narrative verdict.

        This pass replaces a dedicated Checker: Claude is asked to flag
        data quality issues (too few chunks, model divergence, recency gaps)
        inline within the structured JSON response.

        Parameters
        ----------
        state : SentimentAgentState
            Current agent-local state. Read fields:
            - ``state["fear_greed_result"]``  : Composite index output.
            - ``state["finbert_result"]``      : FinBERT probabilities.
            - ``state["vader_result"]``        : VADER compound score.
            - ``state["retrieved_chunks"]``    : For chunk count context.
            - ``state["task_query"]`` / ``state["financial_metrics_summary"]``

        Returns
        -------
        str
            Raw Claude response text (JSON string) stored in
            ``state["brain_reasoning"]``. The caller parses this into the
            final ``sentiment_analysis_summary``.
        """
        shared      = state.get("shared_manager_ref", {})
        task_query  = state.get("task_query", shared.get("task_query", ""))
        fin_summary = state.get("financial_metrics_summary", shared.get("financial_metrics_summary", {}))
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

        # Concise financial context for Claude
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
            "Only attribute a sentiment cause to an aspect listed above with a non-empty evidence_span.\n\n"
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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "sentiment.brain_analyze")
                    sentry_sdk.capture_exception(exc)
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
        """
        EXECUTOR — Four-Step MCP Sentiment Pipeline.

        Executes the fixed tool sequence:
            retrieve_social_data → analyze_finbert → score_vader
            → calculate_fear_greed

        Results from each step are written directly into ``state``.
        Individual tool failures are recorded in
        ``state["extraction_errors"]`` but do not abort the pipeline —
        subsequent steps receive empty payloads and run defensively.

        Pipeline Steps
        --------------
        Step 1 — ``retrieve_social_data``
            Uses ``plan["retrieval_query"]``, ``plan["ticker"]``, and
            ``plan["days_back"]`` from the Brain's retrieval plan.
            Populates ``state["retrieved_chunks"]`` and
            ``state["sources_metadata"]``.

        Step 2 — ``analyze_finbert``
            Passes ``state["retrieved_chunks"]`` to ProsusAI/finbert.
            Skipped (returns empty dict) if no chunks were retrieved.
            Populates ``state["finbert_result"]``.

        Step 3 — ``score_vader``
            Passes ``state["retrieved_chunks"]`` to NLTK VADER.
            Skipped if no chunks were retrieved.
            Populates ``state["vader_result"]``.

        Step 4 — ``calculate_fear_greed``
            Passes ``state["finbert_result"]`` and ``state["vader_result"]``
            to the weighted aggregator.
            Skipped if both upstream results are empty.
            Populates ``state["fear_greed_result"]``.

        Parameters
        ----------
        session : ClientSession
            Active MCP client session connected to the sentiment server.
        state : SentimentAgentState
            Mutable agent-local state. This method writes into:
            ``state["retrieved_chunks"]``, ``state["sources_metadata"]``,
            ``state["finbert_result"]``, ``state["vader_result"]``,
            ``state["fear_greed_result"]``, ``state["extraction_errors"]``.
        plan : dict[str, Any]
            Output from ``_brain_plan()`` containing retrieval parameters.

        Returns
        -------
        None
            All results are written directly into ``state``.
        """

        # Atomically clear per-attempt state. Historical attempts remain
        # immutable in attempts[] and their errors are reported separately.
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

        # -- Internal helper: safe MCP call -----------------------------------
        shared = state.get("shared_manager_ref", {})
        directives_for_attempt = state.get("manager_directives", shared.get("manager_directives", {}))
        session_id = session_from_directives(directives_for_attempt)

        async def _call(tool: str, args: dict) -> dict:
            _publish_progress(
                session_id, "agent_tool_call", agent="sentiment",
                message=f"Sentiment Agent: running '{tool}'...",
                detail={"tool": tool},
            )
            try:
                if sentry_enabled():
                    import sentry_sdk
                    sentry_sdk.add_breadcrumb(
                        category="mcp.sentiment",
                        message=f"Calling {tool}",
                        data={"tool": tool},
                        level="info",
                    )
                result = await session.call_tool(tool, arguments=args)
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
                if sentry_enabled():
                    import sentry_sdk
                    with sentry_sdk.push_scope() as scope:
                        scope.set_tag("tool", tool)
                        scope.set_tag("component", "mcp.sentiment")
                        sentry_sdk.capture_exception(exc)
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

        # -- Step 2: analyze_finbert -----------------------------------------
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

        # -- Step 3: score_vader ---------------------------------------------
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

        # -- Step 4: calculate_fear_greed ------------------------------------
        fb_has_data = state["finbert_result"].get("total_chunks", 0) > 0
        vd_has_data = state["vader_result"].get("total_chunks", 0) > 0

        if fb_has_data or vd_has_data:
            log.info("[Executor] Step 4 — calculate_fear_greed.")
            # Pass custom weights from directives if provided
            directives = directives_for_attempt
            fear_greed_args: dict[str, Any] = {
                "finbert_result": state["finbert_result"],
                "vader_result":   state["vader_result"],
            }
            if "finbert_weight" in directives:
                fear_greed_args["finbert_weight"] = float(directives["finbert_weight"])
            if "vader_weight" in directives:
                fear_greed_args["vader_weight"] = float(directives["vader_weight"])

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

        urls = [str(item.get("url", "")).strip() for item in state["sources_metadata"] if isinstance(item, dict)]
        unique_documents = len({url for url in urls if url}) or len({
            str(chunk).strip() for chunk in state["retrieved_chunks"] if str(chunk).strip()
        })
        unique_domains = len({urlparse(url).netloc.lower() for url in urls if urlparse(url).netloc})
        model_has_data = fb_has_data or vd_has_data
        if not state["retrieved_chunks"] or not model_has_data:
            status = "ERROR" if state["extraction_errors"] and not state["retrieved_chunks"] else "NO_DATA"
        elif unique_documents < 3 or unique_domains < 2:
            status = "LOW_COVERAGE"
        else:
            status = "VALID"
        score = state["fear_greed_result"].get("score")
        polarity = state["fear_greed_result"].get("polarity")
        if polarity is None and score is not None:
            polarity = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
        evidence_factor = min(1.0, unique_documents / 5.0) * min(1.0, unique_domains / 3.0)
        model_confidence = float(state["fear_greed_result"].get("confidence") or 0.0)
        state["fear_greed_result"]["confidence"] = round(model_confidence * evidence_factor, 4)
        state["fear_greed_result"].update({
            "status": status, "polarity": polarity,
            "coverage": {"unique_documents": unique_documents, "unique_domains": unique_domains,
                         "source_independence_factor": round(evidence_factor, 4)},
        })
        state["sentiment_status"] = status
        state["sentiment_polarity"] = polarity
        attempts.append({
            "attempt_number": len(attempts) + 1, "plan": dict(plan), "status": status,
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
    # Each node returns a partial-state dict. Heavy logic stays in _brain_plan /
    # _execute_sentiment_pipeline / _brain_analyze — unchanged. The MCP session
    # is read from self._session (set once per run() before the graph runs).

    async def _node_brain_plan(self, state: SentimentAgentState) -> dict[str, Any]:
        # Increment once per cycle, mirroring the old while-loop.
        new_counter = state["loop_counter"] + 1
        log.info("[Run] Iteration %d", new_counter)
        plan = await self._brain_plan(state)
        return {"loop_counter": new_counter, "current_plan": plan}

    async def _node_execute(self, state: SentimentAgentState) -> dict[str, Any]:
        await self._execute_sentiment_pipeline(
            self._session, state, state.get("current_plan", {})
        )
        # _execute_sentiment_pipeline mutates several state keys in place;
        # return them so the merged graph state reflects the update.
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
        # Brain Pass 2 — pure LLM interpretation, no MCP tools needed.
        state["aspect_observations"] = extract_aspect_observations(
            state["retrieved_chunks"], state["sources_metadata"],
            state.get("finbert_result", {}).get("chunk_scores", []),
        )
        raw_analysis = await self._brain_analyze(state)
        return {"brain_reasoning": raw_analysis, "aspect_observations": state["aspect_observations"]}

    def _should_continue(self, state: SentimentAgentState):
        """
        Conditional edge after the Executor. Mirrors the old while-loop:
          - retrieved any chunks        → analyze (success, exit loop)
          - loop_counter >= max_loops   → analyze (guardrail, even with no data)
          - otherwise                   → brain_plan (retry, broader query)
        """
        directives = state["manager_directives"]
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        if state["retrieved_chunks"]:
            log.info("[Run] Executor retrieved %d chunks — exiting loop.",
                     len(state["retrieved_chunks"]))
            return "analyze"
        if state["loop_counter"] >= max_loops:
            log.warning("[Run] loop_counter=%d >= max_loops=%d — exiting to analyze.",
                        state["loop_counter"], max_loops)
            return "analyze"
        return "brain_plan"

    def _build_graph(self):
        """
        Compile the internal state machine:
            START → brain_plan → execute → (conditional) → brain_plan | analyze
            analyze → END
        """
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
        """
        PRIMARY ENTRY GATEWAY — Drives the Full Lifecycle Loop.

        Called by the Manager Agent to initiate a sentiment analysis task.
        Manages the complete MCP client session lifecycle, spawns the
        agent-local state, drives the two-tiered Brain → Executor loop,
        and commits verified sentiment results into ``shared_state``.

        Lifecycle
        ---------
        1. Open an async MCP client session via ``stdio_client``.
        2. Initialise ``SentimentAgentState`` from ``shared_state``.
        3. Enter the loop (max ``max_loops`` iterations):
            a. Increment ``loop_counter``.
            b. Brain Pass 1 — ``_brain_plan()`` → retrieval plan.
            c. Executor — ``_execute_sentiment_pipeline()`` → all results.
            d. If chunks > 0: break (success). Else if more loops remain:
               continue with a broader query on the next iteration.
        4. Brain Pass 2 — ``_brain_analyze()`` → narrative interpretation.
        5. Assemble ``sentiment_analysis_summary`` from all state fields.
        6. Commit summary into ``shared_state["sentiment_analysis_summary"]``.
        7. Return the mutated ``shared_state`` to the Manager Agent.

        Loop Exit Conditions
        --------------------
        - Chunks > 0 after any Executor run (normal success path).
        - ``loop_counter >= max_loops`` (safety guardrail, even if no data).
        The Brain's second pass always runs regardless of loop exit reason.

        Parameters
        ----------
        shared_state : SharedManagerState
            The public state contract from the Manager Agent.
            Read fields: ``task_query``, ``manager_directives``,
            ``aggregated_research_context``, ``financial_metrics_summary``.

        Returns
        -------
        SharedManagerState
            The same dict mutated in-place to include:
            ``shared_state["sentiment_analysis_summary"]`` — a dict with
            all verified sentiment scores, labels, narrative, and metadata.

        Raises
        ------
        RuntimeError
            If the MCP server subprocess cannot be started.
        """
        directives = shared_state.get("manager_directives", {})
        max_loops  = int(directives.get("max_loops", self._default_max_loops))

        log.info(
            "SentimentAgent.run() started — task_query='%s' max_loops=%d",
            shared_state.get("task_query", ""),
            max_loops,
        )

        # -- Initialise agent-local state -------------------------------------
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

        # -- Open MCP client session, then drive the compiled graph -----------
        # Topology: brain_plan → execute → (conditional) → brain_plan | analyze
        # The brain_analyze step (node "analyze") is a pure LLM call and does
        # not touch the session; it simply runs as the graph's terminal node.
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

        # brain_analyze ran inside the graph's terminal node; its raw output is
        # now in state["brain_reasoning"].
        raw_analysis = state["brain_reasoning"]

        # Parse Brain's analysis JSON (graceful fallback if malformed)
        try:
            analysis = json.loads(raw_analysis)
        except json.JSONDecodeError:
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

        # -- Assemble sentiment_analysis_summary ------------------------------
        fg = state["fear_greed_result"]
        fb = state["finbert_result"]
        vd = state["vader_result"]

        sentiment_analysis_summary: dict[str, Any] = {
            "status":               state.get("sentiment_status", "NO_DATA"),
            "polarity":             state.get("sentiment_polarity"),
            # Core Fear/Greed index
            "ticker":               extract_ticker(
                                        shared_state.get("task_query", ""),
                                        directives,
                                        shared_state.get("financial_metrics_summary", {}),
                                    ),
            "fear_greed_score":     fg.get("score"),
            "fear_greed_label":     fg.get("label"),
            "fear_greed_confidence": fg.get("confidence"),

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
            "sources_metadata":      state["sources_metadata"],
            "aspect_observations":   state.get("aspect_observations", []),
            "causal_aspects_supported": supported_causal_aspects(state.get("aspect_observations", [])),

            # Brain interpretation
            "overall_sentiment":    analysis.get("overall_sentiment"),
            "conviction_level":     analysis.get("conviction_level"),
            "model_agreement":      analysis.get("model_agreement"),
            "key_signals":          analysis.get("key_signals", []),
            "narrative":            analysis.get("narrative", ""),
            "risk_flags":           analysis.get("risk_flags", []),
            "data_quality_note":    analysis.get("data_quality_note", ""),

            # Full diagnostics (for Report Writer / downstream agents)
            "brain_reasoning":       raw_analysis,
            "fear_greed_diagnostics": fg.get("diagnostics", {}),

            # Execution metadata
            "loop_iterations_used": state["loop_counter"],
            "extraction_errors":    state["extraction_errors"],
            "prior_attempt_errors": state.get("prior_attempt_errors", []),
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
