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
import re
import sys
from typing import Any

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# State contract imports
# ---------------------------------------------------------------------------
# SharedManagerState and SentimentAgentState are declared in agents/state.py
# — the single source of truth for all state TypedDicts across the platform.
from agents.state import SentimentAgentState, SharedManagerState
from langsmith import traceable
from core.observability import sentry_enabled

# ---------------------------------------------------------------------------
# Logging — stderr only; stdout is reserved for MCP JSON-RPC
# NOTE: logging.basicConfig() must NOT be called in library/agent code.
# Configuration belongs exclusively in entry points (api/main.py or __main__).
# ---------------------------------------------------------------------------
log = logging.getLogger("sentiment-agent")


# ---------------------------------------------------------------------------
# Brain system prompts — module-level constants
# ---------------------------------------------------------------------------

_BRAIN_PLAN_SYSTEM_PROMPT = """\
You are the Planning Brain of a SentimentAgent on a financial analysis platform.

Your role: Given a research task and optional financial context, produce a
concise, structured retrieval plan that tells the Executor exactly how to
query the social data RAG pipeline.

You will receive:
  - TASK QUERY             : The original research objective.
  - TICKER                 : Target stock ticker (may be inferred).
  - MANAGER DIRECTIVES     : Configuration hints (days_back, etc.).
  - FINANCIAL CONTEXT      : Optional summary from FinancialAnalystAgent.
  - RESEARCH CONTEXT SAMPLE: First 800 chars of aggregated research chunks.
  - LOOP ITERATION         : Current iteration number.

Output format (strict JSON — no markdown fences, no preamble):
{
  "retrieval_query" : "<semantic search query string, 5-12 words>",
  "ticker"          : "<TICKER or null if not applicable>",
  "days_back"       : <integer, 1-30>,
  "reasoning"       : "<1-2 sentence explanation of the plan>"
}

Rules:
- Output ONLY valid JSON. No explanation outside the JSON object.
- retrieval_query must be specific and sentiment-focused, e.g.:
    "NVIDIA AI chip earnings investor sentiment market reaction"
- days_back: use manager_directives if provided; otherwise default to 7.
- If ticker is ambiguous, extract it from the task_query using context clues.
"""

_BRAIN_ANALYZE_SYSTEM_PROMPT = """\
You are the Analysis Brain of a SentimentAgent on a financial analysis platform.

Your role: Interpret the combined FinBERT + VADER + Fear/Greed sentiment signals
in the context of the research task, and produce a concise, actionable narrative
verdict for the Manager Agent.

You will receive:
  - TASK QUERY        : The original research objective.
  - TICKER            : Target company ticker.
  - FEAR/GREED RESULT : Composite score [-1, +1] and five-band label.
  - FINBERT RESULT    : Deep NLP probabilities (bullish/bearish/neutral).
  - VADER RESULT      : Rule-based compound score and label.
  - CHUNK COUNT       : Number of social/news chunks analysed.
  - FINANCIAL CONTEXT : Optional key metrics from FinancialAnalystAgent.

Your output (strict JSON — no markdown fences, no preamble):
{
  "overall_sentiment"  : "Bullish" | "Bearish" | "Neutral" | "Mixed",
  "conviction_level"   : "High" | "Medium" | "Low",
  "key_signals"        : ["<signal 1>", "<signal 2>", "<signal 3>"],
  "model_agreement"    : "Strong" | "Moderate" | "Weak",
  "narrative"          : "<2-4 sentence actionable interpretation for the Manager Agent>",
  "risk_flags"         : ["<flag 1>", ...],
  "data_quality_note"  : "<brief note on data coverage quality, or empty string>"
}

Rules:
- Output ONLY valid JSON. No explanation outside the JSON object.
- narrative must synthesise FinBERT + VADER + Fear/Greed — do not just repeat numbers.
- Be explicit about model_agreement: Strong = both models align; Weak = diverge > 0.3.
- risk_flags: list data risks (e.g. too few chunks, high neutrality, recency gap).
  Return empty list [] if no flags.
- conviction_level: High if |fear_greed_score| > 0.4 and models agree; Low if < 0.15
  or models strongly diverge.
"""


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
        Anthropic model identifier. Defaults to ``"claude-sonnet-4-20250514"``.
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

    def __init__(
        self,
        server_script_path: str,
        model: str = "claude-sonnet-4-20250514",
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
        log.info(
            "SentimentAgent initialised — model=%s, server=%s",
            model, server_script_path,
        )

    # =========================================================================
    # INTERNAL UTILITY — ticker extraction
    # =========================================================================

    def _extract_ticker(
        self,
        task_query: str,
        directives: dict[str, Any],
        financial_summary: dict[str, Any],
    ) -> str | None:
        """
        Resolve the target ticker symbol from multiple context sources.

        Resolution order (highest priority first):
            1. ``manager_directives["ticker"]``   — explicit Manager override.
            2. ``financial_metrics_summary["ticker"]`` — already resolved by
               FinancialAnalystAgent upstream.
            3. Regex scan of ``task_query`` for 1-5 uppercase letter tokens,
               filtering known stop-words.

        Parameters
        ----------
        task_query : str
            Natural-language research question from the Manager Agent.
        directives : dict[str, Any]
            Manager directives dictionary (may be empty).
        financial_summary : dict[str, Any]
            Financial metrics summary from FinancialAnalystAgent (may be empty).

        Returns
        -------
        str | None
            Uppercase ticker symbol, or ``None`` if extraction fails.
        """
        # Priority 1: explicit directive
        if directives.get("ticker"):
            return str(directives["ticker"]).upper()

        # Priority 2: already resolved by FinancialAnalystAgent
        if financial_summary.get("ticker"):
            return str(financial_summary["ticker"]).upper()

        # Priority 3: heuristic regex scan of task_query
        stop_words = {
            "A", "I", "THE", "AND", "OR", "IN", "OF", "FOR", "AI", "Q1",
            "Q2", "Q3", "Q4", "US", "SEC", "PE", "YOY", "TTM", "EPS",
            "RAG", "NLP", "ETF", "IPO", "GDP", "CEO", "CFO", "MCP",
        }
        candidates = re.findall(r"\b([A-Z]{1,5})\b", task_query)
        for candidate in candidates:
            if candidate not in stop_words:
                log.info("SentimentAgent inferred ticker from task_query: %s", candidate)
                return candidate

        log.warning("SentimentAgent: could not infer ticker from task_query='%s'", task_query)
        return None

    # =========================================================================
    # LAYER 2 — BRAIN (Cognitive / LLM Interpretation Layer)
    # =========================================================================

    @traceable(name="sentiment.brain_plan", run_type="llm")
    def _brain_plan(self, state: SentimentAgentState) -> dict[str, Any]:
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
            - ``state["shared_manager_ref"]``  : task_query, directives,
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
        shared     = state["shared_manager_ref"]
        task_query = shared.get("task_query", "")
        directives = shared.get("manager_directives", {})
        fin_summary = shared.get("financial_metrics_summary", {})
        research_ctx = shared.get("aggregated_research_context", [])

        ticker = self._extract_ticker(task_query, directives, fin_summary)

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
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=256,
                system=_BRAIN_PLAN_SYSTEM_PROMPT,
                messages=state["messages"],
            )
            raw = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw})

            # Strip markdown fences if present
            raw = raw.replace("```json", "").replace("```", "").strip()
            plan = json.loads(raw)
            log.info("[Brain-Plan] Plan: query='%s' ticker=%s days_back=%s",
                     plan.get("retrieval_query"), plan.get("ticker"), plan.get("days_back"))
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
    def _brain_analyze(self, state: SentimentAgentState) -> str:
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
            - ``state["shared_manager_ref"]``  : task_query, financial summary.

        Returns
        -------
        str
            Raw Claude response text (JSON string) stored in
            ``state["brain_reasoning"]``. The caller parses this into the
            final ``sentiment_analysis_summary``.
        """
        shared      = state["shared_manager_ref"]
        task_query  = shared.get("task_query", "")
        fin_summary = shared.get("financial_metrics_summary", {})
        fg          = state["fear_greed_result"]
        fb          = state["finbert_result"]
        vd          = state["vader_result"]

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
            f"FINANCIAL CONTEXT:\n{fin_ctx or 'Not available'}\n\n"
            "Produce your sentiment interpretation JSON verdict."
        )

        state["messages"].append({"role": "user", "content": user_content})
        log.info("[Brain-Analyze] Invoking Claude for sentiment interpretation...")

        try:
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=768,
                system=_BRAIN_ANALYZE_SYSTEM_PROMPT,
                messages=state["messages"],
            )
            raw = response.content[0].text.strip()
            state["messages"].append({"role": "assistant", "content": raw})
            raw = raw.replace("```json", "").replace("```", "").strip()
            log.info("[Brain-Analyze] Interpretation complete.")
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

        # -- Internal helper: safe MCP call -----------------------------------
        async def _call(tool: str, args: dict) -> dict:
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
                return payload
            except Exception as exc:
                state["extraction_errors"].append(f"{tool} exception: {exc}")
                log.exception("[Executor] %s call failed: %s", tool, exc)
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
        days_back       = int(plan.get("days_back", 7))

        log.info("[Executor] Step 1 — retrieve_social_data: query='%s' ticker=%s days=%d",
                 retrieval_query, ticker, days_back)

        retrieval_args: dict[str, Any] = {"query": retrieval_query, "days_back": days_back}
        if ticker:
            retrieval_args["ticker"] = ticker

        social_data = await _call("retrieve_social_data", retrieval_args)

        state["retrieved_chunks"] = social_data.get("chunks", [])
        state["sources_metadata"] = social_data.get("sources", [])
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
                "bullish_prob": 0.0, "bearish_prob": 0.0, "neutral_prob": 0.0,
                "label": "Neutral", "total_chunks": 0, "skipped_chunks": 0,
            }
            state["extraction_errors"].append(
                "analyze_finbert skipped: no social data chunks were retrieved."
            )

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
                "compound": 0.0, "positive_mean": 0.0, "negative_mean": 0.0,
                "neutral_mean": 0.0, "label": "Neutral", "total_chunks": 0, "skipped_chunks": 0,
            }
            state["extraction_errors"].append(
                "score_vader skipped: no social data chunks were retrieved."
            )

        # -- Step 4: calculate_fear_greed ------------------------------------
        fb_has_data = state["finbert_result"].get("total_chunks", 0) > 0
        vd_has_data = state["vader_result"].get("total_chunks", 0) > 0

        if fb_has_data or vd_has_data:
            log.info("[Executor] Step 4 — calculate_fear_greed.")
            # Pass custom weights from directives if provided
            directives = state["shared_manager_ref"].get("manager_directives", {})
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
                "score": 0.0, "label": "Neutral", "finbert_score": 0.0,
                "vader_score": 0.0, "weights": {}, "confidence": 0.0, "diagnostics": {},
            }
            state["extraction_errors"].append(
                "calculate_fear_greed skipped: both FinBERT and VADER produced no data."
            )

        log.info(
            "[Executor] Pipeline complete — chunks=%d finbert=%s vader=%s fg_score=%.4f fg_label=%s",
            len(state["retrieved_chunks"]),
            state["finbert_result"].get("label", "N/A"),
            state["vader_result"].get("label", "N/A"),
            state["fear_greed_result"].get("score", 0.0),
            state["fear_greed_result"].get("label", "N/A"),
        )

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
            "shared_manager_ref": shared_state,
        }

        # -- Open MCP client session ------------------------------------------
        async with stdio_client(self._server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:

                await session.initialize()
                log.info("[Run] MCP session initialised.")

                # -- Main Brain → Executor loop -------------------------------
                while state["loop_counter"] < max_loops:
                    state["loop_counter"] += 1
                    log.info("[Run] Iteration %d / %d", state["loop_counter"], max_loops)

                    # Brain Pass 1: plan the retrieval
                    plan = self._brain_plan(state)

                    # Executor: run the four-step sentiment pipeline
                    await self._execute_sentiment_pipeline(session, state, plan)

                    # Exit if we got data; otherwise retry with a broader query
                    if state["retrieved_chunks"]:
                        log.info(
                            "[Run] Executor retrieved %d chunks — exiting loop.",
                            len(state["retrieved_chunks"]),
                        )
                        break
                    else:
                        log.warning(
                            "[Run] Zero chunks retrieved on iteration %d. "
                            "%s loop iteration(s) remaining.",
                            state["loop_counter"],
                            max_loops - state["loop_counter"],
                        )

        # -- Brain Pass 2: interpret all signals (outside MCP session) --------
        # Runs after the MCP session closes; pure LLM call, no tools needed.
        raw_analysis = self._brain_analyze(state)
        state["brain_reasoning"] = raw_analysis

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
            # Core Fear/Greed index
            "ticker":               self._extract_ticker(
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
        }

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