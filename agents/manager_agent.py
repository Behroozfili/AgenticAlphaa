"""
agents/manager_agent.py
------------------------
Production-grade ManagerAgent — Central Orchestration Layer for the
Alpha-Agent Node platform.

Architecture
------------

    ┌────────────────────────────────────────────────────────────────┐
    │                   ManagerAgent — StateGraph                    │
    │                                                                │
    │  NODES (LangGraph)                                             │
    │    node: hydrate       → _node_hydrate()                       │
    │    node: brain_route   → _node_brain_route()                   │
    │    node: dispatch      → _node_dispatch()                      │
    │    node: evaluate      → _node_evaluate()                      │
    │    node: persist       → _node_persist()                       │
    │    node: finalise      → _node_finalise()                      │
    │    node: abort         → _node_abort()                         │
    │                                                                │
    │  EDGES                                                         │
    │    START → hydrate → brain_route                               │
    │    brain_route →(conditional)→ dispatch | finalise | abort     │
    │    dispatch → evaluate → persist                               │
    │    persist  →(conditional)→ brain_route | dispatch | abort     │
    │                                                                │
    │  GUARDRAIL: max_routing_loops (default: 8)                     │
    └────────────────────────────────────────────────────────────────┘

Routing Actions (Brain vocabulary)
-----------------------------------
    "run_research"     → dispatch ResearchAgent
    "run_financial"    → dispatch FinancialAnalystAgent
    "run_sentiment"    → dispatch SentimentAgent
    "rerun_research"   → re-dispatch ResearchAgent (with updated directives)
    "rerun_financial"  → re-dispatch FinancialAnalystAgent
    "rerun_sentiment"  → re-dispatch SentimentAgent
    "finalise"         → synthesise final report and exit loop
    "abort"            → exit loop with error state (guardrail hit)

State contract
--------------
    Input  : task_query (str), manager_directives (dict), user_preferences (dict)
    Output : SharedManagerState with final_report populated

Dependencies
------------
    pip install anthropic
    All specialist agents must be importable from agents/
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Union

import anthropic

from langgraph.graph import END, StateGraph

from agents.financial_agent import FinancialAnalystAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from agents.state import (
    EvaluationSnapshot,
    ManagerGraphState,
    SharedManagerState,
)
from memory.manager_memory import EvaluationFeedback, ManagerMemory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("manager-agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL             = "claude-sonnet-4-20250514"
_DEFAULT_MAX_ROUTING_LOOPS = 8

_VALID_ACTIONS: frozenset[str] = frozenset({
    "run_research", "run_financial", "run_sentiment",
    "rerun_research", "rerun_financial", "rerun_sentiment",
    "finalise", "abort",
})

# ---------------------------------------------------------------------------
# Brain system prompts
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = """\
You are the Routing Brain of the ManagerAgent on the Alpha-Agent Node platform.

Your role: Analyse the current SharedManagerState and memory context, then decide
the SINGLE most logical next action to take in the orchestration pipeline.

Available actions:
  "run_research"     — Dispatch ResearchAgent (first time).
  "run_financial"    — Dispatch FinancialAnalystAgent (first time).
  "run_sentiment"    — Dispatch SentimentAgent (first time).
  "rerun_research"   — Re-dispatch ResearchAgent (quality was insufficient).
  "rerun_financial"  — Re-dispatch FinancialAnalystAgent (data was incomplete).
  "rerun_sentiment"  — Re-dispatch SentimentAgent (no chunks retrieved).
  "finalise"         — All required agents have run successfully. Synthesise output.
  "abort"            — Loop guardrail hit or unrecoverable error. Exit gracefully.

Typical execution order:
  run_research → run_financial → run_sentiment → finalise

Routing rules:
  1. Always start with run_research unless research context already exists.
  2. run_financial only after research context is non-empty.
  3. run_sentiment only after financial_metrics_summary exists.
  4. finalise only when all three agents have completed with "success" or "partial".
  5. rerun_* only if the last evaluation score < 50 or next_action == "rerun_*".
  6. abort if loop_counter >= max_routing_loops.

Output format (strict JSON — no markdown, no preamble):
{
  "action"           : "<one of the 8 valid actions>",
  "reasoning"        : "<1-2 sentences justifying the choice>",
  "directive_updates": {}
}

directive_updates: a flat dict of manager_directives keys to update before dispatch.
  Example: {"search_depth": "advanced", "days_back": 14}
  Return {} if no changes needed.
"""

_EVALUATOR_SYSTEM_PROMPT = """\
You are the Quality Evaluator Brain of the ManagerAgent on the Alpha-Agent Node platform.

Your role: Critically assess the output produced by a specialist agent and decide
whether it is sufficient for the Manager to proceed to the next pipeline stage.

You will receive:
  - AGENT NAME    : The specialist agent that just ran.
  - STATE SUMMARY : Key output fields from SharedManagerState.
  - MEMORY CONTEXT: Short-term session history.

Evaluation criteria:
  ResearchAgent  → aggregated_research_context must have >= 3 chunks, non-empty.
  FinancialAgent → financial_metrics_summary must have a non-None composite_score
                   and validation_passed == True.
  SentimentAgent → sentiment_analysis_summary must have a non-None fear_greed_score
                   and total_chunks_analyzed >= 1.

Output format (strict JSON — no markdown, no preamble):
{
  "passed"      : true | false,
  "score"       : <int 0-100>,
  "issues"      : ["<specific issue>", ...],
  "next_action" : "<recommended routing action from the valid set>",
  "reasoning"   : "<1-2 sentence explanation>"
}

Rules:
- Be strict: partial or empty outputs must NOT pass (passed=false).
- next_action must be one of the 8 valid routing actions.
- If passed=true, next_action should advance the pipeline (not rerun).
"""

_FINALISER_SYSTEM_PROMPT = """\
You are the Synthesis Brain of the ManagerAgent on the Alpha-Agent Node platform.

Your role: Produce the final, comprehensive investment analysis report by synthesising
the outputs of all three specialist agents into a single coherent narrative.

You will receive the complete SharedManagerState summary containing:
  - TASK QUERY             : The original research objective.
  - RESEARCH CONTEXT       : Key themes from the ResearchAgent.
  - FINANCIAL METRICS      : Verified ratios from the FinancialAnalystAgent.
  - SENTIMENT ANALYSIS     : Fear/Greed + narrative from the SentimentAgent.
  - EXECUTION HISTORY      : Which agents ran and their outcomes.

Your output must be a clean, well-structured investment analysis report in plain text
(NOT JSON). It should cover:

  1. Executive Summary (2-3 sentences)
  2. Financial Health (key metrics, grade, interpretation)
  3. Market Sentiment (Fear/Greed score, label, narrative)
  4. Research Highlights (top 3-5 themes from research context)
  5. Risk Factors (data quality flags, model disagreements, macro risks)
  6. Conclusion & Outlook (1-2 sentence directional view)

Write in professional, analytical English. Be concise — target 400-600 words.
Do NOT include any JSON or code blocks.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_chunk_text(chunk: Union[str, dict]) -> str:
    """
    Safely extract plain text from a research context chunk.
    Handles both legacy plain-string chunks and dict chunks with a "text" key.
    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("text", str(chunk))
    return str(chunk)


def _feedback_to_snapshot(fb: EvaluationFeedback) -> EvaluationSnapshot:
    """Convert an EvaluationFeedback dataclass to a plain EvaluationSnapshot dict."""
    return EvaluationSnapshot(
        step=        fb.step,
        passed=      fb.passed,
        score=       fb.score,
        next_action= fb.next_action,
        issues=      list(fb.issues),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ManagerAgent
# ══════════════════════════════════════════════════════════════════════════════

class ManagerAgent:
    """
    Central Orchestration Layer for the Alpha-Agent Node platform.

    The ManagerAgent is the top-level entry point for all analysis tasks.
    It owns the SharedManagerState lifecycle, drives the specialist agent
    pipeline, maintains cognitive memory across steps, and synthesises
    the final report.

    Responsibilities
    ----------------
    1. Ingest the user's task query and hydrate a fresh SharedManagerState.
    2. Query ManagerMemory for relevant short-term and long-term context.
    3. Consult the Brain (Claude) to decide the next routing action.
    4. Dispatch the chosen specialist agent and await its result.
    5. Evaluate the agent's output quality via a second Brain call.
    6. Persist execution milestones and insights back into ManagerMemory.
    7. Loop until all agents complete or the guardrail is hit.
    8. Synthesise the final investment analysis report.

    Parameters
    ----------
    research_agent : ResearchAgent
        Instantiated ResearchAgent (LangGraph-based).
    financial_agent : FinancialAnalystAgent
        Instantiated FinancialAnalystAgent (MCP-based).
    sentiment_agent : SentimentAgent
        Instantiated SentimentAgent (MCP-based).
    memory : ManagerMemory
        Injected ManagerMemory instance (short + long term).
    model : str
        Anthropic model identifier for Brain calls.
    max_routing_loops : int
        Hard limit on Brain→Dispatch iterations. Default: 8.
    """

    def __init__(
        self,
        research_agent:    ResearchAgent,
        financial_agent:   FinancialAnalystAgent,
        sentiment_agent:   SentimentAgent,
        memory:            ManagerMemory,
        model:             str = _DEFAULT_MODEL,
        max_routing_loops: int = _DEFAULT_MAX_ROUTING_LOOPS,
    ) -> None:
        # Use AsyncAnthropic so Brain calls never block the event loop
        self._llm               = anthropic.AsyncAnthropic()
        self._model             = model
        self._max_routing_loops = max_routing_loops
        self._memory            = memory

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
        """
        Initialise a fresh SharedManagerState with all required fields.
        Called once at the start of every ``run()``.
        """
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
        """Pull relevant context from ManagerMemory (short-term + long-term)."""
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
        """
        BRAIN PASS 1 — Routing Decision (async).

        Uses AsyncAnthropic so the event loop is never blocked.
        """
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
                max_tokens=256,
                system=_ROUTER_SYSTEM_PROMPT,
                messages=self._memory.get_messages(),
            )
            raw = response.content[0].text.strip()
            self._memory.add_message(role="assistant", content=raw)
            raw = raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw)

            action = decision.get("action", "")
            if action not in _VALID_ACTIONS:
                log.warning("[Brain-Route] Invalid action '%s' — defaulting to abort.", action)
                decision["action"] = "abort"

            log.info(
                "[Brain-Route] Decision: action=%s reasoning='%s'",
                decision["action"],
                decision.get("reasoning", "")[:80],
            )
            return decision

        except Exception as exc:
            log.exception("[Brain-Route] Claude call failed: %s", exc)
            agents_run = self._memory.agents_run()
            if "ResearchAgent" not in agents_run:
                fallback_action = "run_research"
            elif "FinancialAnalystAgent" not in agents_run:
                fallback_action = "run_financial"
            elif "SentimentAgent" not in agents_run:
                fallback_action = "run_sentiment"
            else:
                fallback_action = "finalise"
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
        """
        BRAIN PASS 2 — Output Quality Evaluation (async).

        Grades the agent's output and recommends a next routing action.
        Does NOT call memory.add_evaluation() — that is the sole
        responsibility of _node_persist() to avoid double-writing.
        """
        if agent_name == "ResearchAgent":
            output_summary = {
                "chunks_count": len(state.get("aggregated_research_context", [])),
                "sample": [
                    _extract_chunk_text(c)[:200]
                    for c in state.get("aggregated_research_context", [])[:2]
                ],
            }
        elif agent_name == "FinancialAnalystAgent":
            fm = state.get("financial_metrics_summary", {})
            output_summary = {
                "ticker":            fm.get("ticker"),
                "composite_score":   fm.get("composite_score", {}).get("score"),
                "grade":             fm.get("composite_score", {}).get("grade"),
                "validation_passed": fm.get("validation_passed"),
                "extraction_errors": fm.get("extraction_errors", []),
                "pe_ratio":          fm.get("pe_ratio", {}).get("pe_ratio"),
            }
        else:
            sm = state.get("sentiment_analysis_summary", {})
            output_summary = {
                "fear_greed_score":  sm.get("fear_greed_score"),
                "fear_greed_label":  sm.get("fear_greed_label"),
                "overall_sentiment": sm.get("overall_sentiment"),
                "conviction_level":  sm.get("conviction_level"),
                "total_chunks":      sm.get("total_chunks_analyzed"),
                "extraction_errors": sm.get("extraction_errors", []),
            }

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
                max_tokens=256,
                system=_EVALUATOR_SYSTEM_PROMPT,
                messages=self._memory.get_messages(),
            )
            raw = response.content[0].text.strip()
            self._memory.add_message(role="assistant", content=raw)
            raw = raw.replace("```json", "").replace("```", "").strip()
            verdict = json.loads(raw)

        except Exception as exc:
            log.exception("[Brain-Evaluate] Evaluation failed: %s", exc)
            verdict = {
                "passed":      True,
                "score":       50,
                "issues":      [f"Evaluation API call failed: {exc}"],
                "next_action": (
                    "run_financial" if agent_name == "ResearchAgent"
                    else ("run_sentiment" if agent_name == "FinancialAnalystAgent"
                          else "finalise")
                ),
                "reasoning": "Evaluation failed; assuming partial pass.",
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

    async def _brain_finalise(self, state: SharedManagerState) -> str:
        """
        BRAIN PASS 3 — Final Report Synthesis (async).

        Synthesises all agent outputs into a single investment analysis report.
        """
        research_sample = state.get("aggregated_research_context", [])[:3]
        fm = state.get("financial_metrics_summary", {})
        sm = state.get("sentiment_analysis_summary", {})

        fin_summary = {
            "ticker":            fm.get("ticker"),
            "company_name":      fm.get("company_name"),
            "sector":            fm.get("sector"),
            "current_price":     fm.get("current_price"),
            "market_cap":        fm.get("market_cap"),
            "composite_score":   fm.get("composite_score", {}),
            "pe_ratio":          fm.get("pe_ratio", {}).get("pe_ratio"),
            "roe":               fm.get("roe", {}).get("roe_pct"),
            "net_margin":        fm.get("net_margin", {}).get("net_margin_pct"),
            "revenue_cagr":      fm.get("revenue_cagr", {}).get("cagr_pct"),
            "validation_passed": fm.get("validation_passed"),
        }
        sent_summary = {
            "fear_greed_score":  sm.get("fear_greed_score"),
            "fear_greed_label":  sm.get("fear_greed_label"),
            "overall_sentiment": sm.get("overall_sentiment"),
            "conviction_level":  sm.get("conviction_level"),
            "model_agreement":   sm.get("model_agreement"),
            "narrative":         sm.get("narrative"),
            "risk_flags":        sm.get("risk_flags", []),
        }

        research_lines = "\n".join(
            f"  - {_extract_chunk_text(c)[:200]}"
            for c in research_sample
        )

        user_content = (
            f"TASK QUERY:\n{state.get('task_query', '')}\n\n"
            f"RESEARCH HIGHLIGHTS (top 3 chunks):\n{research_lines}\n\n"
            f"FINANCIAL METRICS SUMMARY:\n{json.dumps(fin_summary, indent=2)}\n\n"
            f"SENTIMENT ANALYSIS SUMMARY:\n{json.dumps(sent_summary, indent=2)}\n\n"
            f"EXECUTION HISTORY:\n"
            + json.dumps(state.get("agent_execution_history", []), indent=2)
            + "\n\nSynthesize the final investment analysis report."
        )

        log.info("[Brain-Finalise] Synthesising final report...")
        try:
            response = await self._llm.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_FINALISER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            report = response.content[0].text.strip()
            log.info("[Brain-Finalise] Report generated (%d chars).", len(report))
            return report

        except Exception as exc:
            log.exception("[Brain-Finalise] Synthesis failed: %s", exc)
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
        """
        Route SharedManagerState to the correct specialist agent and await
        its ``run()`` method. Records execution time and outcome.
        """
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

        try:
            state   = await agent.run(state)
            elapsed = round(time.time() - t0, 2)

            result_keys = _infer_result_keys(agent_key, state)

            record.outcome     = "success"
            record.duration_s  = elapsed
            record.result_keys = result_keys

            state["agent_execution_history"].append({
                "agent_name":    agent_class,
                "dispatched_at": record.dispatched_at,
                "outcome":       "success",
                "duration_s":    elapsed,
                "result_keys":   result_keys,
                "error_message": None,
            })
            state["orchestrator_logs"].append(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"[SUCCESS] {agent_class} completed in {elapsed}s — keys={result_keys}"
            )
            log.info("[Dispatch] %s completed in %.2fs.", agent_class, elapsed)

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            record.outcome       = "error"
            record.duration_s    = elapsed
            record.error_message = str(exc)

            state["agent_execution_history"].append({
                "agent_name":    agent_class,
                "dispatched_at": record.dispatched_at,
                "outcome":       "error",
                "duration_s":    elapsed,
                "result_keys":   [],
                "error_message": str(exc),
            })
            state["orchestrator_logs"].append(
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"[ERROR] {agent_class} failed: {exc}"
            )
            log.exception("[Dispatch] %s raised an exception.", agent_class)

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
        """
        Write execution milestones into ManagerMemory.

        Stores evaluation feedback and extracts salient facts into long-term
        memory. Does NOT call add_evaluation() — that was already done in
        _node_evaluate() to avoid double-writing.
        """
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

    async def _node_hydrate(self, g: ManagerGraphState) -> dict:
        """
        NODE: hydrate — Session Setup & State Initialisation.

        Applies long-term memory preferences to directives.
        """
        shared     = g["shared_state"]
        directives = dict(shared.get("manager_directives", {}))

        if "search_depth" not in directives:
            saved_depth = self._memory.get_preference("search_depth")
            if saved_depth:
                directives["search_depth"] = saved_depth
                shared["manager_directives"] = directives

        ticker = directives.get("ticker")
        log.info("[Node:Hydrate] session=%s ticker=%s", g["session_id"], ticker)
        return {"shared_state": shared, "ticker": ticker}

    async def _node_brain_route(self, g: ManagerGraphState) -> dict:
        """
        NODE: brain_route — Routing Decision.

        Increments loop_counter, recalls memory, awaits _brain_route()
        to get the next action, and merges any directive_updates.
        """
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
        return {
            "loop_counter": new_counter,
            "last_action":  action,
            "shared_state": shared,
            "ticker":       updated_ticker,
        }

    async def _node_dispatch(self, g: ManagerGraphState) -> dict:
        """
        NODE: dispatch — Specialist Agent Execution.

        Maps last_action to the correct specialist agent and awaits agent.run().
        """
        action    = g["last_action"]
        shared    = g["shared_state"]
        agent_key = action.removeprefix("rerun_").removeprefix("run_")

        log.info("[Node:Dispatch] action=%s agent_key=%s", action, agent_key)
        shared = await self._dispatch(action=action, state=shared)

        return {"shared_state": shared, "last_agent_key": agent_key}

    async def _node_evaluate(self, g: ManagerGraphState) -> dict:
        """
        NODE: evaluate — Brain Quality Assessment.

        Awaits _brain_evaluate(), stores the EvaluationFeedback in memory
        exactly once, and snapshots it into graph state for routing.
        """
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

        snapshot = _feedback_to_snapshot(evaluation)

        log.info(
            "[Node:Evaluate] %s → passed=%s score=%d next=%s",
            agent_class, evaluation.passed, evaluation.score, evaluation.next_action,
        )
        return {
            "evaluation_passed": evaluation.passed,
            "last_evaluation":   snapshot,
        }

    async def _node_persist(self, g: ManagerGraphState) -> dict:
        """
        NODE: persist — Memory Storage.

        Reads last_evaluation directly from graph state (not memory layer)
        so routing is deterministic regardless of memory availability.

        When evaluation failed, overrides last_action to the rerun_* action
        so _should_continue_after_persist() routes back to dispatch correctly.
        """
        agent_key = g["last_agent_key"]
        shared    = g["shared_state"]

        snapshot: EvaluationSnapshot | None = g.get("last_evaluation")

        if snapshot is not None:
            # Reconstruct a minimal EvaluationFeedback for _persist()
            # _persist() only reads step/score/issues — no add_evaluation() call
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
            rerun_action = snapshot["next_action"]
            # Ensure we always produce a rerun_* action on failure
            if rerun_action.startswith("run_") and not rerun_action.startswith("rerun_"):
                rerun_action = "rerun_" + rerun_action.removeprefix("run_")
            updated_action = rerun_action
            log.info(
                "[Node:Persist] Eval failed — overriding action to '%s'", updated_action
            )

        log.info("[Node:Persist] Memory persisted for agent_key=%s", agent_key)
        return {"last_action": updated_action}

    async def _node_finalise(self, g: ManagerGraphState) -> dict:
        """
        NODE: finalise — Final Report Synthesis.

        Awaits _brain_finalise(), writes the report into shared_state,
        and persists long-term memory to disk.
        """
        shared     = g["shared_state"]
        ticker     = g["ticker"]
        session_id = g["session_id"]

        final_report = await self._brain_finalise(shared)
        shared["final_report"] = final_report

        if ticker:
            self._memory.store_ticker_insight(ticker, {
                "last_task_query":          shared.get("task_query", ""),
                "last_final_report_length": len(final_report),
            })
        self._memory.store_heuristic(
            f"session_{session_id}_loops_used", g["loop_counter"]
        )
        self._memory.persist_long_term()

        log.info(
            "[Node:Finalise] Report generated (%d chars). Long-term memory persisted.",
            len(final_report),
        )
        return {"shared_state": shared}

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
        self._memory.persist_long_term()

        log.warning(
            "[Node:Abort] Orchestration aborted at loop %d / %d.",
            loop, self._max_routing_loops,
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
        """
        CONDITIONAL EDGE — Route after persist node.

        Reads last_action which was already overridden by _node_persist()
        to a rerun_* action when evaluation failed. This makes routing
        deterministic — no ambiguity between run_* and rerun_*.

        Returns
        -------
        "brain_route"  evaluation passed — advance to next Brain decision
        "dispatch"     evaluation failed — rerun the same agent directly
        "abort"        guardrail hit
        """
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
        """
        Build and compile the ManagerAgent LangGraph StateGraph.

        Graph topology
        --------------
        START → hydrate → brain_route
                              │
                  ┌───────────┼──────────────┐
                  ▼           ▼              ▼
              dispatch     finalise        abort
                  │            │              │
              evaluate        END            END
                  │
              persist
                  │
          ┌───────┴───────────┐
          ▼                   ▼
      brain_route          dispatch
      (eval passed)        (eval failed → rerun)
        """
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

    async def run(
        self,
        task_query:         str,
        manager_directives: dict[str, Any] | None = None,
        user_preferences:   dict[str, Any] | None = None,
    ) -> SharedManagerState:
        """
        PRIMARY ENTRY POINT — Invoke the compiled LangGraph StateGraph.

        Parameters
        ----------
        task_query : str
            The user's natural-language analysis objective.
        manager_directives : dict[str, Any] | None
            Initial configuration hints: ticker, max_loops, search_depth,
            days_back, peers.
        user_preferences : dict[str, Any] | None
            Cross-session preferences stored in long-term memory.

        Returns
        -------
        SharedManagerState
            Fully populated state with ``final_report`` and all specialist
            agent outputs committed.
        """
        session_id  = str(uuid.uuid4())[:8]
        directives  = dict(manager_directives or {})
        preferences = dict(user_preferences or {})

        self._memory.new_session(session_id=session_id, task_query=task_query)
        for k, v in preferences.items():
            self._memory.store_preference(k, v)

        log.info(
            "ManagerAgent.run() started — session=%s task='%s'",
            session_id, task_query[:80],
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
            raise RuntimeError(f"ManagerAgent internal graph failed: {exc}") from exc

        final_shared = final["shared_state"]
        log.info(
            "ManagerAgent.run() complete — session=%s loops=%d report_chars=%d",
            session_id,
            final["loop_counter"],
            len(final_shared.get("final_report", "")),
        )
        return final_shared


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _infer_result_keys(agent_key: str, state: SharedManagerState) -> list[str]:
    """
    Infer which SharedManagerState keys an agent populated.

    Parameters
    ----------
    agent_key : str
        One of ``"research"`` | ``"financial"`` | ``"sentiment"``.
    state : SharedManagerState
        State after the agent's ``run()`` returned.

    Returns
    -------
    list[str]
        List of state keys that appear non-empty after the agent ran.
    """
    key_map = {
        "research":  "aggregated_research_context",
        "financial": "financial_metrics_summary",
        "sentiment": "sentiment_analysis_summary",
    }
    result_key = key_map.get(agent_key)
    if result_key and state.get(result_key):
        return [result_key]
    return []