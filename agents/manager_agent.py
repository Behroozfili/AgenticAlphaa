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


"""

from __future__ import annotations

import json
import logging
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
    infer_result_keys
)

import anthropic

from langgraph.graph import END, StateGraph
from langsmith import traceable

from agents.financial_agent import FinancialAnalystAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from agents.state import (
    EvaluationSnapshot,
    ManagerGraphState,
    SharedManagerState,
)
from memory.manager_memory import EvaluationFeedback, ManagerMemory
from core.error_handler import add_breadcrumb, capture_exception, capture_message
from core.progress_bus import publish as _publish_progress
from agents.prompts.promt_manager_agent import (ROUTER_SYSTEM_PROMPT ,EVALUATOR_SYSTEM_PROMPT,FINALISER_SYSTEM_PROMPT,VALID_ACTIONS,ACTION_DESCRIPTIONS)
# QA / validation — checks final_report narration against the period
# provenance tags financial_agent.py attaches to each calculated ratio
# (e.g. catches an annual net_margin narrated as if it described the
# current quarter). Non-blocking: see its usage in _node_finalise below.
# Adjust this import path to wherever validate_period_consistency.py
# actually lives in your project layout (e.g. validation/, tools/qa/).
from evaluation.validate_period_consistency import (
    check_narration_vs_period, detect_material_period_conflicts,
)
from evaluation.recommendation_gate import build_recommendation_gate, contains_actionable_recommendation
from evaluation.canonical_synthesis import build_canonical_synthesis_payload
from evaluation.publication_policy import assemble_publication
from evaluation.final_publication import validate_final_publication
from evaluation.financial_sanity import evaluate_financial_sanity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# NOTE: logging.basicConfig() must NOT be called in library/agent code —
# it hijacks the root logger for the entire process. Configuration belongs
# exclusively in the entry point (api/main.py or __main__ blocks).
log = logging.getLogger("manager-agent")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL             = "claude-haiku-4-5"
_DEFAULT_MAX_ROUTING_LOOPS = 8




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
        llm_client:        anthropic.AsyncAnthropic | None = None,
    ) -> None:
        # Use AsyncAnthropic so Brain calls never block the event loop.
        # Accept an injected client so tests can pass a mock without
        # making real API calls.
        self._llm               = llm_client or anthropic.AsyncAnthropic()
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
                    extract_chunk_text(c)[:200]
                    for c in state.get("aggregated_research_context", [])[:2]
                ],
            }
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
            output_summary = {
                "status":            sentiment_status,
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
            if sentiment_status != "VALID":
                issue = f"Sentiment status is {sentiment_status}; only VALID may advance as successful sentiment."
                verdict = {
                    "passed": False, "score": 40 if sentiment_status == "LOW_COVERAGE" else 0,
                    "issues": [issue], "next_action": "rerun_sentiment",
                    "reasoning": "Missing, errored, or low-coverage input cannot be treated as neutral sentiment.",
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
            financial_failure = agent_name == "FinancialAnalystAgent"
            verdict = {
                "passed":      False if financial_failure else True,
                "score":       0 if financial_failure else 50,
                "issues":      [f"Evaluation API call failed: {exc}"],
                "next_action": (
                    "run_financial" if agent_name == "ResearchAgent"
                    else ("rerun_financial" if financial_failure
                          else "finalise")
                ),
                "reasoning": (
                    "Critical financial evaluation failed closed; valuation cannot advance."
                    if financial_failure else "Evaluation failed; assuming partial pass."
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
        """
        BRAIN PASS 3 — Final Report Synthesis (async).

        Synthesises all agent outputs into a single investment analysis report.
        """
        # Prioritize SEC filing chunks (sec_edgar_filing / sec_edgar_search)
        # ahead of news/rag chunks. A flat [:3] + [:200 chars] truncation
        # (the original behavior) cut every chunk off after ~200
        # characters — nowhere near enough to reach mda/risk_factors text.
        # A later fix gave filing chunks a flat 8000-char budget instead,
        # but that still wasn't enough once financial_statements (often
        # tens of thousands of characters) was requested ALONGSIDE mda in
        # the same call: financial_statements comes first in the JSON
        # (document order) and alone consumed the entire flat budget,
        # so mda never reached the finalizer prompt even though it was
        # successfully fetched (confirmed via direct tool-output
        # inspection). _format_filing_chunk_fairly() now parses the JSON
        # and gives each requested section its OWN budget, so one large
        # section can no longer starve another. Non-filing chunks
        # (news/rag) stay at a smaller flat budget since they're already
        # short, individual article snippets rather than large structured
        # documents.
        all_chunks = state.get("aggregated_research_context", [])

        # If ResearchAgent successfully produced a synthesized summary
        # (see tools/research_tools/context_synthesizer.py), PREFER it
        # alone over the raw chunks — this is where the actual volume
        # reduction happens. The raw chunks are still present in
        # aggregated_research_context (never removed — see that module's
        # docstring on why), so nothing here is lost for downstream
        # consumers that need the raw data (e.g. a numeric faithfulness
        # validator reading the full trace directly); this only changes
        # what the FINALIZER PROMPT itself is built from. Falls back to
        # the previous fair-budgeted raw-chunk selection if no synthesis
        # chunk is present (e.g. synthesis failed and returned None
        # upstream) — so a synthesis failure degrades gracefully instead
        # of losing research content from the report entirely.
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
            # Generous budget: this is already a dense, LLM-compressed
            # summary of everything gathered, not raw source text, so it
            # can afford a larger allowance than any single raw chunk.
            research_lines = "\n".join(
                f"  - {extract_chunk_text(c)[:6000]}" for c in synthesis_chunks
            )
        else:
            log.info(
                "_brain_finalise: no synthesis chunk found — falling back to "
                "raw filing/news chunk fair-budgeting"
            )
            # Prioritize SEC filing chunks (sec_edgar_filing / sec_edgar_search)
            # ahead of news/rag chunks. A flat [:3] + [:200 chars] truncation
            # (the original behavior) cut every chunk off after ~200
            # characters — nowhere near enough to reach mda/risk_factors text.
            # A later fix gave filing chunks a flat 8000-char budget instead,
            # but that still wasn't enough once financial_statements (often
            # tens of thousands of characters) was requested ALONGSIDE mda in
            # the same call: financial_statements comes first in the JSON
            # (document order) and alone consumed the entire flat budget,
            # so mda never reached the finalizer prompt even though it was
            # successfully fetched (confirmed via direct tool-output
            # inspection). _format_filing_chunk_fairly() now parses the JSON
            # and gives each requested section its OWN budget, so one large
            # section can no longer starve another. Non-filing chunks
            # (news/rag) stay at a smaller flat budget since they're already
            # short, individual article snippets rather than large structured
            # documents.
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
            # PERIOD TAGS — do not drop these. Each ratio's "_period" states
            # which reporting period its own inputs actually came from (see
            # financial_agent.py's _call() helper) — e.g. net_margin may be
            # "quarterly:2026-Q1" (a single 3-month quarter) while roe is
            # "annual:2025" (this pipeline has no quarterly balance-sheet
            # source). Passing the bare numbers without these tags is what
            # let a real bug through previously: a quarterly-only net_margin
            # got narrated in the same sentence as *nine-month* cumulative
            # revenue/income figures pulled from raw filing text in
            # RESEARCH HIGHLIGHTS, which use YET ANOTHER period (a 10-Q
            # typically reports both a 3-month-standalone column and a
            # 9-month-year-to-date column) — three different period bases
            # silently merged into one paragraph. See the numeric-fidelity
            # guidance below for the rule this data enables.
            "net_margin_period": fm.get("net_margin", {}).get("_period"),
            # NOTE: current_ratio/de_ratio were previously OMITTED here while
            # every other ratio got its raw value extracted. That left
            # "current_ratio" and "de_ratio" only reachable via
            # composite_score["sub_scores"] — which holds 0-10 NORMALISED
            # contributions to the composite score, not the raw ratios — but
            # under the SAME key names. The finaliser LLM had no raw value to
            # cite for these two metrics, so it pulled the normalised
            # sub-scores instead (e.g. reporting "current ratio of 3.57"
            # when the actual ratio was 1.07, and "debt-to-equity of 0.0"
            # when the actual ratio was 3.87 / "high_leverage"). Providing
            # the raw values explicitly, under unambiguous keys, removes the
            # only path by which that misattribution could happen.
            "current_ratio":     fm.get("current_ratio"),
            "de_ratio":          fm.get("de_ratio", {}).get("de_ratio"),
            "de_ratio_period":   fm.get("de_ratio", {}).get("_period"),
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

            # Phase 1/2 ratios. Simple single-number ratios follow the same
            # raw-value + "_period" pattern as roe/net_margin/de_ratio above;
            # reverse_dcf and altman_z_score have several fields worth
            # keeping together (assumptions/converged; zone/caveat) so they're
            # passed through whole, same as dcf_valuation/composite_score.
            "net_debt_to_ebitda":        fm.get("net_debt_to_ebitda", {}).get("net_debt_to_ebitda"),
            "net_debt_to_ebitda_period": fm.get("net_debt_to_ebitda", {}).get("_period"),
            "fcf_margin":                fm.get("fcf_margin", {}).get("fcf_margin_pct"),
            "fcf_margin_period":         fm.get("fcf_margin", {}).get("_period"),
            "fcf_yield":                 fm.get("fcf_yield", {}).get("fcf_yield_pct"),
            "fcf_yield_period":          fm.get("fcf_yield", {}).get("_period"),
            "ocf_ratio":                 fm.get("ocf_ratio", {}).get("ocf_ratio"),
            "ocf_ratio_period":          fm.get("ocf_ratio", {}).get("_period"),
            "reverse_dcf":               fm.get("reverse_dcf", {}),
            # roic is a documented APPROXIMATION -- see its "note" field
            # (tax_rate_pct is a 21% statutory placeholder, not this
            # company's actual effective rate). Pass "is_approximation" and
            # "note" through explicitly so the finaliser can't cite it as a
            # precise figure the way it must not cite composite_score
            # sub_scores as raw ratios (see numeric-fidelity guidance above).
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

            # Deterministic analysis layer from FinancialAnalystAgent (rule-
            # based, no LLM on its side -- see financial_agent.py's
            # _build_consistency_checks/_build_data_quality_score). The
            # finaliser can use these directly instead of re-deriving trend
            # direction or a reliability read from the raw ratios itself.
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
        # Persist the exact factual contract used by both the first synthesis
        # and any regeneration so traces/tests can audit source selection.
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
                max_tokens=3584,
                temperature=0.0,
                system=FINALISER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            report = response.content[0].text.strip()
            if response.stop_reason == "max_tokens":
                log.warning("[Brain-Finalise] Report may be truncated by max_tokens (%d chars).", len(report))
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
            _publish_progress(
                session_id, "dispatch_end", agent=agent_key,
                message=f"{friendly} completed successfully ({elapsed}s) — returning to Manager",
                detail={"outcome": "success", "duration_s": elapsed, "result_keys": result_keys},
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
                detail={"outcome": "error", "duration_s": elapsed, "error": str(exc)},
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

    @traceable(name="hydrate", run_type="chain")
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
        _publish_progress(
            g["session_id"], "hydrate", agent="manager",
            message=f"Initial state prepared for {ticker or 'analysis'}",
            detail={"ticker": ticker},
        )
        return {"shared_state": shared, "ticker": ticker}

    @traceable(name="brain_route", run_type="chain")
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

    @traceable(name="evaluate", run_type="chain")
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
        """
        NODE: persist — Memory Storage.

        Reads last_evaluation directly from graph state (not memory layer)
        so routing is deterministic regardless of memory availability.

        When evaluation failed, the next action is only forced to a
        rerun_* action if the evaluator's own next_action targets the SAME
        agent that just failed (i.e. it genuinely is a retry). If the
        evaluator instead recommends advancing to a different, not-yet-run
        agent despite the failure, that agent's action keeps its original
        run_/rerun_ label as recommended — it is not force-relabelled as a
        rerun, since it never ran before. See DC-5 below for why this
        distinction matters.
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
            next_action    = snapshot["next_action"]
            next_agent_key = next_action.removeprefix("rerun_").removeprefix("run_")

            if next_agent_key == agent_key:
                # The evaluator wants THIS SAME agent retried — that is a
                # genuine rerun, so force the rerun_* prefix regardless of
                # whether the evaluator's raw next_action already said
                # "rerun_X" or (loosely) "run_X" — this is the one case the
                # original unconditional rewrite was actually meant for.
                updated_action = "rerun_" + next_agent_key
                log.info(
                    "[Node:Persist] Eval failed — retrying same agent '%s', "
                    "action='%s'", agent_key, updated_action,
                )
            else:
                # The evaluator wants to move on to a DIFFERENT agent
                # despite this failure (e.g. "financial data is incomplete,
                # but proceed to sentiment anyway"). That is a genuine
                # first-time dispatch of that other agent, not a rerun.
                #
                # DC-5: the previous version unconditionally rewrote next_action
                # to a rerun_* action on any evaluation failure, which mislabeled
                # this case in orchestrator_logs as "action=rerun_sentiment" even
                # though SentimentAgent had never run before — corrupting the
                # audit trail this logging exists to provide, even though the
                # actual dispatch still worked (agent_key is re-derived from
                # the action string via removeprefix on both "run_" and
                # "rerun_", so the wrong label never affected which agent was
                # actually called). Preserve the evaluator's own action here
                # instead of forcing a rerun_* prefix onto it.
                updated_action = next_action
                log.info(
                    "[Node:Persist] Eval failed on '%s' — evaluator recommends "
                    "advancing to a different agent, action='%s' (label "
                    "preserved, not forced to rerun_*)",
                    agent_key, updated_action,
                )

        log.info("[Node:Persist] Memory persisted for agent_key=%s", agent_key)
        return {"last_action": updated_action}

    @traceable(name="finalise", run_type="chain")
    async def _node_finalise(self, g: ManagerGraphState) -> dict:
        """
        NODE: finalise — Final Report Synthesis.

        Awaits _brain_finalise(), writes the report into shared_state,
        and persists long-term memory to disk.
        """
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
            log.error("[Node:Finalise] Material period conflict limited publication: %s", structured_findings)
            return {"shared_state": shared}

        shared["recommendation_gate"] = build_recommendation_gate(shared)
        final_report = await self._brain_finalise(shared)
        validation = validate_final_publication(final_report, shared)
        shared["final_publication_validation"] = validation
        shared["publication_validation_attempts"] = [{"attempt": 1, "validation": validation}]
        if not validation["passed"]:
            first_candidate = final_report
            final_report = await self._brain_finalise(shared, validation["required_regeneration_context"])
            second = validate_final_publication(final_report, shared)
            shared["final_publication_validation"] = second
            shared["publication_validation_attempts"].append({"attempt": 2, "validation": second})
            if not second["passed"]:
                shared["blocked_final_report"] = final_report
                shared["first_blocked_final_report"] = first_candidate
                shared.update(assemble_publication(final_report, second, shared))
                return {"shared_state": shared}

        # -- QA: period-consistency check (non-blocking) ----------------------
        # Cross-checks each ratio's "_period" provenance tag (set by
        # financial_agent.py — e.g. "annual:2026" vs "quarterly:2026-Q2")
        # against how final_report actually narrates that number (quarterly
        # vs annual language nearby). This is what would have caught the
        # NVDA net_margin bug (a FY2026-annual 55.6% net margin narrated
        # next to Q1 FY2027 quarterly figures) automatically, on every run,
        # without needing a human to notice the mismatch after the fact.
        #
        # Deliberately non-blocking: a false positive here (e.g. an annual
        # figure that happens to sit near an unrelated "quarterly" mention)
        # should not prevent the user from getting their report. Findings
        # are logged, attached to shared_state for downstream inspection/
        # dashboards, and optionally sent to Sentry — nothing more.
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
            validation["critical_findings"]=[*validation.get("critical_findings",[]),*extra]
            validation["required_regeneration_context"]=validation["critical_findings"]
            shared["final_publication_validation"]=validation
            log.error("[Node:Finalise] Material period conflict limited report publication.")
        shared.update(assemble_publication(final_report, shared.get("final_publication_validation",validation), shared))
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

    @traceable(name="ManagerAgent.run", run_type="chain")
    async def run(
        self,
        task_query:         str,
        manager_directives: dict[str, Any] | None = None,
        user_preferences:   dict[str, Any] | None = None,
        client_session_id:  str | None = None,
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
        client_session_id : str | None
            If the caller (e.g. the API route) already generated a
            session_id — typically because the frontend needs to know it
            BEFORE the analysis starts, in order to open an SSE progress
            stream (see core/progress_bus.py) without racing the first
            events — that id is used verbatim instead of generating a new
            one. Falls back to a fresh UUID if not provided, so existing
            callers are unaffected.

        Returns
        -------
        SharedManagerState
            Fully populated state with ``final_report`` and all specialist
            agent outputs committed.
        """
        session_id  = client_session_id or str(uuid.uuid4())[:8]
        directives  = dict(manager_directives or {})
        preferences = dict(user_preferences or {})

        # Propagate the session id to specialist agents via manager_directives
        # (the one free-form channel that's part of the public SharedManagerState
        # contract) so their internal Brain/Executor/Checker layers can publish
        # their own progress events under the same session, via
        # state["shared_manager_ref"]["manager_directives"]["_progress_session_id"].
        directives["_progress_session_id"] = session_id

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
