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
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Union

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
from core.observability import sentry_enabled
from core.progress_bus import publish as _publish_progress

# QA / validation — checks final_report narration against the period
# provenance tags financial_agent.py attaches to each calculated ratio
# (e.g. catches an annual net_margin narrated as if it described the
# current quarter). Non-blocking: see its usage in _node_finalise below.
# Adjust this import path to wherever validate_period_consistency.py
# actually lives in your project layout (e.g. validation/, tools/qa/).
from evaluation.validate_period_consistency import check_narration_vs_period

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

_VALID_ACTIONS: frozenset[str] = frozenset({
    "run_research", "run_financial", "run_sentiment",
    "rerun_research", "rerun_financial", "rerun_sentiment",
    "finalise", "abort",
})

# Human-readable descriptions for each action — single source of truth.
# The prompt is generated from this dict so the list never drifts from
# the actual frozenset (M-6 fix).
_ACTION_DESCRIPTIONS: dict[str, str] = {
    "run_research":    "Dispatch ResearchAgent (first time).",
    "run_financial":   "Dispatch FinancialAnalystAgent (first time).",
    "run_sentiment":   "Dispatch SentimentAgent (first time).",
    "rerun_research":  "Re-dispatch ResearchAgent (quality was insufficient).",
    "rerun_financial": "Re-dispatch FinancialAnalystAgent (data was incomplete).",
    "rerun_sentiment": "Re-dispatch SentimentAgent (no chunks retrieved).",
    "finalise":        "All required agents have run successfully. Synthesise output.",
    "abort":           "Loop guardrail hit or unrecoverable error. Exit gracefully.",
}

# Build the actions block dynamically — guaranteed in sync with _VALID_ACTIONS.
_actions_block: str = """
""".join(
    f'  "{action}"{"." * (18 - len(action))} — {desc}'
    for action, desc in _ACTION_DESCRIPTIONS.items()
)

# ---------------------------------------------------------------------------
# Brain system prompts
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = f"""\
You are the Routing Brain of the ManagerAgent on the Alpha-Agent Node platform.

Your role: Analyse the current SharedManagerState and memory context, then decide
the SINGLE most logical next action to take in the orchestration pipeline.

Available actions:
{_actions_block}

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
{{
  "action"           : "<one of the 8 valid actions>",
  "reasoning"        : "<1-2 sentences justifying the choice>",
  "directive_updates": {{}}
}}

directive_updates: a flat dict of manager_directives keys to update before dispatch.
  Example: {{"search_depth": "advanced", "days_back": 14}}
  Return {{}} if no changes needed.
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
  2. Financial Health (key metrics, grade, interpretation, DCF valuation,
     capital allocation — see guidance below)
  3. Market Sentiment (Fear/Greed score, label, narrative)
  4. Research Highlights (top 3-5 themes from research context)
  5. Risk Factors (data quality flags, model disagreements, macro risks)
  6. Scenario Analysis (Bull / Base / Bear) — see guidance below
  7. Management Commentary Assessment — see guidance below
  8. Conclusion & Outlook (1-2 sentence directional view)

Write in professional, analytical English. Be concise — target 550-800 words.
Do NOT include any JSON or code blocks.

SCENARIO ANALYSIS GUIDANCE (section 6): Write three short paragraphs — Bull
case, Base case, Bear case — each grounded in data you actually have (the
financial metrics, sentiment signals, peer comparison, and risk flags
already provided). The Bull case should name the specific positive signals
that would need to persist or strengthen; the Bear case should name the
specific risk flags or metrics that would need to worsen; the Base case is
the most-likely path given the current mix of signals. If dcf_valuation's
bear/base/bull per-share values are available, cite the matching one in
each paragraph as a numeric anchor (see DCF guidance below) — this is
still not a probability, just "here is what this scenario implies for
value under these growth assumptions."

Do NOT assign a numeric probability (e.g. "60% likely") to any scenario.
An LLM-generated percentage for a market scenario is not a statistically
grounded estimate — it would look quantitative without being backed by any
actual probabilistic model, which is misleading to a reader making
investment decisions. If you want to convey relative likelihood, use plain
language ("the base case is the most consistent with current signals")
rather than a fabricated number.

MANAGEMENT COMMENTARY ASSESSMENT GUIDANCE (section 7): If MD&A (Management's
Discussion and Analysis) text is present in the research context, write 2-3
sentences characterizing management's own framing of the period — are they
emphasizing growth investments, defending margins, flagging headwinds,
citing specific strategic bets (e.g. AI infrastructure spend, capacity
expansion)? Quote or closely paraphrase specific language ONLY if it's
distinctive and material (per the copyright rules already governing this
system — short paraphrase, not verbatim reproduction beyond a few words).

This is a QUALITATIVE read of tone and stated priorities, not a scored
metric — do not invent a "management quality score" or letter grade for
this section; there is no rigorous basis for one from a single filing's
MD&A alone. If no MD&A text is present in the research context, write one
sentence noting it wasn't available rather than fabricating an assessment.

DCF & CAPITAL ALLOCATION GUIDANCE (within section 2, Financial Health): If
"dcf_valuation" data is present and has no error, it contains THREE DCF runs
— "bear", "base", "bull" — each at a different FCF growth assumption, NOT
a single point estimate. Report the base-case enterprise_value / per-share
value as the headline number, but ALSO state the bear-to-bull range, and
explicitly compare it to the current market price:
  - If even the bull case is well below market price, say so plainly —
    this means the market is pricing in growth beyond what even an
    optimistic multi-year projection here captures (longer runway, lower
    effective discount rate, or momentum/speculative factors). This is a
    genuine, useful finding — do not describe it as a modelling failure.
  - If the base or bear case is close to market price, that's a
    different, more reassuring signal — say so too.
Always carry forward the "note" field's caveat (simplified WACC, no
net-debt adjustment) in your own words. In the Scenario Analysis section
(section 6), use the matching bear/base/bull DCF per-share values as the
numeric anchor for each narrative case, so the scenarios aren't purely
qualitative.

If "dcf_monte_carlo" data is also present and has no error, it gives a
CONTINUOUS probability range (P10/P50/P90 percentiles from simulating the
growth-rate assumption) rather than three discrete points — mention this
range once, briefly, in the Financial Health section as a complement to
the bear/base/bull scenarios (e.g. "a Monte Carlo simulation over the
growth-rate assumption puts the P10–P90 range at $X–$Y per share").
IMPORTANT: this only randomises the growth-rate input (per its own "note"
field) — do NOT describe it as a fully calibrated probabilistic valuation
model; carry forward its scope caveat the same way you do for the DCF
scenarios' caveat.

If "peer_comparison" data is present, cite BOTH the raw multiple comparison
(primary P/E vs peer avg P/E) AND, if available, the
"growth_adjusted_comparison" sub-field (primary PEG vs peer-average PEG).
These two can disagree — a stock can have a much higher P/E than peers
while still being cheaper on a growth-adjusted (PEG) basis if it's growing
faster, or vice versa. When they disagree, say so explicitly rather than
picking one framing; that disagreement is itself informative for the
reader. If growth_adjusted_comparison's interpretation is
"insufficient_data", note that the growth-adjusted view wasn't available
rather than omitting the topic entirely.

If "capital_allocation" data is present, briefly note the balance between
buybacks, dividends, and capex (e.g. "capital return via buybacks
significantly exceeds capex, suggesting management sees limited high-return
reinvestment opportunities at scale" — only if the data actually supports
that reading). If either dcf_valuation or capital_allocation is missing/
unavailable, skip it rather than fabricating a value.

IMPORTANT — numeric fidelity: When citing a financial ratio (P/E, ROE, net
margin, current ratio, D/E, revenue CAGR, etc.), always use the top-level raw
value provided for that metric (e.g. the "current_ratio" or "de_ratio" key).
NEVER use the numbers inside "composite_score.sub_scores" as if they were the
ratio itself — those are normalised 0-10 contributions to the composite
score, not the actual metric value, even though some share the same name.

IMPORTANT — period fidelity: Several metrics carry a matching "<metric>_period"
field (e.g. "net_margin_period", "roe_period", "de_ratio_period",
"revenue_cagr_period") stating which reporting period that metric's OWN
inputs actually came from — values look like "quarterly:2026-Q1" (a single
3-month quarter), "annual:2025" (a full fiscal year), "ttm", "mrq", or
"mixed". These periods are NOT guaranteed to agree with each other, and
that's fine — e.g. net_margin can legitimately be "quarterly:2026-Q1" while
roe is "annual:2025" (this pipeline has no quarterly balance-sheet source).
Follow these rules when writing:
  1. Never state or imply that a metric describes a different period than
     its own "<metric>_period" says. If net_margin_period is
     "quarterly:2026-Q1", say "Q1 FY2026 net margin" (or similar) — do NOT
     call it the "nine-month" or "full-year" net margin.
  2. RESEARCH HIGHLIGHTS may contain raw revenue/net-income figures quoted
     directly from the filing text (e.g. "$241.8 billion in revenue over
     nine months"). These are a DIFFERENT, independent source from
     FINANCIAL METRICS SUMMARY and are NOT guaranteed to be the same period
     as any "<metric>_period" — a 10-Q commonly reports both a 3-month
     standalone column and a 9-month year-to-date column. Do NOT combine a
     research-highlight revenue/income figure with a financial-metrics
     ratio in the same sentence (e.g. "$241.8B in revenue... with a net
     margin of X%") unless their periods actually match — state each
     figure's period explicitly instead, in separate sentences if needed,
     or simply don't restate the ratio in that sentence.
  3. If you're unsure a research-highlight figure and a metrics-summary
     ratio share a period, err toward NOT putting them in the same
     sentence rather than guessing they align.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_news_chunk_fairly(chunk_text: str, max_articles: int = 5, per_article_budget: int = 220) -> str:
    """
    Format a news_search chunk so multiple articles survive, instead of
    the flat 200-char truncation cutting off partway through the FIRST
    article's title alone.

    Regression this guards against: news_search's raw JSON puts
    query/effective_query/total_results BEFORE the "articles" list, and
    a flat [:200] truncation on the whole serialized blob rarely reaches
    past the first article's title — meaning a chunk with 8 real,
    specific, often highly-relevant articles (e.g. "JPMorgan reversed
    course on Tesla, price target +227%") contributed effectively ZERO
    information to the finalizer, even though the tool call succeeded and
    fetched all 8. This surfaces title + a short snippet of description
    for up to `max_articles`, so the finalizer actually sees the specific
    events, not just a truncated fragment of the first headline.

    Falls back to a flat truncation if the embedded JSON can't be parsed.
    """
    idx = chunk_text.rfind("─")
    json_start = chunk_text.find("{", idx) if idx != -1 else chunk_text.find("{")
    if json_start == -1:
        return chunk_text[:per_article_budget * 2]

    try:
        payload = json.loads(chunk_text[json_start:])
    except (json.JSONDecodeError, ValueError):
        return chunk_text[:per_article_budget * 2]

    articles = payload.get("articles")
    if not isinstance(articles, list) or not articles:
        return chunk_text[:per_article_budget * 2]

    header = f"[NEWS: query={payload.get('effective_query', payload.get('query', '?'))!r}, {payload.get('total_results', '?')} total results]"
    lines = [header]
    for a in articles[:max_articles]:
        title = a.get("title", "")
        desc = a.get("description", "")
        published = a.get("published_at", "")
        lines.append(f"  • [{published}] {title} — {desc[:per_article_budget]}")
    return "\n".join(lines)


def _extract_chunk_text(chunk: Union[str, dict]) -> str:
    """
    Safely extract plain text from a research context chunk.

    DC-3: The isinstance(chunk, str) branch is kept for safety but is
    currently unreachable — all callers in the pipeline produce dict chunks
    with a "text" key. If the pipeline schema changes, this guard prevents
    a silent crash.
    """
    if isinstance(chunk, str):
        # NOTE: unreachable in the current pipeline (all chunks are dicts).
        # Kept as a defensive fallback in case the schema changes upstream.
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("text", str(chunk))
    return str(chunk)


def _format_filing_chunk_fairly(chunk_text: str, per_section_budget: int = 3000) -> str:
    """
    Format a sec_edgar_filing chunk so every requested section gets a FAIR
    character budget, instead of naively truncating the whole serialized
    JSON blob.

    Regression this guards against: sec_edgar_filing's "sections" dict
    preserves document order — for a 10-Q that's typically
    financial_statements, then mda, then risk_factors. financial_statements
    is routinely tens of thousands of characters (full balance sheet,
    income statement, cash flow statement, and notes). When the Brain
    requests sections=["financial_statements", "mda"] together (observed
    routinely in real traces) and the WHOLE chunk is truncated to a single
    flat budget, financial_statements alone consumes the entire budget —
    mda is fully present in the fetched data (confirmed via direct tool
    output inspection) but never reaches the finalizer prompt at all,
    because the flat truncation cuts the string before mda's text even
    starts. Parsing the JSON and budgeting PER SECTION fixes this: each
    section gets its own space regardless of how large its neighbors are.

    Falls back to a flat truncation of the raw text if the embedded JSON
    can't be parsed (e.g. unexpected format) — never raises, never returns
    less information than the old behavior would have.
    """
    # The chunk's raw text looks like:
    #   ──────────────
    #   [TOOL: sec_edgar_filing] | [QUERY: TSLA]
    #   ──────────────
    #   {...json...}
    # Find the JSON payload by locating the last line of dashes and taking
    # everything after it.
    marker = "\n" + "─" * 10  # tolerate variable dash-line lengths
    idx = chunk_text.rfind("─")
    json_start = chunk_text.find("{", idx) if idx != -1 else chunk_text.find("{")
    if json_start == -1:
        return chunk_text[:per_section_budget * 2]  # not JSON at all; flat fallback

    try:
        payload = json.loads(chunk_text[json_start:])
    except (json.JSONDecodeError, ValueError):
        return chunk_text[:per_section_budget * 2]  # unparseable; flat fallback

    sections = payload.get("sections")
    if not isinstance(sections, dict) or not sections:
        return chunk_text[:per_section_budget * 2]  # no sections dict; flat fallback

    header = (
        f"[SEC FILING: {payload.get('ticker', '?')} {payload.get('form_type', '?')}, "
        f"filed {payload.get('filed_at', '?')}]"
    )
    parts = [header]
    for name, text in sections.items():
        parts.append(f"--- {name} ---\n{text[:per_section_budget]}")
    return "\n".join(parts)


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
                system=_ROUTER_SYSTEM_PROMPT,
                messages=self._memory.get_messages(),
            )
            if response.stop_reason == "max_tokens":
                log.warning("[Brain-Route] Response truncated by max_tokens — JSON may be invalid.")
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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "brain_route")
                    sentry_sdk.capture_exception(exc)
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
                temperature=0.0,
                max_tokens=768,
                system=_EVALUATOR_SYSTEM_PROMPT,
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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "brain_evaluate")
                    scope.set_tag("agent_name", agent_name)
                    sentry_sdk.capture_exception(exc)
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
            if "[SYNTHESIZED RESEARCH SUMMARY" in _extract_chunk_text(c)[:60]
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
                f"  - {_extract_chunk_text(c)[:6000]}" for c in synthesis_chunks
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
                if "[TOOL: sec_edgar_filing]" in _extract_chunk_text(c)[:150]
            ]
            news_chunks = [
                c for c in all_chunks
                if "[TOOL: news_search]" in _extract_chunk_text(c)[:150]
            ]
            other_chunks = [c for c in all_chunks if c not in filing_chunks and c not in news_chunks]
            research_sample = filing_chunks[:2] + news_chunks[:2] + other_chunks[:3]
            research_lines = "\n".join(
                f"  - {_format_filing_chunk_fairly(_extract_chunk_text(c))}" if c in filing_chunks
                else f"  - {_format_news_chunk_fairly(_extract_chunk_text(c))}" if c in news_chunks
                else f"  - {_extract_chunk_text(c)[:200]}"
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
                max_tokens=3584,
                temperature=0.0,
                system=_FINALISER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            report = response.content[0].text.strip()
            if response.stop_reason == "max_tokens":
                log.warning("[Brain-Finalise] Report may be truncated by max_tokens (%d chars).", len(report))
            log.info("[Brain-Finalise] Report generated (%d chars).", len(report))
            return report

        except Exception as exc:
            log.exception("[Brain-Finalise] Synthesis failed: %s", exc)
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "brain_finalise")
                    sentry_sdk.capture_exception(exc)
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
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.add_breadcrumb(
                    category="manager_agent.dispatch",
                    message=f"Dispatching {agent_class}",
                    data={"agent_class": agent_class, "action": action},
                    level="info",
                )
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

            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("agent_name", agent_class)
                    scope.set_tag("action", action)
                    sentry_sdk.capture_exception(exc)

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

        snapshot = _feedback_to_snapshot(evaluation)

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
        final_report = await self._brain_finalise(shared)
        shared["final_report"] = final_report

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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "qa.period_consistency")
                    sentry_sdk.capture_exception(exc)

        shared["qa_period_findings"] = period_findings
        if period_findings:
            log.warning(
                "[Node:Finalise] %d period-consistency issue(s) found in "
                "%s report: %s",
                len(period_findings), ticker or "?", period_findings,
            )
            if sentry_enabled():
                import sentry_sdk
                sentry_sdk.capture_message(
                    f"Period-consistency issue(s) in {ticker or '?'} report "
                    f"({len(period_findings)} finding(s))",
                    level="warning",
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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "memory.persist_long_term")
                    sentry_sdk.capture_exception(exc)

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
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "memory.persist_long_term")
                    sentry_sdk.capture_exception(exc)

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