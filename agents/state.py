"""
agents/state.py — Contract-Based State Definitions
====================================================
Single source of truth for ALL state TypedDicts used across the
Alpha-Agent Node platform.

  Level 1 — SharedManagerState:
      The public API contract between the Manager Agent and ALL specialist
      downstream agents (Research Agent, Financial Analyst Agent, Sentiment
      Agent, Report Writer Agent, etc.).
      Only fields that cross agent boundaries live here.
      Each specialist agent owns exactly the fields it writes; all others
      are treated as read-only.

  Level 2 — Agent-Private States (all declared in this file):
      Each specialist agent has its own isolated private TypedDict defined
      below. These are instantiated inside the agent's run() method and
      destroyed when run() returns. They are NEVER passed back to the Manager.

      ResearchAgentState     → private memory for the Research Agent loop.
      FinancialAgentState    → private memory for the Financial Analyst Agent loop.
      SentimentAgentState    → private memory for the Sentiment Agent loop.

Design principle (Contract-Based Design):
    Manager passes SharedManagerState in
        → Specialist Agent hydrates its private state from SharedManagerState
        → Specialist Agent processes internally using its private state
        → Specialist Agent commits its results into its owned field(s)
        → Returns the mutated SharedManagerState out to the Manager.

    Internal loop states are NEVER leaked across agent boundaries.

Field Ownership Map
-------------------
    Field                        Owner Agent
    ───────────────────────────  ─────────────────────────
    task_query                   Manager (writes), all (read)
    manager_directives           Manager (writes), all (read)
    aggregated_research_context  ResearchAgent
    financial_metrics_summary    FinancialAnalystAgent
    sentiment_analysis_summary   SentimentAgent
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from typing_extensions import TypedDict


# ══════════════════════════════════════════════════════════════════
# Level 1 — Public contract: Manager ↔ All Specialist Agents
# ══════════════════════════════════════════════════════════════════

class SharedManagerState(TypedDict, total=False):
    """
    Global public interface shared between the Manager Agent and all
    downstream specialist agents on the Alpha-Agent Node platform.

    This TypedDict is the single source of truth for cross-agent communication.
    It is instantiated by the Manager Agent, passed to each specialist agent
    in sequence (or in parallel where safe), and returned with the relevant
    owned fields populated.

    Ownership Rules
    ---------------
    - Each field is OWNED by exactly one agent (the agent that writes it).
    - All other agents treat non-owned fields as READ-ONLY.
    - The Manager Agent owns ``task_query`` and ``manager_directives``.
    - Specialist agents must NEVER overwrite fields owned by other agents.

    Fields
    ------
    task_query : str
        The natural-language research question or analysis objective posed
        by the Manager Agent. Read by all specialist agents to understand
        the scope of the task.
        Example: "What is NVIDIA's competitive position in AI chips for Q1 2025?"

        Owner  : Manager Agent
        Readers: All specialist agents

    manager_directives : dict[str, Any]
        Optional configuration hints from the Manager Agent that guide
        specialist agent behaviour without hard-coding logic into agents.

        Recognised directive keys (cross-agent):
            "ticker"           (str)   : Target stock ticker, e.g. "NVDA".
            "max_loops"        (int)   : Override internal loop limit (default: 3).
            "search_depth"     (str)   : "basic" | "advanced" for search tools.
            "days_back"        (int)   : News recency window in days.
            "required_sources" (list)  : Tools that must be called, e.g.
                                         ["tavily_search", "sec_edgar_filing"].
            "form_type"        (str)   : SEC filing type, e.g. "10-K".
            "peers"            (list)  : Peer ticker symbols for comparison,
                                         e.g. ["AMD", "INTC", "QCOM"].

        Owner  : Manager Agent
        Readers: All specialist agents

    aggregated_research_context : list[str]
        Accumulates all raw text chunks gathered during the Research Agent's
        loop. Populated by the Research Agent and consumed as background
        knowledge by the Financial Analyst Agent, Sentiment Agent, and
        Report Writer Agent.
        Starts as an empty list; the Research Agent appends to it.

        Owner  : Research Agent
        Readers: Financial Analyst Agent, Sentiment Agent, Report Writer Agent

    financial_metrics_summary : dict[str, Any]
        Structured dictionary of verified financial metrics committed by the
        FinancialAnalystAgent after completing its three-tiered extraction
        and validation loop.

        Guaranteed top-level keys (present even if value is None):
            "ticker"                   (str | None)  : Resolved ticker symbol.
            "company_name"             (str | None)  : Official company name.
            "sector"                   (str | None)  : GICS sector.
            "industry"                 (str | None)  : GICS industry.
            "current_price"            (float | None): Latest market price (USD).
            "market_cap"               (float | None): Market capitalisation (USD).
            "beta"                     (float | None): 5-year monthly beta.
            "52w_high"                 (float | None): 52-week high price (USD).
            "52w_low"                  (float | None): 52-week low price (USD).
            "pe_ratio"                 (dict)        : P/E ratio + interpretation.
            "roe"                      (dict)        : Return on Equity (%).
            "net_margin"               (dict)        : Net Profit Margin (%).
            "de_ratio"                 (dict)        : Debt-to-Equity ratio.
            "revenue_cagr"             (dict)        : Revenue CAGR (%).
            "composite_score"          (dict)        : Weighted health score 0-100
                                                       and letter grade A-F.
            "annual_revenue_history"   (list[dict])  : [{year, revenue, yoy_growth}]
            "revenue_growth_ttm"       (float | None): Trailing 12-month YoY growth.
            "xbrl_total_assets"        (list[dict])  : Most recent SEC assets entry.
            "xbrl_total_liabilities"   (list[dict])  : Most recent SEC liabilities.
            "loop_iterations_used"     (int)         : Extraction loop iterations.
            "validation_passed"        (bool)        : Whether Checker constraints passed.
            "extraction_errors"        (list[str])   : Any per-tool errors encountered.

        Owner  : Financial Analyst Agent
        Readers: Sentiment Agent, Report Writer Agent, Manager Agent

    sentiment_analysis_summary : dict[str, Any]
        Structured dictionary of market sentiment signals produced by the
        SentimentAgent after running its Brain → Executor pipeline.

        Guaranteed top-level keys (present even if value is None):
            "ticker"              (str | None)  : Ticker analysed.
            "fear_greed_score"    (float | None): Composite score [-1.0, +1.0].
            "fear_greed_label"    (str | None)  : "Extreme Fear" … "Extreme Greed".
            "fear_greed_confidence" (float)     : |score| heuristic [0, 1].
            "finbert_label"       (str | None)  : "Bullish" | "Bearish" | "Neutral".
            "finbert_bullish_prob" (float)      : Mean bullish probability [0, 1].
            "finbert_bearish_prob" (float)      : Mean bearish probability [0, 1].
            "finbert_neutral_prob" (float)      : Mean neutral probability [0, 1].
            "vader_compound"      (float)       : Mean VADER compound [-1, +1].
            "vader_label"         (str | None)  : "Bullish" | "Bearish" | "Neutral".
            "total_chunks_analyzed" (int)       : Social/news chunks processed.
            "sources_metadata"    (list[dict])  : Source metadata for each chunk.
            "brain_reasoning"     (str)         : Claude's final interpretation
                                                   of the combined signals.
            "loop_iterations_used" (int)        : Executor loop iterations run.
            "extraction_errors"   (list[str])   : Per-tool error messages, if any.

        Owner  : Sentiment Agent
        Readers: Report Writer Agent, Manager Agent
    """

    task_query:                   str
    manager_directives:           dict[str, Any]
    aggregated_research_context:  list[str]
    financial_metrics_summary:    dict[str, Any]
    sentiment_analysis_summary:   dict[str, Any]


# ══════════════════════════════════════════════════════════════════
# Level 2 — Private memory: Research Agent internal loop
# ══════════════════════════════════════════════════════════════════

class ResearchAgentState(TypedDict):
    """
    Isolated private memory for the Research Agent's internal LangGraph
    state machine.

    This state is instantiated fresh at the start of each ``run()`` call
    from the contents of SharedManagerState, and destroyed after the loop
    completes. It is NEVER passed back to the Manager directly.

    Fields
    ------
    messages : Annotated[list[dict], operator.add]
        Running conversation log between the Brain node and Claude.
        Uses ``operator.add`` so LangGraph automatically appends new messages
        rather than overwriting the list on each state update.
        Each entry is a dict with keys:
            "role"    : "user" | "assistant"
            "content" : str

    context_chunks : Annotated[list[str], operator.add]
        Accumulator for all text snippets gathered by the Executor node
        across all loop iterations.
        Uses ``operator.add`` so chunks are appended, never overwritten.
        On loop completion, this list is copied into
        ``SharedManagerState["aggregated_research_context"]``.

    loop_counter : int
        Monotonically increasing counter incremented by the Executor node
        at the start of each iteration.
        Used by ``_should_continue()`` to enforce the max-loop guardrail.

    validation_feedback : str
        Textual critique written by the Checker node when the gathered
        context is insufficient. Fed back into the Brain node as additional
        context for the next planning cycle.
        Empty string ("") signals no feedback (first iteration or last pass).

    is_complete : bool
        Completion flag set by the Checker node.
        True  → context is sufficient; graph routes to END.
        False → context needs improvement; graph loops back to Brain.

    shared_manager_ref : SharedManagerState
        A read-only structured reference copy of the original shared state.
        Stored here so every node has access to ``task_query`` and
        ``manager_directives`` without needing them injected separately.
        Must not be mutated by any node.
    """

    messages:            Annotated[list[dict], operator.add]
    context_chunks:      Annotated[list[str],  operator.add]
    loop_counter:        int
    validation_feedback: str
    is_complete:         bool
    shared_manager_ref:  SharedManagerState


# ══════════════════════════════════════════════════════════════════
# Level 2 — Private memory: Financial Analyst Agent internal loop
# ══════════════════════════════════════════════════════════════════

class FinancialAgentState(TypedDict):
    """
    Isolated private memory for the FinancialAnalystAgent's internal
    three-tiered execution loop (Brain → Executors → Checker).

    This state is instantiated fresh at the start of each ``run()`` call
    from the contents of SharedManagerState, and destroyed after the loop
    completes. It is NEVER passed back to the Manager directly.

    On loop completion, the agent distils the relevant fields into a clean
    ``financial_metrics_summary`` dict and writes it into SharedManagerState.

    Fields
    ------
    messages : list[dict]
        Running conversational log between the Brain node and Claude used
        for multi-turn planning across loop iterations.
        Each entry is a dict with keys:
            "role"    : "user" | "assistant"
            "content" : str
        Unlike ResearchAgentState, this field is NOT annotated with
        operator.add — the Financial Agent manages appending manually
        so the Brain can inject system-level feedback at any position.

    raw_numerical_data : dict[str, Any]
        Storage for all raw financial statements and market data retrieved
        from MCP tool servers during ``_execute_data_extraction()``.

        Expected keys after a successful extraction:
            "ticker"             (str)       : Resolved ticker symbol.
            "yahoo_ratios"       (dict)       : Full yfinance info payload.
            "revenue_growth"     (dict)       : Annual + quarterly income data.
            "xbrl_financials"    (dict)       : SEC EDGAR XBRL balance-sheet data.
            "extraction_errors"  (list[str])  : Per-tool error messages, if any.

    calculated_ratios : dict[str, Any]
        Storage for all computed accounting metrics returned by the
        FinancialRatioCalculator MCP server during
        ``_execute_ratio_computation()``.

        Expected keys after a successful computation:
            "pe"              (dict) : P/E ratio + interpretation label.
            "roe"             (dict) : Return on Equity (%) + label.
            "net_margin"      (dict) : Net Profit Margin (%) + label.
            "de_ratio"        (dict) : Debt-to-Equity ratio + label.
            "cagr"            (dict) : Revenue CAGR (%) + label.
            "composite_score" (dict) : Weighted health score 0-100 + grade A-F.

    loop_counter : int
        Monotonically increasing counter incremented once at the start of
        each iteration of the Brain → Executor → Checker cycle.
        Compared against ``manager_directives["max_loops"]`` (default 3)
        to enforce the safety guardrail in ``run()``.

    validation_feedback : str
        Actionable textual critique produced by ``_check_data_quality()``
        when one or more data integrity constraints fail.
        Fed back into ``_brain()`` as additional context for the next
        planning cycle so the Brain can adjust its tool priorities.
        Empty string ("") signals no outstanding issues (first iteration
        or all constraints passed on the previous iteration).

    is_complete : bool
        Completion flag set by ``_check_data_quality()``.
        True  → all data integrity constraints passed; ``run()`` exits the
                loop and commits results to SharedManagerState.
        False → at least one constraint failed; loop continues with
                updated ``validation_feedback`` for the Brain.

    shared_manager_ref : SharedManagerState
        A read-only structured reference to the original SharedManagerState
        passed in by the Manager Agent at the start of ``run()``.
        Stored here so all three internal layers (Brain, Executors, Checker)
        can access ``task_query`` and ``manager_directives`` without
        requiring them to be injected as separate parameters.
        Must NOT be mutated by any internal node.
    """

    messages:            list[dict]
    raw_numerical_data:  dict[str, Any]
    calculated_ratios:   dict[str, Any]
    loop_counter:        int
    validation_feedback: str
    is_complete:         bool
    shared_manager_ref:  SharedManagerState


# ══════════════════════════════════════════════════════════════════
# Level 2 — Private memory: Sentiment Agent internal loop
# ══════════════════════════════════════════════════════════════════

class SentimentAgentState(TypedDict):
    """
    Isolated private memory for the SentimentAgent's internal
    two-tiered execution loop (Brain → Executor).

    This state is instantiated fresh at the start of each ``run()`` call
    from the contents of SharedManagerState, and destroyed after the loop
    completes. It is NEVER passed back to the Manager directly.

    On loop completion, the agent distils its findings into a clean
    ``sentiment_analysis_summary`` dict and writes it into SharedManagerState.

    Note: SentimentAgent uses a Brain → Executor two-tier architecture
    (no dedicated Checker layer). The Brain performs both planning and
    final interpretation of the aggregated sentiment signals.

    Fields
    ------
    messages : list[dict]
        Running conversational log between the Brain and Claude across
        loop iterations. Used for multi-turn reasoning where the Brain
        evaluates Executor results and decides follow-up actions.
        Each entry is a dict with keys:
            "role"    : "user" | "assistant"
            "content" : str
        Not annotated with operator.add — the Brain manages appending
        manually to allow system-level context injection at any position.

    retrieved_chunks : list[str]
        Accumulates raw social and news text chunks fetched from the
        RAG pipeline via the ``retrieve_social_data`` MCP tool call.
        Populated by the Executor and consumed by both FinBERT and VADER.

    sources_metadata : list[dict]
        Parallel list of source metadata dicts for each entry in
        ``retrieved_chunks``. Forwarded into the final summary for
        downstream auditability.
        Keys per entry: ticker, source_type, published_at, url, title,
        rrf_score.

    finbert_result : dict[str, Any]
        Raw JSON payload returned by the ``analyze_finbert`` MCP tool.
        Populated by the Executor.
        Expected keys: bullish_prob, bearish_prob, neutral_prob, label,
        total_chunks, skipped_chunks.

    vader_result : dict[str, Any]
        Raw JSON payload returned by the ``score_vader`` MCP tool.
        Populated by the Executor.
        Expected keys: compound, positive_mean, negative_mean,
        neutral_mean, label, total_chunks, skipped_chunks.

    fear_greed_result : dict[str, Any]
        Raw JSON payload returned by the ``calculate_fear_greed`` MCP tool.
        Populated by the Executor after both FinBERT and VADER have run.
        Expected keys: score, label, finbert_score, vader_score,
        weights, confidence, diagnostics.

    brain_reasoning : str
        Claude's final narrative interpretation of the aggregated
        sentiment signals, written during the Brain's analysis pass.
        Committed into ``sentiment_analysis_summary["brain_reasoning"]``.
        Empty string before the Brain's final pass runs.

    loop_counter : int
        Monotonically increasing counter incremented once per Executor
        cycle. Used to enforce the ``max_loops`` safety guardrail
        inside ``run()``. Unlike ResearchAgent and FinancialAgent, the
        Sentiment Agent loop is typically single-pass (loop_counter
        reaches 1 on a clean run).

    extraction_errors : list[str]
        Accumulates per-tool error strings recorded by the Executor
        when an MCP tool call fails or returns an error payload.
        Forwarded into ``sentiment_analysis_summary["extraction_errors"]``.

    shared_manager_ref : SharedManagerState
        A read-only structured reference to the original SharedManagerState
        passed in by the Manager Agent at the start of ``run()``.
        Stored here so both the Brain and Executor can access
        ``task_query``, ``manager_directives``, and
        ``aggregated_research_context`` without separate parameter passing.
        Must NOT be mutated by any internal layer.
    """

    messages:           list[dict]
    retrieved_chunks:   list[str]
    sources_metadata:   list[dict]
    finbert_result:     dict[str, Any]
    vader_result:       dict[str, Any]
    fear_greed_result:  dict[str, Any]
    brain_reasoning:    str
    loop_counter:       int
    extraction_errors:  list[str]
    shared_manager_ref: SharedManagerState
