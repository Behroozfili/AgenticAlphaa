
#agents/state.py — Contract-Based State Definitions


from __future__ import annotations

import operator
from typing import Annotated, Any, Union

from typing_extensions import NotRequired, TypedDict


class FinancialFact(TypedDict, total=False):
    """Period- and scope-aware contract for a sourced financial fact."""
    metric: str
    value: float | int | str | None
    unit: str | None
    ticker: str | None
    fiscal_year: int | None
    fiscal_quarter: str | None
    period_start: str | None
    period_end: str | None
    duration_months: int | None
    filing_type: str | None
    accession_number: str | None
    metric_scope: str | None
    gaap_status: str | None
    source_url: str | None
    provenance_status: str


class TypedClaim(TypedDict, total=False):
    """Claim whose epistemic category cannot be silently inferred by prose."""
    claim_type: str
    text: str
    source_url: str | None
    accession_number: str | None
    speaker: str | None
    derivation: str | None
    confidence: str
    validation_errors: list[str]


# ══════════════════════════════════════════════════════════════════
# Level 1 — Public contract: Manager ↔ All Specialist Agents
# ══════════════════════════════════════════════════════════════════

class SharedManagerState(TypedDict, total=True):
    

    task_query:                   str
    manager_directives:           dict[str, Any]
    aggregated_research_context:  list[Union[str, dict]]
    financial_metrics_summary:    dict[str, Any]
    sentiment_analysis_summary:   dict[str, Any]
    agent_execution_history:      list[dict[str, Any]]
    orchestrator_logs:            list[str]
    final_report:                 str


# ══════════════════════════════════════════════════════════════════
# Level 2 — Private memory: Research Agent internal loop
# ══════════════════════════════════════════════════════════════════

class ResearchAgentState(TypedDict):
    

    messages:            Annotated[list[dict], operator.add]
    context_chunks:      Annotated[list[Union[str, dict]], operator.add]
    called_tool_signatures: Annotated[list[str], operator.add]
    loop_counter:        int
    validation_feedback: str
    is_complete:         bool
    task_query:          str
    manager_directives:  dict[str, Any]


# ══════════════════════════════════════════════════════════════════
# Level 2 — Private memory: Financial Analyst Agent internal loop
# ══════════════════════════════════════════════════════════════════

class FinancialAgentState(TypedDict):
  

    messages:            list[dict]
    raw_numerical_data:  dict[str, Any]
    calculated_ratios:   dict[str, Any]
    loop_counter:        int
    validation_feedback: str
    is_complete:         bool
    task_query:          str
    manager_directives:  dict[str, Any]
    intents:             list[str]
    extra_tools:         list[str]
    rationale:           str
    financial_sanity:    dict[str, Any]


# ══════════════════════════════════════════════════════════════════
# Level 2 — Private memory: Sentiment Agent internal loop
# ══════════════════════════════════════════════════════════════════

class SentimentAgentState(TypedDict):
    

    messages:           list[dict]
    retrieved_chunks:   list[str]
    sources_metadata:   list[dict]
    finbert_result:     dict[str, Any]
    vader_result:       dict[str, Any]
    fear_greed_result:  dict[str, Any]
    brain_reasoning:    str
    loop_counter:       int
    extraction_errors:  list[str]
    prior_attempt_errors: list[str]
    attempts:            list[dict[str, Any]]
    sentiment_status:    str
    sentiment_polarity:  str | None
    task_query:          str
    manager_directives:  dict[str, Any]
    
    financial_metrics_summary:   dict[str, Any]
    aggregated_research_context: list[Union[str, dict]]
 
    current_plan:        dict[str, Any]
    aspect_observations: NotRequired[list[dict[str, Any]]]


# ══════════════════════════════════════════════════════════════════
# Level 3 — Private memory: Manager Agent LangGraph internal state
# ══════════════════════════════════════════════════════════════════

class EvaluationSnapshot(TypedDict, total=False):
   
    step:        str
    passed:      bool
    score:       int
    next_action: str
    issues:      list[str]


class ManagerGraphState(TypedDict):
    

    shared_state:      SharedManagerState
    loop_counter:      int
    last_action:       str
    last_agent_key:    str
    evaluation_passed: bool
    last_evaluation:   NotRequired[EvaluationSnapshot | None]
    ticker:            str | None
    session_id:        str
