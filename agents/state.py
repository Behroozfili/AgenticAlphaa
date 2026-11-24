
#agents/state.py — Contract-Based State Definitions


from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from typing import Annotated, Any, TypeAlias, Union

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


class ForecastContextPayload(TypedDict):
    """JSON-safe serialized ForecastContext stored in graph state."""
    ticker: str
    as_of: str
    target: str
    horizon_trading_days: int
    lookback_days: int
    benchmark: str
    analysis_mode: str
    cutoff_policy: str
    schema_version: str
    dataset_version: str
    feature_version: str
    model_version: str
    calendar_version: str
    label: str


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_MAX_FEATURE_DEPTH = 4
_MAX_FEATURE_CONTAINER_ITEMS = 100
_MAX_FEATURE_KEY_LENGTH = 128
_MAX_FEATURE_TEXT_LENGTH = 4096


def _materialize_json_value(
    value: object,
    *,
    path: str,
    depth: int,
    active_containers: set[int],
) -> JSONValue:
    if depth > _MAX_FEATURE_DEPTH:
        raise ValueError(f"{path} exceeds maximum JSON depth {_MAX_FEATURE_DEPTH}")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        if isinstance(value, str) and len(value) > _MAX_FEATURE_TEXT_LENGTH:
            raise ValueError(f"{path} text exceeds {_MAX_FEATURE_TEXT_LENGTH} characters")
        return value
    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a cycle")
        if len(value) > _MAX_FEATURE_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds {_MAX_FEATURE_CONTAINER_ITEMS} items")
        active_containers.add(identity)
        try:
            return [
                _materialize_json_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(identity)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{path} contains a cycle")
        if len(value) > _MAX_FEATURE_CONTAINER_ITEMS:
            raise ValueError(f"{path} exceeds {_MAX_FEATURE_CONTAINER_ITEMS} items")
        active_containers.add(identity)
        try:
            materialized: dict[str, JSONValue] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key or len(key) > _MAX_FEATURE_KEY_LENGTH:
                    raise ValueError(f"{path} keys must be non-empty compact strings")
                materialized[key] = _materialize_json_value(
                    item,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                    active_containers=active_containers,
                )
            return materialized
        finally:
            active_containers.remove(identity)
    raise TypeError(f"{path} contains non-JSON-safe {type(value).__name__}")


def validate_forecast_feature_values(values: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    """Validate and recursively materialize compact JSON-native feature values."""
    if not isinstance(values, Mapping):
        raise TypeError("feature values must be a mapping")
    materialized = _materialize_json_value(
        values,
        path="values",
        depth=0,
        active_containers=set(),
    )
    if not isinstance(materialized, dict):  # Root is guaranteed by the Mapping check.
        raise TypeError("feature values must materialize to a dictionary")
    return materialized


class ForecastFeatureSlot(TypedDict, total=False):
    """Compact versioned feature payload; raw evidence stays out of State."""
    version: str
    values: dict[str, JSONValue]
    evidence_ids: list[str]
    missing: bool
    degraded: bool
    errors: list[str]


class MonthlyForecastSlot(TypedDict, total=False):
    status: str
    ticker: str
    as_of: str
    target_date: str
    horizon_trading_days: int
    analysis_mode: str
    cutoff_policy: str
    benchmark: str
    run_id: str
    universe_manifest_hash: str | None
    model_variant: str | None
    feature_materialized_at: str | None
    action: str
    probability_positive: float | None
    expected_return_21d: float | None
    expected_excess_return: float | None
    confidence_interval: list[float] | None
    confidence: float | None
    evidence_ids: list[str]
    warnings: list[str]
    missing: bool
    degraded: bool
    graph_coverage: bool | None
    quality_gate_passed: bool
    quality_gate_failures: list[str]
    specialist_contributions: dict[str, JSONValue]
    decision_drivers: list[dict[str, JSONValue]]
    versions: dict[str, str | None]
    narrative: str
    implementation: dict[str, str | int]


# ══════════════════════════════════════════════════════════════════
# Level 1 — Public contract: Manager ↔ All Specialist Agents
# ══════════════════════════════════════════════════════════════════

class _LegacySharedManagerState(TypedDict, total=True):
    

    task_query:                   str
    manager_directives:           dict[str, Any]
    aggregated_research_context:  list[Union[str, dict]]
    financial_metrics_summary:    dict[str, Any]
    sentiment_analysis_summary:   dict[str, Any]
    agent_execution_history:      list[dict[str, Any]]
    orchestrator_logs:            list[str]
    final_report:                 str


class SharedManagerState(_LegacySharedManagerState, total=False):
    """Legacy required state plus optional monthly-forecast migration slots."""
    forecast_context: ForecastContextPayload
    research_features: ForecastFeatureSlot
    financial_features: ForecastFeatureSlot
    sentiment_features: ForecastFeatureSlot
    market_features: ForecastFeatureSlot
    monthly_forecast: MonthlyForecastSlot


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


# ══════════════════════════════════════════════════════════════
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
