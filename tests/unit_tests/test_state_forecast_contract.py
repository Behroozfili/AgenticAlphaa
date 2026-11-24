import json
import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType

import pytest

from agents.state import (
    ForecastContextPayload,
    SharedManagerState,
    validate_forecast_feature_values,
)
from core.forecast_contract import ForecastContext


LEGACY_REQUIRED_KEYS = {
    "task_query",
    "manager_directives",
    "aggregated_research_context",
    "financial_metrics_summary",
    "sentiment_analysis_summary",
    "agent_execution_history",
    "orchestrator_logs",
    "final_report",
}


class CustomMapping(Mapping[str, object]):
    def __init__(self, values: dict[str, object]):
        self._values = values

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def legacy_state():
    return {
        "task_query": "Analyze MSFT",
        "manager_directives": {},
        "aggregated_research_context": [],
        "financial_metrics_summary": {},
        "sentiment_analysis_summary": {},
        "agent_execution_history": [],
        "orchestrator_logs": [],
        "final_report": "",
    }


def test_new_state_fields_are_optional_and_legacy_required_keys_are_unchanged():
    assert SharedManagerState.__required_keys__ == frozenset(LEGACY_REQUIRED_KEYS)
    assert {
        "forecast_context", "research_features", "financial_features",
        "sentiment_features", "market_features", "monthly_forecast",
    } <= SharedManagerState.__optional_keys__


def test_legacy_state_json_serialization_is_unchanged():
    state: SharedManagerState = legacy_state()
    assert json.loads(json.dumps(state)) == legacy_state()


def test_serialized_context_fits_typed_state_without_custom_encoder():
    context = ForecastContext(
        ticker="MSFT",
        as_of="2024-05-31T20:00:00Z",
        schema_version="schema-v1",
        dataset_version="dataset-v1",
        feature_version="features-v1",
        model_version="model-v1",
    )
    payload: ForecastContextPayload = context.to_dict()
    state: SharedManagerState = {**legacy_state(), "forecast_context": payload}
    restored = json.loads(json.dumps(state))
    assert ForecastContext.from_dict(restored["forecast_context"]) == context


def test_feature_values_accept_compact_recursive_json_and_serialize():
    values = {
        "momentum": 0.25,
        "missing": None,
        "flags": [True, False],
        "nested": {"period": "2025-Q1", "scores": [1, 2.5]},
    }
    validated = validate_forecast_feature_values(values)
    assert json.loads(json.dumps(validated, allow_nan=False)) == values
    assert validated is not values


def test_feature_values_materialize_nested_mapping_types_and_lists_without_aliasing():
    source_list = [MappingProxyType({"proxy_leaf": 3})]
    custom = CustomMapping({"custom_leaf": source_list})
    source = MappingProxyType({"nested": custom})

    validated = validate_forecast_feature_values(source)

    assert validated == {"nested": {"custom_leaf": [{"proxy_leaf": 3}]}}
    assert type(validated) is dict
    assert type(validated["nested"]) is dict
    assert type(validated["nested"]["custom_leaf"]) is list
    assert type(validated["nested"]["custom_leaf"][0]) is dict
    assert validated["nested"]["custom_leaf"] is not source_list
    assert json.loads(json.dumps(validated, allow_nan=False)) == validated

    source_list.append({"later": 4})
    assert validated == {"nested": {"custom_leaf": [{"proxy_leaf": 3}]}}


def test_feature_values_reject_cycles():
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycle"):
        validate_forecast_feature_values({"cyclic": cyclic})


@pytest.mark.parametrize("invalid", [
    {"tuple": (1, 2)},
    {"bytes": b"raw"},
    {"object": object()},
    {"nan": math.nan},
    {"infinity": math.inf},
    {"too_deep": {"a": {"b": {"c": {"d": {"e": 1}}}}}},
    {"too_long": "x" * 4097},
])
def test_feature_values_reject_non_json_or_non_compact_payloads(invalid):
    with pytest.raises((TypeError, ValueError)):
        validate_forecast_feature_values(invalid)


def test_feature_values_reject_non_string_or_empty_keys():
    with pytest.raises(ValueError):
        validate_forecast_feature_values({"nested": {1: "bad"}})
    with pytest.raises(ValueError):
        validate_forecast_feature_values({"": 1})
