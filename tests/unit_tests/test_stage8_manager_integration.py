from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent, _manager_trace_inputs, _manager_trace_outputs
from agents.monthly_forecast_agent import MonthlyForecastAgent
from agents.executor_helpers.monthly_forecast_helper import (
    STAGE08_IMPLEMENTATION_IDENTITY,
    assemble_monthly_forecast,
)
from agents.prompts.prompt_manager_agent import MONTHLY_MANAGER_SYSTEM_PROMPT
from agents.prompts.prompt_monthly_forecast import MONTHLY_FORECAST_SYSTEM_PROMPT
from core.progress_bus import close_session, get_queue


FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "stage8_manager_integration.json").read_text()
)


def feature_slot(kind: str, context: dict, run_id: str, *, degraded: bool = False) -> dict:
    values = {"dataset_version": context["dataset_version"], "run_id": run_id}
    if kind == "research":
        values.update({
            "timestamps": [context["as_of"]], "graph_coverage": True,
            "vector_coverage": True, "source_diversity": 2,
        })
    return {
        "version": f"{kind}-v1", "values": values,
        "evidence_ids": [f"{kind}:{context['ticker'].lower()}"],
        "missing": False, "degraded": degraded, "errors": [],
    }


def complete_state(ticker: str = "MSFT", run_id: str = "run-msft") -> dict:
    context = deepcopy(FIXTURE["contexts"][ticker])
    universe_manifest_hash = f"sha256:universe-{ticker.lower()}-2026q3"
    model_variant = f"full_model_v1-{ticker.lower()}"
    feature_materialized_at = context["as_of"]
    prediction = {
        **deepcopy(FIXTURE["predictions"][ticker]),
        "ticker": ticker, "as_of": context["as_of"],
        "target_date": context["target"], "horizon_trading_days": 21,
        "analysis_mode": context["analysis_mode"], "benchmark": context["benchmark"],
        "schema_version": context["schema_version"], "run_id": run_id,
        "model_version": context["model_version"],
        "feature_version": context["feature_version"],
        "dataset_version": context["dataset_version"],
        "action_policy_version": "action_policy_v1",
        "universe_manifest_hash": universe_manifest_hash,
        "model_variant": model_variant,
        "feature_materialized_at": feature_materialized_at,
        "evidence_ids": [f"market:{ticker.lower()}"],
        "missing": False, "degraded": False,
    }
    market = feature_slot("market", context, run_id)
    market["version"] = context["feature_version"]
    market["values"]["feature_materialized_at"] = feature_materialized_at
    market["values"]["quant_prediction"] = prediction
    return {
        "task_query": f"Analyze {ticker}",
        "manager_directives": {"ticker": ticker, "run_id": run_id},
        "forecast_context": context,
        "research_features": feature_slot("research", context, run_id),
        "financial_features": feature_slot("financial", context, run_id),
        "sentiment_features": feature_slot("sentiment", context, run_id),
        "market_features": market,
    }


def queued_events(session_id: str) -> list[dict]:
    queue = get_queue(session_id)
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    close_session(session_id)
    return events


def assert_no_private_sentinel(value) -> None:
    assert "PRIVATE_SENTINEL" not in json.dumps(value, sort_keys=True)


def test_complete_state_preserves_quant_and_schema():
    state = complete_state()
    result = assemble_monthly_forecast(state).to_dict()
    quant = state["market_features"]["values"]["quant_prediction"]
    assert result["status"] == "READY"
    for key in (
        "probability_positive", "expected_return_21d", "expected_excess_return",
        "confidence_interval", "confidence", "action", "universe_manifest_hash",
        "model_variant", "feature_materialized_at",
    ):
        assert result[key] == quant[key]
    assert {
        "status", "ticker", "as_of", "target_date", "horizon_trading_days",
        "analysis_mode", "benchmark", "run_id", "action", "probability_positive",
        "expected_return_21d", "expected_excess_return", "confidence_interval",
        "confidence", "evidence_ids", "warnings", "missing", "degraded",
        "graph_coverage", "versions", "narrative", "implementation",
        "universe_manifest_hash", "model_variant", "feature_materialized_at",
    } <= result.keys()
    for key in ("universe_manifest_hash", "model_variant", "feature_materialized_at"):
        assert result["versions"][key] == quant[key]


def test_authoritative_quant_provenance_survives_exactly_without_placeholders():
    state = complete_state()
    prediction = state["market_features"]["values"]["quant_prediction"]
    expected = {
        "universe_manifest_hash": "sha256:manifest-9b7f3c2a",
        "model_variant": "full_model_v1-no_graph",
        "feature_materialized_at": "2026-08-31T20:17:43Z",
    }
    prediction.update(expected)
    state["market_features"]["values"]["feature_materialized_at"] = expected[
        "feature_materialized_at"
    ]
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == "READY"
    assert {key: result[key] for key in expected} == expected
    assert {key: result["versions"][key] for key in expected} == expected


@pytest.mark.parametrize(
    "field",
    ("universe_manifest_hash", "model_variant", "feature_materialized_at"),
)
def test_missing_required_quant_provenance_fails_closed(field):
    state = complete_state()
    state["market_features"]["values"]["quant_prediction"].pop(field)
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == result["action"] == "INSUFFICIENT_DATA"
    assert result["probability_positive"] is None
    assert "QUANT_PREDICTION_REJECTED" in result["warnings"]
    assert {key for key in (
        "universe_manifest_hash", "model_variant", "feature_materialized_at",
    )} <= result.keys()


def test_tampered_feature_materialization_provenance_fails_closed():
    state = complete_state()
    state["market_features"]["values"]["quant_prediction"][
        "feature_materialized_at"
    ] = "2026-08-31T20:17:44Z"
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == result["action"] == "INSUFFICIENT_DATA"
    assert result["confidence"] is None
    assert "QUANT_PREDICTION_REJECTED" in result["warnings"]


def test_incomplete_state_is_insufficient_without_confidence():
    state = complete_state()
    state.pop("market_features")
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == result["action"] == "INSUFFICIENT_DATA"
    assert result["probability_positive"] is None
    assert result["confidence"] is None


def test_optional_degradation_keeps_quant_but_warns():
    state = complete_state()
    state["sentiment_features"]["degraded"] = True
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == "DEGRADED"
    assert result["action"] == "BUY"
    assert "SENTIMENT_FEATURES_DEGRADED" in result["warnings"]


def test_future_evidence_is_rejected_fail_closed():
    state = complete_state()
    state["research_features"]["values"]["timestamps"] = ["2026-09-01T00:00:00Z"]
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["probability_positive"] is None
    assert "RESEARCH_FEATURES_FUTURE_EVIDENCE_REJECTED" in result["warnings"]


def test_changed_quant_action_is_rejected_fail_closed():
    state = complete_state()
    state["market_features"]["values"]["quant_prediction"]["action"] = "HOLD"
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["confidence"] is None
    assert "QUANT_PREDICTION_REJECTED" in result["warnings"]


@pytest.mark.asyncio
async def test_llm_cannot_change_numeric_prediction_or_action():
    llm = AsyncMock()
    llm.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="SELL; probability=0.01; return=-99%")]
    )
    agent = MonthlyForecastAgent(llm_client=llm)
    result = await agent.run(complete_state())
    assert result["monthly_forecast"]["action"] == "BUY"
    assert result["monthly_forecast"]["probability_positive"] == 0.61
    assert "SELL" not in result["monthly_forecast"]["narrative"]
    assert "action BUY" in result["monthly_forecast"]["narrative"]


class Specialist:
    def __init__(self, kind: str):
        self.kind = kind

    async def run(self, state):
        await asyncio.sleep(0)
        context = state["forecast_context"]
        run_id = state["manager_directives"]["run_id"]
        return {**state, f"{self.kind}_features": feature_slot(self.kind, context, run_id)}


class MarketFeatures:
    def __init__(self, context, run_id):
        self.context, self.run_id = context, run_id

    def to_state_slot(self):
        slot = feature_slot("market", self.context, self.run_id)
        slot["version"] = self.context["feature_version"]
        slot["values"]["feature_materialized_at"] = self.context["as_of"]
        return slot


class QuantService:
    def build_features(self, *, context, market, **_):
        assert market["ticker"] == context.ticker
        return MarketFeatures(context.to_dict(), market["run_id"])

    def forecast(self, *, context, market_features, run_id):
        base = FIXTURE["predictions"][context.ticker]
        return {
            **deepcopy(base), "ticker": context.ticker,
            "as_of": context.to_dict()["as_of"], "target_date": context.to_dict()["target"],
            "horizon_trading_days": 21, "analysis_mode": context.analysis_mode.value,
            "benchmark": context.benchmark, "schema_version": context.schema_version,
            "run_id": run_id, "model_version": context.model_version,
            "feature_version": context.feature_version, "dataset_version": context.dataset_version,
            "action_policy_version": "action_policy_v1",
            "universe_manifest_hash": f"sha256:universe-{context.ticker.lower()}-2026q3",
            "model_variant": f"full_model_v1-{context.ticker.lower()}",
            "feature_materialized_at": context.to_dict()["as_of"],
            "evidence_ids": [f"market:{context.ticker.lower()}"],
            "missing": False, "degraded": False,
        }


def manager() -> ManagerAgent:
    memory = MagicMock()
    return ManagerAgent(
        Specialist("research"), Specialist("financial"), Specialist("sentiment"), memory,
        llm_client=AsyncMock(), monthly_forecast_agent=MonthlyForecastAgent(),
        market_forecast_service=QuantService(),
        market_data_provider=lambda context: {"ticker": context.ticker, "run_id": f"run-{context.ticker.lower()}"},
    )


@pytest.mark.asyncio
async def test_manager_node_order_and_typed_join():
    result = await manager().run(
        "Analyze MSFT", {"ticker": "MSFT", "run_id": "run-msft", "forecast_context": FIXTURE["contexts"]["MSFT"]},
    )
    assert result["monthly_node_order"] == [
        "context", "research", "financial", "sentiment", "controlled_join",
        "market_features", "monthly_forecast", "canonical_synthesis", "publication",
    ]
    assert result["monthly_forecast"]["status"] == "READY"
    for field in ("universe_manifest_hash", "model_variant", "feature_materialized_at"):
        assert result["canonical_synthesis"]["forecast"]["versions"][field] == result[
            "monthly_forecast"
        ][field]
    assert result["canonical_synthesis"]["dcf_semantics"] == "LONG_TERM_VALUATION_FEATURE_NOT_21_DAY_FORECAST"


@pytest.mark.asyncio
async def test_two_concurrent_manager_runs_have_no_cross_talk():
    subject = manager()
    msft, nvda = await asyncio.gather(
        subject.run("Analyze MSFT", {"ticker": "MSFT", "run_id": "run-msft", "forecast_context": FIXTURE["contexts"]["MSFT"]}, client_session_id="stage08-concurrent-msft"),
        subject.run("Analyze NVDA", {"ticker": "NVDA", "run_id": "run-nvda", "forecast_context": FIXTURE["contexts"]["NVDA"]}, client_session_id="stage08-concurrent-nvda"),
    )
    assert (msft["monthly_forecast"]["ticker"], msft["monthly_forecast"]["action"]) == ("MSFT", "BUY")
    assert (nvda["monthly_forecast"]["ticker"], nvda["monthly_forecast"]["action"]) == ("NVDA", "AVOID")
    assert msft["forecast_context"]["as_of"] != nvda["forecast_context"]["as_of"]
    assert msft["monthly_forecast"]["run_id"] != nvda["monthly_forecast"]["run_id"]
    msft_events = queued_events("stage08-concurrent-msft")
    nvda_events = queued_events("stage08-concurrent-nvda")
    assert msft_events and nvda_events
    assert {event["run_id"] for event in msft_events} == {"run-msft"}
    assert {event["run_id"] for event in nvda_events} == {"run-nvda"}
    assert {event["detail"]["ticker"] for event in msft_events} == {"MSFT"}
    assert {event["detail"]["ticker"] for event in nvda_events} == {"NVDA"}


def test_prompt_rename_and_constraints_are_atomic():
    prompt_dir = Path(__file__).parents[2] / "agents" / "prompts"
    assert not (prompt_dir / "promt_manager_agent.py").exists()
    assert (prompt_dir / "prompt_manager_agent.py").exists()
    assert "21-trading-session" in MONTHLY_MANAGER_SYSTEM_PROMPT
    assert "never" in MONTHLY_FORECAST_SYSTEM_PROMPT.lower()


def test_manager_trace_processors_allowlist_identity_and_redact_private_state():
    context = FIXTURE["contexts"]["MSFT"]
    safe_input = _manager_trace_inputs({
        "task_query": "PRIVATE_SENTINEL query",
        "manager_directives": {
            "forecast_context": context, "run_id": "run-msft",
            "secret": "PRIVATE_SENTINEL secret", "raw_payload": {"x": "PRIVATE_SENTINEL"},
        },
        "user_preferences": {"token": "PRIVATE_SENTINEL"},
        "client_session_id": "PRIVATE_SENTINEL-session",
    })
    unsafe_output = complete_state()
    unsafe_output.update({
        "task_query": "PRIVATE_SENTINEL query", "final_report": "PRIVATE_SENTINEL report",
        "raw_evidence": ["PRIVATE_SENTINEL evidence"],
        "monthly_forecast": {
            **assemble_monthly_forecast(complete_state()).to_dict(),
            "narrative": "PRIVATE_SENTINEL narrative",
        },
    })
    safe_output = _manager_trace_outputs(unsafe_output)
    assert_no_private_sentinel(safe_input)
    assert_no_private_sentinel(safe_output)
    for processed in (safe_input, safe_output):
        metadata = processed["metadata"]
        assert metadata["run_id"] == "run-msft"
        assert metadata["ticker"] == "MSFT"
        assert metadata["as_of"] == context["as_of"]
        assert metadata["target"] == context["target"]
        assert metadata["horizon"] == 21
        assert metadata["analysis_mode"] == "LIVE"
        assert metadata["dataset_version"] == "dataset-v1"
        assert metadata["feature_version"] == "monthly-market-features-v1"
        assert metadata["model_version"] == "model-v1"
        assert metadata["schema_version"] == "forecast-v1"
        assert (metadata["stage"], metadata["stage_revision"], metadata["attempt"], metadata["agent"]) == ("08", 1, 2, "integration")
        assert metadata["implementation_version"] == "stage08-r1-a2"
    serialized = json.dumps(safe_output, sort_keys=True)
    for forbidden in ("final_report", "narrative", "evidence_ids", "probability_positive", "expected_return_21d"):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_monthly_success_lifecycle_is_measured_identity_bound_and_payload_free():
    session_id = "stage08-success-events"
    result = await manager().run(
        "PRIVATE_SENTINEL query",
        {"ticker": "MSFT", "run_id": "run-msft", "forecast_context": FIXTURE["contexts"]["MSFT"]},
        client_session_id=session_id,
    )
    events = queued_events(session_id)
    assert [event["type"] for event in events] == [
        "forecast_context", "feature_start", "feature_complete", "feature_start",
        "feature_complete", "monthly_forecast_start", "monthly_forecast_complete",
    ]
    assert result["monthly_forecast"]["probability_positive"] == 0.61
    for event in events:
        assert event["run_id"] == "run-msft"
        detail = event["detail"]
        assert (detail["stage"], detail["stage_revision"], detail["attempt"], detail["agent"]) == ("08", 1, 2, "integration")
        assert detail["ticker"] == "MSFT"
        assert detail["latency_ms"] >= 0
        assert_no_private_sentinel(event)
        serialized = json.dumps(event, sort_keys=True)
        for forbidden in ("probability_positive", "expected_return_21d", "expected_excess_return", "evidence_ids", "narrative", "task_query"):
            assert forbidden not in serialized

    assert STAGE08_IMPLEMENTATION_IDENTITY == {
        "stage": "08",
        "stage_revision": 1,
        "attempt": 2,
        "agent": "integration",
        "implementation_version": "stage08-r1-a2",
    }


class FailingQuantService(QuantService):
    def forecast(self, **_):
        raise RuntimeError("PRIVATE_SENTINEL exception text")


@pytest.mark.asyncio
async def test_monthly_error_event_is_stable_redacted_and_exception_is_reraised():
    subject = manager()
    subject._market_forecast_service = FailingQuantService()
    session_id = "stage08-error-events"
    with pytest.raises(RuntimeError, match="PRIVATE_SENTINEL exception text"):
        await subject.run(
            "PRIVATE_SENTINEL query",
            {"ticker": "MSFT", "run_id": "run-msft", "forecast_context": FIXTURE["contexts"]["MSFT"]},
            client_session_id=session_id,
        )
    events = queued_events(session_id)
    assert events[-1]["type"] == "forecast_error"
    assert events[-1]["message"] == "STAGE08_MONTHLY_FAIL_CLOSED"
    assert events[-1]["run_id"] == "run-msft"
    assert events[-1]["detail"]["degraded_flags"] == ["monthly_fail_closed"]
    assert events[-1]["detail"]["latency_ms"] >= 0
    assert_no_private_sentinel(events[-1])
