"""Stage 10B route, persistence and request-isolation acceptance tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.dependencies import get_manager_memory
from api.routes.analyze import AnalyzeRequest, _forecast_row, _outcome_row, router


MONTHLY_PAYLOAD = {
    "query": "Generate a point in time monthly forecast",
    "ticker": "msft",
    "as_of": "2024-01-31T21:00:00Z",
    "horizon": 21,
    "mode": "historical_backtest",
    "run_id": "b41d7dc2-d608-4c08-b99b-87f90f064829",
}


def monthly_result(run_id=MONTHLY_PAYLOAD["run_id"], ticker="MSFT"):
    return {
        "market_features": {
            "version": "monthly-features-v1", "evidence_ids": ["ev-1"],
            "values": {"ticker": ticker, "as_of": "2024-01-31T21:00:00Z",
                "dataset_version": "historical-dataset-v1", "schema_version": "forecast-schema-v2",
                "feature_materialized_at": "2024-01-31T21:00:00Z", "momentum_21d": 0.04},
        },
        "execution_status": "completed",
        "publication_status": "PUBLISHED",
        "final_publication_validation": {"passed": True},
        "monthly_forecast": {
            "status": "READY", "ticker": ticker,
            "as_of": "2024-01-31T21:00:00Z", "target_date": "2024-03-01",
            "horizon_trading_days": 21, "analysis_mode": "HISTORICAL_BACKTEST",
            "benchmark": "SPY", "run_id": run_id,
            "universe_manifest_hash": "a" * 64, "model_variant": "FULL_MODEL",
            "feature_materialized_at": "2024-01-31T21:00:00Z",
            "action": "BUY", "probability_positive": 0.7,
            "expected_return_21d": 0.05, "expected_excess_return": 0.03,
            "confidence_interval": [0.01, 0.08], "confidence": 0.8,
            "evidence_ids": ["ev-1"], "warnings": ["compact warning"],
            "missing": False, "degraded": False, "graph_coverage": True,
            "versions": {
                "schema_version": "forecast-schema-v2",
                "dataset_version": "historical-dataset-v1",
                "feature_version": "monthly-features-v1",
                "model_version": "monthly-model-v1",
                "action_policy_version": "action_policy_v1",
            },
            "narrative": "must not be duplicated in the compact forecast",
        },
    }


def make_client(manager, database=None, telemetry_sink=None):
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")
    application.state.manager_agent = manager
    if database is None:
        database = MagicMock()
        database.rpc.return_value.execute.return_value.data = {"status": "SAVED", "run_id": MONTHLY_PAYLOAD["run_id"]}
    application.state.supabase = database
    if telemetry_sink is not None:
        application.state.telemetry_sink = telemetry_sink
    memory = MagicMock()
    memory.long.get_all_preferences.return_value = {}
    application.dependency_overrides[get_manager_memory] = lambda: memory
    return TestClient(application, raise_server_exceptions=False)


def test_monthly_request_builds_canonical_context_and_compact_response():
    manager = AsyncMock()
    manager.run.return_value = monthly_result()
    response = make_client(manager).post("/api/v1/analyze", json=MONTHLY_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["forecast_context"]["ticker"] == "MSFT"
    assert body["forecast_context"]["analysis_mode"] == "HISTORICAL_BACKTEST"
    assert body["forecast_context"]["target"] == "2024-03-01"
    assert body["forecast_run_id"] == MONTHLY_PAYLOAD["run_id"]
    assert body["monthly_forecast"]["probability_positive"] == 0.7
    assert body["monthly_forecast"]["narrative"] == "must not be duplicated in the compact forecast"
    directives = manager.run.await_args.kwargs["manager_directives"]
    assert directives["forecast_context"] == body["forecast_context"]
    assert directives["run_id"] == MONTHLY_PAYLOAD["run_id"]


def test_invalid_month_end_context_fails_with_stable_422():
    payload = {**MONTHLY_PAYLOAD, "as_of": "2024-01-30T21:00:00Z"}
    response = make_client(AsyncMock()).post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == {"code": "INVALID_FORECAST_CONTEXT"}


def test_horizon_other_than_21_is_rejected():
    with pytest.raises(ValueError):
        AnalyzeRequest(**{**MONTHLY_PAYLOAD, "horizon": 20})


def test_monthly_persistence_uses_idempotent_versioned_rows():
    manager = AsyncMock()
    manager.run.return_value = monthly_result()
    from postgrest.exceptions import APIError
    database = MagicMock()
    acknowledgement = MagicMock(data={"status": "SAVED", "run_id": MONTHLY_PAYLOAD["run_id"]})
    database.rpc.return_value.execute.side_effect = [ConnectionError("temporary transport"), acknowledgement]
    client = make_client(manager, database)
    response = client.post("/api/v1/analyze", json=MONTHLY_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["persistence_status"] == "SAVED"
    name, bundle = database.rpc.call_args.args
    assert name == "persist_monthly_bundle"
    database.table.assert_not_called()  # No independent partial writes.
    assert database.rpc.return_value.execute.call_count == 2
    analysis_row, forecast_row = bundle["p_analysis"], bundle["p_forecast"]
    assert analysis_row["as_of"] == MONTHLY_PAYLOAD["as_of"]
    assert analysis_row["evidence_ids"] == ["ev-1"]
    assert forecast_row["feature_snapshot_id"] == bundle["p_snapshot"]["snapshot_id"]
    assert forecast_row["predicted_ticker_return"] == 0.05
    assert forecast_row["predicted_benchmark_return"] == pytest.approx(0.02)
    assert "narrative" not in str(forecast_row)
    first_hash = bundle["p_request_hash"]
    database.rpc.return_value.execute.side_effect = APIError({"code": "PAA01", "message": "FORECAST_RUN_CONFLICT", "details": None, "hint": None})
    conflicting = client.post("/api/v1/analyze", json={**MONTHLY_PAYLOAD, "query": "Different research question"})
    assert conflicting.status_code == 409
    assert conflicting.json()["detail"] == {"code": "FORECAST_RUN_CONFLICT"}
    assert database.rpc.call_args.args[1]["p_request_hash"] != first_hash
    assert database.rpc.return_value.execute.call_count == 3  # Conflicts are not retried.
    database.table.assert_not_called()



def test_monthly_forecast_row_fails_closed_without_required_provenance():
    forecast = monthly_result()["monthly_forecast"]
    assert _forecast_row(forecast) is not None
    for field in ("universe_manifest_hash", "feature_materialized_at"):
        incomplete = {**forecast, field: None}
        assert _forecast_row(incomplete) is None


def test_monthly_lifecycle_is_local_compact_measured_and_persistence_aware():
    manager = AsyncMock()
    manager.run.return_value = monthly_result()
    events = []
    response = make_client(manager, telemetry_sink=events.append).post(
        "/api/v1/analyze", json=MONTHLY_PAYLOAD,
    )
    assert response.status_code == 200
    assert [event["event"] for event in events] == [
        "stage10b.analysis.started",
        "stage10b.persistence.completed",
        "stage10b.analysis.completed",
    ]
    terminal = events[-1]
    assert terminal["run_id"] == MONTHLY_PAYLOAD["run_id"]
    assert terminal["ticker"] == "MSFT"
    assert terminal["as_of"] == MONTHLY_PAYLOAD["as_of"]
    assert terminal["evidence_count"] == 1
    assert terminal["latency_ms"] >= 0
    assert not ({"query", "user_id", "final_report", "raw_evidence"} & set(terminal))
    assert "must not be duplicated" not in repr(events)


def test_persistence_failure_event_is_stable_and_nonfatal():
    manager = AsyncMock()
    manager.run.return_value = monthly_result()
    database = MagicMock()
    database.rpc.side_effect = RuntimeError("PRIVATE_DATABASE_EXCEPTION")
    events = []
    response = make_client(
        manager, database=database, telemetry_sink=events.append,
    ).post("/api/v1/analyze", json=MONTHLY_PAYLOAD)
    assert response.status_code == 200
    persistence = next(
        event for event in events
        if event["event"] == "stage10b.persistence.failed"
    )
    assert persistence["error_code"] == "ANALYSIS_PERSIST_FAILED"
    assert "PRIVATE_DATABASE_EXCEPTION" not in repr(events)


def test_outcome_persistence_fails_closed_until_observable():
    context = {
        "ticker": "MSFT", "as_of": "2024-01-31T21:00:00Z",
        "target": "2024-03-01", "horizon_trading_days": 21,
        "benchmark": "SPY", "dataset_version": "historical-dataset-v1",
    }
    outcome = {
        "ticker": "MSFT", "as_of": context["as_of"], "target_date": context["target"],
        "horizon_trading_days": 21, "benchmark": "SPY", "price_policy_version": "v1",
        "ticker_start_adjusted_close": 100, "ticker_end_adjusted_close": 105,
        "benchmark_start_adjusted_close": 100, "benchmark_end_adjusted_close": 102,
        "ticker_total_return": 0.05, "benchmark_total_return": 0.02,
        "excess_return": 0.03, "source_snapshot_ids": ["s1"],
        "observable_at": "2999-01-01T00:00:00Z",
    }
    assert _outcome_row({"forecast_outcome": outcome}, context) is None
    outcome["observable_at"] = "2024-03-02T00:00:00Z"
    assert _outcome_row({"forecast_outcome": outcome}, context)["excess_return"] == 0.03


@pytest.mark.asyncio
async def test_concurrent_requests_keep_ticker_as_of_and_run_id_isolated():
    from api.routes.analyze import analyze

    seen = []

    class Manager:
        async def run(self, **kwargs):
            directives = kwargs["manager_directives"]
            seen.append(directives)
            await __import__("asyncio").sleep(0)
            context = directives["forecast_context"]
            return monthly_result(directives["run_id"], context["ticker"])

    request = MagicMock()
    request.app.state.manager_agent = Manager()
    request.app.state.supabase = MagicMock()
    memory_a = MagicMock()
    memory_b = MagicMock()
    memory_a.long.get_all_preferences.return_value = {"request": "a"}
    memory_b.long.get_all_preferences.return_value = {"request": "b"}
    payload_a = AnalyzeRequest(**MONTHLY_PAYLOAD)
    payload_b = AnalyzeRequest(**{
        **MONTHLY_PAYLOAD,
        "ticker": "AAPL",
        "as_of": "2024-02-29T21:00:00Z",
        "run_id": "8f0590e2-35cf-4b3e-a57b-9a8bc1656175",
    })
    first, second = await __import__("asyncio").gather(
        analyze(payload_a, request, memory_a),
        analyze(payload_b, request, memory_b),
    )
    assert (first.forecast_context["ticker"], first.forecast_run_id) == (
        "MSFT", MONTHLY_PAYLOAD["run_id"],
    )
    assert (second.forecast_context["ticker"], second.forecast_run_id) == (
        "AAPL", "8f0590e2-35cf-4b3e-a57b-9a8bc1656175",
    )
    assert {item["ticker"] for item in seen} == {"MSFT", "AAPL"}
