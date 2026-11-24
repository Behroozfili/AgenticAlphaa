"""Stage 10B asynchronous backtest job route tests."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from api.routes import backtest


RUN_ID = "94e6e7f4-43a2-42aa-91cb-e73b92f81a7d"


def report_fixture():
    return {
        "manifest": {
            "run_id": RUN_ID,
            "report_hash": "a" * 64,
            "backtest_version": "walk-forward-v1",
            "universe_version": "u1",
            "universe_manifest_hash": "b" * 64,
            "cutoff_schedule": {"version": "c1"},
            "horizon_trading_days": 21,
            "fold_method": "expanding_window_observable_outcomes_only",
            "minimum_training_cutoffs": 36,
            "fold_count": 2,
            "transaction_cost_bps_one_way": 5,
            "gross_results_reported": True,
            "dataset_version": "d1",
            "dataset_manifest_hash": "c" * 64,
            "eligible_rows_sha256": "d" * 64,
            "feature_version": "f1",
            "model_version": "m1",
            "schema_version": "s1",
            "model_variants": ["FULL_MODEL"],
            "action_policy_version": "a1",
            "action_thresholds": {"buy": 0.55, "avoid": 0.45},
        },
        "folds": [{"fold_id": "one"}, {"fold_id": "two"}],
        "records": [{"id": 1}, {"id": 2}, {"id": 3}],
        "metrics": {"FULL_MODEL": {"mae": 0.1}},
        "leakage_audit": {"future_record_count": 0},
    }


def request_with_state(database=None, telemetry_sink=None):
    request = MagicMock()
    request.app.state = SimpleNamespace()
    if database is not None:
        request.app.state.supabase = database
    if telemetry_sink is not None:
        request.app.state.telemetry_sink = telemetry_sink
    return request


def test_reproducibility_hash_is_input_only_and_run_id_independent():
    first = report_fixture()
    second = deepcopy(first)
    second["manifest"]["run_id"] = "22222222-2222-4222-8222-222222222222"
    second["manifest"]["report_hash"] = "e" * 64
    second["metrics"] = {"OUTPUT_ONLY": {"mae": 999}}
    second["records"] = [{"output": "changed"}]
    second["leakage_audit"] = {"output": "changed"}
    assert backtest._reproducibility_hash(first) == backtest._reproducibility_hash(second)


def test_reproducibility_hash_changes_for_every_identity_family(monkeypatch):
    report = report_fixture()
    baseline = backtest._reproducibility_hash(report)
    mutations = [
        ("universe_version", "u2"),
        ("universe_manifest_hash", "e" * 64),
        ("cutoff_schedule", {"version": "c2"}),
        ("transaction_cost_bps_one_way", 20),
        ("dataset_version", "d2"),
        ("feature_version", "f2"),
        ("action_policy_version", "a2"),
        ("model_variants", ["MARKET_ONLY"]),
    ]
    for field, value in mutations:
        changed = deepcopy(report)
        changed["manifest"][field] = value
        assert backtest._reproducibility_hash(changed) != baseline
    changed = deepcopy(report)
    changed["folds"][0]["fold_id"] = "different-fold"
    assert backtest._reproducibility_hash(changed) != baseline
    for constant in (
        "BACKTEST_MODEL_VERSION", "BACKTEST_CODE_VERSION", "BACKTEST_CONFIG_VERSION",
    ):
        with monkeypatch.context() as scoped:
            scoped.setattr(backtest, constant, f"different-{constant.lower()}")
            assert backtest._reproducibility_hash(report) != baseline


@pytest.mark.asyncio
async def test_create_is_nonblocking_and_same_run_id_is_idempotent(monkeypatch):
    def delayed_report(*_args, **_kwargs):
        return report_fixture()

    monkeypatch.setattr(backtest, "build_report", delayed_report)
    request = request_with_state()
    payload = backtest.BacktestRequest(run_id=RUN_ID)
    first = await backtest.create_backtest(payload, request)
    second = await backtest.create_backtest(payload, request)
    assert first.run_id == RUN_ID
    assert first.status == "PLANNED"
    assert second.run_id == first.run_id
    assert len(request.app.state.stage10b_backtest_jobs) == 1
    assert len(request.app.state.stage10b_backtest_tasks) == 1
    await asyncio.gather(*tuple(request.app.state.stage10b_backtest_tasks))


@pytest.mark.asyncio
async def test_completed_job_exposes_compact_artifact_and_persists(monkeypatch):
    monkeypatch.setattr(backtest, "build_report", lambda *_args, **_kwargs: report_fixture())
    table = MagicMock()
    database = MagicMock()
    database.table.return_value = table
    events = []
    request = request_with_state(database, events.append)
    payload = backtest.BacktestRequest(run_id=RUN_ID)
    await backtest.create_backtest(payload, request)
    await asyncio.gather(*tuple(request.app.state.stage10b_backtest_tasks))
    status = await backtest.get_backtest(RUN_ID, request)
    artifact = await backtest.get_backtest_artifact(RUN_ID, request)
    assert status.status == "COMPLETED"
    assert artifact["fold_count"] == 2
    assert artifact["record_count"] == 3
    assert "records" not in artifact
    database.table.assert_called_with("backtest_runs")
    persisted = table.upsert.call_args.args[0]
    assert persisted["status"] == "COMPLETED"
    assert persisted["config_version"] == "stage10b-r1-a2"
    assert persisted["artifact_refs"]["report_hash"] == "a" * 64
    assert len(persisted["fold_definition"]["definition_sha256"]) == 64
    assert table.upsert.call_args.kwargs == {
        "on_conflict": "run_id", "ignore_duplicates": True,
    }
    assert [event["event"] for event in events] == [
        "stage10b.backtest.started", "stage10b.backtest.completed",
    ]
    assert events[-1]["dataset_version"] == "d1"
    assert events[-1]["model_version"] == "m1"
    assert events[-1]["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_failed_job_returns_only_stable_error_code(monkeypatch):
    monkeypatch.setattr(
        backtest,
        "build_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")),
    )
    events = []
    request = request_with_state(telemetry_sink=events.append)
    await backtest.create_backtest(backtest.BacktestRequest(run_id=RUN_ID), request)
    await asyncio.gather(*tuple(request.app.state.stage10b_backtest_tasks))
    job = await backtest.get_backtest(RUN_ID, request)
    assert job.status == "FAILED"
    assert job.error_code == "BACKTEST_EXECUTION_FAILED"
    assert "PRIVATE_SENTINEL" not in job.model_dump_json()
    assert events[-1]["event"] == "stage10b.backtest.failed"
    assert events[-1]["error_code"] == "BACKTEST_EXECUTION_FAILED"
    assert events[-1]["latency_ms"] >= 0
    assert "PRIVATE_SENTINEL" not in repr(events)


@pytest.mark.asyncio
async def test_raising_backtest_telemetry_sink_is_nonfatal(monkeypatch):
    monkeypatch.setattr(backtest, "build_report", lambda *_args, **_kwargs: report_fixture())

    def failing_sink(_event):
        raise RuntimeError("PRIVATE_SINK_EXCEPTION")

    request = request_with_state(telemetry_sink=failing_sink)
    await backtest.create_backtest(backtest.BacktestRequest(run_id=RUN_ID), request)
    await asyncio.gather(*tuple(request.app.state.stage10b_backtest_tasks))
    job = await backtest.get_backtest(RUN_ID, request)
    assert job.status == "COMPLETED"


@pytest.mark.asyncio
async def test_missing_and_not_ready_artifacts_fail_closed():
    request = request_with_state()
    with pytest.raises(Exception) as missing:
        await backtest.get_backtest(RUN_ID, request)
    assert missing.value.status_code == 404
    jobs, _ = backtest._job_store(request.app)
    jobs[RUN_ID] = {
        "run_id": RUN_ID, "status": "RUNNING", "artifact": None,
    }
    with pytest.raises(Exception) as pending:
        await backtest.get_backtest_artifact(RUN_ID, request)
    assert pending.value.status_code == 409
