import dataclasses
from pathlib import Path
from types import MappingProxyType

from evaluation.monthly_backtest import BacktestReport, run_monthly_backtest
from evaluation.point_in_time_dataset import load_point_in_time_fixture
from evaluation.temporal_leakage import audit_backtest


FIXTURE = Path("tests/fixtures/stage9_walk_forward.json")


def test_complete_backtest_has_zero_future_records():
    dataset = load_point_in_time_fixture(FIXTURE)
    report = run_monthly_backtest(dataset)
    assert report.leakage_audit["status"] == "PASS"
    assert report.leakage_audit["future_record_count"] == 0
    assert report.leakage_audit["checked_forecasts"] == len(report.records)


def test_audit_detects_future_training_outcome():
    dataset = load_point_in_time_fixture(FIXTURE)
    report = run_monthly_backtest(dataset)
    record = dict(report.records[0])
    record["training_max_outcome_materialized_at"] = record["outcome_materialized_at"]
    tampered = BacktestReport(
        manifest=report.manifest,
        folds=report.folds,
        records=(MappingProxyType(record),) + report.records[1:],
        metrics=report.metrics,
        leakage_audit=MappingProxyType({}),
    )
    events = []
    audit = audit_backtest(dataset, tampered, event_sink=lambda event: events.append(dict(event)))
    assert audit.future_record_count == 1
    assert audit.findings[0].code == "FUTURE_TRAINING_OUTCOME"
    assert events[0]["boundary"] == "leakage_audit"
    assert events[0]["outcome"] == "fail_closed"
    assert events[0]["error_code"] == "STAGE09_FUTURE_RECORDS_DETECTED"
    assert not ({"findings", "records", "labels", "predictions"} & set(events[0]))


def test_audit_detects_feature_materialized_after_as_of():
    dataset = load_point_in_time_fixture(FIXTURE)
    report = run_monthly_backtest(dataset)
    first = dataset.rows[0]
    changed_snapshot = dataclasses.replace(first.snapshot, feature_materialized_at=first.outcome_materialized_at)
    changed_dataset = dataclasses.replace(dataset, rows=(dataclasses.replace(first, snapshot=changed_snapshot),) + dataset.rows[1:])
    audit = audit_backtest(changed_dataset, report)
    assert any(item.code == "FEATURE_AFTER_AS_OF" for item in audit.findings)
