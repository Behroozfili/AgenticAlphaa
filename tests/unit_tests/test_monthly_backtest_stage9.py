from pathlib import Path
import math

from core.forecast_contract import parse_utc_datetime
from evaluation.monthly_backtest import MODEL_VARIANTS, build_expanding_folds, run_monthly_backtest
from evaluation.point_in_time_dataset import classify_scientific_report, load_point_in_time_fixture


FIXTURE = Path("tests/fixtures/stage9_walk_forward.json")


def test_expanding_folds_use_only_observable_past_outcomes():
    dataset = load_point_in_time_fixture(FIXTURE)
    folds = build_expanding_folds(dataset.rows)
    assert folds
    assert folds[0].training_cutoffs >= 36
    assert [fold.training_cutoffs for fold in folds] == sorted(fold.training_cutoffs for fold in folds)
    for fold in folds:
        forecast_at = parse_utc_datetime(fold.forecast_as_of)
        assert all(parse_utc_datetime(row.snapshot.as_of) < forecast_at for row in fold.training_rows)
        assert all(parse_utc_datetime(row.outcome_materialized_at) <= forecast_at for row in fold.training_rows)


def test_backtest_compares_identical_rows_and_all_required_variants():
    report = run_monthly_backtest(load_point_in_time_fixture(FIXTURE))
    assert tuple(report.manifest["model_variants"]) == MODEL_VARIANTS
    assert len(set(report.manifest["variant_row_identity_sha256"].values())) == 1
    expected = report.manifest["forecast_count_per_variant"]
    assert expected > 0
    for variant in MODEL_VARIANTS:
        rows = [row for row in report.records if row["model_variant"] == variant]
        assert len(rows) == expected
        assert report.metrics[variant]["coverage"]["fraction"] == 1.0


def test_report_is_reproducible_and_records_required_provenance():
    dataset = load_point_in_time_fixture(FIXTURE)
    first = run_monthly_backtest(dataset)
    second = run_monthly_backtest(dataset)
    assert first.manifest["report_hash"] == second.manifest["report_hash"]
    required = {
        "ticker", "as_of", "target_date", "horizon_trading_days", "analysis_mode",
        "universe_manifest_hash", "dataset_version", "feature_version", "model_version",
        "action_policy_version", "model_variant", "benchmark", "run_id", "evidence_ids",
        "missing", "degraded", "feature_materialized_at", "forecast_materialized_at",
        "outcome_materialized_at",
    }
    assert required <= set(first.records[0])
    assert first.manifest["scientific_report_level"] == "PRELIMINARY"
    assert first.manifest["transaction_cost_bps_one_way"] == 10.0
    assert first.manifest["gross_results_reported"] is True


def test_class_prior_baseline_is_always_hold_and_full_model_underperformance_is_not_hidden():
    report = run_monthly_backtest(load_point_in_time_fixture(FIXTURE))
    baseline = [row for row in report.records if row["model_variant"] == "class_prior_always_hold"]
    assert {row["action"] for row in baseline} == {"HOLD"}
    assert "full_model" in report.metrics
    assert "momentum_21d" in report.metrics


def test_telemetry_enabled_is_scientifically_identical_and_ordered():
    dataset = load_point_in_time_fixture(FIXTURE)
    disabled = run_monthly_backtest(dataset).to_dict()
    events = []
    enabled = run_monthly_backtest(dataset, event_sink=lambda event: events.append(dict(event))).to_dict()
    assert enabled == disabled
    assert enabled["manifest"]["report_hash"] == "213ad73328c244c495d0a465ed1bc7b02fb1ab2348a803e6ab807948f3aad779"
    assert events[0]["boundary"] == "run" and events[0]["outcome"] == "started"
    assert events[-1]["boundary"] == "run" and events[-1]["outcome"] == "degraded"
    assert {event["boundary"] for event in events} >= {
        "run", "fold_model_fit", "fold_prediction", "run_progress",
        "metric_aggregation", "leakage_audit",
    }
    assert all(event["stage"] == "09" and event["stage_revision"] == 1
               and event["attempt"] == 2 and event["agent"] == "quant"
               and event["implementation_version"] == "stage09-r1-a2" for event in events)
    assert all(math.isfinite(event["latency_ms"]) and event["latency_ms"] >= 0.0 for event in events)
    progress = [event for event in events if event["boundary"] == "run_progress"]
    assert [event["completed_folds"] for event in progress] == list(range(1, 12))
    forbidden = {"report", "records", "features", "labels", "predictions", "coefficients", "evidence_ids"}
    assert all(not (forbidden & set(event)) for event in events)


def test_scientific_oos_threshold_is_exactly_499_versus_500():
    assert classify_scientific_report(
        ticker_count=10, monthly_cutoff_count=60, eligible_oos_forecasts=499,
    ) == "PRELIMINARY"
    assert classify_scientific_report(
        ticker_count=10, monthly_cutoff_count=60, eligible_oos_forecasts=500,
    ) == "SCIENTIFIC"
    assert classify_scientific_report(
        ticker_count=9, monthly_cutoff_count=60, eligible_oos_forecasts=500,
    ) == "PRELIMINARY"
    assert classify_scientific_report(
        ticker_count=10, monthly_cutoff_count=59, eligible_oos_forecasts=500,
    ) == "PRELIMINARY"
