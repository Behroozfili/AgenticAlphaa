from pathlib import Path
import json
import math

import pytest

from core.forecast_contract import HORIZON_TRADING_DAYS, nth_trading_session_after, parse_utc_datetime
from evaluation.point_in_time_dataset import (
    STAGE09_RUNTIME_IDENTITY, load_point_in_time_fixture, materialize_synthetic_fixture,
)


FIXTURE = Path("tests/fixtures/stage9_walk_forward.json")


def test_fixture_materializes_reproducibly_with_preliminary_manifest():
    first = load_point_in_time_fixture(FIXTURE)
    second = load_point_in_time_fixture(FIXTURE)
    assert len(first.rows) == 96
    assert first.manifest_hash == second.manifest_hash
    assert first.manifest["eligible_rows_sha256"] == second.manifest["eligible_rows_sha256"]
    assert first.manifest["scientific_report_level"] == "PRELIMINARY"
    assert first.manifest["ticker_count"] == 2
    assert first.manifest["monthly_cutoff_count"] == 48


def test_every_row_has_exact_target_and_forecast_before_outcome():
    dataset = load_point_in_time_fixture(FIXTURE)
    for row in dataset.rows:
        as_of = parse_utc_datetime(row.snapshot.as_of)
        assert row.snapshot.horizon_trading_days == HORIZON_TRADING_DAYS
        assert row.snapshot.target_date == nth_trading_session_after(as_of.date(), 21).isoformat()
        assert parse_utc_datetime(row.snapshot.feature_materialized_at) <= as_of
        assert as_of < parse_utc_datetime(row.outcome_materialized_at)


def test_fixture_contains_explicit_optional_missingness_not_zero_fill():
    dataset = load_point_in_time_fixture(FIXTURE)
    missing = [row for row in dataset.rows if row.snapshot.missing_indicators["sentiment_features__missing"]]
    assert missing
    assert all(row.snapshot.values["sentiment_aggregate_30"] is None for row in missing)


def test_fixture_telemetry_has_immutable_identity_finite_latency_and_no_payload():
    events = []
    load_point_in_time_fixture(FIXTURE, event_sink=lambda event: events.append(dict(event)))
    assert len(events) == 1
    event = events[0]
    assert {key: event[key] for key in STAGE09_RUNTIME_IDENTITY} == dict(STAGE09_RUNTIME_IDENTITY)
    assert event["boundary"] == "fixture_materialization"
    assert event["outcome"] == "success"
    assert math.isfinite(event["latency_ms"]) and event["latency_ms"] >= 0.0
    assert not ({"rows", "features", "labels", "predictions", "evidence_ids"} & set(event))


def test_fixture_failure_event_redacts_private_input_and_omits_exception_text():
    events = []
    with pytest.raises(TypeError, match="generation_spec"):
        materialize_synthetic_fixture({}, fixture_sha256="x", run_id="PRIVATE_SENTINEL-run",
                                      event_sink=lambda event: events.append(dict(event)))
    assert events[0]["outcome"] == "fail_closed"
    assert events[0]["error_code"] == "STAGE09_FIXTURE_REJECTED"
    rendered = repr(events)
    assert "PRIVATE_SENTINEL" not in rendered
    assert "generation_spec" not in rendered


def test_source_rows_never_stand_in_for_out_of_sample_forecasts():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tickers = [f"T{index:02d}" for index in range(10)]
    payload["generation_spec"].update({"tickers": tickers, "monthly_cutoffs": 60})
    payload["universe"]["constituents"] = [
        {
            "ticker": ticker,
            "effective_from": "2018-01-01",
            "effective_to": None,
            "benchmark": "SPY",
            "source_ref": "fixture:stage9-scientific-counterexample",
        }
        for ticker in tickers
    ]
    dataset = materialize_synthetic_fixture(payload, fixture_sha256="counterexample")
    assert dataset.manifest["eligible_row_count"] == 600
    assert dataset.manifest["ticker_count"] == 10
    assert dataset.manifest["monthly_cutoff_count"] == 60
    assert dataset.manifest["scientific_report_level"] == "PRELIMINARY"
