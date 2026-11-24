from datetime import date, datetime, timezone

import pytest

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError, VersionMismatchError
from core.forecast_contract import (
    AnalysisMode,
    FORWARD_EXCESS_RETURN_LABEL,
    ForecastContext,
    assert_not_after_as_of,
    assert_version_match,
    compute_forward_return_labels,
    is_us_regular_trading_session,
    last_trading_session_of_month,
    nth_trading_session_after,
    parse_utc_datetime,
)


def make_context(**overrides):
    values = {
        "ticker": "msft",
        "as_of": "2024-05-31T20:00:00Z",
        "schema_version": "schema-v1",
        "dataset_version": "dataset-v1",
        "feature_version": "features-v1",
        "model_version": "model-v1",
    }
    values.update(overrides)
    return ForecastContext(**values)


def test_aware_timestamp_is_normalized_to_utc():
    parsed = parse_utc_datetime("2024-05-31T16:00:00-04:00")
    assert parsed == datetime(2024, 5, 31, 20, tzinfo=timezone.utc)


def test_naive_timestamp_fails_closed():
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_utc_datetime("2024-05-31T16:00:00")


@pytest.mark.parametrize("closed", [
    date(2024, 3, 29),  # Good Friday
    date(2024, 6, 19),  # Juneteenth
    date(2024, 6, 22),  # weekend
    date(2018, 12, 5),  # presidential memorial closure
])
def test_regular_session_calendar_excludes_closures(closed):
    assert not is_us_regular_trading_session(closed)


def test_month_end_and_exact_21_session_target():
    assert last_trading_session_of_month(2024, 3) == date(2024, 3, 28)
    assert nth_trading_session_after(date(2024, 5, 31)) == date(2024, 7, 2)


def test_xnys_saturday_new_year_does_not_close_preceding_friday():
    assert is_us_regular_trading_session(date(2021, 12, 31))
    assert last_trading_session_of_month(2021, 12) == date(2021, 12, 31)


def test_carter_closure_changes_exact_2025_target():
    assert not is_us_regular_trading_session(date(2025, 1, 9))
    assert nth_trading_session_after(date(2024, 12, 31)) == date(2025, 2, 3)
    context = make_context(as_of="2024-12-31T21:00:00Z")
    assert context.target == date(2025, 2, 3)


def test_forecast_context_normalizes_and_derives_target():
    context = make_context(analysis_mode="historical_backtest")
    assert context.ticker == "MSFT"
    assert context.benchmark == "SPY"
    assert context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST
    assert context.target == date(2024, 7, 2)
    assert context.horizon_trading_days == 21


def test_context_rejects_non_month_end_cutoff():
    with pytest.raises(ValueError, match="last regular"):
        make_context(as_of="2024-05-30T20:00:00Z")


def test_context_requires_actual_close_and_handles_black_friday_early_close():
    with pytest.raises(ValueError, match="regular-session close"):
        make_context(as_of="2024-05-31T19:59:59Z")
    context = make_context(as_of="2024-11-29T18:00:00Z")
    assert context.as_of.hour == 18


def test_context_rejects_wrong_target_or_horizon():
    with pytest.raises(ValueError, match="21st"):
        make_context(target="2024-07-03")
    with pytest.raises(ValueError, match="exactly 21"):
        make_context(horizon_trading_days=20)


def test_context_serialization_round_trip_is_stable():
    context = make_context()
    payload = context.to_dict()
    assert payload["label"] == FORWARD_EXCESS_RETURN_LABEL
    assert ForecastContext.from_dict(payload) == context
    payload["calendar_version"] = "wrong"
    with pytest.raises(VersionMismatchError):
        ForecastContext.from_dict(payload)


def test_historical_round_trip_cannot_silently_degrade_to_live():
    context = make_context(analysis_mode=AnalysisMode.HISTORICAL_BACKTEST)
    restored = ForecastContext.from_dict(context.to_dict())
    assert restored.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST


@pytest.mark.parametrize("missing_field", [
    "ticker",
    "as_of",
    "target",
    "horizon_trading_days",
    "lookback_days",
    "benchmark",
    "analysis_mode",
    "schema_version",
    "dataset_version",
    "feature_version",
    "model_version",
    "calendar_version",
    "label",
])
def test_deserialization_rejects_every_missing_canonical_field(missing_field):
    payload = make_context(analysis_mode=AnalysisMode.HISTORICAL_BACKTEST).to_dict()
    payload.pop(missing_field)
    with pytest.raises(ValueError, match="missing required"):
        ForecastContext.from_dict(payload)


def test_future_and_missing_evidence_fail_closed():
    cutoff = "2024-05-31T20:00:00Z"
    assert assert_not_after_as_of("2024-05-31T19:59:59Z", cutoff).tzinfo is timezone.utc
    with pytest.raises(TemporalLeakageError):
        assert_not_after_as_of("2024-05-31T20:00:01Z", cutoff)
    with pytest.raises(DataUnavailableAsOfError):
        assert_not_after_as_of(None, cutoff)


def test_calendar_outside_versioned_range_fails_closed():
    with pytest.raises(DataUnavailableAsOfError):
        is_us_regular_trading_session(date(2101, 1, 3))


def test_version_guard():
    assert_version_match(expected="v1", actual="v1", component="features")
    with pytest.raises(VersionMismatchError, match="features"):
        assert_version_match(expected="v1", actual="v2", component="features")


def test_forward_excess_return_semantics_use_adjusted_prices():
    labels = compute_forward_return_labels(
        ticker_start_adjusted_close=100,
        ticker_target_adjusted_close=110,
        benchmark_start_adjusted_close=200,
        benchmark_target_adjusted_close=210,
    )
    assert labels["ticker_total_return"] == pytest.approx(0.10)
    assert labels["benchmark_total_return"] == pytest.approx(0.05)
    assert labels[FORWARD_EXCESS_RETURN_LABEL] == pytest.approx(0.05)
    assert labels["excess_return_direction"] == "POSITIVE"


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_label_prices_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError):
        compute_forward_return_labels(
            ticker_start_adjusted_close=bad,
            ticker_target_adjusted_close=110,
            benchmark_start_adjusted_close=200,
            benchmark_target_adjusted_close=210,
        )
