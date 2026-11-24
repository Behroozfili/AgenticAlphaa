from datetime import date

import pytest

from core.forecast_contract import ForecastContext, ForecastCutoffPolicy


def _context(**overrides):
    values = {
        "ticker": "MSFT", "as_of": "2024-05-30T20:00:00Z",
        "schema_version": "schema-v1", "dataset_version": "dataset-v1",
        "feature_version": "features-v1", "model_version": "model-v1",
        "cutoff_policy": ForecastCutoffPolicy.MONTH_TO_DATE,
    }
    values.update(overrides)
    return ForecastContext(**values)


def test_month_to_date_accepts_completed_regular_session_and_round_trips():
    context = _context()
    assert context.cutoff_policy is ForecastCutoffPolicy.MONTH_TO_DATE
    assert context.target == date(2024, 7, 1)
    assert ForecastContext.from_dict(context.to_dict()) == context


def test_month_to_date_still_rejects_weekends_and_non_close_timestamps():
    with pytest.raises(ValueError, match="regular US trading session"):
        _context(as_of="2024-05-25T20:00:00Z")
    with pytest.raises(ValueError, match="regular-session close"):
        _context(as_of="2024-05-30T19:59:00Z")


def test_legacy_payload_without_cutoff_policy_remains_month_end():
    month_end = _context(
        as_of="2024-05-31T20:00:00Z", cutoff_policy=ForecastCutoffPolicy.MONTH_END,
    )
    payload = month_end.to_dict()
    payload.pop("cutoff_policy")
    assert ForecastContext.from_dict(payload).cutoff_policy is ForecastCutoffPolicy.MONTH_END

