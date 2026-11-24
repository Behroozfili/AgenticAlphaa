from datetime import date

from core.forecast_contract import ForecastCutoffPolicy
from evaluation.train_live_quant_artifact import _monthly_cutoffs


def test_month_to_date_cutoffs_use_requested_day_or_previous_session():
    cutoffs = list(_monthly_cutoffs(
        date(2026, 8, 1), date(2026, 10, 31),
        cutoff_policy=ForecastCutoffPolicy.MONTH_TO_DATE,
        month_day=18,
    ))
    assert cutoffs == [date(2026, 8, 18), date(2026, 9, 18), date(2026, 10, 16)]


def test_month_end_cutoffs_ignore_month_day():
    cutoffs = list(_monthly_cutoffs(
        date(2026, 8, 1), date(2026, 9, 30),
        cutoff_policy=ForecastCutoffPolicy.MONTH_END,
        month_day=18,
    ))
    assert cutoffs == [date(2026, 8, 31), date(2026, 9, 30)]
