"""Versioned, point-in-time contract for monthly equity forecasts."""

from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError, VersionMismatchError

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
HORIZON_TRADING_DAYS = 21
DEFAULT_BENCHMARK = "SPY"
FORWARD_EXCESS_RETURN_LABEL = "forward_21d_excess_return"
CALENDAR_VERSION = "XNYS-regular-v1"
_MIN_YEAR, _MAX_YEAR = 1998, 2100
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


class AnalysisMode(StrEnum):
    LIVE = "LIVE"
    HISTORICAL_BACKTEST = "HISTORICAL_BACKTEST"


class ForecastCutoffPolicy(StrEnum):
    MONTH_END = "MONTH_END"
    MONTH_TO_DATE = "MONTH_TO_DATE"


def parse_utc_datetime(value: datetime | str, *, field_name: str = "timestamp") -> datetime:
    """Parse an aware timestamp and normalize it to UTC; never guess timezone."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith(("Z", "z")):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be ISO-8601") from exc
    else:
        raise TypeError(f"{field_name} must be datetime or ISO-8601 text")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _observed(day: date) -> date:
    return day - timedelta(days=1) if day.weekday() == 5 else (
        day + timedelta(days=1) if day.weekday() == 6 else day
    )


def _xnys_new_year_closure(nominal_year: int) -> date | None:
    """XNYS does not close Friday when January 1 falls on Saturday."""
    new_year = date(nominal_year, 1, 1)
    if new_year.weekday() == 5:
        return None
    if new_year.weekday() == 6:
        return new_year + timedelta(days=1)
    return new_year


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    length = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * length) // 451
    month = (h + length - 7 * m + 114) // 31
    return date(year, month, (h + length - 7 * m + 114) % 31 + 1)


_EXCEPTIONAL_CLOSURES = frozenset({
    date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14),
    date(2004, 6, 11), date(2007, 1, 2), date(2012, 10, 29),
    date(2012, 10, 30), date(2018, 12, 5),
    date(2025, 1, 9),
})


def _require_year(year: int) -> None:
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        raise DataUnavailableAsOfError(
            f"{CALENDAR_VERSION} supports {_MIN_YEAR}..{_MAX_YEAR}; cannot classify {year}"
        )


@lru_cache(maxsize=None)
def us_regular_market_holidays(year: int) -> frozenset[date]:
    """Full-day XNYS closures for one supported calendar year."""
    _require_year(year)
    holidays: set[date] = set()
    for nominal_year in (year, year + 1):
        closure = _xnys_new_year_closure(nominal_year)
        if closure is not None and closure.year == year:
            holidays.add(closure)
    holidays.update({
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    })
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    holidays.update(day for day in _EXCEPTIONAL_CLOSURES if day.year == year)
    return frozenset(holidays)


def is_us_regular_trading_session(day: date) -> bool:
    if not isinstance(day, date) or isinstance(day, datetime):
        raise TypeError("day must be a date")
    _require_year(day.year)
    return day.weekday() < 5 and day not in us_regular_market_holidays(day.year)


def nth_trading_session_after(day: date, sessions: int = HORIZON_TRADING_DAYS) -> date:
    """Return the Nth regular US session strictly after the supplied date."""
    if not isinstance(sessions, int) or isinstance(sessions, bool) or sessions <= 0:
        raise ValueError("sessions must be a positive integer")
    cursor, found = day, 0
    while found < sessions:
        cursor += timedelta(days=1)
        if is_us_regular_trading_session(cursor):
            found += 1
    return cursor


def last_trading_session_of_month(year: int, month: int) -> date:
    cursor = date(year, month, calendar.monthrange(year, month)[1])
    while not is_us_regular_trading_session(cursor):
        cursor -= timedelta(days=1)
    return cursor


def regular_session_close_utc(day: date) -> datetime:
    """Return the actual close for a supported regular monthly cutoff session."""
    if not is_us_regular_trading_session(day):
        raise DataUnavailableAsOfError(f"{day} is not a regular US trading session")
    thanksgiving = _nth_weekday(day.year, 11, 3, 4)
    close_hour = 13 if day == thanksgiving + timedelta(days=1) else 16
    return datetime(day.year, day.month, day.day, close_hour, tzinfo=NEW_YORK).astimezone(UTC)


def assert_not_after_as_of(
    evidence_timestamp: datetime | str | None,
    as_of: datetime | str,
    *,
    field_name: str = "evidence_timestamp",
) -> datetime:
    cutoff = parse_utc_datetime(as_of, field_name="as_of")
    if evidence_timestamp is None:
        raise DataUnavailableAsOfError(f"{field_name} is required for point-in-time use")
    observed = parse_utc_datetime(evidence_timestamp, field_name=field_name)
    if observed > cutoff:
        raise TemporalLeakageError(
            f"{field_name} {observed.isoformat()} is after as_of {cutoff.isoformat()}"
        )
    return observed


def assert_version_match(*, expected: str, actual: str, component: str) -> None:
    if expected != actual:
        raise VersionMismatchError(
            f"{component} version mismatch: expected {expected!r}, got {actual!r}"
        )


def compute_forward_return_labels(
    *,
    ticker_start_adjusted_close: float,
    ticker_target_adjusted_close: float,
    benchmark_start_adjusted_close: float,
    benchmark_target_adjusted_close: float,
) -> dict[str, float | str]:
    values = {
        "ticker_start_adjusted_close": ticker_start_adjusted_close,
        "ticker_target_adjusted_close": ticker_target_adjusted_close,
        "benchmark_start_adjusted_close": benchmark_start_adjusted_close,
        "benchmark_target_adjusted_close": benchmark_target_adjusted_close,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    ticker_return = ticker_target_adjusted_close / ticker_start_adjusted_close - 1.0
    benchmark_return = benchmark_target_adjusted_close / benchmark_start_adjusted_close - 1.0
    excess_return = ticker_return - benchmark_return
    return {
        "ticker_total_return": ticker_return,
        "benchmark_total_return": benchmark_return,
        FORWARD_EXCESS_RETURN_LABEL: excess_return,
        "excess_return_direction": "POSITIVE" if excess_return > 0 else "NEGATIVE",
    }


def _parse_target(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime target must be timezone-aware")
        return value.astimezone(NEW_YORK).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError("target must be an ISO-8601 date") from exc
    raise TypeError("target must be a date or ISO-8601 date text")


@dataclass(frozen=True, slots=True, kw_only=True)
class ForecastContext:
    ticker: str
    as_of: datetime | str
    schema_version: str
    dataset_version: str
    feature_version: str
    model_version: str
    target: date | datetime | str | None = None
    horizon_trading_days: int = HORIZON_TRADING_DAYS
    lookback_days: int = 365
    benchmark: str = DEFAULT_BENCHMARK
    analysis_mode: AnalysisMode | str = AnalysisMode.LIVE
    cutoff_policy: ForecastCutoffPolicy | str = ForecastCutoffPolicy.MONTH_END

    def __post_init__(self) -> None:
        ticker, benchmark = self.ticker.strip().upper(), self.benchmark.strip().upper()
        if not _TICKER.fullmatch(ticker) or not _TICKER.fullmatch(benchmark):
            raise ValueError("ticker and benchmark must be canonical symbols")
        if self.horizon_trading_days != HORIZON_TRADING_DAYS:
            raise ValueError("forecast horizon must be exactly 21 trading sessions")
        if not isinstance(self.lookback_days, int) or isinstance(self.lookback_days, bool) or self.lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer")
        versions = {
            "schema_version": self.schema_version, "dataset_version": self.dataset_version,
            "feature_version": self.feature_version, "model_version": self.model_version,
        }
        for name, value in versions.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        try:
            mode = self.analysis_mode if isinstance(self.analysis_mode, AnalysisMode) else AnalysisMode(str(self.analysis_mode).upper())
        except ValueError as exc:
            raise ValueError("analysis_mode must be LIVE or HISTORICAL_BACKTEST") from exc
        try:
            cutoff_policy = (
                self.cutoff_policy if isinstance(self.cutoff_policy, ForecastCutoffPolicy)
                else ForecastCutoffPolicy(str(self.cutoff_policy).upper())
            )
        except ValueError as exc:
            raise ValueError("cutoff_policy must be MONTH_END or MONTH_TO_DATE") from exc
        cutoff = parse_utc_datetime(self.as_of, field_name="as_of")
        market_day = cutoff.astimezone(NEW_YORK).date()
        if not is_us_regular_trading_session(market_day):
            raise ValueError("as_of must be a regular US trading session")
        if cutoff_policy is ForecastCutoffPolicy.MONTH_END:
            month_end = last_trading_session_of_month(market_day.year, market_day.month)
            if market_day != month_end:
                raise ValueError(f"as_of must be the last regular US trading session of its month ({month_end})")
        expected_close = regular_session_close_utc(market_day)
        if cutoff != expected_close:
            raise ValueError(f"as_of must equal the regular-session close ({expected_close.isoformat()})")
        expected_target = nth_trading_session_after(market_day)
        target = expected_target if self.target is None else _parse_target(self.target)
        if target != expected_target:
            raise ValueError(f"target must be the 21st regular trading session after as_of ({expected_target})")
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "as_of", cutoff)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "analysis_mode", mode)
        object.__setattr__(self, "cutoff_policy", cutoff_policy)
        for name, value in versions.items():
            object.__setattr__(self, name, value.strip())

    @property
    def target_date(self) -> date:
        return self.target  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of.isoformat().replace("+00:00", "Z"),  # type: ignore[union-attr]
            "target": self.target.isoformat(),  # type: ignore[union-attr]
            "horizon_trading_days": self.horizon_trading_days,
            "lookback_days": self.lookback_days,
            "benchmark": self.benchmark,
            "analysis_mode": self.analysis_mode.value,  # type: ignore[union-attr]
            "cutoff_policy": self.cutoff_policy.value,  # type: ignore[union-attr]
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "calendar_version": CALENDAR_VERSION,
            "label": FORWARD_EXCESS_RETURN_LABEL,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ForecastContext":
        allowed = {
            "ticker", "as_of", "target", "horizon_trading_days", "lookback_days",
            "benchmark", "analysis_mode", "schema_version", "dataset_version",
            "feature_version", "model_version", "cutoff_policy",
        }
        required = (allowed - {"cutoff_policy"}) | {"calendar_version", "label"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing required ForecastContext fields: {sorted(missing)}")
        unknown = set(payload) - (allowed | {"calendar_version", "label"})
        if unknown:
            raise ValueError(f"unknown ForecastContext fields: {sorted(unknown)}")
        assert_version_match(
            expected=CALENDAR_VERSION,
            actual=str(payload["calendar_version"]),
            component="calendar",
        )
        assert_version_match(
            expected=FORWARD_EXCESS_RETURN_LABEL,
            actual=str(payload["label"]),
            component="label",
        )
        values = {key: value for key, value in payload.items() if key in allowed}
        values.setdefault("cutoff_policy", ForecastCutoffPolicy.MONTH_END)
        return cls(**values)


__all__ = [
    "AnalysisMode", "CALENDAR_VERSION", "DEFAULT_BENCHMARK", "ForecastCutoffPolicy",
    "FORWARD_EXCESS_RETURN_LABEL", "ForecastContext", "HORIZON_TRADING_DAYS",
    "assert_not_after_as_of", "assert_version_match", "compute_forward_return_labels",
    "is_us_regular_trading_session", "last_trading_session_of_month",
    "nth_trading_session_after", "parse_utc_datetime", "us_regular_market_holidays",
    "regular_session_close_utc",
]
