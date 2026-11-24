"""Mechanical future-information audit for Stage 09 datasets and reports."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, TYPE_CHECKING

from core.forecast_contract import (
    HORIZON_TRADING_DAYS,
    nth_trading_session_after,
    parse_utc_datetime,
    regular_session_close_utc,
)
from evaluation.point_in_time_dataset import Stage09EventSink, emit_stage09_event

if TYPE_CHECKING:
    from evaluation.monthly_backtest import BacktestReport
    from evaluation.point_in_time_dataset import PointInTimeDataset


LEAKAGE_AUDIT_VERSION = "stage09-temporal-leakage-v1"


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    code: str
    identity: str
    detail: str


@dataclass(frozen=True, slots=True)
class LeakageReport:
    future_record_count: int
    checked_source_rows: int
    checked_forecasts: int
    findings: tuple[LeakageFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_version": LEAKAGE_AUDIT_VERSION,
            "status": "PASS" if self.future_record_count == 0 else "FAIL",
            "future_record_count": self.future_record_count,
            "checked_source_rows": self.checked_source_rows,
            "checked_forecasts": self.checked_forecasts,
            "findings": [
                {"code": item.code, "identity": item.identity, "detail": item.detail}
                for item in self.findings
            ],
        }


def _audit_backtest(dataset: "PointInTimeDataset", report: "BacktestReport") -> LeakageReport:
    findings: list[LeakageFinding] = []
    for row in dataset.rows:
        identity = f"{row.snapshot.ticker}|{row.snapshot.as_of}"
        as_of = parse_utc_datetime(row.snapshot.as_of)
        finalized = parse_utc_datetime(row.snapshot.finalized_at)
        feature_at = parse_utc_datetime(row.snapshot.feature_materialized_at)
        outcome_at = parse_utc_datetime(row.outcome_materialized_at)
        expected_target = nth_trading_session_after(as_of.date(), HORIZON_TRADING_DAYS)
        if row.snapshot.target_date != expected_target.isoformat():
            findings.append(LeakageFinding("TARGET_NOT_EXACT_21ST_SESSION", identity, row.snapshot.target_date))
        if finalized > as_of or feature_at > as_of:
            findings.append(LeakageFinding("FEATURE_AFTER_AS_OF", identity, row.snapshot.feature_materialized_at))
        if outcome_at < regular_session_close_utc(expected_target):
            findings.append(LeakageFinding("OUTCOME_BEFORE_OBSERVABLE", identity, row.outcome_materialized_at))
        if finalized >= outcome_at:
            findings.append(LeakageFinding("FORECAST_NOT_BEFORE_OUTCOME", identity, row.outcome_materialized_at))
    for item in report.records:
        identity = f"{item['model_variant']}|{item['ticker']}|{item['as_of']}"
        forecast_at = parse_utc_datetime(item["forecast_materialized_at"])
        as_of = parse_utc_datetime(item["as_of"])
        outcome_at = parse_utc_datetime(item["outcome_materialized_at"])
        train_as_of = parse_utc_datetime(item["training_max_as_of"])
        train_outcome = parse_utc_datetime(item["training_max_outcome_materialized_at"])
        if forecast_at != as_of:
            findings.append(LeakageFinding("FORECAST_TIME_NOT_AS_OF", identity, str(item["forecast_materialized_at"])))
        if forecast_at >= outcome_at:
            findings.append(LeakageFinding("FORECAST_NOT_BEFORE_OUTCOME", identity, str(item["outcome_materialized_at"])))
        if train_as_of >= as_of:
            findings.append(LeakageFinding("TRAINING_CUTOFF_NOT_PAST", identity, str(item["training_max_as_of"])))
        if train_outcome > forecast_at:
            findings.append(LeakageFinding("FUTURE_TRAINING_OUTCOME", identity, str(item["training_max_outcome_materialized_at"])))
        if int(item["horizon_trading_days"]) != HORIZON_TRADING_DAYS:
            findings.append(LeakageFinding("INVALID_HORIZON", identity, str(item["horizon_trading_days"])))
    return LeakageReport(
        future_record_count=len(findings),
        checked_source_rows=len(dataset.rows),
        checked_forecasts=len(report.records),
        findings=tuple(findings),
    )


def audit_backtest(
    dataset: "PointInTimeDataset", report: "BacktestReport", *,
    event_sink: Stage09EventSink | None = None, run_id: str = "stage09-r1-a1",
) -> LeakageReport:
    started_at = time.perf_counter()
    try:
        result = _audit_backtest(dataset, report)
    except Exception:
        emit_stage09_event(event_sink, boundary="leakage_audit", outcome="fail_closed",
            kind="error", started_at=started_at, run_id=run_id,
            audit_version=LEAKAGE_AUDIT_VERSION, error_code="STAGE09_AUDIT_REJECTED")
        raise
    failed = result.future_record_count > 0
    emit_stage09_event(event_sink, boundary="leakage_audit",
        outcome="fail_closed" if failed else "success", kind="error" if failed else "lifecycle",
        started_at=started_at, run_id=run_id, audit_version=LEAKAGE_AUDIT_VERSION,
        row_count=result.checked_source_rows, forecast_count=result.checked_forecasts,
        error_code="STAGE09_FUTURE_RECORDS_DETECTED" if failed else None,
        degraded_flags=["future_records"] if failed else [])
    return result


__all__ = ["LEAKAGE_AUDIT_VERSION", "LeakageFinding", "LeakageReport", "audit_backtest"]
