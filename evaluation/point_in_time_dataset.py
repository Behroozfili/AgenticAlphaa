"""Deterministic point-in-time dataset materialization for Stage 09 backtests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.forecast_contract import (
    AnalysisMode,
    HORIZON_TRADING_DAYS,
    last_trading_session_of_month,
    nth_trading_session_after,
    regular_session_close_utc,
)
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION
from tools.market_tools.historical_dataset_builder import (
    FeatureSnapshot,
    LabeledSnapshot,
    eligible_row_content_hash,
    validate_universe_manifest,
)
from tools.market_tools.market_data import MARKET_DATA_VERSION


POINT_IN_TIME_DATASET_VERSION = "stage09-point-in-time-dataset-v1"
SCIENTIFIC_MINIMUMS = MappingProxyType({"tickers": 10, "cutoffs": 60, "forecasts": 500})
STAGE09_RUNTIME_IDENTITY = MappingProxyType({
    "stage": "09",
    "stage_revision": 1,
    "attempt": 2,
    "agent": "quant",
    "implementation_version": "stage09-r1-a2",
    "backtest_version": "stage09-expanding-walk-forward-v1",
})
Stage09EventSink = Callable[[Mapping[str, Any]], None]
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
_EVENT_CONTEXT_FIELDS = frozenset({
    "run_id", "fold_id", "model_variant", "ticker", "as_of", "target",
    "horizon", "analysis_mode", "dataset_version",
    "feature_version", "model_version", "schema_version", "metrics_version",
    "audit_version", "fixture_version", "error_code", "degraded_flags",
    "row_count", "fold_count", "forecast_count", "evidence_count",
    "completed_folds", "total_folds", "training_cutoffs", "variant_count",
})
_COUNT_FIELDS = frozenset({
    "row_count", "fold_count", "forecast_count", "evidence_count",
    "completed_folds", "total_folds", "training_cutoffs", "variant_count",
})


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if "PRIVATE_SENTINEL" in value.upper() or not _SAFE_TEXT.fullmatch(value):
        return "REDACTED"
    return value


def emit_stage09_event(
    event_sink: Stage09EventSink | None, *, boundary: str, outcome: str,
    started_at: float, kind: str = "lifecycle", **context: Any,
) -> None:
    """Emit a bounded payload-safe event; sink failures never alter science."""
    if event_sink is None:
        return
    latency_ms = max(0.0, (time.perf_counter() - started_at) * 1000.0)
    event: dict[str, Any] = {
        **dict(STAGE09_RUNTIME_IDENTITY),
        "kind": _safe_text(kind) or "lifecycle",
        "boundary": _safe_text(boundary) or "REDACTED",
        "outcome": _safe_text(outcome) or "REDACTED",
        "latency_ms": float(latency_ms),
    }
    for name, value in context.items():
        if name not in _EVENT_CONTEXT_FIELDS:
            continue
        if name in _COUNT_FIELDS:
            if isinstance(value, bool):
                continue
            try:
                event[name] = max(0, min(int(value), 1_000_000))
            except (TypeError, ValueError, OverflowError):
                continue
        elif name == "degraded_flags":
            if isinstance(value, (list, tuple, set)):
                flags = [_safe_text(item) for item in value]
                event[name] = sorted({item for item in flags if item})[:16]
        else:
            safe = _safe_text(value)
            if safe is not None:
                event[name] = safe
    try:
        event_sink(MappingProxyType(event))
    except Exception:
        return


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def classify_scientific_report(
    *, ticker_count: int, monthly_cutoff_count: int, eligible_oos_forecasts: int,
) -> str:
    """Classify only from executed walk-forward OOS coverage, never source rows."""
    values = (ticker_count, monthly_cutoff_count, eligible_oos_forecasts)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("scientific threshold inputs must be non-negative integers")
    return (
        "SCIENTIFIC"
        if ticker_count >= SCIENTIFIC_MINIMUMS["tickers"]
        and monthly_cutoff_count >= SCIENTIFIC_MINIMUMS["cutoffs"]
        and eligible_oos_forecasts >= SCIENTIFIC_MINIMUMS["forecasts"]
        else "PRELIMINARY"
    )


def _month(year: int, month: int, offset: int) -> tuple[int, int]:
    absolute = year * 12 + month - 1 + offset
    return absolute // 12, absolute % 12 + 1


@dataclass(frozen=True, slots=True)
class PointInTimeDataset:
    rows: tuple[LabeledSnapshot, ...]
    universe_manifest: Mapping[str, Any]
    manifest: Mapping[str, Any]
    manifest_hash: str

    def row_keys(self) -> tuple[str, ...]:
        return tuple(f"{r.snapshot.ticker}|{r.snapshot.as_of}" for r in self.rows)


def _feature_values(index: int, ticker_index: int, missing_every: int) -> tuple[dict[str, float | None], dict[str, bool]]:
    phase = index * 0.71 + ticker_index * 0.83
    sentiment = None if index % missing_every == 0 else math.sin(phase - 0.35)
    financial = None if (index + ticker_index) % (missing_every + 1) == 0 else math.cos(phase / 2.0) * 0.12
    graph = 0.0 if (index + ticker_index) % 7 == 0 else 1.0
    values: dict[str, float | None] = {
        "market_momentum_21": math.sin(phase - 0.71) * 0.035,
        "market_volatility_21": 0.12 + abs(math.cos(phase)) * 0.08,
        "market_beta_63": 0.85 + ticker_index * 0.12 + math.sin(index / 5.0) * 0.08,
        "market_relative_strength_21": math.sin(phase - 0.5) * 0.025,
        "research_source_diversity": float(2 + (index + ticker_index) % 4),
        "research_graph_coverage": graph,
        "sentiment_aggregate_30": sentiment,
        "financial_valuation_gap": financial,
    }
    missing = {
        "research_features__missing": False,
        "financial_features__missing": financial is None,
        "sentiment_features__missing": sentiment is None,
    }
    return values, missing


def _materialize_synthetic_fixture(payload: Mapping[str, Any], *, fixture_sha256: str) -> PointInTimeDataset:
    """Materialize a compact, deterministic fixture; never contacts a provider."""
    spec = payload.get("generation_spec")
    if not isinstance(spec, Mapping):
        raise TypeError("fixture requires generation_spec")
    tickers = tuple(str(item).upper() for item in spec.get("tickers", ()))
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("fixture tickers must be unique and non-empty")
    cutoff_count = int(spec.get("monthly_cutoffs", 0))
    if cutoff_count < 37:
        raise ValueError("fixture requires at least 37 monthly cutoffs")
    year, month = (int(part) for part in str(spec["first_month"]).split("-"))
    benchmark = str(spec.get("benchmark") or "SPY").upper()
    missing_every = int(spec.get("missing_every", 6))
    universe = validate_universe_manifest(payload["universe"])
    rows: list[LabeledSnapshot] = []
    for index in range(cutoff_count):
        cutoff_year, cutoff_month = _month(year, month, index)
        cutoff_day = last_trading_session_of_month(cutoff_year, cutoff_month)
        as_of = regular_session_close_utc(cutoff_day)
        target_day = nth_trading_session_after(cutoff_day, HORIZON_TRADING_DAYS)
        target_close = regular_session_close_utc(target_day)
        for ticker_index, ticker in enumerate(tickers):
            values, missing = _feature_values(index, ticker_index, missing_every)
            excess = (
                0.021 * math.sin(index * 0.71 + ticker_index * 0.83)
                + 0.28 * float(values["market_momentum_21"] or 0.0)
                + 0.003 * (float(values["research_graph_coverage"] or 0.0) - 0.5)
            )
            benchmark_return = 0.003 + 0.006 * math.cos(index * 0.31)
            ticker_return = benchmark_return + excess
            snapshot_seed = {
                "ticker": ticker,
                "as_of": as_of.isoformat(),
                "target_date": target_day.isoformat(),
                "values": values,
                "fixture_sha256": fixture_sha256,
            }
            snapshot = FeatureSnapshot(
                ticker=ticker,
                as_of=as_of.isoformat().replace("+00:00", "Z"),
                target_date=target_day.isoformat(),
                horizon_trading_days=HORIZON_TRADING_DAYS,
                benchmark=benchmark,
                analysis_mode=AnalysisMode.HISTORICAL_BACKTEST.value,
                dataset_version=MARKET_DATA_VERSION,
                feature_version=MARKET_FEATURE_VERSION,
                values=MappingProxyType(values),
                evidence_ids=(f"stage09-{ticker.lower()}-{index:03d}",),
                missing_indicators=MappingProxyType(missing),
                degraded=any(missing.values()),
                finalized_at=as_of.isoformat().replace("+00:00", "Z"),
                snapshot_hash=stable_hash(snapshot_seed),
                feature_materialized_at=as_of.isoformat().replace("+00:00", "Z"),
            )
            rows.append(LabeledSnapshot(
                snapshot=snapshot,
                ticker_total_return=float(ticker_return),
                benchmark_total_return=float(benchmark_return),
                forward_21d_excess_return=float(ticker_return - benchmark_return),
                direction_positive=1 if ticker_return - benchmark_return > 0.0 else 0,
                outcome_materialized_at=target_close.isoformat().replace("+00:00", "Z"),
            ))
    ordered = tuple(sorted(rows, key=lambda row: (row.snapshot.as_of, row.snapshot.ticker)))
    eligible_hash = eligible_row_content_hash(ordered)
    # A source dataset cannot establish OOS coverage before walk-forward folds
    # execute. The final report reclassifies from its common eligible OOS set.
    report_level = classify_scientific_report(
        ticker_count=len(tickers), monthly_cutoff_count=cutoff_count,
        eligible_oos_forecasts=0,
    )
    manifest = {
        "schema_version": "1.0",
        "dataset_builder_version": POINT_IN_TIME_DATASET_VERSION,
        "fixture_sha256": fixture_sha256,
        "universe_version": universe["universe_version"],
        "universe_manifest_hash": universe["manifest_hash"],
        "dataset_version": MARKET_DATA_VERSION,
        "feature_version": MARKET_FEATURE_VERSION,
        "ticker_count": len(tickers),
        "monthly_cutoff_count": cutoff_count,
        "eligible_row_count": len(ordered),
        "eligible_rows_sha256": eligible_hash,
        "cutoff_schedule": "last_regular_us_trading_session_of_calendar_month",
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "scientific_report_level": report_level,
        "scientific_minimums": dict(SCIENTIFIC_MINIMUMS),
    }
    manifest_hash = stable_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    return PointInTimeDataset(
        rows=ordered,
        universe_manifest=MappingProxyType(universe),
        manifest=MappingProxyType(manifest),
        manifest_hash=manifest_hash,
    )


def materialize_synthetic_fixture(
    payload: Mapping[str, Any], *, fixture_sha256: str,
    event_sink: Stage09EventSink | None = None, run_id: str = "stage09-r1-a1",
) -> PointInTimeDataset:
    started_at = time.perf_counter()
    try:
        result = _materialize_synthetic_fixture(payload, fixture_sha256=fixture_sha256)
    except Exception:
        emit_stage09_event(event_sink, boundary="fixture_materialization", outcome="fail_closed",
            kind="error", started_at=started_at, run_id=run_id,
            error_code="STAGE09_FIXTURE_REJECTED")
        raise
    emit_stage09_event(event_sink, boundary="fixture_materialization", outcome="success",
        started_at=started_at, run_id=run_id, fixture_version=POINT_IN_TIME_DATASET_VERSION,
        row_count=len(result.rows), evidence_count=sum(len(row.snapshot.evidence_ids) for row in result.rows),
        degraded_flags=["preliminary_fixture"] if result.manifest["scientific_report_level"] == "PRELIMINARY" else [])
    return result


def load_point_in_time_fixture(
    path: str | Path, *, event_sink: Stage09EventSink | None = None,
    run_id: str = "stage09-r1-a1",
) -> PointInTimeDataset:
    fixture_path = Path(path)
    raw = fixture_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("fixture_purpose") != "stage09_deterministic_walk_forward_regression_only":
        raise ValueError("fixture is not approved for Stage 09 regression")
    return materialize_synthetic_fixture(payload, fixture_sha256=hashlib.sha256(raw).hexdigest(),
                                         event_sink=event_sink, run_id=run_id)


__all__ = [
    "POINT_IN_TIME_DATASET_VERSION", "STAGE09_RUNTIME_IDENTITY", "PointInTimeDataset",
    "SCIENTIFIC_MINIMUMS", "Stage09EventSink", "classify_scientific_report",
    "emit_stage09_event",
    "load_point_in_time_fixture", "materialize_synthetic_fixture", "stable_hash",
]
