"""Leakage-safe expanding-window backtest over deterministic Quant forecasts."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from core.forecast_contract import HORIZON_TRADING_DAYS, parse_utc_datetime
from evaluation.forecast_metrics import compute_forecast_metrics
from evaluation.point_in_time_dataset import (
    PointInTimeDataset, Stage09EventSink, classify_scientific_report,
    emit_stage09_event, stable_hash,
)
from tools.market_tools.forecast_model import (
    ACTION_POLICY_VERSION,
    MINIMUM_TRAINING_CUTOFFS,
    apply_action_policy,
    fit_deterministic_model,
    predict,
)
from tools.market_tools.historical_dataset_builder import (
    FeatureSnapshot,
    LabeledSnapshot,
    build_data_readiness,
)


BACKTEST_VERSION = "stage09-expanding-walk-forward-v1"
TRANSACTION_COST_BPS_ONE_WAY = 10.0
MODEL_VARIANTS = (
    "class_prior_always_hold",
    "momentum_21d",
    "market_only",
    "no_graph",
    "no_sentiment",
    "no_financial",
    "full_model",
)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _row_key(row: LabeledSnapshot) -> str:
    return f"{row.snapshot.ticker}|{row.snapshot.as_of}"


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_id: str
    forecast_as_of: str
    training_rows: tuple[LabeledSnapshot, ...]
    test_rows: tuple[LabeledSnapshot, ...]
    training_cutoffs: int
    training_max_as_of: str
    training_max_outcome_materialized_at: str

    def manifest(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "forecast_as_of": self.forecast_as_of,
            "training_cutoffs": self.training_cutoffs,
            "training_row_keys": [_row_key(row) for row in self.training_rows],
            "test_row_keys": [_row_key(row) for row in self.test_rows],
            "training_max_as_of": self.training_max_as_of,
            "training_max_outcome_materialized_at": self.training_max_outcome_materialized_at,
        }


@dataclass(frozen=True, slots=True)
class BacktestReport:
    manifest: Mapping[str, Any]
    folds: tuple[Mapping[str, Any], ...]
    records: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    leakage_audit: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": dict(self.manifest),
            "folds": [dict(item) for item in self.folds],
            "records": [dict(item) for item in self.records],
            "metrics": dict(self.metrics),
            "leakage_audit": dict(self.leakage_audit),
        }


def build_expanding_folds(
    rows: Iterable[LabeledSnapshot], *, minimum_training_cutoffs: int = MINIMUM_TRAINING_CUTOFFS,
) -> tuple[WalkForwardFold, ...]:
    if minimum_training_cutoffs < MINIMUM_TRAINING_CUTOFFS:
        raise ValueError(f"minimum_training_cutoffs cannot be below {MINIMUM_TRAINING_CUTOFFS}")
    ordered = tuple(sorted(rows, key=lambda row: (row.snapshot.as_of, row.snapshot.ticker)))
    cutoffs = sorted({row.snapshot.as_of for row in ordered})
    folds: list[WalkForwardFold] = []
    for cutoff in cutoffs:
        forecast_at = parse_utc_datetime(cutoff, field_name="forecast_as_of")
        test = tuple(row for row in ordered if row.snapshot.as_of == cutoff)
        train = tuple(row for row in ordered if (
            parse_utc_datetime(row.snapshot.as_of) < forecast_at
            and parse_utc_datetime(row.outcome_materialized_at, field_name="outcome_materialized_at") <= forecast_at
        ))
        training_cutoffs = len({row.snapshot.as_of for row in train})
        if training_cutoffs < minimum_training_cutoffs or not test:
            continue
        folds.append(WalkForwardFold(
            fold_id=f"fold-{len(folds) + 1:03d}",
            forecast_as_of=cutoff,
            training_rows=train,
            test_rows=test,
            training_cutoffs=training_cutoffs,
            training_max_as_of=max(row.snapshot.as_of for row in train),
            training_max_outcome_materialized_at=max(row.outcome_materialized_at for row in train),
        ))
    if not folds:
        raise ValueError("no expanding fold has 36 observable historical cutoffs")
    return tuple(folds)


def _include_feature(name: str, variant: str) -> bool:
    lowered = name.lower()
    if variant == "market_only":
        return lowered.startswith("market_")
    if variant == "no_graph":
        return "graph" not in lowered
    if variant == "no_sentiment":
        return not lowered.startswith("sentiment_")
    if variant == "no_financial":
        return not lowered.startswith("financial_")
    return True


def _project_values(values: Mapping[str, float | None], variant: str) -> dict[str, float | None]:
    projected = {name: value for name, value in values.items() if _include_feature(name, variant)}
    if not projected:
        raise ValueError(f"variant {variant} selected no features")
    return projected


def _project_row(row: LabeledSnapshot, variant: str) -> LabeledSnapshot:
    values = _project_values(row.snapshot.values, variant)
    payload = row.snapshot.canonical_payload()
    payload["values"] = values
    snapshot = dataclasses.replace(
        row.snapshot,
        values=MappingProxyType(values),
        snapshot_hash=stable_hash(payload),
    )
    return dataclasses.replace(row, snapshot=snapshot)


def _baseline_prediction(variant: str, train: tuple[LabeledSnapshot, ...], test: LabeledSnapshot) -> dict[str, Any]:
    if variant == "class_prior_always_hold":
        probability = sum(row.direction_positive for row in train) / len(train)
        expected_return = sum(row.ticker_total_return for row in train) / len(train)
        expected_excess = sum(row.forward_21d_excess_return for row in train) / len(train)
        action = "HOLD"
    elif variant == "momentum_21d":
        momentum = test.snapshot.values.get("market_momentum_21")
        if momentum is None:
            raise ValueError("momentum baseline requires market_momentum_21")
        probability = 0.6 if float(momentum) > 0.0 else (0.4 if float(momentum) < 0.0 else 0.5)
        expected_excess = float(momentum)
        expected_return = float(momentum) + sum(row.benchmark_total_return for row in train) / len(train)
        action = apply_action_policy(probability, expected_excess)
    else:
        raise ValueError(f"unknown baseline {variant}")
    return {
        "probability_positive": float(probability),
        "expected_return_21d": float(expected_return),
        "expected_excess_return": float(expected_excess),
        "action": action,
        "model_version": f"stage09-{variant}-v1",
    }


def _record(
    *, dataset: PointInTimeDataset, fold: WalkForwardFold, row: LabeledSnapshot,
    variant: str, prediction: Mapping[str, Any], run_id: str,
) -> dict[str, Any]:
    snapshot = row.snapshot
    return {
        "ticker": snapshot.ticker,
        "as_of": snapshot.as_of,
        "target_date": snapshot.target_date,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "analysis_mode": snapshot.analysis_mode,
        "universe_manifest_hash": dataset.manifest["universe_manifest_hash"],
        "dataset_version": snapshot.dataset_version,
        "feature_version": snapshot.feature_version,
        "model_version": str(prediction["model_version"]),
        "action_policy_version": ACTION_POLICY_VERSION,
        "model_variant": variant,
        "benchmark": snapshot.benchmark,
        "run_id": run_id,
        "fold_id": fold.fold_id,
        "evidence_ids": list(snapshot.evidence_ids),
        "missing": bool(any(snapshot.missing_indicators.values())),
        "degraded": snapshot.degraded,
        "missing_indicators": dict(snapshot.missing_indicators),
        "feature_materialized_at": snapshot.feature_materialized_at,
        "forecast_materialized_at": snapshot.as_of,
        "outcome_materialized_at": row.outcome_materialized_at,
        "training_max_as_of": fold.training_max_as_of,
        "training_max_outcome_materialized_at": fold.training_max_outcome_materialized_at,
        "probability_positive": float(prediction["probability_positive"]),
        "expected_return_21d": float(prediction["expected_return_21d"]),
        "expected_excess_return": float(prediction["expected_excess_return"]),
        "action": str(prediction["action"]),
        "actual_ticker_return": row.ticker_total_return,
        "actual_benchmark_return": row.benchmark_total_return,
        "actual_excess_return": row.forward_21d_excess_return,
        "actual_direction_positive": row.direction_positive,
    }


def _run_monthly_backtest(
    dataset: PointInTimeDataset, *, run_id: str = "stage09-r1-a1",
    minimum_training_cutoffs: int = MINIMUM_TRAINING_CUTOFFS,
    transaction_cost_bps_one_way: float = TRANSACTION_COST_BPS_ONE_WAY,
    event_sink: Stage09EventSink | None = None,
) -> BacktestReport:
    if transaction_cost_bps_one_way != TRANSACTION_COST_BPS_ONE_WAY:
        raise ValueError("Stage 09 v1 requires the pre-registered 10 bps one-way cost")
    folds = build_expanding_folds(dataset.rows, minimum_training_cutoffs=minimum_training_cutoffs)
    records: list[dict[str, Any]] = []
    expected_per_variant = sum(len(fold.test_rows) for fold in folds)
    for fold in folds:
        fitted: dict[str, Any] = {}
        for variant in MODEL_VARIANTS:
            if variant in {"class_prior_always_hold", "momentum_21d"}:
                continue
            fit_started = time.perf_counter()
            try:
                train = tuple(_project_row(row, variant) for row in fold.training_rows)
                readiness = build_data_readiness(
                    rows=train,
                    universe_manifest=dataset.universe_manifest,
                    run_id=f"{run_id}-{fold.fold_id}-{variant}",
                    source_hashes={"stage09_fixture_sha256": str(dataset.manifest["fixture_sha256"])},
                )
                fitted[variant] = fit_deterministic_model(
                    rows=train,
                    readiness=readiness,
                    train_through=fold.forecast_as_of,
                    run_id=f"{run_id}-{fold.fold_id}-{variant}",
                )
            except Exception:
                emit_stage09_event(event_sink, boundary="fold_model_fit", outcome="fail_closed",
                    kind="error", started_at=fit_started, run_id=run_id, fold_id=fold.fold_id,
                    model_variant=variant, training_cutoffs=fold.training_cutoffs,
                    dataset_version=str(dataset.manifest["dataset_version"]),
                    feature_version=str(dataset.manifest["feature_version"]),
                    error_code="STAGE09_MODEL_FIT_REJECTED")
                raise
            emit_stage09_event(event_sink, boundary="fold_model_fit", outcome="success",
                started_at=fit_started, run_id=run_id, fold_id=fold.fold_id,
                model_variant=variant, training_cutoffs=fold.training_cutoffs,
                row_count=len(train), evidence_count=sum(len(row.snapshot.evidence_ids) for row in train),
                dataset_version=str(dataset.manifest["dataset_version"]),
                feature_version=str(dataset.manifest["feature_version"]),
                model_version=fitted[variant].model_version)
        for variant in MODEL_VARIANTS:
            for row in fold.test_rows:
                prediction_started = time.perf_counter()
                try:
                    if variant in {"class_prior_always_hold", "momentum_21d"}:
                        prediction = _baseline_prediction(variant, fold.training_rows, row)
                    else:
                        values = _project_values(row.snapshot.values, variant)
                        prediction = predict(
                            fitted[variant],
                            values,
                            context={
                                "ticker": row.snapshot.ticker,
                                "as_of": row.snapshot.as_of,
                                "target": row.snapshot.target_date,
                                "horizon_trading_days": HORIZON_TRADING_DAYS,
                                "analysis_mode": row.snapshot.analysis_mode,
                                "dataset_version": row.snapshot.dataset_version,
                                "feature_version": row.snapshot.feature_version,
                                "model_version": fitted[variant].model_version,
                            },
                            run_id=f"{run_id}-{fold.fold_id}-{variant}",
                        ).to_dict()
                except Exception:
                    emit_stage09_event(event_sink, boundary="fold_prediction", outcome="fail_closed",
                        kind="error", started_at=prediction_started, run_id=run_id,
                        fold_id=fold.fold_id, model_variant=variant, ticker=row.snapshot.ticker,
                        as_of=row.snapshot.as_of, target=row.snapshot.target_date,
                        horizon=str(HORIZON_TRADING_DAYS), analysis_mode=row.snapshot.analysis_mode,
                        dataset_version=row.snapshot.dataset_version,
                        feature_version=row.snapshot.feature_version,
                        error_code="STAGE09_PREDICTION_REJECTED")
                    raise
                emit_stage09_event(event_sink, boundary="fold_prediction", outcome="success",
                    started_at=prediction_started, run_id=run_id, fold_id=fold.fold_id,
                    model_variant=variant, ticker=row.snapshot.ticker, as_of=row.snapshot.as_of,
                    target=row.snapshot.target_date, horizon=str(HORIZON_TRADING_DAYS),
                    analysis_mode=row.snapshot.analysis_mode, dataset_version=row.snapshot.dataset_version,
                    feature_version=row.snapshot.feature_version,
                    model_version=str(prediction["model_version"]), evidence_count=len(row.snapshot.evidence_ids),
                    degraded_flags=["optional_feature_missing"] if any(row.snapshot.missing_indicators.values()) else [])
                records.append(_record(
                    dataset=dataset, fold=fold, row=row, variant=variant,
                    prediction=prediction, run_id=run_id,
                ))
        emit_stage09_event(event_sink, boundary="run_progress", outcome="progress",
            kind="progress", started_at=time.perf_counter(), run_id=run_id,
            completed_folds=len({item["fold_id"] for item in records}), total_folds=len(folds),
            forecast_count=len(records), variant_count=len(MODEL_VARIANTS))
    row_identity_by_variant = {
        variant: stable_hash(sorted(
            f"{item['fold_id']}|{item['ticker']}|{item['as_of']}"
            for item in records if item["model_variant"] == variant
        )) for variant in MODEL_VARIANTS
    }
    if len(set(row_identity_by_variant.values())) != 1:
        raise AssertionError("model variants did not evaluate identical folds and rows")
    eligible_oos_by_variant = {
        variant: sum(item["model_variant"] == variant for item in records)
        for variant in MODEL_VARIANTS
    }
    if len(set(eligible_oos_by_variant.values())) != 1:
        raise AssertionError("model variants did not retain identical eligible OOS coverage")
    eligible_oos_forecasts = next(iter(eligible_oos_by_variant.values()))
    executed_ticker_count = len({
        item["ticker"] for item in records if item["model_variant"] == MODEL_VARIANTS[0]
    })
    report_level = classify_scientific_report(
        ticker_count=executed_ticker_count,
        monthly_cutoff_count=int(dataset.manifest["monthly_cutoff_count"]),
        eligible_oos_forecasts=eligible_oos_forecasts,
    )
    metrics = {
        variant: compute_forecast_metrics(
            (item for item in records if item["model_variant"] == variant),
            expected_forecasts=expected_per_variant,
            transaction_cost_bps_one_way=transaction_cost_bps_one_way,
            event_sink=event_sink, run_id=run_id, model_variant=variant,
        ) for variant in MODEL_VARIANTS
    }
    manifest = {
        "schema_version": "1.0",
        "backtest_version": BACKTEST_VERSION,
        "run_id": run_id,
        "universe_version": dataset.manifest["universe_version"],
        "universe_manifest_hash": dataset.manifest["universe_manifest_hash"],
        "date_range": {"start": dataset.rows[0].snapshot.as_of, "end": dataset.rows[-1].snapshot.as_of},
        "cutoff_schedule": dataset.manifest["cutoff_schedule"],
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "fold_method": "expanding_window_observable_outcomes_only",
        "minimum_training_cutoffs": minimum_training_cutoffs,
        "fold_count": len(folds),
        "forecast_count_per_variant": expected_per_variant,
        "transaction_cost_bps_one_way": transaction_cost_bps_one_way,
        "gross_results_reported": True,
        "action_thresholds": {
            "buy": {"probability_positive_min": 0.55, "expected_excess_return_min": 0.02},
            "avoid": {"probability_positive_max": 0.45, "expected_excess_return_max": -0.02},
        },
        "action_policy_version": ACTION_POLICY_VERSION,
        "dataset_version": dataset.manifest["dataset_version"],
        "feature_version": dataset.manifest["feature_version"],
        "dataset_manifest_hash": dataset.manifest_hash,
        "eligible_rows_sha256": dataset.manifest["eligible_rows_sha256"],
        "model_variants": list(MODEL_VARIANTS),
        "variant_row_identity_sha256": row_identity_by_variant,
        "scientific_report_level": report_level,
        "scientific_minimums": dataset.manifest["scientific_minimums"],
        "limitations": [
            "Synthetic local fixture results are engineering evidence, not investment-performance evidence.",
            "The report remains PRELIMINARY below 10 tickers, 60 cutoffs, or 500 eligible out-of-sample forecasts.",
        ],
    }
    from evaluation.temporal_leakage import audit_backtest

    provisional = BacktestReport(
        manifest=MappingProxyType(manifest),
        folds=tuple(MappingProxyType(fold.manifest()) for fold in folds),
        records=tuple(MappingProxyType(item) for item in records),
        metrics=MappingProxyType(metrics),
        leakage_audit=MappingProxyType({}),
    )
    audit = audit_backtest(dataset, provisional, event_sink=event_sink, run_id=run_id).to_dict()
    if audit["future_record_count"]:
        raise AssertionError("walk-forward leakage audit failed closed")
    manifest["report_hash"] = stable_hash({
        "manifest": manifest,
        "folds": [fold.manifest() for fold in folds],
        "records": records,
        "metrics": metrics,
        "leakage_audit": audit,
    })
    return BacktestReport(
        manifest=MappingProxyType(manifest),
        folds=provisional.folds,
        records=provisional.records,
        metrics=provisional.metrics,
        leakage_audit=MappingProxyType(audit),
    )


def run_monthly_backtest(
    dataset: PointInTimeDataset, *, run_id: str = "stage09-r1-a1",
    minimum_training_cutoffs: int = MINIMUM_TRAINING_CUTOFFS,
    transaction_cost_bps_one_way: float = TRANSACTION_COST_BPS_ONE_WAY,
    event_sink: Stage09EventSink | None = None,
) -> BacktestReport:
    started_at = time.perf_counter()
    emit_stage09_event(event_sink, boundary="run", outcome="started", started_at=started_at,
        run_id=run_id, row_count=len(dataset.rows), variant_count=len(MODEL_VARIANTS),
        dataset_version=str(dataset.manifest["dataset_version"]),
        feature_version=str(dataset.manifest["feature_version"]))
    try:
        result = _run_monthly_backtest(
            dataset, run_id=run_id, minimum_training_cutoffs=minimum_training_cutoffs,
            transaction_cost_bps_one_way=transaction_cost_bps_one_way, event_sink=event_sink,
        )
    except Exception:
        emit_stage09_event(event_sink, boundary="run", outcome="fail_closed", kind="error",
            started_at=started_at, run_id=run_id, row_count=len(dataset.rows),
            variant_count=len(MODEL_VARIANTS), error_code="STAGE09_RUN_FAILED")
        raise
    emit_stage09_event(event_sink, boundary="run", outcome="degraded"
        if result.manifest["scientific_report_level"] == "PRELIMINARY" else "success",
        started_at=started_at, run_id=run_id, fold_count=len(result.folds),
        forecast_count=len(result.records), variant_count=len(MODEL_VARIANTS),
        dataset_version=str(result.manifest["dataset_version"]),
        feature_version=str(result.manifest["feature_version"]),
        degraded_flags=["preliminary_fixture"]
        if result.manifest["scientific_report_level"] == "PRELIMINARY" else [])
    return result


__all__ = [
    "BACKTEST_VERSION", "MODEL_VARIANTS", "TRANSACTION_COST_BPS_ONE_WAY",
    "BacktestReport", "WalkForwardFold", "build_expanding_folds", "run_monthly_backtest",
]
