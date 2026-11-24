"""Transparent deterministic logistic/ridge baseline for 21-session forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError, VersionMismatchError
from core.forecast_contract import CALENDAR_VERSION, HORIZON_TRADING_DAYS, parse_utc_datetime
from tools.market_tools import quant_lifecycle_event, validate_run_id
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION, MarketFeatures
from tools.market_tools.historical_dataset_builder import (DataReadiness, LabeledSnapshot,
    eligible_row_content_hash)


MODEL_FAMILY_VERSION = "logistic-ridge-21d-v1"
ACTION_POLICY_VERSION = "action_policy_v1"
MINIMUM_TRAINING_CUTOFFS = 36
BUY_PROBABILITY = 0.55
BUY_EXCESS_RETURN = 0.02
AVOID_PROBABILITY = 0.45
AVOID_EXCESS_RETURN = -0.02


def apply_action_policy(probability_positive: float, expected_excess_return: float) -> str:
    if probability_positive >= BUY_PROBABILITY and expected_excess_return >= BUY_EXCESS_RETURN:
        return "BUY"
    if probability_positive <= AVOID_PROBABILITY and expected_excess_return <= AVOID_EXCESS_RETURN:
        return "AVOID"
    return "HOLD"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, l2: float = 1.0) -> np.ndarray:
    """Deterministic Newton/IRLS logistic fit with an unpenalized intercept."""
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.diag(np.r_[0.0, np.full(x.shape[1], l2)])
    for _ in range(100):
        probability = _sigmoid(design @ beta)
        weights = np.clip(probability * (1.0 - probability), 1e-9, None)
        gradient = design.T @ (probability - y) + penalty @ beta
        hessian = design.T @ (design * weights[:, None]) + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.diag(np.r_[0.0, np.full(x.shape[1], alpha)])
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


@dataclass(frozen=True, slots=True)
class DeterministicModel:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    logistic_coefficients: tuple[float, ...]
    return_coefficients: tuple[float, ...]
    excess_coefficients: tuple[float, ...]
    residual_interval: tuple[float, float]
    model_version: str
    feature_version: str
    dataset_version: str
    action_policy_version: str
    training_start: str
    training_end: str
    training_rows: int
    training_cutoffs: int
    training_data_hash: str
    readiness_manifest_hash: str
    eligible_rows_hash: str

    def artifact(self) -> dict[str, Any]:
        return {
            "model_family_version": MODEL_FAMILY_VERSION,
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "action_policy_version": self.action_policy_version,
            "horizon_trading_days": HORIZON_TRADING_DAYS,
            "feature_names": list(self.feature_names),
            "imputer": {"strategy": "training_fold_median", "statistics": list(self.medians)},
            "scaler": {"strategy": "training_fold_standard", "means": list(self.means), "scales": list(self.scales)},
            "logistic": {"l2": 1.0, "coefficients_with_intercept": list(self.logistic_coefficients)},
            "ridge_return": {"alpha": 1.0, "coefficients_with_intercept": list(self.return_coefficients)},
            "ridge_excess_return": {"alpha": 1.0, "coefficients_with_intercept": list(self.excess_coefficients)},
            "residual_interval": list(self.residual_interval),
            "training_start": self.training_start, "training_end": self.training_end,
            "training_rows": self.training_rows, "training_cutoffs": self.training_cutoffs,
            "training_data_hash": self.training_data_hash,
            "readiness_manifest_hash": self.readiness_manifest_hash,
            "eligible_rows_hash": self.eligible_rows_hash,
        }


@dataclass(frozen=True, slots=True)
class ForecastPrediction:
    probability_positive: float
    expected_return_21d: float
    expected_excess_return: float
    confidence_interval: tuple[float, float]
    action: str
    confidence: float
    model_version: str
    feature_version: str
    dataset_version: str
    action_policy_version: str
    missing: bool = False
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "probability_positive": self.probability_positive,
            "expected_return_21d": self.expected_return_21d,
            "expected_excess_return": self.expected_excess_return,
            "confidence_interval": list(self.confidence_interval),
            "action": self.action, "confidence": self.confidence,
            "model_version": self.model_version, "feature_version": self.feature_version,
            "dataset_version": self.dataset_version,
            "action_policy_version": self.action_policy_version,
            "missing": self.missing, "degraded": self.degraded,
        }


def _training_matrix(rows: tuple[LabeledSnapshot, ...]) -> tuple[tuple[str, ...], np.ndarray]:
    candidates = sorted({name for row in rows for name in row.snapshot.values})
    selected = tuple(
        name for name in candidates
        if any(row.snapshot.values.get(name) is not None for row in rows)
    )
    if not selected:
        raise DataUnavailableAsOfError("no observed training features")
    matrix = np.asarray([
        [np.nan if row.snapshot.values.get(name) is None else float(row.snapshot.values[name]) for name in selected]
        for row in rows
    ], dtype=float)
    if np.isinf(matrix).any():
        raise ValueError("training features must be finite or missing")
    return selected, matrix


def _fit_deterministic_model(
    *, rows: Iterable[LabeledSnapshot], readiness: DataReadiness,
    train_through: datetime | str | None = None,
) -> DeterministicModel:
    """Fit only after a PASS readiness gate, using only the supplied past fold."""
    readiness.require_pass()
    ordered = tuple(sorted(rows, key=lambda row: (row.snapshot.as_of, row.snapshot.ticker)))
    if not ordered:
        raise DataUnavailableAsOfError("training rows are empty")
    if int(readiness.manifest.get("eligible_forecast_count", -1)) != len(ordered):
        raise ValueError("DATA_READINESS row count does not match training rows")
    expected_rows_hash = readiness.manifest.get("content_hashes", {}).get("eligible_rows_sha256")
    if not expected_rows_hash or eligible_row_content_hash(ordered) != expected_rows_hash:
        raise ValueError("training rows do not match DATA_READINESS eligible-row content hash")
    cutoff_limit = parse_utc_datetime(train_through, field_name="train_through") if train_through is not None else None
    if cutoff_limit is not None and any(parse_utc_datetime(row.snapshot.as_of) > cutoff_limit for row in ordered):
        raise TemporalLeakageError("training fold contains a row after train_through")
    cutoffs = {row.snapshot.as_of for row in ordered}
    if len(cutoffs) < MINIMUM_TRAINING_CUTOFFS:
        raise DataUnavailableAsOfError(f"at least {MINIMUM_TRAINING_CUTOFFS} monthly training cutoffs are required")
    versions = {(row.snapshot.dataset_version, row.snapshot.feature_version) for row in ordered}
    if versions != {(str(readiness.manifest["dataset_version"]), str(readiness.manifest["feature_version"]))}:
        raise VersionMismatchError("training row versions do not match DATA_READINESS")
    y_class = np.asarray([row.direction_positive for row in ordered], dtype=float)
    if set(y_class) != {0.0, 1.0}:
        raise DataUnavailableAsOfError("logistic training requires both direction classes")
    names, raw = _training_matrix(ordered)
    medians = np.nanmedian(raw, axis=0)
    imputed = np.where(np.isnan(raw), medians, raw)
    means = np.mean(imputed, axis=0)
    scales = np.std(imputed, axis=0, ddof=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    x = (imputed - means) / scales
    y_return = np.asarray([row.ticker_total_return for row in ordered], dtype=float)
    y_excess = np.asarray([row.forward_21d_excess_return for row in ordered], dtype=float)
    logistic = _fit_logistic(x, y_class)
    ridge_return = _fit_ridge(x, y_return)
    ridge_excess = _fit_ridge(x, y_excess)
    residuals = y_return - np.column_stack((np.ones(len(x)), x)) @ ridge_return
    interval = tuple(float(item) for item in np.quantile(residuals, (0.05, 0.95)))
    training_payload = [{
        "snapshot_hash": row.snapshot.snapshot_hash,
        "ticker_return": row.ticker_total_return,
        "benchmark_return": row.benchmark_total_return,
        "excess_return": row.forward_21d_excess_return,
        "direction": row.direction_positive,
    } for row in ordered]
    training_hash = _stable_hash(training_payload)
    artifact_seed = {
        "family": MODEL_FAMILY_VERSION, "features": names,
        "medians": medians.round(14).tolist(), "means": means.round(14).tolist(),
        "scales": scales.round(14).tolist(), "logistic": logistic.round(14).tolist(),
        "ridge_return": ridge_return.round(14).tolist(), "ridge_excess": ridge_excess.round(14).tolist(),
        "residual_interval": list(interval), "training_hash": training_hash,
        "readiness_manifest_hash": readiness.manifest_hash,
    }
    model_version = f"{MODEL_FAMILY_VERSION}+sha256:{_stable_hash(artifact_seed)}"
    return DeterministicModel(
        feature_names=names, medians=tuple(float(v) for v in medians),
        means=tuple(float(v) for v in means), scales=tuple(float(v) for v in scales),
        logistic_coefficients=tuple(float(v) for v in logistic),
        return_coefficients=tuple(float(v) for v in ridge_return),
        excess_coefficients=tuple(float(v) for v in ridge_excess),
        residual_interval=interval, model_version=model_version,
        feature_version=str(readiness.manifest["feature_version"]),
        dataset_version=str(readiness.manifest["dataset_version"]),
        action_policy_version=ACTION_POLICY_VERSION,
        training_start=ordered[0].snapshot.as_of, training_end=ordered[-1].snapshot.as_of,
        training_rows=len(ordered), training_cutoffs=len(cutoffs),
        training_data_hash=training_hash, readiness_manifest_hash=readiness.manifest_hash,
        eligible_rows_hash=str(expected_rows_hash),
    )


def fit_deterministic_model(
    *, rows: Iterable[LabeledSnapshot], readiness: DataReadiness,
    train_through: datetime | str | None = None, run_id: str | None = None,
) -> DeterministicModel:
    started_at = time.perf_counter()
    materialized = tuple(rows)
    identity = validate_run_id(run_id) if run_id is not None else None
    lifecycle_context = ({
        "ticker": materialized[-1].snapshot.ticker, "as_of": materialized[-1].snapshot.as_of,
        "target": materialized[-1].snapshot.target_date,
        "horizon_trading_days": materialized[-1].snapshot.horizon_trading_days,
        "analysis_mode": materialized[-1].snapshot.analysis_mode,
        "dataset_version": materialized[-1].snapshot.dataset_version,
        "feature_version": materialized[-1].snapshot.feature_version,
        "schema_version": materialized[-1].snapshot.schema_version,
        "calendar_version": materialized[-1].snapshot.calendar_version,
    } if materialized else {})
    try:
        result = _fit_deterministic_model(rows=materialized, readiness=readiness,
                                          train_through=train_through)
    except Exception:
        quant_lifecycle_event(context=lifecycle_context, boundary="model_fit", started_at=started_at,
            outcome="fail_closed", run_id=identity, counts={"input_count": len(materialized)},
            error_code="QUANT_MODEL_REJECTED")
        raise
    model_context = {**lifecycle_context, "model_version": result.model_version}
    quant_lifecycle_event(context=model_context, boundary="model_fit", started_at=started_at,
        outcome="success", run_id=identity,
        counts={"row_count": result.training_rows, "cutoff_count": result.training_cutoffs,
                "feature_count": len(result.feature_names), "model_count": 3})
    return result


def _prediction_vector(model: DeterministicModel, features: MarketFeatures | Mapping[str, Any]) -> np.ndarray:
    if isinstance(features, MarketFeatures):
        if features.feature_version != model.feature_version or features.dataset_version != model.dataset_version:
            raise VersionMismatchError("prediction feature versions do not match model")
        values = features.numeric_vector()
        missing_required = features.missing
    elif isinstance(features, Mapping):
        values = features
        missing_required = False
    else:
        raise TypeError("features must be MarketFeatures or a numeric mapping")
    if missing_required:
        raise DataUnavailableAsOfError("required market feature coverage is missing")
    row: list[float] = []
    for index, name in enumerate(model.feature_names):
        value = values.get(name)
        if value is None:
            row.append(model.medians[index])
        else:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("prediction features must be finite or missing")
            row.append(number)
    return (np.asarray(row) - np.asarray(model.means)) / np.asarray(model.scales)


def _predict(model: DeterministicModel, features: MarketFeatures | Mapping[str, Any]) -> ForecastPrediction:
    x = _prediction_vector(model, features)
    design = np.r_[1.0, x]
    probability = float(_sigmoid(np.asarray([design @ np.asarray(model.logistic_coefficients)]))[0])
    expected_return = float(design @ np.asarray(model.return_coefficients))
    expected_excess = float(design @ np.asarray(model.excess_coefficients))
    low = expected_return + model.residual_interval[0]
    high = expected_return + model.residual_interval[1]
    action = apply_action_policy(probability, expected_excess)
    confidence = max(0.0, min(1.0, abs(probability - 0.5) * 2.0))
    return ForecastPrediction(
        probability_positive=probability, expected_return_21d=expected_return,
        expected_excess_return=expected_excess, confidence_interval=(low, high),
        action=action, confidence=confidence, model_version=model.model_version,
        feature_version=model.feature_version, dataset_version=model.dataset_version,
        action_policy_version=model.action_policy_version,
    )


def predict(
    model: DeterministicModel, features: MarketFeatures | Mapping[str, Any], *,
    context: Mapping[str, Any] | object | None = None, run_id: str | None = None,
) -> ForecastPrediction:
    started_at = time.perf_counter()
    identity = validate_run_id(run_id) if run_id is not None else None
    lifecycle_context: Mapping[str, Any] | object
    if context is not None:
        lifecycle_context = context
    elif isinstance(features, MarketFeatures):
        lifecycle_context = {
            "ticker": features.ticker, "as_of": features.as_of,
            "dataset_version": features.dataset_version,
            "feature_version": features.feature_version, "model_version": model.model_version,
            "schema_version": features.schema_version,
            "calendar_version": features.calendar_version,
        }
    else:
        lifecycle_context = {"dataset_version": model.dataset_version,
                             "feature_version": model.feature_version,
                             "model_version": model.model_version,
                             "calendar_version": CALENDAR_VERSION}
    try:
        result = _predict(model, features)
    except Exception:
        quant_lifecycle_event(context=lifecycle_context, boundary="prediction", started_at=started_at,
            outcome="fail_closed", run_id=identity, error_code="QUANT_PREDICTION_REJECTED")
        raise
    evidence_count = len(features.evidence_ids) if isinstance(features, MarketFeatures) else 0
    quant_lifecycle_event(context=lifecycle_context, boundary="prediction", started_at=started_at,
        outcome="success", run_id=identity, evidence_count=evidence_count,
        counts={"result_count": 1})
    return result


__all__ = [
    "ACTION_POLICY_VERSION", "AVOID_EXCESS_RETURN", "AVOID_PROBABILITY",
    "BUY_EXCESS_RETURN", "BUY_PROBABILITY", "MINIMUM_TRAINING_CUTOFFS",
    "MODEL_FAMILY_VERSION", "DeterministicModel", "ForecastPrediction",
    "apply_action_policy", "fit_deterministic_model", "predict",
]
