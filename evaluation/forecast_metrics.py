"""Pure deterministic forecast and investment metrics for walk-forward results."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from evaluation.point_in_time_dataset import Stage09EventSink, emit_stage09_event


METRICS_VERSION = "stage09-forecast-metrics-v1"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2:
        return None
    left, right = _rank(actual), _rank(predicted)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return None
    ranks = _rank(probabilities)
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    total = len(labels)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (probabilities >= low) & ((probabilities < high) if index < bins - 1 else (probabilities <= high))
        if selected.any():
            result += float(selected.sum()) / total * abs(float(labels[selected].mean()) - float(probabilities[selected].mean()))
    return result


def _direction(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    predicted = (probabilities >= 0.5).astype(int)
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    recalls = []
    if tp + fn:
        recalls.append(tp / (tp + fn))
    if tn + fp:
        recalls.append(tn / (tn + fp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "balanced_accuracy": float(sum(recalls) / len(recalls)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "roc_auc": _roc_auc(labels, probabilities),
    }


def _investment(records: list[Mapping[str, Any]], transaction_cost: float) -> dict[str, float | None]:
    positions = {"BUY": 1.0, "AVOID": -1.0, "HOLD": 0.0}
    ordered = sorted(records, key=lambda item: (str(item["ticker"]), str(item["as_of"])))
    previous: dict[str, float] = defaultdict(float)
    gross_returns: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []
    active_hits: list[float] = []
    for item in ordered:
        ticker = str(item["ticker"])
        position = positions[str(item["action"])]
        actual = float(item["actual_excess_return"])
        turnover = abs(position - previous[ticker])
        gross = position * actual
        net = gross - turnover * transaction_cost
        turnovers.append(turnover)
        gross_returns.append(gross)
        net_returns.append(net)
        if position:
            active_hits.append(1.0 if gross > 0.0 else 0.0)
        previous[ticker] = position
    net_array = np.asarray(net_returns, dtype=float)
    wealth = np.cumprod(1.0 + net_array)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdowns = np.r_[1.0, wealth] / peaks - 1.0
    std = float(np.std(net_array, ddof=1)) if len(net_array) > 1 else 0.0
    return {
        "hit_rate": float(np.mean(active_hits)) if active_hits else None,
        "gross_excess_return_mean": float(np.mean(gross_returns)),
        "net_excess_return_mean": float(np.mean(net_returns)),
        "transaction_cost_mean": float(np.mean(np.asarray(turnovers) * transaction_cost)),
        "turnover_mean": float(np.mean(turnovers)),
        "max_drawdown_net": float(abs(np.min(drawdowns))),
        "sharpe_net_annualized": float(np.mean(net_array) / std * math.sqrt(12.0)) if std > 0.0 else None,
    }


def _compute_forecast_metrics(
    records: Iterable[Mapping[str, Any]], *, expected_forecasts: int,
    transaction_cost_bps_one_way: float = 10.0,
) -> dict[str, Any]:
    rows = list(records)
    if expected_forecasts <= 0:
        raise ValueError("expected_forecasts must be positive")
    if not rows:
        raise ValueError("at least one forecast record is required")
    labels = np.asarray([int(row["actual_direction_positive"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row["probability_positive"]) for row in rows], dtype=float)
    actual = np.asarray([float(row["actual_excess_return"]) for row in rows], dtype=float)
    predicted = np.asarray([float(row["expected_excess_return"]) for row in rows], dtype=float)
    if not np.isfinite(probabilities).all() or not ((0.0 <= probabilities) & (probabilities <= 1.0)).all():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("returns must be finite")
    error = predicted - actual
    return {
        "metrics_version": METRICS_VERSION,
        "forecast_count": len(rows),
        "direction": _direction(labels, probabilities),
        "probability": {
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "ece_10_bin": _ece(labels, probabilities, bins=10),
        },
        "amount": {
            "mae_excess_return": float(np.mean(np.abs(error))),
            "rmse_excess_return": float(np.sqrt(np.mean(error ** 2))),
            "spearman_ic": _spearman(actual, predicted),
        },
        "investment": _investment(rows, transaction_cost_bps_one_way / 10_000.0),
        "coverage": {
            "eligible_forecasts": len(rows),
            "expected_forecasts": expected_forecasts,
            "fraction": float(len(rows) / expected_forecasts),
        },
    }


def compute_forecast_metrics(
    records: Iterable[Mapping[str, Any]], *, expected_forecasts: int,
    transaction_cost_bps_one_way: float = 10.0,
    event_sink: Stage09EventSink | None = None, run_id: str = "stage09-r1-a1",
    model_variant: str = "unspecified",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    materialized = list(records)
    try:
        result = _compute_forecast_metrics(
            materialized, expected_forecasts=expected_forecasts,
            transaction_cost_bps_one_way=transaction_cost_bps_one_way,
        )
    except Exception:
        emit_stage09_event(event_sink, boundary="metric_aggregation", outcome="fail_closed",
            kind="error", started_at=started_at, run_id=run_id, model_variant=model_variant,
            metrics_version=METRICS_VERSION, forecast_count=len(materialized),
            error_code="STAGE09_METRICS_REJECTED")
        raise
    degraded = result["coverage"]["fraction"] < 1.0
    emit_stage09_event(event_sink, boundary="metric_aggregation",
        outcome="degraded" if degraded else "success", started_at=started_at,
        run_id=run_id, model_variant=model_variant, metrics_version=METRICS_VERSION,
        forecast_count=len(materialized),
        degraded_flags=["incomplete_coverage"] if degraded else [])
    return result


__all__ = ["METRICS_VERSION", "compute_forecast_metrics"]
