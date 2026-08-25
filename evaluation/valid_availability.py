"""Deterministic metric-availability scoring for trace comparisons."""

from __future__ import annotations

from typing import Any


_META_FIELDS = {
    "_inputs", "_period", "source", "selected_source", "source_url",
    "exact_period", "formula", "fallback_reason", "confidence",
    "is_approximation", "semantic_name", "display_name", "interpretation",
    "validation_status", "status", "note", "error", "normalization",
}
_INVALID_STATES = {"UNAVAILABLE", "INVALID", "INCOMPATIBLE_PERIOD"}
_VALID_STATES = {"VALID_PRIMARY", "FALLBACK", "VALID"}
_NON_METRIC_CONTAINERS = {
    "consistency_checks", "data_quality", "financial_facts", "financial_sanity",
    "quality_score", "valuation_input_assessments",
}


def _has_numeric_value(metric: dict[str, Any]) -> bool:
    return any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for key, value in metric.items()
        if key not in _META_FIELDS and not key.startswith("_")
    )


def _period_is_clear(metric: dict[str, Any]) -> bool:
    exact = metric.get("exact_period")
    if isinstance(exact, dict):
        basis = str(exact.get("basis") or "").upper()
        end = exact.get("period_end")
        if basis in {"TTM", "ANNUAL", "DURATION", "SNAPSHOT", "INSTANT", "QUARTERLY"} and end:
            return True
    period = str(metric.get("_period") or "").strip()
    return bool(period) and not period.upper().startswith("MIXED") and "UNKNOWN" not in period.upper()


def _source_is_clear(metric: dict[str, Any]) -> bool:
    source = metric.get("source") or metric.get("selected_source") or metric.get("source_url")
    return bool(source) and str(source).lower() not in {"unavailable", "unknown", "none"}


def valid_metric_inventory(financial_summary: dict[str, Any]) -> dict[str, Any]:
    """Count usable metrics, prioritizing validity plus explicit period/source."""
    details: dict[str, dict[str, Any]] = {}
    for name, raw in financial_summary.items():
        if (
            name in _NON_METRIC_CONTAINERS
            or not isinstance(raw, dict)
            or not _has_numeric_value(raw)
        ):
            continue
        state = str(raw.get("validation_status") or "").upper()
        available = not raw.get("error") and state not in _INVALID_STATES
        period_clear = _period_is_clear(raw)
        source_clear = _source_is_clear(raw)
        valid_state = state in _VALID_STATES or (not state and period_clear and source_clear)
        valid = available and valid_state
        details[name] = {
            "available": available,
            "valid": valid,
            "period_clear": period_clear,
            "source_clear": source_clear,
            "validation_status": state or None,
        }

    return {
        "available_metric_count": sum(x["available"] for x in details.values()),
        "valid_metric_count": sum(x["valid"] for x in details.values()),
        "valid_with_period_and_source_count": sum(
            x["valid"] and x["period_clear"] and x["source_clear"]
            for x in details.values()
        ),
        "metric_count": len(details),
        "metrics": details,
    }


def compare_valid_availability(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Compare quality-aware availability, not merely non-null value counts."""
    baseline = valid_metric_inventory(before)
    candidate = valid_metric_inventory(after)
    before_good = {
        name for name, item in baseline["metrics"].items()
        if item["valid"] and item["period_clear"] and item["source_clear"]
    }
    after_good = {
        name for name, item in candidate["metrics"].items()
        if item["valid"] and item["period_clear"] and item["source_clear"]
    }
    return {
        "before": baseline,
        "after": candidate,
        "valid_with_period_and_source_delta": len(after_good) - len(before_good),
        "newly_valid_with_period_and_source": sorted(after_good - before_good),
        "lost_valid_with_period_and_source": sorted(before_good - after_good),
        "passes_non_regression": before_good <= after_good,
    }
