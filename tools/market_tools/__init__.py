"""Deterministic Stage 07 market package and compact lifecycle boundary."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from core.observability import elapsed_latency_ms, forecast_trace_metadata


QUANT_IMPLEMENTATION_IDENTITY = MappingProxyType({
    "stage": "07", "stage_revision": 1, "attempt": 4,
    "agent": "quant", "implementation_version": "stage07-r1-a4",
})
QUANT_IMPLEMENTATION_VERSION = str(QUANT_IMPLEMENTATION_IDENTITY["implementation_version"])
QUANT_LIFECYCLE_BOUNDARIES = frozenset({
    "cache_validation", "market_bundle", "feature_build", "snapshot_finalization",
    "outcome_materialization", "data_readiness", "model_fit", "prediction",
    "service_forecast",
})
_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TRACE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
_MAX_EVIDENCE_REFERENCES = 64
_COUNT_KEYS = frozenset({
    "input_count", "result_count", "row_count", "eligible_count", "excluded_count",
    "ticker_count", "cutoff_count", "feature_count", "model_count", "cache_count",
    "action_count", "missing_count",
})
_OUTCOMES = frozenset({"success", "degraded", "blocked", "fail_closed"})
_DEGRADED_FLAGS = frozenset({
    "optional_feature_degradation", "required_market_missing", "preliminary_fixture",
    "source_coverage_unverified", "excluded_rows", "insufficient_data",
    "cutoff_policy_distribution_shift",
})
_ERROR_CODES = frozenset({
    "QUANT_CACHE_REJECTED", "QUANT_BUNDLE_REJECTED", "QUANT_FEATURE_REJECTED",
    "QUANT_SNAPSHOT_REJECTED", "QUANT_OUTCOME_REJECTED", "QUANT_READINESS_REJECTED",
    "QUANT_MODEL_REJECTED", "QUANT_PREDICTION_REJECTED", "QUANT_SERVICE_REJECTED",
})
_lifecycle_log = logging.getLogger("quant-lifecycle")


def validate_run_id(value: Any) -> str:
    """Return one opaque whitespace-free run reference or fail closed."""
    if not isinstance(value, str):
        raise TypeError("run_id must be text")
    identity = value.strip()
    if identity != value:
        raise ValueError('run_id must already be canonical')
    if not _RUN_ID.fullmatch(identity) or "PRIVATE_SENTINEL" in identity.upper():
        raise ValueError("run_id must be an opaque compact reference")
    return identity


def validate_evidence_references(values: Any) -> tuple[str, ...]:
    """Validate bounded opaque references; raw text and URLs are never transformed."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("evidence_ids must be a sequence of opaque references")
    if len(values) > _MAX_EVIDENCE_REFERENCES:
        raise ValueError("too many evidence references")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("evidence references must be text")
        reference = value.strip()
        if reference != value:
            raise ValueError('evidence references must already be canonical')
        if (not _REFERENCE.fullmatch(reference)
                or "PRIVATE_SENTINEL" in reference.upper()
                or reference.lower().startswith(("http:", "https:"))):
            raise ValueError("evidence references must be opaque compact IDs")
        result.add(reference)
    return tuple(sorted(result))


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(int(number), 1_000_000)) if math.isfinite(number) else 0


def _safe_trace_context(context: Mapping[str, Any] | object) -> dict[str, Any]:
    """Copy only compact allowlisted identity fields, dropping unsafe values."""
    if isinstance(context, Mapping):
        source = context
    else:
        serializer = getattr(context, "to_dict", None)
        candidate = serializer() if callable(serializer) else {}
        source = candidate if isinstance(candidate, Mapping) else {}
    safe: dict[str, Any] = {}
    for field in (
        "ticker", "as_of", "target", "analysis_mode", "cutoff_policy", "dataset_version",
        "feature_version", "model_version", "schema_version", "calendar_version",
    ):
        value = source.get(field)
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, (date, datetime)):
            value = value.isoformat()
        if (isinstance(value, str) and value == value.strip()
                and _TRACE_TEXT.fullmatch(value)
                and "PRIVATE_SENTINEL" not in value.upper()):
            safe[field] = value
    horizon = source.get("horizon_trading_days")
    if isinstance(horizon, int) and not isinstance(horizon, bool) and horizon > 0:
        safe["horizon_trading_days"] = horizon
    return safe


def quant_lifecycle_event(
    *, context: Mapping[str, Any] | object, boundary: str, started_at: float,
    outcome: str, run_id: str | None = None, evidence_count: int = 0,
    counts: Mapping[str, Any] | None = None,
    degraded_flags: Sequence[str] | None = None, error_code: str | None = None,
) -> dict[str, Any]:
    """Emit measured allowlisted metadata, never source/model/input/output payloads."""
    if boundary not in QUANT_LIFECYCLE_BOUNDARIES:
        raise ValueError("unknown Quant lifecycle boundary")
    identity = validate_run_id(run_id) if run_id is not None else None
    safe_flags = sorted({str(flag) for flag in (degraded_flags or ()) if str(flag) in _DEGRADED_FLAGS})
    event = dict(forecast_trace_metadata(
        _safe_trace_context(context), run_id=identity,
        stage=str(QUANT_IMPLEMENTATION_IDENTITY["stage"]),
        stage_revision=int(QUANT_IMPLEMENTATION_IDENTITY["stage_revision"]),
        attempt=int(QUANT_IMPLEMENTATION_IDENTITY["attempt"]),
        agent=str(QUANT_IMPLEMENTATION_IDENTITY["agent"]),
        latency_ms=elapsed_latency_ms(started_at),
        evidence_count=_bounded_count(evidence_count), degraded_flags=safe_flags,
    ))
    event.update({
        "implementation_version": QUANT_IMPLEMENTATION_VERSION,
        "boundary": boundary,
        "outcome": outcome if outcome in _OUTCOMES else "fail_closed",
    })
    for key, value in (counts or {}).items():
        if key in _COUNT_KEYS:
            event[key] = _bounded_count(value)
    if error_code:
        event["error_code"] = error_code if error_code in _ERROR_CODES else "QUANT_SERVICE_REJECTED"
    _lifecycle_log.info("QUANT_LIFECYCLE %s", json.dumps(event, sort_keys=True, separators=(",", ":")))
    return event


from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION, MarketFeatures, build_market_features
from tools.market_tools.forecast_model import (ACTION_POLICY_VERSION, MODEL_FAMILY_VERSION,
    DeterministicModel, ForecastPrediction, apply_action_policy, fit_deterministic_model, predict)
from tools.market_tools.historical_dataset_builder import (DATASET_BUILDER_VERSION, DataReadiness,
    FeatureSnapshot, LabeledSnapshot, build_data_readiness, build_feature_snapshot,
    labeled_snapshot_from_mapping, materialize_outcome, validate_universe_manifest)
from tools.market_tools.market_data import (MARKET_DATA_VERSION, CorporateAction, MarketDataBundle,
    MarketSeries, PriceBar, build_market_bundle, load_adjusted_ohlcv_cache, rows_to_cache)
from tools.market_tools.market_provider import MarketHistoryFetcher, YahooMarketDataProvider
from tools.market_tools.market_server import MARKET_SERVER_VERSION, MarketForecastService

__all__ = [
    "ACTION_POLICY_VERSION", "DATASET_BUILDER_VERSION", "MARKET_DATA_VERSION",
    "MARKET_FEATURE_VERSION", "MARKET_SERVER_VERSION", "MODEL_FAMILY_VERSION",
    "QUANT_IMPLEMENTATION_IDENTITY", "QUANT_IMPLEMENTATION_VERSION",
    "QUANT_LIFECYCLE_BOUNDARIES", "CorporateAction", "DataReadiness",
    "DeterministicModel", "FeatureSnapshot", "ForecastPrediction", "LabeledSnapshot",
    "MarketDataBundle", "MarketFeatures", "MarketForecastService", "MarketHistoryFetcher",
    "MarketSeries", "PriceBar", "YahooMarketDataProvider",
    "apply_action_policy", "build_data_readiness", "build_feature_snapshot",
    "build_market_bundle", "build_market_features", "fit_deterministic_model",
    "labeled_snapshot_from_mapping", "load_adjusted_ohlcv_cache", "materialize_outcome",
    "predict", "quant_lifecycle_event", "rows_to_cache", "validate_evidence_references",
    "validate_run_id", "validate_universe_manifest",
]
