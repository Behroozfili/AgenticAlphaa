"""Deterministic point-in-time sentiment features for monthly forecasts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import pstdev
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from core.forecast_contract import ForecastContext, parse_utc_datetime
from core.observability import elapsed_latency_ms, forecast_trace_metadata
from tools.sentiment_tools.fear_greed_calculator import FEAR_GREED_MODEL_VERSION
from tools.sentiment_tools.finbert_analyzer import FINBERT_MODEL_VERSION
from tools.sentiment_tools.vader_scorer import VADER_MODEL_VERSION


SENTIMENT_FEATURE_VERSION = "monthly-sentiment-features-v1"
SENTIMENT_IMPLEMENTATION_IDENTITY = MappingProxyType({
    "stage": "06",
    "stage_revision": 1,
    "attempt": 4,
    "agent": "sentiment",
    "implementation_version": "stage06-r1-a4",
})
SENTIMENT_IMPLEMENTATION_VERSION = str(
    SENTIMENT_IMPLEMENTATION_IDENTITY["implementation_version"]
)
SENTIMENT_WINDOWS = (7, 14, 30)
SENTIMENT_LIFECYCLE_BOUNDARIES = frozenset({
    "whole_sentiment_run", "brain_plan", "point_in_time_retrieval",
    "finbert_inference", "vader_scoring", "fear_greed_fusion",
    "brain_analyze", "feature_aggregation_and_pre_shared_state",
})
_LIFECYCLE_ROUTE = {
    "whole_sentiment_run": ("sentiment_agent", "run"),
    "brain_plan": ("sentiment_agent", "plan"),
    "point_in_time_retrieval": ("sentiment_mcp", "retrieve"),
    "finbert_inference": ("sentiment_mcp", "finbert"),
    "vader_scoring": ("sentiment_mcp", "vader"),
    "fear_greed_fusion": ("sentiment_mcp", "fusion"),
    "brain_analyze": ("sentiment_agent", "analyze"),
    "feature_aggregation_and_pre_shared_state": ("sentiment_adapter", "build_features"),
}
_LIFECYCLE_COUNT_KEYS = frozenset({
    "input_count", "result_count", "evidence_count", "eligible_count",
    "rejected_count", "missing_count", "model_count", "loop_count",
})
_LIFECYCLE_OUTCOMES = frozenset({"success", "unavailable", "degraded", "fail_closed"})
_LIFECYCLE_ERROR_CODES = frozenset({
    "SENTIMENT_TOOL_FAIL_CLOSED", "SENTIMENT_TOOL_DATA_UNAVAILABLE",
    "SENTIMENT_BRAIN_PLAN_FAILED", "SENTIMENT_BRAIN_ANALYZE_FAILED",
    "SENTIMENT_NO_ELIGIBLE_EVIDENCE", "SENTIMENT_OPERATION_FAIL_CLOSED",
})
_lifecycle_log = logging.getLogger("sentiment-lifecycle")


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(numeric):
        return 0
    return max(0, min(int(numeric), 1_000_000))


def sentiment_lifecycle_event(
    *, forecast_context: ForecastContext | Mapping[str, Any], run_id: str,
    boundary: str, started_at: float, outcome: str,
    counts: Mapping[str, Any] | None = None, missing: bool = False,
    degraded: bool = False, degraded_flags: Sequence[str] | None = None,
    error_code: str | None = None, error_type: str | None = None,
    model_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Emit one measured, compact, payload-free Stage 06 lifecycle event."""
    if boundary not in SENTIMENT_LIFECYCLE_BOUNDARIES:
        raise ValueError("unknown sentiment lifecycle boundary")
    context, identity = require_sentiment_context(forecast_context, run_id)
    provider, operation = _LIFECYCLE_ROUTE[boundary]
    safe_outcome = outcome if outcome in _LIFECYCLE_OUTCOMES else "fail_closed"
    safe_flags = sorted({
        str(flag) for flag in (degraded_flags or ())
        if str(flag) in {
            "missing_evidence", "missing_finbert", "missing_vader",
            "missing_fusion", "low_coverage", "tool_unavailable",
            "analysis_unavailable", "planning_degraded", "feature_degraded",
        }
    })
    event = dict(forecast_trace_metadata(
        context, run_id=identity,
        stage=str(SENTIMENT_IMPLEMENTATION_IDENTITY["stage"]),
        stage_revision=int(SENTIMENT_IMPLEMENTATION_IDENTITY["stage_revision"]),
        attempt=int(SENTIMENT_IMPLEMENTATION_IDENTITY["attempt"]),
        agent=str(SENTIMENT_IMPLEMENTATION_IDENTITY["agent"]),
        latency_ms=elapsed_latency_ms(started_at),
        evidence_count=_bounded_count((counts or {}).get("evidence_count", 0)),
        degraded_flags=safe_flags,
    ))
    event.update({
        "implementation_version": SENTIMENT_IMPLEMENTATION_VERSION,
        "boundary": boundary, "provider": provider, "operation": operation,
        "outcome": safe_outcome, "missing": bool(missing),
        "degraded": bool(degraded),
    })
    for key, value in (counts or {}).items():
        if key in _LIFECYCLE_COUNT_KEYS:
            event[key] = _bounded_count(value)
    if error_code:
        event["error_code"] = (
            error_code if error_code in _LIFECYCLE_ERROR_CODES
            else "SENTIMENT_OPERATION_FAIL_CLOSED"
        )
    if error_type:
        event["error_type"] = (
            error_type if error_type in {
                "RuntimeError", "TimeoutError", "ConnectionError", "ValueError",
                "LookupError", "JSONDecodeError", "Exception",
            } else "Exception"
        )
    expected_versions = {
        "finbert": FINBERT_MODEL_VERSION,
        "vader": VADER_MODEL_VERSION,
        "fusion": FEAR_GREED_MODEL_VERSION,
    }
    safe_versions = {
        key: expected_versions[key]
        for key, value in (model_versions or {}).items()
        if key in expected_versions and value == expected_versions[key]
    }
    if safe_versions:
        event["sentiment_model_versions"] = dict(sorted(safe_versions.items()))
    _lifecycle_log.info(
        "SENTIMENT_LIFECYCLE %s",
        json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    return event


def require_sentiment_context(
    forecast_context: ForecastContext | Mapping[str, Any] | None,
    run_id: str | None,
    *,
    ticker: str | None = None,
) -> tuple[ForecastContext, str]:
    """Validate the immutable identity required by every Stage 06 run."""
    if forecast_context is None:
        raise ValueError("forecast_context is required for every sentiment execution")
    context = (
        forecast_context
        if isinstance(forecast_context, ForecastContext)
        else ForecastContext.from_dict(dict(forecast_context))
    )
    identity = str(run_id or "").strip()
    if not identity or len(identity) > 160 or "\n" in identity or "\r" in identity:
        raise ValueError("run_id must be compact non-empty text")
    if ticker is not None and str(ticker).strip().upper() != context.ticker:
        raise ValueError("ticker does not match ForecastContext.ticker")
    return context, identity


def valid_composite_score(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score) and -1.0 <= score <= 1.0


def normalise_aspect_names(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return {"<invalid-causal-aspects-used>"}
    return {str(item).strip().lower() for item in value if str(item).strip()}


def causal_audit(
    analysis: dict[str, Any], causal_aspects_supported: list[str],
) -> dict[str, Any]:
    claimed = normalise_aspect_names(analysis.get("causal_aspects_used"))
    supported = normalise_aspect_names(causal_aspects_supported)
    unsupported = sorted(claimed - supported)
    return {
        "claimed_causal_aspects": sorted(claimed),
        "supported_causal_aspects": sorted(supported),
        "unsupported_causal_aspects": unsupported,
        "causal_mismatch_detected": bool(unsupported),
    }


def attempt_rank(attempt: dict[str, Any]) -> tuple[Any, ...]:
    coverage = attempt.get("retrieval_stats", {}) or {}
    fear_greed = ((attempt.get("model_outputs", {}) or {}).get("fear_greed", {}) or {})
    status_rank = {"ERROR": 0, "NO_DATA": 1, "LOW_COVERAGE": 2, "VALID": 3}
    return (
        bool(attempt.get("quality_passed")),
        status_rank.get(str(attempt.get("status")), -1),
        valid_composite_score(fear_greed.get("score")),
        bool(attempt.get("execution_success")),
        int(coverage.get("unique_domains") or 0),
        int(coverage.get("unique_documents") or 0),
        float(fear_greed.get("confidence") or 0.0),
        -int(attempt.get("attempt_number") or 0),
    )


def _canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    query = urlencode(sorted(
        (key, item) for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid", "ref", "source"}
    ))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", query, ""))


def _evidence_id(metadata: Mapping[str, Any], *, dataset_version: str) -> str:
    for key in ("evidence_id", "document_id", "url_hash", "source_id", "content_hash", "document_key"):
        value = str(metadata.get(key) or "").strip()
        if value:
            digest = hashlib.sha256(
                f"{dataset_version}|{key}|{value}".encode("utf-8")
            ).hexdigest()
            return f"sentiment:{digest}"
    canonical = _canonical_url(metadata.get("url"))
    if canonical:
        digest = hashlib.sha256(
            f"{dataset_version}|url|{canonical}".encode("utf-8")
        ).hexdigest()
        return f"sentiment:{digest}"
    stable = "|".join(str(metadata.get(key) or "").strip() for key in (
        "ticker", "published_at", "source_type", "title",
    ))
    digest = hashlib.sha256(
        f"{dataset_version}|metadata|{stable}".encode("utf-8")
    ).hexdigest()
    return f"sentiment:{digest}"


def select_point_in_time_evidence(
    *,
    texts: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    forecast_context: ForecastContext | Mapping[str, Any],
    run_id: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter and deduplicate evidence before scoring; never guess timestamps."""
    context, _ = require_sentiment_context(forecast_context, run_id)
    eligible: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    lower_bound = context.as_of - timedelta(days=max(SENTIMENT_WINDOWS))
    for index, text in enumerate(texts):
        item = metadata[index] if index < len(metadata) and isinstance(metadata[index], Mapping) else {}
        temporal_values: list[datetime] = []
        temporal_invalid = False
        for field_name in ("published_at", "effective_from"):
            raw_value = item.get(field_name)
            if raw_value is None:
                continue
            try:
                parsed_value = parse_utc_datetime(
                    raw_value, field_name=field_name,
                )
            except (TypeError, ValueError):
                errors.append(f"{field_name.upper()}_INVALID")
                temporal_invalid = True
                break
            if parsed_value > context.as_of:
                errors.append("FUTURE_EVIDENCE_REJECTED")
                temporal_invalid = True
                break
            temporal_values.append(parsed_value)
        if temporal_invalid:
            continue
        if not temporal_values:
            errors.append("EVIDENCE_TIMESTAMP_MISSING")
            continue
        observed = max(temporal_values)
        if observed < lower_bound:
            errors.append("OUTSIDE_30D_WINDOW_REJECTED")
            continue
        effective_to = item.get("effective_to")
        if effective_to is not None:
            try:
                if parse_utc_datetime(effective_to, field_name="effective_to") <= context.as_of:
                    errors.append("EXPIRED_EVIDENCE_REJECTED")
                    continue
            except (TypeError, ValueError):
                errors.append("EFFECTIVE_TO_INVALID")
                continue
        item_ticker = str(item.get("ticker") or context.ticker).strip().upper()
        if item_ticker != context.ticker:
            errors.append("TICKER_MISMATCH_REJECTED")
            continue
        item_dataset = str(item.get("dataset_version") or context.dataset_version).strip()
        if item_dataset != context.dataset_version:
            errors.append("DATASET_VERSION_MISMATCH_REJECTED")
            continue
        evidence_id = _evidence_id(item, dataset_version=context.dataset_version)
        if evidence_id in seen:
            errors.append("DUPLICATE_EVIDENCE_REJECTED")
            continue
        seen.add(evidence_id)
        canonical = _canonical_url(item.get("url"))
        domain = urlsplit(canonical).netloc.lower() if canonical else str(item.get("domain") or "").lower()
        eligible.append({
            "index": index,
            "text": str(text),
            "published_at": observed,
            "evidence_id": evidence_id,
            "domain": domain[4:] if domain.startswith("www.") else domain,
            "source_type": str(item.get("source_type") or "unknown"),
        })
    return eligible, sorted(set(errors))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _item_scores(
    eligible: Sequence[Mapping[str, Any]],
    finbert_result: Mapping[str, Any],
    vader_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fb_chunks = finbert_result.get("chunk_scores") if isinstance(finbert_result.get("chunk_scores"), list) else []
    vd_chunks = vader_result.get("chunk_scores") if isinstance(vader_result.get("chunk_scores"), list) else []
    if _result_status(finbert_result, available=bool(fb_chunks)) in _UNAVAILABLE_MODEL_STATUSES:
        fb_chunks = []
    if _result_status(vader_result, available=bool(vd_chunks)) in _UNAVAILABLE_MODEL_STATUSES:
        vd_chunks = []
    scored: list[dict[str, Any]] = []
    for item in eligible:
        index = int(item["index"])
        fb = fb_chunks[index] if index < len(fb_chunks) and isinstance(fb_chunks[index], Mapping) else {}
        vd = vd_chunks[index] if index < len(vd_chunks) and isinstance(vd_chunks[index], Mapping) else {}
        bullish, bearish, compound = _number(fb.get("bullish_prob")), _number(fb.get("bearish_prob")), _number(vd.get("compound"))
        components: list[tuple[float, float]] = []
        if bullish is not None and bearish is not None:
            components.append((bullish - bearish, 0.65))
        if compound is not None:
            components.append((compound, 0.35))
        if not components:
            continue
        weight = sum(component_weight for _, component_weight in components)
        score = sum(value * component_weight for value, component_weight in components) / weight
        scored.append({**item, "score": max(-1.0, min(1.0, score))})
    return scored


_UNAVAILABLE_MODEL_STATUSES = frozenset({
    "ERROR", "FAILED", "NO_DATA", "DATA_UNAVAILABLE", "UNAVAILABLE", "MISSING",
    "UNKNOWN",
})
_ERROR_MODEL_STATUSES = frozenset({"ERROR", "FAILED"})
_MODEL_STATUS_ALLOWLIST = frozenset({
    "VALID", "LOW_COVERAGE", "ERROR", "FAILED", "NO_DATA",
    "DATA_UNAVAILABLE", "UNAVAILABLE", "MISSING",
})
_SELECTED_STATUS_ALLOWLIST = frozenset({
    "VALID", "LOW_COVERAGE", "ERROR", "NO_DATA",
})
_POLARITY_ALLOWLIST = frozenset({"BULLISH", "BEARISH", "NEUTRAL"})


def _canonical_status(value: Any, *, allowed: frozenset[str], default: str) -> str:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        return "UNKNOWN"
    status = value.strip().upper()
    if (
        not status
        or len(status) > 32
        or not status.isascii()
        or any(not (character.isalnum() or character == "_") for character in status)
        or status not in allowed
    ):
        return "UNKNOWN"
    return status


def _result_status(result: Mapping[str, Any], *, available: bool) -> str:
    return _canonical_status(
        result.get("status"),
        allowed=_MODEL_STATUS_ALLOWLIST,
        default="VALID" if available else "NO_DATA",
    )


def _selected_attempt_status(selected: Mapping[str, Any]) -> str:
    return _canonical_status(
        selected.get("status"), allowed=_SELECTED_STATUS_ALLOWLIST,
        default="UNKNOWN",
    )


def _canonical_polarity(value: Any) -> tuple[str | None, bool]:
    if value is None or value == "":
        return None, False
    if not isinstance(value, str):
        return None, True
    polarity = value.strip().upper()
    if (
        not polarity
        or len(polarity) > 16
        or not polarity.isascii()
        or polarity not in _POLARITY_ALLOWLIST
    ):
        return None, True
    return polarity, False


def canonical_sentiment_model_status(
    result: Mapping[str, Any], *, available: bool,
) -> str:
    """Return a fixed provider model-health status safe for State/telemetry."""
    return _result_status(result, available=available)


def canonical_sentiment_polarity(value: Any) -> tuple[str | None, bool]:
    """Return canonical polarity and whether an explicit value was invalid."""
    return _canonical_polarity(value)


def _model_contribution_counts(
    eligible: Sequence[Mapping[str, Any]],
    finbert_result: Mapping[str, Any],
    vader_result: Mapping[str, Any],
) -> tuple[int, int]:
    """Count eligible evidence with a usable contribution from each model."""
    fb_chunks = (
        finbert_result.get("chunk_scores")
        if isinstance(finbert_result.get("chunk_scores"), list) else []
    )
    vd_chunks = (
        vader_result.get("chunk_scores")
        if isinstance(vader_result.get("chunk_scores"), list) else []
    )
    finbert_count = 0
    vader_count = 0
    for item in eligible:
        index = int(item["index"])
        fb = (
            fb_chunks[index]
            if index < len(fb_chunks) and isinstance(fb_chunks[index], Mapping) else {}
        )
        vd = (
            vd_chunks[index]
            if index < len(vd_chunks) and isinstance(vd_chunks[index], Mapping) else {}
        )
        if _number(fb.get("bullish_prob")) is not None and _number(fb.get("bearish_prob")) is not None:
            finbert_count += 1
        if _number(vd.get("compound")) is not None:
            vader_count += 1
    return finbert_count, vader_count


def _safe_coverage_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, count)


@dataclass(frozen=True, slots=True)
class MonthlySentimentFeatures:
    aggregate_7d: float | None
    aggregate_14d: float | None
    aggregate_30d: float | None
    momentum_7d_vs_30d: float | None
    dispersion_30d: float | None
    volume_7d: int
    volume_14d: int
    volume_30d: int
    source_diversity_30d: int
    confidence: float | None
    polarity: str | None
    availability: Mapping[str, bool]
    missing_flags: tuple[str, ...]
    degraded_flags: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    model_versions: Mapping[str, str]
    model_statuses: Mapping[str, str]
    dataset_version: str
    feature_version: str
    run_id: str

    def to_state_slot(self) -> dict[str, Any]:
        values = {
            "aggregate_7d": self.aggregate_7d,
            "aggregate_14d": self.aggregate_14d,
            "aggregate_30d": self.aggregate_30d,
            "momentum_7d_vs_30d": self.momentum_7d_vs_30d,
            "dispersion_30d": self.dispersion_30d,
            "volume_7d": self.volume_7d,
            "volume_14d": self.volume_14d,
            "volume_30d": self.volume_30d,
            "source_diversity_30d": self.source_diversity_30d,
            "confidence": self.confidence,
            "polarity": self.polarity,
            "availability": dict(self.availability),
            "model_versions": dict(self.model_versions),
            "model_statuses": dict(self.model_statuses),
            "dataset_version": self.dataset_version,
            "run_id": self.run_id,
        }
        return {
            "version": self.feature_version,
            "values": values,
            "evidence_ids": list(self.evidence_ids),
            "missing": bool(self.missing_flags),
            "degraded": bool(self.degraded_flags),
            "errors": list(self.degraded_flags),
            "missing_flags": list(self.missing_flags),
            "degraded_flags": list(self.degraded_flags),
        }


def build_monthly_sentiment_features(
    *,
    texts: Sequence[str],
    sources_metadata: Sequence[Mapping[str, Any]],
    finbert_result: Mapping[str, Any],
    vader_result: Mapping[str, Any],
    fear_greed_result: Mapping[str, Any],
    selected_attempt: Mapping[str, Any] | None = None,
    forecast_context: ForecastContext | Mapping[str, Any] | None,
    run_id: str | None,
) -> MonthlySentimentFeatures:
    """Revalidate evidence after scoring and build a compact 7/14/30-day slot."""
    started_at = time.perf_counter()
    context, identity = require_sentiment_context(forecast_context, run_id)
    eligible, rejections = select_point_in_time_evidence(
        texts=texts, metadata=sources_metadata,
        forecast_context=context, run_id=identity,
    )
    scored = _item_scores(eligible, finbert_result, vader_result)
    aggregates: dict[int, float | None] = {}
    volumes: dict[int, int] = {}
    for days in SENTIMENT_WINDOWS:
        start = context.as_of - timedelta(days=days)
        values = [float(item["score"]) for item in scored if item["published_at"] >= start]
        volumes[days] = len(values)
        aggregates[days] = round(sum(values) / len(values), 6) if values else None
    thirty_scores = [float(item["score"]) for item in scored]
    dispersion = round(pstdev(thirty_scores), 6) if thirty_scores else None
    momentum = (
        round(float(aggregates[7]) - float(aggregates[30]), 6)
        if aggregates[7] is not None and aggregates[30] is not None else None
    )
    finbert_count, vader_count = _model_contribution_counts(
        eligible, finbert_result, vader_result,
    )
    finbert_status = _result_status(finbert_result, available=finbert_count > 0)
    vader_status = _result_status(vader_result, available=vader_count > 0)
    fusion_score_valid = valid_composite_score(fear_greed_result.get("score"))
    fusion_status = _result_status(fear_greed_result, available=fusion_score_valid)
    finbert_available = finbert_count > 0 and finbert_status not in _UNAVAILABLE_MODEL_STATUSES
    vader_available = vader_count > 0 and vader_status not in _UNAVAILABLE_MODEL_STATUSES
    fusion_available = fusion_score_valid and fusion_status not in _UNAVAILABLE_MODEL_STATUSES
    confidence = _number(fear_greed_result.get("confidence"))
    confidence = (
        max(0.0, min(1.0, confidence))
        if confidence is not None and fusion_available else None
    )
    polarity, polarity_invalid = _canonical_polarity(fear_greed_result.get("polarity"))
    polarity_invalid = polarity_invalid or fear_greed_result.get("polarity_invalid") is True
    if not fusion_available:
        polarity = None

    selected = selected_attempt if isinstance(selected_attempt, Mapping) else {}
    coverage = selected.get("retrieval_stats") if isinstance(selected.get("retrieval_stats"), Mapping) else {}
    if not coverage:
        candidate_coverage = (
            fear_greed_result.get("retrieval_stats")
            or fear_greed_result.get("coverage")
            or {}
        )
        coverage = candidate_coverage if isinstance(candidate_coverage, Mapping) else {}
    unique_documents = _safe_coverage_count(coverage.get("unique_documents"))
    unique_domains = _safe_coverage_count(coverage.get("unique_domains"))
    coverage_known = unique_documents is not None and unique_domains is not None
    coverage_quality = bool(
        coverage_known and unique_documents >= 3 and unique_domains >= 2
    )
    selected_status = _selected_attempt_status(selected)
    quality_known = (
        "quality_passed" in selected
        and isinstance(selected.get("quality_passed"), bool)
        and selected_status != "UNKNOWN"
    )
    selected_quality = bool(
        quality_known
        and selected.get("quality_passed")
        and selected_status == "VALID"
    )

    availability = {
        **{f"window_{days}d": aggregates[days] is not None for days in SENTIMENT_WINDOWS},
        "finbert": finbert_available,
        "vader": vader_available,
        "fusion": fusion_available,
        "selected_attempt_quality": selected_quality,
        "coverage_quality": coverage_quality,
        "polarity": polarity is not None,
    }
    # The 30-day aggregate is the required monthly signal. The 7/14-day
    # windows are optional momentum enrichments; availability and None values
    # preserve those gaps for deterministic model missing indicators.
    missing = {"window_30d"} if not availability["window_30d"] else set()
    for model_name in ("finbert", "vader", "fusion"):
        if not availability[model_name]:
            missing.add(model_name)
    if not quality_known:
        missing.add("selected_attempt_quality")
    if selected_status == "UNKNOWN":
        missing.add("selected_attempt_status")
    if not coverage_known:
        missing.add("coverage_quality")
    if polarity is None:
        missing.add("polarity")
    missing_flags = tuple(sorted(missing))
    degraded = set(rejections)
    if len(scored) < len(eligible):
        degraded.add("UNSCORED_EVIDENCE")
    if not eligible:
        degraded.add("NO_ELIGIBLE_POINT_IN_TIME_EVIDENCE")
    for model_name, status, available in (
        ("FINBERT", finbert_status, finbert_available),
        ("VADER", vader_status, vader_available),
        ("FUSION", fusion_status, fusion_available),
    ):
        if not available:
            suffix = "ERROR" if status in _ERROR_MODEL_STATUSES else "UNAVAILABLE"
            degraded.add(f"{model_name}_{suffix}")
            if status == "UNKNOWN":
                degraded.add(f"{model_name}_STATUS_UNKNOWN")
    if finbert_available != vader_available:
        degraded.add("PARTIAL_MODEL_COVERAGE")
    if not quality_known:
        degraded.add("SELECTED_ATTEMPT_QUALITY_UNKNOWN")
    elif not selected_quality:
        degraded.add("SELECTED_ATTEMPT_QUALITY_FAILED")
    if not coverage_known:
        degraded.add("COVERAGE_UNKNOWN")
    elif not coverage_quality or selected_status == "LOW_COVERAGE":
        degraded.add("LOW_COVERAGE")
    if selected_status == "UNKNOWN":
        degraded.add("SELECTED_ATTEMPT_STATUS_UNKNOWN")
    if polarity_invalid:
        degraded.add("POLARITY_UNKNOWN")
    elif polarity is None:
        degraded.add("POLARITY_UNAVAILABLE")
    evidence_ids = tuple(sorted(str(item["evidence_id"]) for item in scored))
    domains = {str(item.get("domain") or "") for item in scored if item.get("domain")}
    model_versions = {}
    if finbert_available:
        model_versions["finbert"] = FINBERT_MODEL_VERSION
    if vader_available:
        model_versions["vader"] = VADER_MODEL_VERSION
    if fusion_available:
        model_versions["fusion"] = FEAR_GREED_MODEL_VERSION
    model_statuses = {
        "finbert": finbert_status,
        "vader": vader_status,
        "fusion": fusion_status,
        "selected_attempt": selected_status or "UNKNOWN",
    }
    features = MonthlySentimentFeatures(
        aggregate_7d=aggregates[7], aggregate_14d=aggregates[14], aggregate_30d=aggregates[30],
        momentum_7d_vs_30d=momentum, dispersion_30d=dispersion,
        volume_7d=volumes[7], volume_14d=volumes[14], volume_30d=volumes[30],
        source_diversity_30d=len(domains), confidence=confidence,
        polarity=polarity,
        availability=availability, missing_flags=missing_flags,
        degraded_flags=tuple(sorted(degraded)), evidence_ids=evidence_ids,
        model_versions=model_versions, model_statuses=model_statuses,
        dataset_version=context.dataset_version,
        feature_version=SENTIMENT_FEATURE_VERSION, run_id=identity,
    )
    sentiment_lifecycle_event(
        forecast_context=context, run_id=identity,
        boundary="feature_aggregation_and_pre_shared_state",
        started_at=started_at,
        outcome="degraded" if features.degraded_flags else "success",
        counts={
            "input_count": len(texts), "eligible_count": len(eligible),
            "rejected_count": len(rejections),
            "evidence_count": len(features.evidence_ids),
            "missing_count": len(features.missing_flags),
            "result_count": len(SENTIMENT_WINDOWS),
        },
        missing=bool(features.missing_flags), degraded=bool(features.degraded_flags),
        degraded_flags=["feature_degraded"] if features.degraded_flags else [],
        error_code="SENTIMENT_NO_ELIGIBLE_EVIDENCE" if not eligible else None,
        model_versions=features.model_versions,
    )
    return features


def compact_sentiment_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return the persistence-safe legacy summary without raw evidence or reasoning."""
    allowed = {
        "status", "polarity", "execution_success", "pipeline_execution_success",
        "quality_passed", "quality_failures", "ticker", "fear_greed_score",
        "fear_greed_label", "fear_greed_confidence", "direction_strength",
        "model_agreement_score", "evidence_independence", "confidence_components",
        "finbert_label", "finbert_bullish_prob", "finbert_bearish_prob",
        "finbert_neutral_prob", "vader_compound", "vader_label",
        "total_chunks_analyzed", "raw_chunk_count", "unique_document_count",
        "unique_domain_count", "chunks_collapsed_to_documents",
        "duplicate_text_chunks_removed", "source_document_counts",
        "causal_aspects_supported", "causal_aspects_used", "causal_audit",
        "causal_mismatch_detected", "overall_sentiment", "conviction_level",
        "model_agreement", "key_signals", "narrative", "risk_flags",
        "data_quality_note", "loop_iterations_used", "selected_attempt_number",
        "extraction_errors", "attempt_extraction_errors", "evidence_ids",
        "model_versions", "missing_flags", "degraded_flags",
    }
    return {key: summary[key] for key in allowed if key in summary}


def canonical_feature_digest(features: MonthlySentimentFeatures) -> str:
    payload = json.dumps(features.to_state_slot(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "SENTIMENT_FEATURE_VERSION", "SENTIMENT_IMPLEMENTATION_IDENTITY",
    "SENTIMENT_IMPLEMENTATION_VERSION",
    "SENTIMENT_WINDOWS", "SENTIMENT_LIFECYCLE_BOUNDARIES",
    "MonthlySentimentFeatures", "require_sentiment_context",
    "canonical_sentiment_model_status", "canonical_sentiment_polarity",
    "sentiment_lifecycle_event",
    "select_point_in_time_evidence", "build_monthly_sentiment_features",
    "compact_sentiment_summary", "canonical_feature_digest", "valid_composite_score",
    "normalise_aspect_names", "causal_audit", "attempt_rank",
]
