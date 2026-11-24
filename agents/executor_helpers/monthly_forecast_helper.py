"""Fail-closed Stage 08 assembly for deterministic monthly forecasts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from core.forecast_contract import (
    ForecastContext,
    assert_not_after_as_of,
    parse_utc_datetime,
)
from tools.market_tools import validate_evidence_references, validate_run_id
from tools.market_tools.forecast_model import apply_action_policy
from tools.market_tools.specialist_fusion import (
    SPECIALIST_FUSION_VARIANT_TOKEN,
    SPECIALIST_FUSION_VERSION,
)


MONTHLY_FORECAST_VERSION = "monthly-forecast-v1"
STAGE08_IMPLEMENTATION_IDENTITY = {
    "stage": "08",
    "stage_revision": 1,
    "attempt": 2,
    "agent": "integration",
    "implementation_version": "stage08-r1-a2",
}

_NUMERIC_FIELDS = (
    "probability_positive",
    "expected_return_21d",
    "expected_excess_return",
    "confidence",
)
_PROVENANCE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
_EVIDENCE_TIME_FIELDS = frozenset({
    "filed_at", "published_at", "published_date", "publication_time",
    "effective_at", "observed_at", "timestamp", "timestamps",
})


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Quant {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Quant {field} must be finite")
    return result


def _quant_provenance(
    prediction: Mapping[str, Any],
    market: Mapping[str, Any],
) -> dict[str, str]:
    provenance: dict[str, str] = {}
    for field in ("universe_manifest_hash", "model_variant"):
        value = prediction.get(field)
        if (
            not isinstance(value, str)
            or value != value.strip()
            or not _PROVENANCE_IDENTITY.fullmatch(value)
            or "PRIVATE_SENTINEL" in value.upper()
        ):
            raise ValueError(f"Quant {field} must be a compact canonical identity")
        provenance[field] = value
    materialized_at = prediction.get("feature_materialized_at")
    if (
        not isinstance(materialized_at, str)
        or materialized_at != materialized_at.strip()
        or len(materialized_at) > 64
    ):
        raise ValueError("Quant feature_materialized_at must be a compact timestamp")
    parse_utc_datetime(materialized_at, field_name="feature_materialized_at")
    if market["values"].get("feature_materialized_at") != materialized_at:
        raise ValueError("Quant feature materialization provenance mismatch")
    provenance["feature_materialized_at"] = materialized_at
    return provenance


def _specialist_contributions(
    prediction: Mapping[str, Any], *, model_variant: str,
) -> dict[str, Any]:
    """Validate the auditable bridge from specialist values to final Quant numbers."""
    if SPECIALIST_FUSION_VARIANT_TOKEN not in model_variant:
        return {}
    if prediction.get("specialist_fusion_version") != SPECIALIST_FUSION_VERSION:
        raise ValueError("Quant specialist fusion version mismatch")
    families = {"research", "financial", "sentiment"}
    payload: dict[str, Any] = {"version": SPECIALIST_FUSION_VERSION}
    for source_name, target_name in (
        ("specialist_scores", "scores"),
        ("specialist_return_adjustments", "return_adjustments"),
        ("specialist_logit_adjustments", "logit_adjustments"),
    ):
        source = prediction.get(source_name)
        if not isinstance(source, Mapping) or set(source) != families:
            raise ValueError(f"Quant {source_name} must contain every specialist")
        payload[target_name] = {
            family: _finite_number(source[family], field=f"{source_name}.{family}")
            for family in sorted(families)
        }
    total = _finite_number(
        prediction.get("specialist_total_return_adjustment"),
        field="specialist_total_return_adjustment",
    )
    if not math.isclose(
        total, sum(payload["return_adjustments"].values()),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("Quant specialist return adjustment total mismatch")
    base = {
        "probability_positive": _finite_number(
            prediction.get("base_model_probability_positive"),
            field="base_model_probability_positive",
        ),
        "expected_return_21d": _finite_number(
            prediction.get("base_model_expected_return_21d"),
            field="base_model_expected_return_21d",
        ),
        "expected_excess_return": _finite_number(
            prediction.get("base_model_expected_excess_return"),
            field="base_model_expected_excess_return",
        ),
    }
    if not math.isclose(
        float(prediction["expected_return_21d"]), base["expected_return_21d"] + total,
        rel_tol=0.0, abs_tol=1e-12,
    ) or not math.isclose(
        float(prediction["expected_excess_return"]), base["expected_excess_return"] + total,
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise ValueError("Quant final return does not include specialist adjustment")
    payload["base_model"] = base
    payload["total_return_adjustment"] = total
    return payload


def _slot(state: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    value = state.get(name)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a typed state-slot mapping")
    if not isinstance(value.get("values"), Mapping):
        raise TypeError(f"{name}.values must be a mapping")
    return value


def _research_driver_reason(
    direction: str, event_signals: list[str], source: str, timestamp: str,
) -> str:
    signal_text = "; ".join(event_signals) if event_signals else "a classified directional event"
    origin = f" Source: {source}." if source and source != "unknown" else ""
    observed = f" Observed at or before {timestamp}." if timestamp else ""
    return f"{direction.title()} research signal: {signal_text}.{origin}{observed}".strip()


def _specialist_driver_reason(
    family: str, direction: str, metrics: Mapping[str, float],
) -> str:
    labels = {
        "valuation_gap": "valuation gap", "revenue_growth": "revenue growth",
        "earnings_growth": "earnings growth", "roe_pct": "return on equity",
        "net_margin_pct": "net margin", "roic_pct": "return on invested capital",
        "debt_ratio": "debt ratio", "current_ratio": "current ratio",
        "aggregate_7d": "7-day sentiment", "aggregate_14d": "14-day sentiment",
        "aggregate_30d": "30-day sentiment",
        "momentum_7d_vs_30d": "7-day versus 30-day momentum",
        "confidence": "confidence",
    }
    details = ", ".join(
        f"{labels.get(name, name.replace('_', ' '))} {value:+.3f}"
        for name, value in metrics.items()
    )
    if not details:
        details = "the validated feature score"
    return f"{direction.title()} {family} signal based on {details}."


def _decision_drivers(
    slots: Mapping[str, Mapping[str, Any] | None],
    contributions: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Expose only compact, evidenced inputs that explain specialist adjustments."""
    drivers: list[dict[str, Any]] = []
    scores = contributions.get("scores") if isinstance(contributions, Mapping) else {}
    adjustments = contributions.get("return_adjustments") if isinstance(contributions, Mapping) else {}
    scores = scores if isinstance(scores, Mapping) else {}
    adjustments = adjustments if isinstance(adjustments, Mapping) else {}

    research = slots.get("research_features")
    if research is not None:
        values = research["values"]
        allowed = set(validate_evidence_references(research.get("evidence_ids", ())))
        claims = values.get("claims")
        if isinstance(claims, (list, tuple)):
            for claim in claims[:5]:
                if not isinstance(claim, Mapping):
                    continue
                evidence_id = str(claim.get("evidence_id") or "")
                direction = str(claim.get("direction") or "")
                if evidence_id not in allowed or direction not in {"POSITIVE", "NEGATIVE"}:
                    continue
                try:
                    strength = _finite_number(claim.get("strength"), field="research_claim.strength")
                    confidence = _finite_number(claim.get("confidence"), field="research_claim.confidence")
                    recency = _finite_number(claim.get("recency_weight"), field="research_claim.recency_weight")
                except ValueError:
                    continue
                if not all(0.0 <= value <= 1.0 for value in (strength, confidence, recency)):
                    continue
                source = str(claim.get("source") or "unknown")[:80]
                timestamp = str(claim.get("timestamp") or "")[:64]
                event_signals = [
                    str(signal)[:80] for signal in (claim.get("event_signals") or [])[:3]
                    if isinstance(signal, str) and signal.strip()
                ]
                drivers.append({
                    "specialist": "research", "evidence_id": evidence_id,
                    "citation_reference_id": str(claim.get("citation_reference_id") or "")[:160],
                    "source": source, "timestamp": timestamp,
                    "direction": direction, "strength": strength,
                    "confidence": confidence, "recency_weight": recency,
                    "event_signals": event_signals,
                    "reason": _research_driver_reason(
                        direction, event_signals, source, timestamp,
                    ),
                    "specialist_score": scores.get("research"),
                    "return_adjustment": adjustments.get("research"),
                })

    metric_names = {
        "financial": (
            "valuation_gap", "revenue_growth", "earnings_growth", "roe_pct",
            "net_margin_pct", "roic_pct", "debt_ratio", "current_ratio",
        ),
        "sentiment": (
            "aggregate_7d", "aggregate_14d", "aggregate_30d",
            "momentum_7d_vs_30d", "confidence",
        ),
    }
    for family in ("financial", "sentiment"):
        slot = slots.get(f"{family}_features")
        if slot is None or family not in scores:
            continue
        metrics = {
            name: float(value)
            for name in metric_names[family]
            if (value := slot["values"].get(name)) is not None
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        }
        score = _finite_number(scores[family], field=f"{family}.score")
        direction = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
        drivers.append({
            "specialist": family, "direction": direction, "score": score,
            "return_adjustment": adjustments.get(family), "metrics": metrics,
            "reason": _specialist_driver_reason(family, direction, metrics),
            "evidence_ids": list(validate_evidence_references(slot.get("evidence_ids", ())))[:3],
        })
    return tuple(drivers)

def _find_future_evidence(value: Any, context: ForecastContext, *, path: str) -> list[str]:
    rejected: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key == "timestamps" and isinstance(item, (list, tuple)):
                for index, timestamp in enumerate(item):
                    timestamp_path = f"{child}[{index}]"
                    try:
                        assert_not_after_as_of(
                            timestamp, context.as_of, field_name=timestamp_path,
                        )
                    except (TypeError, ValueError):
                        rejected.append(timestamp_path)
            elif key in _EVIDENCE_TIME_FIELDS and item is not None:
                try:
                    assert_not_after_as_of(item, context.as_of, field_name=child)
                except (TypeError, ValueError):
                    rejected.append(child)
            else:
                rejected.extend(_find_future_evidence(item, context, path=child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rejected.extend(_find_future_evidence(item, context, path=f"{path}[{index}]"))
    return rejected


def _identity_matches(prediction: Mapping[str, Any], context: ForecastContext, run_id: str) -> bool:
    expected = {
        "ticker": context.ticker,
        "as_of": context.as_of.isoformat().replace("+00:00", "Z"),
        "target_date": context.target_date.isoformat(),
        "horizon_trading_days": context.horizon_trading_days,
        "analysis_mode": context.analysis_mode.value,
        "benchmark": context.benchmark,
        "schema_version": context.schema_version,
        "run_id": run_id,
    }
    return all(prediction.get(key) == value for key, value in expected.items())


def specialist_quality_gate(state: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Require usable, evidenced feature slots from every monthly specialist."""
    failures: list[str] = []
    for name in ("research_features", "financial_features", "sentiment_features"):
        slot = state.get(name)
        code = name.removesuffix("_features").upper()
        if not isinstance(slot, Mapping):
            failures.append(f"{code}_SLOT_MISSING")
            continue
        values = slot.get("values")
        evidence = slot.get("evidence_ids")
        if slot.get("missing") is True:
            failures.append(f"{code}_REQUIRED_INPUT_MISSING")
        if slot.get("degraded") is True:
            failures.append(f"{code}_QUALITY_DEGRADED")
        if not isinstance(values, Mapping) or not values:
            failures.append(f"{code}_VALUES_EMPTY")
        if not isinstance(evidence, (list, tuple)) or not evidence:
            failures.append(f"{code}_EVIDENCE_EMPTY")
    result = tuple(sorted(set(failures)))
    return not result, result


@dataclass(frozen=True, slots=True)
class MonthlyForecastResult:
    status: str
    ticker: str
    as_of: str
    target_date: str
    horizon_trading_days: int
    analysis_mode: str
    cutoff_policy: str
    benchmark: str
    run_id: str
    universe_manifest_hash: str | None
    model_variant: str | None
    feature_materialized_at: str | None
    action: str
    probability_positive: float | None
    expected_return_21d: float | None
    expected_excess_return: float | None
    confidence_interval: tuple[float, float] | None
    confidence: float | None
    evidence_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    missing: bool
    degraded: bool
    graph_coverage: bool | None
    quality_gate_passed: bool
    quality_gate_failures: tuple[str, ...]
    versions: Mapping[str, str | None]
    specialist_contributions: Mapping[str, Any]
    decision_drivers: tuple[Mapping[str, Any], ...]
    narrative: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ticker": self.ticker,
            "as_of": self.as_of,
            "target_date": self.target_date,
            "horizon_trading_days": self.horizon_trading_days,
            "analysis_mode": self.analysis_mode,
            "cutoff_policy": self.cutoff_policy,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "universe_manifest_hash": self.universe_manifest_hash,
            "model_variant": self.model_variant,
            "feature_materialized_at": self.feature_materialized_at,
            "action": self.action,
            "probability_positive": self.probability_positive,
            "expected_return_21d": self.expected_return_21d,
            "expected_excess_return": self.expected_excess_return,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
            "missing": self.missing,
            "degraded": self.degraded,
            "graph_coverage": self.graph_coverage,
            "quality_gate_passed": self.quality_gate_passed,
            "quality_gate_failures": list(self.quality_gate_failures),
            "versions": dict(self.versions),
            "specialist_contributions": dict(self.specialist_contributions),
            "decision_drivers": [dict(driver) for driver in self.decision_drivers],
            "narrative": self.narrative,
            "implementation": dict(STAGE08_IMPLEMENTATION_IDENTITY),
        }

    def to_state_slot(self) -> dict[str, Any]:
        return self.to_dict()

    def with_narrative(self, narrative: str) -> "MonthlyForecastResult":
        return replace(self, narrative=narrative[:4000])


def _insufficient(
    *, context: ForecastContext, run_id: str, warnings: set[str],
    evidence_ids: tuple[str, ...] = (), graph_coverage: bool | None = None,
    universe_manifest_hash: str | None = None,
    model_variant: str | None = None,
    feature_materialized_at: str | None = None,
    quality_gate_passed: bool = False,
    quality_gate_failures: tuple[str, ...] = (),
) -> MonthlyForecastResult:
    versions = {
        "schema_version": context.schema_version,
        "dataset_version": context.dataset_version,
        "feature_version": context.feature_version,
        "model_version": context.model_version,
        "action_policy_version": "action_policy_v1",
        "monthly_forecast_version": MONTHLY_FORECAST_VERSION,
        "universe_manifest_hash": universe_manifest_hash,
        "model_variant": model_variant,
        "feature_materialized_at": feature_materialized_at,
    }
    return MonthlyForecastResult(
        status="INSUFFICIENT_DATA", ticker=context.ticker,
        as_of=context.as_of.isoformat().replace("+00:00", "Z"),
        target_date=context.target_date.isoformat(),
        horizon_trading_days=context.horizon_trading_days,
        analysis_mode=context.analysis_mode.value, cutoff_policy=context.cutoff_policy.value,
        benchmark=context.benchmark,
        run_id=run_id, universe_manifest_hash=universe_manifest_hash,
        model_variant=model_variant,
        feature_materialized_at=feature_materialized_at,
        action="INSUFFICIENT_DATA",
        probability_positive=None, expected_return_21d=None,
        expected_excess_return=None, confidence_interval=None, confidence=None,
        evidence_ids=evidence_ids, warnings=tuple(sorted(warnings)),
        missing=True, degraded=True, graph_coverage=graph_coverage,
        quality_gate_passed=quality_gate_passed,
        quality_gate_failures=quality_gate_failures,
        versions=versions, specialist_contributions={}, decision_drivers=(),
    )


def assemble_monthly_forecast(state: Mapping[str, Any]) -> MonthlyForecastResult:
    """Read only typed Shared State and preserve Quant values byte-for-value."""
    context_payload = state.get("forecast_context")
    if not isinstance(context_payload, Mapping):
        raise ValueError("forecast_context is required")
    context = ForecastContext.from_dict(dict(context_payload))
    directives = state.get("manager_directives")
    directives = directives if isinstance(directives, Mapping) else {}
    run_id = validate_run_id(str(directives.get("run_id") or state.get("run_id") or ""))
    quality_passed, quality_failures = specialist_quality_gate(state)

    slots = {
        name: _slot(state, name)
        for name in ("research_features", "financial_features", "sentiment_features", "market_features")
    }
    evidence: set[str] = set()
    warnings: set[str] = set()
    graph_coverage: bool | None = None
    for name, slot in slots.items():
        if slot is None:
            warnings.add(f"MISSING_{name.upper()}")
            continue
        try:
            evidence.update(validate_evidence_references(slot.get("evidence_ids", ())))
        except (TypeError, ValueError):
            warnings.add(f"{name.upper()}_EVIDENCE_REJECTED")
        if slot.get("missing"):
            warnings.add(f"{name.upper()}_MISSING")
        if slot.get("degraded"):
            warnings.add(f"{name.upper()}_DEGRADED")
        values = slot["values"]
        slot_run_id = values.get("run_id")
        if slot_run_id is not None and slot_run_id != run_id:
            warnings.add(f"{name.upper()}_RUN_ID_MISMATCH")
        slot_dataset = values.get("dataset_version")
        if slot_dataset is not None and slot_dataset != context.dataset_version:
            warnings.add(f"{name.upper()}_DATASET_VERSION_MISMATCH")
        future = _find_future_evidence(values, context, path=name)
        if future:
            warnings.add(f"{name.upper()}_FUTURE_EVIDENCE_REJECTED")
        if name == "research_features":
            candidate = values.get("graph_coverage")
            graph_coverage = candidate if isinstance(candidate, bool) else None

    market = slots["market_features"]
    prediction = market["values"].get("quant_prediction") if market else None
    provenance: dict[str, str | None] = {
        "universe_manifest_hash": None,
        "model_variant": None,
        "feature_materialized_at": None,
    }
    fatal = {
        warning for warning in warnings
        if warning.startswith("MISSING_MARKET_FEATURES")
        or warning.startswith("MARKET_FEATURES_MISSING")
        or warning.endswith("_RUN_ID_MISMATCH")
        or warning.endswith("_DATASET_VERSION_MISMATCH")
        or warning.endswith("_FUTURE_EVIDENCE_REJECTED")
        or warning.endswith("_EVIDENCE_REJECTED")
    }
    if not isinstance(prediction, Mapping):
        fatal.add("MISSING_QUANT_PREDICTION")
    else:
        try:
            provenance.update(_quant_provenance(prediction, market))
        except ValueError:
            fatal.add("QUANT_PREDICTION_REJECTED")
    if fatal:
        return _insufficient(
            context=context, run_id=run_id, warnings=warnings | fatal,
            evidence_ids=tuple(sorted(evidence)), graph_coverage=graph_coverage,
            quality_gate_passed=quality_passed,
            quality_gate_failures=quality_failures,
            **provenance,
        )
    assert isinstance(prediction, Mapping)
    try:
        if not _identity_matches(prediction, context, run_id):
            raise ValueError("Quant prediction identity mismatch")
        if prediction.get("model_version") != context.model_version:
            raise ValueError("Quant model version mismatch")
        if prediction.get("feature_version") != context.feature_version:
            raise ValueError("Quant feature version mismatch")
        if prediction.get("dataset_version") != context.dataset_version:
            raise ValueError("Quant dataset version mismatch")
        if prediction.get("action_policy_version") != "action_policy_v1":
            raise ValueError("Quant action policy version mismatch")
        if prediction.get("missing") is True:
            raise ValueError("Quant prediction reports missing required inputs")
        numeric = {field: _finite_number(prediction.get(field), field=field) for field in _NUMERIC_FIELDS}
        if not 0.0 <= numeric["probability_positive"] <= 1.0:
            raise ValueError("Quant probability_positive must be within [0, 1]")
        if not 0.0 <= numeric["confidence"] <= 1.0:
            raise ValueError("Quant confidence must be within [0, 1]")
        interval = prediction.get("confidence_interval")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("Quant confidence_interval must contain two values")
        low = _finite_number(interval[0], field="confidence_interval[0]")
        high = _finite_number(interval[1], field="confidence_interval[1]")
        if low > high:
            raise ValueError("Quant confidence_interval is reversed")
        canonical_action = apply_action_policy(
            numeric["probability_positive"], numeric["expected_excess_return"]
        )
        if prediction.get("action") != canonical_action:
            raise ValueError("Quant action does not match action_policy_v1")
        contributions = _specialist_contributions(
            prediction, model_variant=str(provenance["model_variant"]),
        )
    except ValueError:
        return _insufficient(
            context=context, run_id=run_id,
            warnings=warnings | {"QUANT_PREDICTION_REJECTED"},
            evidence_ids=tuple(sorted(evidence)), graph_coverage=graph_coverage,
            quality_gate_passed=quality_passed,
            quality_gate_failures=quality_failures,
            **provenance,
        )

    try:
        evidence.update(validate_evidence_references(prediction.get("evidence_ids", ())))
    except (TypeError, ValueError):
        return _insufficient(
            context=context, run_id=run_id,
            warnings=warnings | {"QUANT_EVIDENCE_REJECTED"},
            evidence_ids=tuple(sorted(evidence)), graph_coverage=graph_coverage,
            quality_gate_passed=quality_passed,
            quality_gate_failures=quality_failures,
            **provenance,
        )
    degraded = bool(warnings) or bool(prediction.get("degraded"))
    versions = {
        "schema_version": context.schema_version,
        "dataset_version": str(prediction.get("dataset_version")),
        "feature_version": str(prediction.get("feature_version")),
        "model_version": str(prediction.get("model_version")),
        "action_policy_version": str(prediction.get("action_policy_version")),
        "monthly_forecast_version": MONTHLY_FORECAST_VERSION,
        **provenance,
    }
    return MonthlyForecastResult(
        status="DEGRADED" if degraded else "READY", ticker=context.ticker,
        as_of=context.as_of.isoformat().replace("+00:00", "Z"),
        target_date=context.target_date.isoformat(),
        horizon_trading_days=context.horizon_trading_days,
        analysis_mode=context.analysis_mode.value, cutoff_policy=context.cutoff_policy.value,
        benchmark=context.benchmark,
        run_id=run_id,
        universe_manifest_hash=provenance["universe_manifest_hash"],
        model_variant=provenance["model_variant"],
        feature_materialized_at=provenance["feature_materialized_at"],
        action=str(prediction["action"]),
        probability_positive=numeric["probability_positive"],
        expected_return_21d=numeric["expected_return_21d"],
        expected_excess_return=numeric["expected_excess_return"],
        confidence_interval=(low, high), confidence=numeric["confidence"],
        evidence_ids=tuple(sorted(evidence)), warnings=tuple(sorted(warnings)),
        missing=False, degraded=degraded, graph_coverage=graph_coverage,
        quality_gate_passed=quality_passed,
        quality_gate_failures=quality_failures,
        versions=versions, specialist_contributions=contributions,
        decision_drivers=_decision_drivers(slots, contributions),
    )


__all__ = [
    "MONTHLY_FORECAST_VERSION", "STAGE08_IMPLEMENTATION_IDENTITY",
    "MonthlyForecastResult", "assemble_monthly_forecast", "specialist_quality_gate",
]
