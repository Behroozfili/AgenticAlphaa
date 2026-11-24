"""Transparent deterministic fusion of live specialist features with Quant output."""

from __future__ import annotations

import math
from typing import Any

from core.error_handler import DataUnavailableAsOfError
from tools.market_tools.feature_builder import MarketFeatures
from tools.market_tools.forecast_model import apply_action_policy


SPECIALIST_FUSION_VERSION = "specialist-fusion-v2"
SPECIALIST_FUSION_VARIANT_TOKEN = "specialist_fusion_v2"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _require_specialists(features: MarketFeatures) -> None:
    missing = [
        family for family in ("research", "financial", "sentiment")
        if features.missing_indicators.get(f"{family}_features__missing", True)
    ]
    if missing:
        raise DataUnavailableAsOfError(
            "specialist fusion requires non-missing Research, Finance and Sentiment features: "
            + ",".join(missing)
        )


def _research_score(values: dict[str, float | None]) -> float:
    directional = _number(values.get("research_directional_score"))
    confidence = _number(values.get("research_claim_confidence"))
    classified_count = _number(values.get("research_classified_claim_count"))
    diversity = _number(values.get("research_source_diversity"))
    vector = _number(values.get("research_vector_coverage"))
    graph = _number(values.get("research_graph_coverage"))
    if None in (directional, confidence, classified_count, diversity, vector, graph):
        raise DataUnavailableAsOfError("research fusion v2 features are incomplete")
    if classified_count <= 0:
        return 0.0
    quality = (
        _clip(diversity / 3.0, 0.0, 1.0)
        + _clip(vector, 0.0, 1.0)
        + _clip(graph, 0.0, 1.0)
        + _clip(confidence, 0.0, 1.0)
    ) / 4.0
    return _clip(directional * quality)

def _scaled_financial(name: str, value: float) -> float:
    if name in {"financial_revenue_growth", "financial_earnings_growth", "financial_valuation_gap"}:
        scale = value if abs(value) <= 2.0 else value / 100.0
        return math.tanh(scale * 2.0)
    if name in {"financial_roe_pct", "financial_net_margin_pct", "financial_roic_pct",
                "financial_fcf_margin_pct", "financial_fcf_yield_pct"}:
        return math.tanh(value / 25.0)
    if name in {"financial_debt_ratio", "financial_net_debt_to_ebitda"}:
        return -math.tanh(max(0.0, value) / 3.0)
    if name in {"financial_current_ratio", "financial_quick_ratio", "financial_ocf_ratio"}:
        return math.tanh(value - 1.0)
    return 0.0


def _financial_score(values: dict[str, float | None]) -> float:
    names = (
        "financial_valuation_gap", "financial_revenue_growth", "financial_earnings_growth",
        "financial_roe_pct", "financial_net_margin_pct", "financial_roic_pct",
        "financial_net_debt_to_ebitda", "financial_debt_ratio",
        "financial_current_ratio", "financial_quick_ratio",
        "financial_fcf_margin_pct", "financial_fcf_yield_pct", "financial_ocf_ratio",
    )
    observed = [(name, number) for name in names if (number := _number(values.get(name))) is not None]
    if len(observed) < 3:
        raise DataUnavailableAsOfError("financial fusion requires at least three observed metrics")
    return _clip(sum(_scaled_financial(name, value) for name, value in observed) / len(observed))


def _sentiment_score(values: dict[str, float | None]) -> float:
    required = {
        name: _number(values.get(name)) for name in (
            "sentiment_aggregate_7d", "sentiment_aggregate_14d",
            "sentiment_aggregate_30d", "sentiment_momentum_7d_vs_30d",
            "sentiment_confidence",
        )
    }
    if any(value is None for value in required.values()):
        raise DataUnavailableAsOfError("sentiment fusion features are incomplete")
    raw = (
        0.45 * required["sentiment_aggregate_7d"]
        + 0.30 * required["sentiment_aggregate_14d"]
        + 0.15 * required["sentiment_aggregate_30d"]
        + 0.10 * required["sentiment_momentum_7d_vs_30d"]
    )
    return _clip(raw * _clip(required["sentiment_confidence"], 0.0, 1.0))


def apply_specialist_fusion(base: dict[str, Any], features: MarketFeatures) -> dict[str, Any]:
    """Return a forecast whose final numeric values include all three specialists."""
    _require_specialists(features)
    values = dict(features.values)
    scores = {
        "research": _research_score(values),
        "financial": _financial_score(values),
        "sentiment": _sentiment_score(values),
    }
    return_adjustments = {
        "research": 0.006 * scores["research"],
        "financial": 0.012 * scores["financial"],
        "sentiment": 0.010 * scores["sentiment"],
    }
    logit_adjustments = {
        "research": 0.20 * scores["research"],
        "financial": 0.35 * scores["financial"],
        "sentiment": 0.30 * scores["sentiment"],
    }
    base_probability = _clip(float(base["probability_positive"]), 1e-9, 1.0 - 1e-9)
    base_logit = math.log(base_probability / (1.0 - base_probability))
    probability = 1.0 / (1.0 + math.exp(-(base_logit + sum(logit_adjustments.values()))))
    alpha = sum(return_adjustments.values())
    expected_return = float(base["expected_return_21d"]) + alpha
    expected_excess = float(base["expected_excess_return"]) + alpha
    interval = [float(item) + alpha for item in base["confidence_interval"]]
    result = dict(base)
    result.update({
        "base_model_probability_positive": float(base["probability_positive"]),
        "base_model_expected_return_21d": float(base["expected_return_21d"]),
        "base_model_expected_excess_return": float(base["expected_excess_return"]),
        "probability_positive": probability,
        "expected_return_21d": expected_return,
        "expected_excess_return": expected_excess,
        "confidence_interval": interval,
        "action": apply_action_policy(probability, expected_excess),
        "confidence": max(0.0, min(1.0, abs(probability - 0.5) * 2.0)),
        "specialist_fusion_version": SPECIALIST_FUSION_VERSION,
        "specialist_scores": scores,
        "specialist_return_adjustments": return_adjustments,
        "specialist_logit_adjustments": logit_adjustments,
        "specialist_total_return_adjustment": alpha,
        "specialist_inputs_required": True,
    })
    return result


__all__ = [
    "SPECIALIST_FUSION_VARIANT_TOKEN", "SPECIALIST_FUSION_VERSION",
    "apply_specialist_fusion",
]
