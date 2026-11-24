"""Deterministic, auditable direction classification for compact research evidence."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Iterable, Mapping

from core.forecast_contract import ForecastContext, parse_utc_datetime


RESEARCH_CLAIM_CLASSIFIER_VERSION = "research-claim-lexicon-v1"

_POSITIVE: tuple[tuple[str, float], ...] = (
    ("raises guidance", 1.0), ("raised guidance", 1.0),
    ("beats estimates", 0.9), ("beat estimates", 0.9),
    ("record revenue", 0.9), ("record profit", 0.9),
    ("revenue growth", 0.55), ("earnings growth", 0.65),
    ("profit growth", 0.65), ("margin expansion", 0.7),
    ("upgrade", 0.6), ("upgraded", 0.6),
    ("approval", 0.7), ("approved", 0.65),
    ("contract win", 0.75), ("wins contract", 0.75),
    ("partnership", 0.4), ("launch", 0.35),
    ("strong demand", 0.6), ("accelerating demand", 0.75),
    ("buyback", 0.45), ("dividend increase", 0.5),
    ("outperform", 0.55),
)
_NEGATIVE: tuple[tuple[str, float], ...] = (
    ("lowers guidance", 1.0), ("lowered guidance", 1.0),
    ("misses estimates", 0.9), ("missed estimates", 0.9),
    ("revenue decline", 0.65), ("earnings decline", 0.7),
    ("profit decline", 0.7), ("margin compression", 0.7),
    ("downgrade", 0.6), ("downgraded", 0.6),
    ("investigation", 0.55), ("antitrust", 0.6),
    ("lawsuit", 0.55), ("data breach", 0.8),
    ("outage", 0.55), ("recall", 0.75),
    ("layoffs", 0.45), ("job cuts", 0.45),
    ("weak demand", 0.65), ("slowing demand", 0.65),
    ("delay", 0.4), ("delayed", 0.4),
    ("fine", 0.45), ("loss", 0.35),
    ("underperform", 0.55),
)
_EXPLICIT_POSITIVE = frozenset({"catalyst", "opportunity", "positive"})
_EXPLICIT_NEGATIVE = frozenset({"risk", "negative", "risk_factor"})
_SOURCE_CONFIDENCE = {
    "sec_edgar_filing": 0.95,
    "sec_edgar_search": 0.90,
    "company_filing": 0.95,
    "regulatory": 0.90,
    "rag_hybrid_query": 0.78,
    "rag_vector_search": 0.74,
    "rag_graph_traverse": 0.74,
    "news_search": 0.72,
    "tavily_search": 0.70,
}


def _matches(text: str, lexicon: tuple[tuple[str, float], ...]) -> list[tuple[str, float]]:
    return [
        (phrase, weight)
        for phrase, weight in lexicon
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
    ]


def _recency_weight(timestamp: Any, context: ForecastContext) -> float:
    try:
        observed = parse_utc_datetime(timestamp, field_name="research_claim.timestamp")
    except (TypeError, ValueError):
        return 0.25
    age_days = max(0.0, (context.as_of - observed).total_seconds() / 86_400.0)
    half_life = max(7.0, context.lookback_days / 2.0)
    return max(0.20, min(1.0, math.pow(0.5, age_days / half_life)))


def classify_research_evidence(
    evidence: Iterable[Mapping[str, Any]], *, context: ForecastContext,
) -> tuple[dict[str, Any], ...]:
    """Create bounded claims; every claim remains tied to one accepted evidence ID."""
    claims: list[dict[str, Any]] = []
    for item in evidence:
        evidence_id = str(item.get("evidence_id") or "").strip()
        if not evidence_id:
            continue
        title = str(item.get("title") or "").strip()
        excerpt = str(item.get("content_excerpt") or "").strip()
        text = f"{title}\n{excerpt}".lower()[:8_000]
        kind = str(item.get("kind") or "").strip().lower()
        positive = _matches(text, _POSITIVE)
        negative = _matches(text, _NEGATIVE)
        pos_weight = sum(weight for _, weight in positive)
        neg_weight = sum(weight for _, weight in negative)
        explicit = kind in (_EXPLICIT_POSITIVE | _EXPLICIT_NEGATIVE)
        if kind in _EXPLICIT_POSITIVE:
            pos_weight += 1.0
        elif kind in _EXPLICIT_NEGATIVE:
            neg_weight += 1.0
        net = pos_weight - neg_weight
        if abs(net) < 0.20:
            direction = "NEUTRAL"
            strength = 0.0
        else:
            direction = "POSITIVE" if net > 0 else "NEGATIVE"
            strength = min(1.0, abs(net) / max(1.0, pos_weight + neg_weight))
        source = str(item.get("source") or "unknown").strip().lower()
        base_confidence = _SOURCE_CONFIDENCE.get(source, 0.65)
        signal_count = len(positive) + len(negative) + int(explicit)
        confidence = min(0.98, base_confidence * (0.78 + min(0.20, signal_count * 0.05)))
        matched = positive if direction == "POSITIVE" else negative
        event_signals = [phrase for phrase, _ in matched[:3]] if direction != "NEUTRAL" else []
        if explicit and not event_signals:
            event_signals = [kind]
        claims.append({
            "evidence_id": evidence_id,
            "citation_reference_id": item.get("citation_reference_id"),
            "source": source,
            "timestamp": str(item.get("timestamp") or ""),
            "direction": direction,
            "strength": round(strength, 8),
            "confidence": round(confidence, 8),
            "recency_weight": round(_recency_weight(item.get("timestamp"), context), 8),
            "event_signals": event_signals,
            "classifier_version": RESEARCH_CLAIM_CLASSIFIER_VERSION,
        })
    return tuple(sorted(claims, key=lambda claim: claim["evidence_id"]))


__all__ = ["RESEARCH_CLAIM_CLASSIFIER_VERSION", "classify_research_evidence"]
