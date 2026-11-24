"""Deterministic guardrail for explanation-only monthly forecast text."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


_ACTION_WORD = re.compile(r"\b(BUY|HOLD|AVOID|SELL)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*%?")
_CITATION = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9._:+-]{0,159})\]")


def _numbers(value: Any) -> set[float]:
    found: set[float] = set()
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        found.add(float(value))
        found.add(float(value) * 100.0)
    elif isinstance(value, Mapping):
        for item in value.values():
            found.update(_numbers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_numbers(item))
    return found


def _pct(value: Any) -> str:
    return f"{float(value):+.2%}"


def deterministic_narrative(result: Any) -> str:
    if result.status == "INSUFFICIENT_DATA":
        return "Insufficient point-in-time data; no numeric forecast or confidence is available."
    qualifier = " with degraded inputs" if result.degraded else ""
    contributions = dict(result.specialist_contributions or {})
    drivers = list(result.decision_drivers or ())
    if not contributions:
        return (
            f"The deterministic Quant model assigns action {result.action}{qualifier}. "
            "The 21-session forecast is distinct from any long-term DCF valuation."
        )
    base = contributions.get("base_model") or {}
    adjustments = contributions.get("return_adjustments") or {}
    sentences = [
        f"Action {result.action}{qualifier}: the base model expected return was "
        f"{_pct(base.get('expected_return_21d', 0.0))}; specialist evidence changed it by "
        f"{_pct(contributions.get('total_return_adjustment', 0.0))} to "
        f"{_pct(result.expected_return_21d)}."
    ]
    research = [driver for driver in drivers if driver.get("specialist") == "research"]
    if research:
        strongest = research[0]
        reason = str(strongest.get("reason") or "A validated directional research event was found.")
        sentences.append(
            f"Research contributed {_pct(adjustments.get('research', 0.0))}: {reason}"
        )
    else:
        sentences.append(
            f"Research produced no classified directional claim and contributed "
            f"{_pct(adjustments.get('research', 0.0))}."
        )
    for family, label in (("financial", "Finance"), ("sentiment", "Sentiment")):
        family_driver = next((driver for driver in drivers if driver.get("specialist") == family), None)
        if family_driver is None:
            continue
        reason = str(family_driver.get("reason") or "No compact reason was returned.")
        sentences.append(
            f"{label} contributed {_pct(adjustments.get(family, 0.0))}: {reason}"
        )
    sentences.append("DCF remains a long-term valuation input, not the 21-session forecast itself.")
    return " ".join(sentences)[:4000]


def narrative_matches_quant(text: str, result: Any) -> bool:
    """Accept only action and numeric claims reconciled with structured output."""
    candidate = str(text or "").strip()
    if not candidate:
        return False
    authoritative = str(result.action).upper()
    for found in _ACTION_WORD.findall(candidate):
        if found.upper() != authoritative:
            return False
    if authoritative in {"BUY", "HOLD", "AVOID"}:
        negation = (
            rf"(?:\b(?:not|never|isn't|is\s+not|do\s+not|don't)\b.{{0,24}}"
            rf"\b{authoritative}\b|\b{authoritative}\b.{{0,24}}"
            rf"\b(?:not|never|isn't)\b)"
        )
        if re.search(negation, candidate, re.IGNORECASE):
            return False

    allowed = {float(result.horizon_trading_days)}
    values = (
        result.probability_positive,
        result.expected_return_21d,
        result.expected_excess_return,
        result.confidence,
        *(result.confidence_interval or ()),
    )
    for value in values:
        if value is not None and math.isfinite(float(value)):
            allowed.add(float(value))
            allowed.add(float(value) * 100.0)
    allowed.update(_numbers(result.specialist_contributions))
    allowed.update(_numbers(result.decision_drivers))
    for match in _NUMBER.finditer(candidate):
        token = match.group(0).strip().removesuffix("%").strip()
        number = float(token)
        if not any(
            math.isclose(number, expected, rel_tol=5e-3, abs_tol=1e-6)
            for expected in allowed
        ):
            return False

    # Opaque audit IDs remain in the structured response, never in user-facing prose.
    if _CITATION.search(candidate):
        return False
    return True


__all__ = ["deterministic_narrative", "narrative_matches_quant"]
