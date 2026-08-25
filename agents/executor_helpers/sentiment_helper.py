from __future__ import annotations

import math
from typing import Any


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

    return {
        str(item).strip().lower()
        for item in value
        if str(item).strip()
    }


def causal_audit(
    analysis: dict[str, Any],
    causal_aspects_supported: list[str],
) -> dict[str, Any]:

    claimed = normalise_aspect_names(
        analysis.get("causal_aspects_used")
    )

    supported = normalise_aspect_names(
        causal_aspects_supported
    )

    unsupported = sorted(claimed - supported)

    return {
        "claimed_causal_aspects": sorted(claimed),
        "supported_causal_aspects": sorted(supported),
        "unsupported_causal_aspects": unsupported,
        "causal_mismatch_detected": bool(unsupported),
    }


def attempt_rank(
    attempt: dict[str, Any],
) -> tuple[Any, ...]:

    coverage = attempt.get("retrieval_stats", {}) or {}

    fear_greed = (
        (attempt.get("model_outputs", {}) or {})
        .get("fear_greed", {})
        or {}
    )

    status_rank = {
        "ERROR": 0,
        "NO_DATA": 1,
        "LOW_COVERAGE": 2,
        "VALID": 3,
    }

    return (
        bool(attempt.get("quality_passed")),
        status_rank.get(
            str(attempt.get("status")),
            -1,
        ),
        valid_composite_score(
            fear_greed.get("score")
        ),
        bool(attempt.get("execution_success")),
        int(coverage.get("unique_domains") or 0),
        int(coverage.get("unique_documents") or 0),
        float(fear_greed.get("confidence") or 0.0),
        -int(attempt.get("attempt_number") or 0),
    )
