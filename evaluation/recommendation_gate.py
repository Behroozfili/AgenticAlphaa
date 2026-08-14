"""Deterministic confidence gate for actionable investment recommendations."""

from __future__ import annotations

import re
from typing import Any

_ACTION = re.compile(r"\b(buy|sell|accumulate|add to (?:the )?position|enter (?:the )?position|initiate (?:a )?position)\b", re.I)


def build_recommendation_gate(shared: dict[str, Any]) -> dict[str, Any]:
    financial = shared.get("financial_metrics_summary", {})
    sentiment = shared.get("sentiment_analysis_summary", {})
    sanity = financial.get("financial_sanity", {})
    reasons: list[str] = []
    if sanity.get("actionable_valuation_allowed") is False or sanity.get("passed") is False:
        reasons.append("financial sanity validation does not allow actionable valuation")
    sentiment_status = sentiment.get("status", "NO_DATA")
    if sentiment_status != "VALID":
        reasons.append(f"sentiment status is {sentiment_status}, not VALID")
    if shared.get("qa_period_findings"):
        reasons.append("unresolved period findings remain")
    cap = str(sanity.get("confidence_cap", "HIGH")).upper()
    if reasons:
        level, mode = "LOW", "SCENARIO_SUMMARY_ONLY"
    elif cap == "MEDIUM":
        level, mode = "MEDIUM", "CONDITIONAL_ONLY"
    else:
        level, mode = "HIGH", "ACTIONABLE_ALLOWED"
    return {"confidence_level": level, "recommendation_mode": mode,
            "actionable_recommendation_allowed": level == "HIGH", "reasons": reasons}


def contains_actionable_recommendation(report: str) -> bool:
    return bool(_ACTION.search(report or ""))
