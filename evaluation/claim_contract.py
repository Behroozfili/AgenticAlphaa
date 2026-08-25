"""Typed claim contract and deterministic provenance validation."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ClaimType(str, Enum):
    FACT = "FACT"
    MANAGEMENT_STATEMENT = "MANAGEMENT_STATEMENT"
    ANALYST_ESTIMATE = "ANALYST_ESTIMATE"
    INFERENCE = "INFERENCE"
    ASSUMPTION = "ASSUMPTION"
    MODEL_OUTPUT = "MODEL_OUTPUT"


_CLAIM_TYPE_ALIASES = {
    "MODEL_INFERENCE": ClaimType.MODEL_OUTPUT,
    "VALUATION_MODEL_INFERENCE": ClaimType.MODEL_OUTPUT,
    "VALUATION_INFERENCE": ClaimType.MODEL_OUTPUT,
    "ESTIMATE": ClaimType.ANALYST_ESTIMATE,
}


def normalize_claim_type(value: Any, *, default: ClaimType = ClaimType.FACT) -> ClaimType:
    """Normalize legacy labels to the project's current ClaimType enum."""
    token = str(value or default.value).strip().upper()
    if token in _CLAIM_TYPE_ALIASES:
        return _CLAIM_TYPE_ALIASES[token]
    try:
        return ClaimType(token)
    except ValueError:
        return default


def validate_claim(claim: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        kind = ClaimType(claim.get("claim_type"))
    except ValueError:
        return ["invalid claim_type"]
    if not str(claim.get("text") or "").strip():
        errors.append("claim text is required")
    if kind in {ClaimType.FACT, ClaimType.MANAGEMENT_STATEMENT, ClaimType.ANALYST_ESTIMATE}:
        if not (claim.get("source_url") or claim.get("accession_number")):
            errors.append(f"{kind.value} requires a source")
    if kind == ClaimType.MANAGEMENT_STATEMENT:
        if not (claim.get("speaker") or claim.get("attributable_entity")):
            errors.append("MANAGEMENT_STATEMENT requires speaker")
    if kind == ClaimType.ASSUMPTION and not claim.get("derivation"):
        errors.append("ASSUMPTION requires derivation")
    if kind == ClaimType.MODEL_OUTPUT and not (
        claim.get("derivation") or claim.get("formula") or claim.get("model_name")
    ):
        errors.append("MODEL_OUTPUT requires derivation, formula, or model_name")
    return errors


def typed_financial_claim(fact: dict[str, Any]) -> dict[str, Any]:
    return {**fact, "claim_type": ClaimType.FACT.value,
            "text": f"{fact.get('metric')} = {fact.get('value')} {fact.get('unit') or ''}".strip(),
            "validation_errors": []}
