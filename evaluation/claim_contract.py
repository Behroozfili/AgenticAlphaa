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
    if kind == ClaimType.MANAGEMENT_STATEMENT and not claim.get("speaker"):
        errors.append("MANAGEMENT_STATEMENT requires speaker")
    if kind == ClaimType.ASSUMPTION and not claim.get("derivation"):
        errors.append("ASSUMPTION requires derivation")
    return errors


def typed_financial_claim(fact: dict[str, Any]) -> dict[str, Any]:
    return {**fact, "claim_type": ClaimType.FACT.value,
            "text": f"{fact.get('metric')} = {fact.get('value')} {fact.get('unit') or ''}".strip(),
            "validation_errors": []}
