"""Evidence-linked aspect extraction for sentiment observations."""
from __future__ import annotations
import re
from typing import Any

ASPECTS = {"valuation": ("valuation", "expensive", "multiple", "priced"), "ai_demand": ("ai demand", "artificial intelligence", "accelerator", "gpu"), "competition": ("competition", "competitor", "market share"), "geopolitical": ("geopolitical", "export restriction", "china", "sanction"), "management": ("management", "guidance", "ceo", "executive"), "earnings": ("earnings", "revenue", "margin", "profit")}

_VALID_POLARITIES = {"BULLISH", "BEARISH", "NEUTRAL"}

_CAUSAL_MARKER = re.compile(
    r"\b(?:because|due to|driv(?:e|es|en|ing)|drove|lead(?:s|ing)? to|result(?:s|ed|ing)? in|"
    r"reflect(?:s|ed|ing)?|support(?:s|ed|ing)?|weigh(?:s|ed|ing)? on|boost(?:s|ed|ing)?|"
    r"hurt(?:s|ing)?|benefit(?:s|ed|ing)?|priced? in|pricing in|attribut(?:e|ed|es|ing) to)\b",
    re.I,
)

def extract_aspect_observations(chunks: list[str], sources: list[dict[str, Any]], chunk_scores: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    scores = chunk_scores or []; observations = []
    for index, chunk in enumerate(chunks):
        text = str(chunk); source = sources[index] if index < len(sources) else {}; score = scores[index] if index < len(scores) else {}
        for aspect, terms in ASPECTS.items():
            match = next((re.search(rf"[^.!?]*\b{re.escape(term)}\b[^.!?]*[.!?]?", text, re.I) for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.I)), None)
            if match:
                evidence = match.group(0).strip()[:500]
                observations.append({"document_url": source.get("url"), "document_title": source.get("title"), "aspect": aspect, "polarity": score.get("label"), "bullish_prob": score.get("bullish_prob"), "bearish_prob": score.get("bearish_prob"), "evidence_span": evidence, "causal_language_supported": bool(_CAUSAL_MARKER.search(evidence))})
    return observations


def valid_causal_observations(
    observations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return only observations that satisfy the causal evidence contract.

    Legacy observations that predate ``causal_language_supported`` remain
    readable as raw observations, but they are not promoted to causal evidence.
    They must be re-extracted or explicitly backfilled with a verified boolean
    flag before they can support a causal sentiment claim.
    """
    valid: list[dict[str, Any]] = []
    for row in observations or []:
        if not isinstance(row, dict):
            continue
        aspect = str(row.get("aspect") or "").strip()
        polarity = str(row.get("polarity") or "").strip().upper()
        evidence = str(row.get("evidence_span") or "").strip()
        source = str(row.get("document_url") or row.get("document_title") or "").strip()
        if (
            aspect in ASPECTS
            and polarity in _VALID_POLARITIES
            and evidence
            and source
            and row.get("causal_language_supported") is True
        ):
            valid.append(row)
    return valid


def supported_causal_aspects(observations: list[dict[str, Any]]) -> list[str]:
    return sorted({row["aspect"] for row in valid_causal_observations(observations)})
