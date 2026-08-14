"""Evidence-linked aspect extraction for sentiment observations."""
from __future__ import annotations
import re
from typing import Any

ASPECTS = {"valuation": ("valuation", "expensive", "multiple", "priced"), "ai_demand": ("ai demand", "artificial intelligence", "accelerator", "gpu"), "competition": ("competition", "competitor", "market share"), "geopolitical": ("geopolitical", "export restriction", "china", "sanction"), "management": ("management", "guidance", "ceo", "executive"), "earnings": ("earnings", "revenue", "margin", "profit")}

def extract_aspect_observations(chunks: list[str], sources: list[dict[str, Any]], chunk_scores: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    scores = chunk_scores or []; observations = []
    for index, chunk in enumerate(chunks):
        text = str(chunk); source = sources[index] if index < len(sources) else {}; score = scores[index] if index < len(scores) else {}
        for aspect, terms in ASPECTS.items():
            match = next((re.search(rf"[^.!?]*\b{re.escape(term)}\b[^.!?]*[.!?]?", text, re.I) for term in terms if re.search(rf"\b{re.escape(term)}\b", text, re.I)), None)
            if match:
                observations.append({"document_url": source.get("url"), "document_title": source.get("title"), "aspect": aspect, "polarity": score.get("label"), "bullish_prob": score.get("bullish_prob"), "bearish_prob": score.get("bearish_prob"), "evidence_span": match.group(0).strip()[:500], "causal_language_supported": True})
    return observations

def supported_causal_aspects(observations: list[dict[str, Any]]) -> list[str]:
    return sorted({row["aspect"] for row in observations if row.get("evidence_span") and row.get("polarity")})
