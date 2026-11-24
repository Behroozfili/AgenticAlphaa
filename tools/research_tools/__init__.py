"""Point-in-time contracts shared by the Stage 04B research tools."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from inspect import signature
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError
from core.forecast_contract import ForecastContext, assert_not_after_as_of, parse_utc_datetime
from core.observability import elapsed_latency_ms, forecast_trace_metadata

RESEARCH_CONTRACT_VERSION = "research-features-v2"
RESEARCH_IMPLEMENTATION_VERSION = "stage04b-r1-a3"
RESEARCH_IMPLEMENTATION_IDENTITY: Mapping[str, Any] = MappingProxyType({
    "stage": "04B",
    "stage_revision": 1,
    "attempt": 3,
    "agent": "research",
    "implementation_version": RESEARCH_IMPLEMENTATION_VERSION,
})
HISTORICAL_PROVIDERS = frozenset({
    "gdelt_archive", "project_archive", "rag_vector", "rag_graph", "rag_hybrid",
    "rag_vector_search", "rag_graph_traverse", "rag_hybrid_query", "sec_archive",
    "sec_edgar_search", "sec_edgar_filing",
})
_TIME_FIELDS = ("filed_at", "published_at", "published_date", "publication_time", "retrieved_at", "timestamp")
_SAFE_OPERATIONS = frozenset({
    "tavily_search", "news_search", "sec_edgar_search", "sec_edgar_filing",
    "rag_vector_search", "rag_graph_traverse", "rag_hybrid_query",
    "comprehensive_analysis", "research_synthesis", "pre_state_leakage_gate",
})
_SAFE_OUTCOMES = frozenset({"SUCCESS", "DEGRADED", "DATA_UNAVAILABLE", "FAIL_CLOSED"})
_SAFE_REJECTION_CODES = frozenset({
    "FUTURE_EVIDENCE", "EVIDENCE_DATA_UNAVAILABLE", "EVIDENCE_TYPE_INVALID",
    "EVIDENCE_VALUE_INVALID", "HISTORICAL_PROVIDER_REJECTED",
})
_EVIDENCE_EXCERPT_MAX_CHARS = 6_000


def _evidence_content_excerpt(record: Mapping[str, Any]) -> str:
    '''Return bounded source text for synthesis without retaining full documents.'''
    existing = record.get('content_excerpt')
    if isinstance(existing, str) and existing.strip():
        return existing.strip()[:_EVIDENCE_EXCERPT_MAX_CHARS]
    parts: list[str] = []
    for key in (
        'summary', 'snippet', 'description', 'chunk_text', 'content', 'text',
        'name', 'type', 'relation',
    ):
        value = record.get(key)
        if value:
            parts.append(f'{key.upper()}: {str(value).strip()}')
    sections = record.get('sections')
    if isinstance(sections, Mapping):
        for name in sorted(sections, key=str):
            section = sections[name]
            if isinstance(section, Mapping):
                keys = ('preview', 'text', 'digest', 'important_excerpts', 'key_figures', 'query_match')
                for key in keys:
                    value = section.get(key)
                    if value:
                        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
                        parts.append(f'SECTION {name} {key}: {text}')
            elif section:
                parts.append(f'SECTION {name}: {section}')
    return '\n'.join(parts)[:_EVIDENCE_EXCERPT_MAX_CHARS]


def research_implementation_identity() -> dict[str, Any]:
    """Return the immutable Stage 04B implementation provenance as a plain mapping."""
    return dict(RESEARCH_IMPLEMENTATION_IDENTITY)


def research_trace_metadata(
    context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    **measurements: Any,
) -> dict[str, Any]:
    """Build allowlisted trace metadata from the single implementation identity."""
    if context is None:
        return research_implementation_identity()
    context_payload = context.to_dict() if isinstance(context, ForecastContext) else dict(context)
    return dict(forecast_trace_metadata(
        context_payload,
        run_id=run_id,
        stage=str(RESEARCH_IMPLEMENTATION_IDENTITY["stage"]),
        stage_revision=int(RESEARCH_IMPLEMENTATION_IDENTITY["stage_revision"]),
        attempt=int(RESEARCH_IMPLEMENTATION_IDENTITY["attempt"]),
        agent=str(RESEARCH_IMPLEMENTATION_IDENTITY["agent"]),
        **measurements,
    ), implementation_version=RESEARCH_IMPLEMENTATION_VERSION)


class ResearchDataUnavailable(DataUnavailableAsOfError):
    """A reproducible point-in-time research source is unavailable."""


class HistoricalProviderRejected(ResearchDataUnavailable):
    """Evidence came from a provider forbidden for historical execution."""


def _rejection_code(exc: Exception) -> str:
    if isinstance(exc, TemporalLeakageError):
        return "FUTURE_EVIDENCE"
    if isinstance(exc, HistoricalProviderRejected):
        return "HISTORICAL_PROVIDER_REJECTED"
    if isinstance(exc, ResearchDataUnavailable):
        return "EVIDENCE_DATA_UNAVAILABLE"
    if isinstance(exc, TypeError):
        return "EVIDENCE_TYPE_INVALID"
    return "EVIDENCE_VALUE_INVALID"


def rejection_counts(rejected: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(item.get("reason_code")) for item in rejected
        if item.get("reason_code") in _SAFE_REJECTION_CODES
    )
    return {key: counts[key] for key in sorted(counts)}


def citation_reference_id(citation: Any) -> str:
    payload = citation if isinstance(citation, Mapping) else {}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return f"citation:{digest}"


def compact_lifecycle(
    *, context: ForecastContext, run_id: str, operation: str,
    latency_ms: float, payload: Mapping[str, Any], input_count: int = 1,
) -> dict[str, Any]:
    if operation not in _SAFE_OPERATIONS:
        raise ValueError("unsupported research lifecycle operation")
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    results = next((
        payload.get(key) for key in ("results", "articles", "filings", "nodes")
        if isinstance(payload.get(key), list)
    ), [])
    citations = payload.get("citations") if isinstance(payload.get("citations"), list) else []
    if not citations:
        citations = [
            item.get("citation") for item in evidence
            if isinstance(item, Mapping) and isinstance(item.get("citation"), Mapping)
        ]
    if evidence:
        evidence_count = len(evidence)
    else:
        evidence_count = len({
            str(item.get("evidence_id"))
            for item in (*results, *citations)
            if isinstance(item, Mapping) and item.get("evidence_id")
        })
    rejected = payload.get("quarantined") if isinstance(payload.get("quarantined"), list) else []
    degraded = bool(payload.get("degraded"))
    status = payload.get("status")
    if status == "DATA_UNAVAILABLE":
        outcome = "DATA_UNAVAILABLE"
    elif payload.get("error") or payload.get("error_code"):
        outcome = "FAIL_CLOSED"
    elif degraded:
        outcome = "DEGRADED"
    else:
        outcome = "SUCCESS"
    if outcome not in _SAFE_OUTCOMES:
        outcome = "FAIL_CLOSED"
    return {
        **research_trace_metadata(
            context, run_id, latency_ms=latency_ms, evidence_count=evidence_count,
        ),
        "operation": operation,
        "provider": operation,
        "outcome": outcome,
        "input_count": max(0, int(input_count)),
        "result_count": len(results) if results else len(evidence),
        "evidence_count": evidence_count,
        "citation_count": len(citations),
        "quarantine_count": len(rejected),
        "rejection_counts": rejection_counts(rejected),
        "degraded": degraded or outcome != "SUCCESS",
        "resume": "not_applicable",
    }


def observed_research_operation(operation: str):
    """Attach measured compact lifecycle metadata without capturing inputs."""
    if operation not in _SAFE_OPERATIONS:
        raise ValueError("unsupported research lifecycle operation")
    def decorate(func):
        func_signature = signature(func)
        @wraps(func)
        async def wrapped(*args, **kwargs):
            started_at = time.perf_counter()
            bound = func_signature.bind_partial(*args, **kwargs)
            context, run_id = require_research_context(
                bound.arguments.get("forecast_context"), bound.arguments.get("run_id")
            )
            try:
                result = await func(*args, **kwargs)
            except Exception:
                result = {
                    **context_metadata(context, run_id),
                    "status": "FAIL_CLOSED",
                    "error_code": "RESEARCH_PROVIDER_FAIL_CLOSED",
                    "evidence": [], "missing": True, "degraded": True,
                }
            if not isinstance(result, dict):
                return result
            result["telemetry"] = compact_lifecycle(
                context=context, run_id=run_id, operation=operation,
                latency_ms=elapsed_latency_ms(started_at), payload=result,
            )
            return result
        return wrapped
    return decorate


def require_research_context(
    forecast_context: ForecastContext | Mapping[str, Any] | None,
    run_id: str | None,
) -> tuple[ForecastContext, str]:
    if forecast_context is None:
        raise ValueError("forecast_context is required for every research execution")
    context = forecast_context if isinstance(forecast_context, ForecastContext) else ForecastContext.from_dict(dict(forecast_context))
    normalized_run_id = str(run_id or "").strip()
    if not normalized_run_id:
        raise ValueError("run_id is required for every research execution")
    return context, normalized_run_id


def context_metadata(context: ForecastContext, run_id: str) -> dict[str, Any]:
    return {
        "ticker": context.ticker,
        "as_of": context.as_of.isoformat().replace("+00:00", "Z"),
        "lookback_days": context.lookback_days,
        "dataset_version": context.dataset_version,
        "analysis_mode": context.analysis_mode.value,
        "run_id": run_id,
    }


def historical_unavailable(context: ForecastContext, run_id: str, *, source: str, reason: str) -> dict[str, Any]:
    return {
        **context_metadata(context, run_id),
        "status": "DATA_UNAVAILABLE",
        "error_code": "HISTORICAL_PROVIDER_UNAVAILABLE",
        "source": source,
        "reason_code": "HISTORICAL_PROVIDER_UNAVAILABLE",
        "evidence": [],
        "missing": True,
        "degraded": True,
    }


def _canonical_timestamp(record: Mapping[str, Any]) -> tuple[str, datetime]:
    for field in _TIME_FIELDS:
        value = record.get(field)
        if value:
            text = str(value)
            if len(text) == 10:
                # SEC/GDELT archives may retain only the publication date. The
                # contract admits a filing whose filed date is on/before the
                # cutoff date; use start-of-day rather than inventing a time
                # after the market close.
                text += "T00:00:00Z"
            return field, parse_utc_datetime(text, field_name=field)
    raise ResearchDataUnavailable("point-in-time evidence requires a known filed/published timestamp")


def deterministic_evidence_id(record: Mapping[str, Any], *, context: ForecastContext, source: str) -> str:
    identity = {
        "ticker": context.ticker,
        "dataset_version": context.dataset_version,
        "source": source,
        "source_identity": record.get("evidence_id") or record.get("accession_number") or record.get("document_id") or record.get("chunk_id") or record.get("observation_id") or record.get("url") or record.get("source_url"),
        "timestamp": next((record.get(key) for key in _TIME_FIELDS if record.get(key)), None),
        "title": record.get("title") or record.get("company") or record.get("name"),
        "citation": record.get("citation"),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    return f"research:{digest}"


def normalize_evidence(
    record: Mapping[str, Any], *, context: ForecastContext, run_id: str,
    source: str, historical: bool = False,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("evidence record must be a mapping")
    record_ticker = str(record.get("ticker") or context.ticker).strip().upper()
    if record_ticker != context.ticker:
        raise ResearchDataUnavailable("evidence ticker does not match ForecastContext.ticker")
    record_version = str(record.get("dataset_version") or context.dataset_version).strip()
    if historical and not record.get("dataset_version"):
        raise ResearchDataUnavailable("historical evidence requires an explicit dataset_version")
    if record_version != context.dataset_version:
        raise ResearchDataUnavailable("evidence dataset_version does not match ForecastContext")
    timestamp_field, timestamp = _canonical_timestamp(record)
    assert_not_after_as_of(timestamp, context.as_of, field_name=timestamp_field)
    if timestamp < context.as_of - timedelta(days=context.lookback_days):
        raise ResearchDataUnavailable(
            f"{timestamp_field} is outside ForecastContext.lookback_days"
        )
    record_source = str(record.get("research_source") or record.get("source") or source).strip().lower()
    provider = str(record.get("provider") or record_source).strip().lower()
    if historical and provider not in HISTORICAL_PROVIDERS:
        raise HistoricalProviderRejected("historical provider is not allowlisted")
    evidence_id = str(record.get("evidence_id") or "").strip() or deterministic_evidence_id(record, context=context, source=record_source)
    citation = record.get("citation") or {
        "source": record_source,
        "url": record.get("url") or record.get("source_url"),
        "accession_number": record.get("accession_number"),
        "document_id": record.get("document_id"),
        "chunk_id": record.get("chunk_id"),
    }
    raw_origins = record.get("retrieval_origins")
    if not isinstance(raw_origins, (list, tuple, set)):
        raw_origins = record.get("origins")
    if not isinstance(raw_origins, (list, tuple, set)):
        raw_origins = [record.get("origin")]
    normalized = {
        "evidence_id": evidence_id,
        "source": record_source,
        "provider": provider,
        "ticker": context.ticker,
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "timestamp_field": timestamp_field,
        "dataset_version": context.dataset_version,
        "run_id": run_id,
        "citation": citation,
        "title": str(record.get("title") or record.get("company") or record.get("name") or "")[:512],
        "kind": str(record.get("kind") or record.get("claim_type") or "evidence")[:64],
        "retrieval_origins": sorted({
            str(origin).strip().lower()
            for origin in raw_origins
            if str(origin or "").strip().lower() in {"vector", "graph"}
        }),
    }
    content_excerpt = _evidence_content_excerpt(record)
    if content_excerpt:
        normalized['content_excerpt'] = content_excerpt
    return normalized


def eligible_evidence(
    records: Iterable[Mapping[str, Any]], *, context: ForecastContext, run_id: str,
    source: str, historical: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for record in records:
        try:
            compact = normalize_evidence(record, context=context, run_id=run_id, source=source, historical=historical)
        except (ResearchDataUnavailable, TemporalLeakageError, TypeError, ValueError) as exc:
            rejected.append({"reason_code": _rejection_code(exc)})
            continue
        accepted.setdefault(compact["evidence_id"], compact)
    return [accepted[key] for key in sorted(accepted)], rejected


@dataclass(frozen=True, slots=True)
class ResearchFeatures:
    catalysts: tuple[str, ...]
    risks: tuple[str, ...]
    timestamps: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    citation_reference_ids: tuple[str, ...]
    source_diversity: int
    claims: tuple[Mapping[str, Any], ...]
    directional_score: float
    claim_confidence: float
    classification_coverage: float
    classified_claim_count: int
    neutral_claim_count: int
    vector_coverage: bool
    graph_coverage: bool
    missing: bool
    degraded: bool
    dataset_version: str
    run_id: str

    def to_state_slot(self) -> dict[str, Any]:
        return {
            "version": RESEARCH_CONTRACT_VERSION,
            "values": {
                "catalysts": list(self.catalysts), "risks": list(self.risks),
                "timestamps": list(self.timestamps),
                "citation_reference_ids": list(self.citation_reference_ids),
                "source_diversity": self.source_diversity,
                "claims": [dict(claim) for claim in self.claims],
                "directional_score": self.directional_score,
                "claim_confidence": self.claim_confidence,
                "classification_coverage": self.classification_coverage,
                "classified_claim_count": self.classified_claim_count,
                "neutral_claim_count": self.neutral_claim_count,
                "vector_coverage": self.vector_coverage, "graph_coverage": self.graph_coverage,
                "dataset_version": self.dataset_version, "run_id": self.run_id,
            },
            "evidence_ids": list(self.evidence_ids),
            "missing": self.missing, "degraded": self.degraded, "errors": [],
        }


def build_research_features(
    evidence: Iterable[Mapping[str, Any]], *, context: ForecastContext, run_id: str,
) -> ResearchFeatures:
    by_id = {str(item["evidence_id"]): dict(item) for item in evidence if item.get("evidence_id")}
    ordered = [by_id[key] for key in sorted(by_id)]
    from tools.research_tools.claim_classifier import classify_research_evidence

    classified = classify_research_evidence(ordered, context=context)
    claims_with_citations = []
    for claim in classified:
        item = by_id[claim["evidence_id"]]
        claims_with_citations.append({
            **claim,
            "citation_reference_id": citation_reference_id(item.get("citation")),
        })
    directional_claims = [
        claim for claim in claims_with_citations
        if claim["direction"] in {"POSITIVE", "NEGATIVE"}
    ]
    signed_weight = sum(
        (1.0 if claim["direction"] == "POSITIVE" else -1.0)
        * float(claim["strength"])
        * float(claim["confidence"])
        * float(claim["recency_weight"])
        for claim in directional_claims
    )
    weight_denominator = sum(
        float(claim["confidence"]) * float(claim["recency_weight"])
        for claim in directional_claims
    )
    directional_score = signed_weight / weight_denominator if weight_denominator else 0.0
    recency_denominator = sum(float(claim["recency_weight"]) for claim in directional_claims)
    claim_confidence = (
        sum(float(claim["confidence"]) * float(claim["recency_weight"])
            for claim in directional_claims) / recency_denominator
        if recency_denominator else 0.0
    )
    top_claims = tuple(sorted(
        directional_claims,
        key=lambda claim: (
            -float(claim["strength"]) * float(claim["confidence"]) * float(claim["recency_weight"]),
            claim["evidence_id"],
        ),
    )[:12])
    sources = {str(item.get("source") or "unknown") for item in ordered}
    catalysts = tuple(
        claim["evidence_id"] for claim in directional_claims
        if claim["direction"] == "POSITIVE"
    )[:20]
    risks = tuple(
        claim["evidence_id"] for claim in directional_claims
        if claim["direction"] == "NEGATIVE"
    )[:20]
    vector_coverage = any(
        item.get("source") == "rag_vector_search"
        or (
            item.get("source") == "rag_hybrid_query"
            and "vector" in item.get("retrieval_origins", ())
        )
        for item in ordered
    )
    graph_coverage = any(
        item.get("source") == "rag_graph_traverse"
        or (
            item.get("source") == "rag_hybrid_query"
            and "graph" in item.get("retrieval_origins", ())
        )
        for item in ordered
    )
    missing = not ordered
    degraded = missing or not (vector_coverage and graph_coverage) or len(sources) < 2
    return ResearchFeatures(
        catalysts=catalysts, risks=risks,
        timestamps=tuple(sorted({str(item["timestamp"]) for item in ordered})),
        evidence_ids=tuple(sorted(by_id)),
        citation_reference_ids=tuple(sorted({citation_reference_id(item.get("citation")) for item in ordered})),
        source_diversity=len(sources), claims=top_claims,
        directional_score=max(-1.0, min(1.0, directional_score)),
        claim_confidence=max(0.0, min(1.0, claim_confidence)),
        classification_coverage=(len(directional_claims) / len(ordered) if ordered else 0.0),
        classified_claim_count=len(directional_claims),
        neutral_claim_count=len(ordered) - len(directional_claims),
        vector_coverage=vector_coverage, graph_coverage=graph_coverage,
        missing=missing, degraded=degraded, dataset_version=context.dataset_version, run_id=run_id,
    )

__all__ = [
    "HISTORICAL_PROVIDERS", "RESEARCH_CONTRACT_VERSION",
    "RESEARCH_IMPLEMENTATION_IDENTITY", "RESEARCH_IMPLEMENTATION_VERSION",
    "ResearchDataUnavailable",
    "HistoricalProviderRejected",
    "ResearchFeatures", "build_research_features", "citation_reference_id",
    "compact_lifecycle", "context_metadata",
    "deterministic_evidence_id", "eligible_evidence", "historical_unavailable",
    "normalize_evidence", "observed_research_operation", "rejection_counts",
    "research_implementation_identity", "research_trace_metadata",
    "require_research_context",
]
