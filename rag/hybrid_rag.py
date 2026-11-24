"""ForecastContext-bound Vector, Graph, and hybrid retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Optional

from langsmith import traceable

from core.forecast_contract import AnalysisMode, ForecastContext
from core.observability import elapsed_latency_ms, forecast_trace_metadata, sentry_enabled

logger = logging.getLogger(__name__)

_TRACE_IDENTITY = {
    "stage": "04A", "stage_revision": 1, "attempt": 2, "agent": "data_rag",
}
_SAFE_ERROR_CODES = frozenset({
    "VECTOR_UNAVAILABLE", "VECTOR_RETRIEVAL_FAILED",
    "GRAPH_UNAVAILABLE", "GRAPH_TRAVERSAL_FAILED", "GRAPH_NOT_REQUESTED",
})

_neo4j_driver = None


def _get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver
    try:
        from neo4j import AsyncGraphDatabase
        _neo4j_driver = AsyncGraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
        )
        return _neo4j_driver
    except Exception:
        logger.warning("GRAPH_DRIVER_UNAVAILABLE")
        return None


try:
    from rag.embedding_manager import get_embedder as _get_embedder
    _embedder = _get_embedder()
except Exception:
    _embedder = None
    logger.warning("EMBEDDER_UNAVAILABLE")

try:
    from rag.retriever import AlphaRetriever
    from rag.vector_store import AlphaVectorStore
    _vector_store = AlphaVectorStore(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"],
    )
    _retriever = AlphaRetriever(vector_store=_vector_store, embedder=_embedder)
except Exception:
    _retriever = None
    logger.warning("VECTOR_RETRIEVER_UNAVAILABLE")


def _context(value: ForecastContext | dict[str, Any] | None) -> ForecastContext | None:
    if value is None or isinstance(value, ForecastContext):
        return value
    if isinstance(value, dict):
        return ForecastContext.from_dict(value)
    raise TypeError("context must be ForecastContext, its serialized dictionary, or None")


def _citation(item: dict[str, Any], *, origin: str) -> dict[str, Any]:
    return {
        "origin": origin,
        "document_id": item.get("document_id"),
        "chunk_id": item.get("chunk_id") or item.get("id"),
        "evidence_id": item.get("evidence_id"),
        "observation_id": item.get("observation_id"),
        "url": item.get("url") or item.get("source_url") or "",
        "title": item.get("title") or "",
        "published_at": item.get("published_at"),
        "retrieved_at": item.get("retrieved_at"),
        "effective_from": item.get("effective_from"),
        "effective_to": item.get("effective_to"),
        "dataset_version": item.get("dataset_version"),
    }


def _json_temporal(value: Any) -> Any:
    """Return a JSON-safe representation for Neo4j temporal values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for method_name in ("isoformat", "iso_format"):
        method = getattr(value, method_name, None)
        if callable(method):
            rendered = method()
            if rendered:
                return str(rendered)
    return str(value)

def _branch_payload(name: str, *, available: bool, reason: str | None = None) -> dict[str, Any]:
    return {"branch": name, "available": available, "degraded": not available, "reason": reason}


def _compact_context(value: Any) -> dict[str, Any]:
    try:
        ctx = _context(value)
    except (TypeError, ValueError):
        ctx = None
    return dict(forecast_trace_metadata(
        ctx.to_dict() if ctx else {},
        stage=_TRACE_IDENTITY["stage"],
        stage_revision=_TRACE_IDENTITY["stage_revision"],
        attempt=_TRACE_IDENTITY["attempt"],
        agent=_TRACE_IDENTITY["agent"],
    ))


def _trace_inputs(operation: str):
    def process(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation": operation,
            "metadata": _compact_context(inputs.get("context")),
            "resume": "not_applicable",
        }
    return process


def _safe_code(value: Any) -> str | None:
    return value if isinstance(value, str) and value in _SAFE_ERROR_CODES else None


def _trace_outputs(operation: str):
    def process(outputs: Any) -> dict[str, Any]:
        payload = outputs if isinstance(outputs, Mapping) else {}
        results = payload.get("results") or payload.get("nodes") or []
        citations = payload.get("citations") or []
        degradation = payload.get("degradation") or {}
        telemetry = payload.get("telemetry") or {}
        evidence_ids = {
            item.get("evidence_id") for item in results
            if isinstance(item, Mapping) and item.get("evidence_id")
        }
        safe: dict[str, Any] = {
            "operation": operation,
            "metadata": _compact_context(payload.get("context")),
            "result_count": len(results) if isinstance(results, list) else 0,
            "evidence_count": len(evidence_ids),
            "citation_count": len(citations) if isinstance(citations, list) else 0,
            "degraded": bool(degradation.get("degraded", False)),
            "branch_available": bool(degradation.get("available", True)),
            "resume": "not_applicable",
        }
        mode = degradation.get("mode")
        if mode in {"hybrid", "vector_only", "graph_only", "unavailable"}:
            safe["degradation_mode"] = mode
        code = _safe_code(payload.get("error_code"))
        if code:
            safe["error_code"] = code
        for field in (
            "latency_ms", "vector_latency_ms", "graph_latency_ms",
            "rrf_latency_ms", "fusion_latency_ms", "hybrid_latency_ms",
        ):
            value = telemetry.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                safe[field] = round(float(value), 3)
        return safe
    return process


def _finalize_branch(
    payload: dict[str, Any], *, branch: str, started_at: float,
    context: ForecastContext | None,
) -> dict[str, Any]:
    results = payload.get("results") or payload.get("nodes") or []
    citations = payload.get("citations") or []
    evidence_count = len({
        item.get("evidence_id") for item in results
        if isinstance(item, Mapping) and item.get("evidence_id")
    })
    degradation = payload.get("degradation") or {}
    reason = degradation.get("reason")
    flags = [reason] if isinstance(reason, str) and reason else []
    metadata = forecast_trace_metadata(
        context.to_dict() if context else {},
        stage=_TRACE_IDENTITY["stage"],
        stage_revision=_TRACE_IDENTITY["stage_revision"],
        attempt=_TRACE_IDENTITY["attempt"],
        agent=_TRACE_IDENTITY["agent"],
        latency_ms=elapsed_latency_ms(started_at),
        evidence_count=evidence_count,
        degraded_flags=flags,
    )
    payload["telemetry"] = {
        **metadata,
        "operation": branch,
        "result_count": len(results),
        "citation_count": len(citations),
        "degraded": bool(degradation.get("degraded", False)),
        "resume": "not_applicable",
    }
    return payload


@traceable(
    run_type="retriever", name="stage04a_vector_retrieval",
    metadata=_TRACE_IDENTITY, process_inputs=_trace_inputs("vector"),
    process_outputs=_trace_outputs("vector"), exceptions_to_handle=(Exception,),
)
async def rag_vector_search(
    query: str,
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
    days_back: Optional[int] = None,
    threshold: float = 0.01,
    context: ForecastContext | dict[str, Any] | None = None,
) -> dict:
    """Search Vector with an optional point-in-time ForecastContext boundary."""
    started_at = time.perf_counter()
    ctx = _context(context)
    ticker = ctx.ticker if ctx else (ticker_filter.upper() if ticker_filter else None)
    if ctx and ticker_filter and ticker_filter.upper() != ctx.ticker:
        raise ValueError("ticker_filter must match ForecastContext.ticker")
    if not _retriever:
        return _finalize_branch({
            "query": query, "results": [], "citations": [],
            "warning": "AlphaRetriever not configured",
            "error_code": "VECTOR_UNAVAILABLE",
            "degradation": _branch_payload("vector", available=False, reason="vector_unavailable"),
        }, branch="vector", started_at=started_at, context=ctx)
    kwargs: dict[str, Any] = {
        "query": query, "ticker": ticker, "days_back": days_back,
        "score_threshold": threshold,
    }
    supports_temporal = getattr(
        getattr(_retriever, "store", None), "supports_temporal_boundary", True
    )
    legacy_live = bool(
        ctx
        and ctx.analysis_mode is AnalysisMode.LIVE
        and supports_temporal is False
    )
    if ctx:
        kwargs.update(
            as_of=ctx.as_of,
            dataset_version=ctx.dataset_version,
            historical=not legacy_live,
        )
    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="rag.vector_search", message="Vector retrieval",
            data={"ticker": ticker, "historical": bool(ctx)}, level="info",
        )
    try:
        raw_chunks = await asyncio.to_thread(_retriever.retrieve_raw, **kwargs)
    except Exception:
        logger.error("VECTOR_RETRIEVAL_FAILED")
        return _finalize_branch({
            "query": query, "results": [], "citations": [],
            "error_code": "VECTOR_RETRIEVAL_FAILED",
            "degradation": _branch_payload("vector", available=False, reason="vector_error"),
        }, branch="vector", started_at=started_at, context=ctx)
    results = []
    for raw in raw_chunks[:top_k]:
        item = {
            "id": raw.get("id"),
            "document_id": raw.get("document_id"),
            "chunk_id": raw.get("chunk_id") or raw.get("id"),
            "evidence_id": raw.get("evidence_id"),
            "ticker": raw.get("ticker"),
            "dataset_version": raw.get("dataset_version"),
            "source_type": raw.get("source_type"),
            "chunk_text": raw.get("text"),
            "score": round(raw.get("freshness_score", raw.get("rrf_score", 0.0)), 6),
            "published_at": raw.get("published_at"),
            "retrieved_at": raw.get("retrieved_at"),
            "effective_from": raw.get("effective_from"),
            "effective_to": raw.get("effective_to"),
            "url": raw.get("url"),
            "title": raw.get("title"),
            "chunk_index": raw.get("chunk_index"),
        }
        item["citation"] = _citation(item, origin="vector")
        results.append(item)
    degradation = _branch_payload("vector", available=True)
    if legacy_live:
        degradation = {
            "branch": "vector", "available": True, "degraded": True,
            "reason": "legacy_live_temporal_projection",
        }
    return _finalize_branch({
        "query": query,
        "context": ctx.to_dict() if ctx else None,
        "results": results,
        "citations": [item["citation"] for item in results],
        "degradation": degradation,
    }, branch="vector", started_at=started_at, context=ctx)


@traceable(
    run_type="retriever", name="stage04a_graph_retrieval",
    metadata=_TRACE_IDENTITY, process_inputs=_trace_inputs("graph"),
    process_outputs=_trace_outputs("graph"), exceptions_to_handle=(Exception,),
)
async def rag_graph_traverse(
    entity: str,
    relation_types: Optional[list[str]] = None,
    max_hops: int = 2,
    limit: int = 20,
    context: ForecastContext | dict[str, Any] | None = None,
) -> dict:
    """Traverse legacy Graph or fail-closed temporal Graph when context is supplied."""
    started_at = time.perf_counter()
    ctx = _context(context)
    relation_types = relation_types or ["ALL"]
    max_hops = max(1, min(max_hops, 3))
    driver = _get_neo4j()
    if not driver:
        return _finalize_branch({
            "entity": entity, "nodes": [], "paths": [], "citations": [],
            "warning": "Neo4j not configured",
            "error_code": "GRAPH_UNAVAILABLE",
            "degradation": _branch_payload("graph", available=False, reason="graph_unavailable"),
        }, branch="graph", started_at=started_at, context=ctx)
    if "ALL" in relation_types:
        rel_clause = f"[*1..{max_hops}]"
    else:
        invalid = set(relation_types) - {
            "COMPETES_WITH", "SUPPLIES_TO", "AFFECTED_BY", "LED_BY",
            "PART_OF", "RELATED_TO", "ACQUIRED_BY",
        }
        if invalid:
            raise ValueError(f"unsupported relation types: {sorted(invalid)}")
        rel_clause = f"[:{'|'.join(relation_types)}*1..{max_hops}]"

    if ctx:
        # Eligibility is resolved before observations are ranked or returned.
        cypher = """
        MATCH (e:EvidenceObservationV2)-[:SUPPORTS_OBSERVATION]->
              (o:TemporalRelationObservationV2)
        WHERE e.is_quarantined = false
          AND ($all_relations OR o.relation_type IN $relation_types)
          AND e.ticker = $ticker
          AND e.dataset_version = $dataset_version
          AND e.published_at IS NOT NULL AND e.published_at <= datetime($as_of)
          AND e.retrieved_at IS NOT NULL AND e.retrieved_at <= datetime($as_of)
          AND (e['filed_at'] IS NULL OR e['filed_at'] <= datetime($as_of))
          AND e.effective_from IS NOT NULL AND e.effective_from <= datetime($as_of)
          AND (e['effective_to'] IS NULL OR e['effective_to'] > datetime($as_of))
        MATCH (o)-[:SOURCE_ENTITY|TARGET_ENTITY]->(s:TemporalEntityV2)
        WHERE toUpper(s.name) = $entity OR toUpper(s.ticker) = $entity
        MATCH (o)-[:SOURCE_ENTITY|TARGET_ENTITY]->(target:TemporalEntityV2)
        WHERE target <> s
        MATCH (e)-[:OBSERVES_CHUNK]->(chunk:AlphaChunkV2)
        OPTIONAL MATCH (document:AlphaDocumentV2)-[:HAS_CHUNK]->(chunk)
        RETURN target.name AS name, target.type AS type,
               o.relation_type AS relation, 1 AS hops,
               [s.name, target.name] AS path_nodes,
               o.observation_id AS observation_id,
               e.evidence_id AS evidence_id, chunk.chunk_id AS chunk_id,
               document.document_id AS document_id,
               document.source_url AS source_url,
               e.published_at AS published_at, e.retrieved_at AS retrieved_at,
               e.effective_from AS effective_from, e['effective_to'] AS effective_to,
               e.dataset_version AS dataset_version
        ORDER BY e.published_at DESC, o.observation_id ASC
        LIMIT $limit
        """
        params = {
            "entity": entity.upper(), "ticker": ctx.ticker,
            "dataset_version": ctx.dataset_version, "as_of": ctx.as_of.isoformat(),
            "all_relations": "ALL" in relation_types,
            "relation_types": relation_types,
            "limit": limit,
        }
    else:
        cypher = f"""
        MATCH path = (s {{name: $entity}})-{rel_clause}-(e)
        RETURN e.name AS name, labels(e)[0] AS type,
               type(last(relationships(path))) AS relation,
               length(path) AS hops, [n IN nodes(path) | n.name] AS path_nodes
        ORDER BY hops ASC LIMIT $limit
        """
        params = {"entity": entity.upper(), "limit": limit}

    nodes: list[dict[str, Any]] = []
    paths: list[list[Any]] = []
    try:
        async with driver.session() as session:
            result = await session.run(cypher, **params)
            async for record in result:
                item = {
                    "name": record["name"], "type": record["type"],
                    "relation": record["relation"], "hops": record["hops"],
                }
                if ctx:
                    for key in (
                        "observation_id", "evidence_id", "chunk_id", "document_id",
                        "source_url", "published_at", "retrieved_at", "effective_from",
                        "effective_to", "dataset_version",
                    ):
                        value = record[key]
                        item[key] = (
                            _json_temporal(value)
                            if key in {
                                "published_at", "retrieved_at", "effective_from",
                                "effective_to",
                            }
                            else value
                        )
                    item["citation"] = _citation(item, origin="graph")
                nodes.append(item)
                paths.append(record["path_nodes"])
    except Exception:
        logger.error("GRAPH_TRAVERSAL_FAILED")
        return _finalize_branch({
            "entity": entity, "nodes": [], "paths": [], "citations": [],
            "error_code": "GRAPH_TRAVERSAL_FAILED",
            "degradation": _branch_payload("graph", available=False, reason="graph_error"),
        }, branch="graph", started_at=started_at, context=ctx)
    return _finalize_branch({
        "entity": entity, "context": ctx.to_dict() if ctx else None,
        "nodes": nodes, "paths": paths,
        "citations": [node["citation"] for node in nodes if "citation" in node],
        "degradation": _branch_payload("graph", available=True),
    }, branch="graph", started_at=started_at, context=ctx)


@traceable(
    run_type="retriever", name="stage04a_hybrid_retrieval",
    metadata=_TRACE_IDENTITY, process_inputs=_trace_inputs("hybrid"),
    process_outputs=_trace_outputs("hybrid"), exceptions_to_handle=(Exception,),
)
async def rag_hybrid_query(
    query: str,
    entity: Optional[str] = None,
    top_k: int = 5,
    max_hops: int = 2,
    fusion: str = "rrf",
    days_back: Optional[int] = None,
    context: ForecastContext | dict[str, Any] | None = None,
) -> dict:
    """Fuse Vector and Graph results under one immutable ForecastContext."""
    started_at = time.perf_counter()
    ctx = _context(context)
    resolved_entity = entity or (ctx.ticker if ctx else _extract_ticker_from_query(query))
    if ctx and entity and entity.upper() != ctx.ticker:
        raise ValueError("entity must match ForecastContext.ticker")
    if fusion not in {"rrf", "weighted", "union"}:
        raise ValueError("fusion must be rrf, weighted, or union")

    if not resolved_entity:
        vector = await rag_vector_search(
            query=query, top_k=top_k, days_back=days_back, context=ctx
        )
        items = _vector_items(vector)
        return _hybrid_response(
            query=query, entity=None, fusion="vector_only", context=ctx,
            results=items[: top_k * 2], vector=vector, graph=None,
            started_at=started_at, fusion_latency_ms=0.0,
        )

    vector, graph = await asyncio.gather(
        rag_vector_search(
            query=query, top_k=top_k, ticker_filter=resolved_entity,
            days_back=days_back, context=ctx,
        ),
        rag_graph_traverse(entity=resolved_entity, max_hops=max_hops, context=ctx),
    )
    vector_items = _vector_items(vector)
    graph_items = _graph_items(graph)
    fusion_started_at = time.perf_counter()
    if fusion == "rrf":
        fused = _rrf(vector_items, graph_items)
    elif fusion == "weighted":
        fused = _weighted(vector_items, graph_items)
    else:
        fused = vector_items + graph_items
    fused.sort(key=lambda item: (-item["score"], _key(item)))
    fusion_latency_ms = elapsed_latency_ms(fusion_started_at)
    return _hybrid_response(
        query=query, entity=resolved_entity, fusion=fusion, context=ctx,
        results=fused[: top_k * 2], vector=vector, graph=graph,
        started_at=started_at, fusion_latency_ms=fusion_latency_ms,
    )


def _vector_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "text": item.get("chunk_text") or "", "origin": "vector",
            "score": item.get("score", 0.0), "rank": index + 1,
            "url": item.get("url") or "", "title": item.get("title") or "",
            "document_id": item.get("document_id"), "chunk_id": item.get("chunk_id"),
            "evidence_id": item.get("evidence_id"), "citation": item.get("citation"),
        }
        for index, item in enumerate(payload.get("results", []))
    ]


def _graph_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "text": f"{item['name']} ({item['type']}) - {item['relation']} [{item['hops']} hop(s)]",
            "origin": "graph", "score": round(1.0 / (item["hops"] + 1), 4),
            "rank": index + 1, "url": item.get("source_url") or "", "title": "",
            "document_id": item.get("document_id"), "chunk_id": item.get("chunk_id"),
            "evidence_id": item.get("evidence_id"),
            "observation_id": item.get("observation_id"), "citation": item.get("citation"),
        }
        for index, item in enumerate(payload.get("nodes", []))
    ]


def _hybrid_response(
    *, query: str, entity: str | None, fusion: str, context: ForecastContext | None,
    results: list[dict[str, Any]], vector: dict[str, Any],
    graph: dict[str, Any] | None, started_at: float | None = None,
    fusion_latency_ms: float = 0.0,
) -> dict[str, Any]:
    vector_available = bool(vector.get("degradation", {}).get("available", True))
    graph_available = bool(graph and graph.get("degradation", {}).get("available", True))
    if vector_available and graph_available:
        mode = "hybrid"
    elif vector_available:
        mode = "vector_only"
    elif graph_available:
        mode = "graph_only"
    else:
        mode = "unavailable"
    reasons = []
    if not vector_available:
        reasons.append(vector.get("degradation", {}).get("reason", "vector_unavailable"))
    if not graph_available:
        reasons.append(
            graph.get("degradation", {}).get("reason", "graph_unavailable")
            if graph else "graph_not_requested"
        )
    citations = []
    seen = set()
    for result in results:
        citation = result.get("citation")
        if not citation:
            continue
        identity = (
            citation.get("origin"), citation.get("evidence_id"),
            citation.get("chunk_id"), citation.get("observation_id"),
        )
        if identity not in seen:
            citations.append(citation)
            seen.add(identity)
    # Fusion may collapse a shared Vector/Graph chunk into one ranked item;
    # retain both provenance paths even when the result identity is shared.
    branch_items = list(vector.get("results", []))
    if graph:
        branch_items.extend(graph.get("nodes", []))
    for item in branch_items:
        citation = item.get("citation")
        if not citation:
            continue
        identity = (
            citation.get("origin"), citation.get("evidence_id"),
            citation.get("chunk_id"), citation.get("observation_id"),
        )
        if identity not in seen:
            citations.append(citation)
            seen.add(identity)
    vector_latency = vector.get("telemetry", {}).get("latency_ms", 0.0)
    graph_latency = graph.get("telemetry", {}).get("latency_ms", 0.0) if graph else 0.0
    hybrid_latency = elapsed_latency_ms(started_at) if started_at is not None else 0.0
    evidence_count = len({
        citation.get("evidence_id") for citation in citations
        if citation.get("evidence_id")
    })
    telemetry = {
        **forecast_trace_metadata(
            context.to_dict() if context else {},
            stage=_TRACE_IDENTITY["stage"],
            stage_revision=_TRACE_IDENTITY["stage_revision"],
            attempt=_TRACE_IDENTITY["attempt"],
            agent=_TRACE_IDENTITY["agent"],
            latency_ms=hybrid_latency,
            evidence_count=evidence_count,
            degraded_flags=reasons,
        ),
        "operation": "hybrid",
        "vector_latency_ms": round(float(vector_latency), 3),
        "graph_latency_ms": round(float(graph_latency), 3),
        "rrf_latency_ms": round(float(fusion_latency_ms), 3) if fusion == "rrf" else 0.0,
        "fusion_latency_ms": round(float(fusion_latency_ms), 3),
        "hybrid_latency_ms": hybrid_latency,
        "result_count": len(results),
        "citation_count": len(citations),
        "degradation_mode": mode,
        "degraded": mode != "hybrid",
        "resume": "not_applicable",
    }
    return {
        "query": query, "entity": entity, "fusion": fusion,
        "context": context.to_dict() if context else None,
        "results": results, "citations": citations,
        "degradation": {
            "degraded": mode != "hybrid", "mode": mode,
            "vector_available": vector_available, "graph_available": graph_available,
            "reasons": reasons,
        },
        "telemetry": telemetry,
    }


def _embed(text: str) -> list[float]:
    if _embedder is not None:
        return _embedder.embed_query(text)
    digest = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return [(digest >> (index * 8) & 0xFF) / 255.0 for index in range(384)]


def _key(item: dict[str, Any]) -> str:
    stable = item.get("chunk_id") or item.get("observation_id") or item.get("evidence_id")
    if stable:
        return str(stable)
    return hashlib.md5(item["text"][:100].encode(), usedforsecurity=False).hexdigest()


_TICKER_STOPWORDS = {
    "AI", "US", "USA", "CEO", "CFO", "IPO", "ETF", "GDP", "Q1", "Q2", "Q3", "Q4",
    "THE", "AND", "FOR", "WITH", "FROM", "INTO", "OVER",
}


def _extract_ticker_from_query(query: str) -> Optional[str]:
    import re
    return next(
        (candidate for candidate in re.findall(r"\b[A-Z]{1,5}\b", query)
         if candidate not in _TICKER_STOPWORDS),
        None,
    )


def _rrf(a: list[dict], b: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    origins: dict[str, set[str]] = {}
    for values in (a, b):
        for rank, item in enumerate(values, 1):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items.setdefault(key, item)
            origins.setdefault(key, set()).add(item.get("origin", "unknown"))
    return [
        {**items[key], "score": round(score, 6), "origins": sorted(origins[key])}
        for key, score in scores.items()
    ]


def _weighted(vector: list[dict], graph: list[dict], weight: float = 0.7) -> list[dict]:
    return (
        [{**item, "score": round(item["score"] * weight, 4)} for item in vector]
        + [{**item, "score": round(item["score"] * (1 - weight), 4)} for item in graph]
    )


__all__ = ["rag_graph_traverse", "rag_hybrid_query", "rag_vector_search"]
