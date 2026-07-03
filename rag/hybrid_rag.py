"""
rag/hybrid_rag.py — HybridRAG
Unified Hybrid RAG: Vector Search (Supabase pgvector) + Graph Traversal (Neo4j).
Uses AlphaEmbedder singleton — no duplicate model loading.

Tools exposed to the agent:
    rag_vector_search   — semantic similarity over stored chunks
    rag_graph_traverse  — relationship traversal from a named entity
    rag_hybrid_query    — fused vector + graph via RRF (recommended)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Optional

from langsmith import traceable
from core.observability import sentry_enabled

logger = logging.getLogger(__name__)

# ── Optional imports — graceful degradation ────────────────────────────────
# NOTE: _sb (raw Supabase client) and _embed() below are no longer used by
# rag_vector_search itself (it now goes through AlphaRetriever/AlphaVectorStore
# instead — see _retriever setup further down). Left in place in case other
# tools in this module rely on them directly.

try:
    from supabase import create_client
    _sb = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"],
    )
except Exception as exc:
    _sb = None
    logger.warning("Supabase not available: %s", exc)

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
    except Exception as exc:
        logger.warning("Neo4j not available: %s", exc)
        return None

try:
    from rag.embedding_manager import get_embedder as _get_embedder
    _embedder = _get_embedder()
except Exception as exc:
    _embedder = None
    logger.warning("AlphaEmbedder not available: %s", exc)

# AlphaRetriever gives ResearchAgent the full 4-stage pipeline (hybrid
# search → freshness reranking → source diversity filter → token budget)
# instead of a raw Supabase RPC call. All three narrowing stages are left
# at their defaults (enabled) here — ResearchAgent's chunks feed straight
# into an LLM prompt, so recency, source diversity, and a bounded context
# size all matter for it, unlike SentimentAgent's retrieve_social_data
# (rag/sentiment_server.py), which explicitly disables freshness reranking
# and diversity filtering since its output feeds FinBERT/VADER batch
# scoring rather than a prompt.
try:
    from rag.retriever import AlphaRetriever
    from rag.vector_store import AlphaVectorStore
    _vector_store = AlphaVectorStore(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"],
    )
    _retriever = AlphaRetriever(vector_store=_vector_store, embedder=_embedder)
except Exception as exc:
    _retriever = None
    logger.warning("AlphaRetriever not available: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Vector Search (Supabase pgvector via alpha_hybrid_search RPC)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(run_type="retriever")
async def rag_vector_search(
    query: str,
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
    days_back: Optional[int] = None,
    threshold: float = 0.01,
) -> dict:
    """
    Semantic + full-text hybrid search over the Supabase knowledge base,
    run through AlphaRetriever's full pipeline: hybrid search → freshness
    reranking → source diversity filter → token budget. (Previously this
    called the Supabase RPC directly and skipped those last three stages;
    SentimentAgent's separate retrieve_social_data path intentionally
    skips freshness/diversity since its consumer is FinBERT/VADER, not an
    LLM prompt — see rag/sentiment_server.py for that rationale.)

    top_k is honoured as a final slice on top of the retriever's own
    stage3_k (diversity-filter output count); if top_k < stage3_k you get
    fewer, if top_k > stage3_k the diversity filter's cap still applies
    since that's the whole point of using it here.
    """
    if not _retriever:
        return {"query": query, "results": [], "warning": "AlphaRetriever not configured"}

    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="rag.vector_search",
            message="Calling AlphaRetriever.retrieve_raw (full pipeline)",
            data={"query": query[:100], "ticker_filter": ticker_filter},
            level="info",
        )

    try:
        raw_chunks = await asyncio.to_thread(
            _retriever.retrieve_raw,
            query=query,
            ticker=ticker_filter.upper() if ticker_filter else None,
            days_back=days_back,
            score_threshold=threshold,
        )
    except Exception as exc:
        logger.error("AlphaRetriever error: %s", exc)
        if sentry_enabled():
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "rag.vector_search")
                sentry_sdk.capture_exception(exc)
        return {"query": query, "results": [], "error": str(exc)}

    results = [
        {
            "id":           r.get("id"),
            "ticker":       r.get("ticker"),
            "source_type":  r.get("source_type"),
            "chunk_text":   r.get("text"),
            # freshness_score reflects the post-reranking score (what
            # actually determined ordering/survival); fall back to the
            # raw rrf_score if freshness reranking somehow wasn't applied.
            "score":        round(r.get("freshness_score", r.get("rrf_score", 0.0)), 6),
            "published_at": r.get("published_at"),
            "url":          r.get("url"),
            "title":        r.get("title"),
            "chunk_index":  r.get("chunk_index"),
        }
        for r in raw_chunks
    ]
    return {"query": query, "results": results[:top_k]}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Graph Traversal (Neo4j)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(run_type="retriever")
async def rag_graph_traverse(
    entity: str,
    relation_types: Optional[list[str]] = None,
    max_hops: int = 2,
    limit: int = 20,
) -> dict:
    """
    Traverse the Neo4j knowledge graph from a starting entity.
    Discovers: competitors, suppliers, geopolitical impacts, leadership.

    Keeps max_hops ≤ 3 to avoid unreliable reasoning chains
    (see Known Limitations in the project README).
    """
    if relation_types is None:
        relation_types = ["ALL"]

    max_hops = min(max_hops, 3)   # guard against hallucination-prone deep traversal

    driver = _get_neo4j()
    if not driver:
        return {"entity": entity, "nodes": [], "warning": "Neo4j not configured"}

    if "ALL" in relation_types:
        rel_clause = f"[*1..{max_hops}]"
    else:
        types_str  = "|".join(relation_types)
        rel_clause = f"[:{types_str}*1..{max_hops}]"

    cypher = f"""
    MATCH path = (s {{name: $entity}})-{rel_clause}-(e)
    RETURN
        e.name                                      AS name,
        labels(e)[0]                                AS type,
        type(last(relationships(path)))             AS relation,
        length(path)                                AS hops,
        [n IN nodes(path) | n.name]                 AS path_nodes
    ORDER BY hops ASC
    LIMIT $limit
    """

    nodes: list[dict] = []
    paths: list[list] = []

    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="rag.graph_traverse",
            message="Running Neo4j cypher traversal",
            data={"entity": entity, "max_hops": max_hops},
            level="info",
        )

    try:
        async with driver.session() as session:
            result = await session.run(cypher, entity=entity.upper(), limit=limit)
            async for rec in result:
                nodes.append({
                    "name":     rec["name"],
                    "type":     rec["type"],
                    "relation": rec["relation"],
                    "hops":     rec["hops"],
                })
                paths.append(rec["path_nodes"])
    except Exception as exc:
        logger.error("Neo4j traversal error (entity=%s): %s", entity, exc)
        if sentry_enabled():
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "rag.graph_traverse")
                scope.set_tag("entity", entity)
                sentry_sdk.capture_exception(exc)
        return {"entity": entity, "nodes": [], "paths": [], "error": str(exc)}

    return {"entity": entity, "nodes": nodes, "paths": paths}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Hybrid Query (Vector + Graph fused with RRF)
# ─────────────────────────────────────────────────────────────────────────────

@traceable(run_type="retriever")
async def rag_hybrid_query(
    query: str,
    entity: Optional[str] = None,
    top_k: int = 5,
    max_hops: int = 2,
    fusion: str = "rrf",           # "rrf" | "weighted" | "union"
    days_back: Optional[int] = None,
) -> dict:
    """
    Combines vector similarity search + graph traversal.
    Fuses results using Reciprocal Rank Fusion (default).

    `entity` is optional. If omitted, a best-effort ticker is extracted
    from `query` (e.g. "AAPL" out of "Apple AAPL earnings"). If no entity
    can be determined at all, the function gracefully degrades to a
    vector-only search instead of raising — a missing entity should never
    crash the pipeline.

    Best for complex queries like:
        "How does the war in Ukraine affect airline stocks?"
    where both semantic context AND entity relationships are needed.
    """
    resolved_entity = entity or _extract_ticker_from_query(query)

    if not resolved_entity:
        logger.info("rag_hybrid_query: no entity provided/extracted — vector-only mode.")
        vec_res = await rag_vector_search(query=query, top_k=top_k, days_back=days_back)
        vec_items = [
            {
                "text":   r["chunk_text"] or "",
                "origin": "vector",
                "score":  r["score"],
                "rank":   i + 1,
                "url":    r.get("url", ""),
                "title":  r.get("title", ""),
            }
            for i, r in enumerate(vec_res.get("results", []))
        ]
        return {
            "query":   query,
            "entity":  None,
            "fusion":  "vector_only",
            "results": vec_items[: top_k * 2],
            "warning": "No entity provided or extractable from query; graph traversal skipped.",
        }

    vec_res, graph_res = await asyncio.gather(
        rag_vector_search(
            query=query,
            top_k=top_k,
            ticker_filter=resolved_entity,
            days_back=days_back,
        ),
        rag_graph_traverse(entity=resolved_entity, max_hops=max_hops),
    )

    vec_items = [
        {
            "text":   r["chunk_text"] or "",
            "origin": "vector",
            "score":  r["score"],
            "rank":   i + 1,
            "url":    r.get("url", ""),
            "title":  r.get("title", ""),
        }
        for i, r in enumerate(vec_res.get("results", []))
    ]

    graph_items = [
        {
            "text":   f"{n['name']} ({n['type']}) — {n['relation']} [{n['hops']} hop(s)]",
            "origin": "graph",
            "score":  round(1.0 / (n["hops"] + 1), 4),
            "rank":   i + 1,
            "url":    "",
            "title":  "",
        }
        for i, n in enumerate(graph_res.get("nodes", []))
    ]

    if fusion == "rrf":
        fused = _rrf(vec_items, graph_items)
    elif fusion == "weighted":
        fused = _weighted(vec_items, graph_items)
    else:                           # union
        fused = vec_items + graph_items

    fused.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query":   query,
        "entity":  resolved_entity,
        "fusion":  fusion,
        "results": fused[: top_k * 2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """Use the shared AlphaEmbedder singleton; fall back to a deterministic hash."""
    if _embedder is not None:
        return _embedder.embed_query(text)   # FIX: use public API, not _encode_batch
    # Fallback: deterministic pseudo-embedding (no model loaded)
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return [(h >> (i * 8) & 0xFF) / 255.0 for i in range(384)]


def _key(item: dict) -> str:
    return hashlib.md5(item["text"][:100].encode()).hexdigest()


# Heuristic ticker extractor: looks for a 1-5 letter ALL-CAPS token that
# isn't a common English stopword/acronym. Used only as a fallback when
# the caller omits `entity` — best-effort, not authoritative.
_TICKER_STOPWORDS = {
    "AI", "US", "USA", "CEO", "CFO", "IPO", "ETF", "GDP", "Q1", "Q2", "Q3", "Q4",
    "THE", "AND", "FOR", "WITH", "FROM", "INTO", "OVER",
}

def _extract_ticker_from_query(query: str) -> Optional[str]:
    import re
    candidates = re.findall(r"\b[A-Z]{1,5}\b", query)
    for c in candidates:
        if c not in _TICKER_STOPWORDS:
            return c
    return None


def _rrf(a: list[dict], b: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items:  dict[str, dict]  = {}
    for lst in (a, b):
        for rank, item in enumerate(lst, 1):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key]  = item
    return [{**items[k], "score": round(v, 6)} for k, v in scores.items()]


def _weighted(vec: list[dict], graph: list[dict], w: float = 0.7) -> list[dict]:
    return (
        [{**i, "score": round(i["score"] * w, 4)} for i in vec]
        + [{**i, "score": round(i["score"] * (1 - w), 4)} for i in graph]
    )