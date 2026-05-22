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

logger = logging.getLogger(__name__)

# ── Optional imports — graceful degradation ────────────────────────────────

try:
    from supabase import create_client
    _sb = create_client(
        os.environ["SUPABASE_URL"],
        # FIX: use the same env-var convention as vector_store.py
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"],
    )
except Exception as exc:
    _sb = None
    logger.warning("Supabase not available: %s", exc)

try:
    from neo4j import AsyncGraphDatabase
    _neo4j = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
except Exception as exc:
    _neo4j = None
    logger.warning("Neo4j not available: %s", exc)

# FIX: use the AlphaEmbedder singleton instead of a separate SentenceTransformer
try:
    from rag.embedding_manager import get_embedder as _get_embedder
    _embedder = _get_embedder()
except Exception as exc:
    _embedder = None
    logger.warning("AlphaEmbedder not available: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Vector Search (Supabase pgvector via alpha_hybrid_search RPC)
# ─────────────────────────────────────────────────────────────────────────────

async def rag_vector_search(
    query: str,
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
    days_back: Optional[int] = None,
    threshold: float = 0.01,            # RRF scores are small; 0.01 is a safe floor
) -> dict:
    """
    Semantic + full-text hybrid search over the Supabase knowledge base.

    Uses the alpha_hybrid_search RPC (same as AlphaVectorStore.hybrid_search)
    so both the agent and the ingestion pipeline stay in sync with the same
    SQL function.

    Returns top-k relevant document chunks with RRF scores.
    """
    if not _sb:
        return {"query": query, "results": [], "warning": "Supabase not configured"}

    embedding = _embed(query)

    # FIX: use alpha_hybrid_search (matches vector_store.py), not match_documents
    params: dict = {
        "query_embedding": embedding,
        "query_text":      query,
        "top_k":           top_k * 2,   # fetch extra; client-side threshold filters down
        "rrf_k":           60,
        "page_offset":     0,
    }
    if ticker_filter:
        params["filter_ticker"] = ticker_filter.upper()
    if days_back is not None:
        params["days_back"] = days_back

    try:
        resp = _sb.rpc("alpha_hybrid_search", params).execute()
        rows: list[dict] = resp.data or []
    except Exception as exc:
        logger.error("Supabase RPC error: %s", exc)
        return {"query": query, "results": [], "error": str(exc)}

    # Apply score threshold + limit
    results = [
        {
            "id":         r.get("id"),
            "ticker":     r.get("ticker"),
            "source_type": r.get("source_type"),
            "chunk_text": r.get("text"),
            "score":      round(r.get("rrf_score", 0.0), 6),
            "published_at": r.get("published_at"),
            "url":        r.get("url"),
            "title":      r.get("title"),
        }
        for r in rows
        if r.get("rrf_score", 0.0) >= threshold
    ]
    return {"query": query, "results": results[:top_k]}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Graph Traversal (Neo4j)
# ─────────────────────────────────────────────────────────────────────────────

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

    if not _neo4j:
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

    try:
        # FIX: correct async session usage for neo4j >= 5.x
        async with _neo4j.session() as session:
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
        return {"entity": entity, "nodes": [], "paths": [], "error": str(exc)}

    return {"entity": entity, "nodes": nodes, "paths": paths}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Hybrid Query (Vector + Graph fused with RRF)
# ─────────────────────────────────────────────────────────────────────────────

async def rag_hybrid_query(
    query: str,
    entity: str,
    top_k: int = 5,
    max_hops: int = 2,
    fusion: str = "rrf",           # "rrf" | "weighted" | "union"
    days_back: Optional[int] = None,
) -> dict:
    """
    Combines vector similarity search + graph traversal.
    Fuses results using Reciprocal Rank Fusion (default).

    Best for complex queries like:
        "How does the war in Ukraine affect airline stocks?"
    where both semantic context AND entity relationships are needed.
    """
    vec_res, graph_res = await asyncio.gather(
        rag_vector_search(
            query=query,
            top_k=top_k,
            ticker_filter=entity,
            days_back=days_back,
        ),
        rag_graph_traverse(entity=entity, max_hops=max_hops),
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
        "entity":  entity,
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