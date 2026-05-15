import os
import hashlib
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional imports — graceful degradation ────────────────────────
try:
    from supabase import create_client
    _sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
except Exception:
    _sb = None
    logger.warning("Supabase not available")

try:
    from neo4j import AsyncGraphDatabase
    _neo4j = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
except Exception:
    _neo4j = None
    logger.warning("Neo4j not available")

try:
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
except Exception:
    _embedder = None
    logger.warning("SentenceTransformer not available")


# ─────────────────────────────────────────────────────────────────
# Tool 1: Vector Search (Supabase pgvector)
# ─────────────────────────────────────────────────────────────────
async def rag_vector_search(
    query: str,
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
    doc_type: str = "all",              # "10-K" | "10-Q" | "research_report" | "all"
    threshold: float = 0.7,
) -> dict:
    """
    Semantic similarity search over Supabase pgvector knowledge base.
    Returns top-k relevant document chunks with similarity scores.
    """
    if not _sb:
        return {"query": query, "results": [], "warning": "Supabase not configured"}

    embedding = _embed(query)

    params = {
        "query_embedding": embedding,
        "match_threshold": threshold,
        "match_count":     top_k,
    }
    if ticker_filter:
        params["ticker_filter"] = ticker_filter.upper()
    if doc_type != "all":
        params["doc_type_filter"] = doc_type

    resp = _sb.rpc("match_documents", params).execute()
    rows = resp.data or []

    return {
        "query":   query,
        "results": [
            {
                "id":         r.get("id"),
                "ticker":     r.get("ticker"),
                "doc_type":   r.get("doc_type"),
                "chunk_text": r.get("content"),
                "score":      round(r.get("similarity", 0.0), 4),
                "metadata":   r.get("metadata", {}),
            }
            for r in rows
        ],
    }


# ─────────────────────────────────────────────────────────────────
# Tool 2: Graph Traversal (Neo4j)
# ─────────────────────────────────────────────────────────────────
async def rag_graph_traverse(
    entity: str,
    relation_types: list = None,       # ["COMPETES_WITH","SUPPLIES_TO","AFFECTED_BY","ALL"]
    max_hops: int = 2,
    limit: int = 20,
) -> dict:
    """
    Traverse Neo4j knowledge graph from a starting entity.
    Discovers: competitors, suppliers, geopolitical impacts, leadership.
    """
    if relation_types is None:
        relation_types = ["ALL"]

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
        e.name                                       AS name,
        labels(e)[0]                                 AS type,
        type(last(relationships(path)))              AS relation,
        length(path)                                 AS hops,
        [n IN nodes(path) | n.name]                  AS path_nodes
    ORDER BY hops ASC
    LIMIT $limit
    """

    nodes = []
    paths = []
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

    return {"entity": entity, "nodes": nodes, "paths": paths}


# ─────────────────────────────────────────────────────────────────
# Tool 3: Hybrid Query (Vector + Graph fused with RRF)
# ─────────────────────────────────────────────────────────────────
async def rag_hybrid_query(
    query: str,
    entity: str,
    top_k: int = 5,
    max_hops: int = 2,
    fusion: str = "rrf",               # "rrf" | "weighted" | "union"
) -> dict:
    """
    Combines vector similarity search + graph traversal.
    Fuses results using Reciprocal Rank Fusion (default).
    Best for complex queries requiring both semantic + relational context.
    """
    vec_res, graph_res = await asyncio.gather(
        rag_vector_search(query=query, top_k=top_k, ticker_filter=entity),
        rag_graph_traverse(entity=entity, max_hops=max_hops),
    )

    vec_items = [
        {"text": r["chunk_text"], "origin": "vector", "score": r["score"], "rank": i + 1}
        for i, r in enumerate(vec_res.get("results", []))
    ]
    graph_items = [
        {
            "text":   f"{n['name']} ({n['type']}) — {n['relation']} [{n['hops']} hop(s)]",
            "origin": "graph",
            "score":  round(1.0 / (n["hops"] + 1), 4),
            "rank":   i + 1,
        }
        for i, n in enumerate(graph_res.get("nodes", []))
    ]

    if fusion == "rrf":
        fused = _rrf(vec_items, graph_items)
    elif fusion == "weighted":
        fused = _weighted(vec_items, graph_items)
    else:
        fused = vec_items + graph_items

    fused.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query":    query,
        "entity":   entity,
        "fusion":   fusion,
        "results":  fused[: top_k * 2],
    }


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def _embed(text: str) -> list[float]:
    if _embedder:
        return _embedder.encode(text).tolist()
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return [(h >> (i * 8) & 0xFF) / 255.0 for i in range(384)]


def _key(item: dict) -> str:
    return hashlib.md5(item["text"][:100].encode()).hexdigest()


def _rrf(a: list, b: list, k: int = 60) -> list:
    scores: dict[str, float] = {}
    items:  dict[str, dict]  = {}
    for lst in (a, b):
        for rank, item in enumerate(lst, 1):
            key = _key(item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            items[key]  = item
    return [{**items[k], "score": round(v, 6)} for k, v in scores.items()]


def _weighted(vec: list, graph: list, w: float = 0.7) -> list:
    for i in vec:
        i["score"] = round(i["score"] * w, 4)
    for i in graph:
        i["score"] = round(i["score"] * (1 - w), 4)
    return vec + graph
