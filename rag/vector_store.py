"""
rag/vector_store.py — AlphaVectorStore
Supabase-backed vector store with HNSW indexing, Hybrid Search (RRF),
metadata filtering, pagination, and score thresholding.

Prerequisite SQL (run once in Supabase SQL editor):
─────────────────────────────────────────────────────────────────────────────
-- 1. Enable pgvector
create extension if not exists vector;

-- 2. Documents table
create table if not exists alpha_documents (
    id            bigserial primary key,
    content_hash  text        not null,
    url_hash      text        not null,
    ticker        text        not null,
    source_type   text        not null,
    published_at  timestamptz not null,
    ingested_at   timestamptz not null,
    chunk_index   int         not null default 0,
    url           text,
    title         text,
    text          text        not null,
    embedding     vector(384),
    fts           tsvector generated always as (to_tsvector('english', text)) stored,
    -- Unique per CHUNK, not per document: a single article produces many
    -- chunks that all share the same url_hash, so the conflict key must
    -- include chunk_index or upserts fail with
    -- "ON CONFLICT DO UPDATE command cannot affect row a second time".
    constraint alpha_documents_url_chunk_key unique (url_hash, chunk_index)
);

-- 3. HNSW index for fast ANN search
create index if not exists alpha_docs_embedding_hnsw
    on alpha_documents
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- 4. GIN index for full-text search
create index if not exists alpha_docs_fts_gin
    on alpha_documents using gin (fts);

-- 5. RRF hybrid search function
create or replace function alpha_hybrid_search(
    query_embedding vector(384),
    query_text      text,
    filter_ticker   text    default null,
    days_back       int     default null,
    top_k           int     default 50,
    rrf_k           int     default 60,
    page_offset     int     default 0
)
returns table (
    id           bigint,
    text         text,
    ticker       text,
    source_type  text,
    published_at timestamptz,
    url          text,
    title        text,
    chunk_index  int,
    rrf_score    float
)
language sql stable as $$
with
vector_ranked as (
    select
        id,
        row_number() over (order by embedding <=> query_embedding) as rank
    from alpha_documents
    where
        (filter_ticker is null or ticker = filter_ticker)
        and (days_back   is null or published_at > now() - (days_back || ' days')::interval)
    order by embedding <=> query_embedding
    limit top_k * 2
),
fts_ranked as (
    select
        id,
        row_number() over (order by ts_rank_cd(fts, plainto_tsquery('english', query_text)) desc) as rank
    from alpha_documents
    where
        fts @@ plainto_tsquery('english', query_text)
        and (filter_ticker is null or ticker = filter_ticker)
        and (days_back   is null or published_at > now() - (days_back || ' days')::interval)
    limit top_k * 2
),
rrf as (
    select
        coalesce(v.id, f.id) as id,
        coalesce(1.0 / (rrf_k + v.rank), 0) +
        coalesce(1.0 / (rrf_k + f.rank), 0) as rrf_score
    from vector_ranked v
    full outer join fts_ranked f on v.id = f.id
    order by rrf_score desc
    limit top_k
)
select
    d.id,
    d.text,
    d.ticker,
    d.source_type,
    d.published_at,
    d.url,
    d.title,
    d.chunk_index,
    r.rrf_score
from rrf r
join alpha_documents d on d.id = r.id
order by r.rrf_score desc
limit top_k
offset page_offset;
$$;
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from supabase import Client, create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AlphaVectorStore
# ---------------------------------------------------------------------------

class AlphaVectorStore:
    """
    Wraps Supabase for vector storage and retrieval.

    Features:
    - Upserts via url_hash (idempotent ingest)
    - HNSW-accelerated cosine similarity (configured in SQL above)
    - Hybrid RRF search via Postgres RPC
    - Metadata filtering: ticker, days_back
    - Pagination: limit + offset
    - Score threshold to suppress low-quality results
    """

    TABLE = "alpha_documents"
    RPC   = "alpha_hybrid_search"

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ) -> None:
        url = supabase_url or os.environ["SUPABASE_URL"]
        key = supabase_key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.client: Client = create_client(url, key)
        logger.info("AlphaVectorStore connected to Supabase.")

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, records: list[dict[str, Any]]) -> int:
        """
        Upsert a batch of embedded records.

        Each record must have:
            embedding, text, and all metadata keys from AlphaProcessor.

        Returns the number of rows upserted.
        """
        if not records:
            return 0

        rows = [self._to_row(r) for r in records]

        # Supabase upsert on url_hash (unique constraint handles idempotency)
        response = (
            self.client
            .table(self.TABLE)
            .upsert(rows, on_conflict="url_hash,chunk_index")
            .execute()
        )
        count = len(response.data or [])
        logger.info("AlphaVectorStore upserted %d rows.", count)
        return count

    # ------------------------------------------------------------------
    # Hybrid Search (RRF)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
        top_k: int = 50,
        rrf_k: int = 60,
        score_threshold: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Call the Postgres RRF function and apply client-side score filtering.

        Args:
            query_embedding:  L2-normalised query vector.
            query_text:       Raw query string for FTS.
            ticker:           Optional filter (e.g. "AAPL").
            days_back:        Only consider docs newer than N days.
            top_k:            Candidates considered inside the RPC.
            rrf_k:            RRF smoothing constant (default 60).
            score_threshold:  Minimum rrf_score to include in results.
            limit:            Number of rows to return (pagination).
            offset:           Row offset (pagination).

        Returns:
            List of result dicts with keys: id, text, ticker, source_type,
            published_at, url, title, chunk_index, rrf_score.
        """
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "query_text":      query_text,
            "top_k":           top_k,
            "rrf_k":           rrf_k,
            "page_offset":     offset,
        }
        if ticker:
            params["filter_ticker"] = ticker
        if days_back is not None:
            params["days_back"] = days_back

        response = self.client.rpc(self.RPC, params).execute()
        rows: list[dict] = response.data or []

        # Client-side score threshold + pagination limit
        filtered = [r for r in rows if r.get("rrf_score", 0) >= score_threshold]
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_row(record: dict[str, Any]) -> dict[str, Any]:
        """Map an embedded record dict to the DB column schema."""
        meta = record.get("metadata", {})
        return {
            "content_hash": meta.get("content_hash"),
            "url_hash":     meta.get("url_hash"),
            "ticker":       meta.get("ticker"),
            "source_type":  meta.get("source_type"),
            "published_at": meta.get("published_at_utc"),
            "ingested_at":  meta.get("ingested_at"),
            "chunk_index":  meta.get("chunk_index", 0),
            "url":          meta.get("url"),
            "title":        meta.get("title"),
            "text":         record.get("text", ""),
            "embedding":    record.get("embedding"),
        }