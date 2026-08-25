"""
rag/vector_store.py — AlphaVectorStore

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
    
        if not records:
            return 0

        rows = [self._to_row(r) for r in records]

       
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
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[str] = None,
        period_end: Optional[str] = None,
        metric_scope: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        
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
        
        if fiscal_year is not None:
            params["filter_fiscal_year"] = fiscal_year
        if fiscal_quarter:
            params["filter_fiscal_quarter"] = fiscal_quarter
        if period_end:
            params["filter_period_end"] = period_end
        if metric_scope:
            params["filter_metric_scope"] = metric_scope

        response = self.client.rpc(self.RPC, params).execute()
        rows: list[dict] = response.data or []

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
            "fiscal_year":  meta.get("fiscal_year"),
            "fiscal_quarter": meta.get("fiscal_quarter"),
            "period_start": meta.get("period_start"),
            "period_end": meta.get("period_end"),
            "duration_months": meta.get("duration_months"),
            "filing_type": meta.get("filing_type"),
            "accession_number": meta.get("accession_number"),
            "metric_scope": meta.get("metric_scope"),
            "gaap_status": meta.get("gaap_status"),
            "text":         record.get("text", ""),
            "embedding":    record.get("embedding"),
        }
