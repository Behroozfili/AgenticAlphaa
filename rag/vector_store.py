"""Legacy/V2 Vector store adapter with fail-closed temporal retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from supabase import Client, create_client

logger = logging.getLogger(__name__)


class TemporalBoundaryUnsupported(RuntimeError):
    """The selected backend cannot enforce point-in-time retrieval."""


@dataclass(frozen=True)
class VectorStoreConfig:
    schema: Literal["legacy", "v2"] = "legacy"
    legacy_table: str = "alpha_documents"
    legacy_rpc: str = "alpha_hybrid_search"
    v2_table: str = "alpha_documents_v2"
    v2_rpc: str = "alpha_hybrid_search_v2"

    @classmethod
    def from_environment(cls) -> "VectorStoreConfig":
        schema = os.getenv("RAG_VECTOR_SCHEMA", "legacy").strip().lower()
        if schema not in {"legacy", "v2"}:
            raise ValueError("RAG_VECTOR_SCHEMA must be 'legacy' or 'v2'")
        return cls(schema=schema)  # type: ignore[arg-type]


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp is required")
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


class AlphaVectorStore:
    V2_CONFLICT_TARGET = ",".join((
        "canonical_source_id", "content_hash", "document_version",
        "document_id", "evidence_id", "dataset_version", "chunk_index",
        "chunk_hash", "embedding_version",
    ))
    TABLE = "alpha_documents"
    RPC = "alpha_hybrid_search"

    def __init__(self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None, *, config: VectorStoreConfig | None = None, client: Client | None = None) -> None:
        self.config = config or VectorStoreConfig.from_environment()
        if client is not None:
            self.client = client
        else:
            url = supabase_url or os.environ["SUPABASE_URL"]
            key = supabase_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
            self.client = create_client(url, key)
        self.table_name = self.config.v2_table if self.config.schema == "v2" else self.config.legacy_table
        self.rpc_name = self.config.v2_rpc if self.config.schema == "v2" else self.config.legacy_rpc

    @property
    def supports_temporal_boundary(self) -> bool:
        return self.config.schema == "v2"

    def upsert(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        rows = [self._to_v2_row(record) if self.config.schema == "v2" else self._to_row(record) for record in records]
        if self.config.schema == "legacy":
            rows = self._prepare_legacy_rows(rows)
        conflict = self.V2_CONFLICT_TARGET if self.config.schema == "v2" else "url_hash,chunk_index"
        response = self.client.table(self.table_name).upsert(rows, on_conflict=conflict).execute()
        return len(response.data or [])

    @staticmethod
    def _prepare_legacy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scope the legacy conflict identity by ticker before batch upsert.

        The deployed legacy constraint is (url_hash, chunk_index). A single
        article can legitimately mention multiple tickers, so the unscoped
        hash makes Postgres reject the entire multi-ticker batch. Persisting a
        deterministic ticker-scoped hash preserves every company observation
        while remaining compatible with the existing database constraint.
        """
        prepared: dict[tuple[object, object], dict[str, Any]] = {}
        incomplete: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            url_hash = str(item.get("url_hash") or "").strip()
            ticker = str(item.get("ticker") or "").strip().upper()
            if not url_hash or not ticker:
                incomplete.append(item)
                continue
            scoped_identity = f"legacy-ticker-url-v1|{ticker}|{url_hash}"
            item["url_hash"] = hashlib.sha256(
                scoped_identity.encode("utf-8")
            ).hexdigest()
            key = (item["url_hash"], item.get("chunk_index", 0))
            prepared.setdefault(key, item)
        return [*prepared.values(), *incomplete]

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
        as_of: object | None = None,
        dataset_version: str | None = None,
        historical: bool = False,
    ) -> list[dict[str, Any]]:
        if historical and not self.supports_temporal_boundary:
            raise TemporalBoundaryUnsupported("Historical retrieval requires the V2 RPC; the legacy backend cannot enforce as_of/dataset boundaries")
        if self.config.schema == "v2":
            return self._hybrid_search_v2(
                query_embedding=query_embedding, ticker=ticker, days_back=days_back,
                score_threshold=score_threshold, limit=limit, as_of=as_of,
                dataset_version=dataset_version,
            )
        if as_of is not None:
            return self._live_legacy_search_as_of(
                query_embedding=query_embedding,
                ticker=ticker,
                days_back=days_back,
                score_threshold=score_threshold,
                limit=limit,
                as_of=as_of,
                dataset_version=dataset_version,
            )
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "query_text": query_text,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "page_offset": offset,
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
        rows = self.client.rpc(self.rpc_name, params).execute().data or []
        return [row for row in rows if row.get("rrf_score", 0) >= score_threshold][:limit]

    def _live_legacy_search_as_of(
        self,
        *,
        query_embedding: list[float],
        ticker: str | None,
        days_back: int | None,
        score_threshold: float,
        limit: int,
        as_of: object,
        dataset_version: str | None,
    ) -> list[dict[str, Any]]:
        """Filter legacy live rows by publication and ingestion before ranking.

        Historical backtests never enter this adapter; hybrid_search keeps
        those fail-closed on the V2 schema.
        """
        if not ticker or not dataset_version:
            raise ValueError("live temporal retrieval requires ticker and dataset_version")
        boundary = _utc(as_of)
        query = (
            self.client.table(self.table_name)
            .select(
                "id,url_hash,ticker,source_type,published_at,ingested_at,"
                "chunk_index,url,title,text,embedding"
            )
            .eq("ticker", ticker.upper())
            .lte("published_at", boundary.isoformat())
            .lte("ingested_at", boundary.isoformat())
            .not_.is_("embedding", "null")
        )
        if days_back is not None:
            if days_back <= 0:
                raise ValueError("days_back must be positive")
            from datetime import timedelta
            query = query.gte(
                "published_at", (boundary - timedelta(days=days_back)).isoformat()
            )
        candidate_limit = max(200, min(1000, max(1, limit) * 40))
        rows = (
            query.order("published_at", desc=True)
            .limit(candidate_limit)
            .execute()
            .data
            or []
        )
        ranked: list[dict[str, Any]] = []
        for row in rows:
            candidate = self._embedding_values(row.get("embedding"))
            similarity = self._cosine_similarity(query_embedding, candidate)
            if similarity < score_threshold:
                continue
            url_hash = str(row.get("url_hash") or "")
            chunk_index = int(row.get("chunk_index") or 0)
            published_at = row.get("published_at")
            ingested_at = row.get("ingested_at")
            ranked.append({
                **{key: value for key, value in row.items() if key != "embedding"},
                "document_id": url_hash or None,
                "chunk_id": self._stable_legacy_id("chunk", url_hash, chunk_index),
                "evidence_id": self._stable_legacy_id(
                    "evidence", url_hash, published_at, ingested_at
                ),
                "dataset_version": dataset_version,
                "retrieved_at": ingested_at,
                "effective_from": published_at,
                "effective_to": None,
                "rrf_score": similarity,
            })
        ranked.sort(
            key=lambda row: (-float(row["rrf_score"]), str(row["chunk_id"]))
        )
        return ranked[:limit]

    @staticmethod
    def _embedding_values(value: Any) -> list[float]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        if not isinstance(value, (list, tuple)):
            return []
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return -1.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return -1.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _stable_legacy_id(kind: str, *parts: object) -> str:
        import hashlib
        payload = "|".join(str(part or "") for part in (kind, *parts))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hybrid_search_v2(self, *, query_embedding: list[float], ticker: str | None, days_back: int | None, score_threshold: float, limit: int, as_of: object | None, dataset_version: str | None) -> list[dict[str, Any]]:
        if not ticker or as_of is None or not dataset_version:
            raise ValueError("V2 retrieval requires ticker, as_of, and dataset_version")
        as_of_dt = _utc(as_of)
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_ticker": ticker.upper(),
            "query_as_of": as_of_dt.isoformat(),
            "query_dataset_version": dataset_version,
            "match_count": limit,
            "similarity_threshold": score_threshold,
        }
        if days_back is not None:
            if days_back <= 0:
                raise ValueError("days_back must be positive")
            params["query_lookback"] = f"{days_back} days"
        rows = self.client.rpc(self.rpc_name, params).execute().data or []
        eligible: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("ticker", "")).upper() != ticker.upper() or row.get("dataset_version") != dataset_version:
                continue
            available_raw = row.get("filed_at") or row.get("published_at") or row.get("effective_from")
            try:
                effective_from = _utc(row["effective_from"]) if row.get("effective_from") else None
                effective_to = _utc(row["effective_to"]) if row.get("effective_to") else None
                if _utc(available_raw) > as_of_dt or (effective_from and effective_from > as_of_dt) or (effective_to and effective_to <= as_of_dt):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            metadata = dict(row.get("metadata") or {})
            eligible.append({
                **metadata,
                **row,
                "text": row.get("document_text", ""),
                "rrf_score": row.get("similarity", 0.0),
                "url": metadata.get("url", metadata.get("source_url", "")),
                "source_type": metadata.get("source_type", "unknown"),
            })
        return eligible[:limit]

    @staticmethod
    def _to_row(record: dict[str, Any]) -> dict[str, Any]:
        meta = record.get("metadata", {})
        return {
            "content_hash": meta.get("content_hash"), "url_hash": meta.get("url_hash"),
            "ticker": meta.get("ticker"), "source_type": meta.get("source_type"),
            "published_at": meta.get("published_at_utc") or meta.get("published_at"),
            "ingested_at": meta.get("ingested_at") or meta.get("retrieved_at"),
            "chunk_index": meta.get("chunk_index", 0), "url": meta.get("url"), "title": meta.get("title"),
            "fiscal_year": meta.get("fiscal_year"), "fiscal_quarter": meta.get("fiscal_quarter"),
            "period_start": meta.get("period_start"), "period_end": meta.get("period_end"),
            "duration_months": meta.get("duration_months"), "filing_type": meta.get("filing_type"),
            "accession_number": meta.get("accession_number"), "metric_scope": meta.get("metric_scope"),
            "gaap_status": meta.get("gaap_status"), "text": record.get("text", ""),
            "embedding": record.get("embedding"),
        }

    @staticmethod
    def _to_v2_row(record: dict[str, Any]) -> dict[str, Any]:
        meta = dict(record.get("metadata", {}))
        return {
            "chunk_id": meta.get("chunk_id"), "document_id": meta.get("document_id"),
            "evidence_id": meta.get("evidence_id"), "canonical_source_id": meta.get("canonical_source_id"),
            "source_url": meta.get("url"), "source_type": meta.get("source_type"),
            "source_version": meta.get("source_version"), "content_hash": meta.get("content_hash"),
            "chunk_hash": meta.get("chunk_hash"), "chunk_index": meta.get("chunk_index", 0),
            "ticker": meta.get("ticker"), "document_text": record.get("text", ""),
            "embedding": record.get("embedding"), "embedding_model": meta.get("embedding_model"),
            "embedding_version": meta.get("embedding_version"), "published_at": meta.get("published_at"),
            "filed_at": meta.get("filed_at"), "retrieved_at": meta.get("retrieved_at"),
            "effective_from": meta.get("effective_from"), "effective_to": meta.get("effective_to"),
            "dataset_version": meta.get("dataset_version"), "document_version": meta.get("document_version"),
            "metadata": meta, "is_quarantined": False, "quarantine_reason": None,
        }


__all__ = ["AlphaVectorStore", "TemporalBoundaryUnsupported", "VectorStoreConfig"]
