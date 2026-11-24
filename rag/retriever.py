"""Point-in-time retrieval with pre-ranking temporal eligibility checks."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

from rag.embedding_manager import AlphaEmbedder, get_embedder
from rag.vector_store import AlphaVectorStore

logger = logging.getLogger(__name__)
DECAY_HALF_LIFE_HOURS = 72
STAGE1_TOP_K = 50
STAGE2_TOP_K = 10
STAGE3_TOP_K = 5
TOKEN_BUDGET = 2_000
CHARS_PER_TOKEN = 4


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("timestamp is required")
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


class AlphaRetriever:
    def __init__(
        self,
        vector_store: AlphaVectorStore,
        embedder: Optional[AlphaEmbedder] = None,
        decay_half_life_hours: float = DECAY_HALF_LIFE_HOURS,
        stage1_k: int = STAGE1_TOP_K,
        stage2_k: int = STAGE2_TOP_K,
        stage3_k: int = STAGE3_TOP_K,
        token_budget: int = TOKEN_BUDGET,
        apply_freshness_rerank: bool = True,
        apply_diversity_filter: bool = True,
        apply_token_budget: bool = True,
    ) -> None:
        self.store = vector_store
        self.embedder = embedder or get_embedder()
        if not math.isfinite(decay_half_life_hours) or decay_half_life_hours <= 0:
            raise ValueError("decay_half_life_hours must be finite and positive")
        self.decay_half_life_hours = decay_half_life_hours
        self.stage1_k, self.stage2_k, self.stage3_k = stage1_k, stage2_k, stage3_k
        self.token_budget = token_budget
        self.apply_freshness_rerank = apply_freshness_rerank
        self.apply_diversity_filter = apply_diversity_filter
        self.apply_token_budget = apply_token_budget

    def retrieve(self, query: str, ticker: Optional[str] = None, days_back: Optional[int] = None, score_threshold: float = 0.01, *, as_of: object | None = None, dataset_version: str | None = None, historical: bool | None = None) -> str:
        return self._format_context(self._run_pipeline(query, ticker, days_back, score_threshold, as_of=as_of, dataset_version=dataset_version, historical=historical))

    def retrieve_raw(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
        score_threshold: float = 0.01,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[str] = None,
        period_end: Optional[str] = None,
        metric_scope: Optional[str] = None,
        *,
        as_of: object | None = None,
        dataset_version: str | None = None,
        historical: bool | None = None,
    ) -> list[dict[str, Any]]:
        return self._run_pipeline(
            query, ticker, days_back, score_threshold,
            fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter, period_end=period_end,
            metric_scope=metric_scope, as_of=as_of, dataset_version=dataset_version,
            historical=historical,
        )

    def _run_pipeline(
        self,
        query: str,
        ticker: Optional[str],
        days_back: Optional[int],
        score_threshold: float,
        *,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[str] = None,
        period_end: Optional[str] = None,
        metric_scope: Optional[str] = None,
        as_of: object | None = None,
        dataset_version: str | None = None,
        historical: bool | None = None,
    ) -> list[dict[str, Any]]:
        historical = historical if historical is not None else as_of is not None
        temporal = as_of is not None
        if temporal and (not ticker or not dataset_version):
            raise ValueError("temporal retrieval requires ticker, as_of, and dataset_version")
        query_vec = self.embedder.embed_query(query)
        search_kwargs: dict[str, Any] = {
            "query_embedding": query_vec, "query_text": query, "ticker": ticker,
            "days_back": days_back, "top_k": self.stage1_k,
            "score_threshold": score_threshold, "limit": self.stage1_k,
            "fiscal_year": fiscal_year, "fiscal_quarter": fiscal_quarter,
            "period_end": period_end, "metric_scope": metric_scope,
        }
        if temporal:
            search_kwargs.update(as_of=as_of, dataset_version=dataset_version, historical=historical)
        candidates = self.store.hybrid_search(**search_kwargs)
        if temporal:
            candidates = self._filter_temporal_candidates(candidates, ticker=ticker or "", as_of=as_of, dataset_version=dataset_version or "")
        reference = _utc(as_of) if temporal else None
        reranked = self._rerank_by_freshness(candidates, reference_time=reference)[: self.stage2_k] if self.apply_freshness_rerank else candidates
        diverse = self._diversity_filter(reranked) if self.apply_diversity_filter else reranked
        return self._apply_token_budget(diverse) if self.apply_token_budget else diverse

    @staticmethod
    def _filter_temporal_candidates(chunks: list[dict[str, Any]], *, ticker: str, as_of: object, dataset_version: str) -> list[dict[str, Any]]:
        boundary = _utc(as_of)
        eligible: list[dict[str, Any]] = []
        for chunk in chunks:
            if str(chunk.get("ticker", "")).upper() != ticker.upper() or chunk.get("dataset_version") != dataset_version:
                continue
            available = chunk.get("filed_at") or chunk.get("published_at") or chunk.get("effective_from")
            try:
                effective_from = _utc(chunk["effective_from"]) if chunk.get("effective_from") else None
                effective_to = _utc(chunk["effective_to"]) if chunk.get("effective_to") else None
                if _utc(available) > boundary or (effective_from and effective_from > boundary) or (effective_to and effective_to <= boundary):
                    continue
            except (TypeError, ValueError):
                continue
            eligible.append(chunk)
        return eligible

    def _rerank_by_freshness(self, chunks: list[dict], reference_time: datetime | None = None) -> list[dict]:
        reference = reference_time or datetime.now(timezone.utc)
        scored: list[tuple[float, dict]] = []
        for chunk in chunks:
            hours_old = self._hours_since(chunk.get("published_at", ""), reference)
            score = chunk.get("rrf_score", chunk.get("similarity", 0.0)) * math.exp(-math.log(2.0) * hours_old / self.decay_half_life_hours)
            scored.append((score, {**chunk, "freshness_score": score, "hours_old": hours_old}))
        scored.sort(key=lambda item: (item[0], item[1].get("chunk_id", "")), reverse=True)
        return [chunk for _, chunk in scored]

    @staticmethod
    def _hours_since(pub_at: Any, now: datetime) -> float:
        try:
            return max((_utc(now) - _utc(pub_at)).total_seconds() / 3600, 0.0)
        except (TypeError, ValueError):
            return 720.0

    def _diversity_filter(self, chunks: list[dict]) -> list[dict]:
        url_count: dict[str, int] = {}
        source_count: dict[str, int] = {}
        selected: list[dict] = []
        for chunk in chunks:
            url, source = chunk.get("url", ""), chunk.get("source_type", "")
            if url_count.get(url, 0) >= 2 or source_count.get(source, 0) >= 3:
                continue
            url_count[url] = url_count.get(url, 0) + 1
            source_count[source] = source_count.get(source, 0) + 1
            selected.append(chunk)
            if len(selected) >= self.stage3_k:
                break
        return selected

    def _apply_token_budget(self, chunks: list[dict]) -> list[dict]:
        char_budget, used, kept = self.token_budget * CHARS_PER_TOKEN, 0, []
        for chunk in chunks:
            size = len(chunk.get("text", ""))
            if used + size > char_budget and kept:
                break
            used += size
            kept.append(chunk)
        return kept

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        if not chunks:
            return "No relevant context found."
        separator = "-" * 60
        lines: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            score = chunk.get("freshness_score", chunk.get("rrf_score", chunk.get("similarity", 0)))
            lines.extend([
                separator,
                f"[{index}] SOURCE: {chunk.get('source_type', 'N/A')} | TICKER: {chunk.get('ticker', 'N/A')} | DATE: {chunk.get('published_at', 'N/A')} | SCORE: {score:.4f}",
                f"DOCUMENT_ID: {chunk.get('document_id', 'N/A')} | CHUNK_ID: {chunk.get('chunk_id', 'N/A')}",
                f"URL: {chunk.get('url', 'N/A')}", "CONTENT:", chunk.get("text", "").strip(),
            ])
        lines.append(separator)
        return "\n".join(lines)


__all__ = ["AlphaRetriever"]
