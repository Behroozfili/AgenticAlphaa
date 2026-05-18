"""
rag/retriever.py — AlphaRetriever
Two-stage retrieval: Hybrid Search → Freshness Reranking → Source Diversity Filter.
Exponential decay freshness scoring, context token budgeting, and citation formatting.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

from rag.embedding_manager import AlphaEmbedder, get_embedder
from rag.vector_store import AlphaVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECAY_HALF_LIFE_HOURS = 72      # exp(-hours_old / 72) → 50% weight at 3 days
STAGE1_TOP_K          = 50      # candidates from hybrid search
STAGE2_TOP_K          = 10      # after freshness reranking
STAGE3_TOP_K          = 5       # after diversity filtering
TOKEN_BUDGET          = 2_000   # approximate token limit for final context
CHARS_PER_TOKEN       = 4       # rough chars-to-tokens estimate

# ---------------------------------------------------------------------------
# AlphaRetriever
# ---------------------------------------------------------------------------

class AlphaRetriever:
    """
    Orchestrates multi-stage retrieval for financial RAG queries.

    Pipeline:
        1. Hybrid Search  (vector + FTS via RRF) → Top 50
        2. Freshness Reranking (exponential decay) → Top 10
        3. Source Diversity Filter                 → Top 5
        4. Context Budgeting (token cap)
        5. Citation-formatted string output
    """

    def __init__(
        self,
        vector_store: AlphaVectorStore,
        embedder: Optional[AlphaEmbedder] = None,
        decay_half_life_hours: float = DECAY_HALF_LIFE_HOURS,
        stage1_k: int = STAGE1_TOP_K,
        stage2_k: int = STAGE2_TOP_K,
        stage3_k: int = STAGE3_TOP_K,
        token_budget: int = TOKEN_BUDGET,
    ) -> None:
        self.store                 = vector_store
        self.embedder              = embedder or get_embedder()
        self.decay_half_life_hours = decay_half_life_hours
        self.stage1_k              = stage1_k
        self.stage2_k              = stage2_k
        self.stage3_k              = stage3_k
        self.token_budget          = token_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
        score_threshold: float = 0.01,
    ) -> str:
        """
        Full retrieval pipeline.

        Returns a citation-formatted context string ready to be injected
        into an LLM prompt.
        """
        # Stage 1 — Hybrid Search
        query_vec = self.embedder.embed_query(query)
        candidates = self.store.hybrid_search(
            query_embedding=query_vec,
            query_text=query,
            ticker=ticker,
            days_back=days_back,
            top_k=self.stage1_k,
            score_threshold=score_threshold,
            limit=self.stage1_k,
        )
        logger.info("Stage 1 candidates: %d", len(candidates))

        # Stage 2 — Freshness Reranking
        reranked = self._rerank_by_freshness(candidates)
        top_reranked = reranked[: self.stage2_k]
        logger.info("Stage 2 after reranking: %d", len(top_reranked))

        # Stage 3 — Source Diversity
        diverse = self._diversity_filter(top_reranked)
        logger.info("Stage 3 after diversity filter: %d", len(diverse))

        # Stage 4 — Context Budget
        budgeted = self._apply_token_budget(diverse)
        logger.info("Stage 4 after token budget: %d chunks", len(budgeted))

        # Stage 5 — Format
        return self._format_context(budgeted)

    def retrieve_raw(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
        score_threshold: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Same pipeline but returns the raw chunk dicts (useful for evaluation)."""
        query_vec  = self.embedder.embed_query(query)
        candidates = self.store.hybrid_search(
            query_embedding=query_vec,
            query_text=query,
            ticker=ticker,
            days_back=days_back,
            top_k=self.stage1_k,
            score_threshold=score_threshold,
            limit=self.stage1_k,
        )
        reranked   = self._rerank_by_freshness(candidates)[: self.stage2_k]
        diverse    = self._diversity_filter(reranked)
        return self._apply_token_budget(diverse)

    # ------------------------------------------------------------------
    # Stage 2 — Freshness Reranking
    # ------------------------------------------------------------------

    def _rerank_by_freshness(self, chunks: list[dict]) -> list[dict]:
        """
        Adjust each chunk's score with an exponential decay:

            freshness_score = rrf_score * exp(−hours_old / half_life)

        Older documents decay exponentially; recent ones are favoured.
        """
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, dict]] = []

        for chunk in chunks:
            rrf_score    = chunk.get("rrf_score", 0.0)
            pub_at_raw   = chunk.get("published_at", "")
            hours_old    = self._hours_since(pub_at_raw, now)
            decay        = math.exp(-hours_old / self.decay_half_life_hours)
            fresh_score  = rrf_score * decay
            chunk        = {**chunk, "freshness_score": fresh_score, "hours_old": hours_old}
            scored.append((fresh_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    @staticmethod
    def _hours_since(pub_at: Any, now: datetime) -> float:
        """Return hours between published_at and now. Returns 720 on parse failure."""
        try:
            if isinstance(pub_at, str):
                dt = datetime.fromisoformat(pub_at)
            elif isinstance(pub_at, datetime):
                dt = pub_at
            else:
                return 720.0

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = now - dt
            return max(delta.total_seconds() / 3600, 0.0)
        except Exception:
            return 720.0

    # ------------------------------------------------------------------
    # Stage 3 — Source Diversity
    # ------------------------------------------------------------------

    def _diversity_filter(self, chunks: list[dict]) -> list[dict]:
        """
        Ensure the top results aren't all from the same URL or source_type.

        Strategy:
        - At most 2 chunks per URL
        - At most 3 chunks per source_type
        - Stop when stage3_k chunks are collected
        """
        url_count:    dict[str, int] = {}
        source_count: dict[str, int] = {}
        selected:     list[dict]     = []

        for chunk in chunks:
            url         = chunk.get("url", "")
            source_type = chunk.get("source_type", "")

            if url_count.get(url, 0) >= 2:
                continue
            if source_count.get(source_type, 0) >= 3:
                continue

            url_count[url]         = url_count.get(url, 0) + 1
            source_count[source_type] = source_count.get(source_type, 0) + 1
            selected.append(chunk)

            if len(selected) >= self.stage3_k:
                break

        return selected

    # ------------------------------------------------------------------
    # Stage 4 — Token Budget
    # ------------------------------------------------------------------

    def _apply_token_budget(self, chunks: list[dict]) -> list[dict]:
        """
        Greedily include chunks until the character budget is exhausted.
        char_budget = token_budget * CHARS_PER_TOKEN
        """
        char_budget  = self.token_budget * CHARS_PER_TOKEN
        used_chars   = 0
        kept: list[dict] = []

        for chunk in chunks:
            text_len = len(chunk.get("text", ""))
            if used_chars + text_len > char_budget and kept:
                break
            used_chars += text_len
            kept.append(chunk)

        return kept

    # ------------------------------------------------------------------
    # Stage 5 — Citation Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        """
        Build a clearly sourced context string for LLM injection.

        Format per chunk:
        ─────────────────────────────────────────
        [1] SOURCE: news | TICKER: AAPL | DATE: 2024-03-15T14:32:00+00:00
        URL: https://...
        CONTENT:
        <chunk text>
        ─────────────────────────────────────────
        """
        if not chunks:
            return "No relevant context found."

        lines: list[str] = []
        separator = "─" * 60

        for i, chunk in enumerate(chunks, start=1):
            pub_at      = chunk.get("published_at", "N/A")
            source_type = chunk.get("source_type", "N/A")
            ticker      = chunk.get("ticker", "N/A")
            url         = chunk.get("url", "N/A")
            text        = chunk.get("text", "").strip()
            fresh_score = chunk.get("freshness_score", chunk.get("rrf_score", 0))

            lines.append(separator)
            lines.append(
                f"[{i}] SOURCE: {source_type} | TICKER: {ticker} | "
                f"DATE: {pub_at} | SCORE: {fresh_score:.4f}"
            )
            lines.append(f"URL: {url}")
            lines.append("CONTENT:")
            lines.append(text)

        lines.append(separator)
        return "\n".join(lines)
