"""
tools/local_social_retriever.py — LocalSocialDataRetriever
============================================================
Retrieves pre-ingested social and news text chunks from the
existing AlphaRetriever / AlphaVectorStore RAG pipeline.

This tool does NOT call any external API.  All data originates
from documents already ingested into Supabase by ingestion.py
(yfinance news + Reddit RSS feeds via AlphaLoader).

Design contract with the RAG layer
───────────────────────────────────
  • AlphaRetriever.retrieve_raw()  →  list[dict]
      Keys guaranteed by vector_store.py hybrid_search():
        id, text, ticker, source_type, published_at,
        url, title, chunk_index, rrf_score

  • AlphaVectorStore  (Supabase / pgvector + FTS hybrid RRF)
  • AlphaEmbedder     (BAAI/bge-small-en-v1.5, singleton)

Public interface
─────────────────
  retriever = LocalSocialDataRetriever()
  result    = retriever.retrieve(query="NVIDIA earnings outlook", ticker="NVDA")
  # result["chunks"] → list of clean text strings ready for sentiment models
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from rag.embedding_manager import get_embedder
from rag.retriever import AlphaRetriever
from rag.vector_store import AlphaVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return Schema
# ---------------------------------------------------------------------------

@dataclass
class SocialRetrievalResult:
    """
    Structured output returned by LocalSocialDataRetriever.retrieve().

    Attributes:
        query       : The semantic query that was executed.
        ticker      : Ticker symbol used as a metadata filter (may be None).
        chunks      : Ordered list of clean text strings for downstream models.
                      Ordered by hybrid RRF score (best first), then freshness.
        sources     : Parallel list of source metadata dicts for each chunk.
                      Keys: ticker, source_type, published_at, url, title, rrf_score.
        total_found : Number of chunks returned after the full retrieval pipeline.
    """
    query:       str
    ticker:      Optional[str]
    chunks:      list[str]      = field(default_factory=list)
    sources:     list[dict]     = field(default_factory=list)
    total_found: int            = 0


# ---------------------------------------------------------------------------
# LocalSocialDataRetriever
# ---------------------------------------------------------------------------

class LocalSocialDataRetriever:
    """
    Sentiment Agent tool — wraps AlphaRetriever to fetch market buzz.

    Connects to the same Supabase instance used by ingestion.py and
    executes the full 4-stage retrieval pipeline defined in retriever.py:
        Stage 1 → Hybrid Search  (vector + FTS / RRF)
        Stage 2 → Freshness Rerank (exponential decay)
        Stage 3 → Source Diversity Filter
        Stage 4 → Token Budget Cap

    Parameters
    ----------
    supabase_url : str, optional
        Supabase project URL.  Defaults to env var SUPABASE_URL.
    supabase_key : str, optional
        Supabase service-role key.  Defaults to env var SUPABASE_SERVICE_ROLE_KEY.
    days_back : int, optional
        Only consider documents newer than this many days.  Default: 7.
    top_k : int, optional
        Maximum number of chunks to return.  Default: 20.
    score_threshold : float, optional
        Minimum RRF score to include a chunk.  Default: 0.01.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        days_back: int = 7,
        top_k: int = 20,
        score_threshold: float = 0.01,
    ) -> None:
        self._days_back       = days_back
        self._top_k           = top_k
        self._score_threshold = score_threshold

        # Initialise the Supabase-backed vector store
        url = supabase_url or os.environ["SUPABASE_URL"]
        key = supabase_key or (
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or os.environ["SUPABASE_SERVICE_KEY"]
        )
        vector_store = AlphaVectorStore(supabase_url=url, supabase_key=key)

        # Reuse the shared singleton embedder (avoids double-loading the model)
        embedder = get_embedder()

        # Build the retriever with tighter stage limits suited for sentiment use
        self._retriever = AlphaRetriever(
            vector_store=vector_store,
            embedder=embedder,
            stage1_k=top_k * 3,   # wide candidate pool
            stage2_k=top_k,        # after freshness rerank
            stage3_k=top_k,        # keep diversity filter light for sentiment
            token_budget=4_000,    # generous budget; sentiment models handle long input
        )
        logger.info(
            "LocalSocialDataRetriever initialised (days_back=%d, top_k=%d).",
            days_back, top_k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> SocialRetrievalResult:
        """
        Execute the RAG retrieval pipeline and return clean text chunks.

        Uses the full AlphaRetriever multi-stage pipeline:
            Hybrid RRF search → freshness rerank → diversity filter → token budget.

        Parameters
        ----------
        query : str
            Semantic search query, e.g. "NVIDIA earnings sentiment" or
            "AAPL market reaction".  Must be non-empty.
        ticker : str, optional
            Stock ticker to narrow results (e.g. "NVDA").  When provided,
            the Postgres filter ``filter_ticker = ticker`` is applied.
        days_back : int, optional
            Override the instance-level days_back for this specific call.

        Returns
        -------
        SocialRetrievalResult
            .chunks  → list of clean text strings (deduplicated, freshness-ordered)
            .sources → parallel list of source metadata dicts

        Raises
        ------
        ValueError
            If query is empty or whitespace-only.
        RuntimeError
            If the Supabase RPC call fails and no fallback data is available.

        Examples
        --------
        >>> tool = LocalSocialDataRetriever()
        >>> result = tool.retrieve("NVIDIA AI chip demand", ticker="NVDA")
        >>> for text in result.chunks:
        ...     print(text[:120])
        """
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string.")

        effective_days_back = days_back if days_back is not None else self._days_back

        logger.info(
            "Retrieving social data: query='%s' ticker=%s days_back=%d",
            query, ticker, effective_days_back,
        )

        try:
            raw_chunks: list[dict] = self._retriever.retrieve_raw(
                query=query,
                ticker=ticker,
                days_back=effective_days_back,
                score_threshold=self._score_threshold,
            )
        except Exception as exc:
            logger.error("RAG retrieval failed: %s", exc)
            # Return an empty result rather than crashing the Sentiment Agent
            return SocialRetrievalResult(query=query, ticker=ticker)

        if not raw_chunks:
            logger.warning(
                "No chunks found for query='%s' ticker=%s.", query, ticker
            )
            return SocialRetrievalResult(query=query, ticker=ticker)

        chunks:  list[str]  = []
        sources: list[dict] = []

        for chunk in raw_chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            chunks.append(text)
            sources.append({
                "ticker":       chunk.get("ticker", "N/A"),
                "source_type":  chunk.get("source_type", "N/A"),
                "published_at": chunk.get("published_at", "N/A"),
                "url":          chunk.get("url", "N/A"),
                "title":        chunk.get("title", "N/A"),
                "rrf_score":    round(chunk.get("rrf_score", 0.0), 6),
            })

        result = SocialRetrievalResult(
            query=query,
            ticker=ticker,
            chunks=chunks,
            sources=sources,
            total_found=len(chunks),
        )
        logger.info(
            "Retrieved %d chunks for query='%s'.", result.total_found, query
        )
        return result

    # ------------------------------------------------------------------
    # Convenience: return only the raw text list (for quick chaining)
    # ------------------------------------------------------------------

    def retrieve_texts(
        self,
        query: str,
        ticker: Optional[str] = None,
        days_back: Optional[int] = None,
    ) -> list[str]:
        """
        Thin wrapper around retrieve() that returns only the text list.

        Useful when callers (e.g. FinBertSentimentAnalyzer) only need
        the raw strings and not the full SocialRetrievalResult.

        Parameters
        ----------
        query : str
            Semantic search query.
        ticker : str, optional
            Stock ticker filter.
        days_back : int, optional
            Override days_back for this call.

        Returns
        -------
        list[str]
            Clean text strings ordered by relevance + freshness.
            Returns an empty list if no data found.
        """
        return self.retrieve(query=query, ticker=ticker, days_back=days_back).chunks
