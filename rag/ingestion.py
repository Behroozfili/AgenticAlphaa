"""
rag/ingestion.py — Full ETL Pipeline
Vector ingestion (Supabase) + Graph ingestion (Neo4j) in one run.

Stages:
    1. Load      — AlphaLoader (yfinance news + Reddit RSS)
    2. Process   — AlphaProcessor (chunk + deduplicate)
    3. Embed     — AlphaEmbedder (BAAI/bge-small-en-v1.5)
    4. Vector    — AlphaVectorStore upsert → Supabase pgvector
    5. Graph     — AlphaGraphStore extract + upsert → Neo4j
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from rag.loader import AlphaLoader, RawDocument
from rag.processor import AlphaProcessor, ProcessedChunk
from rag.embedding_manager import get_embedder
from rag.vector_store import AlphaVectorStore
from rag.graph_store import AlphaGraphStore
from core.observability import init_sentry, sentry_enabled

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logger = logging.getLogger("IngestionPipeline")


async def run_ingestion_pipeline(
    tickers: list[str],
    skip_graph: bool = False,       # set True to run vector-only (faster / cheaper)
) -> None:
    """
    Full ETL: Load → Process → Embed → Vector upsert → Graph upsert.

    Args:
        tickers:    Stock tickers to ingest (e.g. ["NVDA", "MSFT"]).
        skip_graph: If True, skips Neo4j graph population (Stage 5).
    """
    logger.info("═══ Ingestion Pipeline START — tickers=%s ═══", tickers)

    # ── Validate env ──────────────────────────────────────────────────────────
    # rag/ must not depend on api/ — read env vars directly here.
    # SUPABASE_KEY is the canonical name; SUPABASE_SERVICE_ROLE_KEY is the
    # raw env var that Settings maps to it (kept for backward compatibility).
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        or os.environ.get("SUPABASE_KEY", "")
    )
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing. Aborting.")
        return

    # ── Init components ───────────────────────────────────────────────────────
    loader       = AlphaLoader(max_news_per_ticker=20, max_rss_per_feed=30)
    processor    = AlphaProcessor()
    embedder     = get_embedder()                      # singleton — loaded once
    vector_store = AlphaVectorStore(supabase_url=supabase_url, supabase_key=supabase_key)
    graph_store  = None
    if not skip_graph:
        graph_store = AlphaGraphStore()
        graph_store.connect()

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — Load
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("─── Stage 1: Loading documents ───")
    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="ingestion",
            message="Stage 1: Loading documents",
            data={"tickers": tickers, "stage": "load"},
            level="info",
        )
    try:
        raw_docs: list[RawDocument] = loader.load(tickers=tickers)
    except Exception as exc:
        logger.error("Load stage failed: %s", exc)
        if sentry_enabled():
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "ingestion.load")
                sentry_sdk.capture_exception(exc)
        return

    logger.info("Raw documents fetched: %d", len(raw_docs))
    if not raw_docs:
        logger.warning("No documents — pipeline terminated.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — Process & chunk
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("─── Stage 2: Processing & chunking ───")
    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="ingestion",
            message="Stage 2: Processing & chunking",
            data={"tickers": tickers, "stage": "process"},
            level="info",
        )
    try:
        chunks: list[ProcessedChunk] = processor.process(raw_docs)
    except Exception as exc:
        logger.error("Process stage failed: %s", exc)
        if sentry_enabled():
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "ingestion.process")
                sentry_sdk.capture_exception(exc)
        return

    logger.info("Processor metrics: %s", processor.metrics.report())
    if not chunks:
        logger.warning("No chunks after deduplication — pipeline terminated.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 + 4 — Embed → Vector upsert
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("─── Stage 3+4: Embedding + Vector upsert ───")
    vector_stage_ok = False
    if sentry_enabled():
        import sentry_sdk
        sentry_sdk.add_breadcrumb(
            category="ingestion",
            message="Stage 3+4: Embedding + Vector upsert",
            data={"tickers": tickers, "stage": "vector"},
            level="info",
        )
    try:
        records = embedder.embed_chunks(chunks)
        upserted = vector_store.upsert(records=records)
        logger.info("Vector upsert: %d/%d chunks stored.", upserted, len(records))
        vector_stage_ok = True

    except Exception as exc:
        logger.error("Embed/vector stage failed: %s", exc)
        if sentry_enabled():
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "ingestion.vector")
                sentry_sdk.capture_exception(exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5 — Graph extraction + upsert (Neo4j)
    # ─────────────────────────────────────────────────────────────────────────
    if not vector_stage_ok:
        logger.warning("Skipping Stage 5: vector stage did not complete successfully.")
    elif skip_graph:
        logger.info("Stage 5 skipped (skip_graph=True).")
    else:
        logger.info("─── Stage 5: Graph extraction + upsert ───")
        if sentry_enabled():
            import sentry_sdk
            sentry_sdk.add_breadcrumb(
                category="ingestion",
                message="Stage 5: Graph extraction + upsert",
                data={"tickers": tickers, "stage": "graph"},
                level="info",
            )
        try:
            graph_docs = await graph_store.extract_batch(raw_docs)
            summary    = graph_store.upsert_batch(graph_docs)
            logger.info(
                "Graph upsert: %d nodes, %d relationships.",
                summary["nodes_merged"], summary["rels_merged"],
            )
        except Exception as exc:
            logger.error("Graph stage failed: %s", exc)
            if sentry_enabled():
                import sentry_sdk
                with sentry_sdk.push_scope() as scope:
                    scope.set_tag("component", "ingestion.graph")
                    sentry_sdk.capture_exception(exc)
        finally:
            if graph_store:
                graph_store.close()

    logger.info("═══ Ingestion Pipeline COMPLETE ═══")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_sentry()
    asyncio.run(run_ingestion_pipeline(["MSFT", "NVDA"]))