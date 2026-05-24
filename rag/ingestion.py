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

import logging
import os

from dotenv import load_dotenv

from rag.loader import AlphaLoader, RawDocument
from rag.processor import AlphaProcessor, ProcessedChunk
from rag.embedding_manager import get_embedder
from rag.vector_store import AlphaVectorStore
from rag.graph_store import AlphaGraphStore

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("IngestionPipeline")


def run_ingestion_pipeline(
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
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
    )
    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing. Aborting.")
        return

    # ── Init components ───────────────────────────────────────────────────────
    loader       = AlphaLoader(max_news_per_ticker=20, max_rss_per_feed=30)
    processor    = AlphaProcessor()
    embedder     = get_embedder()                      # singleton — loaded once
    vector_store = AlphaVectorStore(supabase_url=supabase_url, supabase_key=supabase_key)
    graph_store  = AlphaGraphStore() if not skip_graph else None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — Load
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("─── Stage 1: Loading documents ───")
    try:
        raw_docs: list[RawDocument] = loader.load(tickers=tickers)
    except Exception as exc:
        logger.error("Load stage failed: %s", exc)
        return

    logger.info("Raw documents fetched: %d", len(raw_docs))
    if not raw_docs:
        logger.warning("No documents — pipeline terminated.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — Process & chunk
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("─── Stage 2: Processing & chunking ───")
    try:
        chunks: list[ProcessedChunk] = processor.process(raw_docs)
    except Exception as exc:
        logger.error("Process stage failed: %s", exc)
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
    try:
        # embed_chunks() is the public API — returns list of
        # {"text": str, "embedding": List[float], "metadata": dict}
        # which is exactly what AlphaVectorStore.upsert() expects.
        records = embedder.embed_chunks(chunks)

        # Batch upsert in one round-trip instead of N individual calls.
        upserted = vector_store.upsert(records=records)
        logger.info("Vector upsert: %d/%d chunks stored.", upserted, len(records))
        vector_stage_ok = True

    except Exception as exc:
        logger.error("Embed/vector stage failed: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5 — Graph extraction + upsert (Neo4j)
    # ─────────────────────────────────────────────────────────────────────────
    if not vector_stage_ok:
        logger.warning("Skipping Stage 5: vector stage did not complete successfully.")
    elif skip_graph:
        logger.info("Stage 5 skipped (skip_graph=True).")
    else:
        logger.info("─── Stage 5: Graph extraction + upsert ───")
        try:
            graph_docs = graph_store.extract_batch(raw_docs)
            summary    = graph_store.upsert_batch(graph_docs)
            logger.info(
                "Graph upsert: %d nodes, %d relationships.",
                summary["nodes_merged"], summary["rels_merged"],
            )
        except Exception as exc:
            logger.error("Graph stage failed: %s", exc)
        finally:
            if graph_store:
                graph_store.close()

    logger.info("═══ Ingestion Pipeline COMPLETE ═══")


if __name__ == "__main__":
    run_ingestion_pipeline(["MSFT", "NVDA"])