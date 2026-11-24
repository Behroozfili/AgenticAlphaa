"""Backfill temporal V2 Vector and Graph evidence from eligible legacy rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import timedelta
from typing import Any, Iterable

from core.forecast_contract import parse_utc_datetime
from rag.embedding_manager import get_embedder
from rag.graph_store import AlphaGraphStore
from rag.loader import RawDocument
from rag.processor import AlphaProcessor
from rag.vector_store import AlphaVectorStore, VectorStoreConfig

_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")
_VERSION = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SOURCE_TYPES = {"news", "rss", "reddit", "filing", "archive"}


def rows_to_documents(
    rows: Iterable[dict[str, Any]], *, ticker: str, dataset_version: str,
) -> list[RawDocument]:
    documents: list[RawDocument] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        url_hash = str(row.get("url_hash") or "").strip()
        chunk_index = int(row.get("chunk_index") or 0)
        identity = (url_hash, chunk_index)
        published_at = str(row.get("published_at") or "").strip()
        retrieved_at = str(row.get("ingested_at") or "").strip()
        text = str(row.get("text") or "").strip()
        url = str(row.get("url") or "").strip()
        if identity in seen or not all((url_hash, published_at, retrieved_at, text, url)):
            continue
        seen.add(identity)
        source_type = str(row.get("source_type") or "archive").strip().lower()
        if source_type not in _SOURCE_TYPES:
            source_type = "archive"
        documents.append(RawDocument(
            title=str(row.get("title") or ticker)[:512],
            content=text,
            url=url,
            source_type=source_type,  # type: ignore[arg-type]
            ticker=ticker,
            published_at=published_at,
            retrieved_at=retrieved_at,
            effective_from=published_at,
            effective_to=None,
            dataset_version=dataset_version,
            source_id=f"legacy-vector:{url_hash}",
            revision=f"legacy-chunk-{chunk_index}",
            metadata={"materialized_at": retrieved_at, "legacy_vector_backfill": True},
        ))
    return documents


async def backfill_graph(
    *, ticker: str, as_of: str, dataset_version: str, limit: int = 20,
    lookback_days: int = 365,
) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    dataset_version = dataset_version.strip()
    if not _TICKER.fullmatch(ticker):
        raise ValueError("ticker must be a compact normalized symbol")
    if not _VERSION.fullmatch(dataset_version):
        raise ValueError("dataset_version must be a compact identifier")
    if limit <= 0 or lookback_days <= 0:
        raise ValueError("limit and lookback_days must be positive")
    cutoff = parse_utc_datetime(as_of, field_name="as_of")
    start = cutoff - timedelta(days=lookback_days)

    vector = AlphaVectorStore(config=VectorStoreConfig(schema="legacy"))
    rows = (
        vector.client.table(vector.table_name)
        .select("url_hash,ticker,source_type,published_at,ingested_at,chunk_index,url,title,text")
        .eq("ticker", ticker)
        .gte("published_at", start.isoformat())
        .lte("published_at", cutoff.isoformat())
        .lte("ingested_at", cutoff.isoformat())
        .order("published_at", desc=True)
        .limit(limit)
        .execute().data or []
    )
    documents = rows_to_documents(rows, ticker=ticker, dataset_version=dataset_version)
    chunks = AlphaProcessor().process(documents)
    records = get_embedder().embed_chunks(chunks)
    vector_v2 = AlphaVectorStore(config=VectorStoreConfig(schema="v2"))
    vector_upserted = vector_v2.upsert(records)
    graph = AlphaGraphStore()
    graph.connect()
    try:
        graph_documents = await graph.extract_batch(documents)
        write = graph.upsert_batch(graph_documents)
    finally:
        graph.close()
    return {
        "ticker": ticker,
        "as_of": cutoff.isoformat(),
        "dataset_version": dataset_version,
        "eligible_rows": len(rows),
        "documents": len(documents),
        "chunks": len(chunks),
        "vector_upserted": vector_upserted,
        "graph_documents": len(graph_documents),
        **write,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.backfill_graph",
        description="Backfill temporal V2 Vector and Neo4j evidence from point-in-time eligible legacy rows",
    )
    parser.add_argument("ticker")
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=365)
    return parser


def cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(backfill_graph(
        ticker=args.ticker, as_of=args.as_of,
        dataset_version=args.dataset_version,
        limit=args.limit, lookback_days=args.lookback_days,
    ))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())


__all__ = ["backfill_graph", "build_parser", "cli", "rows_to_documents"]
