"""
seed_rag_data.py — Quick RAG database seeder for local/test environments.

Runs the REAL ingestion pipeline (rag/ingestion.py) for a small list of
tickers, including BOTH stages:
    - Vector stage  : Supabase pgvector  (alpha_documents + alpha_hybrid_search)
    - Graph stage   : Neo4j knowledge graph (entities + relationships via Claude)

This populates Supabase's `alpha_documents` table with real news + Reddit
RSS content for each ticker, embedded via AlphaEmbedder (BAAI/bge-small-en-v1.5),
AND extracts entities/relationships from each document into Neo4j.

Usage
-----
    python seed_rag_data.py
    python seed_rag_data.py NVDA MSFT AAPL TSLA

Requirements
------------
    - .env (or environment) must have:
        SUPABASE_URL
        SUPABASE_SERVICE_ROLE_KEY   (or SUPABASE_KEY)
        ANTHROPIC_API_KEY           (required for graph entity extraction)
        NEO4J_URI                   (e.g. "neo4j+s://xxxx.databases.neo4j.io")
        NEO4J_USER                  (default: "neo4j")
        NEO4J_PASSWORD
    - The alpha_documents table + alpha_hybrid_search RPC must already
      exist in Supabase (see the SQL block at the top of rag/vector_store.py).

Note
----
If NEO4J_URI / NEO4J_PASSWORD are missing or unreachable, AlphaGraphStore
logs a warning and silently skips graph writes (no crash) — the vector
stage still completes normally either way.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from rag.ingestion import run_ingestion_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Default seed tickers if none are passed on the command line.
DEFAULT_TICKERS = ["NVDA", "MSFT", "AAPL", "TSLA", "AMD"]


async def main(tickers: list[str]) -> None:
    print(f"\n🌱 Seeding RAG database (vector + graph) for tickers: {tickers}\n")
    await run_ingestion_pipeline(tickers=tickers, skip_graph=False)
    print("\n✅ Seeding complete. Verify with:\n")
    print("   -- Vector store:")
    print("   select ticker, count(*) from alpha_documents group by ticker;\n")
    print("   -- Graph store (Neo4j Browser / cypher-shell):")
    print("   MATCH (n) RETURN labels(n)[0] AS type, count(*) ORDER BY count(*) DESC;\n")


if __name__ == "__main__":
    requested = sys.argv[1:] or DEFAULT_TICKERS
    asyncio.run(main([t.upper() for t in requested]))