"""
seed_rag_data.py 
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