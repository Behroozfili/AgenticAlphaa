"""
sentiment_server.py — Sentiment Agent MCP Server
==================================================
Exposes the four Sentiment Agent tools over the stdio MCP protocol.
The Sentiment Agent acts as an MCP Client, connecting to this server
and invoking these tools dynamically via standard protocol requests.

Registered tools
─────────────────
  1. retrieve_social_data   — AlphaRetriever via rag/retriever.py (RAG pipeline)
  2. analyze_finbert        — FinBertSentimentAnalyzer  (ProsusAI/finbert)
  3. score_vader            — VaderLexiconScorer         (NLTK VADER)
  4. calculate_fear_greed   — FearGreedIndexCalculator   (weighted aggregation)

RAG Integration
────────────────
retrieve_social_data now uses AlphaRetriever.retrieve_raw() directly from
rag/retriever.py — the same single entry point used by research_server.py.
LocalSocialDataRetriever has been removed to eliminate the duplicate RAG path.

Output contract for retrieve_social_data
──────────────────────────────────────────
{
  "chunks":          list[str],        -- plain text, ready for FinBERT/VADER
  "sources_metadata": list[dict],      -- parallel metadata per chunk
  "total_retrieved": int
}

Each sources_metadata entry:
{
  "ticker":       str | None,
  "source_type":  str | None,
  "published_at": str | None,
  "url":          str | None,
  "title":        str | None,
  "rrf_score":    float
}

Structural pattern
───────────────────
  Mirrors research_server.py exactly:
    • mcp.server.Server          as the application host
    • @app.list_tools()          to advertise tool schemas
    • @app.call_tool()           to route and execute calls
    • stdio_server()             as the transport layer
    • match/case                 for clean routing
    • json.dumps(result)         as the universal serialization contract

Usage
──────
  python sentiment_server.py          # stdio mode (for LangGraph MCP client)
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.observability import init_sentry, sentry_enabled
init_sentry()

from tools.sentiment_tools.fear_greed_calculator import FearGreedIndexCalculator
from tools.sentiment_tools.finbert_analyzer import FinBertSentimentAnalyzer
from tools.sentiment_tools.vader_scorer import VaderLexiconScorer
from rag.retriever import AlphaRetriever
from rag.vector_store import AlphaVectorStore
from rag.embedding_manager import get_embedder

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sentiment-mcp")

# ---------------------------------------------------------------------------
# Lazy-initialised tool singletons
# ---------------------------------------------------------------------------

_retriever:  AlphaRetriever             | None = None
_finbert:    FinBertSentimentAnalyzer   | None = None
_vader:      VaderLexiconScorer         | None = None
_fear_greed: FearGreedIndexCalculator   | None = None


def _get_retriever() -> AlphaRetriever:
    global _retriever
    if _retriever is None:
        vector_store = AlphaVectorStore(
            supabase_url=os.environ["SUPABASE_URL"],
            supabase_key=(
                os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
                or os.environ["SUPABASE_KEY"]
            ),
        )
        _retriever = AlphaRetriever(
            vector_store=vector_store,
            embedder=get_embedder(),
        )
    return _retriever


def _get_finbert() -> FinBertSentimentAnalyzer:
    global _finbert
    if _finbert is None:
        _finbert = FinBertSentimentAnalyzer()
    return _finbert


def _get_vader() -> VaderLexiconScorer:
    global _vader
    if _vader is None:
        _vader = VaderLexiconScorer()
    return _vader


def _get_fear_greed() -> FearGreedIndexCalculator:
    global _fear_greed
    if _fear_greed is None:
        finbert_w   = float(os.environ.get("FEAR_GREED_FINBERT_WEIGHT", "0.65"))
        vader_w     = float(os.environ.get("FEAR_GREED_VADER_WEIGHT",   "0.35"))
        _fear_greed = FearGreedIndexCalculator(
            finbert_weight=finbert_w,
            vader_weight=vader_w,
        )
    return _fear_greed


# ---------------------------------------------------------------------------
# RAG retrieval helper
# ---------------------------------------------------------------------------

def _retrieve_social_data(
    query:     str,
    ticker:    str | None,
    days_back: int,
) -> dict[str, Any]:
    """
    Fetch chunks from the RAG pipeline via AlphaRetriever.retrieve_raw().

    Uses retrieve_raw() instead of retrieve() so we get structured chunk
    dicts with metadata rather than a pre-formatted context string.
    This keeps the output contract identical to what the SentimentAgent
    expects: separate ``chunks`` and ``sources_metadata`` lists.

    Parameters
    ----------
    query     : Natural language query for hybrid search.
    ticker    : Optional ticker filter passed to Supabase.
    days_back : Recency window in days.

    Returns
    -------
    dict with keys:
        chunks           : list[str]   — plain text per chunk
        sources_metadata : list[dict]  — parallel metadata per chunk
        total_retrieved  : int
    """
    raw_chunks: list[dict] = _get_retriever().retrieve_raw(
        query=query,
        ticker=ticker,
        days_back=days_back,
    )

    chunks:   list[str]  = []
    metadata: list[dict] = []

    for chunk in raw_chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        chunks.append(text)
        metadata.append({
            "ticker":       chunk.get("ticker"),
            "source_type":  chunk.get("source_type"),
            "published_at": chunk.get("published_at"),
            "url":          chunk.get("url"),
            "title":        chunk.get("title"),
            "rrf_score":    chunk.get("rrf_score", chunk.get("freshness_score", 0.0)),
        })

    return {
        "chunks":           chunks,
        "sources_metadata": metadata,
        "total_retrieved":  len(chunks),
    }


# ---------------------------------------------------------------------------
# Serialization helper — dataclass → JSON-safe dict
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    """
    Recursively convert dataclasses, lists, and primitives to JSON-safe types.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

app = Server("sentiment-agent-mcp")


# ══════════════════════════════════════════════════════════════════
# LIST TOOLS
# ══════════════════════════════════════════════════════════════════

@app.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[

        # ── 1. RAG-based social data retrieval ────────────────────
        Tool(
            name="retrieve_social_data",
            description=(
                "Retrieve pre-ingested social and news text chunks from the RAG pipeline. "
                "Interfaces with AlphaRetriever (Supabase pgvector + FTS hybrid search) to "
                "fetch recent market buzz: yfinance news, Reddit RSS posts, and financial "
                "news articles relevant to a company or ticker. "
                "Returns a list of clean text chunks ready for sentiment analysis."
            ),
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Semantic search query, e.g. 'NVIDIA earnings sentiment' or "
                            "'Apple market reaction Q3'. Must be non-empty."
                        ),
                    },
                    "ticker": {
                        "type": "string",
                        "description": (
                            "Stock ticker to narrow results (e.g. 'NVDA', 'AAPL'). "
                            "When provided, the Postgres filter is applied. Optional."
                        ),
                    },
                    "days_back": {
                        "type": "integer",
                        "description": (
                            "Only consider documents ingested within the last N days. "
                            "Default: 7. Increase for longer time horizons."
                        ),
                        "default": 7,
                        "minimum": 1,
                        "maximum": 90,
                    },
                },
            },
        ),

        # ── 2. FinBertSentimentAnalyzer ────────────────────────────
        Tool(
            name="analyze_finbert",
            description=(
                "Run financial sentiment analysis using ProsusAI/finbert. "
                "FinBERT is fine-tuned on Financial PhraseBank and produces three "
                "class probabilities: bullish, bearish, neutral. "
                "Accepts a list of text chunks (typically from retrieve_social_data). "
                "Returns aggregated mean probabilities and a dominant label."
            ),
            inputSchema={
                "type": "object",
                "required": ["texts"],
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of text chunks to analyse. "
                            "Typically the 'chunks' field from retrieve_social_data output. "
                            "Empty strings are filtered automatically."
                        ),
                        "minItems": 1,
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Inference batch size (default: 16).",
                        "default": 16,
                        "minimum": 1,
                        "maximum": 64,
                    },
                },
            },
        ),

        # ── 3. VaderLexiconScorer ──────────────────────────────────
        Tool(
            name="score_vader",
            description=(
                "Score text chunks using NLTK VADER lexicon. "
                "VADER is optimised for social media and short financial text. "
                "Returns mean compound score [-1, +1] and aggregated pos/neg/neu means. "
                "Complements FinBERT — use both for the Fear/Greed calculation."
            ),
            inputSchema={
                "type": "object",
                "required": ["texts"],
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of text chunks to score. "
                            "Typically the 'chunks' field from retrieve_social_data output. "
                            "Empty strings are filtered automatically."
                        ),
                        "minItems": 1,
                    },
                },
            },
        ),

        # ── 4. FearGreedIndexCalculator ────────────────────────────
        Tool(
            name="calculate_fear_greed",
            description=(
                "Aggregate FinBERT and VADER outputs into a unified Fear/Greed index score. "
                "Applies weighted fusion (default: FinBERT 65%, VADER 35%) and normalises "
                "the result to [-1.0 (Extreme Fear / Bearish), +1.0 (Extreme Greed / Bullish)]. "
                "Returns a score, five-band label, confidence heuristic, and full diagnostics. "
                "Labels: 'Extreme Fear' | 'Fear' | 'Neutral' | 'Greed' | 'Extreme Greed'."
            ),
            inputSchema={
                "type": "object",
                "required": ["finbert_result", "vader_result"],
                "properties": {
                    "finbert_result": {
                        "type": "object",
                        "description": "The JSON output from analyze_finbert tool call.",
                        "required": ["bullish_prob", "bearish_prob", "neutral_prob"],
                        "properties": {
                            "bullish_prob":   {"type": "number"},
                            "bearish_prob":   {"type": "number"},
                            "neutral_prob":   {"type": "number"},
                            "label":          {"type": "string"},
                            "total_chunks":   {"type": "integer"},
                            "skipped_chunks": {"type": "integer"},
                        },
                    },
                    "vader_result": {
                        "type": "object",
                        "description": "The JSON output from score_vader tool call.",
                        "required": ["compound"],
                        "properties": {
                            "compound":      {"type": "number"},
                            "positive_mean": {"type": "number"},
                            "negative_mean": {"type": "number"},
                            "neutral_mean":  {"type": "number"},
                            "label":         {"type": "string"},
                            "total_chunks":  {"type": "integer"},
                        },
                    },
                    "finbert_weight": {
                        "type": "number",
                        "description": "Override FinBERT weight (0-1). Default: 0.65.",
                        "default": 0.65,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "vader_weight": {
                        "type": "number",
                        "description": "Override VADER weight (0-1). Default: 0.35.",
                        "default": 0.35,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
            },
        ),

    ])


# ══════════════════════════════════════════════════════════════════
# CALL TOOL
# ══════════════════════════════════════════════════════════════════

@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """
    Route an incoming MCP tool call to the correct tool implementation.

    retrieve_social_data runs in a thread via asyncio.to_thread() because
    AlphaRetriever.retrieve_raw() is synchronous (network I/O to Supabase).
    All other tools also run in threads as they involve heavy model inference.
    """
    log.info("→ tool=%s args=%s", name, json.dumps(arguments, ensure_ascii=False))

    try:
        match name:

            # ── 1. retrieve_social_data ────────────────────────────
            case "retrieve_social_data":
                result = await asyncio.to_thread(
                    _retrieve_social_data,
                    query=arguments["query"],
                    ticker=arguments.get("ticker"),
                    days_back=int(arguments.get("days_back", 7)),
                )

            # ── 2. analyze_finbert ─────────────────────────────────
            case "analyze_finbert":
                texts      = arguments["texts"]
                batch_size = int(arguments.get("batch_size", 16))
                analyzer   = _get_finbert()
                original_bs         = analyzer.batch_size
                analyzer.batch_size = batch_size
                result_obj = await asyncio.to_thread(analyzer.analyze, texts)
                analyzer.batch_size = original_bs
                result = _to_dict(result_obj)

            # ── 3. score_vader ─────────────────────────────────────
            case "score_vader":
                result_obj = await asyncio.to_thread(
                    _get_vader().score,
                    texts=arguments["texts"],
                )
                result = _to_dict(result_obj)

            # ── 4. calculate_fear_greed ────────────────────────────
            case "calculate_fear_greed":
                fw = arguments.get("finbert_weight")
                vw = arguments.get("vader_weight")

                if fw is not None or vw is not None:
                    fw          = float(fw if fw is not None else 0.65)
                    vw          = float(vw if vw is not None else 0.35)
                    calculator  = FearGreedIndexCalculator(
                        finbert_weight=fw,
                        vader_weight=vw,
                    )
                else:
                    calculator = _get_fear_greed()

                result_obj = await asyncio.to_thread(
                    calculator.calculate_from_dict,
                    finbert_dict=arguments["finbert_result"],
                    vader_dict=arguments["vader_result"],
                )
                result = _to_dict(result_obj)

            case _:
                raise ValueError(f"Unknown tool: '{name}'")

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2),
                )
            ]
        )

    except Exception as exc:
        log.exception("Tool '%s' raised an exception.", name)
        if sentry_enabled():
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("tool", name)
                scope.set_tag("server", "sentiment-agent-mcp")
                sentry_sdk.capture_exception(exc)
        error_payload = {"error": str(exc), "tool": name, "arguments": arguments}
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps(error_payload, ensure_ascii=False, indent=2),
                )
            ],
            isError=True,
        )


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

async def main() -> None:
    log.info("Sentiment Agent MCP Server starting (stdio mode)...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())