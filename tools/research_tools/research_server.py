"""
Research Agent — MCP Server
============================
Registers the following tools and exposes them over stdio MCP protocol:

  1. tavily_search      — real-time web search
  2. news_search        — financial news via NewsAPI
  3. sec_edgar_search   — EDGAR full-text filing search
  4. sec_edgar_filing   — fetch & parse a specific SEC filing
  5. rag_vector_search  — semantic search over Supabase pgvector
  6. rag_graph_traverse — Neo4j entity-relationship traversal
  7. rag_hybrid_query   — RRF fusion of vector + graph
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, ListToolsResult

# ── Tool implementations ───────────────────────────────────────────
from tavily_search import tavily_search
from news_search   import news_search
from sec_edgar     import sec_edgar_search, sec_edgar_filing
from rag.hybrid_rag import rag_vector_search, rag_graph_traverse, rag_hybrid_query
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("research-mcp")

app = Server("research-agent-mcp")


# ══════════════════════════════════════════════════════════════════
# LIST TOOLS  —  advertise every tool to the MCP client
# ══════════════════════════════════════════════════════════════════
@app.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=[

        # ── 1. Tavily Search ───────────────────────────────────────
        Tool(
            name="tavily_search",
            description="Real-time web search via Tavily API. Best for recent news, company events, macro developments.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default: 5)",
                        "default": 5
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "'basic' = fast snippet, 'advanced' = full page scrape",
                        "default": "basic"
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these domains (optional)"
                    },
                    "topic": {
                        "type": "string",
                        "enum": ["general", "news", "finance"],
                        "default": "finance"
                    },
                },
            },
        ),

        # ── 2. NewsAPI ─────────────────────────────────────────────
        Tool(
            name="news_search",
            description="Fetch financial news articles via NewsAPI. Supports date range, language, and sort order.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "News search query (e.g. 'NVIDIA earnings Q4')"
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD (default: 30 days ago)"
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD (optional)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code, e.g. 'en', 'de'",
                        "default": "en"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevancy", "popularity", "publishedAt"],
                        "default": "publishedAt"
                    },
                    "page_size": {
                        "type": "integer",
                        "description": "Number of articles (default: 10, max: 100)",
                        "default": 10
                    },
                },
            },
        ),

        # ── 3. SEC EDGAR — search ──────────────────────────────────
        Tool(
            name="sec_edgar_search",
            description="Full-text search across SEC EDGAR filings. Returns filing metadata: company, form type, date, accession number.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term against EDGAR full-text index"
                    },
                    "ticker": {
                        "type": "string",
                        "description": "Filter by stock ticker (e.g. 'NVDA')"
                    },
                    "form_type": {
                        "type": "string",
                        "description": "Filter by form: '10-K', '10-Q', '8-K'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max filings to return (default: 5)",
                        "default": 5
                    },
                },
            },
        ),

        # ── 4. SEC EDGAR — fetch filing ────────────────────────────
        Tool(
            name="sec_edgar_filing",
            description="Fetch and parse the latest SEC filing (10-K/10-Q) for a ticker. Returns named sections: business, risk_factors, mda, financial_statements.",
            inputSchema={
                "type": "object",
                "required": ["ticker"],
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. 'NVDA')"
                    },
                    "form_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K"],
                        "default": "10-K"
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["business", "risk_factors", "mda", "financial_statements", "all"]
                        },
                        "description": "Sections to extract",
                        "default": ["all"]
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters per section (default: 8000)",
                        "default": 8000
                    },
                },
            },
        ),

        # ── 5. RAG — Vector Search ─────────────────────────────────
        Tool(
            name="rag_vector_search",
            description="Semantic similarity search over Supabase pgvector knowledge base. Finds relevant chunks from pre-ingested 10-K reports and research documents.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query for semantic search"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default: 5)",
                        "default": 5
                    },
                    "ticker_filter": {
                        "type": "string",
                        "description": "Filter results to a specific ticker (optional)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum RRF score (default: 0.01)",
                        "default": 0.01
                    },
                },
            },
        ),

        # ── 6. RAG — Graph Traversal ───────────────────────────────
        Tool(
            name="rag_graph_traverse",
            description="Traverse Neo4j knowledge graph to discover entity relationships. Finds competitors, suppliers, geopolitical impacts. Relations: COMPETES_WITH, SUPPLIES_TO, AFFECTED_BY, LED_BY, PART_OF, RELATED_TO, ACQUIRED_BY.",
            inputSchema={
                "type": "object",
                "required": ["entity"],
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Starting entity: ticker or company name (e.g. 'NVDA' or 'NVIDIA')"
                    },
                    "relation_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["COMPETES_WITH", "SUPPLIES_TO", "AFFECTED_BY", "LED_BY", "PART_OF", "RELATED_TO", "ACQUIRED_BY", "ALL"]
                        },
                        "default": ["ALL"]
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Traversal depth (1-3, default: 2)",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 3
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max nodes to return (default: 20)",
                        "default": 20
                    },
                },
            },
        ),

        # ── 7. RAG — Hybrid Query ──────────────────────────────────
        Tool(
            name="rag_hybrid_query",
            description="Fuses vector similarity search + graph traversal via Reciprocal Rank Fusion. Best for complex queries requiring both semantic matching AND relationship context. Example: 'How does Taiwan conflict affect NVIDIA supply chain?'",
            inputSchema={
                "type": "object",
                "required": ["query", "entity"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language research query"
                    },
                    "entity": {
                        "type": "string",
                        "description": "Primary entity for graph traversal (ticker or company name)"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Vector results to retrieve (default: 5)",
                        "default": 5
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Graph traversal depth (default: 2)",
                        "default": 2
                    },
                    "fusion": {
                        "type": "string",
                        "enum": ["rrf", "weighted", "union"],
                        "description": "'rrf'=Reciprocal Rank Fusion, 'weighted'=score-weighted, 'union'=all results",
                        "default": "rrf"
                    },
                },
            },
        ),

    ])


# ══════════════════════════════════════════════════════════════════
# CALL TOOL  —  route each call to the correct implementation
# ══════════════════════════════════════════════════════════════════
@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    log.info("→ tool=%s args=%s", name, json.dumps(arguments, ensure_ascii=False))

    try:
        match name:

            case "tavily_search":
                result = await tavily_search(
                    query=arguments["query"],
                    max_results=arguments.get("max_results", 5),
                    search_depth=arguments.get("search_depth", "basic"),
                    include_domains=arguments.get("include_domains"),
                    topic=arguments.get("topic", "finance"),
                )

            case "news_search":
                result = await news_search(
                    query=arguments["query"],
                    from_date=arguments.get("from_date"),
                    to_date=arguments.get("to_date"),
                    language=arguments.get("language", "en"),
                    sort_by=arguments.get("sort_by", "publishedAt"),
                    page_size=arguments.get("page_size", 10),
                )

            case "sec_edgar_search":
                result = await sec_edgar_search(
                    query=arguments["query"],
                    ticker=arguments.get("ticker"),
                    form_type=arguments.get("form_type"),
                    max_results=arguments.get("max_results", 5),
                )

            case "sec_edgar_filing":
                result = await sec_edgar_filing(
                    ticker=arguments["ticker"],
                    form_type=arguments.get("form_type", "10-K"),
                    sections=arguments.get("sections", ["all"]),
                    max_chars=arguments.get("max_chars", 8000),
                )

            case "rag_vector_search":
                result = await rag_vector_search(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 5),
                    ticker_filter=arguments.get("ticker_filter"),
                    threshold=arguments.get("threshold", 0.01),
                )

            case "rag_graph_traverse":
                result = await rag_graph_traverse(
                    entity=arguments["entity"],
                    relation_types=arguments.get("relation_types", ["ALL"]),
                    max_hops=arguments.get("max_hops", 2),
                    limit=arguments.get("limit", 20),
                )

            case "rag_hybrid_query":
                result = await rag_hybrid_query(
                    query=arguments["query"],
                    entity=arguments["entity"],
                    top_k=arguments.get("top_k", 5),
                    max_hops=arguments.get("max_hops", 2),
                    fusion=arguments.get("fusion", "rrf"),
                )

            case _:
                raise ValueError(f"Unknown tool: {name}")

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        )

    except Exception as exc:
        log.exception("Tool %s failed", name)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"error": str(exc), "tool": name}))],
            isError=True,
        )


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════
async def main():
    log.info("Research Agent MCP Server starting...")
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())