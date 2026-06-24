import asyncio
from typing import Optional

from tools.research_tools.tavily_search import tavily_search
from tools.research_tools.sec_edgar import sec_edgar_filing, sec_edgar_search


async def comprehensive_analysis(
    ticker: str,
    company_name: Optional[str] = None,
    topic_query: Optional[str] = None,     # descriptive query for Tavily, e.g. "Apple AI spending strategy 2026"
    form_type: str = "10-Q",
    sections: list = None,                 # e.g. ["mda"]
    max_results: int = 5,
) -> dict:
    """
    Hybrid research call: runs Tavily (soft/analyst-sentiment data) and SEC EDGAR
    (hard/official filing data) concurrently and returns both, so the agent
    doesn't have to choose one source or risk feeding a long natural-language
    query into EDGAR's full-text search (which degrades on long queries).

    - Tavily gets the full descriptive `topic_query` (or falls back to
      "<company_name or ticker> AI strategy" if not given) — it handles
      long natural-language queries well.
    - SEC EDGAR gets `ticker` / `form_type` as separate structured params,
      never the raw long query, so it isn't truncated by the defensive
      6-term cap in sec_edgar.py's _sanitize_query.

    Returns: {"ticker": ..., "news": <tavily_search result>, "filing": <sec_edgar_filing result>}
    """
    if sections is None:
        sections = ["mda"]

    name_for_query = company_name or ticker
    query = topic_query or f"{name_for_query} AI strategy spending"

    news_task = tavily_search(query=query, max_results=max_results, topic="finance")
    filing_task = sec_edgar_filing(ticker=ticker, form_type=form_type, sections=sections)

    news_results, filing_results = await asyncio.gather(
        news_task, filing_task, return_exceptions=True
    )

    if isinstance(news_results, Exception):
        news_results = {"query": query, "error": str(news_results)}
    if isinstance(filing_results, Exception):
        filing_results = {"ticker": ticker, "error": str(filing_results)}

    return {
        "ticker": ticker,
        "news": news_results,
        "filing": filing_results,
    }