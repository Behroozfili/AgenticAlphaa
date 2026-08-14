import os
import httpx
from typing import Optional

TAVILY_API_URL = "https://api.tavily.com/search"

# Tavily is a supplementary/discovery source, not the primary evidence for
# financial analysis (SEC filings are). Kept intentionally lean: fewer
# results, and each result's content capped -- "basic" search_depth can
# still return a few thousand characters per result, and "advanced" scrapes
# full pages. Uncapped, a handful of Tavily calls in a research loop can
# rival or exceed a single SEC EDGAR preview in size despite being a
# lower-priority source.
_DEFAULT_MAX_RESULTS = 3
_SNIPPET_CHARS = 500


async def tavily_search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    search_depth: str = "basic",       # "basic" | "advanced"
    include_domains: Optional[list] = None,
    topic: str = "finance",            # "general" | "news" | "finance"
    snippet_chars: int = _SNIPPET_CHARS,
) -> dict:
    """
    Real-time web search via Tavily API.
    Returns structured results: title, url, snippet (capped to
    `snippet_chars`), score, published_date.

    Defaults are intentionally conservative (3 results, 500-char snippets):
    this is a discovery/supplementary source, not the primary evidence a
    financial analysis should be built on -- callers needing the full page
    content for a specific promising result should re-fetch that one URL
    directly rather than requesting more/longer results from every hit.
    """
    api_key = os.environ["TAVILY_API_KEY"]

    payload = {
        "api_key":        api_key,
        "query":          query,
        "search_depth":   search_depth,
        "max_results":    max_results,
        "topic":          topic,
        "include_answer": True,
    }
    if include_domains:
        payload["include_domains"] = include_domains

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(TAVILY_API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return {
        "query":   query,
        "answer":  data.get("answer"),
        "results": [
            {
                "title":          r.get("title", ""),
                "url":            r.get("url", ""),
                "snippet":        r.get("content", "")[:snippet_chars],
                "score":          round(r.get("score", 0.0), 4),
                "published_date": r.get("published_date"),
            }
            for r in data.get("results", [])
        ],
    }