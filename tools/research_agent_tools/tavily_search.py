import os
import httpx
from typing import Optional

TAVILY_API_URL = "https://api.tavily.com/search"


async def tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",       # "basic" | "advanced"
    include_domains: Optional[list] = None,
    topic: str = "finance",            # "general" | "news" | "finance"
) -> dict:
    """
    Real-time web search via Tavily API.
    Returns structured results: title, url, snippet, score, published_date.
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
                "snippet":        r.get("content", ""),
                "score":          round(r.get("score", 0.0), 4),
                "published_date": r.get("published_date"),
            }
            for r in data.get("results", [])
        ],
    }
