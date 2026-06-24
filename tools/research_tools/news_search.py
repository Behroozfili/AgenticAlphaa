import os
import httpx
from typing import Optional
from datetime import datetime, timedelta

NEWSAPI_URL = "https://newsapi.org/v2/everything"

SORT_BY_MAP = {
    "relevance": "relevancy",
    "relevancy": "relevancy",
    "popularity": "popularity",
    "publishedAt": "publishedAt",
    "published_at": "publishedAt",
    "date": "publishedAt",
}


async def news_search(
    query: str,
    from_date: Optional[str] = None,   # YYYY-MM-DD
    to_date: Optional[str] = None,     # YYYY-MM-DD
    language: str = "en",
    sort_by: str = "publishedAt",      # "relevancy" | "popularity" | "publishedAt"
    page_size: int = 10,
) -> dict:
    """
    Fetch financial news articles via NewsAPI.
    Returns: total_results, list of articles with title, url, source, published_at.
    """
    api_key = os.environ["NEWSAPI_KEY"]

    if not from_date:
        from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    sort_by = SORT_BY_MAP.get(sort_by, "publishedAt")

    params = {
        "apiKey":   api_key,
        "q":        query,
        "from":     from_date,
        "language": language,
        "sortBy":   sort_by,
        "pageSize": min(page_size, 100),
    }
    if to_date:
        params["to"] = to_date

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(NEWSAPI_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    articles = [
        {
            "title":        a.get("title", ""),
            "description":  a.get("description", ""),
            "url":          a.get("url", ""),
            "source":       a.get("source", {}).get("name", ""),
            "author":       a.get("author"),
            "published_at": a.get("publishedAt", ""),
        }
        for a in data.get("articles", [])
        if a.get("title") != "[Removed]"
    ]

    return {
        "query":         query,
        "total_results": data.get("totalResults", 0),
        "articles":      articles,
    }