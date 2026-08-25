import os
import re
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

# NewsAPI's /v2/everything endpoint indexes tens of thousands of sources,
# many of which aren't financial news at all (dev blogs, code hosts,
# aggregators). Without an explicit exclude list, a query with few true
# matches can surface something like a GitHub repo README as its top (and
# only) "article" purely because it shares a couple of words with the
# query. This default list can be extended via the exclude_domains param;
# pass exclude_domains=[] to disable exclusion entirely.
DEFAULT_EXCLUDED_DOMAINS = [
    "github.com",
    "news.ycombinator.com",
    "stackoverflow.com",
    "reddit.com",       # Reddit sentiment is ingested separately via RSS
    "medium.com",       # frequently low-signal, un-vetted opinion posts
]

# Relative-time phrases that add nothing to relevance (from_date/to_date
# already scope the request by date) but make the literal word-match
# harder to satisfy. Observed in production: "Apple AAPL earnings revenue
# guidance last 14 days" returned 0 results partly because of this
# trailing phrase competing for a match against every other word.
_RELATIVE_TIME_PATTERN = re.compile(
    r"\b(last|past|previous|recent)\s+\d*\s*(day|days|week|weeks|month|months|"
    r"year|years|quarter|quarters)\b",
    re.IGNORECASE,
)


_GENERIC_FILLER_WORDS = {
    "stock", "stocks", "share", "shares", "price", "prices", "market",
    "markets", "movement", "movements", "sentiment", "analysis",
    "outlook", "performance", "update", "updates", "news", "report",
    "reports", "review", "overview", "trend", "trends", "today",
    "latest", "current",
}



def _extract_terms(query: str) -> list[str]:
    """
    Clean and reduce a raw query into the list of terms that actually
    drive relevance: strip relative-time phrases, drop generic filler
    words, and dedupe case-insensitively. Shared by _build_boolean_query
    (for the NewsAPI request) and the post-fetch relevance filter (for
    checking what NewsAPI actually matched on).
    """
    cleaned = _RELATIVE_TIME_PATTERN.sub(" ", query)
    raw_terms = [t for t in cleaned.replace('"', "").split() if t]

    if not raw_terms:
        return []

    anchor, tail = raw_terms[0], raw_terms[1:]
    substantive_tail = [t for t in tail if t.lower() not in _GENERIC_FILLER_WORDS]
    terms = [anchor] + substantive_tail

    seen: set[str] = set()
    deduped_terms = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            deduped_terms.append(t)
    return deduped_terms


def _build_boolean_query(terms: list[str]) -> str:
    """
    Build the NewsAPI boolean query string from an already-cleaned term
    list: AND all of them if there are few, otherwise anchor the first two
    and OR the rest so the query doesn't over-constrain.
    """
    if not terms:
        return ""
    if len(terms) <= 3:
        return " AND ".join(terms)
    head, rest = terms[:2], terms[2:]
    return f'{" AND ".join(head)} AND ({" OR ".join(rest)})'


def _restructure_query(query: str) -> str:
    """Convenience wrapper: extract terms and build the boolean query in
    one call. Falls back to the original query string if nothing survives
    extraction (e.g. an all-filler or empty query)."""
    terms = _extract_terms(query)
    return _build_boolean_query(terms) or query



def _count_term_matches(article: dict, terms: list[str]) -> list[str]:
    haystack = f"{article.get('title', '')} {article.get('description', '')}".lower()
    return [t for t in terms if t.lower() in haystack]


def _min_required_matches(n_terms: int) -> int:
   
    return min(2, n_terms)


async def news_search(
    query: str,
    from_date: Optional[str] = None,   # YYYY-MM-DD
    to_date: Optional[str] = None,     # YYYY-MM-DD
    language: str = "en",
    sort_by: str = "relevancy",        # "relevancy" | "popularity" | "publishedAt"
    page_size: int = 5,
    exclude_domains: Optional[list[str]] = None,
) -> dict:
    """
    Fetch financial news articles via NewsAPI.
    Returns: total_results, list of articles with title, url, source, published_at.

    page_size defaults to 5 (was 10). News is a supplementary/recency
    source relative to SEC filings, not the primary evidence for financial
    fundamentals -- and the same story is routinely republished across many
    outlets, so a larger page_size mostly buys near-duplicate coverage of
    one event rather than genuinely new information. Callers explicitly
    surveying market reaction/coverage breadth can still raise it.

    sort_by defaults to "relevancy" (was "publishedAt"). Sorting by recency
    means the newest article wins even when it's a weak/coincidental match —
    for a niche query with few true hits, that surfaces irrelevant results.
    Relevancy sorting is the correct default; callers can still opt into
    publishedAt/popularity explicitly when recency genuinely is the goal
    (e.g. "latest news on X").

    exclude_domains defaults to DEFAULT_EXCLUDED_DOMAINS (filters out
    non-financial-news noise like GitHub/HN/Reddit/Medium). Pass an empty
    list to disable filtering, or a custom list to override the defaults.

    The query is defensively restructured before hitting NewsAPI — see
    _restructure_query(). Multi-word natural-language queries (which the
    calling LLM keeps producing despite prompt guidance to use keywords)
    get rewritten from an implicit AND-of-all-words into an explicit
    "first_term AND (rest OR'd)" form, since requiring every word in a
    full sentence to co-occur in one article routinely returns 0 results.
    """
    api_key = os.environ["NEWSAPI_KEY"]

    if not from_date:
        from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")

    sort_by = SORT_BY_MAP.get(sort_by, "relevancy")

    if exclude_domains is None:
        exclude_domains = DEFAULT_EXCLUDED_DOMAINS

    effective_query = _restructure_query(query)
    terms = _extract_terms(query)

    params = {
        "apiKey":   api_key,
        "q":        effective_query,
        "from":     from_date,
        "language": language,
        "sortBy":   sort_by,
        "pageSize": min(page_size, 100),
    }
    if to_date:
        params["to"] = to_date
    if exclude_domains:
        params["excludeDomains"] = ",".join(exclude_domains)

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(NEWSAPI_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    raw_articles = [
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


    min_matches = _min_required_matches(len(terms))
    filtered_articles = []
    for article in raw_articles:
        matched = _count_term_matches(article, terms)
        if len(matched) >= min_matches:
            filtered_articles.append({**article, "matched_terms": matched})

    articles_were_filtered = len(filtered_articles) < len(raw_articles)

    if filtered_articles:
        articles = filtered_articles
    else:
    
        articles = raw_articles

    total_results = data.get("totalResults", 0)

    result = {
        "query":            query,
        "effective_query":  effective_query,
        "total_results":    total_results,
        "articles":         articles,
    }

    if articles_were_filtered and filtered_articles:
        result["relevance_filter_applied"] = True
        result["relevance_filter_note"] = (
            f"{len(raw_articles) - len(filtered_articles)} of "
            f"{len(raw_articles)} article(s) from NewsAPI were dropped for "
            f"not containing at least {min_matches} of the query terms "
            f"{terms!r} in their title/description — likely mentioned the "
            "company only in passing rather than being about this topic."
        )
    elif articles_were_filtered and not filtered_articles:
        result["relevance_filter_bypassed"] = True
        result["relevance_filter_note"] = (
            f"All {len(raw_articles)} article(s) from NewsAPI failed the "
            f"local relevance check (title/description didn't contain "
            f"enough of {terms!r}), but they're returned anyway rather "
            "than hiding potentially-useful results — review titles "
            "carefully before treating these as strongly on-topic."
        )

   
    if total_results <= 1 or len(articles) <= 1:
        result["low_confidence"] = True
        result["warning"] = (
            f"Only {total_results} raw result(s) from NewsAPI, "
            f"{len(articles)} after relevance filtering, for "
            f"effective_query={effective_query!r} (original query="
            f"{query!r}). If this keeps happening for a well-covered "
            "topic, try a shorter query with just the company/ticker and "
            "one keyword, or use tavily_search / rag_hybrid_query instead "
            "— they handle natural-language queries better than NewsAPI."
        )

    return result