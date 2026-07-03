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

# Generic finance vocabulary that appears in nearly EVERY article about a
# company's stock, regardless of what actually happened — these words
# describe the *category* of content ("this is stock-related, price-
# related, sentiment-related") rather than the specific topic. OR-ing them
# in (as an earlier version of this function did) made queries match
# almost any article mentioning the company at all, which is too broad —
# e.g. "Apple stock price movement analyst ratings sentiment" would match
# literally any Apple stock article. They're dropped entirely rather than
# OR'd, so only genuinely distinguishing terms drive the match.
_GENERIC_FILLER_WORDS = {
    "stock", "stocks", "share", "shares", "price", "prices", "market",
    "markets", "movement", "movements", "sentiment", "analysis",
    "outlook", "performance", "update", "updates", "news", "report",
    "reports", "review", "overview", "trend", "trends", "today",
    "latest", "current",
}

# NOTE: an earlier version of this function tried to auto-drop a
# "redundant ticker" token immediately after the company name (e.g. "AAPL"
# in "Apple AAPL earnings"). That heuristic was removed: it can't reliably
# distinguish a genuinely redundant ticker from a real short keyword in
# the same position — e.g. "Apple AI strategy" got mangled into
# "Apple AND strategy" because "AI" matched the same "1-5 uppercase
# letters right after the company name" pattern as a ticker would. Wrongly
# discarding a real keyword is worse than the minor cost of occasionally
# keeping a redundant ticker as an extra AND term (which usually still
# matches fine, since financial articles commonly mention the ticker
# alongside the company name anyway). So ticker-looking tokens are now
# just treated like any other term.


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


# NewsAPI's own relevancy matching can be looser than the boolean query
# implies — an AND'd term can technically satisfy the query by appearing
# once, in passing, in a long article that isn't really ABOUT that topic
# (e.g. a broad market-wrap article that mentions the company once while
# covering several others). The boolean query controls what CAN come
# back; this local filter controls what's actually TRUSTED as relevant,
# by requiring a minimum number of the query's own terms to show up in
# the article's title/description specifically (not just somewhere in the
# full body NewsAPI indexed) — title/description are a much stronger
# relevance signal than "appears somewhere in the article text".
def _count_term_matches(article: dict, terms: list[str]) -> list[str]:
    haystack = f"{article.get('title', '')} {article.get('description', '')}".lower()
    return [t for t in terms if t.lower() in haystack]


def _min_required_matches(n_terms: int) -> int:
    # Always require the anchor (first term) plus at least one more
    # distinguishing term when there is one — a single-term match (often
    # just the company name) isn't enough to trust an article as
    # genuinely on-topic when the original query had more to it.
    return min(2, n_terms)


async def news_search(
    query: str,
    from_date: Optional[str] = None,   # YYYY-MM-DD
    to_date: Optional[str] = None,     # YYYY-MM-DD
    language: str = "en",
    sort_by: str = "relevancy",        # "relevancy" | "popularity" | "publishedAt"
    page_size: int = 10,
    exclude_domains: Optional[list[str]] = None,
) -> dict:
    """
    Fetch financial news articles via NewsAPI.
    Returns: total_results, list of articles with title, url, source, published_at.

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

    # Local relevance filter: NewsAPI's boolean match can be satisfied by
    # a term appearing once, anywhere in the full article body — including
    # broad market-wrap pieces that mention the company only in passing.
    # Require a minimum number of the query's own terms to actually show
    # up in title/description (a much stronger signal than "somewhere in
    # the full text") before trusting an article as genuinely relevant.
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
        # Don't silently return nothing just because title/description
        # were terse — fall back to the unfiltered set rather than hiding
        # results the caller could still find useful, but flag it.
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

    # Surface low-confidence results explicitly rather than silently
    # returning a single weak match as if it were a solid hit. Callers
    # (agents/LLMs consuming this tool) can use this to decide whether to
    # retry with a narrower/shorter query or fall back to another tool.
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