import os
import httpx
from datetime import date, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping, Optional

from core.forecast_contract import AnalysisMode, ForecastContext
from tools.research_tools import (
    context_metadata,
    eligible_evidence,
    historical_unavailable,
    observed_research_operation,
    require_research_context,
)

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


def _normalize_published_date(value: Any) -> Any:
    """Convert Tavily's RFC-2822 news timestamps to contract ISO-8601."""
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@observed_research_operation("tavily_search")
async def tavily_search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    search_depth: str = "basic",       # "basic" | "advanced"
    include_domains: Optional[list] = None,
    topic: str = "news",               # "general" | "news" | "finance"
    snippet_chars: int = _SNIPPET_CHARS,
    start_date: Optional[str] = None,   # YYYY-MM-DD; bounded by ForecastContext
    end_date: Optional[str] = None,     # YYYY-MM-DD; bounded by ForecastContext
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    archived_records: list[Mapping[str, Any]] | None = None,
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
    context, run_identity = require_research_context(forecast_context, run_id)
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        if archived_records is None:
            return historical_unavailable(
                context, run_identity, source="tavily_search",
                reason="Current Tavily/web results cannot represent the past; no approved local archive was supplied.",
            )
        evidence, rejected = eligible_evidence(
            archived_records, context=context, run_id=run_identity,
            source="tavily_search", historical=True,
        )
        return {
            **context_metadata(context, run_identity), "query": query,
            "answer": None, "results": evidence, "evidence": evidence,
            "quarantined": rejected, "status": "OK" if evidence else "DATA_UNAVAILABLE",
            "missing": not evidence, "degraded": bool(rejected) or not evidence,
        }

    api_key = os.environ["TAVILY_API_KEY"]

    # Bind live web retrieval to the feature month represented by the
    # immutable ForecastContext. Explicit bounds may narrow, never widen it.
    context_start = (context.as_of - timedelta(days=context.lookback_days)).date()
    context_end = context.as_of.date()
    requested_start = date.fromisoformat(start_date) if start_date else context_start
    requested_end = date.fromisoformat(end_date) if end_date else context_end
    effective_start = max(requested_start, context_start)
    effective_end = min(requested_end, context_end)
    if effective_start > effective_end:
        raise ValueError("Tavily date range does not overlap ForecastContext")

    payload = {
        "api_key":        api_key,
        "query":          query,
        "search_depth":   search_depth,
        "max_results":    max_results,
        "topic":          topic,
        "include_answer": True,
        "start_date":     effective_start.isoformat(),
        "end_date":       effective_end.isoformat(),
    }
    if include_domains:
        payload["include_domains"] = include_domains

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(TAVILY_API_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    raw_results = [
        {
            "title": r.get("title", ""), "url": r.get("url", ""),
            "snippet": r.get("content", "")[:snippet_chars],
            "score": round(r.get("score", 0.0), 4),
            "published_date": _normalize_published_date(r.get("published_date")),
            "ticker": context.ticker, "dataset_version": context.dataset_version,
            "provider": "tavily_live",
        }
        for r in data.get("results", [])
    ]
    evidence, rejected = eligible_evidence(
        raw_results, context=context, run_id=run_identity,
        source="tavily_search", historical=False,
    )
    eligible_urls = {
        str(item.get("citation", {}).get("url")): item for item in evidence
        if isinstance(item.get("citation"), dict)
    }
    results = [
        {**item, **eligible_urls[str(item.get("url"))]}
        for item in raw_results if str(item.get("url")) in eligible_urls
    ]
    return {
        **context_metadata(context, run_identity),
        "query":   query,
        "date_range": {
            "start": effective_start.isoformat(),
            "end": effective_end.isoformat(),
        },
        "answer": None,
        "answer_quarantined": bool(data.get("answer")),
        "results": results,
        "evidence": evidence,
        "quarantined": rejected,
        "status": "OK" if evidence else "DATA_UNAVAILABLE",
        "missing": not evidence,
        "degraded": bool(rejected) or not evidence,
    }
