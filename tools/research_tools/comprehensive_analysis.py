import asyncio
from typing import Any, Mapping, Optional

from core.forecast_contract import ForecastContext
from tools.research_tools.tavily_search import tavily_search
from tools.research_tools.sec_edgar import sec_edgar_filing
from tools.research_tools import context_metadata, observed_research_operation, require_research_context


@observed_research_operation("comprehensive_analysis")
async def comprehensive_analysis(
    ticker: str,
    company_name: Optional[str] = None,
    topic_query: Optional[str] = None,     # descriptive query for Tavily, e.g. "Apple AI spending strategy 2026"
    form_type: str = "10-Q",
    sections: list = None,                 # e.g. ["mda"]
    max_results: int = 3,
    filing_mode: str = "preview",          # "preview" | "full" -- passed to sec_edgar_filing
    filing_query: Optional[str] = None,    # passed to sec_edgar_filing's `query` (keyword lookup within the section)
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    archived_web_records: list[Mapping[str, Any]] | None = None,
    archived_filing: Mapping[str, Any] | None = None,
) -> dict:
    """
    Hybrid research call: runs Tavily (soft/analyst-sentiment data) and SEC EDGAR
    (hard/official filing data) concurrently and returns both, so the agent
    doesn't have to choose one source or risk feeding a long natural-language
    query into EDGAR's full-text search (which degrades on long queries).

    - Tavily gets the full descriptive `topic_query` (or falls back to
      "<company_name or ticker> AI strategy" if not given) — it handles
      long natural-language queries well. `max_results` defaults to 3 and
      each result's content is capped by tavily_search's own snippet_chars
      default (500) -- Tavily is a supplementary/discovery source here, not
      the primary evidence.
    - SEC EDGAR gets `ticker` / `form_type` as separate structured params,
      never the raw long query, so it isn't truncated by the defensive
      6-term cap in sec_edgar.py's _sanitize_query. `filing_mode` defaults
      to "preview" (dense, compressed: total_chars, digest/key_figures,
      important_excerpts -- see sec_edgar_filing's own docstring), NOT the
      full section text. Pass filing_mode="full" once you already know you
      need the full text, or filing_query to jump straight to a keyword
      match inside the section regardless of mode.

    Returns: {"ticker": ..., "news": <tavily_search result>, "filing": <sec_edgar_filing result>}
    `filing` is the same shape sec_edgar_filing itself returns -- in the
    default preview mode, each requested section is a dict with
    total_chars/digest/key_figures/important_excerpts, NOT a plain string
    of section text.
    """
    context, run_identity = require_research_context(forecast_context, run_id)
    if ticker.strip().upper() != context.ticker:
        raise ValueError("ticker must match ForecastContext.ticker")
    ticker = context.ticker
    if sections is None:
        sections = ["mda"]

    name_for_query = company_name or ticker
    query = topic_query or f"{name_for_query} AI strategy spending"

    news_task = tavily_search(
        query=query, max_results=max_results, topic="news",
        forecast_context=context, run_id=run_identity,
        archived_records=archived_web_records,
    )
    filing_task = sec_edgar_filing(
        ticker=ticker, form_type=form_type, sections=sections,
        mode=filing_mode, query=filing_query,
        forecast_context=context, run_id=run_identity,
        archived_filing=archived_filing,
    )

    news_results, filing_results = await asyncio.gather(
        news_task, filing_task, return_exceptions=True
    )

    if isinstance(news_results, Exception):
        news_results = {"status": "FAIL_CLOSED", "error_code": "TAVILY_FAIL_CLOSED", "evidence": [], "missing": True, "degraded": True}
    if isinstance(filing_results, Exception):
        filing_results = {"status": "FAIL_CLOSED", "error_code": "SEC_FAIL_CLOSED", "evidence": [], "missing": True, "degraded": True}

    return {
        **context_metadata(context, run_identity),
        "ticker": ticker,
        "news": news_results,
        "filing": filing_results,
        "forecast_context": context.to_dict(),
        "missing": bool(news_results.get("missing")) and bool(filing_results.get("missing")),
        "degraded": bool(news_results.get("degraded")) or bool(filing_results.get("degraded")),
    }
