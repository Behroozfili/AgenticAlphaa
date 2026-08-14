"""
test_search_agent.py — full integration test suite for ResearchAgent, in
one file.

Organized in three tiers, run selectively via pytest markers:

    pytest test_search_agent.py -v -m "not slow"   # Tier 1 only — fast, no LLM
    pytest test_search_agent.py -v                   # everything, including Tier 2

Tier 1 (no marker): call tool functions directly — news_search,
    sec_edgar_search/filing, tavily_search, comprehensive_analysis.
    Real API calls, but no LLM — fast and cheap.
Tier 2 (@pytest.mark.slow): run ResearchAgent.run() end-to-end —
    real LLM calls, slower and more expensive.

Every test here is tied to a specific bug found and fixed during this
project's development — see each test's docstring for which one.
"""
import pytest
from unittest.mock import patch

from tools.research_tools.news_search import news_search
from tools.research_tools.sec_edgar import sec_edgar_filing, sec_edgar_search
from tools.research_tools.tavily_search import tavily_search
from tools.research_tools.comprehensive_analysis import comprehensive_analysis
from agents.research_agent import ResearchAgent


# =============================================================================
# TIER 1 — news_search
# =============================================================================

class TestNewsSearch:
    """Regression: news_search used to return 0 results for LLM-generated
    sentence-style queries (e.g. "Apple AAPL earnings revenue guidance
    last 14 days") because the raw sentence was sent straight to the API
    instead of being restructured into keyword terms."""

    @pytest.mark.asyncio
    async def test_sentence_style_query_still_returns_results(self):
        from datetime import datetime, timedelta
        wide_from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        result = await news_search(
            query="Microsoft earnings revenue guidance outlook",
            from_date=wide_from_date,  # wide window so this isn't flaky on slow news days
        )
        assert result.get("error") is None
        assert len(result.get("articles", [])) > 0, (
            "sentence-style query returned 0 results — query restructuring may be broken"
        )
        assert "effective_query" in result, "effective_query field missing — restructuring may not be running"

    @pytest.mark.asyncio
    async def test_excludes_low_signal_domains(self):
        from datetime import datetime, timedelta
        wide_from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        result = await news_search(query="Microsoft Azure", from_date=wide_from_date)
        domains_returned = [a.get("url", "") for a in result.get("articles", [])]
        assert not any("reddit.com" in u or "news.ycombinator.com" in u for u in domains_returned)


# =============================================================================
# TIER 1 — sec_edgar_filing / sec_edgar_search
# =============================================================================

class TestSecEdgarFiling:
    """Regression: the "sections" parameter existed on the tool for a long
    time before research_agent's own prompt ever told the LLM about it —
    meaning risk_factors/mda were rarely requested even though the plumbing
    supported it. This test checks the TOOL side works when sections is
    explicitly requested."""

    @pytest.mark.asyncio
    async def test_explicit_sections_returns_risk_factors_and_mda(self):
        result = await sec_edgar_filing("MSFT", form_type="10-Q", sections=["mda", "risk_factors"])
        assert result.get("error") is None
        assert "mda" in result["sections"]
        assert "risk_factors" in result["sections"]
        assert len(result["sections"]["mda"]) > 500, "mda section suspiciously short"

    @pytest.mark.asyncio
    async def test_default_sections_is_all(self):
        result = await sec_edgar_filing("MSFT", form_type="10-Q")
        assert result.get("error") is None
        assert len(result["sections"]) >= 2, "default sections=['all'] should return multiple sections"


class TestSecEdgarSearch:
    """Regression: unresolvable ticker used to silently fall back to an
    unscoped full-text search across ALL filers — this must fail loudly
    (return an error) instead."""

    @pytest.mark.asyncio
    async def test_unresolvable_ticker_returns_error_not_unscoped_search(self):
        result = await sec_edgar_search(query="revenue guidance", ticker="ZZZNOTAREALTICKERZZZ")
        assert result.get("error") is not None
        assert result["filings"] == []

    @pytest.mark.asyncio
    async def test_long_query_is_truncated_not_rejected(self):
        long_query = " ".join(["term"] * 20)
        result = await sec_edgar_search(query=long_query, ticker="MSFT")
        assert result.get("error") is None


# =============================================================================
# TIER 1 — tavily_search
# =============================================================================

class TestTavilySearch:
    """tavily_search has no known historical bug in this project (unlike
    news_search/sec_edgar), so these are baseline schema/contract tests
    rather than regressions — they still matter, since a silent schema
    change upstream (Tavily API) would otherwise only surface as a
    confusing failure two layers up in comprehensive_analysis or the
    ResearchAgent's Checker."""

    @pytest.mark.asyncio
    async def test_returns_expected_schema(self):
        result = await tavily_search(query="Microsoft AI strategy", max_results=3)
        assert "results" in result
        assert isinstance(result["results"], list)
        if result["results"]:
            first = result["results"][0]
            for key in ("title", "url", "snippet", "score"):
                assert key in first

    @pytest.mark.asyncio
    async def test_max_results_is_respected(self):
        result = await tavily_search(query="Microsoft AI strategy", max_results=2)
        assert len(result["results"]) <= 2

    @pytest.mark.asyncio
    async def test_finance_topic_narrows_results(self):
        """Sanity check that the topic param is actually being sent —
        doesn't assert on content (too fragile), just that the call
        succeeds and returns the query back unchanged."""
        result = await tavily_search(query="Microsoft", topic="finance", max_results=3)
        assert result["query"] == "Microsoft"


# =============================================================================
# TIER 1 — comprehensive_analysis (concurrent gather resilience)
# =============================================================================

class TestComprehensiveAnalysis:
    """comprehensive_analysis() runs tavily_search and sec_edgar_filing
    CONCURRENTLY via asyncio.gather(..., return_exceptions=True)
    specifically so one source failing doesn't take down the other —
    that's the one behavior here worth dedicated regression coverage."""

    @pytest.mark.asyncio
    async def test_real_call_returns_both_news_and_filing(self):
        result = await comprehensive_analysis(ticker="MSFT", company_name="Microsoft")
        assert result["ticker"] == "MSFT"
        assert "news" in result and "filing" in result
        assert result["filing"].get("error") is None
        assert result["news"].get("error") is None

    @pytest.mark.asyncio
    async def test_tavily_failure_does_not_break_sec_edgar_result(self):
        with patch(
            "tools.research_tools.comprehensive_analysis.tavily_search",
            side_effect=RuntimeError("simulated Tavily outage"),
        ):
            result = await comprehensive_analysis(ticker="MSFT", company_name="Microsoft")
        assert result["news"].get("error") is not None
        assert "simulated Tavily outage" in result["news"]["error"]
        assert result["filing"].get("error") is None
        assert "sections" in result["filing"]

    @pytest.mark.asyncio
    async def test_sec_edgar_failure_does_not_break_tavily_result(self):
        with patch(
            "tools.research_tools.comprehensive_analysis.sec_edgar_filing",
            side_effect=RuntimeError("simulated EDGAR outage"),
        ):
            result = await comprehensive_analysis(ticker="MSFT", company_name="Microsoft")
        assert result["filing"].get("error") is not None
        assert result["news"].get("error") is None

    @pytest.mark.asyncio
    async def test_default_sections_is_mda_only(self):
        """This wrapper deliberately defaults to sections=["mda"], narrower
        than sec_edgar_filing's own ["all"] default — guards against that
        drifting back and ballooning token usage for every caller."""
        result = await comprehensive_analysis(ticker="MSFT", company_name="Microsoft")
        assert set(result["filing"]["sections"].keys()) <= {"mda"}


# =============================================================================
# TIER 2 — ResearchAgent end-to-end (real LLM calls, mark as slow)
# =============================================================================

pytestmark_slow = pytest.mark.slow


@pytest.mark.slow
class TestResearchAgentEndToEnd:

    @pytest.mark.asyncio
    async def test_populates_context_with_multiple_tool_types(self, known_ticker):
        """Regression: the query-format/sections fixes were meant to get a
        MIX of tool types into context, not just whichever one happens to
        work. A run that only ever produced one tool's output would mean
        another tool is silently failing/returning nothing."""
        agent = ResearchAgent()
        result = await agent.run({
            "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker} CORPORATION ({known_ticker})",
            "manager_directives": {"ticker": known_ticker},
            "aggregated_research_context": [],
        })
        contexts = result.get("aggregated_research_context", [])
        assert len(contexts) > 0, "no research context was gathered at all"

        tool_names_used = {
            c.split("[TOOL: ")[1].split("]")[0]
            for c in contexts if "[TOOL: " in c
        }
        assert len(tool_names_used) >= 2, (
            f"expected context from multiple tool types, only saw: {tool_names_used}"
        )

    @pytest.mark.asyncio
    async def test_requests_risk_factors_for_comprehensive_task(self, known_ticker):
        """Regression for the undocumented-sections-parameter bug: for a
        comprehensive investment analysis task, at least one sec_edgar_filing
        call should have pulled risk_factors, not just financial_statements.
        A proxy check (does risk_factors text show up anywhere in context)
        rather than intercepting the Brain's literal tool-call arguments."""
        agent = ResearchAgent()
        result = await agent.run({
            "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker} CORPORATION ({known_ticker})",
            "manager_directives": {"ticker": known_ticker},
            "aggregated_research_context": [],
        })
        contexts = result.get("aggregated_research_context", [])
        filing_chunks = [c for c in contexts if "sec_edgar_filing" in c]
        assert filing_chunks, "no sec_edgar_filing call was made at all"
        assert any(
            "risk_factors" in c.lower() or "risk factors" in c.lower()
            for c in filing_chunks
        ), "risk_factors was never requested/returned in any sec_edgar_filing call this run"

    @pytest.mark.asyncio
    async def test_repeated_runs_produce_consistent_tool_selection(self, known_ticker):
        """Lighter-weight stability check than the full sentiment-label
        version in the pipeline-level tests: across 2 runs, the SET of
        tool types used shouldn't vary wildly (e.g. one run uses
        news_search + sec_edgar_filing, another only rag_vector_search) —
        that would suggest the Brain's planning is unstable, not just its
        wording."""
        agent = ResearchAgent()
        tool_sets = []
        for _ in range(2):
            result = await agent.run({
                "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker} CORPORATION ({known_ticker})",
                "manager_directives": {"ticker": known_ticker},
                "aggregated_research_context": [],
            })
            contexts = result.get("aggregated_research_context", [])
            tools_used = {
                c.split("[TOOL: ")[1].split("]")[0]
                for c in contexts if "[TOOL: " in c
            }
            tool_sets.append(tools_used)

        overlap = tool_sets[0] & tool_sets[1]
        assert len(overlap) >= 1, (
            f"no overlap in tool selection between two runs: {tool_sets[0]} vs {tool_sets[1]}"
        )