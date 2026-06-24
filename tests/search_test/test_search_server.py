"""
Tests for: tools/research_tools/research_server.py
Phase: 2b — Research Tools (MCP server dispatch layer)

Scope decision: tavily_search/news_search/sec_edgar_search/sec_edgar_filing
are already fully tested in their own test files. This file only tests the
`call_tool()` async dispatcher: correct routing by `name`, correct argument
mapping (including default values), the "unknown tool" error path, and the
Sentry-capture-on-exception error path.
"""
import json
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from tools.research_tools.research_server import call_tool


# ---------------------------------------------------------------------------
# Routing — each tool name dispatches to the correct underlying function
# ---------------------------------------------------------------------------

class TestCallToolRouting:
    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.tavily_search", new_callable=AsyncMock)
    async def test_routes_to_tavily_search_with_defaults(self, mock_tavily):
        mock_tavily.return_value = {"query": "x", "results": []}

        result = await call_tool("tavily_search", {"query": "NVDA news"})

        mock_tavily.assert_called_once_with(
            query="NVDA news", max_results=5, search_depth="basic",
            include_domains=None, topic="finance",
        )
        assert result.isError is False

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.news_search", new_callable=AsyncMock)
    async def test_routes_to_news_search_with_overrides(self, mock_news):
        mock_news.return_value = {"query": "x", "articles": []}

        await call_tool("news_search", {
            "query": "NVDA", "from_date": "2024-01-01", "page_size": 20,
        })

        mock_news.assert_called_once_with(
            query="NVDA", from_date="2024-01-01", to_date=None,
            language="en", sort_by="publishedAt", page_size=20,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.sec_edgar_search", new_callable=AsyncMock)
    async def test_routes_to_sec_edgar_search(self, mock_search):
        mock_search.return_value = {"query": "x", "filings": []}

        await call_tool("sec_edgar_search", {"query": "10-K", "ticker": "NVDA"})

        mock_search.assert_called_once_with(
            query="10-K", ticker="NVDA", form_type=None, max_results=5,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.sec_edgar_filing", new_callable=AsyncMock)
    async def test_routes_to_sec_edgar_filing_with_default_sections(self, mock_filing):
        mock_filing.return_value = {"ticker": "NVDA", "sections": {}}

        await call_tool("sec_edgar_filing", {"ticker": "NVDA"})

        mock_filing.assert_called_once_with(
            ticker="NVDA", form_type="10-K", sections=["all"], max_chars=8000,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.rag_vector_search", new_callable=AsyncMock)
    async def test_routes_to_rag_vector_search(self, mock_rag):
        mock_rag.return_value = {"results": []}

        await call_tool("rag_vector_search", {"query": "earnings"})

        mock_rag.assert_called_once_with(
            query="earnings", top_k=5, ticker_filter=None, threshold=0.01,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.rag_hybrid_query", new_callable=AsyncMock)
    async def test_routes_to_rag_hybrid_query_requires_entity(self, mock_rag):
        mock_rag.return_value = {"results": []}

        await call_tool("rag_hybrid_query", {"query": "earnings", "entity": "NVDA"})

        mock_rag.assert_called_once_with(
            query="earnings", entity="NVDA", top_k=5, max_hops=2, fusion="rrf",
        )


# ---------------------------------------------------------------------------
# Unknown tool name
# ---------------------------------------------------------------------------

class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self):
        result = await call_tool("not_a_real_tool", {})

        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert "Unknown tool" in payload["error"]


# ---------------------------------------------------------------------------
# Required argument missing -> KeyError -> caught and wrapped as error result
# ---------------------------------------------------------------------------

class TestMissingRequiredArgument:
    @pytest.mark.asyncio
    async def test_missing_query_returns_error_result_not_raises(self):
        # tavily_search requires "query"; omitting it should raise KeyError
        # internally, caught by call_tool's except block (not propagated).
        result = await call_tool("tavily_search", {})

        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert payload["tool"] == "tavily_search"
        assert "error" in payload


# ---------------------------------------------------------------------------
# Sentry capture on exception
# ---------------------------------------------------------------------------

class TestSentryCaptureOnException:
    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.tavily_search", new_callable=AsyncMock)
    @patch("tools.research_tools.research_server.sentry_enabled", return_value=True)
    async def test_sentry_captures_exception_with_tool_and_server_tags(
        self, mock_enabled, mock_tavily
    ):
        mock_tavily.side_effect = RuntimeError("api down")

        with patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc:
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope

            result = await call_tool("tavily_search", {"query": "x"})

            scope.set_tag.assert_any_call("tool", "tavily_search")
            scope.set_tag.assert_any_call("server", "research-agent-mcp")
            mock_capture_exc.assert_called_once()
            assert result.isError is True

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.tavily_search", new_callable=AsyncMock)
    @patch("tools.research_tools.research_server.sentry_enabled", return_value=False)
    async def test_sentry_not_invoked_when_disabled(self, mock_enabled, mock_tavily):
        mock_tavily.side_effect = RuntimeError("api down")

        with patch("sentry_sdk.capture_exception") as mock_capture_exc:
            result = await call_tool("tavily_search", {"query": "x"})
            mock_capture_exc.assert_not_called()
            assert result.isError is True