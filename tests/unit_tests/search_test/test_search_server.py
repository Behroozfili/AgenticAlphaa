"""Offline unit coverage for the Research MCP dispatcher."""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.forecast_contract import ForecastContext
from tools.research_tools.research_server import call_tool, list_tools


FORECAST_CONTEXT = ForecastContext(
    ticker="NVDA",
    as_of="2024-01-31T21:00:00Z",
    schema_version="v1",
    dataset_version="dataset-v1",
    feature_version="feature-v1",
    model_version="model-v1",
)
RUN_ID = "run-search-server-test"


def valid_arguments(**overrides):
    """Return dispatcher arguments with the mandatory immutable identity."""
    return {
        "forecast_context": FORECAST_CONTEXT.to_dict(),
        "run_id": RUN_ID,
        **overrides,
    }


class TestCallToolRouting:
    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.tavily_search", new_callable=AsyncMock)
    async def test_routes_to_tavily_search_with_defaults(self, mock_tavily):
        mock_tavily.return_value = {"query": "x", "results": []}

        result = await call_tool(
            "tavily_search", valid_arguments(query="NVDA news")
        )

        mock_tavily.assert_awaited_once_with(
            query="NVDA news",
            max_results=3,
            search_depth="basic",
            include_domains=None,
            topic="news",
            snippet_chars=500,
            start_date=None,
            end_date=None,
            forecast_context=FORECAST_CONTEXT,
            run_id=RUN_ID,
            archived_records=None,
        )
        assert result.isError is False

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.news_search", new_callable=AsyncMock)
    async def test_routes_to_news_search_with_overrides(self, mock_news):
        mock_news.return_value = {"query": "x", "articles": []}

        await call_tool(
            "news_search",
            valid_arguments(
                query="NVDA", from_date="2024-01-01", page_size=20
            ),
        )

        mock_news.assert_awaited_once_with(
            query="NVDA",
            from_date="2024-01-01",
            to_date=None,
            language="en",
            sort_by="relevancy",
            page_size=20,
            exclude_domains=None,
            forecast_context=FORECAST_CONTEXT,
            run_id=RUN_ID,
            archived_records=None,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.sec_edgar_search", new_callable=AsyncMock)
    async def test_routes_to_sec_edgar_search(self, mock_search):
        mock_search.return_value = {"query": "x", "filings": []}

        await call_tool(
            "sec_edgar_search",
            valid_arguments(query="10-K", ticker="NVDA"),
        )

        mock_search.assert_awaited_once_with(
            query="10-K",
            ticker="NVDA",
            form_type=None,
            max_results=5,
            forecast_context=FORECAST_CONTEXT,
            run_id=RUN_ID,
            archived_records=None,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.sec_edgar_filing", new_callable=AsyncMock)
    async def test_routes_to_sec_edgar_filing_with_defaults(self, mock_filing):
        mock_filing.return_value = {"ticker": "NVDA", "sections": {}}

        await call_tool(
            "sec_edgar_filing", valid_arguments(ticker="NVDA")
        )

        mock_filing.assert_awaited_once_with(
            ticker="NVDA",
            form_type="10-K",
            sections=[],
            mode="preview",
            preview_chars=4000,
            query=None,
            query_window=3000,
            offset=0,
            length=None,
            max_chars=25000,
            accession_number=None,
            forecast_context=FORECAST_CONTEXT,
            run_id=RUN_ID,
            archived_filing=None,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.sec_edgar_filing", new_callable=AsyncMock)
    async def test_routes_accession_number_without_forcing_default_form(self, mock_filing):
        mock_filing.return_value = {"ticker": "NVDA", "sections": {}}

        await call_tool(
            "sec_edgar_filing",
            valid_arguments(
                ticker="NVDA",
                accession_number="0001045810-25-000023",
                sections=["risk_factors"],
            ),
        )

        assert mock_filing.await_args.kwargs["accession_number"] == "0001045810-25-000023"
        assert mock_filing.await_args.kwargs["form_type"] is None
        assert mock_filing.await_args.kwargs["forecast_context"] == FORECAST_CONTEXT
        assert mock_filing.await_args.kwargs["run_id"] == RUN_ID

    @pytest.mark.asyncio
    async def test_every_schema_requires_execution_identity(self):
        result = await list_tools()

        assert result.tools
        for tool in result.tools:
            assert {"forecast_context", "run_id"} <= set(tool.inputSchema["required"])

        filing = next(item for item in result.tools if item.name == "sec_edgar_filing")
        assert "accession_number" in filing.inputSchema["properties"]
        assert "sec_edgar_search" in filing.description
        assert "exact filing" in filing.description

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.rag_vector_search", new_callable=AsyncMock)
    async def test_routes_to_rag_vector_search(self, mock_rag):
        mock_rag.return_value = {"results": []}

        await call_tool(
            "rag_vector_search", valid_arguments(query="earnings")
        )

        mock_rag.assert_awaited_once_with(
            query="earnings",
            top_k=5,
            ticker_filter=None,
            threshold=0.01,
            days_back=FORECAST_CONTEXT.lookback_days,
            context=FORECAST_CONTEXT,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.rag_graph_traverse", new_callable=AsyncMock)
    async def test_routes_to_rag_graph_traverse(self, mock_rag):
        mock_rag.return_value = {"results": []}

        await call_tool(
            "rag_graph_traverse", valid_arguments(entity="NVDA")
        )

        mock_rag.assert_awaited_once_with(
            entity="NVDA",
            relation_types=["ALL"],
            max_hops=2,
            limit=20,
            context=FORECAST_CONTEXT,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.rag_hybrid_query", new_callable=AsyncMock)
    async def test_routes_to_rag_hybrid_query(self, mock_rag):
        mock_rag.return_value = {"results": []}

        await call_tool(
            "rag_hybrid_query",
            valid_arguments(query="earnings", entity="NVDA"),
        )

        mock_rag.assert_awaited_once_with(
            query="earnings",
            entity="NVDA",
            top_k=5,
            max_hops=2,
            fusion="rrf",
            days_back=FORECAST_CONTEXT.lookback_days,
            context=FORECAST_CONTEXT,
        )

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.comprehensive_analysis", new_callable=AsyncMock)
    async def test_routes_to_comprehensive_analysis(self, mock_analysis):
        mock_analysis.return_value = {"ticker": "NVDA", "news": {}, "filing": {}}

        await call_tool(
            "comprehensive_analysis",
            valid_arguments(ticker="NVDA", sections=["mda"]),
        )

        mock_analysis.assert_awaited_once_with(
            ticker="NVDA",
            company_name=None,
            topic_query=None,
            form_type="10-Q",
            sections=["mda"],
            max_results=3,
            filing_mode="preview",
            filing_query=None,
            forecast_context=FORECAST_CONTEXT,
            run_id=RUN_ID,
            archived_web_records=None,
            archived_filing=None,
        )


class TestFailClosedErrors:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_stable_private_error(self):
        result = await call_tool("not_a_real_tool", valid_arguments(secret="do-not-return"))

        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert payload == {
            "status": "FAIL_CLOSED",
            "error_code": "RESEARCH_TOOL_FAIL_CLOSED",
            "tool": "unknown",
            "missing": True,
            "degraded": True,
        }
        assert "do-not-return" not in result.content[0].text

    @pytest.mark.asyncio
    async def test_missing_query_returns_stable_error_not_exception_text(self):
        result = await call_tool("tavily_search", valid_arguments())

        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert payload["tool"] == "tavily_search"
        assert payload["error_code"] == "RESEARCH_TOOL_FAIL_CLOSED"
        assert "query" not in payload
        assert "exception" not in payload

    @pytest.mark.asyncio
    async def test_missing_identity_fails_closed_before_dispatch(self):
        with patch(
            "tools.research_tools.research_server.tavily_search",
            new_callable=AsyncMock,
        ) as mock_tavily:
            result = await call_tool("tavily_search", {"query": "private query"})

        mock_tavily.assert_not_awaited()
        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert payload["error_code"] == "RESEARCH_TOOL_FAIL_CLOSED"
        assert "private query" not in result.content[0].text


class TestPrivacySafeSentry:
    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.tavily_search", new_callable=AsyncMock)
    @patch("tools.research_tools.research_server.sentry_enabled", return_value=True)
    async def test_sentry_captures_stable_message_without_exception(
        self, mock_enabled, mock_tavily, caplog
    ):
        mock_tavily.side_effect = RuntimeError("private provider failure")
        fake_sentry = MagicMock()

        with patch.dict(sys.modules, {"sentry_sdk": fake_sentry}):
            result = await call_tool(
                "tavily_search", valid_arguments(query="private query")
            )

        scope = fake_sentry.push_scope.return_value.__enter__.return_value
        scope.set_tag.assert_any_call("tool", "tavily_search")
        scope.set_tag.assert_any_call("server", "research-agent-mcp")
        fake_sentry.capture_exception.assert_not_called()
        fake_sentry.capture_message.assert_called_once_with(
            "RESEARCH_TOOL_FAIL_CLOSED", level="error"
        )
        rendered = result.content[0].text + caplog.text
        assert "private provider failure" not in rendered
        assert "private query" not in rendered
        assert result.isError is True

    @pytest.mark.asyncio
    @patch("tools.research_tools.research_server.tavily_search", new_callable=AsyncMock)
    @patch("tools.research_tools.research_server.sentry_enabled", return_value=False)
    async def test_sentry_not_imported_when_disabled(self, mock_enabled, mock_tavily):
        mock_tavily.side_effect = RuntimeError("private provider failure")
        fake_sentry = MagicMock()

        with patch.dict(sys.modules, {"sentry_sdk": fake_sentry}):
            result = await call_tool(
                "tavily_search", valid_arguments(query="private query")
            )

        fake_sentry.capture_exception.assert_not_called()
        fake_sentry.capture_message.assert_not_called()
        assert result.isError is True
