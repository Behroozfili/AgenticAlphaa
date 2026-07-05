"""
Tests for: tools/research_tools/tavily_search.py
Phase: 2b — Research Tools

Mocking strategy: httpx.AsyncClient.post is mocked. The TAVILY_API_KEY env
var is mocked via monkeypatch.setenv since the function reads it eagerly
with os.environ["TAVILY_API_KEY"] (raises KeyError if unset — tested below).
"""
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from tools.research_tools.tavily_search import tavily_search


def make_async_client_mock(post_return):
    client = AsyncMock()
    client.post.return_value = post_return
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


class TestTavilySearch:
    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_happy_path_returns_structured_results(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {
            "answer": "NVDA beat estimates.",
            "results": [
                {"title": "NVDA Q3 Earnings", "url": "https://x.com/a",
                 "content": "NVDA beat...", "score": 0.987654, "published_date": "2024-11-01"},
            ],
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await tavily_search("NVDA earnings")

        assert result["query"] == "NVDA earnings"
        assert result["answer"] == "NVDA beat estimates."
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.9877  # rounded to 4 dp

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_missing_api_key_raises_keyerror(self, mock_client_cls, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(KeyError):
            await tavily_search("anything")
        mock_client_cls.assert_not_called()

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_include_domains_added_to_payload_when_provided(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"answer": None, "results": []}
        resp.raise_for_status.return_value = None
        client_mock = make_async_client_mock(resp)
        mock_client_cls.return_value = client_mock

        await tavily_search("query", include_domains=["reuters.com"])

        sent_payload = client_mock.__aenter__.return_value.post.call_args.kwargs["json"]
        assert sent_payload["include_domains"] == ["reuters.com"]

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_no_results_returns_empty_list(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"answer": None, "results": []}
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await tavily_search("nonexistent topic xyz")
        assert result["results"] == []
        assert result["answer"] is None

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_http_error_propagates(self, mock_client_cls, monkeypatch):
        import httpx
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad request", request=MagicMock(), response=MagicMock()
        )
        mock_client_cls.return_value = make_async_client_mock(resp)

        with pytest.raises(httpx.HTTPStatusError):
            await tavily_search("query")

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_missing_score_defaults_to_zero(self, mock_client_cls, monkeypatch):
        """If Tavily omits 'score' on a result, round(r.get('score', 0.0), 4) should not crash."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {
            "answer": None,
            "results": [{"title": "x", "url": "y", "content": "z"}],  # no "score" key
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await tavily_search("query")
        assert result["results"][0]["score"] == 0.0