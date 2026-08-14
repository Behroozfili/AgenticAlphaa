"""
Tests for: tools/research_tools/news_search.py
Phase: 2b — Research Tools

Mocking strategy: httpx.AsyncClient.get is mocked. NEWSAPI_KEY env var
required by os.environ["NEWSAPI_KEY"] is set/unset via monkeypatch.
We also freeze datetime.utcnow() indirectly by checking from_date is a
valid date string rather than asserting an exact value (avoids flaky
tests around the 30-day default window's exact day boundary).
"""
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from tools.research_tools.news_search import news_search


def make_async_client_mock(get_return):
    client = AsyncMock()
    client.get.return_value = get_return
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


class TestNewsSearch:
    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_happy_path_returns_articles(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {
            "totalResults": 1,
            "articles": [{
                "title": "NVDA surges", "description": "desc", "url": "https://x.com",
                "source": {"name": "Reuters"}, "author": "Jane Doe",
                "publishedAt": "2024-11-01T12:00:00Z",
            }],
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await news_search("NVDA")

        assert result["query"] == "NVDA"
        assert result["total_results"] == 1
        assert result["articles"][0]["source"] == "Reuters"

    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_removed_articles_are_filtered_out(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {
            "totalResults": 2,
            "articles": [
                {"title": "[Removed]", "url": "x", "source": {"name": "y"}, "publishedAt": "z"},
                {"title": "Real Article", "url": "x2", "source": {"name": "y2"}, "publishedAt": "z2"},
            ],
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await news_search("query")
        assert len(result["articles"]) == 1
        assert result["articles"][0]["title"] == "Real Article"

    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_missing_api_key_raises_keyerror(self, mock_client_cls, monkeypatch):
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        with pytest.raises(KeyError):
            await news_search("query")
        mock_client_cls.assert_not_called()

    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_default_from_date_is_30_days_back(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"totalResults": 0, "articles": []}
        resp.raise_for_status.return_value = None
        client_mock = make_async_client_mock(resp)
        mock_client_cls.return_value = client_mock

        await news_search("query")

        sent_params = client_mock.__aenter__.return_value.get.call_args.kwargs["params"]
        expected_from = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        assert sent_params["from"] == expected_from

    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_explicit_from_date_is_respected(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"totalResults": 0, "articles": []}
        resp.raise_for_status.return_value = None
        client_mock = make_async_client_mock(resp)
        mock_client_cls.return_value = client_mock

        await news_search("query", from_date="2024-01-01")

        sent_params = client_mock.__aenter__.return_value.get.call_args.kwargs["params"]
        assert sent_params["from"] == "2024-01-01"

    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_page_size_capped_at_100(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"totalResults": 0, "articles": []}
        resp.raise_for_status.return_value = None
        client_mock = make_async_client_mock(resp)
        mock_client_cls.return_value = client_mock

        await news_search("query", page_size=500)

        sent_params = client_mock.__aenter__.return_value.get.call_args.kwargs["params"]
        assert sent_params["pageSize"] == 100

    @pytest.mark.asyncio
    @patch("tools.research_tools.news_search.httpx.AsyncClient")
    async def test_to_date_only_added_when_provided(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"totalResults": 0, "articles": []}
        resp.raise_for_status.return_value = None
        client_mock = make_async_client_mock(resp)
        mock_client_cls.return_value = client_mock

        await news_search("query")  # no to_date

        sent_params = client_mock.__aenter__.return_value.get.call_args.kwargs["params"]
        assert "to" not in sent_params