"""
Tests for: tools/research_tools/tavily_search.py
Phase: 2b — Research Tools

Mocking strategy: httpx.AsyncClient.post is mocked. The TAVILY_API_KEY env
var is mocked via monkeypatch.setenv since the function reads it eagerly
with os.environ["TAVILY_API_KEY"] (raises KeyError if unset — tested below).
"""
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from tools.research_tools.tavily_search import tavily_search as _tavily_search
from core.forecast_contract import ForecastContext
from tools.research_tools import deterministic_evidence_id

LIVE_CONTEXT = ForecastContext(
    ticker="NVDA", as_of="2024-11-29T18:00:00Z", lookback_days=30,
    schema_version="v1", dataset_version="dataset-v1",
    feature_version="feature-v1", model_version="model-v1",
)


async def tavily_search(*args, **kwargs):
    kwargs.setdefault("forecast_context", LIVE_CONTEXT)
    kwargs.setdefault("run_id", "run-tavily-test")
    return await _tavily_search(*args, **kwargs)


def make_async_client_mock(post_return):
    client = AsyncMock()
    client.post.return_value = post_return
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


class TestTavilySearch:
    @pytest.mark.asyncio
    async def test_historical_without_archive_fails_closed_without_network(self, monkeypatch):
        historical = ForecastContext.from_dict(
            {**LIVE_CONTEXT.to_dict(), "analysis_mode": "HISTORICAL_BACKTEST"}
        )
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = await _tavily_search(
            "NVDA history", forecast_context=historical, run_id="run-historical"
        )
        assert result["status"] == "DATA_UNAVAILABLE"
        assert result["evidence"] == []

    @pytest.mark.asyncio
    async def test_historical_archive_rejects_future_and_current_web(self):
        historical = ForecastContext.from_dict(
            {**LIVE_CONTEXT.to_dict(), "analysis_mode": "HISTORICAL_BACKTEST"}
        )
        eligible = {
            "provider": "gdelt_archive", "ticker": "NVDA",
            "dataset_version": "dataset-v1", "published_at": "2024-11-01T12:00:00Z",
            "url": "https://archive.test/eligible", "title": "eligible",
        }
        future = {**eligible, "url": "https://archive.test/future", "published_at": "2024-12-01T12:00:00Z"}
        current = {**eligible, "url": "https://web.test/current", "provider": "tavily_live"}
        result = await _tavily_search(
            "NVDA history", forecast_context=historical, run_id="run-historical",
            archived_records=[eligible, future, current],
        )
        assert result["status"] == "OK"
        assert len(result["evidence"]) == 1
        assert len(result["quarantined"]) == 2

    def test_evidence_id_is_deterministic_and_version_sensitive(self):
        record = {"url": "https://archive.test/a", "published_at": "2024-11-01T12:00:00Z", "title": "a"}
        first = deterministic_evidence_id(record, context=LIVE_CONTEXT, source="gdelt_archive")
        second = deterministic_evidence_id(dict(record), context=LIVE_CONTEXT, source="gdelt_archive")
        other_context = ForecastContext.from_dict(
            {**LIVE_CONTEXT.to_dict(), "dataset_version": "dataset-v2"}
        )
        assert first == second
        assert first != deterministic_evidence_id(record, context=other_context, source="gdelt_archive")
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
        assert result["answer"] is None
        assert result["answer_quarantined"] is True
        assert len(result["results"]) == 1
        assert result["results"][0]["score"] == 0.9877  # rounded to 4 dp

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_request_is_bounded_to_forecast_context_window(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {"answer": None, "results": []}
        resp.raise_for_status.return_value = None
        client_mock = make_async_client_mock(resp)
        mock_client_cls.return_value = client_mock

        result = await tavily_search(
            "NVDA", start_date="2020-01-01", end_date="2030-01-01"
        )

        payload = client_mock.__aenter__.return_value.post.call_args.kwargs["json"]
        assert payload["start_date"] == "2024-10-30"
        assert payload["end_date"] == "2024-11-29"
        assert result["date_range"] == {
            "start": "2024-10-30", "end": "2024-11-29"
        }

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_rfc_news_timestamp_is_normalized_before_temporal_gate(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {
            "answer": None,
            "results": [{
                "title": "NVDA news", "url": "https://x.com/rfc",
                "content": "Monthly evidence", "score": 0.9,
                "published_date": "Fri, 01 Nov 2024 12:00:00 GMT",
            }],
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await tavily_search("NVDA")

        assert result["status"] == "OK"
        assert len(result["evidence"]) == 1
        assert result["evidence"][0]["timestamp"] == "2024-11-01T12:00:00Z"

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_missing_api_key_fails_closed(self, mock_client_cls, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = await tavily_search("anything")
        assert result["error_code"] == "RESEARCH_PROVIDER_FAIL_CLOSED"
        assert result["telemetry"]["outcome"] == "FAIL_CLOSED"
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
    async def test_http_error_fails_closed(self, mock_client_cls, monkeypatch):
        import httpx
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "bad request", request=MagicMock(), response=MagicMock()
        )
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await tavily_search("query")
        assert result["error_code"] == "RESEARCH_PROVIDER_FAIL_CLOSED"
        assert result["telemetry"]["outcome"] == "FAIL_CLOSED"

    @pytest.mark.asyncio
    @patch("tools.research_tools.tavily_search.httpx.AsyncClient")
    async def test_missing_score_defaults_to_zero(self, mock_client_cls, monkeypatch):
        """If Tavily omits 'score' on a result, round(r.get('score', 0.0), 4) should not crash."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        resp = MagicMock()
        resp.json.return_value = {
            "answer": None,
            "results": [{"title": "x", "url": "y", "content": "z", "published_date": "2024-11-01"}],  # no "score" key
        }
        resp.raise_for_status.return_value = None
        mock_client_cls.return_value = make_async_client_mock(resp)

        result = await tavily_search("query")
        assert result["results"][0]["score"] == 0.0
