"""
Tests for: api/routes/analyze.py
Phase: 7 — API Layer (Integration)

Mocking strategy: a minimal standalone FastAPI app is built in this test
file (NOT api.main.app) with the analyze_router mounted, app.state.supabase
and app.state.manager_agent set to MagicMocks, and get_manager_memory
overridden via app.dependency_overrides (the exact pattern the file's own
docstring recommends). This isolates the route's logic from main.py's
startup/lifespan concerns entirely.
"""
from unittest.mock import MagicMock, AsyncMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.analyze import router, AnalyzeRequest, _persist_analysis
from api.dependencies import get_manager_memory
from api.core.exceptions import AgentError


def make_app(manager_agent=None, supabase=None, memory_override=None):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.supabase = supabase or MagicMock()
    app.state.manager_agent = manager_agent or AsyncMock()

    if memory_override is not None:
        mock_memory = memory_override
    else:
        mock_memory = MagicMock()
        mock_memory.long.get_all_preferences.return_value = {}

    app.dependency_overrides[get_manager_memory] = lambda: mock_memory

    return app, mock_memory


def make_client(**kwargs):
    app, memory = make_app(**kwargs)
    return TestClient(app, raise_server_exceptions=False), app, memory


VALID_PAYLOAD = {"query": "Is NVIDIA a good buy for Q1 2025?", "ticker": "nvda"}


# ---------------------------------------------------------------------------
# AnalyzeRequest — Pydantic validation
# ---------------------------------------------------------------------------

class TestAnalyzeRequestValidation:
    def test_ticker_uppercased(self):
        req = AnalyzeRequest(query="x" * 15, ticker="nvda")
        assert req.ticker == "NVDA"

    def test_ticker_none_allowed(self):
        req = AnalyzeRequest(query="x" * 15, ticker=None)
        assert req.ticker is None

    def test_ticker_with_digits_rejected(self):
        with pytest.raises(ValueError):
            AnalyzeRequest(query="x" * 15, ticker="NV3A")

    def test_ticker_too_long_rejected(self):
        with pytest.raises(ValueError):
            AnalyzeRequest(query="x" * 15, ticker="TOOLONG")

    def test_query_too_short_rejected(self):
        with pytest.raises(ValueError):
            AnalyzeRequest(query="short")

    def test_search_depth_must_be_basic_or_advanced(self):
        with pytest.raises(ValueError):
            AnalyzeRequest(query="x" * 15, search_depth="ultra")

    def test_days_back_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            AnalyzeRequest(query="x" * 15, days_back=400)

    def test_defaults_applied(self):
        req = AnalyzeRequest(query="x" * 15)
        assert req.user_id == "anonymous"
        assert req.search_depth == "advanced"
        assert req.days_back == 14
        assert req.include_sentiment is True


# ---------------------------------------------------------------------------
# POST /analyze — happy path
# ---------------------------------------------------------------------------

class TestAnalyzeEndpointHappyPath:
    def test_returns_200_with_structured_response(self):
        manager_agent = AsyncMock()
        manager_agent.run.return_value = {
            "final_report": "Buy recommendation.",
            "financial_metrics_summary": {"pe": 20},
            "sentiment_analysis_summary": {"label": "Bullish"},
            "aggregated_research_context": ["a", "b", "c"],
            "agent_execution_history": [],
            "orchestrator_logs": [],
        }
        client, app, memory = make_client(manager_agent=manager_agent)

        resp = client.post("/api/v1/analyze", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["final_report"] == "Buy recommendation."
        assert body["research_context_chunks"] == 3
        assert body["financial_metrics"] == {"pe": 20}

    def test_manager_directives_built_from_request_fields(self):
        manager_agent = AsyncMock()
        manager_agent.run.return_value = {}
        client, app, memory = make_client(manager_agent=manager_agent)

        client.post("/api/v1/analyze", json={
            "query": "Is NVIDIA a good buy?", "ticker": "amd",
            "search_depth": "basic", "days_back": 30, "include_sentiment": False,
        })

        _, kwargs = manager_agent.run.call_args
        directives = kwargs["manager_directives"]
        assert directives["ticker"] == "AMD"
        assert directives["search_depth"] == "basic"
        assert directives["days_back"] == 30
        assert directives["include_sentiment"] is False

    def test_user_preferences_pulled_from_injected_memory(self):
        manager_agent = AsyncMock()
        manager_agent.run.return_value = {}
        memory = MagicMock()
        memory.long.get_all_preferences.return_value = {"fmt": "concise"}
        client, app, _ = make_client(manager_agent=manager_agent, memory_override=memory)

        client.post("/api/v1/analyze", json=VALID_PAYLOAD)

        _, kwargs = manager_agent.run.call_args
        assert kwargs["user_preferences"] == {"fmt": "concise"}

    def test_missing_result_fields_default_gracefully(self):
        """ManagerAgent.run() returning a sparse dict should not crash response building."""
        manager_agent = AsyncMock()
        manager_agent.run.return_value = {}  # nothing at all
        client, app, _ = make_client(manager_agent=manager_agent)

        resp = client.post("/api/v1/analyze", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert body["final_report"] == ""
        assert body["research_context_chunks"] == 0
        assert body["financial_metrics"] == {}


# ---------------------------------------------------------------------------
# POST /analyze — failure path (ManagerAgent.run raises)
# ---------------------------------------------------------------------------

class TestAnalyzeEndpointFailurePath:
    def test_manager_agent_exception_returns_agent_error_response(self):
        manager_agent = AsyncMock()
        manager_agent.run.side_effect = RuntimeError("graph crashed")
        client, app, _ = make_client(manager_agent=manager_agent)

        resp = client.post("/api/v1/analyze", json=VALID_PAYLOAD)

        # AgentError raised by the route -> no exception_handler registered in
        # this minimal test app, so FastAPI's default 500 applies UNLESS we
        # register one. We verify the route itself raises AgentError correctly
        # by checking the response is a server error and the underlying cause.
        assert resp.status_code == 500

    def test_agent_error_includes_original_exception_message_as_detail(self):
        from api.main import alpha_agent_exception_handler
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.add_exception_handler(AgentError, alpha_agent_exception_handler)
        manager_agent = AsyncMock()
        manager_agent.run.side_effect = RuntimeError("graph crashed")
        app.state.supabase = MagicMock()
        app.state.manager_agent = manager_agent
        app.state.sentry_ok = False
        memory = MagicMock()
        memory.long.get_all_preferences.return_value = {}
        app.dependency_overrides[get_manager_memory] = lambda: memory

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/analyze", json=VALID_PAYLOAD)

        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "AGENT_ERROR"
        assert "graph crashed" in body["detail"]

    def test_persist_still_attempted_after_failure(self):
        manager_agent = AsyncMock()
        manager_agent.run.side_effect = RuntimeError("crashed")
        mock_supabase = MagicMock()
        client, app, _ = make_client(manager_agent=manager_agent, supabase=mock_supabase)

        client.post("/api/v1/analyze", json=VALID_PAYLOAD)

        mock_supabase.table.assert_called_with("analyses")
        insert_payload = mock_supabase.table.return_value.insert.call_args[0][0]
        assert insert_payload["status"] == "failed"
        assert insert_payload["error_message"] == "crashed"


# ---------------------------------------------------------------------------
# _persist_analysis — Supabase write + fire-and-forget error handling
# ---------------------------------------------------------------------------

class TestPersistAnalysis:
    @pytest.mark.asyncio
    async def test_writes_expected_row_shape(self):
        request = MagicMock()
        mock_db = MagicMock()
        request.app.state.supabase = mock_db
        req = AnalyzeRequest(query="x" * 15, ticker="NVDA")

        await _persist_analysis(
            request=request, analysis_id="id-1", user_id="u1", req=req,
            result={"final_report": "report text", "financial_metrics_summary": {"pe": 1}},
            status="completed", error_message=None,
            created_at="t0", completed_at="t1", duration_s=2.5,
        )

        mock_db.table.assert_called_once_with("analyses")
        payload = mock_db.table.return_value.insert.call_args[0][0]
        assert payload["analysis_id"] == "id-1"
        assert payload["ticker"] == "NVDA"
        assert payload["final_report"] == "report text"
        assert payload["duration_s"] == 2.5

    @pytest.mark.asyncio
    async def test_supabase_failure_is_caught_and_does_not_raise(self):
        request = MagicMock()
        request.app.state.supabase.table.side_effect = RuntimeError("db down")
        req = AnalyzeRequest(query="x" * 15)

        await _persist_analysis(
            request=request, analysis_id="id-1", user_id="u1", req=req,
            result={}, status="completed", error_message=None,
            created_at="t0", completed_at="t1", duration_s=1.0,
        )  # must not raise — fire-and-forget pattern