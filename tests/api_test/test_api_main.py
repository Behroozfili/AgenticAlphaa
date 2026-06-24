"""
Tests for: api/main.py
Phase: 7 — API Layer (Integration)

GOOD NEWS confirmed before writing these tests: the Phase 0 bug
(`SentimentAgent()` called without `server_script_path`) has ALREADY been
fixed in this file — line 108 now correctly passes
`SentimentAgent(server_script_path=script_path)`. No bug-documenting test
is needed for that anymore.

NEW FINDING (documented, not a crash bug): the comment above `_is_prod`
claims it is "evaluated inside lifespan (not at import time)" — but the
actual code is a MODULE-LEVEL statement evaluated once at IMPORT time.
See TestIsProdEagerEvaluation below, which documents the actual (eager)
behavior. Tests that monkeypatch APP_ENV after import will NOT see
docs_url/CORS change without a fresh module reload.

Mocking strategy:
  - supabase.create_client, ResearchAgent, FinancialAnalystAgent,
    SentimentAgent, and ManagerAgent are all patched at the `api.main`
    import point for lifespan tests, so no real network/LLM calls happen.
  - sys.exit is patched to actually raise SystemExit so we can assert on
    it without killing the test process.
  - TestClient(app) is used for endpoint-level tests with lifespan
    patched to a no-op so /health and /readiness don't need real agents.
"""
import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-supabase-key")

import api.main as main_module
from api.core.exceptions import ConfigurationError, AgentError
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Module-level eager-evaluation finding
# ---------------------------------------------------------------------------

class TestIsProdEagerEvaluation:
    def test_is_prod_reflects_app_env_at_import_time_only(self, monkeypatch):
        original = main_module._is_prod
        monkeypatch.setenv("APP_ENV", "production")
        assert main_module._is_prod == original


# ---------------------------------------------------------------------------
# lifespan() — startup sequence
# ---------------------------------------------------------------------------

def make_fake_app():
    app = MagicMock()
    app.state = MagicMock()
    return app


class TestLifespanStartupSuccess:
    @pytest.mark.asyncio
    async def test_happy_path_populates_app_state(self):
        fake_app = make_fake_app()

        with patch("api.main.validate_settings"), \
             patch("api.main.init_sentry", return_value=True), \
             patch("api.main.init_langsmith", return_value=False), \
             patch("api.main.create_client") as mock_create_client, \
             patch("api.main.ResearchAgent"), \
             patch("api.main.FinancialAnalystAgent"), \
             patch("api.main.SentimentAgent") as mock_sentiment_cls, \
             patch("api.main.ManagerMemory"), \
             patch("api.main.ManagerAgent") as mock_manager_cls:

            mock_supabase = MagicMock()
            mock_create_client.return_value = mock_supabase

            async with main_module.lifespan(fake_app):
                pass

        assert fake_app.state.sentry_ok is True
        assert fake_app.state.supabase is mock_supabase
        mock_sentiment_cls.assert_called_once()
        _, kwargs = mock_sentiment_cls.call_args
        assert "server_script_path" in kwargs
        mock_manager_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_manager_agent_receives_all_three_specialist_agents(self):
        fake_app = make_fake_app()

        with patch("api.main.validate_settings"), \
             patch("api.main.init_sentry", return_value=False), \
             patch("api.main.init_langsmith", return_value=False), \
             patch("api.main.create_client"), \
             patch("api.main.ResearchAgent") as mock_research_cls, \
             patch("api.main.FinancialAnalystAgent") as mock_financial_cls, \
             patch("api.main.SentimentAgent") as mock_sentiment_cls, \
             patch("api.main.ManagerMemory"), \
             patch("api.main.ManagerAgent") as mock_manager_cls:

            research_instance = mock_research_cls.return_value
            financial_instance = mock_financial_cls.return_value
            sentiment_instance = mock_sentiment_cls.return_value

            async with main_module.lifespan(fake_app):
                pass

            _, kwargs = mock_manager_cls.call_args
            assert kwargs["research_agent"] is research_instance
            assert kwargs["financial_agent"] is financial_instance
            assert kwargs["sentiment_agent"] is sentiment_instance

    @pytest.mark.asyncio
    async def test_system_memory_uses_system_user_id(self):
        fake_app = make_fake_app()

        with patch("api.main.validate_settings"), \
             patch("api.main.init_sentry", return_value=False), \
             patch("api.main.init_langsmith", return_value=False), \
             patch("api.main.create_client"), \
             patch("api.main.ResearchAgent"), \
             patch("api.main.FinancialAnalystAgent"), \
             patch("api.main.SentimentAgent"), \
             patch("api.main.ManagerMemory") as mock_memory_cls, \
             patch("api.main.ManagerAgent"):

            async with main_module.lifespan(fake_app):
                pass

            _, kwargs = mock_memory_cls.call_args
            assert kwargs["user_id"] == "system"


class TestLifespanStartupFailure:
    @pytest.mark.asyncio
    async def test_configuration_error_exits_process(self):
        fake_app = make_fake_app()

        with patch("api.main.validate_settings", side_effect=ConfigurationError("missing key")), \
             patch("api.main.sys.exit", side_effect=SystemExit(1)) as mock_exit:

            with pytest.raises(SystemExit):
                async with main_module.lifespan(fake_app):
                    pass

            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_supabase_connection_failure_exits_process(self):
        fake_app = make_fake_app()

        with patch("api.main.validate_settings"), \
             patch("api.main.init_sentry", return_value=False), \
             patch("api.main.init_langsmith", return_value=False), \
             patch("api.main.create_client", side_effect=ConnectionError("supabase down")), \
             patch("api.main.sys.exit", side_effect=SystemExit(1)) as mock_exit:

            with pytest.raises(SystemExit):
                async with main_module.lifespan(fake_app):
                    pass

            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_supabase_failure_does_not_attempt_agent_init(self):
        fake_app = make_fake_app()

        with patch("api.main.validate_settings"), \
             patch("api.main.init_sentry", return_value=False), \
             patch("api.main.init_langsmith", return_value=False), \
             patch("api.main.create_client", side_effect=ConnectionError("down")), \
             patch("api.main.sys.exit", side_effect=SystemExit(1)), \
             patch("api.main.ResearchAgent") as mock_research_cls:

            with pytest.raises(SystemExit):
                async with main_module.lifespan(fake_app):
                    pass

            mock_research_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Health / readiness endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client_no_lifespan():
    @asynccontextmanager
    async def noop_lifespan(app):
        app.state.sentry_ok = False
        app.state.supabase = MagicMock()
        yield

    with patch.object(main_module.app.router, "lifespan_context", noop_lifespan):
        # raise_server_exceptions=False mimics a real deployed server (uvicorn):
        # it returns whatever the ASGI app/exception handlers produced instead
        # of re-raising the original exception into the test for debugging.
        # Without this, TestClient's default debug behavior makes it look like
        # the generic Exception handler "isn't being called" — it IS, but the
        # client re-raises before showing you the response.
        with TestClient(main_module.app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoint:
    def test_returns_200_ok(self, client_no_lifespan):
        resp = client_no_lifespan.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestReadinessEndpoint:
    def test_returns_ready_when_supabase_reachable(self, client_no_lifespan):
        main_module.app.state.supabase.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock()
        resp = client_no_lifespan.get("/readiness")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_returns_503_when_supabase_unreachable(self, client_no_lifespan):
        main_module.app.state.supabase.table.side_effect = ConnectionError("down")
        resp = client_no_lifespan.get("/readiness")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"
        assert resp.json()["supabase"] == "unreachable"


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

class TestAlphaAgentExceptionHandler:
    def test_agent_error_returns_structured_json_with_correct_status(self, client_no_lifespan):
        @main_module.app.get("/__raise_agent_error_test")
        async def _raise():
            raise AgentError(message="pipeline failed", detail="x")

        main_module.app.state.sentry_ok = False
        resp = client_no_lifespan.get("/__raise_agent_error_test")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "AGENT_ERROR"
        assert body["message"] == "pipeline failed"
        assert "trace_id" in body

    def test_sentry_capture_called_when_sentry_enabled(self, client_no_lifespan):
        @main_module.app.get("/__raise_for_sentry_test")
        async def _raise():
            raise AgentError(message="boom")

        main_module.app.state.sentry_ok = True
        with patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture:
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope
            resp = client_no_lifespan.get("/__raise_for_sentry_test")
            mock_capture.assert_called_once()
        assert resp.status_code == 500


class TestUnhandledExceptionHandler:
    def test_dev_mode_includes_exception_detail(self, client_no_lifespan):
        @main_module.app.get("/__raise_generic_error_test")
        async def _raise():
            raise RuntimeError("something broke")

        main_module.app.state.sentry_ok = False
        resp = client_no_lifespan.get("/__raise_generic_error_test")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "INTERNAL_ERROR"
        if main_module.settings.APP_ENV != "production":
            assert "detail" in body
            assert "something broke" in body["detail"]

    def test_response_always_includes_trace_id(self, client_no_lifespan):
        @main_module.app.get("/__raise_generic_error_test_2")
        async def _raise():
            raise RuntimeError("boom")

        main_module.app.state.sentry_ok = False
        resp = client_no_lifespan.get("/__raise_generic_error_test_2")
        assert "trace_id" in resp.json()
        assert len(resp.json()["trace_id"]) == 8


# ---------------------------------------------------------------------------
# request_timing_middleware
# ---------------------------------------------------------------------------

class TestRequestTimingMiddleware:
    def test_response_passes_through_unchanged(self, client_no_lifespan):
        resp = client_no_lifespan.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_logs_method_path_status_duration(self, client_no_lifespan, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="api.main"):
            client_no_lifespan.get("/health")
        assert any("GET" in r.message and "/health" in r.message for r in caplog.records)