"""
Tests for: core/observability.py
Phase: 5 — Core Infrastructure

IMPORTANT: _sentry_enabled / _langsmith_enabled are module-level globals
with NO reset function provided by the module itself. We reset them
directly via monkeypatch.setattr on the module object between tests to
avoid state leaking across tests (idempotency is a documented feature,
so without resetting, every test after the first init_sentry() call would
silently see _sentry_enabled=True regardless of DSN).
"""
from unittest.mock import patch
import pytest

import core.observability as obs
from core.observability import (
    init_sentry,
    sentry_enabled,
    init_langsmith,
    langsmith_enabled,
)


@pytest.fixture(autouse=True)
def reset_observability_state():
    obs._sentry_enabled = False
    obs._langsmith_enabled = False
    yield
    obs._sentry_enabled = False
    obs._langsmith_enabled = False


# ---------------------------------------------------------------------------
# init_sentry
# ---------------------------------------------------------------------------

class TestInitSentry:
    def test_no_dsn_returns_false_and_stays_disabled(self, monkeypatch):
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        result = init_sentry()
        assert result is False
        assert sentry_enabled() is False

    def test_blank_dsn_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "   ")
        result = init_sentry()
        assert result is False

    def test_valid_dsn_initialises_and_returns_true(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as mock_init:
            result = init_sentry()
        assert result is True
        assert sentry_enabled() is True
        mock_init.assert_called_once()

    def test_explicit_app_env_passed_to_sentry_init(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as mock_init:
            init_sentry(app_env="production")
        _, kwargs = mock_init.call_args
        assert kwargs["environment"] == "production"

    def test_app_env_falls_back_to_env_var_then_development(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        monkeypatch.setenv("APP_ENV", "staging")
        with patch("sentry_sdk.init") as mock_init:
            init_sentry()
        _, kwargs = mock_init.call_args
        assert kwargs["environment"] == "staging"

    def test_app_env_defaults_to_development_when_unset(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        monkeypatch.delenv("APP_ENV", raising=False)
        with patch("sentry_sdk.init") as mock_init:
            init_sentry()
        _, kwargs = mock_init.call_args
        assert kwargs["environment"] == "development"

    def test_send_default_pii_is_false(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as mock_init:
            init_sentry()
        _, kwargs = mock_init.call_args
        assert kwargs["send_default_pii"] is False

    def test_idempotent_second_call_is_noop(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as mock_init:
            init_sentry()
            init_sentry()
        mock_init.assert_called_once()  # second call short-circuits

    def test_sentry_init_exception_disables_gracefully(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://malformed-dsn")
        with patch("sentry_sdk.init", side_effect=Exception("bad dsn")):
            result = init_sentry()
        assert result is False
        assert sentry_enabled() is False

    def test_import_error_disables_gracefully(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch.dict("sys.modules", {"sentry_sdk": None}):
            result = init_sentry()
        assert result is False


# ---------------------------------------------------------------------------
# sentry_enabled
# ---------------------------------------------------------------------------

class TestSentryEnabled:
    def test_reflects_current_state(self, monkeypatch):
        assert sentry_enabled() is False
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init"):
            init_sentry()
        assert sentry_enabled() is True


# ---------------------------------------------------------------------------
# init_langsmith
# ---------------------------------------------------------------------------

class TestInitLangsmith:
    def test_no_api_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        result = init_langsmith()
        assert result is False
        assert langsmith_enabled() is False

    def test_blank_api_key_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "  ")
        result = init_langsmith()
        assert result is False

    def test_valid_key_sets_langchain_env_vars(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key-123")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        result = init_langsmith()
        assert result is True
        assert langsmith_enabled() is True
        import os
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-key-123"
        assert os.environ["LANGCHAIN_PROJECT"] == "alpha-agent-node"

    def test_custom_project_name_used_when_set(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-custom-project")
        init_langsmith()
        import os
        assert os.environ["LANGCHAIN_PROJECT"] == "my-custom-project"

    def test_idempotent_second_call_is_noop(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        first = init_langsmith()
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)  # remove key
        second = init_langsmith()  # should still return True (cached state)
        assert first is True
        assert second is True


# ---------------------------------------------------------------------------
# langsmith_enabled
# ---------------------------------------------------------------------------

class TestLangsmithEnabled:
    def test_reflects_current_state(self, monkeypatch):
        assert langsmith_enabled() is False
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        init_langsmith()
        assert langsmith_enabled() is True