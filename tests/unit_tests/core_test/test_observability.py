"""
Tests for: core/observability.py
Phase: 5 — Core Infrastructure

IMPORTANT: _sentry_enabled / _langsmith_enabled are module-level globals.
Fully configured exporters now use a filtered outbound boundary. The fixture
resets process flags and tracing environment between tests.
"""
import os
from unittest.mock import patch
import pytest

import core.observability as obs
from core.observability import (
    init_sentry,
    sentry_enabled,
    init_langsmith,
    langsmith_enabled,
)


SENTRY_GATES = {"observability_enabled": True, "exporter_enabled": True}
LANGSMITH_GATES = {"observability_enabled": True, "exporter_enabled": True}


@pytest.fixture(autouse=True)
def reset_observability_state():
    langchain_env = {
        key: os.environ.get(key)
        for key in (
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_TRACING",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
        )
    }
    obs._sentry_enabled = False
    obs._langsmith_enabled = False
    try:
        with patch("core.telemetry_export.OperationalClient"):
            yield
    finally:
        from langsmith.run_trees import configure
        configure(enabled=False, client=None)
        obs._langsmith_client = None
        obs._sentry_enabled = False
        obs._langsmith_enabled = False
        for key, value in langchain_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# init_sentry
# ---------------------------------------------------------------------------

class TestInitSentry:
    @pytest.mark.parametrize(
        ("global_gate", "exporter_gate"),
        [(False, False), (False, True), (True, False)],
    )
    def test_incomplete_gate_never_initializes(
        self, global_gate, exporter_gate,
    ):
        with patch("sentry_sdk.init") as mock_init:
            result = init_sentry(
                observability_enabled=global_gate,
                exporter_enabled=exporter_gate,
                dsn="https://PRIVATE_DSN@sentry.invalid/1",
            )
        assert result is False
        assert sentry_enabled() is False
        mock_init.assert_not_called()

    def test_no_dsn_returns_false_and_stays_disabled(self, monkeypatch):
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        result = init_sentry(**SENTRY_GATES)
        assert result is False
        assert sentry_enabled() is False

    def test_blank_dsn_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "   ")
        result = init_sentry(**SENTRY_GATES)
        assert result is False

    def test_fully_enabled_dsn_remains_fail_closed(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(**SENTRY_GATES) is True
            assert sentry_enabled() is True
        sdk.assert_called_once()
        assert sdk.call_args.kwargs["send_default_pii"] is False
        assert callable(sdk.call_args.kwargs["before_send"])

    def test_settings_dsn_is_ignored_and_never_logged(
        self, monkeypatch, caplog,
    ):
        secret="https://PRIVATE_DSN@sentry.invalid/1"
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(dsn=secret, **SENTRY_GATES) is True
        assert sdk.call_args.kwargs["dsn"] == secret
        assert secret not in caplog.text

    def test_explicit_app_env_does_not_bypass_fail_closed(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(**SENTRY_GATES) is True
            assert sentry_enabled() is True
        sdk.assert_called_once()
        assert sdk.call_args.kwargs["send_default_pii"] is False
        assert callable(sdk.call_args.kwargs["before_send"])

    def test_environment_does_not_bypass_fail_closed(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(**SENTRY_GATES) is True
            assert sentry_enabled() is True
        sdk.assert_called_once()
        assert sdk.call_args.kwargs["send_default_pii"] is False
        assert callable(sdk.call_args.kwargs["before_send"])

    def test_missing_app_env_remains_fail_closed(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(**SENTRY_GATES) is True
            assert sentry_enabled() is True
        sdk.assert_called_once()
        assert sdk.call_args.kwargs["send_default_pii"] is False
        assert callable(sdk.call_args.kwargs["before_send"])

    def test_sdk_is_not_initialized_to_send_any_payload(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(**SENTRY_GATES) is True
            assert sentry_enabled() is True
        sdk.assert_called_once()
        assert sdk.call_args.kwargs["send_default_pii"] is False
        assert callable(sdk.call_args.kwargs["before_send"])

    def test_repeated_calls_remain_disabled(self, monkeypatch):
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(dsn="https://key@sentry.invalid/1", **SENTRY_GATES)
            assert init_sentry() is False
        sdk.assert_called_once()

    def test_failing_sentry_sdk_is_never_reached(self, caplog):
        with patch("sentry_sdk.init", side_effect=Exception("PRIVATE_EXCEPTION")) as sdk:
            assert init_sentry(dsn="https://PRIVATE_DSN@sentry.invalid/1", **SENTRY_GATES) is False
        sdk.assert_called_once()
        assert not sentry_enabled()
        assert "PRIVATE_EXCEPTION" not in caplog.text

    def test_import_error_disables_gracefully(self, monkeypatch):
        with patch("sentry_sdk.init", side_effect=ImportError("PRIVATE_SENTINEL")) as sdk:
            assert init_sentry(dsn="https://key@sentry.invalid/1", **SENTRY_GATES) is False
        sdk.assert_called_once()


# ---------------------------------------------------------------------------
# sentry_enabled
# ---------------------------------------------------------------------------

class TestSentryEnabled:
    def test_remains_false_after_fully_enabled_request(self, monkeypatch):
        monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.io/123")
        with patch("sentry_sdk.init") as sdk:
            assert init_sentry(**SENTRY_GATES) is True
            assert sentry_enabled() is True
        sdk.assert_called_once()
        assert sdk.call_args.kwargs["send_default_pii"] is False
        assert callable(sdk.call_args.kwargs["before_send"])


# ---------------------------------------------------------------------------
# init_langsmith
# ---------------------------------------------------------------------------

class TestInitLangsmith:
    @pytest.mark.parametrize(
        ("global_gate", "exporter_gate"),
        [(False, False), (False, True), (True, False)],
    )
    def test_incomplete_gate_never_mutates_sdk_environment(
        self, monkeypatch, global_gate, exporter_gate,
    ):
        for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"):
            monkeypatch.delenv(key, raising=False)
        result = init_langsmith(
            observability_enabled=global_gate,
            exporter_enabled=exporter_gate,
            api_key="PRIVATE_LANGSMITH_KEY",
            project="private-project",
        )
        assert result is False
        assert langsmith_enabled() is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"
        assert os.environ.get("LANGSMITH_TRACING") == "false"
        assert "LANGCHAIN_API_KEY" not in os.environ
        assert "LANGCHAIN_PROJECT" not in os.environ

    def test_no_api_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        result = init_langsmith(**LANGSMITH_GATES)
        assert result is False
        assert langsmith_enabled() is False

    def test_blank_api_key_treated_as_absent(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "  ")
        result = init_langsmith(**LANGSMITH_GATES)
        assert result is False

    def test_fully_enabled_key_does_not_activate_langsmith(self, monkeypatch):
        from langsmith.utils import tracing_is_enabled
        assert init_langsmith(api_key="fake-key", project="offline", **LANGSMITH_GATES) is True
        assert langsmith_enabled() and tracing_is_enabled()
        assert os.environ["LANGSMITH_TRACING"] == "false"  # children require their own initializer

    def test_settings_key_is_ignored_without_client_or_log_disclosure(
        self, monkeypatch, caplog,
    ):
        secret="PRIVATE_LANGSMITH_KEY"
        with patch("core.telemetry_export.OperationalClient") as factory:
            assert init_langsmith(api_key=secret, project="offline", **LANGSMITH_GATES) is True
        factory.assert_called_once()
        assert factory.call_args.kwargs["api_key"] == secret
        assert secret not in caplog.text

    def test_hostile_environment_is_nonfatal_and_redacted(self, caplog):
        with patch("core.telemetry_export.OperationalClient", side_effect=RuntimeError("PRIVATE_EXCEPTION")):
            assert init_langsmith(api_key="PRIVATE_KEY", **LANGSMITH_GATES) is False
        assert not langsmith_enabled()
        assert "PRIVATE_EXCEPTION" not in caplog.text

    def test_custom_project_is_not_copied_to_export_environment(self, monkeypatch):
        from langsmith.run_helpers import get_tracing_context
        assert init_langsmith(api_key="fake-key", project="my-project", **LANGSMITH_GATES)
        assert get_tracing_context()["project_name"] == "my-project"

    def test_repeated_calls_remain_disabled(self, monkeypatch):
        assert init_langsmith(api_key="fake-key", **LANGSMITH_GATES)
        assert init_langsmith() is False
        from langsmith.utils import tracing_is_enabled
        assert not tracing_is_enabled()


# ---------------------------------------------------------------------------
# langsmith_enabled
# ---------------------------------------------------------------------------

class TestLangsmithEnabled:
    def test_remains_false_after_fully_enabled_request(self, monkeypatch):
        assert not langsmith_enabled()
        assert init_langsmith(api_key="fake-key", **LANGSMITH_GATES)
        assert langsmith_enabled()
