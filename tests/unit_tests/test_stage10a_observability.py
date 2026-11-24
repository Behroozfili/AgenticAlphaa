"""Attempt-3 fail-closed exporter checks for the Stage 10A runtime."""

from __future__ import annotations

import itertools
import math
import os
import socket
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import core.observability as core_observability
import langsmith_trace
from api.config import Settings
from api.main import _configure_optional_observability
from core.error_handler import add_breadcrumb, capture_exception, capture_message
from langsmith.utils import tracing_is_enabled
from langsmith_trace import emit_stage10a_event, stage10a_event


PRIVATE_SENTINEL = "PRIVATE_SENTINEL_REQUEST_QUERY_PREFERENCES_STATE_EXCEPTION"
_TRACE_ENV_KEYS = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
)


@pytest.fixture(autouse=True)
def reset_export_state():
    prior = {key: os.environ.get(key) for key in _TRACE_ENV_KEYS}
    core_observability._sentry_enabled = False
    core_observability._langsmith_enabled = False
    langsmith_trace.client = None
    try:
        with patch("core.telemetry_export.OperationalClient"):
            yield
    finally:
        core_observability._sentry_enabled = False
        core_observability._langsmith_enabled = False
        langsmith_trace.client = None
        from langsmith.run_trees import configure
        configure(enabled=False, client=None)
        core_observability._langsmith_client = None
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _fully_enabled_settings() -> Settings:
    return Settings(
        _env_file=None,
        ANTHROPIC_API_KEY="test-anthropic-key",
        SUPABASE_URL="https://offline.invalid",
        SUPABASE_SERVICE_ROLE_KEY="test-supabase-key",
        NEO4J_URI="bolt://offline.invalid",
        NEO4J_USER="neo4j",
        NEO4J_PASSWORD="test-password",
        APP_ENV="test",
        OBSERVABILITY_ENABLED=True,
        SENTRY_ENABLED=True,
        LANGSMITH_ENABLED=True,
        SENTRY_DSN=f"https://{PRIVATE_SENTINEL}@sentry.invalid/1",
        LANGSMITH_API_KEY=PRIVATE_SENTINEL,
        LANGSMITH_PROJECT="private-project",
    )


def test_attempt3_runtime_identity_is_exact_and_has_no_stale_prior_attempt_value():
    identity = dict(langsmith_trace.STAGE10A_RUNTIME_IDENTITY)
    assert identity == {
        "stage": "10A",
        "stage_revision": 1,
        "attempt": 3,
        "agent": "core",
        "implementation_version": "stage10a-r1-a3",
    }
    event = stage10a_event(
        "api.request.completed",
        metadata={"run_id": "attempt-3-run", "status": "COMPLETED"},
        started_at=time.perf_counter(),
    )
    assert event["attempt"] == 3
    assert event["implementation_version"] == "stage10a-r1-a3"
    assert "stage10a-r1-a1" not in repr(event)
    assert "stage10a-r1-a2" not in repr(event)
    assert math.isfinite(event["latency_ms"])
    assert event["latency_ms"] >= 0


def test_api_hostile_legacy_environment_matrix_always_disables_exporters(
    caplog, monkeypatch,
):
    for enabled, sentry, dsn, tracing, key in itertools.product((False, True), repeat=5):
        settings=Settings(_env_file=None, OBSERVABILITY_ENABLED=enabled,
            SENTRY_ENABLED=sentry, SENTRY_DSN="https://key@sentry.invalid/1" if dsn else "",
            LANGSMITH_ENABLED=tracing, LANGSMITH_API_KEY="fake-key" if key else "")
        with patch("sentry_sdk.init") as sentry_init:
            expected=(enabled and sentry and dsn, enabled and tracing and key)
            assert _configure_optional_observability(settings) == expected
            assert tracing_is_enabled() == expected[1]
            assert sentry_init.called == expected[0]


def test_fully_enabled_configuration_makes_zero_external_calls(caplog, monkeypatch):
    with patch("sentry_sdk.init") as sentry_init, patch("core.telemetry_export.OperationalClient") as factory:
        assert _configure_optional_observability(_fully_enabled_settings()) == (True, True)
    assert callable(sentry_init.call_args.kwargs["before_send"])
    assert factory.call_args.kwargs["omit_traced_runtime_info"] is True
    assert PRIVATE_SENTINEL not in caplog.text


def test_raw_request_manager_and_exception_values_cannot_reach_exporter():
    from core.telemetry_export import OperationalClient, Client, sentry_event
    core_observability._sentry_enabled=True
    core_observability._langsmith_enabled=True
    unsafe={"task_query":PRIVATE_SENTINEL,"manager_state":PRIVATE_SENTINEL,"user_id":PRIVATE_SENTINEL}
    event=sentry_event({"request":unsafe,"extra":unsafe,"user":unsafe,
        "exception":{"values":[{"type":"RuntimeError","value":PRIVATE_SENTINEL,
        "stacktrace":{"frames":[{"filename":"module.py","function":"run","lineno":1,"vars":unsafe}]}}]}}, {})
    assert PRIVATE_SENTINEL not in repr(event)
    assert event["exception"]["values"][0]["type"]=="RuntimeError"
    with patch.object(Client,"create_run") as create, patch.object(Client,"update_run") as update:
        # The fixture mocks the constructor; bind the real exported class methods.
        import importlib.util
        spec=importlib.util.spec_from_file_location("boundary_check", "core/telemetry_export.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        client=object.__new__(module.OperationalClient)
        client.create_run("manager",unsafe,"chain",extra={"metadata":unsafe},error=PRIVATE_SENTINEL)
        client.update_run("00000000-0000-0000-0000-000000000001",outputs=unsafe,error=PRIVATE_SENTINEL,events=[unsafe])
        assert PRIVATE_SENTINEL not in repr(create.call_args)
        assert PRIVATE_SENTINEL not in repr(update.call_args)
        assert update.call_args.kwargs["error"]=="OPERATION_FAILED"


def test_external_validation_client_is_fail_closed_but_fake_injection_remains_local():
    with patch("langsmith_trace.Client") as client_type:
        with pytest.raises(RuntimeError, match="not configured"):
            langsmith_trace.get_langsmith_client()
    client_type.assert_not_called()

    fake = object()
    langsmith_trace.client = fake
    assert langsmith_trace.get_langsmith_client() is fake


def test_initializer_and_local_sink_failures_are_nonfatal_and_payload_free(caplog):
    def failing_sink(_event):
        raise RuntimeError(PRIVATE_SENTINEL)

    with patch("api.main.init_sentry", side_effect=RuntimeError(PRIVATE_SENTINEL)), patch(
        "api.main.init_langsmith", side_effect=RuntimeError(PRIVATE_SENTINEL)
    ):
        assert _configure_optional_observability(_fully_enabled_settings()) == (False, False)

    event = emit_stage10a_event(
        "api.request.failed",
        metadata={"run_id": "failure-probe", "status": "FAILED"},
        started_at=time.perf_counter(),
        sink=failing_sink,
    )
    assert event is not None
    assert event["attempt"] == 3
    assert PRIVATE_SENTINEL not in repr(event)
    assert PRIVATE_SENTINEL not in caplog.text


def test_fail_closed_disposition_is_justified_by_unaudited_legacy_paths():
    assert core_observability.EXTERNAL_EXPORTERS_REQUEST_SAFE is True
    with patch("sentry_sdk.init") as sdk:
        assert _configure_optional_observability(_fully_enabled_settings())==(True,True)
    assert callable(sdk.call_args.kwargs["before_send"])
    assert langsmith_trace.get_langsmith_client() is core_observability.telemetry_client()
