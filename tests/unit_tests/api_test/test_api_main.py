"""Stage 10A tests for the FastAPI lifecycle and safe error boundary."""

from __future__ import annotations

from contextlib import asynccontextmanager
from itertools import product
import os
import socket
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
import core.observability as core_observability
from api.config import Settings
from api.core.exceptions import AgentError
from api.dependencies import ManagerAgentFactory
from core.error_handler import TemporalLeakageError, VersionMismatchError
from langsmith.utils import tracing_is_enabled


def runtime_settings(**overrides):
    values = {
        "ANTHROPIC_API_KEY": "test-key",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        "NEO4J_URI": "bolt://example",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "APP_ENV": "test",
        "OBSERVABILITY_ENABLED": False,
        "SENTRY_ENABLED": False,
        "LANGSMITH_ENABLED": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_lifespan_keeps_only_clients_and_stateless_factory_on_app_state():
    application = MagicMock()
    application.state.runtime_settings = runtime_settings()
    events = []
    application.state.telemetry_sink = events.append
    supabase = MagicMock()
    neo4j = MagicMock()

    with patch("api.main.create_client", return_value=supabase), patch(
        "api.main.GraphDatabase.driver", return_value=neo4j
    ), patch("api.main.init_sentry", return_value=False) as sentry, patch(
        "api.main.init_langsmith", return_value=False
    ) as langsmith:
        async with main_module.lifespan(application):
            assert application.state.supabase is supabase
            assert application.state.neo4j_driver is neo4j
            assert isinstance(application.state.manager_factory, ManagerAgentFactory)
            assert application.state.manager_agent is application.state.manager_factory
        neo4j.close.assert_called_once_with()
        sentry.assert_called_once_with(
            app_env="test",
            observability_enabled=False,
            exporter_enabled=False,
            dsn=application.state.runtime_settings.SENTRY_DSN,
        )
        langsmith.assert_called_once_with(
            observability_enabled=False,
            exporter_enabled=False,
            api_key=application.state.runtime_settings.LANGSMITH_API_KEY,
            project=application.state.runtime_settings.LANGSMITH_PROJECT,
        )
    assert [event["event"] for event in events] == [
        "api.startup.started",
        "api.startup.completed",
        "api.shutdown.started",
        "api.shutdown.completed",
    ]
    assert len({event["run_id"] for event in events}) == 1
    assert all(event["stage"] == "10A" and event["stage_revision"] == 1 for event in events)
    assert all(event["attempt"] == 3 and event["agent"] == "core" for event in events)
    assert all(event["implementation_version"] == "stage10a-r1-a3" for event in events)
    assert all(event.get("latency_ms", 0) >= 0 for event in events)


@pytest.mark.asyncio
async def test_lifespan_propagates_startup_failure_without_sys_exit(caplog):
    application = MagicMock()
    application.state.runtime_settings = runtime_settings()
    events = []
    application.state.telemetry_sink = events.append
    with caplog.at_level("ERROR", logger="api.main"):
        with patch("api.main.create_client", side_effect=ConnectionError("PRIVATE_SENTINEL")):
            with pytest.raises(ConnectionError, match="PRIVATE_SENTINEL"):
                async with main_module.lifespan(application):
                    pass
    assert [event["event"] for event in events] == [
        "api.startup.started", "api.startup.failed"
    ]
    assert events[-1]["error_code"] == "API_STARTUP_FAILED"
    assert "PRIVATE_SENTINEL" not in caplog.text


@pytest.mark.asyncio
async def test_complete_exporter_gates_use_settings_and_fail_nonfatally(caplog):
    runtime = runtime_settings(
        OBSERVABILITY_ENABLED=True,
        SENTRY_ENABLED=True,
        LANGSMITH_ENABLED=True,
        SENTRY_DSN="PRIVATE_DSN",
        LANGSMITH_API_KEY="PRIVATE_KEY",
        LANGSMITH_PROJECT="private-project",
    )
    application = MagicMock()
    application.state.runtime_settings = runtime
    application.state.telemetry_sink = None
    with caplog.at_level("WARNING", logger="api.main"), patch(
        "api.main.init_sentry", side_effect=RuntimeError("PRIVATE_SENTINEL")
    ) as sentry, patch(
        "api.main.init_langsmith", side_effect=RuntimeError("PRIVATE_SENTINEL")
    ) as langsmith, patch("api.main.create_client", return_value=MagicMock()), patch(
        "api.main.GraphDatabase.driver", return_value=MagicMock()
    ):
        async with main_module.lifespan(application):
            assert application.state.sentry_ok is False
            assert application.state.langsmith_ok is False
    sentry.assert_called_once_with(
        app_env="test", observability_enabled=True, exporter_enabled=True, dsn="PRIVATE_DSN"
    )
    langsmith.assert_called_once_with(
        observability_enabled=True,
        exporter_enabled=True,
        api_key="PRIVATE_KEY",
        project="private-project",
    )
    assert "PRIVATE_SENTINEL" not in caplog.text
    assert "PRIVATE_DSN" not in caplog.text
    assert "PRIVATE_KEY" not in caplog.text


INCOMPLETE_EXPORTER_GATE_MATRIX = [
    pytest.param(
        observability_enabled,
        sentry_enabled,
        sentry_credential,
        langsmith_enabled,
        langsmith_credential,
        id=(
            f"obs-{int(observability_enabled)}-sentry-{int(sentry_enabled)}-"
            f"dsn-{int(sentry_credential)}-langsmith-{int(langsmith_enabled)}-"
            f"key-{int(langsmith_credential)}"
        ),
    )
    for (
        observability_enabled,
        sentry_enabled,
        sentry_credential,
        langsmith_enabled,
        langsmith_credential,
    ) in product((False, True), repeat=5)
    if not (
        observability_enabled
        and sentry_enabled
        and sentry_credential
        and langsmith_enabled
        and langsmith_credential
    )
]


@pytest.mark.parametrize(
    (
        "observability_enabled",
        "sentry_enabled",
        "sentry_credential",
        "langsmith_enabled",
        "langsmith_credential",
    ),
    INCOMPLETE_EXPORTER_GATE_MATRIX,
)
def test_hostile_legacy_exporter_state_is_cleared_for_every_incomplete_gate(
    monkeypatch,
    observability_enabled,
    sentry_enabled,
    sentry_credential,
    langsmith_enabled,
    langsmith_credential,
):
    private = "PRIVATE_SENTINEL"
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", private)
    core_observability._sentry_enabled = True
    core_observability._langsmith_enabled = True
    runtime = runtime_settings(
        OBSERVABILITY_ENABLED=observability_enabled,
        SENTRY_ENABLED=sentry_enabled,
        SENTRY_DSN=f"https://{private}@sentry.invalid/1" if sentry_credential else "",
        LANGSMITH_ENABLED=langsmith_enabled,
        LANGSMITH_API_KEY=private if langsmith_credential else "",
    )

    try:
        with patch("sentry_sdk.init") as sentry_init, patch(
            "sentry_sdk.capture_exception"
        ) as sentry_capture, patch("core.telemetry_export.OperationalClient") as langsmith_client, patch.object(
            socket, "create_connection"
        ) as create_connection, patch.object(socket, "socket") as socket_type:
            expected = (observability_enabled and sentry_enabled and sentry_credential, observability_enabled and langsmith_enabled and langsmith_credential)
            assert main_module._configure_optional_observability(runtime) == expected

        assert core_observability.sentry_enabled() == expected[0]
        assert core_observability.langsmith_enabled() == expected[1]
        assert tracing_is_enabled() == expected[1]
        assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
        assert os.environ["LANGSMITH_TRACING"] == "false"
        assert "LANGCHAIN_API_KEY" not in os.environ
        assert sentry_init.called == expected[0]
        sentry_capture.assert_not_called()
        assert langsmith_client.called == expected[1]
        create_connection.assert_not_called()
        socket_type.assert_not_called()
    finally:
        from langsmith.run_trees import configure
        configure(enabled=False, client=None)
        core_observability._langsmith_client = None
        core_observability._sentry_enabled = False
        core_observability._langsmith_enabled = False


@pytest.fixture
def client():
    application = main_module.create_app(runtime_settings=runtime_settings())

    @asynccontextmanager
    async def no_io_lifespan(app):
        app.state.sentry_ok = False
        app.state.supabase = MagicMock()
        yield

    application.router.lifespan_context = no_io_lifespan
    with TestClient(application, raise_server_exceptions=False) as test_client:
        yield test_client, application


def test_health_exposes_runtime_version(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "runtime_version": "stage10a-runtime-v1",
    }


def test_readiness_is_fail_closed(client):
    test_client, application = client
    application.state.supabase.table.side_effect = ConnectionError("down")
    response = test_client.get("/readiness")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_alpha_agent_error_keeps_stable_envelope(client):
    test_client, application = client

    @application.get("/__stage10a_agent_error")
    async def fail():
        raise AgentError("public message", detail="public detail")

    response = test_client.get("/__stage10a_agent_error")
    assert response.status_code == 500
    assert response.json()["error"] == "AGENT_ERROR"
    assert len(response.json()["trace_id"]) == 8


@pytest.mark.parametrize(
    ("exception", "expected_code", "forbidden"),
    [
        (TemporalLeakageError("secret timestamp 2099-01-01"), "TEMPORAL_BOUNDARY_ERROR", "2099"),
        (VersionMismatchError("secret-model-v99"), "VERSION_MISMATCH", "v99"),
    ],
)
def test_core_contract_errors_are_sanitized(client, exception, expected_code, forbidden):
    test_client, application = client

    @application.get(f"/__stage10a_core_error_{expected_code}")
    async def fail():
        raise exception

    response = test_client.get(f"/__stage10a_core_error_{expected_code}")
    assert response.json()["error"] == expected_code
    assert forbidden not in response.text
    assert response.json()["detail"] is None


def test_unhandled_error_never_exposes_internal_detail(client):
    test_client, application = client

    @application.get("/__stage10a_unhandled")
    async def fail():
        raise RuntimeError("database-password-is-secret")

    response = test_client.get("/__stage10a_unhandled")
    assert response.status_code == 500
    assert "database-password-is-secret" not in response.text


def test_existing_route_skeletons_are_registered(client):
    _, application = client
    paths = set(application.openapi()["paths"])
    assert "/api/v1/analyze" in paths
    assert any(path.startswith("/api/v1/analyze/stream") for path in paths)


def _telemetry_client(sink):
    application = main_module.create_app(
        runtime_settings=runtime_settings(), telemetry_sink=sink
    )

    @asynccontextmanager
    async def no_io_lifespan(app):
        app.state.sentry_ok = False
        app.state.supabase = MagicMock()
        yield

    application.router.lifespan_context = no_io_lifespan
    return application


def test_request_events_share_one_opaque_id_and_have_terminal_latency():
    events = []
    application = _telemetry_client(events.append)
    with TestClient(application, raise_server_exceptions=False) as test_client:
        assert test_client.get("/health").status_code == 200
    assert [event["event"] for event in events] == [
        "api.request.started", "api.request.completed"
    ]
    assert len({event["run_id"] for event in events}) == 1
    assert len(events[0]["run_id"]) == 32
    assert events[-1]["latency_ms"] >= 0
    assert all(event["stage"] == "10A" and event["agent"] == "core" for event in events)


@pytest.mark.asyncio
async def test_concurrent_requests_have_distinct_correlated_ids():
    import asyncio
    from starlette.requests import Request
    from starlette.responses import Response

    events = []
    application = _telemetry_client(events.append)

    async def invoke():
        request = Request({
            "type": "http",
            "app": application,
            "method": "GET",
            "path": "/health",
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1),
            "scheme": "http",
            "http_version": "1.1",
        })

        async def call_next(_request):
            await asyncio.sleep(0)
            return Response(status_code=200)

        return await main_module.request_timing_middleware(request, call_next)

    first, second = await asyncio.gather(invoke(), invoke())
    assert [first.status_code, second.status_code] == [200, 200]
    by_id = {}
    for event in events:
        by_id.setdefault(event["run_id"], []).append(event["event"])
    assert len(by_id) == 2
    assert all(names == ["api.request.started", "api.request.completed"] for names in by_id.values())


def test_failing_telemetry_sink_cannot_change_request_behavior():
    def failing_sink(_event):
        raise RuntimeError("PRIVATE_SENTINEL")

    application = _telemetry_client(failing_sink)
    with TestClient(application, raise_server_exceptions=False) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_failed_request_emits_payload_free_terminal_event(caplog):
    events = []
    application = _telemetry_client(events.append)

    @application.get("/__private_failure")
    async def fail():
        raise RuntimeError("PRIVATE_SENTINEL")

    with caplog.at_level("ERROR", logger="api.main"):
        with TestClient(application, raise_server_exceptions=False) as test_client:
            response = test_client.get("/__private_failure")
    assert response.status_code == 500
    assert [event["event"] for event in events] == [
        "api.request.started", "api.request.failed"
    ]
    assert events[-1]["error_code"] == "REQUEST_EXECUTION_FAILED"
    assert "PRIVATE_SENTINEL" not in str(events)
    assert "PRIVATE_SENTINEL" not in caplog.text
