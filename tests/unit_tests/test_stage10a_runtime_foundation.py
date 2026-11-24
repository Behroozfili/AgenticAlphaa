"""Cross-module acceptance checks for the Stage 10A runtime foundation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.main import create_app
from api.config import Settings
from api.dependencies import ManagerAgentFactory
from memory.manager_memory import ManagerMemory


def test_runtime_contract_serializes_through_canonical_forecast_context():
    runtime = Settings(_env_file=None)
    context = runtime.forecast_context(
        ticker="msft",
        as_of="2024-05-31T20:00:00Z",
        analysis_mode="HISTORICAL_BACKTEST",
    )
    restored = type(context).from_dict(context.to_dict())
    assert restored == context
    assert restored.horizon_trading_days == 21
    assert restored.dataset_version == runtime.HISTORICAL_DATASET_VERSION


def test_two_request_memories_have_zero_short_and_monthly_crosstalk():
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    factory = ManagerAgentFactory(client, MagicMock(), Settings(_env_file=None))
    first = factory.create_memory(user_id="user-a")
    second = factory.create_memory(user_id="user-b")
    assert isinstance(first, ManagerMemory)
    first.new_session("a", "first")
    second.new_session("b", "second")
    first.add_message("user", "alpha")
    first.store_monthly_insight(
        ticker="MSFT", as_of="2024-05-31T20:00:00Z", horizon=21,
        analysis_mode="LIVE", schema_version="v2", insight={"request": "a"},
    )
    assert second.get_messages() == []
    assert second.long.monthly_insights == {}
    assert first.short is not second.short
    assert first.long is not second.long


def test_factory_retains_clients_but_not_manager_or_memory_instances():
    factory = ManagerAgentFactory(MagicMock(), MagicMock(), Settings(_env_file=None))
    fields = set(factory.__dataclass_fields__)
    assert fields == {
        "supabase_client", "neo4j_driver", "runtime_settings", "manager_builder",
        "market_forecast_service", "market_data_provider",
    }
    assert not hasattr(factory, "memory")
    assert not hasattr(factory, "manager")


def test_real_request_lifecycle_and_dependency_events_share_exactly_one_run_id():
    events = []
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []

    class FakeManager:
        async def run(self, **_kwargs):
            return {}

    runtime = Settings(_env_file=None, APP_ENV="test")
    factory = ManagerAgentFactory(
        client,
        MagicMock(),
        runtime,
        manager_builder=lambda _memory: FakeManager(),
    )
    application = create_app(runtime_settings=runtime, telemetry_sink=events.append)

    @asynccontextmanager
    async def no_io_lifespan(app):
        app.state.sentry_ok = False
        app.state.supabase = client
        app.state.manager_factory = factory
        app.state.manager_agent = factory
        yield

    application.router.lifespan_context = no_io_lifespan
    with TestClient(application, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/api/v1/analyze",
            json={
                "query": "PRIVATE_SENTINEL confidential question",
                "ticker": "MSFT",
                "user_id": "PRIVATE_SENTINEL",
            },
        )
    assert response.status_code == 200
    assert [event["event"] for event in events] == [
        "api.request.started",
        "manager.memory.created",
        "api.request.completed",
    ]
    assert len({event["run_id"] for event in events}) == 1
    assert "PRIVATE_SENTINEL" not in str(events)
    assert all(event["implementation_version"] == "stage10a-r1-a3" for event in events)
    assert all(event.get("latency_ms", 0) >= 0 for event in events)
