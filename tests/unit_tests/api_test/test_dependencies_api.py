"""
Tests for: api/dependencies.py
Phase: 7 — API Layer (Integration)

Mocking strategy: FastAPI's Request and app.state are both MagicMock —
no real ASGI app or HTTP server needed since these are plain dependency
functions (not route handlers), callable directly with a faked Request.
ManagerMemory is NOT mocked — we let the real class run, but inject a
mocked supabase_client (per ManagerMemory's own DI design from Phase 4)
so no real Supabase calls happen.
"""
from unittest.mock import MagicMock

import pytest

from api.dependencies import (
    ManagerAgentFactory,
    get_manager_agent,
    get_manager_memory,
    get_user_id,
)
from api.config import Settings
from memory.manager_memory import ManagerMemory


def make_request(headers=None, supabase_client=None):
    request = MagicMock()
    request.headers = headers or {}
    request.app.state.supabase = supabase_client or MagicMock()
    return request


# ---------------------------------------------------------------------------
# get_user_id
# ---------------------------------------------------------------------------

class TestGetUserId:
    def test_returns_only_middleware_verified_user_id(self):
        request = make_request(headers={"X-User-Id": "forged-user"})
        request.state.user_id = "verified-user-123"
        assert get_user_id(request) == "verified-user-123"

    def test_falls_back_to_default_user_id_when_header_absent(self, monkeypatch):
        import api.config as config_module
        config_module.get_settings.cache_clear()
        monkeypatch.setenv("DEFAULT_USER_ID", "anonymous_fallback")
        config_module.get_settings.cache_clear()
        # Re-import settings reference used inside dependencies.py
        import api.dependencies as deps_module
        monkeypatch.setattr(deps_module, "settings", config_module.get_settings())

        request = make_request(headers={})
        assert get_user_id(request) == "anonymous_fallback"

    def test_empty_header_value_falls_back_to_default(self, monkeypatch):
        import api.dependencies as deps_module
        monkeypatch.setattr(deps_module.settings, "DEFAULT_USER_ID", "fallback_id")

        request = make_request(headers={"X-User-Id": ""})
        # empty string is falsy -> `or` falls through to default
        assert get_user_id(request) == "fallback_id"


# ---------------------------------------------------------------------------
# get_manager_memory
# ---------------------------------------------------------------------------

class TestGetManagerMemory:
    def test_creates_manager_memory_with_resolved_user_id_and_supabase_client(self):
        mock_client = MagicMock()
        response = MagicMock()
        response.data = []
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = response

        request = make_request(supabase_client=mock_client)

        memory = get_manager_memory(request, user_id="user-456")

        assert isinstance(memory, ManagerMemory)
        assert memory.long._user_id == "user-456"
        assert memory.long._db is mock_client

    def test_uses_app_state_supabase_not_a_new_connection(self):
        mock_client = MagicMock()
        response = MagicMock()
        response.data = []
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = response
        request = make_request(supabase_client=mock_client)

        memory = get_manager_memory(request, user_id="u1")

        # The exact same mock object instance must be reused (no new client created)
        assert memory.long._db is request.app.state.supabase

    def test_long_term_memory_is_loaded_during_construction(self):
        """
        ManagerMemory's long-term layer is loaded via LongTermMemory.create()
        per the docstring's claim "load() is called in ManagerMemory.__init__".
        We verify the supabase client's .table() was actually invoked (i.e.
        load() ran), not just that ManagerMemory was constructed.
        """
        mock_client = MagicMock()
        response = MagicMock()
        response.data = [{"operational_heuristics": {"h": 1}, "ticker_insights": {},
                          "user_preferences": {}}]
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = response
        request = make_request(supabase_client=mock_client)

        memory = get_manager_memory(request, user_id="u1")

        mock_client.table.assert_called_with("long_term_memory")
        assert memory.long.operational_heuristics == {"h": 1}

    def test_memory_event_reuses_request_run_id_and_is_payload_free(self):
        events = []
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        request = make_request(supabase_client=mock_client)
        request.state.stage10a_run_id = "a" * 32
        request.app.state.telemetry_sink = events.append
        get_manager_memory(request, user_id="PRIVATE_SENTINEL")
        assert [event["event"] for event in events] == ["manager.memory.created"]
        assert events[0]["run_id"] == "a" * 32
        assert events[0]["latency_ms"] >= 0
        assert "PRIVATE_SENTINEL" not in str(events)


class TestManagerAgentFactory:
    @pytest.mark.asyncio
    async def test_each_run_builds_fresh_manager_and_memory(self):
        import asyncio

        memories = []

        class FakeManager:
            def __init__(self, memory):
                self.memory = memory

            async def run(self, *, client_session_id, **_kwargs):
                self.memory.new_session(client_session_id, client_session_id)
                self.memory.add_message("user", client_session_id)
                await asyncio.sleep(0)
                return {"session": self.memory.short.session_id, "count": len(self.memory.get_messages())}

        client = MagicMock()
        client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        factory = ManagerAgentFactory(
            supabase_client=client,
            neo4j_driver=MagicMock(),
            runtime_settings=Settings(_env_file=None),
            manager_builder=lambda memory: memories.append(memory) or FakeManager(memory),
        )
        events = []
        first, second = await asyncio.gather(
            factory.run(
                task_query="a", client_session_id="session-a",
                _stage10a_run_id="a" * 32, _stage10a_event_sink=events.append,
            ),
            factory.run(
                task_query="b", client_session_id="session-b",
                _stage10a_run_id="b" * 32, _stage10a_event_sink=events.append,
            ),
        )
        assert first == {"session": "session-a", "count": 1}
        assert second == {"session": "session-b", "count": 1}
        assert memories[0] is not memories[1]
        assert memories[0].get_messages() != memories[1].get_messages()
        by_id = {}
        for event in events:
            by_id.setdefault(event["run_id"], []).append(event["event"])
        assert by_id == {
            "a" * 32: ["manager.memory.created", "manager.factory.created"],
            "b" * 32: ["manager.memory.created", "manager.factory.created"],
        }
        assert all(event["latency_ms"] >= 0 for event in events)

    def test_dependency_binds_fresh_manager_to_supplied_memory(self):
        memory = MagicMock(spec=ManagerMemory)
        manager = object()
        factory = MagicMock(spec=ManagerAgentFactory)
        factory.for_request.return_value = manager
        assert get_manager_agent(memory=memory, factory=factory) is manager
        factory.for_request.assert_called_once_with(memory)

    def test_factory_failure_is_correlated_redacted_and_reraised(self, caplog):
        events = []
        factory = ManagerAgentFactory(
            MagicMock(), MagicMock(), Settings(_env_file=None),
            manager_builder=lambda _memory: (_ for _ in ()).throw(RuntimeError("PRIVATE_SENTINEL")),
        )
        with pytest.raises(RuntimeError, match="PRIVATE_SENTINEL"):
            factory.for_request(
                MagicMock(spec=ManagerMemory), run_id="c" * 32, sink=events.append
            )
        assert [event["event"] for event in events] == ["manager.factory.failed"]
        assert events[0]["run_id"] == "c" * 32
        assert events[0]["error_code"] == "MANAGER_CREATE_FAILED"
        assert "PRIVATE_SENTINEL" not in str(events)

    @pytest.mark.asyncio
    async def test_failing_sink_does_not_change_factory_run(self):
        class FakeManager:
            async def run(self, **_kwargs):
                return {"status": "ok"}

        client = MagicMock()
        client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        factory = ManagerAgentFactory(
            client, MagicMock(), Settings(_env_file=None), manager_builder=lambda _memory: FakeManager()
        )
        result = await factory.run(
            task_query="PRIVATE_SENTINEL",
            _stage10a_run_id="d" * 32,
            _stage10a_event_sink=lambda _event: (_ for _ in ()).throw(RuntimeError("sink")),
        )
        assert result == {"status": "ok"}
