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

from api.dependencies import get_user_id, get_manager_memory
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
    def test_returns_x_user_id_header_when_present(self):
        request = make_request(headers={"X-User-Id": "user-123"})
        assert get_user_id(request) == "user-123"

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