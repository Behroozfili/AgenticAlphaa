"""
Tests for: memory/manager_memory.py
Phase: 4 — Memory Layer

Mocking strategy: a MagicMock is injected as `supabase_client` directly into
LongTermMemory/ManagerMemory constructors (the class explicitly supports DI
for this purpose, keeping __init__ free of network I/O per its own
docstring). We mock the `.table().select().eq().execute()` and
`.table().upsert().execute()` chains to control load()/persist() behavior
without ever touching real Supabase or env vars.
"""
import time
from unittest.mock import MagicMock
import pytest

from memory.manager_memory import (
    ShortTermMemory,
    LongTermMemory,
    ManagerMemory,
    AgentExecutionRecord,
    EvaluationFeedback,
)


def make_eval(step="research", passed=True, score=80, issues=None, next_action="proceed"):
    return EvaluationFeedback(
        step=step, timestamp=time.time(), passed=passed, score=score,
        issues=issues or [], next_action=next_action, raw_verdict="{}",
    )


# ---------------------------------------------------------------------------
# ShortTermMemory
# ---------------------------------------------------------------------------

class TestShortTermMemoryLifecycle:
    def test_reset_initialises_session_fields(self):
        mem = ShortTermMemory()
        mem.reset(session_id="s1", task_query="Analyse NVDA")
        assert mem.session_id == "s1"
        assert mem.task_query == "Analyse NVDA"
        assert mem.messages == []
        assert mem.agent_log == []
        assert mem.eval_feedback == []

    def test_reset_clears_prior_session_data(self):
        mem = ShortTermMemory()
        mem.reset("s1", "q1")
        mem.add_message("user", "hello")
        mem.log_dispatch("ResearchAgent", {})
        mem.reset("s2", "q2")
        assert mem.messages == []
        assert mem.agent_log == []


class TestShortTermMemoryMessages:
    def test_add_message_appends_role_and_content(self):
        mem = ShortTermMemory()
        mem.add_message("user", "hello")
        assert mem.get_messages() == [{"role": "user", "content": "hello"}]

    def test_get_messages_returns_copy_not_reference(self):
        mem = ShortTermMemory()
        mem.add_message("user", "hi")
        msgs = mem.get_messages()
        msgs.append({"role": "x", "content": "y"})
        assert len(mem.get_messages()) == 1  # internal list unaffected

    def test_max_messages_trims_oldest_fifo(self):
        mem = ShortTermMemory(max_messages=3)
        for i in range(5):
            mem.add_message("user", f"msg{i}")
        msgs = mem.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg2"
        assert msgs[-1]["content"] == "msg4"


class TestShortTermMemoryAgentLog:
    def test_log_dispatch_returns_mutable_record(self):
        mem = ShortTermMemory()
        record = mem.log_dispatch("ResearchAgent", {"ticker": "NVDA"})
        assert isinstance(record, AgentExecutionRecord)
        assert record.agent_name == "ResearchAgent"
        assert record.outcome == "pending"

    def test_directives_are_deep_copied_not_referenced(self):
        mem = ShortTermMemory()
        directives = {"ticker": "NVDA"}
        record = mem.log_dispatch("ResearchAgent", directives)
        directives["ticker"] = "AAPL"  # mutate original after dispatch
        assert record.directives["ticker"] == "NVDA"  # unaffected

    def test_caller_can_update_record_in_place_after_dispatch(self):
        mem = ShortTermMemory()
        record = mem.log_dispatch("ResearchAgent", {})
        record.outcome = "success"
        record.duration_s = 12.4
        record.result_keys = ["aggregated_research_context"]
        last = mem.get_last_dispatch()
        assert last.outcome == "success"
        assert last.duration_s == 12.4

    def test_get_last_dispatch_returns_none_when_empty(self):
        mem = ShortTermMemory()
        assert mem.get_last_dispatch() is None

    def test_agents_run_returns_ordered_names(self):
        mem = ShortTermMemory()
        mem.log_dispatch("ResearchAgent", {})
        mem.log_dispatch("FinancialAgent", {})
        assert mem.agents_run() == ["ResearchAgent", "FinancialAgent"]

    def test_get_agent_log_returns_copy(self):
        mem = ShortTermMemory()
        mem.log_dispatch("ResearchAgent", {})
        log_copy = mem.get_agent_log()
        log_copy.append("garbage")
        assert len(mem.get_agent_log()) == 1


class TestShortTermMemoryEvaluations:
    def test_add_evaluation_and_get_last(self):
        mem = ShortTermMemory()
        ev = make_eval(step="research", score=90)
        mem.add_evaluation(ev)
        assert mem.get_last_evaluation() is ev

    def test_get_last_evaluation_none_when_empty(self):
        mem = ShortTermMemory()
        assert mem.get_last_evaluation() is None

    def test_get_evaluations_returns_copy(self):
        mem = ShortTermMemory()
        mem.add_evaluation(make_eval())
        evals = mem.get_evaluations()
        evals.append("garbage")
        assert len(mem.get_evaluations()) == 1


class TestShortTermMemoryContextDict:
    def test_to_context_dict_shape_with_no_evaluation(self):
        mem = ShortTermMemory()
        mem.reset("s1", "Analyse NVDA")
        ctx = mem.to_context_dict()
        assert ctx["session_id"] == "s1"
        assert ctx["task_query"] == "Analyse NVDA"
        assert ctx["agents_dispatched"] == []
        assert ctx["last_evaluation"] is None

    def test_to_context_dict_includes_dispatched_agents(self):
        mem = ShortTermMemory()
        mem.reset("s1", "q")
        record = mem.log_dispatch("ResearchAgent", {})
        record.outcome = "success"
        record.duration_s = 5.0
        record.result_keys = ["context"]
        ctx = mem.to_context_dict()
        assert ctx["agents_dispatched"] == [
            {"agent": "ResearchAgent", "outcome": "success", "duration": 5.0, "keys": ["context"]}
        ]

    def test_to_context_dict_includes_last_evaluation(self):
        mem = ShortTermMemory()
        mem.reset("s1", "q")
        mem.add_evaluation(make_eval(step="financial", passed=False, score=40,
                                     issues=["stale data"], next_action="re_run"))
        ctx = mem.to_context_dict()
        assert ctx["last_evaluation"] == {
            "step": "financial", "passed": False, "score": 40,
            "next_action": "re_run", "issues": ["stale data"],
        }

    def test_session_elapsed_s_increases_over_time(self):
        mem = ShortTermMemory()
        mem.reset("s1", "q")
        mem.session_start_ts = time.time() - 10
        ctx = mem.to_context_dict()
        assert ctx["session_elapsed_s"] >= 10.0


# ---------------------------------------------------------------------------
# LongTermMemory — construction (DI keeps __init__ I/O-free)
# ---------------------------------------------------------------------------

class TestLongTermMemoryInit:
    def test_injected_client_used_directly_no_env_needed(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        client = MagicMock()
        ltm = LongTermMemory(user_id="u1", supabase_client=client)
        assert ltm._db is client

    def test_missing_env_and_no_client_raises_valueerror(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        with pytest.raises(ValueError):
            LongTermMemory(user_id="u1")

    def test_user_id_falls_back_to_default_user_id_env(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_USER_ID", "env_user")
        client = MagicMock()
        ltm = LongTermMemory(user_id="", supabase_client=client)
        assert ltm._user_id == "env_user"

    def test_init_does_not_call_load(self, monkeypatch):
        """__init__ must stay I/O-free — load() is opt-in via create()."""
        client = MagicMock()
        LongTermMemory(user_id="u1", supabase_client=client)
        client.table.assert_not_called()

    def test_create_classmethod_calls_load(self, monkeypatch):
        client = MagicMock()
        response = MagicMock()
        response.data = []
        client.table.return_value.select.return_value.eq.return_value.execute.return_value = response

        LongTermMemory.create(user_id="u1", supabase_client=client)
        client.table.assert_called_once_with("long_term_memory")


# ---------------------------------------------------------------------------
# LongTermMemory — load()
# ---------------------------------------------------------------------------

class TestLongTermMemoryLoad:
    def _build(self, client):
        return LongTermMemory(user_id="u1", supabase_client=client)

    def test_load_populates_fields_from_existing_row(self):
        client = MagicMock()
        response = MagicMock()
        response.data = [{
            "operational_heuristics": {"h1": "v1"},
            "ticker_insights": {"NVDA": {"sector": "Tech"}},
            "user_preferences": {"report_format": "concise"},
        }]
        client.table.return_value.select.return_value.eq.return_value.execute.return_value = response

        ltm = self._build(client)
        ltm.load()

        assert ltm.operational_heuristics == {"h1": "v1"}
        assert ltm.ticker_insights == {"NVDA": {"sector": "Tech"}}
        assert ltm.user_preferences == {"report_format": "concise"}

    def test_load_no_existing_row_starts_empty(self):
        client = MagicMock()
        response = MagicMock()
        response.data = []
        client.table.return_value.select.return_value.eq.return_value.execute.return_value = response

        ltm = self._build(client)
        ltm.load()

        assert ltm.operational_heuristics == {}
        assert ltm.ticker_insights == {}

    def test_load_exception_resets_to_empty_without_raising(self):
        client = MagicMock()
        client.table.return_value.select.return_value.eq.return_value.execute.side_effect = (
            RuntimeError("supabase down")
        )

        ltm = self._build(client)
        ltm.load()  # must not raise

        assert ltm.operational_heuristics == {}
        assert ltm.ticker_insights == {}
        assert ltm.user_preferences == {}

    def test_load_filters_by_correct_user_id(self):
        client = MagicMock()
        response = MagicMock()
        response.data = []
        client.table.return_value.select.return_value.eq.return_value.execute.return_value = response

        ltm = LongTermMemory(user_id="specific_user", supabase_client=client)
        ltm.load()

        client.table.return_value.select.return_value.eq.assert_called_once_with(
            "user_id", "specific_user"
        )


# ---------------------------------------------------------------------------
# LongTermMemory — persist()
# ---------------------------------------------------------------------------

class TestLongTermMemoryPersist:
    def test_persist_upserts_with_correct_payload_and_conflict_key(self):
        client = MagicMock()
        ltm = LongTermMemory(user_id="u1", supabase_client=client)
        ltm.operational_heuristics = {"h": 1}
        ltm.ticker_insights = {"NVDA": {}}
        ltm.user_preferences = {"p": 1}

        ltm.persist()

        client.table.assert_called_with("long_term_memory")
        _, kwargs = client.table.return_value.upsert.call_args
        assert kwargs["on_conflict"] == "user_id"
        payload = client.table.return_value.upsert.call_args[0][0]
        assert payload["user_id"] == "u1"
        assert payload["operational_heuristics"] == {"h": 1}

    def test_persist_exception_propagates(self):
        client = MagicMock()
        client.table.return_value.upsert.return_value.execute.side_effect = RuntimeError("down")
        ltm = LongTermMemory(user_id="u1", supabase_client=client)
        with pytest.raises(RuntimeError):
            ltm.persist()


# ---------------------------------------------------------------------------
# LongTermMemory — operational heuristics (capped FIFO)
# ---------------------------------------------------------------------------

class TestHeuristics:
    def _build(self):
        return LongTermMemory(user_id="u1", supabase_client=MagicMock())

    def test_store_and_get_heuristic(self):
        ltm = self._build()
        ltm.store_heuristic("k1", "v1")
        assert ltm.get_heuristic("k1") == "v1"

    def test_get_missing_heuristic_returns_default(self):
        ltm = self._build()
        assert ltm.get_heuristic("missing", default="fallback") == "fallback"

    def test_eviction_at_cap_removes_oldest(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock(), max_heuristics=2)
        ltm.store_heuristic("k1", "v1")
        ltm.store_heuristic("k2", "v2")
        ltm.store_heuristic("k3", "v3")  # should evict k1
        assert ltm.get_heuristic("k1") is None
        assert ltm.get_heuristic("k3") == "v3"
        assert len(ltm.get_all_heuristics()) == 2

    def test_updating_existing_key_does_not_trigger_eviction(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock(), max_heuristics=2)
        ltm.store_heuristic("k1", "v1")
        ltm.store_heuristic("k2", "v2")
        ltm.store_heuristic("k1", "updated")  # update existing, at cap but key exists
        assert len(ltm.get_all_heuristics()) == 2
        assert ltm.get_heuristic("k1") == "updated"

    def test_get_all_heuristics_returns_copy(self):
        ltm = self._build()
        ltm.store_heuristic("k1", "v1")
        copy = ltm.get_all_heuristics()
        copy["k2"] = "injected"
        assert "k2" not in ltm.get_all_heuristics()


# ---------------------------------------------------------------------------
# LongTermMemory — ticker insights (capped, merge-upsert)
# ---------------------------------------------------------------------------

class TestTickerInsights:
    def _build(self, **kw):
        return LongTermMemory(user_id="u1", supabase_client=MagicMock(), **kw)

    def test_store_and_get_ticker_insight(self):
        ltm = self._build()
        ltm.store_ticker_insight("nvda", {"sector": "Tech"})
        result = ltm.get_ticker_insight("NVDA")
        assert result["sector"] == "Tech"

    def test_ticker_is_uppercased(self):
        ltm = self._build()
        ltm.store_ticker_insight("nvda", {"a": 1})
        assert "NVDA" in ltm.ticker_insights
        assert "nvda" not in ltm.ticker_insights

    def test_repeated_calls_merge_not_overwrite(self):
        ltm = self._build()
        ltm.store_ticker_insight("NVDA", {"sector": "Tech"})
        ltm.store_ticker_insight("NVDA", {"last_grade": "A"})
        result = ltm.get_ticker_insight("NVDA")
        assert result["sector"] == "Tech"
        assert result["last_grade"] == "A"

    def test_last_updated_timestamp_added(self):
        ltm = self._build()
        ltm.store_ticker_insight("NVDA", {"sector": "Tech"})
        assert "last_updated" in ltm.get_ticker_insight("NVDA")

    def test_get_unknown_ticker_returns_empty_dict(self):
        ltm = self._build()
        assert ltm.get_ticker_insight("ZZZZ") == {}

    def test_eviction_at_cap_removes_oldest_ticker(self):
        ltm = self._build(max_ticker_insights=2)
        ltm.store_ticker_insight("NVDA", {"a": 1})
        ltm.store_ticker_insight("AAPL", {"a": 1})
        ltm.store_ticker_insight("MSFT", {"a": 1})  # evicts NVDA
        assert ltm.get_ticker_insight("NVDA") == {}
        assert "MSFT" in ltm.ticker_insights


# ---------------------------------------------------------------------------
# LongTermMemory — user preferences
# ---------------------------------------------------------------------------

class TestPreferences:
    def test_store_and_get_preference(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock())
        ltm.store_preference("report_format", "concise")
        assert ltm.get_preference("report_format") == "concise"

    def test_get_missing_preference_returns_default(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock())
        assert ltm.get_preference("missing", default="x") == "x"

    def test_get_all_preferences_returns_copy(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock())
        ltm.store_preference("k", "v")
        copy = ltm.get_all_preferences()
        copy["injected"] = True
        assert "injected" not in ltm.get_all_preferences()


# ---------------------------------------------------------------------------
# LongTermMemory — recall()
# ---------------------------------------------------------------------------

class TestLongTermRecall:
    def test_recall_without_ticker_omits_ticker_insight_key(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock())
        result = ltm.recall()
        assert "ticker_insight" not in result
        assert "heuristics" in result
        assert "total_tickers_cached" in result

    def test_recall_with_ticker_includes_insight(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock())
        ltm.store_ticker_insight("NVDA", {"sector": "Tech"})
        result = ltm.recall(ticker="NVDA")
        assert result["ticker_insight"]["sector"] == "Tech"

    def test_total_tickers_cached_reflects_count(self):
        ltm = LongTermMemory(user_id="u1", supabase_client=MagicMock())
        ltm.store_ticker_insight("NVDA", {})
        ltm.store_ticker_insight("AAPL", {})
        result = ltm.recall()
        assert result["total_tickers_cached"] == 2


# ---------------------------------------------------------------------------
# ManagerMemory — unified facade (delegation correctness)
# ---------------------------------------------------------------------------

class TestManagerMemoryFacade:
    @pytest.fixture
    def manager(self):
        client = MagicMock()
        response = MagicMock()
        response.data = []
        client.table.return_value.select.return_value.eq.return_value.execute.return_value = response
        return ManagerMemory(user_id="u1", supabase_client=client)

    def test_new_session_delegates_to_short_term_reset(self, manager):
        manager.new_session(session_id="s1", task_query="Analyse NVDA")
        assert manager.short.session_id == "s1"

    def test_add_message_delegates_to_short_term(self, manager):
        manager.add_message(role="user", content="hi")
        assert manager.get_messages() == [{"role": "user", "content": "hi"}]

    def test_log_dispatch_returns_record_from_short_term(self, manager):
        record = manager.log_dispatch("ResearchAgent", {"ticker": "NVDA"})
        assert isinstance(record, AgentExecutionRecord)
        assert manager.agents_run() == ["ResearchAgent"]

    def test_add_evaluation_and_get_last_delegate_correctly(self, manager):
        ev = make_eval()
        manager.add_evaluation(ev)
        assert manager.get_last_evaluation() is ev

    def test_long_term_delegation_heuristics(self, manager):
        manager.store_heuristic("k", "v")
        assert manager.get_heuristic("k") == "v"

    def test_long_term_delegation_ticker_insight(self, manager):
        manager.store_ticker_insight("NVDA", {"sector": "Tech"})
        assert manager.get_ticker_insight("NVDA")["sector"] == "Tech"

    def test_long_term_delegation_preference(self, manager):
        manager.store_preference("fmt", "concise")
        assert manager.get_preference("fmt") == "concise"

    def test_persist_long_term_calls_long_persist(self, manager):
        manager.long.persist = MagicMock()
        manager.persist_long_term()
        manager.long.persist.assert_called_once()

    def test_recall_merges_short_and_long_term(self, manager):
        manager.new_session(session_id="s1", task_query="q")
        manager.store_heuristic("k", "v")
        result = manager.recall()
        assert "short_term" in result
        assert "long_term" in result
        assert result["short_term"]["session_id"] == "s1"
        assert result["long_term"]["heuristics"]["k"] == "v"

    def test_recall_with_ticker_passes_through_to_long_term(self, manager):
        manager.store_ticker_insight("NVDA", {"sector": "Tech"})
        result = manager.recall(ticker="NVDA")
        assert result["long_term"]["ticker_insight"]["sector"] == "Tech"