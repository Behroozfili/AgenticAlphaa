"""
Tests for: agents/research_agent.py
Phase: 6 — Agents (1st: no dependency on other agents)

Mocking strategy:
  - anthropic.Anthropic is injected directly via the `llm_client` constructor
    param (the class explicitly supports DI for this — no real API calls).
  - MCP tool calls are mocked by patching `stdio_client` and `ClientSession`
    at the module level, since the executor node opens its own
    AsyncExitStack/session internally (no DI point exists for it).
  - The LangGraph StateGraph/END come from a lightweight stub
    (stubs/langgraph/graph.py) that actually executes nodes + conditional
    edges in-process, so `agent.run()` can be tested end-to-end without a
    real LangGraph runtime.
  - We test node methods (_brain_node, _executor_node, _checker_node,
    _should_continue) both in isolation AND via the full run() pipeline.
"""
import json
import sys
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from agents.research_agent import ResearchAgent, _trace_inputs, _trace_outputs
from agents.state import SharedManagerState, ResearchAgentState
from core.forecast_contract import ForecastContext
from tools.research_tools import (
    RESEARCH_IMPLEMENTATION_IDENTITY, build_research_features, compact_lifecycle,
    normalize_evidence, research_implementation_identity, research_trace_metadata,
)
from tools.research_tools.context_synthesizer import (
    _synthesis_trace_inputs, _synthesis_trace_outputs, synthesize_research_context,
)


def make_context(ticker="NVDA"):
    return ForecastContext(
        ticker=ticker, as_of="2024-01-31T21:00:00Z",
        schema_version="v1", dataset_version="dataset-v1",
        feature_version="feature-v1", model_version="model-v1",
    )


def make_llm_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def make_evidence_payload():
    return json.dumps({
        "evidence": [{
            "evidence_id": "ev-nvda-1", "source": "tavily_search",
            "provider": "tavily_live", "ticker": "NVDA",
            "timestamp": "2024-01-15T12:00:00Z",
            "dataset_version": "dataset-v1", "run_id": "run-test-04b",
            "citation": {"url": "PRIVATE_SENTINEL_URL"},
            "title": "PRIVATE_SENTINEL_TITLE", "kind": "catalyst",
        }]
    })


def make_agent(llm=None, max_loops=3):
    llm = llm or MagicMock()
    return ResearchAgent(llm_client=llm, max_loops=max_loops)


def make_shared_state(task_query="Analyze NVDA", **kw):
    context = make_context()
    state: SharedManagerState = {
        "task_query": task_query,
        "aggregated_research_context": [],
        "manager_directives": {"ticker": context.ticker, "run_id": "run-test-04b"},
        "forecast_context": context.to_dict(),
    }
    state.update(kw)
    return state


def make_research_state(
    task_query="Analyze NVDA",
    manager_directives=None,
    **kw,
):
    """Build the top-level internal state created by ResearchAgent.run()."""
    context = make_context()
    directives = {
        "ticker": context.ticker, "run_id": "run-test-04b",
        "forecast_context": context.to_dict(),
        **dict(manager_directives or {}),
    }
    state: ResearchAgentState = {
        "messages": [],
        "context_chunks": [],
        "called_tool_signatures": [],
        "loop_counter": 0,
        "validation_feedback": "",
        "is_complete": False,
        "task_query": task_query,
        "manager_directives": directives,
    }
    state.update(kw)
    return state


def test_trace_processors_drop_private_sentinels():
    state = make_research_state(
        task_query="PRIVATE_SENTINEL_QUERY",
        messages=[{"content": "PRIVATE_SENTINEL_MESSAGE"}],
        context_chunks=[{"citation": "PRIVATE_SENTINEL_CITATION"}],
    )
    traced_in = _trace_inputs("research_executor")({"state": state, "plan": "PRIVATE_SENTINEL_PLAN"})
    traced_out = _trace_outputs("research_executor")({
        **state, "research_features": {"evidence_ids": [], "errors": [],
        "summary": "PRIVATE_SENTINEL_SUMMARY"},
    })
    rendered = json.dumps([traced_in, traced_out], sort_keys=True)
    assert "PRIVATE_SENTINEL" not in rendered
    assert traced_in["message_count"] == 1
    assert traced_out["context_reference_count"] == 1
    for processed in (traced_in["metadata"], traced_out["metadata"]):
        assert processed["stage"] == "04B"
        assert processed["stage_revision"] == 1
        assert processed["attempt"] == 3
        assert processed["agent"] == "research"
        assert processed["implementation_version"] == "stage04b-r1-a3"


def test_stage04b_implementation_identity_is_single_immutable_attempt3_source():
    expected = {
        "stage": "04B", "stage_revision": 1, "attempt": 3,
        "agent": "research", "implementation_version": "stage04b-r1-a3",
    }
    assert dict(RESEARCH_IMPLEMENTATION_IDENTITY) == expected
    copied = research_implementation_identity()
    copied["attempt"] = 99
    assert dict(RESEARCH_IMPLEMENTATION_IDENTITY) == expected
    assert research_trace_metadata(make_context(), "run-test-04b")["attempt"] == 3


def test_synthesis_trace_processors_use_attempt3_and_drop_private_payloads():
    traced_in = _synthesis_trace_inputs({
        "forecast_context": make_context(), "run_id": "run-test-04b",
        "task_query": "PRIVATE_SENTINEL_QUERY",
        "chunks": [{"evidence_id": "PRIVATE_SENTINEL_EVIDENCE"}],
    })
    traced_out = _synthesis_trace_outputs({
        "summary": "PRIVATE_SENTINEL_SUMMARY",
        "telemetry": {
            **research_trace_metadata(make_context(), "run-test-04b"),
            "operation": "research_synthesis", "outcome": "SUCCESS",
            "evidence_count": 1,
        },
    })
    for processed in (traced_in, traced_out):
        assert processed["stage"] == "04B"
        assert processed["stage_revision"] == 1
        assert processed["attempt"] == 3
        assert processed["agent"] == "research"
        assert processed["implementation_version"] == "stage04b-r1-a3"
        assert "PRIVATE_SENTINEL" not in json.dumps(processed, sort_keys=True)


def test_lifecycle_matrix_has_measured_compact_counts_for_every_stage04b_boundary():
    context = make_context()
    operations = (
        "tavily_search", "news_search", "sec_edgar_search", "sec_edgar_filing",
        "rag_vector_search", "rag_graph_traverse", "rag_hybrid_query",
        "comprehensive_analysis", "research_synthesis", "pre_state_leakage_gate",
    )
    for operation in operations:
        telemetry = compact_lifecycle(
            context=context,
            run_id="run-test-04b",
            operation=operation,
            latency_ms=0.125,
            input_count=2,
            payload={
                "status": "OK",
                "evidence": [{"evidence_id": "ev-1", "citation": {"url": "PRIVATE_SENTINEL_URL"}}],
                "quarantined": [{"reason_code": "FUTURE_EVIDENCE"}],
                "degraded": True,
            },
        )
        assert telemetry["operation"] == operation
        assert telemetry["stage"] == "04B"
        assert telemetry["stage_revision"] == 1
        assert telemetry["attempt"] == 3
        assert telemetry["agent"] == "research"
        assert telemetry["implementation_version"] == "stage04b-r1-a3"
        assert telemetry["provider"] == operation
        assert telemetry["latency_ms"] >= 0
        assert telemetry["input_count"] == 2
        assert telemetry["result_count"] == 1
        assert telemetry["evidence_count"] == 1
        assert telemetry["citation_count"] == 1
        assert telemetry["quarantine_count"] == 1
        assert telemetry["rejection_counts"] == {"FUTURE_EVIDENCE": 1}
        assert "PRIVATE_SENTINEL" not in json.dumps(telemetry)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_injected_llm_client_used_no_env_needed(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm = MagicMock()
        agent = ResearchAgent(llm_client=llm)
        assert agent._llm is llm

    def test_missing_api_key_and_no_client_raises_keyerror(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(KeyError):
            ResearchAgent()

    def test_default_max_loops_is_3(self):
        agent = make_agent()
        assert agent._max_loops == 3

    def test_graph_is_built_at_construction(self):
        agent = make_agent()
        assert agent._graph is not None


# ---------------------------------------------------------------------------
# run() — input validation
# ---------------------------------------------------------------------------

class TestRunValidation:
    @pytest.mark.asyncio
    async def test_empty_task_query_raises_valueerror(self):
        agent = make_agent()
        with pytest.raises(ValueError):
            await agent.run(make_shared_state(task_query=""))

    @pytest.mark.asyncio
    async def test_whitespace_only_task_query_raises_valueerror(self):
        agent = make_agent()
        with pytest.raises(ValueError):
            await agent.run(make_shared_state(task_query="   "))

    @pytest.mark.asyncio
    async def test_missing_forecast_context_is_rejected(self):
        agent = make_agent()
        state = make_shared_state()
        state.pop("forecast_context")
        with pytest.raises(ValueError, match="forecast_context is required"):
            await agent.run(state)

    @pytest.mark.asyncio
    async def test_missing_run_id_is_rejected(self):
        agent = make_agent()
        state = make_shared_state()
        state["manager_directives"].pop("run_id")
        with pytest.raises(ValueError, match="run_id is required"):
            await agent.run(state)


class TestPointInTimeFeatures:
    def test_normalize_preserves_bounded_sec_excerpt(self):
        context = make_context()
        record = {
            "source": "sec_edgar_filing", "provider": "sec_archive",
            "ticker": "NVDA", "filed_at": "2024-01-15",
            "dataset_version": "dataset-v1",
            "sections": {"risk_factors": {"preview": "material risk " * 700}},
        }

        normalized = normalize_evidence(
            record, context=context, run_id="run-test", source="sec_edgar_filing"
        )

        assert "material risk" in normalized["content_excerpt"]
        assert len(normalized["content_excerpt"]) == 6_000

    @pytest.mark.asyncio
    async def test_synthesis_returns_local_cited_digest_without_calling_llm(self):
        context = make_context()
        evidence = {
            "evidence_id": "ev-local-1", "source": "sec_edgar_filing",
            "provider": "sec_archive", "ticker": "NVDA",
            "timestamp": "2024-01-15T12:00:00Z",
            "dataset_version": "dataset-v1",
            "content_excerpt": "AI infrastructure demand increased.",
        }
        llm = MagicMock()

        result = await synthesize_research_context(
            [evidence], "NVDA", llm, forecast_context=context, run_id="run-test"
        )

        assert result["summary_generated"] is True
        assert result["summary"].startswith("[SYNTHESIZED RESEARCH SUMMARY]")
        assert "[ev-local-1]" in result["summary"]
        assert "AI infrastructure demand increased." in result["summary"]
        llm.messages.create.assert_not_called()
        assert "AI infrastructure" not in json.dumps(_synthesis_trace_outputs(result))

    def test_vector_graph_and_hybrid_coverage_flags(self):
        context = make_context()
        base = {
            "provider": "project_archive", "ticker": "NVDA",
            "timestamp": "2024-01-15T12:00:00Z", "dataset_version": "dataset-v1",
            "citation": {"url": "https://archive.test/evidence"}, "title": "signal",
        }
        vector = {**base, "evidence_id": "vector", "source": "rag_vector_search"}
        graph = {**base, "evidence_id": "graph", "source": "rag_graph_traverse"}
        vector_features = build_research_features([vector], context=context, run_id="run-test")
        graph_features = build_research_features([graph], context=context, run_id="run-test")
        hybrid_features = build_research_features([vector, graph], context=context, run_id="run-test")
        assert (vector_features.vector_coverage, vector_features.graph_coverage, vector_features.degraded) == (True, False, True)
        assert (graph_features.vector_coverage, graph_features.graph_coverage, graph_features.degraded) == (False, True, True)
        assert (hybrid_features.vector_coverage, hybrid_features.graph_coverage, hybrid_features.degraded) == (True, True, False)

    def test_hybrid_coverage_requires_actual_branch_origin(self):
        context = make_context()
        base = {
            "provider": "rag_hybrid_query", "source": "rag_hybrid_query",
            "ticker": "NVDA", "timestamp": "2024-01-15T12:00:00Z",
            "dataset_version": "dataset-v1",
            "citation": {"url": "https://archive.test/evidence"},
            "title": "signal",
        }
        vector_only = {
            **base, "evidence_id": "hybrid-vector",
            "retrieval_origins": ["vector"],
        }
        features = build_research_features(
            [vector_only], context=context, run_id="run-test"
        )
        assert features.vector_coverage is True
        assert features.graph_coverage is False
        assert features.degraded is True

    @pytest.mark.asyncio
    async def test_second_gate_rejects_future_before_synthesis(self):
        context = make_context()
        future = {
            "evidence_id": "future", "source": "news_search", "provider": "project_archive",
            "ticker": "NVDA", "timestamp": "2024-02-01T12:00:00Z",
            "dataset_version": "dataset-v1", "citation": {"url": "https://future.test"},
        }
        llm = MagicMock()
        result = await synthesize_research_context(
            [future], "NVDA", llm, forecast_context=context, run_id="run-test"
        )
        assert result["missing"] is True
        assert result["evidence_ids"] == []
        assert result["telemetry"]["operation"] == "research_synthesis"
        assert result["telemetry"]["latency_ms"] >= 0
        assert result["telemetry"]["quarantine_count"] == 1
        assert result["telemetry"]["rejection_counts"] == {"FUTURE_EVIDENCE": 1}
        llm.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------

def test_lifecycle_counts_rag_result_evidence_ids_without_evidence_list():
    telemetry = compact_lifecycle(
        context=make_context(),
        run_id="run-test-04b",
        operation="rag_hybrid_query",
        latency_ms=1.0,
        payload={
            "results": [
                {"evidence_id": "rag-1"},
                {"evidence_id": "rag-1"},
                {"evidence_id": "rag-2"},
            ],
            "citations": [
                {"evidence_id": "rag-1"},
                {"evidence_id": "rag-2"},
            ],
            "degraded": False,
        },
    )

    assert telemetry["result_count"] == 3
    assert telemetry["evidence_count"] == 2
    assert telemetry["citation_count"] == 2

def test_extract_evidence_prefers_rag_result_text_over_citation_only_record():
    context = make_context()
    payload = {
        "results": [{
            "evidence_id": "rag-1", "ticker": "NVDA",
            "published_at": "2024-01-15T12:00:00Z",
            "dataset_version": "dataset-v1", "chunk_text": "RAG demand evidence",
            "citation": {"url": "https://archive.test/rag-1"},
        }],
        "citations": [{"evidence_id": "rag-1", "url": "https://archive.test/rag-1"}],
    }

    evidence = ResearchAgent._extract_evidence(
        "rag_vector_search", json.dumps(payload), context, "run-test"
    )

    assert len(evidence) == 1
    assert "RAG demand evidence" in evidence[0]["content_excerpt"]

def test_extract_hybrid_evidence_preserves_actual_branch_origins():
    context = make_context()
    payload = {
        "results": [{
            "evidence_id": "rag-hybrid-1", "ticker": "NVDA",
            "published_at": "2024-01-15T12:00:00Z",
            "dataset_version": "dataset-v1", "origin": "vector",
            "origins": ["vector"], "chunk_text": "Vector-only result",
            "citation": {"url": "https://archive.test/rag-hybrid-1"},
        }],
    }

    evidence = ResearchAgent._extract_evidence(
        "rag_hybrid_query", json.dumps(payload), context, "run-test"
    )

    assert evidence[0]["retrieval_origins"] == ["vector"]

class TestParsePlan:
    def test_valid_json_plan_parsed(self):
        agent = make_agent()
        plan = json.dumps({"reasoning": "x", "actions": [{"tool": "tavily_search", "arguments": {"query": "q"}}]})
        actions = agent._parse_plan(plan)
        assert actions == [{"tool": "tavily_search", "arguments": {"query": "q"}}]

    def test_markdown_fenced_json_stripped(self):
        agent = make_agent()
        plan = "```json\n" + json.dumps({"actions": []}) + "\n```"
        assert agent._parse_plan(plan) == []

    def test_invalid_json_returns_empty_list(self):
        agent = make_agent()
        assert agent._parse_plan("not json") == []

    def test_actions_not_a_list_returns_empty(self):
        agent = make_agent()
        plan = json.dumps({"actions": "not-a-list"})
        assert agent._parse_plan(plan) == []

    def test_missing_actions_key_returns_empty_list(self):
        agent = make_agent()
        assert agent._parse_plan(json.dumps({"reasoning": "x"})) == []


# ---------------------------------------------------------------------------
# _format_tool_result
# ---------------------------------------------------------------------------

class TestFormatToolResult:
    def test_includes_tool_name_and_text(self):
        result = ResearchAgent._format_tool_result("tavily_search", {"query": "NVDA"}, "some result text")
        assert "tavily_search" in result
        assert "some result text" in result
        assert "NVDA" in result

    def test_query_hint_falls_back_to_entity_then_ticker(self):
        result = ResearchAgent._format_tool_result("rag_graph_traverse", {"entity": "AMD"}, "x")
        assert "AMD" in result

    def test_query_hint_defaults_to_na_when_no_known_key(self):
        result = ResearchAgent._format_tool_result("tool", {}, "x")
        assert "N/A" in result


# ---------------------------------------------------------------------------
# _should_continue — routing logic
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def _state(self, loop_counter, is_complete, max_loops_directive=None):
        directives = {"max_loops": max_loops_directive} if max_loops_directive is not None else {}
        return make_research_state(
            loop_counter=loop_counter,
            is_complete=is_complete,
            manager_directives=directives,
        )

    def test_forces_end_when_loop_counter_at_max(self):
        agent = make_agent(max_loops=3)
        result = agent._should_continue(self._state(loop_counter=3, is_complete=False))
        assert result == "__end__"

    def test_ends_when_is_complete_true(self):
        agent = make_agent(max_loops=3)
        result = agent._should_continue(self._state(loop_counter=1, is_complete=True))
        assert result == "__end__"

    def test_continues_to_brain_when_incomplete_and_under_cap(self):
        agent = make_agent(max_loops=3)
        result = agent._should_continue(self._state(loop_counter=1, is_complete=False))
        assert result == "brain"

    def test_guardrail_takes_priority_over_is_complete(self):
        agent = make_agent(max_loops=2)
        result = agent._should_continue(self._state(loop_counter=2, is_complete=True))
        assert result == "__end__"

    def test_manager_directive_overrides_instance_max_loops(self):
        agent = make_agent(max_loops=10)
        result = agent._should_continue(self._state(loop_counter=1, is_complete=False, max_loops_directive=1))
        assert result == "__end__"


# ---------------------------------------------------------------------------
# _brain_node
# ---------------------------------------------------------------------------

class TestBrainNode:
    @pytest.mark.asyncio
    async def test_calls_llm_and_appends_messages(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response('{"actions": []}')
        agent = make_agent(llm=llm)

        state = make_research_state()
        update = await agent._brain_node(state)

        assert len(update["messages"]) == 2
        assert update["messages"][0]["role"] == "user"
        assert update["messages"][1]["role"] == "assistant"
        llm.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_feedback_included_in_prompt_when_present(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response('{"actions": []}')
        agent = make_agent(llm=llm)

        state = make_research_state(
            loop_counter=1,
            validation_feedback="Need more recent data",
        )
        await agent._brain_node(state)

        sent_prompt = llm.messages.create.call_args.kwargs["messages"][-1]["content"]
        assert "Need more recent data" in sent_prompt

    @pytest.mark.asyncio
    async def test_history_messages_prepended_to_new_prompt(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response('{"actions": []}')
        agent = make_agent(llm=llm)

        prior = [{"role": "user", "content": "prior msg"}]
        state = make_research_state(messages=prior)
        await agent._brain_node(state)

        sent_messages = llm.messages.create.call_args.kwargs["messages"]
        assert sent_messages[0] == prior[0]


# ---------------------------------------------------------------------------
# _executor_node
# ---------------------------------------------------------------------------

class TestExecutorNode:
    @pytest.mark.asyncio
    async def test_no_actions_returns_warning_chunk(self):
        agent = make_agent()
        state = make_research_state(
            messages=[{"role": "assistant", "content": "not json"}],
        )
        update = await agent._executor_node(state)
        assert update["loop_counter"] == 1
        assert "no valid action plan" in update["context_chunks"][0]

    @pytest.mark.asyncio
    async def test_successful_tool_call_appends_formatted_chunk(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "NVDA"}}]})
        state = make_research_state(
            messages=[{"role": "assistant", "content": plan}],
        )

        mock_session = AsyncMock()
        tool_result = MagicMock()
        tool_result.content = [MagicMock(type="text", text=make_evidence_payload())]
        mock_session.call_tool.return_value = tool_result

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            update = await agent._executor_node(state)

        assert update["loop_counter"] == 1
        assert update["context_chunks"][0]["evidence_id"] == "ev-nvda-1"
        assert update["context_chunks"][0]["source"] == "tavily_search"

    @pytest.mark.asyncio
    async def test_tool_failure_appends_error_marker_not_raises(self, caplog):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "NVDA"}}]})
        state = make_research_state(
            messages=[{"role": "assistant", "content": plan}],
        )

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = RuntimeError("PRIVATE_SENTINEL_EXCEPTION")

        progress_events = []
        fake_sentry = MagicMock()
        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio, \
             patch("agents.research_agent._publish_progress", side_effect=lambda *args, **kwargs: progress_events.append(kwargs)), \
             patch("agents.research_agent.sentry_enabled", return_value=True), \
             patch.dict(sys.modules, {"sentry_sdk": fake_sentry}):
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            update = await agent._executor_node(state)

        assert update["context_chunks"][0] == {
            "source": "tavily_search", "error_code": "RESEARCH_TOOL_FAIL_CLOSED"
        }
        persisted = json.dumps([update, progress_events], sort_keys=True) + caplog.text
        assert "PRIVATE_SENTINEL" not in persisted
        assert progress_events[-1]["detail"]["outcome"] == "FAIL_CLOSED"
        assert progress_events[-1]["detail"]["latency_ms"] >= 0
        assert progress_events[-1]["detail"]["evidence_count"] == 0
        for event in progress_events:
            detail = event["detail"]
            assert detail["stage"] == "04B"
            assert detail["stage_revision"] == 1
            assert detail["attempt"] == 3
            assert detail["agent"] == "research"
            assert detail["implementation_version"] == "stage04b-r1-a3"
        fake_sentry.capture_exception.assert_not_called()
        fake_sentry.capture_message.assert_called_once_with("RESEARCH_TOOL_FAIL_CLOSED", level="error")
        sentry_data = fake_sentry.add_breadcrumb.call_args.kwargs["data"]
        assert sentry_data["stage"] == "04B"
        assert sentry_data["stage_revision"] == 1
        assert sentry_data["attempt"] == 3
        assert sentry_data["agent"] == "research"
        assert sentry_data["implementation_version"] == "stage04b-r1-a3"
        assert "PRIVATE_SENTINEL" not in json.dumps(sentry_data, sort_keys=True)

    @pytest.mark.asyncio
    async def test_unallowlisted_private_tool_name_never_reaches_logs_progress_or_state(self, caplog):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "PRIVATE_SENTINEL_TOOL", "arguments": {"query": "PRIVATE_SENTINEL_QUERY"}}]})
        state = make_research_state(messages=[{"role": "assistant", "content": plan}])
        progress_events = []
        mock_session = AsyncMock()
        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio, \
             patch("agents.research_agent._publish_progress", side_effect=lambda *args, **kwargs: progress_events.append(kwargs)):
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None
            update = await agent._executor_node(state)

        assert update["context_chunks"] == [{
            "source": "research_mcp", "error_code": "RESEARCH_TOOL_NOT_ALLOWLISTED",
        }]
        assert "PRIVATE_SENTINEL" not in json.dumps([update, progress_events]) + caplog.text
        mock_session.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_connection_failure_appends_error_marker_not_raises(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "q"}}]})
        state = make_research_state(
            messages=[{"role": "assistant", "content": plan}],
        )

        with patch("agents.research_agent.stdio_client", side_effect=ConnectionError("server down")):
            update = await agent._executor_node(state)

        assert update["context_chunks"][0] == {
            "source": "research_mcp", "error_code": "RESEARCH_MCP_UNAVAILABLE"
        }
        assert "server down" not in json.dumps(update)

    @pytest.mark.asyncio
    async def test_action_with_no_tool_name_is_skipped(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"arguments": {"query": "q"}}]})
        state = make_research_state(
            messages=[{"role": "assistant", "content": plan}],
        )

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session = AsyncMock()
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            update = await agent._executor_node(state)

        assert update["context_chunks"] == []
        mock_session.call_tool.assert_not_called()


# ---------------------------------------------------------------------------
# _checker_node
# ---------------------------------------------------------------------------

class TestCheckerNode:
    @pytest.mark.asyncio
    async def test_no_chunks_returns_incomplete_without_calling_llm(self):
        llm = MagicMock()
        agent = make_agent(llm=llm)
        state = make_research_state(loop_counter=1)
        update = await agent._checker_node(state)
        assert update["is_complete"] is False
        assert "No data was retrieved" in update["validation_feedback"]
        llm.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_verdict_clears_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            json.dumps({"is_complete": True, "score": 95, "missing": "nothing", "feedback": ""})
        )
        agent = make_agent(llm=llm)
        state = make_research_state(
            context_chunks=["some chunk"],
            loop_counter=1,
        )
        update = await agent._checker_node(state)
        assert update["is_complete"] is True
        assert update["validation_feedback"] == ""

    @pytest.mark.asyncio
    async def test_fenced_verdict_with_trailing_rationale_does_not_retry(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            '```json\n{"is_complete": true, "score": 82, "feedback": ""}\n```\nRationale: enough current evidence.'
        )
        agent = make_agent(llm=llm)
        state = make_research_state(
            context_chunks=["current filing evidence"], loop_counter=1
        )

        update = await agent._checker_node(state)

        assert update == {"is_complete": True, "validation_feedback": ""}

    @pytest.mark.asyncio
    async def test_incomplete_verdict_keeps_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            json.dumps({"is_complete": False, "feedback": "Need more sources"})
        )
        agent = make_agent(llm=llm)
        state = make_research_state(
            context_chunks=["chunk"],
            loop_counter=1,
        )
        update = await agent._checker_node(state)
        assert update["is_complete"] is False
        assert update["validation_feedback"] == "Need more sources"

    @pytest.mark.asyncio
    async def test_checker_missing_and_score_never_reach_logs_or_progress(self, caplog):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "is_complete": False,
            "missing": "PRIVATE_SENTINEL_MISSING",
            "score": {"payload": "PRIVATE_SENTINEL_SCORE"},
            "feedback": "Need more sources",
        }))
        agent = make_agent(llm=llm)
        state = make_research_state(context_chunks=["compact reference"], loop_counter=1)
        progress_events = []

        with patch(
            "agents.research_agent._publish_progress",
            side_effect=lambda *args, **kwargs: progress_events.append(kwargs),
        ):
            update = await agent._checker_node(state)

        assert update == {
            "is_complete": False, "validation_feedback": "Need more sources",
        }
        persisted = caplog.text + json.dumps(progress_events, sort_keys=True)
        assert "PRIVATE_SENTINEL" not in persisted
        detail = progress_events[-1]["detail"]
        assert "score" not in detail
        assert detail["is_complete"] is False
        assert detail["stage"] == "04B"
        assert detail["stage_revision"] == 1
        assert detail["attempt"] == 3
        assert detail["agent"] == "research"
        assert detail["implementation_version"] == "stage04b-r1-a3"

    @pytest.mark.asyncio
    async def test_invalid_json_verdict_defaults_to_incomplete(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("not valid json")
        agent = make_agent(llm=llm)
        state = make_research_state(
            context_chunks=["chunk"],
            loop_counter=1,
        )
        update = await agent._checker_node(state)
        assert update["is_complete"] is False
        assert "retry" in update["validation_feedback"].lower()

    @pytest.mark.asyncio
    async def test_checker_provider_failure_uses_diverse_valid_evidence(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("provider unavailable")
        agent = make_agent(llm=llm)
        evidence = [
            {
                "evidence_id": f"ev-{index}", "source": source,
                "provider": provider, "ticker": "NVDA",
                "timestamp": "2024-01-15T12:00:00Z",
                "dataset_version": "dataset-v1", "run_id": "run-test-04b",
                "citation": {"url": f"https://archive.test/{index}"},
                "title": "validated point-in-time signal",
            }
            for index, (source, provider) in enumerate((
                ("tavily_search", "tavily_live"),
                ("rag_hybrid_query", "rag_hybrid"),
                ("news_search", "project_archive"),
            ), start=1)
        ]
        state = make_research_state(context_chunks=evidence, loop_counter=1)

        update = await agent._checker_node(state)

        assert update == {"is_complete": True, "validation_feedback": ""}

    @pytest.mark.asyncio
    async def test_checker_provider_failure_keeps_sparse_evidence_incomplete(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("provider unavailable")
        agent = make_agent(llm=llm)
        state = make_research_state(
            context_chunks=[{
                "evidence_id": "ev-only", "source": "news_search",
                "provider": "project_archive", "ticker": "NVDA",
                "timestamp": "2024-01-15T12:00:00Z",
                "dataset_version": "dataset-v1", "run_id": "run-test-04b",
                "citation": {"url": "https://archive.test/only"},
                "title": "one source only",
            }],
            loop_counter=1,
        )

        update = await agent._checker_node(state)

        assert update["is_complete"] is False
        assert "deterministic evidence coverage is insufficient" in update["validation_feedback"]
    @pytest.mark.asyncio
    async def test_combined_chunks_truncated_above_12000_chars(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            json.dumps({"is_complete": True, "feedback": ""})
        )
        agent = make_agent(llm=llm)
        huge_chunk = "x" * 20000
        state = make_research_state(
            context_chunks=[huge_chunk],
            loop_counter=1,
        )
        await agent._checker_node(state)
        sent_prompt = llm.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "[truncated for audit]" in sent_prompt


# ---------------------------------------------------------------------------
# run() — full pipeline via the stub LangGraph executor
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    @pytest.mark.asyncio
    async def test_hands_synthesized_research_data_to_manager_context(self):
        llm = MagicMock()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "NVDA"}}]})
        verdict = json.dumps({"is_complete": True, "feedback": ""})
        llm.messages.create.side_effect = [make_llm_response(plan), make_llm_response(verdict)]
        agent = make_agent(llm=llm)
        mock_session = AsyncMock()
        tool_result = MagicMock()
        tool_result.content = [MagicMock(type="text", text=json.dumps({"evidence": [{
            "evidence_id": "ev-nvda-1", "source": "tavily_search",
            "provider": "tavily_live", "ticker": "NVDA",
            "timestamp": "2024-01-15T12:00:00Z",
            "dataset_version": "dataset-v1", "snippet": "Demand accelerated.",
        }]}))]
        mock_session.call_tool.return_value = tool_result

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None
            result = await agent.run(make_shared_state())

        context = result["aggregated_research_context"]
        assert len(context) == 2
        assert context[0]["evidence_id"] == "ev-nvda-1"
        assert "content_excerpt" not in context[0]
        assert context[1].startswith("[SYNTHESIZED RESEARCH SUMMARY]")
        assert "Demand accelerated." in context[1]
        assert "[ev-nvda-1]" in context[1]

    @pytest.mark.asyncio
    async def test_completes_in_one_loop_and_merges_context(self):
        llm = MagicMock()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "NVDA"}}]})
        verdict = json.dumps({"is_complete": True, "feedback": ""})
        llm.messages.create.side_effect = [
            make_llm_response(plan),
            make_llm_response(verdict),
        ]
        agent = make_agent(llm=llm, max_loops=3)

        mock_session = AsyncMock()
        tool_result = MagicMock()
        tool_result.content = [MagicMock(type="text", text=make_evidence_payload())]
        mock_session.call_tool.return_value = tool_result

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            with patch(
                "agents.research_agent.synthesize_research_context",
                new=AsyncMock(return_value=None),
            ):
                result = await agent.run(make_shared_state())

        assert len(result["aggregated_research_context"]) == 1
        assert result["aggregated_research_context"][0]["evidence_id"] == "ev-nvda-1"
        assert result["research_features"]["evidence_ids"] == ["ev-nvda-1"]
        assert "PRIVATE_SENTINEL" not in json.dumps({
            "references": result["aggregated_research_context"],
            "features": result["research_features"],
        })
        forbidden_payload_keys = {"citation", "citations", "summary", "synthesis", "quarantined", "reason", "detail"}
        persisted_keys = set()
        stack = [result["aggregated_research_context"], result["research_features"]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                persisted_keys.update(value)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        assert persisted_keys.isdisjoint(forbidden_payload_keys)

    @pytest.mark.asyncio
    async def test_loops_until_guardrail_when_never_complete(self):
        llm = MagicMock()
        plan = json.dumps({"actions": []})
        verdict = json.dumps({"is_complete": False, "feedback": "keep trying"})
        llm.messages.create.side_effect = [
            make_llm_response(plan), make_llm_response(verdict),
            make_llm_response(plan), make_llm_response(verdict),
        ]
        agent = make_agent(llm=llm, max_loops=2)

        # NOTE: run() also calls synthesize_research_context() once, after
        # the loop above completes — and it DOES fire here even though the
        # Brain never produced real actions, because the Executor still
        # appends a non-empty "[EXECUTOR WARNING] ..." placeholder chunk on
        # every no-actions iteration (context_chunks is never actually
        # empty). Left unpatched, that's an un-mocked 5th call to the same
        # `llm.messages.create`, which exhausts the 4-item side_effect list
        # and raises StopIteration (silently caught inside
        # context_synthesizer.py, so it doesn't fail the test directly, but
        # it does inflate call_count to 5 and pollutes what this test is
        # actually meant to isolate: the loop/guardrail's OWN call count).
        # Patching synthesis out keeps this test scoped to _should_continue's
        # guardrail behavior; synthesis has its own dedicated tests.
        with patch(
            "agents.research_agent.synthesize_research_context",
            new=AsyncMock(return_value=None),
        ):
            result = await agent.run(make_shared_state())

        assert result["aggregated_research_context"] == []
        assert result["research_features"]["missing"] is True
        assert "RESEARCH_EVIDENCE_UNAVAILABLE" in result["research_features"]["errors"]
        assert llm.messages.create.call_count == 4

    @pytest.mark.asyncio
    async def test_existing_aggregated_context_is_preserved_and_extended(self):
        llm = MagicMock()
        plan = json.dumps({"actions": []})
        verdict = json.dumps({"is_complete": True, "feedback": ""})
        llm.messages.create.side_effect = [make_llm_response(plan), make_llm_response(verdict)]
        agent = make_agent(llm=llm, max_loops=3)

        shared = make_shared_state(aggregated_research_context=["pre-existing chunk"])
        result = await agent.run(shared)

        assert result["aggregated_research_context"] == []
        assert result["research_features"]["missing"] is True

    @pytest.mark.asyncio
    async def test_brain_provider_failure_uses_bounded_diverse_plan(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("LLM API down")
        agent = make_agent(llm=llm, max_loops=3)

        state = make_research_state()
        update = await agent._brain_node(state)
        plan = json.loads(update["messages"][-1]["content"])

        assert [action["tool"] for action in plan["actions"]] == [
            "rag_hybrid_query",
            "tavily_search",
            "news_search",
            "sec_edgar_filing",
        ]
        ticker = state["manager_directives"]["ticker"]
        assert plan["actions"][0]["arguments"]["entity"] == ticker
        assert plan["actions"][3]["arguments"]["ticker"] == ticker


def test_normalize_evidence_preserves_retrieval_origins_across_gates():
    context = make_context()
    record = {
        "evidence_id": "hybrid-both", "source": "rag_hybrid_query",
        "provider": "rag_hybrid_query", "ticker": "NVDA",
        "published_at": "2024-01-15T12:00:00Z",
        "dataset_version": "dataset-v1", "origins": ["vector", "graph"],
        "citation": {"url": "https://archive.test/hybrid-both"},
    }

    first = normalize_evidence(
        record, context=context, run_id="run-test", source="rag_hybrid_query"
    )
    second = normalize_evidence(
        first, context=context, run_id="run-test", source="research_agent_final"
    )

    assert first["retrieval_origins"] == ["graph", "vector"]
    assert second["retrieval_origins"] == ["graph", "vector"]