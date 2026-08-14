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
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from agents.research_agent import ResearchAgent
from agents.state import SharedManagerState, ResearchAgentState


def make_llm_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def make_agent(llm=None, max_loops=3):
    llm = llm or MagicMock()
    return ResearchAgent(llm_client=llm, max_loops=max_loops)


def make_shared_state(task_query="Analyze NVDA", **kw):
    state: SharedManagerState = {
        "task_query": task_query,
        "aggregated_research_context": [],
        "manager_directives": {},
    }
    state.update(kw)
    return state


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


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------

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
        return {
            "loop_counter": loop_counter,
            "is_complete": is_complete,
            "shared_manager_ref": {"manager_directives": directives},
        }

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

        state: ResearchAgentState = {
            "messages": [], "context_chunks": [], "loop_counter": 0,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
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

        state: ResearchAgentState = {
            "messages": [], "context_chunks": [], "loop_counter": 1,
            "validation_feedback": "Need more recent data", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
        await agent._brain_node(state)

        sent_prompt = llm.messages.create.call_args.kwargs["messages"][-1]["content"]
        assert "Need more recent data" in sent_prompt

    @pytest.mark.asyncio
    async def test_history_messages_prepended_to_new_prompt(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response('{"actions": []}')
        agent = make_agent(llm=llm)

        prior = [{"role": "user", "content": "prior msg"}]
        state: ResearchAgentState = {
            "messages": prior, "context_chunks": [], "loop_counter": 0,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
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
        state: ResearchAgentState = {
            "messages": [{"role": "assistant", "content": "not json"}],
            "context_chunks": [], "loop_counter": 0, "validation_feedback": "",
            "is_complete": False, "shared_manager_ref": make_shared_state(),
        }
        update = await agent._executor_node(state)
        assert update["loop_counter"] == 1
        assert "no valid action plan" in update["context_chunks"][0]

    @pytest.mark.asyncio
    async def test_successful_tool_call_appends_formatted_chunk(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "NVDA"}}]})
        state: ResearchAgentState = {
            "messages": [{"role": "assistant", "content": plan}],
            "context_chunks": [], "loop_counter": 0, "validation_feedback": "",
            "is_complete": False, "shared_manager_ref": make_shared_state(),
        }

        mock_session = AsyncMock()
        tool_result = MagicMock()
        tool_result.content = [MagicMock(type="text", text="search results here")]
        mock_session.call_tool.return_value = tool_result

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            update = await agent._executor_node(state)

        assert update["loop_counter"] == 1
        assert "search results here" in update["context_chunks"][0]
        assert "tavily_search" in update["context_chunks"][0]

    @pytest.mark.asyncio
    async def test_tool_failure_appends_error_marker_not_raises(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "NVDA"}}]})
        state: ResearchAgentState = {
            "messages": [{"role": "assistant", "content": plan}],
            "context_chunks": [], "loop_counter": 0, "validation_feedback": "",
            "is_complete": False, "shared_manager_ref": make_shared_state(),
        }

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = RuntimeError("tool crashed")

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            update = await agent._executor_node(state)

        assert "[TOOL ERROR]" in update["context_chunks"][0]
        assert "tool crashed" in update["context_chunks"][0]

    @pytest.mark.asyncio
    async def test_mcp_connection_failure_appends_error_marker_not_raises(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"tool": "tavily_search", "arguments": {"query": "q"}}]})
        state: ResearchAgentState = {
            "messages": [{"role": "assistant", "content": plan}],
            "context_chunks": [], "loop_counter": 0, "validation_feedback": "",
            "is_complete": False, "shared_manager_ref": make_shared_state(),
        }

        with patch("agents.research_agent.stdio_client", side_effect=ConnectionError("server down")):
            update = await agent._executor_node(state)

        assert "[MCP CONNECTION ERROR]" in update["context_chunks"][0]

    @pytest.mark.asyncio
    async def test_action_with_no_tool_name_is_skipped(self):
        agent = make_agent()
        plan = json.dumps({"actions": [{"arguments": {"query": "q"}}]})
        state: ResearchAgentState = {
            "messages": [{"role": "assistant", "content": plan}],
            "context_chunks": [], "loop_counter": 0, "validation_feedback": "",
            "is_complete": False, "shared_manager_ref": make_shared_state(),
        }

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
        state: ResearchAgentState = {
            "messages": [], "context_chunks": [], "loop_counter": 1,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
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
        state: ResearchAgentState = {
            "messages": [], "context_chunks": ["some chunk"], "loop_counter": 1,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
        update = await agent._checker_node(state)
        assert update["is_complete"] is True
        assert update["validation_feedback"] == ""

    @pytest.mark.asyncio
    async def test_incomplete_verdict_keeps_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            json.dumps({"is_complete": False, "feedback": "Need more sources"})
        )
        agent = make_agent(llm=llm)
        state: ResearchAgentState = {
            "messages": [], "context_chunks": ["chunk"], "loop_counter": 1,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
        update = await agent._checker_node(state)
        assert update["is_complete"] is False
        assert update["validation_feedback"] == "Need more sources"

    @pytest.mark.asyncio
    async def test_invalid_json_verdict_defaults_to_incomplete(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("not valid json")
        agent = make_agent(llm=llm)
        state: ResearchAgentState = {
            "messages": [], "context_chunks": ["chunk"], "loop_counter": 1,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
        update = await agent._checker_node(state)
        assert update["is_complete"] is False
        assert "retry" in update["validation_feedback"].lower()

    @pytest.mark.asyncio
    async def test_combined_chunks_truncated_above_12000_chars(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            json.dumps({"is_complete": True, "feedback": ""})
        )
        agent = make_agent(llm=llm)
        huge_chunk = "x" * 20000
        state: ResearchAgentState = {
            "messages": [], "context_chunks": [huge_chunk], "loop_counter": 1,
            "validation_feedback": "", "is_complete": False,
            "shared_manager_ref": make_shared_state(),
        }
        await agent._checker_node(state)
        sent_prompt = llm.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "[truncated for audit]" in sent_prompt


# ---------------------------------------------------------------------------
# run() — full pipeline via the stub LangGraph executor
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
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
        tool_result.content = [MagicMock(type="text", text="result data")]
        mock_session.call_tool.return_value = tool_result

        with patch("agents.research_agent.ClientSession") as mock_session_cls, \
             patch("agents.research_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            result = await agent.run(make_shared_state())

        assert len(result["aggregated_research_context"]) == 1
        assert "result data" in result["aggregated_research_context"][0]

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

        assert len(result["aggregated_research_context"]) == 2
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

        assert result["aggregated_research_context"][0] == "pre-existing chunk"
        assert len(result["aggregated_research_context"]) == 2

    @pytest.mark.asyncio
    async def test_graph_exception_wrapped_as_runtimeerror(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("LLM API down")
        agent = make_agent(llm=llm, max_loops=3)

        with pytest.raises(RuntimeError, match="ResearchAgent internal graph failed"):
            await agent.run(make_shared_state())