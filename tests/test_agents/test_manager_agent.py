"""
Tests for: agents/manager_agent.py
Phase: 6 — Agents (4th: orchestrates research/financial/sentiment agents)

CRITICAL BLOCKING BUGS FOUND (must be fixed before this file can even be
imported in the real project):

  BUG 1 (line ~115): `_actions_block: str = "<literal newline>".join(...)`
  has a LITERAL newline character inside an unterminated single-line
  string literal instead of the escape sequence "\n". This is a
  SyntaxError — `python -c "import agents.manager_agent"` fails
  immediately with "SyntaxError: unterminated string literal".
  Fix: change to `_actions_block: str = "\n".join(...)` (escaped \n).

  BUG 2 (line ~145, _ROUTER_SYSTEM_PROMPT): this is an f-string (f-triple-
  quoted) because it interpolates {_actions_block}. The literal JSON example
  block inside it uses single braces which Python's f-string parser
  tries to evaluate as expressions. An empty `{}` is a SyntaxError
  ("f-string: valid expression required before '}'").
  Fix: double all literal braces in the example block (`{{`, `}}`)
  except the genuine `{_actions_block}` interpolation.

This test file's LOCAL copy of manager_agent.py has BOTH bugs fixed so
the actual orchestration logic can be tested. Apply the same two fixes
to the real project file before running this suite for real.

Mocking strategy: research_agent/financial_agent/sentiment_agent are
MagicMock instances with `.run` as an AsyncMock (matching the real
Specialist Agent `async def run(shared_state) -> shared_state` contract).
ManagerMemory is a MagicMock with all Phase 4 facade methods stubbed.
llm_client is an AsyncMock matching anthropic.AsyncAnthropic's
`.messages.create()` interface.
"""
import json
import time
from unittest.mock import MagicMock, AsyncMock
import pytest

from agents.manager_agent import (
    ManagerAgent,
    _extract_chunk_text,
    _feedback_to_snapshot,
    _infer_result_keys,
)
from memory.manager_memory import EvaluationFeedback


def make_llm_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def make_specialist_agent(name="ResearchAgent"):
    agent = MagicMock()
    agent.__class__.__name__ = name
    agent.run = AsyncMock(side_effect=lambda state: state)
    return agent


def make_memory():
    memory = MagicMock()
    memory.recall.return_value = {"short_term": {}, "long_term": {"heuristics": {}}}
    memory.agents_run.return_value = []
    memory.get_messages.return_value = []
    memory.get_preference.return_value = None
    memory.log_dispatch.return_value = MagicMock(
        dispatched_at=time.time(), outcome="pending", duration_s=None,
        error_message=None, result_keys=[],
    )
    return memory


def make_manager(llm=None, memory=None, research=None, financial=None, sentiment=None, max_loops=8):
    return ManagerAgent(
        research_agent=research or make_specialist_agent("ResearchAgent"),
        financial_agent=financial or make_specialist_agent("FinancialAnalystAgent"),
        sentiment_agent=sentiment or make_specialist_agent("SentimentAgent"),
        memory=memory or make_memory(),
        llm_client=llm or AsyncMock(),
        max_routing_loops=max_loops,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestExtractChunkText:
    def test_dict_chunk_extracts_text_field(self):
        assert _extract_chunk_text({"text": "hello", "other": 1}) == "hello"

    def test_dict_chunk_without_text_falls_back_to_str(self):
        result = _extract_chunk_text({"other": 1})
        assert result == str({"other": 1})

    def test_string_chunk_returned_as_is(self):
        assert _extract_chunk_text("plain string") == "plain string"

    def test_other_type_falls_back_to_str(self):
        assert _extract_chunk_text(42) == "42"


class TestFeedbackToSnapshot:
    def test_converts_all_fields(self):
        fb = EvaluationFeedback(
            step="research", timestamp=time.time(), passed=True, score=80,
            issues=["x"], next_action="run_financial", raw_verdict="{}",
        )
        snap = _feedback_to_snapshot(fb)
        assert snap["step"] == "research"
        assert snap["passed"] is True
        assert snap["score"] == 80
        assert snap["next_action"] == "run_financial"
        assert snap["issues"] == ["x"]

    def test_issues_list_is_copied_not_referenced(self):
        original_issues = ["x"]
        fb = EvaluationFeedback(
            step="research", timestamp=time.time(), passed=True, score=80,
            issues=original_issues, next_action="finalise", raw_verdict="{}",
        )
        snap = _feedback_to_snapshot(fb)
        original_issues.append("y")
        assert snap["issues"] == ["x"]


class TestInferResultKeys:
    def test_research_with_data_returns_key(self):
        state = {"aggregated_research_context": ["chunk"]}
        assert _infer_result_keys("research", state) == ["aggregated_research_context"]

    def test_research_empty_returns_empty_list(self):
        state = {"aggregated_research_context": []}
        assert _infer_result_keys("research", state) == []

    def test_financial_with_data_returns_key(self):
        state = {"financial_metrics_summary": {"ticker": "NVDA"}}
        assert _infer_result_keys("financial", state) == ["financial_metrics_summary"]

    def test_unknown_agent_key_returns_empty_list(self):
        assert _infer_result_keys("unknown", {}) == []


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_max_routing_loops_is_8(self):
        manager = make_manager()
        assert manager._max_routing_loops == 8

    def test_agents_dict_maps_correct_keys(self):
        research = make_specialist_agent("ResearchAgent")
        financial = make_specialist_agent("FinancialAnalystAgent")
        sentiment = make_specialist_agent("SentimentAgent")
        manager = make_manager(research=research, financial=financial, sentiment=sentiment)
        assert manager._agents["research"] is research
        assert manager._agents["financial"] is financial
        assert manager._agents["sentiment"] is sentiment

    def test_graph_compiled_at_init(self):
        manager = make_manager()
        assert manager._graph is not None


# ---------------------------------------------------------------------------
# _hydrate_state
# ---------------------------------------------------------------------------

class TestHydrateState:
    def test_initialises_all_required_fields(self):
        manager = make_manager()
        state = manager._hydrate_state("Analyze NVDA", {"ticker": "NVDA"})
        assert state["task_query"] == "Analyze NVDA"
        assert state["manager_directives"] == {"ticker": "NVDA"}
        assert state["aggregated_research_context"] == []
        assert state["financial_metrics_summary"] == {}
        assert state["final_report"] == ""


# ---------------------------------------------------------------------------
# _brain_route
# ---------------------------------------------------------------------------

class TestBrainRoute:
    @pytest.mark.asyncio
    async def test_valid_action_returned(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "action": "run_research", "reasoning": "start", "directive_updates": {},
        }))
        manager = make_manager(llm=llm)

        decision = await manager._brain_route(
            state={"task_query": "q", "manager_directives": {}},
            memory_recall={"short_term": {}, "long_term": {}},
            loop_counter=1,
        )
        assert decision["action"] == "run_research"

    @pytest.mark.asyncio
    async def test_invalid_action_defaults_to_abort(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "action": "not_a_real_action", "reasoning": "x", "directive_updates": {},
        }))
        manager = make_manager(llm=llm)

        decision = await manager._brain_route(
            state={"task_query": "q", "manager_directives": {}},
            memory_recall={"short_term": {}, "long_term": {}},
            loop_counter=1,
        )
        assert decision["action"] == "abort"

    @pytest.mark.asyncio
    async def test_api_exception_falls_back_based_on_agents_run(self):
        llm = AsyncMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        memory = make_memory()
        memory.agents_run.return_value = ["ResearchAgent"]
        manager = make_manager(llm=llm, memory=memory)

        decision = await manager._brain_route(
            state={"task_query": "q", "manager_directives": {}},
            memory_recall={"short_term": {}, "long_term": {}},
            loop_counter=1,
        )
        assert decision["action"] == "run_financial"

    @pytest.mark.asyncio
    async def test_api_exception_fallback_finalise_when_all_agents_run(self):
        llm = AsyncMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        memory = make_memory()
        memory.agents_run.return_value = ["ResearchAgent", "FinancialAnalystAgent", "SentimentAgent"]
        manager = make_manager(llm=llm, memory=memory)

        decision = await manager._brain_route(
            state={"task_query": "q", "manager_directives": {}},
            memory_recall={"short_term": {}, "long_term": {}},
            loop_counter=1,
        )
        assert decision["action"] == "finalise"


# ---------------------------------------------------------------------------
# _brain_evaluate
# ---------------------------------------------------------------------------

class TestBrainEvaluate:
    @pytest.mark.asyncio
    async def test_research_agent_builds_correct_summary_and_feedback(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "passed": True, "score": 90, "issues": [], "next_action": "run_financial",
        }))
        manager = make_manager(llm=llm)
        state = {"aggregated_research_context": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}

        feedback = await manager._brain_evaluate("ResearchAgent", state, {"short_term": {}})
        assert feedback.step == "research"
        assert feedback.passed is True
        assert feedback.score == 90

    @pytest.mark.asyncio
    async def test_financial_agent_evaluation(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "passed": False, "score": 30, "issues": ["low score"], "next_action": "rerun_financial",
        }))
        manager = make_manager(llm=llm)
        state = {"financial_metrics_summary": {"ticker": "NVDA", "composite_score": {"score": 40}}}

        feedback = await manager._brain_evaluate("FinancialAnalystAgent", state, {"short_term": {}})
        assert feedback.step == "financial"
        assert feedback.passed is False
        assert feedback.next_action == "rerun_financial"

    @pytest.mark.asyncio
    async def test_sentiment_agent_evaluation(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "passed": True, "score": 70, "issues": [], "next_action": "finalise",
        }))
        manager = make_manager(llm=llm)
        state = {"sentiment_analysis_summary": {"fear_greed_score": 0.3}}

        feedback = await manager._brain_evaluate("SentimentAgent", state, {"short_term": {}})
        assert feedback.step == "sentiment"

    @pytest.mark.asyncio
    async def test_api_exception_defaults_to_partial_pass(self):
        llm = AsyncMock()
        llm.messages.create.side_effect = RuntimeError("down")
        manager = make_manager(llm=llm)

        feedback = await manager._brain_evaluate("ResearchAgent", {}, {"short_term": {}})
        assert feedback.passed is True
        assert feedback.score == 50
        assert feedback.next_action == "run_financial"

    @pytest.mark.asyncio
    async def test_does_not_call_memory_add_evaluation(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({"passed": True, "score": 90}))
        memory = make_memory()
        manager = make_manager(llm=llm, memory=memory)

        await manager._brain_evaluate("ResearchAgent", {}, {"short_term": {}})
        memory.add_evaluation.assert_not_called()


# ---------------------------------------------------------------------------
# _brain_finalise
# ---------------------------------------------------------------------------

class TestBrainFinalise:
    @pytest.mark.asyncio
    async def test_returns_report_text(self):
        llm = AsyncMock()
        llm.messages.create.return_value = make_llm_response("Final report text.")
        manager = make_manager(llm=llm)

        report = await manager._brain_finalise({"task_query": "q", "agent_execution_history": []})
        assert report == "Final report text."

    @pytest.mark.asyncio
    async def test_api_exception_returns_fallback_report(self):
        llm = AsyncMock()
        llm.messages.create.side_effect = RuntimeError("synthesis failed")
        manager = make_manager(llm=llm)

        report = await manager._brain_finalise({
            "task_query": "q", "agent_execution_history": [],
            "financial_metrics_summary": {"composite_score": {"score": 70}},
            "sentiment_analysis_summary": {"fear_greed_label": "Greed"},
        })
        assert "REPORT GENERATION FAILED" in report
        assert "70" in report
        assert "Greed" in report


# ---------------------------------------------------------------------------
# _dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    @pytest.mark.asyncio
    async def test_successful_dispatch_updates_history(self):
        research = make_specialist_agent("ResearchAgent")
        manager = make_manager(research=research)
        state = {"manager_directives": {}, "orchestrator_logs": [], "agent_execution_history": []}

        result = await manager._dispatch("run_research", state)

        assert result["agent_execution_history"][0]["outcome"] == "success"
        research.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_agent_key_returns_state_unchanged(self):
        manager = make_manager()
        state = {"manager_directives": {}, "orchestrator_logs": [], "agent_execution_history": []}

        result = await manager._dispatch("run_unknown_agent", state)
        assert result["agent_execution_history"] == []

    @pytest.mark.asyncio
    async def test_agent_exception_recorded_as_error_outcome(self):
        research = make_specialist_agent("ResearchAgent")
        research.run.side_effect = RuntimeError("agent crashed")
        manager = make_manager(research=research)
        state = {"manager_directives": {}, "orchestrator_logs": [], "agent_execution_history": []}

        result = await manager._dispatch("run_research", state)

        assert result["agent_execution_history"][0]["outcome"] == "error"
        assert "agent crashed" in result["agent_execution_history"][0]["error_message"]

    @pytest.mark.asyncio
    async def test_rerun_prefix_maps_to_same_agent_key(self):
        financial = make_specialist_agent("FinancialAnalystAgent")
        manager = make_manager(financial=financial)
        state = {"manager_directives": {}, "orchestrator_logs": [], "agent_execution_history": []}

        await manager._dispatch("rerun_financial", state)
        financial.run.assert_called_once()


# ---------------------------------------------------------------------------
# _persist
# ---------------------------------------------------------------------------

class TestPersist:
    def test_research_stores_chunk_count_heuristic(self):
        memory = make_memory()
        manager = make_manager(memory=memory)
        state = {
            "manager_directives": {"ticker": "NVDA"},
            "aggregated_research_context": ["a", "b"],
        }
        fb = EvaluationFeedback(step="research", timestamp=time.time(), passed=True,
                                score=90, issues=[], next_action="run_financial", raw_verdict="{}")

        manager._persist("research", state, fb)

        memory.store_heuristic.assert_called_once_with("NVDA_research_chunks", 2)

    def test_financial_stores_ticker_insight_and_heuristic(self):
        memory = make_memory()
        manager = make_manager(memory=memory)
        state = {
            "manager_directives": {"ticker": "NVDA"},
            "financial_metrics_summary": {"composite_score": {"score": 80, "grade": "A"}, "sector": "Tech"},
        }
        fb = EvaluationFeedback(step="financial", timestamp=time.time(), passed=True,
                                score=80, issues=[], next_action="run_sentiment", raw_verdict="{}")

        manager._persist("financial", state, fb)

        memory.store_ticker_insight.assert_called_once()
        memory.store_heuristic.assert_called_once_with("NVDA_financial_score", 80)

    def test_sentiment_stores_ticker_insight(self):
        memory = make_memory()
        manager = make_manager(memory=memory)
        state = {
            "manager_directives": {"ticker": "NVDA"},
            "sentiment_analysis_summary": {"fear_greed_score": 0.3, "fear_greed_label": "Greed"},
        }
        fb = EvaluationFeedback(step="sentiment", timestamp=time.time(), passed=True,
                                score=70, issues=[], next_action="finalise", raw_verdict="{}")

        manager._persist("sentiment", state, fb)

        memory.store_ticker_insight.assert_called_once()

    def test_no_ticker_skips_persistence(self):
        memory = make_memory()
        manager = make_manager(memory=memory)
        state = {"manager_directives": {}, "aggregated_research_context": ["a"]}
        fb = EvaluationFeedback(step="research", timestamp=time.time(), passed=True,
                                score=90, issues=[], next_action="run_financial", raw_verdict="{}")

        manager._persist("research", state, fb)

        memory.store_heuristic.assert_not_called()


# ---------------------------------------------------------------------------
# Conditional edge routers
# ---------------------------------------------------------------------------

class TestShouldRoute:
    def test_guardrail_forces_abort(self):
        manager = make_manager(max_loops=3)
        g = {"last_action": "run_research", "loop_counter": 3}
        assert manager._should_route(g) == "abort"

    def test_finalise_action_routes_to_finalise(self):
        manager = make_manager(max_loops=8)
        g = {"last_action": "finalise", "loop_counter": 1}
        assert manager._should_route(g) == "finalise"

    def test_run_prefix_routes_to_dispatch(self):
        manager = make_manager(max_loops=8)
        g = {"last_action": "run_research", "loop_counter": 1}
        assert manager._should_route(g) == "dispatch"

    def test_rerun_prefix_routes_to_dispatch(self):
        manager = make_manager(max_loops=8)
        g = {"last_action": "rerun_financial", "loop_counter": 1}
        assert manager._should_route(g) == "dispatch"

    def test_unknown_action_routes_to_abort(self):
        manager = make_manager(max_loops=8)
        g = {"last_action": "something_weird", "loop_counter": 1}
        assert manager._should_route(g) == "abort"

    def test_abort_action_routes_to_abort(self):
        manager = make_manager(max_loops=8)
        g = {"last_action": "abort", "loop_counter": 1}
        assert manager._should_route(g) == "abort"


class TestShouldContinueAfterPersist:
    def test_guardrail_forces_abort(self):
        manager = make_manager(max_loops=2)
        g = {"loop_counter": 2, "evaluation_passed": False, "last_action": "rerun_research"}
        assert manager._should_continue_after_persist(g) == "abort"

    def test_eval_passed_routes_to_brain_route(self):
        manager = make_manager(max_loops=8)
        g = {"loop_counter": 1, "evaluation_passed": True, "last_action": "run_research"}
        assert manager._should_continue_after_persist(g) == "brain_route"

    def test_eval_failed_with_rerun_action_routes_to_dispatch(self):
        manager = make_manager(max_loops=8)
        g = {"loop_counter": 1, "evaluation_passed": False, "last_action": "rerun_research"}
        assert manager._should_continue_after_persist(g) == "dispatch"

    def test_eval_failed_without_rerun_prefix_routes_to_brain_route(self):
        manager = make_manager(max_loops=8)
        g = {"loop_counter": 1, "evaluation_passed": False, "last_action": "run_research"}
        assert manager._should_continue_after_persist(g) == "brain_route"


# ---------------------------------------------------------------------------
# run() — full pipeline via the stub LangGraph executor
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    @pytest.mark.asyncio
    async def test_happy_path_runs_all_three_agents_and_finalises(self):
        llm = AsyncMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"action": "run_research", "reasoning": "x", "directive_updates": {}})),
            make_llm_response(json.dumps({"passed": True, "score": 90, "issues": [], "next_action": "run_financial"})),
            make_llm_response(json.dumps({"action": "run_financial", "reasoning": "x", "directive_updates": {}})),
            make_llm_response(json.dumps({"passed": True, "score": 90, "issues": [], "next_action": "run_sentiment"})),
            make_llm_response(json.dumps({"action": "run_sentiment", "reasoning": "x", "directive_updates": {}})),
            make_llm_response(json.dumps({"passed": True, "score": 90, "issues": [], "next_action": "finalise"})),
            make_llm_response(json.dumps({"action": "finalise", "reasoning": "x", "directive_updates": {}})),
            make_llm_response("Final synthesized report."),
        ]

        research = make_specialist_agent("ResearchAgent")
        research.run = AsyncMock(side_effect=lambda s: {**s, "aggregated_research_context": ["chunk1", "chunk2", "chunk3"]})
        financial = make_specialist_agent("FinancialAnalystAgent")
        financial.run = AsyncMock(side_effect=lambda s: {**s, "financial_metrics_summary": {"ticker": "NVDA", "composite_score": {"score": 80}}})
        sentiment = make_specialist_agent("SentimentAgent")
        sentiment.run = AsyncMock(side_effect=lambda s: {**s, "sentiment_analysis_summary": {"fear_greed_score": 0.3}})

        manager = make_manager(llm=llm, research=research, financial=financial, sentiment=sentiment, max_loops=8)

        result = await manager.run(task_query="Analyze NVDA", manager_directives={"ticker": "NVDA"})

        assert result["final_report"] == "Final synthesized report."
        research.run.assert_called_once()
        financial.run.assert_called_once()
        sentiment.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_guardrail_aborts_when_brain_loops_forever(self):
        llm = AsyncMock()
        responses = []
        for _ in range(10):
            responses.append(make_llm_response(json.dumps({
                "action": "run_research", "reasoning": "x", "directive_updates": {},
            })))
            responses.append(make_llm_response(json.dumps({
                "passed": False, "score": 10, "issues": [], "next_action": "rerun_research",
            })))
        llm.messages.create.side_effect = responses

        research = make_specialist_agent("ResearchAgent")
        manager = make_manager(llm=llm, research=research, max_loops=3)

        result = await manager.run(task_query="Analyze NVDA")

        assert result.get("final_report", "") == ""
        assert "[ABORT]" in "".join(result["orchestrator_logs"])

    @pytest.mark.asyncio
    async def test_graph_exception_wrapped_as_runtimeerror(self):
        llm = AsyncMock()
        llm.messages.create.side_effect = RuntimeError("totally broken")
        memory = make_memory()
        memory.recall.side_effect = RuntimeError("memory layer broken")
        manager = make_manager(llm=llm, memory=memory)

        with pytest.raises(RuntimeError, match="ManagerAgent internal graph failed"):
            await manager.run(task_query="Analyze NVDA")

    @pytest.mark.asyncio
    async def test_memory_new_session_called_with_task_query(self):
        llm = AsyncMock()
        memory = make_memory()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"action": "finalise", "reasoning": "x", "directive_updates": {}})),
            make_llm_response("report"),
        ]
        manager = make_manager(llm=llm, memory=memory)

        await manager.run(task_query="Analyze NVDA", user_preferences={"fmt": "concise"})

        memory.new_session.assert_called_once()
        memory.store_preference.assert_called_once_with("fmt", "concise")