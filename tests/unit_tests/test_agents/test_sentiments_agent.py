"""
Tests for: agents/sentiment_agent.py
Phase: 6 — Agents (3rd: per phase plan, after Phase 0 sentiment bug fix)

Mocking strategy: same DI pattern as financial_agent — llm_client injected,
MCP session injected directly into _execute_sentiment_pipeline (explicit
param), and stdio_client/ClientSession patched only for the full run()
pipeline test.

NOTE: __init__ requires `server_script_path` as a REQUIRED positional arg
(no default) — this is the exact constructor shape referenced by the
Phase 0 bug in api/main.py:107 (`SentimentAgent()` called with no args,
which raises TypeError: missing required positional argument).
"""
import json
import math
import time
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp import StdioServerParameters
from agents.sentiment_agent import (
    SentimentAgent,
    _mcp_trace_inputs,
    _mcp_trace_outputs,
    _publish_lifecycle_progress,
    _sentiment_call_tool,
    _trace_outputs,
)
from agents.executor_helpers.sentiment_helper import (
    SENTIMENT_LIFECYCLE_BOUNDARIES,
    sentiment_lifecycle_event,
)
from core.forecast_contract import ForecastContext


TEST_FORECAST_CONTEXT = ForecastContext(
    ticker="NVDA", as_of="2026-06-30T20:00:00Z",
    schema_version="2.1", dataset_version="sentiment-fixture-v1",
    feature_version="monthly-sentiment-features-v1", model_version="test-model-v1",
).to_dict()


def make_llm_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def make_tool_result(payload: dict):
    result = MagicMock()
    result.content = [MagicMock(text=json.dumps(payload))]
    return result


def make_agent(llm=None, max_loops=2):
    llm = llm or MagicMock()
    return SentimentAgent(server_script_path="dummy_path.py", llm_client=llm, max_loops=max_loops)


def make_shared_state(task_query="Analyze NVDA sentiment", **kw):
    state = {
        "task_query": task_query, "manager_directives": {},
        "forecast_context": TEST_FORECAST_CONTEXT, "run_id": "sentiment-test-run",
    }
    state.update(kw)
    return state


def make_agent_state(shared=None, **overrides):
    shared = shared or make_shared_state()
    directives = dict(shared.get("manager_directives", {}))
    directives.setdefault("forecast_context", shared.get("forecast_context", TEST_FORECAST_CONTEXT))
    directives.setdefault("run_id", shared.get("run_id", "sentiment-test-run"))
    directives.setdefault("ticker", "NVDA")
    state = {
        "messages": [], "retrieved_chunks": [], "sources_metadata": [],
        "finbert_result": {}, "vader_result": {}, "fear_greed_result": {},
        "brain_reasoning": "", "loop_counter": 0, "extraction_errors": [],
        "prior_attempt_errors": [], "attempts": [],
        "sentiment_status": "VALID", "sentiment_polarity": "NEUTRAL",
        "task_query": shared.get("task_query", ""),
        "manager_directives": directives,
        "financial_metrics_summary": shared.get("financial_metrics_summary", {}),
        "aggregated_research_context": shared.get("aggregated_research_context", []),
        "current_plan": {}, "shared_manager_ref": shared,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# __init__ — documents the Phase 0 constructor bug shape
# ---------------------------------------------------------------------------

class TestInitRequiresServerScriptPath:
    def test_missing_server_script_path_raises_typeerror(self):
        with pytest.raises(TypeError):
            SentimentAgent()  # type: ignore[call-arg]

    def test_explicit_path_builds_correct_server_params(self):
        agent = SentimentAgent(server_script_path="/path/to/sentiment_server.py", llm_client=MagicMock())
        assert agent._server_params.args == ["/path/to/sentiment_server.py"]

    def test_default_max_loops_is_2(self):
        agent = make_agent()
        assert agent._default_max_loops == 2

    def test_injected_mcp_server_params_overrides_script_path(self):
        custom_params = StdioServerParameters(command="python", args=["custom.py"])
        agent = SentimentAgent(
            server_script_path="ignored.py", llm_client=MagicMock(), mcp_server_params=custom_params,
        )
        assert agent._server_params is custom_params


# ---------------------------------------------------------------------------
# _extract_ticker — 3-tier resolution order
# ---------------------------------------------------------------------------

class TestExtractTicker:
    def test_directive_has_highest_priority(self):
        agent = make_agent()
        ticker = agent._extract_ticker("query AAPL", {"ticker": "nvda"}, {"ticker": "AMD"})
        assert ticker == "NVDA"

    def test_financial_summary_second_priority(self):
        agent = make_agent()
        ticker = agent._extract_ticker("some query", {}, {"ticker": "amd"})
        assert ticker == "AMD"

    def test_regex_fallback_when_others_absent(self):
        agent = make_agent()
        ticker = agent._extract_ticker("Analyze NVDA sentiment", {}, {})
        assert ticker == "NVDA"

    def test_stop_words_filtered_including_sentiment_specific_ones(self):
        agent = make_agent()
        ticker = agent._extract_ticker("RAG NLP analysis for the CEO", {}, {})
        assert ticker is None

    def test_returns_none_when_unresolvable(self):
        agent = make_agent()
        assert agent._extract_ticker("just a generic query", {}, {}) is None


# ---------------------------------------------------------------------------
# _brain_plan
# ---------------------------------------------------------------------------

class TestBrainPlan:
    @pytest.mark.asyncio
    async def test_valid_plan_parsed_and_returned(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "retrieval_query": "NVDA sentiment", "ticker": "NVDA", "days_back": 7, "reasoning": "x",
        }))
        agent = make_agent(llm=llm)
        state = make_agent_state()

        plan = await agent._brain_plan(state)

        assert plan["retrieval_query"] == "NVDA sentiment"
        assert plan["ticker"] == "NVDA"
        assert len(state["messages"]) == 2

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_stripped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            "```json\n" + json.dumps({"retrieval_query": "q", "ticker": None, "days_back": 7}) + "\n```"
        )
        agent = make_agent(llm=llm)
        plan = await agent._brain_plan(make_agent_state())
        assert plan["retrieval_query"] == "q"

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_default_plan(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("not valid json")
        agent = make_agent(llm=llm)
        state = make_agent_state(shared=make_shared_state(task_query="Analyze NVDA"))

        plan = await agent._brain_plan(state)

        assert "NVDA" in plan["retrieval_query"]
        assert plan["days_back"] == 30
        assert "Default plan" in plan["reasoning"]

    @pytest.mark.asyncio
    async def test_api_exception_falls_back_to_default_plan_without_raising(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        plan = await agent._brain_plan(make_agent_state())
        assert "Default plan" in plan["reasoning"]

    @pytest.mark.asyncio
    async def test_directive_days_back_used_in_default_plan(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("down")
        agent = make_agent(llm=llm)
        state = make_agent_state(shared=make_shared_state(manager_directives={"days_back": 14}))
        plan = await agent._brain_plan(state)
        assert plan["days_back"] == 30


# ---------------------------------------------------------------------------
# _brain_analyze
# ---------------------------------------------------------------------------

class TestBrainAnalyze:
    @pytest.mark.asyncio
    async def test_returns_raw_text_with_fences_stripped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            "```json\n" + json.dumps({"overall_sentiment": "Bullish"}) + "\n```"
        )
        agent = make_agent(llm=llm)
        result = await agent._brain_analyze(make_agent_state())
        assert result == json.dumps({"overall_sentiment": "Bullish"})

    @pytest.mark.asyncio
    async def test_api_exception_returns_fallback_json_string(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        result = await agent._brain_analyze(make_agent_state())
        parsed = json.loads(result)
        assert parsed["status"] == "ERROR"
        assert parsed["overall_sentiment"] is None
        assert "API down" not in parsed["narrative"]
        assert "Brain API call failed" in parsed["risk_flags"]

    @pytest.mark.asyncio
    async def test_includes_chunk_count_in_prompt(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response('{"overall_sentiment": "Neutral"}')
        agent = make_agent(llm=llm)
        state = make_agent_state(retrieved_chunks=["a", "b", "c"])
        await agent._brain_analyze(state)
        sent_content = llm.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "CHUNK COUNT ANALYSED: 3" in sent_content

    @pytest.mark.asyncio
    async def test_prompt_enforces_strict_causal_flag_contract(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            '{"overall_sentiment": "Neutral", "causal_aspects_used": []}'
        )
        agent = make_agent(llm=llm)
        await agent._brain_analyze(make_agent_state(retrieved_chunks=["valuation mentioned"]))
        sent_content = llm.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "ONLY when that exact observation has causal_language_supported=True" in sent_content
        assert "may only be described as 'mentioned in evidence'" in sent_content


# ---------------------------------------------------------------------------
# _execute_sentiment_pipeline
# ---------------------------------------------------------------------------

class TestExecuteSentimentPipeline:
    @pytest.mark.asyncio
    async def test_happy_path_runs_all_four_steps(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"chunks": ["chunk1", "chunk2"], "sources": [{"x": 1}]}),
            make_tool_result({"bullish_prob": 0.6, "bearish_prob": 0.2, "total_chunks": 2}),
            make_tool_result({"compound": 0.3, "total_chunks": 2}),
            make_tool_result({"score": 0.4, "label": "Greed"}),
        ]
        state = make_agent_state()
        plan = {"retrieval_query": "NVDA sentiment", "ticker": "NVDA", "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        assert state["retrieved_chunks"] == ["chunk1", "chunk2"]
        assert state["finbert_result"]["bullish_prob"] == 0.6
        assert state["vader_result"]["compound"] == 0.3
        assert state["fear_greed_result"]["label"] == "Greed"
        assert session.call_tool.call_count == 4

    @pytest.mark.asyncio
    async def test_zero_chunks_skips_finbert_vader_and_fear_greed(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"chunks": [], "sources": []})
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        assert state["retrieved_chunks"] == []
        assert state["finbert_result"]["status"] == "NO_DATA"
        assert state["finbert_result"]["label"] is None
        assert state["vader_result"]["status"] == "NO_DATA"
        assert state["fear_greed_result"]["score"] is None
        assert session.call_tool.call_count == 1
        assert state["extraction_errors"] == []

    @pytest.mark.asyncio
    async def test_ticker_omitted_from_args_when_none(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"chunks": [], "sources": []})
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        call_args = session.call_tool.call_args_list[0].args[1]
        assert call_args["ticker"] == "NVDA"
        assert call_args["forecast_context"] == TEST_FORECAST_CONTEXT
        assert call_args["run_id"] == "sentiment-test-run"

    @pytest.mark.asyncio
    async def test_custom_weights_from_directives_passed_to_fear_greed(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"chunks": ["a"], "sources": []}),
            make_tool_result({"total_chunks": 1}),
            make_tool_result({"total_chunks": 1}),
            make_tool_result({"score": 0.1}),
        ]
        state = make_agent_state(shared=make_shared_state(
            manager_directives={"finbert_weight": 0.8, "vader_weight": 0.2}
        ))
        plan = {"retrieval_query": "q", "ticker": "NVDA", "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        fg_call_args = session.call_tool.call_args_list[3].args[1]
        assert fg_call_args["finbert_weight"] == 0.8
        assert fg_call_args["vader_weight"] == 0.2

    @pytest.mark.asyncio
    async def test_tool_error_in_payload_recorded_in_extraction_errors(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"error": "rate limited"})
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        assert state["extraction_errors"] == [
            "retrieve_social_data: SENTIMENT_TOOL_DATA_UNAVAILABLE"
        ]

    @pytest.mark.asyncio
    async def test_tool_exception_recorded_and_pipeline_continues(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = RuntimeError("connection lost")
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        assert state["extraction_errors"] == [
            "retrieve_social_data: SENTIMENT_TOOL_FAIL_CLOSED"
        ]
        assert state["retrieved_chunks"] == []

    @pytest.mark.asyncio
    async def test_fear_greed_runs_if_only_finbert_has_data(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"chunks": ["a"], "sources": []}),
            make_tool_result({"total_chunks": 1, "bullish_prob": 0.5}),
            make_tool_result({"total_chunks": 0}),
            make_tool_result({"score": 0.2}),
        ]
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        assert state["fear_greed_result"]["score"] == 0.2
        assert session.call_tool.call_count == 4

    @pytest.mark.asyncio
    async def test_missing_composite_score_cannot_be_valid(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({
                "chunks": ["a", "b", "c"],
                "sources_metadata": [
                    {"url": "https://one.example/a"},
                    {"url": "https://two.example/b"},
                    {"url": "https://three.example/c"},
                ],
                "unique_document_count": 3,
                "unique_domain_count": 3,
            }),
            make_tool_result({"total_chunks": 3}),
            make_tool_result({"total_chunks": 3}),
            make_tool_result({"score": None, "status": "NO_DATA"}),
        ]
        state = make_agent_state()

        await agent._execute_sentiment_pipeline(
            session,
            state,
            {"retrieval_query": "q", "ticker": "NVDA", "days_back": 7},
        )

        assert state["sentiment_status"] == "ERROR"
        assert state["attempts"][0]["quality_passed"] is False
        assert "missing or invalid composite sentiment score" in state["attempts"][0]["quality_failures"]


class TestAttemptSelectionAndCausalAudit:
    def test_low_coverage_retries_while_budget_remains(self):
        agent = make_agent(max_loops=2)
        state = make_agent_state(
            loop_counter=1,
            sentiment_status="LOW_COVERAGE",
            retrieved_chunks=["one document"],
        )
        assert agent._should_continue(state) == "brain_plan"

    def test_causal_mismatch_is_detected_fail_safe(self):
        audit = SentimentAgent._causal_audit(
            {"causal_aspects_used": ["valuation", "competition"]},
            ["ai_demand"],
        )
        assert audit["causal_mismatch_detected"] is True
        assert audit["unsupported_causal_aspects"] == ["competition", "valuation"]

    def test_best_attempt_selection_does_not_assume_last_is_best(self):
        agent = make_agent()
        state = make_agent_state(attempts=[
            {
                "attempt_number": 1,
                "status": "LOW_COVERAGE",
                "execution_success": True,
                "quality_passed": False,
                "quality_failures": ["unique document coverage 2 is below 3"],
                "polarity": "BULLISH",
                "retrieval_stats": {"unique_documents": 2, "unique_domains": 2},
                "retrieval": {
                    "chunks": ["strong first", "independent first"],
                    "sources_metadata": [],
                },
                "model_outputs": {
                    "finbert": {"total_chunks": 2},
                    "vader": {"total_chunks": 2},
                    "fear_greed": {"score": 0.4, "confidence": 0.8},
                },
                "errors": [],
            },
            {
                "attempt_number": 2,
                "status": "NO_DATA",
                "execution_success": True,
                "quality_passed": False,
                "quality_failures": ["no sentiment documents retrieved"],
                "polarity": None,
                "retrieval_stats": {"unique_documents": 0, "unique_domains": 0},
                "retrieval": {"chunks": [], "sources_metadata": []},
                "model_outputs": {
                    "finbert": {}, "vader": {},
                    "fear_greed": {"score": None, "confidence": 0.0},
                },
                "errors": [],
            },
        ])

        selected = agent._select_best_attempt(state)

        assert selected["attempt_number"] == 1
        assert state["retrieved_chunks"] == ["strong first", "independent first"]
        assert state["fear_greed_result"]["selected_attempt_number"] == 1


# ---------------------------------------------------------------------------
# run() — full pipeline
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    @pytest.mark.asyncio
    async def test_missing_forecast_context_fails_before_mcp_or_llm(self):
        agent = make_agent()
        with pytest.raises(ValueError, match="forecast_context is required"):
            await agent.run({"task_query": "Analyze NVDA", "manager_directives": {}})

    @pytest.mark.asyncio
    async def test_completes_in_one_loop_when_chunks_found(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"retrieval_query": "q", "ticker": "NVDA", "days_back": 7})),
            make_llm_response(json.dumps({
                "overall_sentiment": "Bullish", "conviction_level": "High",
                "key_signals": [], "model_agreement": "Strong", "narrative": "Good news",
                "risk_flags": [], "data_quality_note": "", "causal_aspects_used": [],
            })),
        ]
        agent = make_agent(llm=llm, max_loops=2)

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [
            make_tool_result({
                "chunks": ["chunk1", "chunk2", "chunk3"],
                "sources_metadata": [
                    {"url": "https://one.example/a"},
                    {"url": "https://two.example/b"},
                    {"url": "https://one.example/c"},
                ],
                "raw_chunk_count": 4,
                "unique_document_count": 3,
                "unique_domain_count": 2,
                "evidence_independence": 0.6667,
            }),
            make_tool_result({"total_chunks": 3, "label": "Bullish"}),
            make_tool_result({"total_chunks": 3, "label": "Bullish"}),
            make_tool_result({
                "score": 0.5, "label": "Greed", "confidence": 0.8,
                "direction_strength": 0.5, "model_agreement_score": 0.9,
                "evidence_independence": 0.6667,
            }),
        ]

        with patch("agents.sentiment_agent.ClientSession") as mock_session_cls, \
             patch("agents.sentiment_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            result = await agent.run(make_shared_state())

        summary = result["sentiment_analysis_summary"]
        assert summary["overall_sentiment"] == "Bullish"
        assert summary["loop_iterations_used"] == 1
        assert summary["total_chunks_analyzed"] == 3
        assert summary["quality_passed"] is True
        assert summary["execution_success"] is True
        assert summary["unique_document_count"] == 3
        assert summary["unique_domain_count"] == 2
        assert "sentiment_features" in result
        assert "brain_reasoning" not in summary
        assert "attempts" not in summary
        assert "sources_metadata" not in summary

    @pytest.mark.asyncio
    async def test_retries_once_when_zero_chunks_on_first_pass(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"retrieval_query": "narrow", "ticker": "NVDA", "days_back": 7})),
            make_llm_response(json.dumps({"retrieval_query": "broader", "ticker": "NVDA", "days_back": 14})),
            make_llm_response(json.dumps({"overall_sentiment": "Neutral"})),
        ]
        agent = make_agent(llm=llm, max_loops=2)

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [
            make_tool_result({"chunks": [], "sources": []}),
            make_tool_result({"chunks": ["found one"], "sources": []}),
            make_tool_result({"total_chunks": 1}),
            make_tool_result({"total_chunks": 1}),
            make_tool_result({"score": 0.1}),
        ]

        with patch("agents.sentiment_agent.ClientSession") as mock_session_cls, \
             patch("agents.sentiment_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            result = await agent.run(make_shared_state())

        summary = result["sentiment_analysis_summary"]
        assert summary["loop_iterations_used"] == 2
        assert summary["total_chunks_analyzed"] == 1
        assert summary["status"] == "LOW_COVERAGE"
        assert summary["quality_passed"] is False
        assert summary["execution_success"] is True

    @pytest.mark.asyncio
    async def test_malformed_brain_analysis_json_falls_back_gracefully(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"retrieval_query": "q", "ticker": "NVDA", "days_back": 7})),
            make_llm_response("not valid json from brain analyze"),
        ]
        agent = make_agent(llm=llm, max_loops=1)

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [
            make_tool_result({
                "chunks": ["a", "b", "c"],
                "sources_metadata": [
                    {"url": "https://one.example/a"},
                    {"url": "https://two.example/b"},
                    {"url": "https://three.example/c"},
                ],
                "unique_document_count": 3,
                "unique_domain_count": 3,
            }),
            make_tool_result({"total_chunks": 3, "label": "Neutral"}),
            make_tool_result({"total_chunks": 3, "label": "Neutral"}),
            make_tool_result({"score": 0.0, "label": "Neutral", "confidence": 1.0}),
        ]

        with patch("agents.sentiment_agent.ClientSession") as mock_session_cls, \
             patch("agents.sentiment_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            result = await agent.run(make_shared_state())

        summary = result["sentiment_analysis_summary"]
        assert summary["overall_sentiment"] is None
        assert "Brain analysis JSON parse failed" in summary["risk_flags"]
        assert "not valid json from brain analyze" not in summary["narrative"]
        assert summary["quality_passed"] is False

    @pytest.mark.asyncio
    async def test_ticker_in_summary_resolved_via_extract_ticker(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"retrieval_query": "q", "ticker": "NVDA", "days_back": 7})),
            make_llm_response(json.dumps({"overall_sentiment": "Neutral"})),
        ]
        agent = make_agent(llm=llm, max_loops=1)

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = make_tool_result({"chunks": [], "sources": []})

        with patch("agents.sentiment_agent.ClientSession") as mock_session_cls, \
             patch("agents.sentiment_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            shared = make_shared_state(manager_directives={"ticker": "NVDA"})
            result = await agent.run(shared)

        assert result["sentiment_analysis_summary"]["ticker"] == "NVDA"

    @staticmethod
    async def _run_typed_status_scenario(
        *, finbert_payload, vader_payload, fusion_payload, low_coverage=False,
    ):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({
                "retrieval_query": "q", "ticker": "NVDA", "days_back": 30,
            })),
            make_llm_response(json.dumps({
                "overall_sentiment": "Bullish", "causal_aspects_used": [],
            })),
        ]
        agent = make_agent(llm=llm, max_loops=1)
        count = 1 if low_coverage else 3
        chunks = [f"sentiment {index}" for index in range(count)]
        sources = [
            {
                "ticker": "NVDA",
                "published_at": f"2026-06-{30 - index:02d}T20:00:00Z",
                "dataset_version": "sentiment-fixture-v1",
                "url": f"https://source{index}.example/article",
            }
            for index in range(count)
        ]
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({
                "chunks": chunks, "sources_metadata": sources,
                "unique_document_count": count,
                "unique_domain_count": count,
                "evidence_independence": 1.0,
            }),
            make_tool_result(finbert_payload),
            make_tool_result(vader_payload),
            make_tool_result(fusion_payload),
        ]
        with patch("agents.sentiment_agent.ClientSession") as session_cls, \
             patch("agents.sentiment_agent.stdio_client") as stdio:
            session_cls.return_value.__aenter__.return_value = session
            session_cls.return_value.__aexit__.return_value = None
            stdio.return_value.__aenter__.return_value = (None, None)
            stdio.return_value.__aexit__.return_value = None
            return await agent.run(make_shared_state())

    @pytest.mark.asyncio
    async def test_full_agent_exposes_one_model_failure_in_typed_features(self):
        result = await self._run_typed_status_scenario(
            finbert_payload={
                "status": "VALID", "total_chunks": 3,
                "chunk_scores": [
                    {"bullish_prob": 0.8, "bearish_prob": 0.1},
                    {"bullish_prob": 0.7, "bearish_prob": 0.2},
                    {"bullish_prob": 0.6, "bearish_prob": 0.2},
                ],
            },
            vader_payload={"status": "ERROR", "total_chunks": 0},
            fusion_payload={
                "status": "VALID", "score": 0.5, "confidence": 0.5,
                "polarity": "BULLISH",
            },
        )

        slot = result["sentiment_features"]
        assert slot["values"]["aggregate_7d"] is not None
        assert slot["degraded"] is True
        assert slot["values"]["availability"]["vader"] is False
        assert "VADER_ERROR" in slot["degraded_flags"]
        assert "vader" not in slot["values"]["model_versions"]

    @pytest.mark.asyncio
    async def test_full_agent_exposes_fusion_failure_in_typed_features(self):
        result = await self._run_typed_status_scenario(
            finbert_payload={
                "status": "VALID", "total_chunks": 3,
                "chunk_scores": [
                    {"bullish_prob": 0.8, "bearish_prob": 0.1},
                    {"bullish_prob": 0.7, "bearish_prob": 0.2},
                    {"bullish_prob": 0.6, "bearish_prob": 0.2},
                ],
            },
            vader_payload={
                "status": "VALID", "total_chunks": 3,
                "chunk_scores": [{"compound": 0.4}, {"compound": 0.3}, {"compound": 0.2}],
            },
            fusion_payload={"status": "ERROR", "score": None},
        )

        slot = result["sentiment_features"]
        assert slot["values"]["aggregate_7d"] is not None
        assert slot["degraded"] is True
        assert slot["values"]["availability"]["fusion"] is False
        assert "FUSION_ERROR" in slot["degraded_flags"]
        assert "fusion" not in slot["values"]["model_versions"]

    @pytest.mark.asyncio
    async def test_full_agent_propagates_low_coverage_in_typed_features(self):
        result = await self._run_typed_status_scenario(
            finbert_payload={
                "status": "VALID", "total_chunks": 1,
                "chunk_scores": [{"bullish_prob": 0.8, "bearish_prob": 0.1}],
            },
            vader_payload={
                "status": "VALID", "total_chunks": 1,
                "chunk_scores": [{"compound": 0.4}],
            },
            fusion_payload={
                "status": "VALID", "score": 0.5, "confidence": 0.5,
                "polarity": "BULLISH",
            },
            low_coverage=True,
        )

        slot = result["sentiment_features"]
        assert slot["values"]["aggregate_7d"] is not None
        assert slot["missing"] is False
        assert slot["degraded"] is True
        assert slot["values"]["availability"]["coverage_quality"] is False
        assert slot["values"]["availability"]["selected_attempt_quality"] is False
        assert "LOW_COVERAGE" in slot["degraded_flags"]

    @pytest.mark.asyncio
    async def test_full_agent_private_statuses_and_polarity_never_cross_boundaries(
        self, caplog,
    ):
        private = "PRIVATE_SENTINEL"
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({
                "retrieval_query": "q", "ticker": "NVDA", "days_back": 30,
            })),
            make_llm_response(json.dumps({
                "overall_sentiment": None, "risk_flags": [],
                "causal_aspects_used": [],
            })),
        ]
        agent = make_agent(llm=llm, max_loops=1)
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({
                "chunks": ["one", "two", "three"],
                "sources_metadata": [
                    {
                        "ticker": "NVDA", "published_at": "2026-06-30T20:00:00Z",
                        "dataset_version": "sentiment-fixture-v1",
                        "url": "https://one.example/article",
                    },
                    {
                        "ticker": "NVDA", "published_at": "2026-06-29T20:00:00Z",
                        "dataset_version": "sentiment-fixture-v1",
                        "url": "https://two.example/article",
                    },
                    {
                        "ticker": "NVDA", "published_at": "2026-06-28T20:00:00Z",
                        "dataset_version": "sentiment-fixture-v1",
                        "url": "https://three.example/article",
                    },
                ],
                "unique_document_count": 3, "unique_domain_count": 3,
                "evidence_independence": 1.0,
            }),
            make_tool_result({
                "status": "VALID", "total_chunks": 3,
                "chunk_scores": [
                    {"bullish_prob": 0.8, "bearish_prob": 0.1},
                    {"bullish_prob": 0.7, "bearish_prob": 0.2},
                    {"bullish_prob": 0.6, "bearish_prob": 0.2},
                ],
            }),
            make_tool_result({
                "status": "VALID", "total_chunks": 3,
                "chunk_scores": [
                    {"compound": 0.4}, {"compound": 0.3}, {"compound": 0.2},
                ],
            }),
            make_tool_result({
                "status": f"{private}!", "score": 0.5, "confidence": 0.9,
                "polarity": private,
            }),
        ]

        with patch("agents.sentiment_agent.ClientSession") as session_cls, \
             patch("agents.sentiment_agent.stdio_client") as stdio, \
             patch("agents.sentiment_agent._publish_forecast") as forecast_publish, \
             patch("agents.sentiment_agent._publish_progress") as progress_publish, \
             patch("agents.sentiment_agent.capture_exception") as captured, \
             caplog.at_level("INFO"):
            session_cls.return_value.__aenter__.return_value = session
            session_cls.return_value.__aexit__.return_value = None
            stdio.return_value.__aenter__.return_value = (None, None)
            stdio.return_value.__aexit__.return_value = None
            result = await agent.run(make_shared_state())

        slot = result["sentiment_features"]
        observable = json.dumps({
            "slot": slot,
            "summary": result["sentiment_analysis_summary"],
            "logs": caplog.text,
            "progress": [
                {"args": call.args, "kwargs": call.kwargs}
                for publisher in (forecast_publish, progress_publish)
                for call in publisher.call_args_list
            ],
            "trace": _trace_outputs("run")(result),
            "sentry": [
                {"args": call.args, "kwargs": call.kwargs}
                for call in captured.call_args_list
            ],
        }, default=str, sort_keys=True)

        assert private not in observable
        assert slot["degraded"] is True
        assert slot["values"]["polarity"] is None
        assert slot["values"]["model_statuses"] == {
            "finbert": "VALID", "vader": "VALID",
            "fusion": "UNKNOWN", "selected_attempt": "ERROR",
        }
        assert "finbert" in slot["values"]["model_versions"]
        assert "vader" in slot["values"]["model_versions"]
        assert "fusion" not in slot["values"]["model_versions"]
        assert slot["values"]["availability"]["fusion"] is False
        assert result["sentiment_analysis_summary"]["polarity"] is None
        assert "POLARITY_UNKNOWN" in slot["degraded_flags"]
        assert "FUSION_STATUS_UNKNOWN" in slot["degraded_flags"]


class TestStage06ObservabilitySentinels:
    @pytest.mark.asyncio
    async def test_mcp_failure_span_and_executor_never_emit_private_payload(
        self, caplog,
    ):
        private = "PRIVATE_SENTINEL"
        context = ForecastContext.from_dict(TEST_FORECAST_CONTEXT)
        session = AsyncMock()
        session.call_tool.side_effect = RuntimeError(private)

        with caplog.at_level("INFO"):
            result = await _sentiment_call_tool(
                session,
                "retrieve_social_data",
                {"query": private, "forecast_context": TEST_FORECAST_CONTEXT,
                 "run_id": "sentiment-test-run"},
                context=context,
                run_id="sentiment-test-run",
            )

        trace_input = _mcp_trace_inputs({
            "session": session, "tool": "retrieve_social_data",
            "arguments": {"query": private},
            "context": context, "run_id": "sentiment-test-run",
        })
        trace_output = _mcp_trace_outputs(result)
        serialized = json.dumps({
            "input": trace_input,
            "output": trace_output,
            "payload": result.content[0].text,
            "logs": caplog.text,
        })
        assert private not in serialized
        assert trace_output["error_code"] == "SENTIMENT_TOOL_FAIL_CLOSED"
        assert trace_output["outcome"] == "fail_closed"
        assert math.isfinite(trace_output["latency_ms"])
        assert trace_output["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_query_and_free_form_error_are_absent_from_logs_progress_and_state(
        self, caplog,
    ):
        private = "PRIVATE_SENTINEL"
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"error": private})
        state = make_agent_state(
            shared=make_shared_state(task_query=private)
        )

        with patch("agents.sentiment_agent._publish_forecast") as forecast_publish, \
        patch("agents.sentiment_agent._publish_progress") as progress_publish, \
             patch("agents.sentiment_agent.capture_exception") as captured, \
             caplog.at_level("INFO"):
            await agent._execute_sentiment_pipeline(
                session, state,
                {"retrieval_query": private, "ticker": "NVDA", "days_back": 30},
            )

        observable = json.dumps({
            "logs": caplog.text,
            "progress": [
                {"args": call.args, "kwargs": call.kwargs}
                for publisher in (forecast_publish, progress_publish)
                for call in publisher.call_args_list
            ],
            "errors": state["extraction_errors"],
            "sentry": [
                {"args": [str(value) for value in call.args], "kwargs": call.kwargs}
                for call in captured.call_args_list
            ],
        }, default=str)
        assert private not in observable
        assert state["extraction_errors"] == [
            "retrieve_social_data: SENTIMENT_TOOL_DATA_UNAVAILABLE"
        ]
        assert progress_publish.call_count >= 4

    def test_complete_lifecycle_matrix_is_measured_versioned_and_payload_free(self):
        context = ForecastContext.from_dict(TEST_FORECAST_CONTEXT)
        private = "PRIVATE_SENTINEL"
        observed = []
        for boundary in sorted(SENTIMENT_LIFECYCLE_BOUNDARIES):
            event = sentiment_lifecycle_event(
                forecast_context=context,
                run_id="sentiment-test-run",
                boundary=boundary,
                started_at=time.perf_counter(),
                outcome="success",
                counts={"evidence_count": 2, "result_count": 1},
                model_versions={"finbert": private},
            )
            observed.append(event)

        assert {event["boundary"] for event in observed} == set(
            SENTIMENT_LIFECYCLE_BOUNDARIES
        )
        for event in observed:
            assert event["stage"] == "06"
            assert event["stage_revision"] == 1
            assert event["attempt"] == 4
            assert event["agent"] == "sentiment"
            assert event["implementation_version"] == "stage06-r1-a4"
            assert event["run_id"] == "sentiment-test-run"
            assert event["ticker"] == "NVDA"
            assert math.isfinite(event["latency_ms"])
            assert event["latency_ms"] >= 0
            assert event["evidence_count"] == 2
            assert private not in json.dumps(event)


def test_ui_progress_maps_internal_boundaries_without_false_completion():
    context = ForecastContext.from_dict(TEST_FORECAST_CONTEXT)
    common = {
        "session_id": "ui-session", "context": context,
        "run_id": "sentiment-test-run", "started_at": time.perf_counter(),
        "outcome": "success", "evidence_count": 2,
    }
    with patch("agents.sentiment_agent._publish_progress") as publish, \
         patch("agents.sentiment_agent._publish_forecast") as publish_forecast:
        _publish_lifecycle_progress(boundary="brain_plan", **common)
        _publish_lifecycle_progress(boundary="finbert_inference", **common)
        _publish_lifecycle_progress(
            boundary="feature_aggregation_and_pre_shared_state", **common
        )
        _publish_lifecycle_progress(boundary="whole_sentiment_run", **common)

    assert [call.args[1] for call in publish.call_args_list] == [
        "agent_brain", "agent_tool_result", "dispatch_end",
    ]
    assert publish.call_args_list[1].kwargs["detail"]["tool"] == "analyze_finbert"
    publish_forecast.assert_called_once()
    assert publish_forecast.call_args.args[1] == "feature_complete"
