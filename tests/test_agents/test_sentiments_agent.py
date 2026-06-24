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
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from mcp import StdioServerParameters
from agents.sentiment_agent import SentimentAgent


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
    state = {"task_query": task_query, "manager_directives": {}}
    state.update(kw)
    return state


def make_agent_state(shared=None, **overrides):
    state = {
        "messages": [], "retrieved_chunks": [], "sources_metadata": [],
        "finbert_result": {}, "vader_result": {}, "fear_greed_result": {},
        "brain_reasoning": "", "loop_counter": 0, "extraction_errors": [],
        "shared_manager_ref": shared or make_shared_state(),
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
    def test_valid_plan_parsed_and_returned(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "retrieval_query": "NVDA sentiment", "ticker": "NVDA", "days_back": 7, "reasoning": "x",
        }))
        agent = make_agent(llm=llm)
        state = make_agent_state()

        plan = agent._brain_plan(state)

        assert plan["retrieval_query"] == "NVDA sentiment"
        assert plan["ticker"] == "NVDA"
        assert len(state["messages"]) == 2

    def test_markdown_fenced_json_stripped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            "```json\n" + json.dumps({"retrieval_query": "q", "ticker": None, "days_back": 7}) + "\n```"
        )
        agent = make_agent(llm=llm)
        plan = agent._brain_plan(make_agent_state())
        assert plan["retrieval_query"] == "q"

    def test_invalid_json_falls_back_to_default_plan(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("not valid json")
        agent = make_agent(llm=llm)
        state = make_agent_state(shared=make_shared_state(task_query="Analyze NVDA"))

        plan = agent._brain_plan(state)

        assert "NVDA" in plan["retrieval_query"]
        assert plan["days_back"] == 7
        assert "Default plan" in plan["reasoning"]

    def test_api_exception_falls_back_to_default_plan_without_raising(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        plan = agent._brain_plan(make_agent_state())
        assert "Default plan" in plan["reasoning"]

    def test_directive_days_back_used_in_default_plan(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("down")
        agent = make_agent(llm=llm)
        state = make_agent_state(shared=make_shared_state(manager_directives={"days_back": 14}))
        plan = agent._brain_plan(state)
        assert plan["days_back"] == 14


# ---------------------------------------------------------------------------
# _brain_analyze
# ---------------------------------------------------------------------------

class TestBrainAnalyze:
    def test_returns_raw_text_with_fences_stripped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            "```json\n" + json.dumps({"overall_sentiment": "Bullish"}) + "\n```"
        )
        agent = make_agent(llm=llm)
        result = agent._brain_analyze(make_agent_state())
        assert result == json.dumps({"overall_sentiment": "Bullish"})

    def test_api_exception_returns_fallback_json_string(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        result = agent._brain_analyze(make_agent_state())
        parsed = json.loads(result)
        assert parsed["overall_sentiment"] == "Neutral"
        assert "Brain API call failed" in parsed["risk_flags"]

    def test_includes_chunk_count_in_prompt(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response('{"overall_sentiment": "Neutral"}')
        agent = make_agent(llm=llm)
        state = make_agent_state(retrieved_chunks=["a", "b", "c"])
        agent._brain_analyze(state)
        sent_content = llm.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "CHUNK COUNT ANALYSED: 3" in sent_content


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
        assert state["finbert_result"]["label"] == "Neutral"
        assert state["vader_result"]["label"] == "Neutral"
        assert state["fear_greed_result"]["label"] == "Neutral"
        assert session.call_tool.call_count == 1
        assert any("analyze_finbert skipped" in e for e in state["extraction_errors"])
        assert any("score_vader skipped" in e for e in state["extraction_errors"])
        assert any("calculate_fear_greed skipped" in e for e in state["extraction_errors"])

    @pytest.mark.asyncio
    async def test_ticker_omitted_from_args_when_none(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"chunks": [], "sources": []})
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        call_args = session.call_tool.call_args_list[0][1]["arguments"]
        assert "ticker" not in call_args

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

        fg_call_args = session.call_tool.call_args_list[3][1]["arguments"]
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

        assert any("retrieve_social_data error" in e for e in state["extraction_errors"])

    @pytest.mark.asyncio
    async def test_tool_exception_recorded_and_pipeline_continues(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = RuntimeError("connection lost")
        state = make_agent_state()
        plan = {"retrieval_query": "q", "ticker": None, "days_back": 7}

        await agent._execute_sentiment_pipeline(session, state, plan)

        assert any("retrieve_social_data exception" in e for e in state["extraction_errors"])
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


# ---------------------------------------------------------------------------
# run() — full pipeline
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    @pytest.mark.asyncio
    async def test_completes_in_one_loop_when_chunks_found(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"retrieval_query": "q", "ticker": "NVDA", "days_back": 7})),
            make_llm_response(json.dumps({
                "overall_sentiment": "Bullish", "conviction_level": "High",
                "key_signals": [], "model_agreement": "Strong", "narrative": "Good news",
                "risk_flags": [], "data_quality_note": "",
            })),
        ]
        agent = make_agent(llm=llm, max_loops=2)

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [
            make_tool_result({"chunks": ["chunk1"], "sources": []}),
            make_tool_result({"total_chunks": 1, "label": "Bullish"}),
            make_tool_result({"total_chunks": 1, "label": "Bullish"}),
            make_tool_result({"score": 0.5, "label": "Greed"}),
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
        assert summary["total_chunks_analyzed"] == 1

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

    @pytest.mark.asyncio
    async def test_malformed_brain_analysis_json_falls_back_gracefully(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"retrieval_query": "q", "ticker": "NVDA", "days_back": 7})),
            make_llm_response("not valid json from brain analyze"),
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

            result = await agent.run(make_shared_state())

        summary = result["sentiment_analysis_summary"]
        assert summary["overall_sentiment"] == "Neutral"
        assert "Brain analysis JSON parse failed" in summary["risk_flags"]

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