"""
Tests for: agents/financial_agent.py
Phase: 6 — Agents (2nd)

Mocking strategy:
  - anthropic.Anthropic injected via `llm_client` (DI point, no real API calls).
  - MCP session is a plain AsyncMock injected directly as the `session`
    argument to _execute_data_extraction/_execute_ratio_computation (both
    take `session` as an explicit parameter — no need to patch stdio_client
    for those). Only the full run() pipeline test needs to patch
    stdio_client/ClientSession since run() opens its own session internally.

KNOWN BUG (see TestInitLoggingBug): __init__'s log.info() call passes only
1 positional arg for a 2-placeholder format string
(`"...model=%s, server=%s", model`) — this raises TypeError the moment the
log record is actually formatted (i.e. whenever root logging level <= INFO).
"""
import json
import logging
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from agents.financial_agent import FinancialAnalystAgent


def make_llm_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def make_tool_result(payload: dict):
    result = MagicMock()
    result.content = [MagicMock(text=json.dumps(payload))]
    return result


def make_agent(llm=None, max_loops=3):
    llm = llm or MagicMock()
    return FinancialAnalystAgent(llm_client=llm, max_loops=max_loops)


def make_shared_state(task_query="Analyze NVDA", **kw):
    state = {"task_query": task_query, "manager_directives": {}}
    state.update(kw)
    return state


def make_agent_state(shared=None, **overrides):
    state = {
        "messages": [], "raw_numerical_data": {}, "calculated_ratios": {},
        "loop_counter": 0, "validation_feedback": "", "is_complete": False,
        "shared_manager_ref": shared or make_shared_state(),
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_injected_llm_used(self):
        llm = MagicMock()
        agent = FinancialAnalystAgent(llm_client=llm)
        assert agent._llm is llm

    def test_default_max_loops(self):
        agent = make_agent()
        assert agent._default_max_loops == 3


class TestInitLoggingBug:
    def test_init_log_call_has_mismatched_format_args_known_bug(self, caplog):
        """
        KNOWN BUG: `log.info("...model=%s, server=%s", model)` supplies only
        one positional arg for a format string with TWO %s placeholders.
        Python's logging module defers string formatting until the record
        is actually emitted, so this raises a TypeError as soon as the
        agent logger's effective level is INFO or lower (e.g. under caplog,
        or once api/main.py configures logging.basicConfig(INFO) at startup)
        — it does NOT raise at call time under default WARNING-level logging.

        This test forces INFO-level capture to prove the bug exists. Once
        fixed (pass `model` for both placeholders, e.g. the server params
        string too), update this test to assert no error is raised.
        """
        with caplog.at_level(logging.INFO, logger="financial-analyst-agent"):
            with pytest.raises(TypeError):
                FinancialAnalystAgent(llm_client=MagicMock())


# ---------------------------------------------------------------------------
# _extract_ticker
# ---------------------------------------------------------------------------

class TestExtractTicker:
    def test_explicit_directive_takes_priority(self):
        agent = make_agent()
        ticker = agent._extract_ticker("Some query about AAPL", {"ticker": "nvda"})
        assert ticker == "NVDA"

    def test_falls_back_to_regex_on_task_query(self):
        agent = make_agent()
        ticker = agent._extract_ticker("Analyze NVDA fundamentals", {})
        assert ticker == "NVDA"

    def test_stop_words_are_skipped(self):
        agent = make_agent()
        ticker = agent._extract_ticker("THE AI sector and EPS for Q1", {})
        assert ticker is None

    def test_returns_none_when_nothing_found(self):
        agent = make_agent()
        ticker = agent._extract_ticker("analyze the market today", {})
        assert ticker is None

    def test_first_valid_candidate_wins(self):
        agent = make_agent()
        ticker = agent._extract_ticker("Compare NVDA against AMD", {})
        assert ticker == "NVDA"


# ---------------------------------------------------------------------------
# _execute_data_extraction
# ---------------------------------------------------------------------------

class TestExecuteDataExtraction:
    @pytest.mark.asyncio
    async def test_no_ticker_records_error_and_returns_early(self):
        agent = make_agent()
        session = AsyncMock()
        state = make_agent_state(shared=make_shared_state(task_query="no ticker here"))

        await agent._execute_data_extraction(session, state)

        assert "Could not determine ticker" in state["raw_numerical_data"]["extraction_errors"]
        session.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_calls_three_tools_and_stores_results(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"pe_ratio": 20, "current_price": 100}),
            make_tool_result({"annual_revenue": [{"revenue": 100}]}),
            make_tool_result({"total_assets": [{"value": 500}]}),
        ]
        state = make_agent_state()

        await agent._execute_data_extraction(session, state)

        assert state["raw_numerical_data"]["ticker"] == "NVDA"
        assert state["raw_numerical_data"]["yahoo_ratios"]["pe_ratio"] == 20
        assert state["raw_numerical_data"]["revenue_growth"]["annual_revenue"][0]["revenue"] == 100
        assert state["raw_numerical_data"]["xbrl_financials"]["total_assets"][0]["value"] == 500
        assert session.call_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_individual_tool_error_field_recorded_but_others_continue(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"error": "rate limited"}),
            make_tool_result({"annual_revenue": []}),
            make_tool_result({"total_assets": []}),
        ]
        state = make_agent_state()

        await agent._execute_data_extraction(session, state)

        errors = state["raw_numerical_data"]["extraction_errors"]
        assert any("tool_get_financial_ratios error" in e for e in errors)
        assert state["raw_numerical_data"]["revenue_growth"]["annual_revenue"] == []

    @pytest.mark.asyncio
    async def test_tool_exception_does_not_abort_remaining_tools(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            RuntimeError("yahoo down"),
            make_tool_result({"annual_revenue": [{"revenue": 1}]}),
            make_tool_result({}),
        ]
        state = make_agent_state()

        await agent._execute_data_extraction(session, state)

        assert state["raw_numerical_data"]["yahoo_ratios"] == {}
        assert any("tool_get_financial_ratios exception" in e for e in state["raw_numerical_data"]["extraction_errors"])
        assert state["raw_numerical_data"]["revenue_growth"]["annual_revenue"] == [{"revenue": 1}]


# ---------------------------------------------------------------------------
# _execute_ratio_computation
# ---------------------------------------------------------------------------

class TestExecuteRatioComputation:
    @pytest.mark.asyncio
    async def test_pe_computed_directly_when_price_and_eps_present(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"pe_ratio": 25, "interpretation": "fairly_valued"})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {"current_price": 100, "eps_trailing": 4},
            "revenue_growth": {}, "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        called_tool = session.call_tool.call_args_list[0][0][0]
        assert called_tool == "tool_calc_pe"
        assert state["calculated_ratios"]["pe"]["pe_ratio"] == 25

    @pytest.mark.asyncio
    async def test_pe_falls_back_to_yahoo_sourced_when_eps_missing(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"composite_score": True})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {"pe_ratio": 30, "current_price": None, "eps_trailing": None},
            "revenue_growth": {}, "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        assert state["calculated_ratios"]["pe"]["pe_ratio"] == 30
        assert state["calculated_ratios"]["pe"]["interpretation"] == "sourced_from_yahoo"

    @pytest.mark.asyncio
    async def test_roe_insufficient_data_when_no_net_income(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {}, "revenue_growth": {"annual_net_income": []}, "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        assert state["calculated_ratios"]["roe"] == {"roe_pct": None, "interpretation": "insufficient_data"}

    @pytest.mark.asyncio
    async def test_roe_computed_from_xbrl_assets_minus_liabilities(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"roe_pct": 20.0})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {},
            "revenue_growth": {"annual_net_income": [{"net_income": 100}]},
            "xbrl_financials": {
                "total_assets": [{"value": 1000}],
                "total_liabilities": [{"value": 400}],
            },
        })

        await agent._execute_ratio_computation(session, state)

        tool_calls = [c[0][0] for c in session.call_tool.call_args_list]
        assert "tool_calc_roe" in tool_calls
        roe_call_args = next(
            c[1]["arguments"] for c in session.call_tool.call_args_list if c[0][0] == "tool_calc_roe"
        )
        assert roe_call_args["shareholders_equity"] == 600

    @pytest.mark.asyncio
    async def test_de_ratio_sourced_from_yahoo_skips_mcp_call(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {"de_ratio": 0.5},
            "revenue_growth": {}, "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        assert state["calculated_ratios"]["de_ratio"] == {
            "de_ratio": 0.5, "interpretation": "sourced_from_yahoo",
        }
        called_tools = [c[0][0] for c in session.call_tool.call_args_list]
        assert "tool_calc_debt_to_equity" not in called_tools

    @pytest.mark.asyncio
    async def test_cagr_insufficient_history_with_one_data_point(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {}, "revenue_growth": {"annual_revenue": [{"revenue": 100}]},
            "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        assert state["calculated_ratios"]["cagr"] == {
            "cagr_pct": None, "interpretation": "insufficient_history",
        }

    @pytest.mark.asyncio
    async def test_cagr_computed_with_oldest_and_newest_revenue(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"cagr_pct": 25.0})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {},
            "revenue_growth": {"annual_revenue": [
                {"revenue": 150}, {"revenue": 120}, {"revenue": 100},
            ]},
            "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        cagr_call_args = next(
            c[1]["arguments"] for c in session.call_tool.call_args_list if c[0][0] == "tool_calc_cagr"
        )
        assert cagr_call_args["start_value"] == 100
        assert cagr_call_args["end_value"] == 150
        assert cagr_call_args["years"] == 2.0

    @pytest.mark.asyncio
    async def test_composite_score_always_called_last(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"score": 70, "grade": "B"})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {}, "revenue_growth": {}, "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        called_tools = [c[0][0] for c in session.call_tool.call_args_list]
        assert called_tools[-1] == "tool_calc_composite_score"
        assert state["calculated_ratios"]["composite_score"]["score"] == 70


# ---------------------------------------------------------------------------
# _check_data_quality
# ---------------------------------------------------------------------------

class TestCheckDataQuality:
    def test_preflight_guard_skips_llm_when_both_empty(self):
        llm = MagicMock()
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert "DATA PRESENCE" in result["failed"]
        llm.messages.create.assert_not_called()

    def test_complete_verdict_clears_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "is_complete": True, "score": 90, "passed": ["DATA PRESENCE"], "failed": [], "issues": [],
        }))
        agent = make_agent(llm=llm)
        state = make_agent_state(raw_numerical_data={"ticker": "NVDA"}, calculated_ratios={"pe": {}})

        result = agent._check_data_quality(state)

        assert result["is_complete"] is True
        assert result["feedback"] == ""

    def test_incomplete_verdict_preserves_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "is_complete": False, "score": 30, "failed": ["VALUATION SANITY"],
            "issues": ["pe too high"], "feedback": "re-check pe ratio",
        }))
        agent = make_agent(llm=llm)
        state = make_agent_state(raw_numerical_data={"ticker": "NVDA"}, calculated_ratios={"pe": {}})

        result = agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert result["feedback"] == "re-check pe ratio"

    def test_markdown_fenced_json_is_stripped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            "```json\n" + json.dumps({"is_complete": True, "score": 90}) + "\n```"
        )
        agent = make_agent(llm=llm)
        state = make_agent_state(raw_numerical_data={"ticker": "NVDA"}, calculated_ratios={"pe": {}})

        result = agent._check_data_quality(state)
        assert result["is_complete"] is True

    def test_invalid_json_returns_incomplete_with_parse_error_marker(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("not valid json at all")
        agent = make_agent(llm=llm)
        state = make_agent_state(raw_numerical_data={"ticker": "NVDA"}, calculated_ratios={"pe": {}})

        result = agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert "CHECKER PARSE ERROR" in result["failed"]

    def test_llm_api_exception_returns_incomplete_with_api_error_marker(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        state = make_agent_state(raw_numerical_data={"ticker": "NVDA"}, calculated_ratios={"pe": {}})

        result = agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert "CHECKER API ERROR" in result["failed"]


# ---------------------------------------------------------------------------
# _brain
# ---------------------------------------------------------------------------

class TestBrain:
    def test_valid_json_plan_parsed_and_messages_appended(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "plan": "Extract financials", "priority_tools": ["tool_get_financial_ratios"],
        }))
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = agent._brain(state)

        assert result["plan"] == "Extract financials"
        assert result["priority_tools"] == ["tool_get_financial_ratios"]
        assert len(state["messages"]) == 2

    def test_non_json_response_falls_back_to_default_tools(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("just plain text, not json")
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = agent._brain(state)

        assert result["plan"] == "just plain text, not json"
        assert "tool_get_financial_ratios" in result["priority_tools"]

    def test_api_exception_returns_default_plan_without_raising(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = agent._brain(state)

        assert "Default plan" in result["plan"]
        assert len(result["priority_tools"]) == 3

    def test_feedback_included_in_prompt(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({"plan": "x", "priority_tools": []}))
        agent = make_agent(llm=llm)
        state = make_agent_state(validation_feedback="missing revenue data")

        agent._brain(state)

        sent_messages = llm.messages.create.call_args.kwargs["messages"]
        assert "missing revenue data" in sent_messages[0]["content"]


# ---------------------------------------------------------------------------
# run() — full pipeline
# ---------------------------------------------------------------------------

class TestRunFullPipeline:
    @pytest.mark.asyncio
    async def test_completes_after_checker_passes_and_assembles_summary(self):
        llm = MagicMock()
        llm.messages.create.side_effect = [
            make_llm_response(json.dumps({"plan": "extract", "priority_tools": []})),
            make_llm_response(json.dumps({"is_complete": True, "score": 90, "passed": [], "failed": [], "issues": []})),
        ]
        agent = make_agent(llm=llm, max_loops=3)

        mock_session = AsyncMock()
        mock_session.call_tool.side_effect = [
            make_tool_result({"pe_ratio": 20, "company_name": "NVIDIA", "current_price": 100}),
            make_tool_result({"annual_revenue": [{"revenue": 100}]}),
            make_tool_result({"total_assets": [{"value": 1}]}),
            make_tool_result({"pe_ratio": 20}),
            make_tool_result({"score": 80, "grade": "B"}),
        ]

        with patch("agents.financial_agent.ClientSession") as mock_session_cls, \
             patch("agents.financial_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            shared = make_shared_state()
            result = await agent.run(shared)

        summary = result["financial_metrics_summary"]
        assert summary["ticker"] == "NVDA"
        assert summary["validation_passed"] is True
        assert summary["loop_iterations_used"] == 1

    @pytest.mark.asyncio
    async def test_guardrail_stops_after_max_loops_when_never_complete(self):
        llm = MagicMock()
        brain_resp = make_llm_response(json.dumps({"plan": "x", "priority_tools": []}))
        checker_resp = make_llm_response(json.dumps({
            "is_complete": False, "score": 10, "failed": ["X"], "issues": [], "feedback": "retry",
        }))
        llm.messages.create.side_effect = [brain_resp, checker_resp] * 2
        agent = make_agent(llm=llm, max_loops=2)

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = make_tool_result({})

        with patch("agents.financial_agent.ClientSession") as mock_session_cls, \
             patch("agents.financial_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            result = await agent.run(make_shared_state())

        summary = result["financial_metrics_summary"]
        assert summary["validation_passed"] is False
        assert summary["loop_iterations_used"] == 2

    @pytest.mark.asyncio
    async def test_directive_max_loops_overrides_default(self):
        llm = MagicMock()
        brain_resp = make_llm_response(json.dumps({"plan": "x", "priority_tools": []}))
        checker_resp = make_llm_response(json.dumps({"is_complete": False, "failed": [], "issues": [], "feedback": "x"}))
        llm.messages.create.side_effect = [brain_resp, checker_resp]
        agent = make_agent(llm=llm, max_loops=5)

        mock_session = AsyncMock()
        mock_session.call_tool.return_value = make_tool_result({})

        with patch("agents.financial_agent.ClientSession") as mock_session_cls, \
             patch("agents.financial_agent.stdio_client") as mock_stdio:
            mock_session_cls.return_value.__aenter__.return_value = mock_session
            mock_session_cls.return_value.__aexit__.return_value = None
            mock_stdio.return_value.__aenter__.return_value = (None, None)
            mock_stdio.return_value.__aexit__.return_value = None

            shared = make_shared_state(manager_directives={"ticker": "NVDA", "max_loops": 1})
            result = await agent.run(shared)

        assert result["financial_metrics_summary"]["loop_iterations_used"] == 1