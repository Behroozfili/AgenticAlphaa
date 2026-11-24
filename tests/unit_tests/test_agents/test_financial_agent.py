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
import inspect
import logging
import math
import time
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from agents.financial_agent import (
    FinancialAnalystAgent, _financial_call_tool,
    _financial_mcp_trace_inputs, _financial_mcp_trace_outputs,
)
from financial_test import STAGE5_RUN_ID, stage5_context
from agents.executor_helpers.monthly_financial_features import build_monthly_financial_features
from core.forecast_contract import ForecastContext
from tools.financial_tools import FINANCIAL_LIFECYCLE_BOUNDARIES, financial_lifecycle_event


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
    supplied_directives = kw.pop("manager_directives", {})
    directives = {
        "ticker": "NVDA", "run_id": STAGE5_RUN_ID,
        "forecast_context": stage5_context("NVDA"),
    }
    directives.update(supplied_directives)
    state = {
        "task_query": task_query, "forecast_context": stage5_context("NVDA"),
        "manager_directives": directives,
    }
    state.update(kw)
    return state


def make_agent_state(shared=None, **overrides):
    shared = shared or make_shared_state()
    state = {
        "messages": [], "raw_numerical_data": {}, "calculated_ratios": {},
        "loop_counter": 0, "validation_feedback": "", "is_complete": False,
        "task_query": shared["task_query"],
        "manager_directives": dict(shared["manager_directives"]),
        "intents": [], "extra_tools": [], "rationale": "",
        "shared_manager_ref": shared,
    }
    state.update(overrides)
    return state


def checker_ready_state():
    return make_agent_state(
        raw_numerical_data={"ticker": "NVDA", "peer_comparison": {"peers": [{"ticker": "AMD"}]}},
        calculated_ratios={}, intents=["peer_relative"],
    )


def test_monthly_feature_adapter_is_fixed_compact_and_missing_is_not_zero():
    features = build_monthly_financial_features(
        raw={
            "ticker": "NVDA",
            "yahoo_ratios": {"current_price": 100.0},
            "xbrl_financials": {"revenue_annual": [{
                "value": 1000.0, "period_end": "2025-12-31",
                "filed_at": "2026-02-01", "accession_number": "acc-1",
                "xbrl_concept": "Revenue", "evidence_id": "financial:e1",
            }]},
        },
        calculated={"dcf": {"base": {"intrinsic_value_per_share": 125.0}}},
        forecast_context=stage5_context(), run_id=STAGE5_RUN_ID,
    )
    slot = features.to_state_slot()
    assert slot["values"]["valuation_gap"] == 0.25
    assert slot["values"]["roe_pct"] is None
    assert "quality_profitability" in slot["missing_flags"]
    assert slot["values"]["dcf_semantics"] == "LONG_TERM_VALUATION_FEATURE_NOT_21_DAY_FORECAST"
    serialized = json.dumps(slot)
    assert "Revenue" not in serialized and "1000.0" not in serialized
    assert slot["evidence_ids"] == ["financial:e1"]


@pytest.mark.asyncio
async def test_safe_financial_mcp_trace_strips_private_inputs_outputs_and_exceptions(caplog):
    marker = "PRIVATE_SENTINEL"
    context = ForecastContext.from_dict(stage5_context())
    raw_arguments = {
        "ticker": "NVDA", "query": marker, "filing_text": marker,
        "accession_number": marker, "dcf_array": [marker],
        "forecast_context": stage5_context(), "run_id": STAGE5_RUN_ID,
    }
    processed_inputs = _financial_mcp_trace_inputs({
        "tool": "tool_get_financial_ratios", "arguments": raw_arguments,
        "context": context, "run_id": STAGE5_RUN_ID,
    })
    assert marker not in json.dumps(processed_inputs, sort_keys=True)
    assert processed_inputs["stage"] == "05" and processed_inputs["attempt"] == 1
    assert processed_inputs["ticker"] == "NVDA" and processed_inputs["input_count"] > 0

    session = AsyncMock()
    session.call_tool.return_value = make_tool_result({
        "current_price": marker, "facts": [marker], "filing_text": marker,
        "dcf": [marker], "error": None,
    })
    with caplog.at_level(logging.INFO):
        wrapped = await _financial_call_tool(
            session, "tool_get_financial_ratios", raw_arguments,
            context=context, run_id=STAGE5_RUN_ID,
        )
    processed_outputs = _financial_mcp_trace_outputs(wrapped)
    assert marker not in json.dumps(processed_outputs, sort_keys=True)
    assert processed_outputs["outcome"] == "success"
    assert processed_outputs["result_count"] == 1

    session.call_tool.side_effect = RuntimeError(marker)
    with caplog.at_level(logging.INFO):
        failed = await _financial_call_tool(
            session, "tool_get_financial_ratios", raw_arguments,
            context=context, run_id=STAGE5_RUN_ID,
        )
    failed_trace = _financial_mcp_trace_outputs(failed)
    assert failed_trace["outcome"] == "fail_closed"
    assert failed_trace["error_code"] == "FINANCIAL_TOOL_FAIL_CLOSED"
    assert marker not in json.dumps(failed_trace, sort_keys=True)
    assert marker not in failed.content[0].text
    assert marker not in caplog.text

    invalid_session = AsyncMock()
    with pytest.raises(ValueError, match="forecast_context is required"):
        await _financial_call_tool(
            invalid_session, "tool_get_financial_ratios", raw_arguments,
            context=None, run_id=STAGE5_RUN_ID,  # type: ignore[arg-type]
        )
    invalid_session.call_tool.assert_not_called()


def test_all_three_financial_mcp_paths_use_only_safe_wrapper():
    methods = (
        FinancialAnalystAgent._execute_sec_filing_chain,
        FinancialAnalystAgent._execute_data_extraction,
        FinancialAnalystAgent._execute_ratio_computation,
    )
    for method in methods:
        source = inspect.getsource(method)
        assert "_financial_call_tool(" in source
        assert "traced_call_tool" not in source


def test_stage05_lifecycle_matrix_is_complete_measured_and_payload_free(caplog):
    marker = "PRIVATE_SENTINEL"
    expected = {
        "sec_submissions", "sec_companyfacts", "sec_filing_selection_and_text",
        "sec_restatement_gate", "yahoo_adjusted_history",
        "yahoo_current_only_fail_closed", "financial_mcp_ratio_dispatcher",
        "monthly_financial_adapter", "agent_extraction", "agent_calculation",
        "agent_checker", "pre_shared_state_gate",
    }
    assert FINANCIAL_LIFECYCLE_BOUNDARIES == expected
    context = ForecastContext.from_dict(stage5_context())
    events = []
    with caplog.at_level(logging.INFO, logger="financial-lifecycle"):
        for boundary in sorted(expected):
            events.append(financial_lifecycle_event(
                context=context, run_id=STAGE5_RUN_ID, boundary=boundary,
                started_at=time.perf_counter(), outcome=marker,
                counts={
                    "input_count": marker, "result_count": float("inf"),
                    "eligible_count": -99, "rejected_count": 10**20,
                },
                missing=True, degraded=True, error_code=marker,
            ))
    assert {event["boundary"] for event in events} == expected
    for event in events:
        assert event["stage"] == "05" and event["stage_revision"] == 1
        assert event["attempt"] == 1 and event["agent"] == "financial"
        assert math.isfinite(event["latency_ms"]) and event["latency_ms"] >= 0
        assert event["outcome"] == "fail_closed"
        assert event["error_code"] == "FINANCIAL_OPERATION_FAIL_CLOSED"
        assert 0 <= event["rejected_count"] <= 1_000_000
        assert marker not in json.dumps(event, sort_keys=True)
    assert marker not in caplog.text

    callsite_sources = {
        "sec_submissions": Path("tools/financial_tools/sec_edgar.py"),
        "sec_companyfacts": Path("tools/financial_tools/sec_edgar.py"),
        "sec_filing_selection_and_text": Path("tools/financial_tools/sec_edgar.py"),
        "sec_restatement_gate": Path("tools/financial_tools/sec_edgar.py"),
        "yahoo_adjusted_history": Path("tools/financial_tools/yahoo_finance.py"),
        "yahoo_current_only_fail_closed": Path("tools/financial_tools/__init__.py"),
        "financial_mcp_ratio_dispatcher": Path("agents/financial_agent.py"),
        "monthly_financial_adapter": Path("agents/executor_helpers/monthly_financial_features.py"),
        "agent_extraction": Path("agents/financial_agent.py"),
        "agent_calculation": Path("agents/financial_agent.py"),
        "agent_checker": Path("agents/financial_agent.py"),
        "pre_shared_state_gate": Path("agents/financial_agent.py"),
    }
    for boundary, path in callsite_sources.items():
        assert boundary in path.read_text(encoding="utf-8")


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
    def test_init_log_call_no_longer_raises_format_mismatch(self, caplog):
        """
        Initialisation emits a fixed event code without model or server path.
        """
        with caplog.at_level(logging.INFO, logger="financial-analyst-agent"):
            FinancialAnalystAgent(llm_client=MagicMock())

        assert any(r.message == "FINANCIAL_AGENT_INITIALIZED" for r in caplog.records)
        assert "model=" not in caplog.text and "server_script=" not in caplog.text


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
    async def test_missing_execution_context_fails_before_provider(self):
        agent = make_agent()
        session = AsyncMock()
        state = make_agent_state(manager_directives={})

        with pytest.raises(ValueError, match="forecast_context is required"):
            await agent._execute_data_extraction(session, state)
        session.call_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_happy_path_calls_overview_tools_and_stores_results(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"pe_ratio": 20, "current_price": 100}),
            make_tool_result({"annual_revenue": [{"revenue": 100}]}),
            make_tool_result({"total_assets": [{"value": 500}]}),
            make_tool_result({"total_equity": 300}),
        ]
        state = make_agent_state()

        await agent._execute_data_extraction(session, state)

        assert state["raw_numerical_data"]["ticker"] == "NVDA"
        assert state["raw_numerical_data"]["yahoo_ratios"]["pe_ratio"] == 20
        assert state["raw_numerical_data"]["revenue_growth"]["annual_revenue"][0]["revenue"] == 100
        assert state["raw_numerical_data"]["xbrl_financials"]["total_assets"][0]["value"] == 500
        assert session.call_tool.call_count == 4
        for call in session.call_tool.call_args_list:
            arguments = call.args[1]
            assert arguments["run_id"] == STAGE5_RUN_ID
            assert arguments["forecast_context"]["ticker"] == "NVDA"

    @pytest.mark.asyncio
    async def test_individual_tool_error_field_recorded_but_others_continue(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            make_tool_result({"error": "rate limited"}),
            make_tool_result({"annual_revenue": []}),
            make_tool_result({"total_assets": []}),
            make_tool_result({}),
        ]
        state = make_agent_state()

        await agent._execute_data_extraction(session, state)

        errors = state["raw_numerical_data"]["extraction_errors"]
        assert "FINANCIAL_TOOL_DATA_UNAVAILABLE" in errors
        assert "rate limited" not in json.dumps(errors)
        assert state["raw_numerical_data"]["revenue_growth"]["annual_revenue"] == []

    @pytest.mark.asyncio
    async def test_tool_exception_does_not_abort_remaining_tools(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.side_effect = [
            RuntimeError("yahoo down"),
            make_tool_result({"annual_revenue": [{"revenue": 1}]}),
            make_tool_result({}),
            make_tool_result({}),
        ]
        state = make_agent_state()

        await agent._execute_data_extraction(session, state)

        assert state["raw_numerical_data"]["yahoo_ratios"] == {}
        assert "FINANCIAL_TOOL_FAIL_CLOSED" in state["raw_numerical_data"]["extraction_errors"]
        assert "yahoo down" not in json.dumps(state["raw_numerical_data"]["extraction_errors"])
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

        assert state["calculated_ratios"]["roe"]["roe_pct"] is None
        assert state["calculated_ratios"]["roe"]["validation_status"] == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_roe_computed_from_xbrl_assets_minus_liabilities(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"roe_pct": 20.0})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {},
            "revenue_growth": {"annual_net_income": [{"net_income": 100, "year": 2025}]},
            "xbrl_financials": {
                "total_assets": [{"value": 1000, "period_end": "2025-12-31", "unit": "USD", "metric_scope": "instant", "validation_status": "VALID_PRIMARY"}],
                "total_liabilities": [{"value": 400, "period_end": "2025-12-31", "unit": "USD", "metric_scope": "instant", "validation_status": "VALID_PRIMARY"}],
            },
        })

        await agent._execute_ratio_computation(session, state)

        tool_calls = [c[0][0] for c in session.call_tool.call_args_list]
        assert "tool_calc_roe" in tool_calls
        roe_call_args = next(c.args[1] for c in session.call_tool.call_args_list if c.args[0] == "tool_calc_roe")
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

        metric = state["calculated_ratios"]["de_ratio"]
        assert metric["de_ratio"] == 0.5
        assert metric["interpretation"] == "sourced_from_yahoo"
        assert metric["_period"] == "snapshot:mrq"
        assert metric["validation_status"] == "FALLBACK"
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

        assert state["calculated_ratios"]["cagr"]["cagr_pct"] is None
        assert state["calculated_ratios"]["cagr"]["validation_status"] == "UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_cagr_computed_with_oldest_and_newest_revenue(self):
        agent = make_agent()
        session = AsyncMock()
        session.call_tool.return_value = make_tool_result({"cagr_pct": 25.0})
        state = make_agent_state(raw_numerical_data={
            "yahoo_ratios": {},
            "revenue_growth": {"annual_revenue": [
                {"revenue": 150, "year": 2024}, {"revenue": 120, "year": 2023}, {"revenue": 100, "year": 2022},
            ]},
            "xbrl_financials": {},
        })

        await agent._execute_ratio_computation(session, state)

        cagr_call_args = next(c.args[1] for c in session.call_tool.call_args_list if c.args[0] == "tool_calc_cagr")
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
    @pytest.mark.asyncio
    async def test_preflight_guard_skips_llm_when_both_empty(self):
        llm = MagicMock()
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = await agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert "DATA PRESENCE" in result["failed"]
        llm.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_verdict_clears_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "is_complete": True, "score": 90, "passed": ["DATA PRESENCE"], "failed": [], "issues": [],
        }))
        agent = make_agent(llm=llm)
        state = checker_ready_state()

        result = await agent._check_data_quality(state)

        assert result["is_complete"] is True
        assert result["feedback"] == ""

    @pytest.mark.asyncio
    async def test_incomplete_verdict_preserves_feedback(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "is_complete": False, "score": 30, "failed": ["VALUATION SANITY"],
            "issues": ["pe too high"], "feedback": "re-check pe ratio",
        }))
        agent = make_agent(llm=llm)
        state = checker_ready_state()

        result = await agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert result["feedback"] == "re-check pe ratio"

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_is_stripped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(
            "```json\n" + json.dumps({"is_complete": True, "score": 90}) + "\n```"
        )
        agent = make_agent(llm=llm)
        state = checker_ready_state()

        result = await agent._check_data_quality(state)
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_invalid_json_returns_incomplete_with_parse_error_marker(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("not valid json at all")
        agent = make_agent(llm=llm)
        state = checker_ready_state()

        result = await agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert "CHECKER PARSE ERROR" in result["failed"]

    @pytest.mark.asyncio
    async def test_llm_api_exception_returns_incomplete_with_api_error_marker(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        state = checker_ready_state()

        result = await agent._check_data_quality(state)

        assert result["is_complete"] is False
        assert "CHECKER API ERROR" in result["failed"]


# ---------------------------------------------------------------------------
# _brain
# ---------------------------------------------------------------------------

class TestBrain:
    @pytest.mark.asyncio
    async def test_valid_json_plan_parsed_and_messages_appended(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "intents": ["valuation"], "extra_tools": [], "rationale": "Valuation requested.",
        }))
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = await agent._brain(state)

        assert result["intents"] == ["valuation"]
        assert result["extra_tools"] == []
        assert len(state["messages"]) == 2

    @pytest.mark.asyncio
    async def test_non_json_response_falls_back_to_default_tools(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("just plain text, not json")
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = await agent._brain(state)

        assert result["intents"] == ["overview"]
        assert result["extra_tools"] == []

    @pytest.mark.asyncio
    async def test_api_exception_returns_default_plan_without_raising(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("API down")
        agent = make_agent(llm=llm)
        state = make_agent_state()

        result = await agent._brain(state)

        assert result["intents"] == ["overview"]
        assert result["extra_tools"] == []

    @pytest.mark.asyncio
    async def test_feedback_included_in_prompt(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({"intents": ["growth"], "extra_tools": [], "rationale": "Growth."}))
        agent = make_agent(llm=llm)
        state = make_agent_state(validation_feedback="missing revenue data")

        await agent._brain(state)

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
        final_state = make_agent_state(
            raw_numerical_data={
                "ticker": "NVDA",
                "yahoo_ratios": {
                    "company_name": "NVIDIA",
                    "sector": "Technology",
                    "current_price": 100,
                    "market_cap": 1_000,
                    "current_ratio": 2.0,
                },
                "xbrl_financials": {},
            },
            calculated_ratios={
                "pe": {"pe_ratio": 20},
                "roe": {"roe_pct": 30},
                "current_ratio_calc": {"current_ratio": 2.0},
                "composite_score": {"score": 80, "grade": "B"},
            },
            financial_sanity={"passed": True, "score": 90},
            is_complete=True, loop_counter=1,
            intents=["overview"],
        )
        agent._graph = MagicMock(ainvoke=AsyncMock(return_value=final_state))

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
        assert summary["company_name"] == "NVIDIA"
        assert summary["current_price"] == 100
        assert summary["pe_ratio"] == {"pe_ratio": 20}
        assert summary["current_ratio"] == {"current_ratio": 2.0}
        assert summary["composite_score"] == {"score": 80, "grade": "B"}
        assert summary["financial_sanity"] == {"passed": True, "score": 90}

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
