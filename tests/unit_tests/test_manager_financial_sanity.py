from unittest.mock import AsyncMock, MagicMock
import json

import pytest

from agents.manager_agent import ManagerAgent


def make_manager(llm):
    memory = MagicMock()
    memory.get_messages.return_value = []
    return ManagerAgent(
        research_agent=MagicMock(), financial_agent=MagicMock(),
        sentiment_agent=MagicMock(), memory=memory, llm_client=llm,
    )


@pytest.mark.asyncio
async def test_financial_critical_failure_blocks_before_llm():
    llm = AsyncMock()
    manager = make_manager(llm)
    state = {"financial_metrics_summary": {
        "ticker": "NVDA",
        "dcf_valuation": {"base": {
            "wacc_pct": 3.0,
            "assumptions": {"terminal_growth_rate_pct": 3.0},
        }},
    }}
    feedback = await manager._brain_evaluate("FinancialAnalystAgent", state, {"short_term": {}})
    assert feedback.passed is False
    assert feedback.score <= 49
    assert feedback.next_action == "rerun_financial"
    llm.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_financial_evaluator_api_error_fails_closed():
    llm = AsyncMock()
    llm.messages.create.side_effect = RuntimeError("down")
    manager = make_manager(llm)
    state = {"financial_metrics_summary": {"ticker": "NVDA"}}
    feedback = await manager._brain_evaluate("FinancialAnalystAgent", state, {"short_term": {}})
    assert feedback.passed is False
    assert feedback.score == 0
    assert feedback.next_action == "rerun_financial"


@pytest.mark.asyncio
async def test_manager_receives_complete_valuation_assumptions():
    llm = AsyncMock()
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [MagicMock(text=json.dumps({
        "passed": True, "score": 90, "issues": [], "next_action": "run_sentiment",
    }))]
    llm.messages.create.return_value = response
    manager = make_manager(llm)
    state = {"financial_metrics_summary": {
        "ticker": "NVDA", "validation_passed": True,
        "forecast": {"assumptions": {"historical_revenue_cagr_pct": 20}},
        "dcf_valuation": {"base": {
            "wacc_components": {"cost_of_equity_pct": 9, "after_tax_cost_of_debt_pct": 4},
            "equity_bridge": {"cash": 100, "total_debt": 50, "complete": True},
            "equity_bridge_complete": True,
            "assumptions": {"terminal_growth_rate_pct": 2.5},
        }},
    }}
    await manager._brain_evaluate("FinancialAnalystAgent", state, {"short_term": {}})
    content = manager._memory.add_message.call_args_list[0].kwargs["content"]
    assert '"wacc_components"' in content
    assert '"equity_bridge"' in content
    assert '"terminal_growth_rate_pct"' in content
    assert '"forecast"' in content
