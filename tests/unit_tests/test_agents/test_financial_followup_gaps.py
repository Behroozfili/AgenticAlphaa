import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import agents.financial_agent as financial_module
from agents.financial_agent import (
    FinancialAnalystAgent,
    _checker_requirements,
    _select_sec_filing_type,
)


def state(intents, *, raw=None, calc=None, query="Analyze AMD"):
    return {
        "messages": [], "raw_numerical_data": raw or {},
        "calculated_ratios": calc or {}, "loop_counter": 1,
        "validation_feedback": "", "is_complete": False,
        "task_query": query, "manager_directives": {},
        "intents": intents, "extra_tools": [], "rationale": "",
    }


def result(**payload):
    item = MagicMock()
    item.content = [MagicMock(text=json.dumps(payload))]
    return item


def metric(name="value", value=1.0):
    return {name: value, "interpretation": "available", "_inputs": {"x": value}, "_period": "ttm"}


class AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_new_ratios_flow_through_run_into_shared_state(monkeypatch):
    calc = {
        "pb": metric("pb_ratio", 4.2),
        "ev_ebitda": metric("ev_ebitda", 18.0),
        "peg": metric("peg_ratio", 1.4),
        "operating_margin": metric("operating_margin_pct", 22.0),
        "roa": metric("roa_pct", 9.0),
        "current_ratio_calc": metric("current_ratio", 2.1),
        "interest_coverage": metric("interest_coverage", 12.0),
        "revenue_cagr_from_growth": metric("cagr_pct", 15.0),
        "dcf_monte_carlo_growth": metric("p50_per_share", 170.0),
    }
    final_state = state(["valuation"], raw={"ticker": "AMD"}, calc=calc)
    final_state["is_complete"] = True

    agent = FinancialAnalystAgent(llm_client=MagicMock())
    agent._graph = MagicMock(ainvoke=AsyncMock(return_value=final_state))
    session = MagicMock(initialize=AsyncMock())
    monkeypatch.setattr(financial_module, "stdio_client", lambda params: AsyncContext((None, None)))
    monkeypatch.setattr(financial_module, "ClientSession", lambda read, write: AsyncContext(session))

    shared = {"task_query": "Analyze AMD", "manager_directives": {}}
    output = await agent.run(shared)
    summary = output["financial_metrics_summary"]

    for key, expected in calc.items():
        assert summary[key] == expected
        assert summary[key]["_period"] == "ttm"
        assert summary[key]["_inputs"]
    assert "pe_ratio" in summary  # legacy contract remains present


@pytest.mark.asyncio
async def test_missing_new_metrics_do_not_break_summary(monkeypatch):
    final_state = state(["overview"], raw={"ticker": "AMD"}, calc={})
    final_state["is_complete"] = True
    agent = FinancialAnalystAgent(llm_client=MagicMock())
    agent._graph = MagicMock(ainvoke=AsyncMock(return_value=final_state))
    monkeypatch.setattr(financial_module, "stdio_client", lambda params: AsyncContext((None, None)))
    monkeypatch.setattr(financial_module, "ClientSession", lambda read, write: AsyncContext(MagicMock(initialize=AsyncMock())))
    output = await agent.run({"task_query": "AMD", "manager_directives": {}})
    assert output["financial_metrics_summary"]["pb"] == {}


@pytest.mark.parametrize(
    ("intents", "calc"),
    [
        (["liquidity"], {"current_ratio_calc": metric(), "ocf_ratio": metric()}),
        (["valuation"], {"pe": metric(), "dcf": metric()}),
        (["financial_health"], {"roe": metric(), "de_ratio": metric(), "altman_z_score": metric()}),
        (["growth"], {"cagr": metric(), "forecast": metric()}),
    ],
)
def test_intent_requirements_accept_relevant_coverage(intents, calc):
    contract = _checker_requirements(state(intents, calc=calc))
    assert contract["passed"] is True


def test_multi_intent_is_union_and_full_report_is_comprehensive():
    multi = _checker_requirements(state(["liquidity", "growth"], calc={}))
    names = {name for group in multi["required_any_of_groups"] for name in group}
    assert {"current_ratio_calc", "ocf_ratio", "cagr", "forecast"} <= names
    assert multi["passed"] is False
    full = _checker_requirements(state(["full_report"], calc={}))
    assert len(full["required_any_of_groups"]) > len(multi["required_any_of_groups"])


@pytest.mark.asyncio
async def test_checker_does_not_require_valuation_for_liquidity(monkeypatch):
    llm = MagicMock()
    llm.messages.create.return_value = result(
        is_complete=True, score=90, passed=["INTENT COVERAGE"],
        failed=[], issues=[], feedback="",
    )
    monkeypatch.setattr(financial_module, "evaluate_financial_sanity", lambda payload: {
        "passed": True, "score": 100, "critical_failures": [],
    })
    agent = FinancialAnalystAgent(llm_client=llm)
    s = state(
        ["liquidity"], raw={"ticker": "AMD", "yahoo_ratios": {}},
        calc={"current_ratio_calc": metric(), "ocf_ratio": metric()},
    )
    verdict = await agent._check_data_quality(s)
    assert verdict["is_complete"] is True
    prompt = llm.messages.create.call_args.kwargs["messages"][0]["content"]
    assert '"liquidity"' in prompt
    assert "ACTIVE INTENT COMPLETENESS CONTRACT" in prompt


@pytest.mark.asyncio
async def test_checker_rejects_llm_pass_when_intent_requirements_missing(monkeypatch):
    llm = MagicMock()
    llm.messages.create.return_value = result(
        is_complete=True, score=95, passed=[], failed=[], issues=[], feedback="",
    )
    monkeypatch.setattr(financial_module, "evaluate_financial_sanity", lambda payload: {
        "passed": True, "score": 100, "critical_failures": [],
    })
    verdict = await FinancialAnalystAgent(llm_client=llm)._check_data_quality(
        state(["valuation"], raw={"ticker": "AMD"}, calc={"pe": metric()}),
    )
    assert verdict["is_complete"] is False
    assert "INTENT-SPECIFIC COVERAGE" in verdict["failed"]


@pytest.mark.parametrize(
    ("query", "intents", "expected"),
    [
        ("AMD quarterly results", ["profitability"], "10-Q"),
        ("AMD material event", ["overview"], "8-K"),
        ("Fundamental analysis of AMD", ["financial_health"], "10-K"),
        ("Investment analysis report AMD", ["full_report"], "10-K"),
        ("Is AMD liquid?", ["liquidity"], None),
    ],
)
def test_sec_filing_type_is_task_aware(query, intents, expected):
    assert _select_sec_filing_type(state(intents, query=query)) == expected


@pytest.mark.asyncio
async def test_sec_chain_uses_returned_cik_and_accession():
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=[
        result(cik="0000002488", company_name="AMD"),
        result(filings=[{
            "accession_number": "0000002488-26-000001",
            "filing_date": "2026-02-01",
            "primary_document": "amd-20251231.htm",
        }]),
        result(
            cik="0000002488", accession_number="0000002488-26-000001",
            text="filing", word_count=1000, validation_status="VALID",
        ),
    ])
    s = state(["full_report"], raw={})
    errors = []
    await FinancialAnalystAgent(llm_client=MagicMock())._execute_sec_filing_chain(
        session, s, "AMD", "10-K", errors,
    )
    calls = session.call_tool.await_args_list
    filing_arguments = calls[2].kwargs.get("arguments") or calls[2].args[1]
    assert filing_arguments == {
        "accession_number": "0000002488-26-000001", "cik": "0000002488",
        "primary_document": "amd-20251231.htm", "form_type": "10-K",
    }
    assert s["raw_numerical_data"]["sec_filing"]["status"] == "complete"
    assert errors == []


@pytest.mark.asyncio
async def test_sec_chain_retries_next_filing_when_first_document_is_invalid():
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=[
        result(cik="0000002488", company_name="AMD"),
        result(filings=[
            {"accession_number": "first", "primary_document": "consent.htm"},
            {"accession_number": "second", "primary_document": "annual-report.htm"},
        ]),
        result(text=None, word_count=0, validation_status="INVALID"),
        result(text="valid filing", word_count=1000, validation_status="VALID"),
    ])
    s = state(["full_report"], raw={})
    errors = []

    await FinancialAnalystAgent(llm_client=MagicMock())._execute_sec_filing_chain(
        session, s, "AMD", "10-K", errors,
    )

    filing = s["raw_numerical_data"]["sec_filing"]
    assert filing["status"] == "complete"
    assert filing["selected_filing"]["accession_number"] == "second"
    assert filing["attempted_filings"] == ["first", "second"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses, expected_calls",
    [([result(error="no cik")], 1),
     ([result(cik="0000002488"), result(filings=[])], 2)],
)
async def test_sec_chain_stops_when_prerequisite_missing(responses, expected_calls):
    session = MagicMock(call_tool=AsyncMock(side_effect=responses))
    s = state(["full_report"], raw={})
    errors = []
    await FinancialAnalystAgent(llm_client=MagicMock())._execute_sec_filing_chain(
        session, s, "AMD", "10-K", errors,
    )
    assert session.call_tool.await_count == expected_calls
    assert s["raw_numerical_data"]["sec_filing"]["status"] == "unavailable"
