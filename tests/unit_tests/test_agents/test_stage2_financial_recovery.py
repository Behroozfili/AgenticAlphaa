from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
from agents.executor_helpers.finance_executor_helper import build_data_quality_score
from agents.executor_helpers.financial_helper import (
    derive_compute_facts,
    map_build_forecast,
    map_current_ratio,
    map_debt_to_equity,
    map_fcf_margin,
    map_ocf_ratio,
    post_de_ratio,
    surface_calculated_ratios,
)
from evaluation.canonical_synthesis import build_canonical_synthesis_payload
from evaluation.final_publication import validate_final_publication
from tools.financial_tools.sec_edgar import _extract_annual
from financial_test import stage5_context
from core.forecast_contract import ForecastContext


def extract_annual(*args, **kwargs):
    kwargs.setdefault("context", ForecastContext.from_dict(stage5_context("NVDA")))
    return _extract_annual(*args, **kwargs)


def fact(value, start, end, *, concept="PrimaryConcept", status="VALID_PRIMARY"):
    return {
        "value": value,
        "period_start": start,
        "period_end": end,
        "fiscal_year": int(end[:4]),
        "unit": "USD",
        "metric_scope": "duration",
        "validation_status": status,
        "xbrl_concept": concept,
        "selected_source": f"sec_xbrl:{concept}",
        "exact_period": {
            "basis": "duration",
            "period_start": start,
            "period_end": end,
        },
    }


def raw(*, yahoo=None, xbrl=None, balance=None):
    return {
        "yahoo_ratios": yahoo or {},
        "xbrl_financials": xbrl or {},
        "balance_sheet_snapshot": balance or {},
        "revenue_growth": {
            "annual_revenue": [
                {"year": 2025, "revenue": 100.0},
                {"year": 2024, "revenue": 90.0},
            ],
        },
    }


def test_same_concept_restatement_prefers_latest_filed_fact():
    payload = {
        "facts": {"us-gaap": {"Revenue": {"units": {"USD": [
            {
                "form": "10-K", "fp": "FY", "fy": 2025,
                "start": "2024-01-01", "end": "2024-12-31",
                "val": 100, "filed": "2025-02-01", "accn": "old",
            },
            {
                "form": "10-K/A", "fp": "FY", "fy": 2025,
                "start": "2024-01-01", "end": "2024-12-31",
                "val": 105, "filed": "2025-03-01", "accn": "new",
            },
        ]}}}},
    }

    selected = extract_annual(payload, ["Revenue"], metric_scope="duration")

    assert selected[0]["value"] == 105
    assert selected[0]["accession_number"] == "new"
    assert selected[0]["validation_status"] == "VALID_PRIMARY"


def test_alternative_xbrl_concept_is_explicit_fallback():
    payload = {
        "facts": {"us-gaap": {"AlternativeRevenue": {"units": {"USD": [{
            "form": "10-K", "fp": "FY", "fy": 2025,
            "start": "2024-01-01", "end": "2024-12-31",
            "val": 105, "filed": "2025-02-01", "accn": "a1",
        }]}}}},
    }

    selected = extract_annual(
        payload, ["PreferredRevenue", "AlternativeRevenue"], metric_scope="duration"
    )

    assert selected[0]["validation_status"] == "FALLBACK"
    assert selected[0]["confidence"] == "medium"
    assert "PreferredRevenue" in selected[0]["fallback_reason"]


def test_direct_yahoo_fcf_recovers_output_without_annual_ttm_mixing():
    inputs = derive_compute_facts(raw(yahoo={
        "free_cash_flow": 500.0,
        "revenue_ttm": 2_000.0,
        "financial_data_period_end": "2026-07-31",
        "beta": 1.1,
    }), {"cagr": {"cagr_pct": 8.0}})

    assert inputs["fcf_base"] == 500.0
    assert inputs["fcf_period"] == "ttm:2026-07-31"
    assert inputs["valuation_base_period"] == "ttm:2026-07-31"
    assert inputs["fcf_metadata"]["validation_status"] == "FALLBACK"
    assert inputs["fcf_metadata"]["confidence"] == "medium"
    assert "Yahoo" in inputs["fcf_metadata"]["fallback_reason"]
    forecast = map_build_forecast(inputs)
    assert forecast is not None
    assert forecast[0]["revenue_base"] == 2_000.0
    assert forecast[1] == "projection_from:ttm:2026-07-31"


def test_sec_fcf_never_borrows_yahoo_ttm_revenue_for_margin():
    start, end = "2025-01-01", "2025-12-31"
    inputs = derive_compute_facts(raw(
        yahoo={
            "free_cash_flow": 999.0,
            "revenue_ttm": 5_000.0,
            "financial_data_period_end": "2026-07-31",
            "beta": 1.1,
        },
        xbrl={
            "operating_cash_flow_annual": [fact(700.0, start, end, concept="OCF")],
            "capex_annual": [fact(200.0, start, end, concept="CapEx")],
        },
    ), {"cagr": {"cagr_pct": 8.0}})

    assert inputs["fcf_base"] == 500.0
    assert inputs["fcf_period"].startswith("annual:")
    assert inputs["revenue_base"] is None
    assert map_fcf_margin(inputs) is None
    assert map_build_forecast(inputs) is None


def test_current_ratio_and_ocf_coverage_have_distinct_semantics():
    end = "2025-12-31"
    inputs = derive_compute_facts(raw(
        xbrl={"operating_cash_flow_annual": [
            fact(300.0, "2025-01-01", end, concept="OCF")
        ]},
        balance={
            "period_end": end,
            "current_assets": 400.0,
            "current_liabilities": 200.0,
        },
    ), {})

    current_plan = map_current_ratio(inputs)
    ocf_plan = map_ocf_ratio(inputs)

    assert current_plan[2]["semantic_name"] == "current_ratio"
    assert current_plan[2]["formula"] == "Current Assets / Current Liabilities"
    assert ocf_plan[2]["semantic_name"] == "operating_cash_flow_coverage"
    assert ocf_plan[2]["formula"] == "Operating Cash Flow / Current Liabilities"


def test_ocf_coverage_rejects_nonmatching_period_end():
    inputs = derive_compute_facts(raw(
        yahoo={
            "operating_cash_flow": 320.0,
            "financial_data_period_end": "2026-03-31",
        },
        xbrl={"operating_cash_flow_annual": [
            fact(300.0, "2025-01-01", "2025-12-31", concept="OCF")
        ]},
        balance={
            "period_end": "2026-06-30",
            "current_liabilities": 200.0,
        },
    ), {})

    assert map_ocf_ratio(inputs) is None
    assert inputs["ocf_coverage_metadata"]["validation_status"] == "UNAVAILABLE"


def test_liabilities_to_equity_is_degraded_approximation_not_standard_de():
    end = "2025-12-31"
    instant = lambda value, concept: {
        "value": value,
        "period_start": None,
        "period_end": end,
        "metric_scope": "instant",
        "unit": "USD",
        "validation_status": "VALID_PRIMARY",
        "xbrl_concept": concept,
    }
    inputs = derive_compute_facts(raw(
        xbrl={
            "total_assets": [instant(1_000.0, "Assets")],
            "total_liabilities": [instant(600.0, "Liabilities")],
        },
    ), {})

    plan = map_debt_to_equity(inputs)
    assert plan[2]["validation_status"] == "DEGRADED"
    assert plan[2]["is_approximation"] is True
    assert plan[2]["semantic_name"] == "total_liabilities_to_equity_approximation"
    result = {"de_ratio": 1.5, **plan[2]}
    post_de_ratio(result, inputs)
    assert "not standard D/E" in result["display_name"]
    assert result["formula"] == "Total Liabilities / Shareholders' Equity"


def test_one_unavailable_metric_does_not_remove_other_results():
    calc = {
        "fcf_margin": {
            "fcf_margin_pct": None,
            "validation_status": "UNAVAILABLE",
            "confidence": "none",
        },
        "current_ratio_calc": {
            "current_ratio": 2.0,
            "validation_status": "VALID_PRIMARY",
            "confidence": "high",
        },
    }
    summary = {}
    surface_calculated_ratios(summary, calc)

    assert summary["fcf_margin"]["validation_status"] == "UNAVAILABLE"
    assert summary["current_ratio_calc"]["current_ratio"] == 2.0


def test_data_quality_and_canonical_contract_accept_additive_metadata():
    metric = {
        "current_ratio": 2.0,
        "_period": "snapshot:2025-12-31",
        "validation_status": "FALLBACK",
        "source": "yahoo_finance.currentRatio",
        "exact_period": {"basis": "snapshot", "period_end": "2025-12-31"},
        "formula": "Current Assets / Current Liabilities",
        "fallback_reason": "components unavailable",
        "confidence": "medium",
        "is_approximation": False,
        "semantic_name": "current_ratio",
    }
    quality = build_data_quality_score(
        {"yahoo_ratios": {"current_ratio": 2.0}}, {"current_ratio_calc": metric}
    )
    assert quality["fallback_metric_count"] == 1

    payload = build_canonical_synthesis_payload({
        "financial_metrics_summary": {"current_ratio": metric},
        "sentiment_analysis_summary": {},
        "recommendation_gate": {},
    })
    published = payload["financial"]["performance_metrics"]["current_ratio"]
    assert published["value"] == 2.0
    assert published["validation_status"] == "FALLBACK"
    assert published["fallback_reason"] == "components unavailable"
    assert published["confidence"] == "medium"


@pytest.mark.asyncio
async def test_manager_contract_preserves_legacy_scalars_and_recovery_metadata():
    llm = AsyncMock()
    response = MagicMock(stop_reason="end_turn")
    response.content = [MagicMock(text="Report")]
    llm.messages.create.return_value = response
    manager = ManagerAgent(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), llm_client=llm
    )
    metric = {
        "current_ratio": 2.0,
        "_period": "snapshot:2025-12-31",
        "validation_status": "FALLBACK",
        "source": "yahoo_finance.currentRatio",
        "exact_period": {"basis": "snapshot", "period_end": "2025-12-31"},
        "formula": "Current Assets / Current Liabilities",
        "fallback_reason": "components unavailable",
        "confidence": "medium",
        "is_approximation": False,
    }
    state = {
        "task_query": "Analyze MSFT",
        "financial_metrics_summary": {"ticker": "MSFT", "current_ratio": metric},
        "sentiment_analysis_summary": {},
        "recommendation_gate": {},
        "aggregated_research_context": [],
        "agent_execution_history": [],
    }

    assert await manager._brain_finalise(state) == "Report"
    canonical = state["canonical_synthesis_payload"]
    published = canonical["financial"]["performance_metrics"]["current_ratio"]
    assert published["value"] == 2.0
    assert published["validation_status"] == "FALLBACK"


def test_publication_contract_ignores_additive_metric_metadata():
    shared = {
        "financial_metrics_summary": {
            "current_price": 100.0,
            "current_ratio": {
                "current_ratio": 2.0,
                "validation_status": "FALLBACK",
                "confidence": "medium",
                "formula": "Current Assets / Current Liabilities",
            },
            "de_ratio": {
                "de_ratio": 1.5,
                "validation_status": "DEGRADED",
                "is_approximation": True,
                "semantic_name": "total_liabilities_to_equity_approximation",
            },
            "financial_facts": [],
        },
        "sentiment_analysis_summary": {"status": "VALID"},
        "recommendation_gate": {"actionable_recommendation_allowed": True},
    }

    result = validate_final_publication("Current ratio is 2.0.", shared)

    assert result["passed"] is True
    assert result["publication_status"] == "PUBLISHED"
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
