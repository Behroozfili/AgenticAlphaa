import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import (
    ManagerAgent,
    agent_result_is_valid,
    is_transient_failure,
    normalize_route_action,
)
from evaluation.canonical_synthesis import build_canonical_synthesis_payload
from evaluation.final_publication import (
    normalize_validation_aware_narrative,
    validate_final_publication,
)
from evaluation.publication_policy import (
    assemble_publication,
    execution_status_from_publication,
)
from evaluation.valid_availability import compare_valid_availability
from evaluation.validate_period_consistency import check_narration_vs_period


FIXTURE = Path(__file__).parents[1] / "fixtures" / "stage3_ticker_regressions.json"


def cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def shared(case):
    return {
        "task_query": f"Analyze {case['ticker']}",
        "financial_metrics_summary": {
            "ticker": case["ticker"],
            **case["financial_metrics_summary"],
            "financial_facts": [],
        },
        "sentiment_analysis_summary": {"status": "VALID", "causal_aspects_supported": []},
        "recommendation_gate": {"actionable_recommendation_allowed": True},
    }


@pytest.mark.parametrize("case", cases(), ids=lambda row: row["ticker"])
def test_company_and_unseen_ticker_narratives_validate(case):
    result = validate_final_publication(case["report"], shared(case))
    assert result["passed"], result["critical_findings"]


def test_fixture_contains_required_seen_and_unseen_tickers():
    fixture_cases = cases()
    assert {x["ticker"] for x in fixture_cases if x["seen"]} == {"AMD", "MSFT", "NVDA"}
    assert any(not x["seen"] and x["ticker"] not in {"AMD", "MSFT", "NVDA"} for x in fixture_cases)


def test_complete_equity_bridge_removes_stale_no_net_debt_caveat():
    fm = cases()[1]["financial_metrics_summary"]
    stale = "The DCF has no net-debt adjustment."
    normalized = normalize_validation_aware_narrative(stale, fm)
    assert "no net-debt adjustment" not in normalized.lower()
    assert "includes the modeled EV-to-equity bridge" in normalized
    assert validate_final_publication(normalized, {"financial_metrics_summary": fm, "sentiment_analysis_summary": {}, "recommendation_gate": {}})["passed"]


def test_reverse_dcf_is_model_output_in_valuation_not_sentiment():
    state = {
        "financial_metrics_summary": {
            "reverse_dcf": {
                "implied_growth_rate_pct": 18.0,
                "formula": "Reverse DCF",
                "interpretation": "model-dependent implied growth",
            }
        },
        "sentiment_analysis_summary": {"status": "VALID"},
        "recommendation_gate": {},
    }
    reverse = build_canonical_synthesis_payload(state)["valuation"]["reverse_dcf"]
    assert reverse["claim_type"] == "MODEL_OUTPUT"
    assert reverse["domain"] == "valuation"
    assert reverse["classification"] == "valuation_model_inference"

    bad_report = "## Market Sentiment\nReverse DCF is a bullish sentiment signal."
    result = validate_final_publication(bad_report, state)
    assert not result["passed"]
    assert any(x["check"] == "reverse_dcf_semantics" for x in result["critical_findings"])


def test_period_validator_is_sentence_local_and_rounding_aware():
    fm = cases()[2]["financial_metrics_summary"]
    valid = "FY2026 annual net margin was about 55.6%. Q1 revenue was discussed separately."
    assert check_narration_vs_period(valid, fm) == []

    invalid = "Q1 net margin was approximately 55.6%."
    findings = check_narration_vs_period(invalid, fm)
    assert any(x["severity"] == "HIGH" for x in findings)


def test_period_validator_does_not_equate_distinct_percentage_points():
    fm = {
        "gross_margin": {
            "gross_margin_pct": 67.94,
            "_period": "annual:2025-07-01/2026-06-30",
        }
    }
    report = "Nine-month gross margin declined to 68.1%."
    assert check_narration_vs_period(report, fm) == []


def test_period_marker_before_comma_does_not_leak_to_next_metric_clause():
    fm = {
        "net_margin": {
            "net_margin_pct": 39.74, "_period": "quarterly:2026-Q4",
        },
        "roe": {"roe_pct": 30.23, "_period": "annual:2026"},
    }
    report = (
        "Profitability metrics include 39.74% net margin in Q4 FY2026, "
        "30.23% ROE."
    )
    assert check_narration_vs_period(report, fm) == []


def test_dcf_price_implication_is_not_low_confidence_sentiment():
    state = {
        "financial_metrics_summary": {},
        "sentiment_analysis_summary": {
            "status": "VALID", "confidence": 0.1,
            "causal_aspects_supported": [],
        },
        "recommendation_gate": {},
    }
    report = (
        "## Financial Health\nThe current DCF valuation implies that the market "
        "price embeds growth beyond the modeled range."
    )
    validation = validate_final_publication(report, state)
    assert validation["checks"]["sentiment_evidence_fidelity"]["passed"]
    assert validation["checks"]["low_confidence_sentiment_scope"]["passed"]


def test_valid_availability_comparison_requires_period_and_source():
    after = cases()[0]["financial_metrics_summary"]
    before = {
        key: {
            value_key: next(
                value for field, value in metric.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
        }
        for key, metric in after.items()
        for value_key in [next(
            field for field, value in metric.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        )]
    }
    comparison = compare_valid_availability(before, after)
    assert comparison["after"]["available_metric_count"] >= comparison["before"]["available_metric_count"]
    assert comparison["valid_with_period_and_source_delta"] > 0
    assert comparison["passes_non_regression"] is True


def test_retry_policy_is_transient_only_and_reuses_valid_results():
    assert is_transient_failure("HTTP 503 service unavailable") is True
    assert is_transient_failure("period metadata is missing") is False

    state = {
        "manager_directives": {"include_sentiment": True},
        "financial_metrics_summary": {
            "validation_passed": True,
            "financial_sanity": {"passed": True},
        },
        "agent_execution_history": [{
            "agent_name": "FinancialAnalystAgent", "outcome": "success", "error_message": None,
        }],
    }
    assert agent_result_is_valid("financial", state) is True
    assert normalize_route_action("rerun_financial", state) == "run_sentiment"


@pytest.mark.asyncio
async def test_dispatch_does_not_run_an_agent_with_a_valid_reusable_result():
    financial = AsyncMock()
    manager = ManagerAgent(MagicMock(), financial, MagicMock(), MagicMock(), llm_client=AsyncMock())
    shared_state = {
        "manager_directives": {},
        "orchestrator_logs": [],
        "financial_metrics_summary": {
            "validation_passed": True,
            "financial_sanity": {"passed": True},
        },
    }
    result = await manager._node_dispatch({
        "last_action": "rerun_financial", "shared_state": shared_state,
    })
    financial.run.assert_not_awaited()
    assert "[REUSE]" in result["shared_state"]["orchestrator_logs"][-1]


def test_failed_publication_cannot_be_top_level_success():
    validation = {
        "passed": False,
        "critical_findings": [{
            "check": "valuation_bridge_consistency",
            "message": "bridge incomplete",
            "severity": "CRITICAL",
        }],
        "warnings": [],
    }
    state = {
        "financial_metrics_summary": {"current_price": 100.0},
        "sentiment_analysis_summary": {"status": "VALID"},
        "recommendation_gate": {"actionable_recommendation_allowed": True},
    }
    publication = assemble_publication("## DCF Valuation\nBad value.", validation, state)
    assert publication["publication_status"] == "LIMITED_REPORT"
    assert publication["publication_succeeded"] is False
    assert execution_status_from_publication(
        publication["publication_status"], validation_passed=False
    ) == "failed"


def test_manager_report_contract_rejects_missing_or_truncated_sections():
    validation = validate_final_publication(
        "## Executive Summary\nPartial.\n\n"
        "[REPORT TRUNCATED BEFORE VALIDATED COMPLETION]",
        {"enforce_report_sections": True},
    )
    findings = validation["checks"]["required_report_structure"]["findings"]
    assert validation["passed"] is False
    assert any("missing required sections" in item for item in findings)
    assert any("token limit" in item for item in findings)
