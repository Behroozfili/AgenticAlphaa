from copy import deepcopy

from evaluation.financial_sanity import (
    evaluate_financial_sanity, valuation_input_finding,
    CRITICAL_MISSING, MATERIAL_MISSING, IMMATERIAL_MISSING,
    NOT_APPLICABLE, ESTIMATED_WITH_DISCLOSURE,
    build_valuation_input_assessments,
)


def valid_financial_summary():
    rows = {
        "bear": [{"revenue_growth_pct": 8, "fcf": 90}, {"revenue_growth_pct": 5, "fcf": 95}],
        "base": [{"revenue_growth_pct": 12, "fcf": 100}, {"revenue_growth_pct": 7, "fcf": 110}],
        "bull": [{"revenue_growth_pct": 16, "fcf": 115}, {"revenue_growth_pct": 9, "fcf": 125}],
    }
    forecast = {
        "assumptions": {
            "historical_revenue_cagr_pct": 25,
            "historical_cagr_used_as_direct_fcf_growth": False,
        },
        "scenarios": {name: {"yearly_forecast": path} for name, path in rows.items()},
    }
    valuation = {
        "wacc_pct": 9.0, "equity_bridge_complete": True,
        "equity_value_per_share": 50,
        "assumptions": {"terminal_growth_rate_pct": 2.5, "forecast_mode": "explicit_fcf_by_year"},
    }
    return {
        "forecast": forecast,
        "dcf_valuation": {
            "bear": deepcopy(valuation), "base": deepcopy(valuation), "bull": deepcopy(valuation),
            "validation": {"is_valid": True},
        },
    }


def rules(result):
    return {item["rule"] for item in result["critical_failures"]}


def test_valid_forecast_and_valuation_pass():
    result = evaluate_financial_sanity(valid_financial_summary())
    assert result["passed"] is True
    assert result["actionable_valuation_allowed"] is True


def test_rejects_historical_cagr_copied_to_constant_fcf_growth():
    data = valid_financial_summary()
    data["forecast"]["assumptions"]["historical_cagr_used_as_direct_fcf_growth"] = True
    result = evaluate_financial_sanity(data)
    assert "HISTORICAL_CAGR_DIRECT_COPY" in rules(result)
    assert result["score"] <= 49


def test_rejects_extreme_constant_growth_without_fade_or_evidence():
    data = valid_financial_summary()
    for row in data["forecast"]["scenarios"]["bull"]["yearly_forecast"]:
        row["revenue_growth_pct"] = 30
    result = evaluate_financial_sanity(data)
    assert "EXTREME_CONSTANT_GROWTH" in rules(result)


def test_rejects_per_share_value_with_incomplete_equity_bridge():
    data = valid_financial_summary()
    data["dcf_valuation"]["base"]["equity_bridge_complete"] = False
    result = evaluate_financial_sanity(data)
    assert "INCOMPLETE_EQUITY_BRIDGE" in rules(result)


def test_rejects_terminal_growth_at_or_above_wacc_and_implausible_wacc():
    data = valid_financial_summary()
    data["dcf_valuation"]["base"]["wacc_pct"] = 3.0
    data["dcf_valuation"]["base"]["assumptions"]["terminal_growth_rate_pct"] = 3.0
    result = evaluate_financial_sanity(data)
    assert {"IMPLAUSIBLE_WACC", "TERMINAL_GROWTH_GUARD"} <= rules(result)


def test_rejects_unordered_scenarios_and_non_deteriorating_bear():
    data = valid_financial_summary()
    data["forecast"]["scenarios"]["bear"]["yearly_forecast"][0] = {
        "revenue_growth_pct": 13, "fcf": 101,
    }
    result = evaluate_financial_sanity(data)
    assert {"SCENARIO_ORDERING", "BEAR_NOT_DETERIORATION"} <= rules(result)


def test_rejects_material_unflagged_one_off_income():
    data = valid_financial_summary()
    data["earnings_quality"] = {"material_one_off_income_pct": 15, "one_off_flagged": False}
    result = evaluate_financial_sanity(data)
    assert "UNFLAGGED_ONE_OFF_INCOME" in rules(result)


def finding(field, classification, materiality, action, value_status="missing", assumed_value=None):
    return valuation_input_finding(
        field, classification, materiality, "test impact", action,
        "medium", "test rationale", value_status=value_status,
        assumed_value=assumed_value, source_or_derivation="test derivation",
    )


def test_critical_missing_bridge_input_blocks_valuation():
    data = valid_financial_summary()
    data["valuation_input_assessments"] = {
        "cash": finding("cash", CRITICAL_MISSING, "critical", "block_valuation")
    }
    result = evaluate_financial_sanity(data)
    assert result["passed"] is False
    assert result["actionable_valuation_allowed"] is False


def test_material_missing_blocks_when_impact_is_unbounded():
    data = valid_financial_summary()
    data["valuation_input_assessments"] = {
        "leases": finding("leases", MATERIAL_MISSING, "material", "reduce_confidence_or_block")
    }
    result = evaluate_financial_sanity(data)
    assert "MATERIAL_VALUATION_INPUT_MISSING" in rules(result)


def test_immaterial_and_not_applicable_missing_do_not_block():
    data = valid_financial_summary()
    data["valuation_input_assessments"] = {
        "leases": finding("leases", IMMATERIAL_MISSING, "immaterial", "continue_with_warning"),
        "minority": finding("minority", NOT_APPLICABLE, "none", "continue", "not_applicable"),
    }
    result = evaluate_financial_sanity(data)
    assert result["passed"] is True
    assert result["actionable_valuation_allowed"] is True
    assert result["confidence_cap"] == "HIGH"


def test_estimated_fallback_is_typed_disclosed_and_prevents_high_confidence():
    data = valid_financial_summary()
    estimate = finding(
        "effective_tax_rate_pct", ESTIMATED_WITH_DISCLOSURE, "material",
        "continue_reduced_confidence", "estimated", 21.0,
    )
    data["valuation_input_assessments"] = {"effective_tax_rate_pct": estimate}
    result = evaluate_financial_sanity(data)
    assert result["passed"] is True
    assert result["confidence_cap"] == "MEDIUM"
    assert result["unresolved_material_assumptions"] is True
    assert result["input_findings"][0]["source_or_derivation"] == "test derivation"


def test_fallback_cost_of_debt_and_tax_are_typed_assumptions_not_facts():
    findings = build_valuation_input_assessments(
        market_cap=1000, total_debt=20, cash=100,
        marketable_securities=None, lease_liabilities=None,
        minority_interest=None, diluted_shares=10,
        diluted_shares_observed=False, cost_of_debt_observed=False,
        tax_rate_observed=False, pre_tax_cost_of_debt_pct=5,
        effective_tax_rate_pct=21, sector="Technology",
    )
    for field in ("pre_tax_cost_of_debt_pct", "effective_tax_rate_pct", "diluted_shares"):
        assert findings[field]["classification"] == ESTIMATED_WITH_DISCLOSURE
        assert findings[field]["value_status"] == "estimated"
        assert findings[field]["source_or_derivation"]
        assert findings[field]["assumed_value"] is not None
