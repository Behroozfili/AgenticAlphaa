"""Deterministic financial-model validation used before any LLM evaluator."""

from __future__ import annotations

from typing import Any

CRITICAL_MISSING = "CRITICAL_MISSING"
MATERIAL_MISSING = "MATERIAL_MISSING"
IMMATERIAL_MISSING = "IMMATERIAL_MISSING"
NOT_APPLICABLE = "NOT_APPLICABLE"
ESTIMATED_WITH_DISCLOSURE = "ESTIMATED_WITH_DISCLOSURE"
OBSERVED = "OBSERVED"


def valuation_input_finding(
    field: str, classification: str, materiality: str, impact: str,
    action: str, confidence: str, rationale: str, *,
    value_status: str, assumed_value: float | None = None,
    source_or_derivation: str | None = None,
) -> dict[str, Any]:
    """Build the canonical missingness/assumption finding contract."""
    return {
        "field": field, "classification": classification,
        "materiality": materiality, "impact": impact, "action": action,
        "confidence": confidence, "rationale": rationale,
        "value_status": value_status, "assumed_value": assumed_value,
        "source_or_derivation": source_or_derivation,
        "validation_status": "RESOLVED" if classification in {OBSERVED, NOT_APPLICABLE} else "DISCLOSED",
    }


def build_valuation_input_assessments(
    *, market_cap: float | None, total_debt: float | None, cash: float | None,
    marketable_securities: float | None, lease_liabilities: float | None,
    minority_interest: float | None, diluted_shares: float | None,
    diluted_shares_observed: bool, cost_of_debt_observed: bool,
    tax_rate_observed: bool, pre_tax_cost_of_debt_pct: float,
    effective_tax_rate_pct: float, sector: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Classify valuation inputs without conflating missing, zero, or N/A."""
    findings: dict[str, dict[str, Any]] = {}

    def observed(field: str, source: str) -> None:
        findings[field] = valuation_input_finding(
            field, OBSERVED, "resolved", "none", "continue", "high",
            "A numeric value was observed; an observed zero remains zero.",
            value_status="observed", source_or_derivation=source,
        )

    for field, value in (("cash", cash), ("total_debt", total_debt)):
        if value is None:
            findings[field] = valuation_input_finding(
                field, CRITICAL_MISSING, "critical", "equity bridge cannot be bounded",
                "block_valuation", "low", f"{field} is required for the EV-to-equity bridge.",
                value_status="missing",
            )
        else:
            observed(field, "Yahoo Finance current/TTM company data")

    if marketable_securities is not None:
        observed("marketable_securities", "reported separately by caller")
    elif cash is not None:
        findings["marketable_securities"] = valuation_input_finding(
            "marketable_securities", NOT_APPLICABLE, "none", "already included in cash bucket",
            "continue", "high", "Yahoo totalCash is treated as the combined cash and short-term-investment bucket.",
            value_status="not_applicable", source_or_derivation="definition of the supplied totalCash bucket",
        )
    else:
        findings["marketable_securities"] = valuation_input_finding(
            "marketable_securities", MATERIAL_MISSING, "material", "non-operating assets unresolved",
            "block_valuation", "low", "Neither cash nor a separate securities balance is available.",
            value_status="missing",
        )

    leverage = total_debt / market_cap if total_debt is not None and market_cap else None
    if lease_liabilities is not None:
        observed("lease_liabilities", "reported company data")
    elif leverage is not None and leverage <= 0.05:
        findings["lease_liabilities"] = valuation_input_finding(
            "lease_liabilities", IMMATERIAL_MISSING, "immaterial", "bounded by low observed leverage",
            "continue_with_warning", "medium", "Total debt is no more than 5% of market capitalization; omitted leases are unlikely to change the rounded valuation materially.",
            value_status="missing",
        )
    else:
        findings["lease_liabilities"] = valuation_input_finding(
            "lease_liabilities", MATERIAL_MISSING, "material", "debt-like claims may change equity value",
            "reduce_confidence_or_block", "low", "Lease liabilities are unavailable and low materiality cannot be demonstrated.",
            value_status="missing",
        )

    if minority_interest is not None:
        observed("minority_interest", "reported company data")
    elif (sector or "").lower() in {"financial services", "financials"}:
        findings["minority_interest"] = valuation_input_finding(
            "minority_interest", MATERIAL_MISSING, "material", "ownership claims may change equity value",
            "reduce_confidence_or_block", "low", "Minority interest is unavailable for a financial company.",
            value_status="missing",
        )
    else:
        findings["minority_interest"] = valuation_input_finding(
            "minority_interest", IMMATERIAL_MISSING, "immaterial", "no reported minority claim signal",
            "continue_with_warning", "medium", "No minority-interest value is reported and no sector-specific materiality signal was found.",
            value_status="missing",
        )

    if diluted_shares is None:
        findings["diluted_shares"] = valuation_input_finding(
            "diluted_shares", CRITICAL_MISSING, "critical", "per-share value cannot be calculated",
            "block_per_share_valuation", "low", "No observed or derived share count is available.", value_status="missing",
        )
    elif diluted_shares_observed:
        observed("diluted_shares", "Yahoo sharesOutstanding")
    else:
        findings["diluted_shares"] = valuation_input_finding(
            "diluted_shares", ESTIMATED_WITH_DISCLOSURE, "material", "per-share denominator is estimated",
            "continue_reduced_confidence", "medium", "Derived as market capitalization divided by current price.",
            value_status="estimated", assumed_value=diluted_shares,
            source_or_derivation="market_cap / current_price",
        )

    for field, observed_flag, value, fallback in (
        ("pre_tax_cost_of_debt_pct", cost_of_debt_observed, pre_tax_cost_of_debt_pct, "5% model fallback"),
        ("effective_tax_rate_pct", tax_rate_observed, effective_tax_rate_pct, "21% US statutory-rate fallback"),
    ):
        if observed_flag:
            observed(field, "company data / derived from company interest expense and debt")
        else:
            findings[field] = valuation_input_finding(
                field, ESTIMATED_WITH_DISCLOSURE, "material", "WACC is assumption-sensitive",
                "continue_reduced_confidence", "medium", fallback,
                value_status="estimated", assumed_value=value, source_or_derivation=fallback,
            )
    return findings


def _scenario_rows(forecast: dict[str, Any], name: str) -> list[dict[str, Any]]:
    scenarios = forecast.get("scenarios", forecast.get("forecast_scenarios", {}))
    scenario = scenarios.get(name, {}) if isinstance(scenarios, dict) else {}
    return scenario.get("yearly_forecast", []) if isinstance(scenario, dict) else []


def evaluate_financial_sanity(financial: dict[str, Any]) -> dict[str, Any]:
    """Return a fail-closed verdict for economically invalid valuation inputs."""
    forecast = financial.get("forecast", {})
    dcf = financial.get("dcf_valuation", financial.get("dcf", {}))
    reverse_dcf = financial.get("reverse_dcf", {})
    earnings_quality = financial.get("earnings_quality", {})
    critical: list[dict[str, str]] = []
    warnings: list[str] = []
    input_findings_map = financial.get("valuation_input_assessments", {})
    if not input_findings_map and isinstance(dcf, dict):
        candidate = dcf.get("base", dcf)
        if isinstance(candidate, dict):
            input_findings_map = candidate.get("input_assessments", {})
    input_findings = list(input_findings_map.values()) if isinstance(input_findings_map, dict) else []
    unresolved_material = False

    def fail(rule: str, message: str) -> None:
        critical.append({"rule": rule, "message": message, "severity": "CRITICAL"})

    for finding in input_findings:
        classification = finding.get("classification")
        action = finding.get("action", "")
        field = finding.get("field", "unknown")
        if classification == CRITICAL_MISSING:
            fail("CRITICAL_VALUATION_INPUT_MISSING", f"{field}: {finding.get('rationale', 'critical input missing')}")
            unresolved_material = True
        elif classification == MATERIAL_MISSING:
            unresolved_material = True
            if action in {"block_valuation", "block_per_share_valuation", "reduce_confidence_or_block"}:
                fail("MATERIAL_VALUATION_INPUT_MISSING", f"{field}: {finding.get('rationale', 'material input missing')}")
            else:
                warnings.append(f"Material missing input {field} reduces valuation confidence.")
        elif classification == IMMATERIAL_MISSING:
            warnings.append(f"Immaterial missing input {field}; valuation continues with disclosure.")
        elif classification == ESTIMATED_WITH_DISCLOSURE:
            unresolved_material = unresolved_material or finding.get("materiality") == "material"
            warnings.append(f"Estimated input {field} is disclosed and reduces confidence.")

    forecast_assumptions = forecast.get("assumptions", {})
    historical_cagr = forecast_assumptions.get("historical_revenue_cagr_pct")
    direct_copy = forecast_assumptions.get("historical_cagr_used_as_direct_fcf_growth") is True

    valuations = [dcf.get(name, {}) for name in ("bear", "base", "bull")]
    if not any(valuations) and isinstance(dcf, dict):
        valuations = [dcf]
    for valuation in valuations:
        assumptions = valuation.get("assumptions", {}) if isinstance(valuation, dict) else {}
        growth = assumptions.get("fcf_growth_rate_pct")
        years = assumptions.get("projection_years", 0)
        mode = assumptions.get("forecast_mode")
        if direct_copy or (
            mode == "legacy_constant_growth" and years and years > 1
            and historical_cagr is not None and growth == historical_cagr
        ):
            fail("HISTORICAL_CAGR_DIRECT_COPY", "Historical revenue CAGR was copied directly into constant multi-year FCF growth.")
        if mode == "legacy_constant_growth" and growth is not None and abs(growth) > 15:
            evidence = assumptions.get("growth_evidence")
            if not evidence:
                fail("EXTREME_CONSTANT_GROWTH", "Extreme constant FCF growth has no fade model or supporting evidence.")

        wacc = valuation.get("wacc_pct", valuation.get("discount_rate_pct")) if isinstance(valuation, dict) else None
        terminal = assumptions.get("terminal_growth_rate_pct")
        if wacc is not None and not 4.0 <= wacc <= 20.0:
            fail("IMPLAUSIBLE_WACC", f"WACC {wacc}% is outside the permitted 4%-20% range.")
        if wacc is not None and terminal is not None and terminal >= wacc:
            fail("TERMINAL_GROWTH_GUARD", "Terminal growth must be strictly below WACC.")
        per_share = valuation.get("equity_value_per_share", valuation.get("intrinsic_value_per_share")) if isinstance(valuation, dict) else None
        if per_share is not None and valuation.get("equity_bridge_complete") is not True:
            fail("INCOMPLETE_EQUITY_BRIDGE", "Per-share intrinsic value was reported without a complete EV-to-equity bridge.")

    rows = {name: _scenario_rows(forecast, name) for name in ("bear", "base", "bull")}
    if any(rows.values()):
        if not all(rows.values()) or len({len(value) for value in rows.values()}) != 1:
            fail("SCENARIO_ORDERING", "Bear/base/bull forecast paths must all exist with equal lengths.")
        else:
            for index in range(len(rows["base"])):
                values = [rows[name][index].get("fcf") for name in ("bear", "base", "bull")]
                if any(value is None for value in values) or not values[0] < values[1] < values[2]:
                    fail("SCENARIO_ORDERING", f"Year {index + 1} does not satisfy bear FCF < base FCF < bull FCF.")
                    break
            bear_growth = rows["bear"][0].get("revenue_growth_pct")
            base_growth = rows["base"][0].get("revenue_growth_pct")
            if bear_growth is None or base_growth is None or bear_growth >= base_growth:
                fail("BEAR_NOT_DETERIORATION", "Bear case is not a deterioration from the base case.")
            for name, path in rows.items():
                growths = [row.get("revenue_growth_pct") for row in path]
                if len(growths) > 1 and None not in growths and abs(growths[0]) > 15 and len(set(growths)) == 1:
                    fail("EXTREME_CONSTANT_GROWTH", f"{name} has exceptional constant growth without fade.")

    scenario_validation = dcf.get("validation", {}) if isinstance(dcf, dict) else {}
    if scenario_validation and scenario_validation.get("is_valid") is False:
        fail("SCENARIO_VALIDATION_FAILED", "DCF scenario validation did not pass.")

    one_off_pct = earnings_quality.get("material_one_off_income_pct")
    if one_off_pct is not None and abs(one_off_pct) >= 10 and earnings_quality.get("one_off_flagged") is not True:
        fail("UNFLAGGED_ONE_OFF_INCOME", "Material one-off/non-operating income was not explicitly flagged.")
    elif not earnings_quality:
        warnings.append("Earnings-quality normalization is not yet available; one-off income could not be verified.")

    if reverse_dcf and reverse_dcf.get("error") is None:
        wording = reverse_dcf.get("interpretation", "").lower()
        if "under this model's assumptions" not in wording:
            fail("REVERSE_DCF_WORDING", "Reverse DCF must state that its result is under the model's assumptions.")

    unique_rules = []
    seen: set[str] = set()
    for item in critical:
        if item["rule"] not in seen:
            unique_rules.append(item)
            seen.add(item["rule"])
    score = max(0, 100 - 20 * len(unique_rules) - 5 * len(warnings))
    if unique_rules:
        score = min(score, 49)
    confidence_cap = "LOW" if unique_rules else ("MEDIUM" if unresolved_material or any(
        item.get("classification") == ESTIMATED_WITH_DISCLOSURE for item in input_findings
    ) else "HIGH")
    return {
        "passed": not unique_rules,
        "score": score,
        "score_cap": 49 if unique_rules else 100,
        "confidence_cap": confidence_cap,
        "actionable_valuation_allowed": not unique_rules,
        "critical_failures": unique_rules,
        "warnings": warnings,
        "input_findings": input_findings,
        "unresolved_material_assumptions": unresolved_material,
        "validator": "deterministic_financial_sanity_v1",
    }
