from agents.executor_helpers.financial_helper import (
    checker_requirements,
    derive_compute_facts,
    map_build_forecast,
    map_gross_margin,
    map_operating_margin,
)


def annual_fact(value, start, end, *, fiscal_year=None, unit="USD"):
    return {
        "value": value,
        "period_start": start,
        "period_end": end,
        "fiscal_year": fiscal_year,
        "unit": unit,
        "metric_scope": "duration",
        "validation_status": "VALID",
    }


def raw_with_xbrl(**series):
    return {
        "yahoo_ratios": {"beta": 1.2},
        "revenue_growth": {
            "annual_revenue": [
                {"year": 2025, "revenue": 100},
                {"year": 2024, "revenue": 90},
            ],
        },
        "xbrl_financials": series,
    }


def test_msft_regression_revenue_2026_is_not_combined_with_cogs_2017():
    raw = raw_with_xbrl(
        revenue_annual=[annual_fact(331_839, "2025-07-01", "2026-06-30", fiscal_year=2026)],
        cost_of_revenue_annual=[annual_fact(34_261, "2016-07-01", "2017-06-30", fiscal_year=2017)],
    )

    inputs = derive_compute_facts(raw, {})

    assert map_gross_margin(inputs) is None
    assert inputs["gross_margin_period"] is None


def test_nvda_regression_operating_margin_requires_exact_shared_period():
    raw = raw_with_xbrl(
        revenue_annual=[annual_fact(26_914, "2021-02-01", "2022-01-30", fiscal_year=2022)],
        operating_income_annual=[annual_fact(130_387, "2025-01-27", "2026-01-25", fiscal_year=2026)],
    )

    inputs = derive_compute_facts(raw, {})

    assert map_operating_margin(inputs) is None
    assert inputs["operating_margin_period"] is None


def test_nvda_regression_stale_capex_cannot_seed_fcf_or_dcf():
    raw = raw_with_xbrl(
        revenue_annual=[annual_fact(208_000, "2025-01-27", "2026-01-25", fiscal_year=2026)],
        operating_cash_flow_annual=[annual_fact(102_718, "2025-01-27", "2026-01-25", fiscal_year=2026)],
        capex_annual=[annual_fact(138.735, "2011-01-31", "2012-01-29", fiscal_year=2012)],
    )

    inputs = derive_compute_facts(raw, {"cagr": {"cagr_pct": 20.0}})

    assert inputs["fcf_base"] is None
    assert inputs["dcf_fcf_base"] is None
    assert map_build_forecast(inputs) is None


def test_compatible_annual_bundle_remains_available_for_valuation():
    start, end = "2025-01-27", "2026-01-25"
    raw = raw_with_xbrl(
        revenue_annual=[annual_fact(208_000, start, end, fiscal_year=2026)],
        operating_cash_flow_annual=[annual_fact(102_718, start, end, fiscal_year=2026)],
        capex_annual=[annual_fact(3_500, start, end, fiscal_year=2026)],
    )

    inputs = derive_compute_facts(raw, {"cagr": {"cagr_pct": 20.0}})
    forecast = map_build_forecast(inputs)

    assert inputs["dcf_fcf_base"] == 99_218
    assert forecast is not None
    assert forecast[0]["source_period"] == f"annual:{start}/{end}"


def test_checker_distinguishes_available_from_period_valid():
    state = {
        "intents": ["cash_flow"],
        "raw_numerical_data": {},
        "calculated_ratios": {
            "fcf_margin": {
                "fcf_margin_pct": 45.0,
                "_period": "MIXED(fcf=annual:2026,revenue=annual:2017)",
                "validation_status": "INCOMPATIBLE_PERIOD",
            },
            "ocf_ratio": {
                "ocf_ratio": 1.1,
                "_period": "annual:2026",
                "validation_status": "VALID",
            },
        },
    }

    contract = checker_requirements(state)

    assert contract["availability"]["fcf_margin"] is True
    assert contract["validity"]["fcf_margin"] is False
    assert contract["passed"] is False
