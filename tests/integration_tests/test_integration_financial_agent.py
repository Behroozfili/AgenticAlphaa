"""
test_financial_agent.py — full integration test suite for
FinancialAnalystAgent, in one file.

Organized in two tiers, run selectively via pytest markers:

    pytest test_financial_agent.py -v -m "not slow"   # Tier 1 only — fast, no LLM
    pytest test_financial_agent.py -v                    # everything, including Tier 2

Tier 1 (no marker): call tool functions directly — get_peer_comparison,
    get_xbrl_financials, discounted_cash_flow / dcf_scenario_range /
    dcf_monte_carlo, debt_to_equity. These are all plain synchronous
    functions (unlike the research tools), real API calls but no LLM —
    fast and cheap.
Tier 2 (@pytest.mark.slow): run FinancialAnalystAgent.run() end-to-end —
    real LLM calls, slower and more expensive.

Every test here is tied to a specific bug found and fixed during this
project's development — see each test's docstring for which one.
"""
import pytest

from tools.financial_tools.yahoo_finance import get_peer_comparison
from tools.financial_tools.sec_edgar import get_xbrl_financials
from tools.financial_tools.financial_ratio_calculator import (
    discounted_cash_flow,
    dcf_scenario_range,
    dcf_monte_carlo,
    debt_to_equity,
)
from agents.financial_agent import FinancialAnalystAgent


# =============================================================================
# TIER 1 — get_peer_comparison
# =============================================================================

class TestPeerComparison:
    """Regression: get_peer_comparison's auto-inference used to silently
    return peers=[] always (fetched yf recommendations, then discarded
    them) — this test forces explicit peers and checks the growth-
    adjusted comparison field exists, since that's the field the fix
    added on top of the pre-existing raw comparison."""

    def test_returns_growth_adjusted_comparison(self):
        result = get_peer_comparison("MSFT", peers=["GOOGL", "META"])
        assert result.get("error") is None
        assert "growth_adjusted_comparison" in result
        gac = result["growth_adjusted_comparison"]
        assert "primary_peg_ratio" in gac
        assert gac["interpretation"] in (
            "cheap_relative_to_growth", "expensive_relative_to_growth",
            "in_line_with_peers", "insufficient_data",
        )

    def test_no_peers_does_not_crash(self):
        """peers=None should degrade gracefully (empty peers, not an exception)."""
        result = get_peer_comparison("MSFT", peers=None)
        assert result.get("error") is None
        assert result["peers"] == []


# =============================================================================
# TIER 1 — get_xbrl_financials (cash-flow tags)
# =============================================================================

class TestXbrlCashFlowTags:
    """Regression: operating_cash_flow_annual / capex_annual / etc. were
    added later — this test would have caught it if the tags were
    silently dropped or renamed."""

    def test_returns_all_cash_flow_series(self):
        result = get_xbrl_financials("MSFT")
        assert result.get("error") is None
        for key in ("operating_cash_flow_annual", "capex_annual",
                    "dividends_paid_annual", "buybacks_annual"):
            assert key in result, f"missing key: {key}"
            assert isinstance(result[key], list)


# =============================================================================
# TIER 1 — DCF valuation (scenario range + Monte Carlo)
# =============================================================================

class TestDcfValuation:
    """Regression: a single-point DCF for a hypergrowth stock reads as
    'broken' without a range — enforce that the scenario/Monte Carlo
    versions actually produce a spread, not three identical numbers."""

    def test_scenario_range_produces_a_spread(self):
        result = dcf_scenario_range(
            fcf_base=60_000_000_000, beta=1.13, base_growth_rate_pct=12.42,
            shares_outstanding=7_400_000_000,
        )
        bear_ev = result["bear"]["enterprise_value"]
        base_ev = result["base"]["enterprise_value"]
        bull_ev = result["bull"]["enterprise_value"]
        assert bear_ev < base_ev < bull_ev, "scenarios should be monotonically ordered"

    def test_monte_carlo_percentiles_are_ordered(self):
        result = dcf_monte_carlo(
            fcf_base=60_000_000_000, beta=1.13, base_growth_rate_pct=12.42,
            shares_outstanding=7_400_000_000, n_simulations=200, seed=1,
        )
        assert result["error"] is None
        assert result["enterprise_value_p10"] <= result["enterprise_value_p50"] <= result["enterprise_value_p90"]

    def test_negative_fcf_returns_error_not_exception(self):
        result = discounted_cash_flow(fcf_base=-1000, beta=1.0)
        assert result["error"] is not None
        assert result["enterprise_value"] is None


# =============================================================================
# TIER 1 — debt_to_equity calculator
# =============================================================================

class TestDebtToEquityLabeling:
    """Regression: the XBRL fallback path in financial_agent.py's calling
    code passed total_liabilities as total_debt, silently inflating D/E
    ~3x and mislabeling it as a precise ratio. This test only checks the
    calculator's own honesty about what it was given — the mislabeling
    bug lived in the CALLING code, not here, so see
    TestFinancialAgentEndToEnd.test_de_ratio_is_honestly_labeled below for
    the full regression coverage of that specific bug."""

    def test_basic_calculation_is_correct(self):
        result = debt_to_equity(total_debt=100, shareholders_equity=50)
        assert result["de_ratio"] == pytest.approx(2.0)

    def test_zero_equity_does_not_divide_by_zero(self):
        result = debt_to_equity(total_debt=100, shareholders_equity=0)
        assert result.get("error") is not None or result.get("de_ratio") is None


# =============================================================================
# TIER 2 — FinancialAnalystAgent end-to-end (real LLM calls, mark as slow)
# =============================================================================

@pytest.mark.slow
class TestFinancialAgentEndToEnd:

    @pytest.mark.asyncio
    async def test_populates_all_expected_fields(self, known_ticker):
        """A passing 'no exception raised' run is NOT enough — several bugs
        this project shipped for a while were cases where the agent ran
        successfully and just never populated a field it should have
        (peer_comparison, forward_pe, capital_allocation). This asserts on
        the PRESENCE and SHAPE of specific fields, not just completion."""
        agent = FinancialAnalystAgent()
        result = await agent.run({
            "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker}",
            "manager_directives": {"ticker": known_ticker},
        })
        summary = result["financial_metrics_summary"]

        required_keys = [
            "ticker", "company_name", "sector", "industry",
            "current_price", "market_cap",
            "pe_ratio", "forward_pe", "peg_ratio", "roe", "net_margin",
            "de_ratio", "revenue_cagr", "composite_score",
            "peer_comparison", "dcf_valuation", "dcf_monte_carlo",
            "capital_allocation",
            "loop_iterations_used", "validation_passed",
        ]
        missing = [k for k in required_keys if k not in summary]
        assert not missing, f"financial_metrics_summary missing keys: {missing}"

        assert summary["ticker"] == known_ticker
        assert summary["current_price"] is not None
        assert summary["composite_score"].get("score") is not None
        assert summary["de_ratio"].get("interpretation") != "unavailable", (
            "known-good ticker should have SOME D/E figure, even if approximated"
        )

    @pytest.mark.asyncio
    async def test_de_ratio_is_honestly_labeled(self, known_ticker):
        """Regression for the de_ratio mislabeling bug: if the XBRL fallback
        path was used (liabilities-to-equity approximation, not true D/E),
        the interpretation field MUST say so — it must never silently
        present an approximation as if it were the precise ratio."""
        agent = FinancialAnalystAgent()
        result = await agent.run({
            "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker}",
            "manager_directives": {"ticker": known_ticker},
        })
        de = result["financial_metrics_summary"]["de_ratio"]
        if de.get("interpretation") not in ("sourced_from_yahoo",):
            assert de.get("interpretation") == "approximation_not_comparable_to_standard_thresholds", (
                "de_ratio came from the XBRL fallback path but wasn't labeled as an approximation"
            )

    @pytest.mark.asyncio
    async def test_repeated_runs_produce_identical_categorical_grade(self, known_ticker):
        """Regression for the pre-temperature=0 instability bug: run the
        same ticker twice and require the composite grade (a categorical
        LLM-adjacent output) to agree. NOTE: makes 2x the LLM calls of a
        single run — expensive. Run on a schedule, not every commit."""
        agent = FinancialAnalystAgent()
        grades = []
        for _ in range(2):
            result = await agent.run({
                "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker}",
                "manager_directives": {"ticker": known_ticker},
            })
            grades.append(result["financial_metrics_summary"]["composite_score"].get("grade"))
        assert grades[0] == grades[1], f"composite grade unstable across runs: {grades}"
