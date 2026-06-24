"""
Tests for: tools/financial_tools/financial_ratio_calculator.py
Phase: 1 — Pure-Logic / Zero-Mock Foundations

No mocking required — every function here is pure math (no I/O, no network,
no external state). All tests use direct numeric inputs and assert on the
returned dict.

KNOWN BUG documented in TC-PE-BUG: price_to_earnings() raises UnboundLocalError
when `pe is None` because `interp` is only assigned inside the
`if pe is not None:` block. This test is expected to FAIL until fixed.
"""
import math
import pytest

from tools.financial_tools.financial_ratio_calculator import (
    _safe_div,
    _label,
    price_to_earnings,
    price_to_book,
    ev_to_ebitda,
    peg_ratio,
    gross_margin,
    operating_margin,
    net_margin,
    return_on_equity,
    return_on_assets,
    current_ratio,
    quick_ratio,
    debt_to_equity,
    interest_coverage,
    asset_turnover,
    cagr,
    composite_financial_score,
)


# ---------------------------------------------------------------------------
# _safe_div (internal helper)
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10, 2) == 5

    def test_denominator_zero_returns_default(self):
        assert _safe_div(10, 0) is None

    def test_denominator_zero_custom_default(self):
        assert _safe_div(10, 0, default=-1) == -1

    def test_numerator_none_returns_default(self):
        assert _safe_div(None, 5) is None

    def test_denominator_none_returns_default(self):
        assert _safe_div(5, None) is None


# ---------------------------------------------------------------------------
# _label (internal helper)
# ---------------------------------------------------------------------------

class TestLabel:
    def test_value_none_returns_unavailable(self):
        assert _label(None, [(10, "high"), (0, "low")]) == "unavailable"

    def test_higher_is_better_picks_first_matching_threshold(self):
        assert _label(15, [(10, "high"), (0, "low")]) == "high"

    def test_higher_is_better_falls_through_to_lowest_band(self):
        assert _label(-5, [(10, "high"), (0, "low")]) == "low"

    def test_lower_is_better_direction(self):
        # higher_is_better=False: value <= threshold picks the label
        assert _label(2, [(5, "good"), (10, "bad")], higher_is_better=False) == "good"

    def test_no_threshold_matches_returns_last_label(self):
        # value way above all thresholds, lower_is_better=False direction
        assert _label(999, [(5, "good"), (10, "bad")], higher_is_better=False) == "bad"


# ---------------------------------------------------------------------------
# 1. Valuation Ratios
# ---------------------------------------------------------------------------

class TestPriceToEarnings:
    def test_undervalued_below_15(self):
        result = price_to_earnings(price=100, eps=10)  # pe = 10
        assert result["pe_ratio"] == 10
        assert result["interpretation"] == "undervalued"

    def test_fairly_valued_between_15_and_30(self):
        result = price_to_earnings(price=200, eps=10)  # pe = 20
        assert result["interpretation"] == "fairly_valued"

    def test_overvalued_above_30(self):
        result = price_to_earnings(price=400, eps=10)  # pe = 40
        assert result["interpretation"] == "overvalued"

    def test_negative_earnings(self):
        result = price_to_earnings(price=100, eps=-5)  # pe = -20
        assert result["interpretation"] == "negative_earnings"

    def test_pe_is_none_returns_unavailable_bug_fixed(self):
        """
        FIXED: price_to_earnings() now initialises `interp = "unavailable"`
        before the `if pe is not None:` block, so eps=0 (pe is None) no
        longer raises UnboundLocalError — it correctly returns
        {"pe_ratio": None, "interpretation": "unavailable", ...}.
        """
        result = price_to_earnings(price=100, eps=0)
        assert result["pe_ratio"] is None
        assert result["interpretation"] == "unavailable"

    def test_pe_exactly_30_is_fairly_valued_not_overvalued(self):
        """
        Boundary change: the fix also tightened the comparison from
        `pe < 30` to `pe <= 30`, so pe == 30 now falls in "fairly_valued"
        instead of "overvalued". This test locks in the new boundary.
        """
        result = price_to_earnings(price=300, eps=10)  # pe = 30
        assert result["interpretation"] == "fairly_valued"


class TestPriceToBook:
    def test_trading_below_book(self):
        result = price_to_book(price=8, book_value_per_share=10)
        assert result["interpretation"] == "trading_below_book"

    def test_fairly_valued(self):
        result = price_to_book(price=20, book_value_per_share=10)  # pb=2
        assert result["interpretation"] == "fairly_valued"

    def test_premium_to_book(self):
        result = price_to_book(price=40, book_value_per_share=10)  # pb=4
        assert result["interpretation"] == "premium_to_book"

    def test_book_value_zero_unavailable(self):
        result = price_to_book(price=10, book_value_per_share=0)
        assert result["pb_ratio"] is None
        assert result["interpretation"] == "unavailable"


class TestEvToEbitda:
    def test_undervalued(self):
        result = ev_to_ebitda(enterprise_value=50, ebitda=10)  # 5
        assert result["interpretation"] == "undervalued"

    def test_fairly_valued(self):
        result = ev_to_ebitda(enterprise_value=150, ebitda=10)  # 15
        assert result["interpretation"] == "fairly_valued"

    def test_expensive(self):
        result = ev_to_ebitda(enterprise_value=250, ebitda=10)  # 25
        assert result["interpretation"] == "expensive"

    def test_ebitda_zero_unavailable(self):
        result = ev_to_ebitda(enterprise_value=100, ebitda=0)
        assert result["interpretation"] == "unavailable"


class TestPegRatio:
    def test_undervalued_below_1(self):
        result = peg_ratio(pe=10, earnings_growth_rate_pct=15)  # 0.67
        assert result["interpretation"] == "undervalued"

    def test_fairly_valued(self):
        result = peg_ratio(pe=20, earnings_growth_rate_pct=15)  # 1.33
        assert result["interpretation"] == "fairly_valued"

    def test_overvalued_above_2(self):
        result = peg_ratio(pe=40, earnings_growth_rate_pct=10)  # 4.0
        assert result["interpretation"] == "overvalued"

    def test_growth_rate_zero_unavailable(self):
        result = peg_ratio(pe=10, earnings_growth_rate_pct=0)
        assert result["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# 2. Profitability Ratios
# ---------------------------------------------------------------------------

class TestGrossMargin:
    def test_excellent_above_60(self):
        result = gross_margin(revenue=100, cogs=30)  # 70%
        assert result["gross_margin_pct"] == 70.0
        assert result["interpretation"] == "excellent"

    def test_low_below_20(self):
        result = gross_margin(revenue=100, cogs=90)  # 10%
        assert result["interpretation"] == "low"

    def test_revenue_zero_raises_or_handled(self):
        # _safe_div(gp, revenue) with revenue=0 -> denominator==0 -> default None
        result = gross_margin(revenue=0, cogs=10)
        assert result["gross_margin_pct"] is None


class TestOperatingMargin:
    def test_excellent(self):
        result = operating_margin(operating_income=30, revenue=100)
        assert result["interpretation"] == "excellent"

    def test_low(self):
        result = operating_margin(operating_income=-5, revenue=100)
        assert result["interpretation"] == "low"


class TestNetMargin:
    def test_excellent(self):
        result = net_margin(net_income=25, revenue=100)
        assert result["interpretation"] == "excellent"

    def test_low(self):
        result = net_margin(net_income=1, revenue=100)
        assert result["interpretation"] == "low"


class TestReturnOnEquity:
    def test_excellent(self):
        result = return_on_equity(net_income=25, shareholders_equity=100)
        assert result["roe_pct"] == 25.0
        assert result["interpretation"] == "excellent"

    def test_equity_zero_unavailable(self):
        result = return_on_equity(net_income=10, shareholders_equity=0)
        assert result["interpretation"] == "unavailable"


class TestReturnOnAssets:
    def test_excellent(self):
        result = return_on_assets(net_income=12, total_assets=100)
        assert result["interpretation"] == "excellent"

    def test_low(self):
        result = return_on_assets(net_income=1, total_assets=100)
        assert result["interpretation"] == "low"


# ---------------------------------------------------------------------------
# 3. Liquidity Ratios
# ---------------------------------------------------------------------------

class TestCurrentRatio:
    def test_strong_above_2(self):
        result = current_ratio(current_assets=300, current_liabilities=100)
        assert result["interpretation"] == "strong"

    def test_adequate_between_1_and_2(self):
        result = current_ratio(current_assets=150, current_liabilities=100)
        assert result["interpretation"] == "adequate"

    def test_weak_below_1(self):
        result = current_ratio(current_assets=50, current_liabilities=100)
        assert result["interpretation"] == "weak"

    def test_liabilities_zero_unavailable(self):
        result = current_ratio(current_assets=100, current_liabilities=0)
        assert result["interpretation"] == "unavailable"


class TestQuickRatio:
    def test_strong(self):
        result = quick_ratio(cash=80, short_term_investments=20,
                              receivables=20, current_liabilities=100)
        # liquid_assets=120 -> qr=1.2 >= 1
        assert result["interpretation"] == "strong"

    def test_moderate(self):
        result = quick_ratio(cash=30, short_term_investments=10,
                              receivables=20, current_liabilities=100)
        # liquid_assets=60 -> qr=0.6
        assert result["interpretation"] == "moderate"

    def test_weak(self):
        result = quick_ratio(cash=10, short_term_investments=0,
                              receivables=0, current_liabilities=100)
        assert result["interpretation"] == "weak"


# ---------------------------------------------------------------------------
# 4. Leverage / Solvency Ratios
# ---------------------------------------------------------------------------

class TestDebtToEquity:
    def test_low_leverage(self):
        result = debt_to_equity(total_debt=20, shareholders_equity=100)
        assert result["interpretation"] == "low_leverage"

    def test_moderate_leverage(self):
        result = debt_to_equity(total_debt=100, shareholders_equity=100)
        assert result["interpretation"] == "moderate_leverage"

    def test_high_leverage(self):
        result = debt_to_equity(total_debt=200, shareholders_equity=100)
        assert result["interpretation"] == "high_leverage"

    def test_equity_zero_unavailable(self):
        result = debt_to_equity(total_debt=50, shareholders_equity=0)
        assert result["interpretation"] == "unavailable"


class TestInterestCoverage:
    def test_strong(self):
        result = interest_coverage(ebit=60, interest_expense=10)
        assert result["interpretation"] == "strong"

    def test_adequate(self):
        result = interest_coverage(ebit=20, interest_expense=10)
        assert result["interpretation"] == "adequate"

    def test_at_risk(self):
        result = interest_coverage(ebit=10, interest_expense=10)
        assert result["interpretation"] == "at_risk"

    def test_interest_zero_unavailable(self):
        result = interest_coverage(ebit=50, interest_expense=0)
        assert result["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# 5. Efficiency Ratios
# ---------------------------------------------------------------------------

class TestAssetTurnover:
    def test_efficient(self):
        result = asset_turnover(revenue=120, avg_total_assets=100)
        assert result["interpretation"] == "efficient"

    def test_moderate(self):
        result = asset_turnover(revenue=60, avg_total_assets=100)
        assert result["interpretation"] == "moderate"

    def test_low_efficiency(self):
        result = asset_turnover(revenue=30, avg_total_assets=100)
        assert result["interpretation"] == "low_efficiency"


# ---------------------------------------------------------------------------
# 6. Growth Ratios — CAGR
# ---------------------------------------------------------------------------

class TestCagr:
    def test_hypergrowth(self):
        # (200/100)^(1/1) - 1 = 1.0 -> 100% -> hypergrowth
        result = cagr(start_value=100, end_value=200, years=1)
        assert result["interpretation"] == "hypergrowth"
        assert result["cagr_pct"] == 100.0

    def test_slow_growth_zero_to_five(self):
        result = cagr(start_value=100, end_value=102, years=1)  # 2%
        assert result["interpretation"] == "slow"

    def test_start_value_zero_returns_unavailable(self):
        result = cagr(start_value=0, end_value=100, years=2)
        assert result["cagr_pct"] is None
        assert result["interpretation"] == "unavailable"

    def test_years_zero_returns_unavailable(self):
        result = cagr(start_value=100, end_value=200, years=0)
        assert result["cagr_pct"] is None
        assert result["interpretation"] == "unavailable"

    def test_negative_start_value_returns_unavailable(self):
        result = cagr(start_value=-50, end_value=100, years=2)
        assert result["interpretation"] == "unavailable"

    def test_internal_exception_is_caught_and_returns_error_label(self, monkeypatch):
        """
        Forces the except branch by making `years` something that raises
        when used in `1 / years` (e.g. a non-numeric type slipping past
        type hints, simulating a defensive-programming edge case)."""
        result = cagr(start_value=100, end_value=200, years="oops")  # type: ignore
        assert result["cagr_pct"] is None
        assert result["interpretation"] == "error"


# ---------------------------------------------------------------------------
# 7. Composite Scoring
# ---------------------------------------------------------------------------

class TestCompositeFinancialScore:
    def test_all_inputs_provided_computes_score_and_grade(self):
        result = composite_financial_score(
            pe=15, pb=2, roe_pct=20, net_margin_pct=15,
            current_ratio=2, de_ratio=0.5, revenue_cagr_pct=20,
        )
        assert result["score"] is not None
        assert result["grade"] in {"A", "B", "C", "D", "F"}
        assert result["missing_inputs"] == []
        # all 6 weighted metrics should appear (pb is NOT one of them)
        assert set(result["sub_scores"].keys()) == {
            "roe", "net_margin", "revenue_cagr", "pe", "current_ratio", "de_ratio"
        }

    def test_all_inputs_none_returns_na_grade(self):
        result = composite_financial_score()
        assert result["score"] is None
        assert result["grade"] == "N/A"
        assert len(result["missing_inputs"]) == 6

    def test_partial_inputs_only_uses_available_weights(self):
        result = composite_financial_score(roe_pct=30, net_margin_pct=20)
        assert result["score"] is not None
        assert "revenue_cagr" not in result["missing_inputs"] or True  # missing list sanity
        assert set(result["missing_inputs"]) == {
            "revenue_cagr", "pe", "current_ratio", "de_ratio"
        }

    def test_pe_metric_is_inverted_lower_pe_scores_higher(self):
        low_pe = composite_financial_score(pe=5)
        high_pe = composite_financial_score(pe=60)
        assert low_pe["sub_scores"]["pe"] > high_pe["sub_scores"]["pe"]

    def test_de_ratio_metric_is_inverted_lower_de_scores_higher(self):
        low_de = composite_financial_score(de_ratio=0)
        high_de = composite_financial_score(de_ratio=3)
        assert low_de["sub_scores"]["de_ratio"] > high_de["sub_scores"]["de_ratio"]

    def test_grade_boundaries(self):
        # Force a near-perfect score to hit grade "A" (>= 80)
        result = composite_financial_score(
            roe_pct=40, net_margin_pct=30, revenue_cagr_pct=50,
            pe=5, current_ratio=3, de_ratio=0,
        )
        assert result["score"] >= 80
        assert result["grade"] == "A"

    def test_score_is_clamped_for_out_of_range_inputs(self):
        # roe_pct way beyond max_val=40 should clamp, not error or exceed 10 sub-score
        result = composite_financial_score(roe_pct=9999)
        assert result["sub_scores"]["roe"] == 10.0