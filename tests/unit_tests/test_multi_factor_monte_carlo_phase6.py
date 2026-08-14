from tools.financial_tools.financial_ratio_calculator import (
    dcf_monte_carlo, dcf_multi_factor_monte_carlo,
)


def test_multi_factor_model_is_reproducible_and_ordered():
    result = dcf_multi_factor_monte_carlo(
        [110, 120, 130, 140, 150], 9.0, 2.5, cash=50, total_debt=20,
        diluted_shares=10, n_simulations=500, seed=7,
    )
    assert result["model_classification"] == "MULTI_FACTOR"
    assert result["enterprise_value_p10"] < result["enterprise_value_p50"] < result["enterprise_value_p90"]
    assert result["intrinsic_value_per_share_p10"] < result["intrinsic_value_per_share_p90"]
    assert len(result["stochastic_drivers"]) == 4
    assert result["correlations"]["demand_margin"] == 0.60


def test_bounds_keep_terminal_growth_below_wacc():
    result = dcf_multi_factor_monte_carlo([100, 105], 5.1, 5.0, n_simulations=200, seed=1)
    assert result["error"] is None
    assert result["rejected_draws"] == 0


def test_invalid_explicit_path_fails_closed():
    result = dcf_multi_factor_monte_carlo([100, -1], 9, 2.5)
    assert result["error"]


def test_legacy_single_variable_model_discloses_limited_scope():
    result = dcf_monte_carlo(100, 1.0, 8.0, n_simulations=100, seed=1)
    assert result["model_classification"] == "LIMITED"
    assert "growth rate only" in result["note"]
