import pytest

from evaluation.forecast_metrics import compute_forecast_metrics


def _row(ticker, as_of, label, probability, actual, predicted, action):
    return {
        "ticker": ticker,
        "as_of": as_of,
        "actual_direction_positive": label,
        "probability_positive": probability,
        "actual_excess_return": actual,
        "expected_excess_return": predicted,
        "action": action,
    }


def test_hand_calculated_direction_probability_amount_and_coverage_metrics():
    rows = [
        _row("A", "2020-01-31T21:00:00Z", 1, 0.9, 0.04, 0.03, "BUY"),
        _row("A", "2020-02-28T21:00:00Z", 0, 0.2, -0.02, -0.01, "AVOID"),
        _row("B", "2020-01-31T21:00:00Z", 1, 0.8, 0.02, 0.01, "BUY"),
        _row("B", "2020-02-28T21:00:00Z", 0, 0.1, -0.04, -0.03, "AVOID"),
    ]
    result = compute_forecast_metrics(rows, expected_forecasts=4)
    assert result["direction"]["balanced_accuracy"] == 1.0
    assert result["direction"]["precision"] == 1.0
    assert result["direction"]["recall"] == 1.0
    assert result["direction"]["f1"] == 1.0
    assert result["direction"]["roc_auc"] == 1.0
    assert result["probability"]["brier"] == pytest.approx(0.025)
    assert result["amount"]["mae_excess_return"] == pytest.approx(0.01)
    assert result["amount"]["rmse_excess_return"] == pytest.approx(0.01)
    assert result["coverage"]["fraction"] == 1.0


def test_investment_metrics_report_gross_net_cost_turnover_drawdown_and_sharpe():
    rows = [
        _row("A", "2020-01-31T21:00:00Z", 1, 0.9, 0.04, 0.03, "BUY"),
        _row("A", "2020-02-28T21:00:00Z", 0, 0.2, -0.02, -0.01, "AVOID"),
    ]
    investment = compute_forecast_metrics(rows, expected_forecasts=2)["investment"]
    assert investment["gross_excess_return_mean"] == pytest.approx(0.03)
    assert investment["turnover_mean"] == pytest.approx(1.5)
    assert investment["transaction_cost_mean"] == pytest.approx(0.0015)
    assert investment["net_excess_return_mean"] == pytest.approx(0.0285)
    assert investment["hit_rate"] == 1.0
    assert "max_drawdown_net" in investment
    assert "sharpe_net_annualized" in investment


def test_invalid_probability_fails_closed():
    with pytest.raises(ValueError, match="probabilities"):
        compute_forecast_metrics([_row("A", "x", 1, 1.1, 0.1, 0.1, "BUY")], expected_forecasts=1)


def test_metric_telemetry_is_count_only_and_failure_uses_stable_code():
    valid = [_row("A", "2020-01-31T21:00:00Z", 1, 0.9, 0.04, 0.03, "BUY")]
    events = []
    compute_forecast_metrics(valid, expected_forecasts=1, model_variant="full_model",
                             event_sink=lambda event: events.append(dict(event)))
    assert len(events) == 1
    assert events[0]["boundary"] == "metric_aggregation"
    assert events[0]["forecast_count"] == 1
    assert not ({"records", "labels", "predictions", "actual_excess_return"} & set(events[0]))
    failed = []
    with pytest.raises(ValueError, match="probabilities"):
        compute_forecast_metrics(
            [_row("A", "PRIVATE_SENTINEL", 1, 1.1, 0.1, 0.1, "BUY")],
            expected_forecasts=1, event_sink=lambda event: failed.append(dict(event)),
        )
    assert failed[0]["outcome"] == "fail_closed"
    assert failed[0]["error_code"] == "STAGE09_METRICS_REJECTED"
    assert "PRIVATE_SENTINEL" not in repr(failed)
