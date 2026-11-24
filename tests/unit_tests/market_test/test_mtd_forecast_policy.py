from dataclasses import replace

from core.forecast_contract import ForecastCutoffPolicy
from tests.unit_tests.market_test import context, feature_set, fixture, training_rows
from tests.unit_tests.market_test.test_market_server import service
from tools.market_tools.historical_dataset_builder import (
    MONTH_TO_DATE_CUTOFF_POLICY_VERSION, build_data_readiness,
)
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.market_server import MarketForecastService


def test_mtd_identity_reaches_quant_and_is_degraded_for_month_end_trained_artifact():
    mtd_context = replace(context(), cutoff_policy=ForecastCutoffPolicy.MONTH_TO_DATE)
    mtd_features = replace(feature_set(), cutoff_policy=ForecastCutoffPolicy.MONTH_TO_DATE.value)

    result = service().forecast(
        context=mtd_context, market_features=mtd_features, run_id="mtd-policy-test",
    )

    assert result["cutoff_policy"] == "MONTH_TO_DATE"
    assert result["degraded"] is True
    assert "cutoff_policy_distribution_shift" in result["degraded_flags"]
    assert result["probability_positive"] is not None
    assert result["expected_return_21d"] is not None

def test_mtd_trained_artifact_is_not_degraded_for_mtd_runtime():
    rows = training_rows()
    gate = build_data_readiness(
        rows=rows, universe_manifest=fixture()["universe"], run_id="mtd-ready",
        cutoff_policy_version=MONTH_TO_DATE_CUTOFF_POLICY_VERSION,
    )
    model = fit_deterministic_model(rows=rows, readiness=gate)
    runtime = MarketForecastService(
        model=model, readiness=gate,
        universe_manifest_hash=gate.manifest["universe_manifest_hash"],
        universe_constituents=tuple(fixture()["universe"]["constituents"]),
    )
    mtd_context = replace(context(), cutoff_policy=ForecastCutoffPolicy.MONTH_TO_DATE)
    mtd_features = replace(
        feature_set(), cutoff_policy=ForecastCutoffPolicy.MONTH_TO_DATE.value
    )
    result = runtime.forecast(
        context=mtd_context, market_features=mtd_features, run_id="mtd-policy-ready",
    )
    assert result["degraded"] is False
    assert "cutoff_policy_distribution_shift" not in result["degraded_flags"]
