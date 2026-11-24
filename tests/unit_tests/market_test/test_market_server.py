from dataclasses import replace

import pytest

from tests.unit_tests.market_test import context, feature_set, fixture, training_rows
from tests.unit_tests.market_test.test_historical_dataset_builder import readiness
from tools.market_tools import QUANT_IMPLEMENTATION_IDENTITY, QUANT_IMPLEMENTATION_VERSION
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.market_server import MARKET_SERVER_VERSION, MarketForecastService


def service():
    rows = training_rows()
    gate = readiness(rows)
    model = fit_deterministic_model(rows=rows, readiness=gate)
    return MarketForecastService(model=model, readiness=gate,
        universe_manifest_hash=gate.manifest["universe_manifest_hash"],
        universe_constituents=tuple(fixture()["universe"]["constituents"]))


def test_service_forecast_has_complete_reproducibility_identity():
    features = feature_set()
    result = service().forecast(context=context(), market_features=features, run_id="run-stage07")
    required = {"ticker", "as_of", "target_date", "horizon_trading_days", "analysis_mode",
        "universe_manifest_hash", "dataset_version", "feature_version", "model_version",
        "action_policy_version", "model_variant", "benchmark", "run_id", "evidence_ids",
        "probability_positive", "expected_return_21d", "expected_excess_return",
        "confidence_interval", "market_server_version", "schema_version", "calendar_version",
        "stage", "stage_revision", "attempt", "agent", "latency_ms", "evidence_count",
        "readiness_manifest_hash", "training_data_hash", "eligible_rows_hash",
        "feature_materialized_at"}
    assert required <= set(result)
    assert result["market_server_version"] == MARKET_SERVER_VERSION
    assert result["horizon_trading_days"] == 21
    assert result["feature_materialized_at"] == features.materialized_at
    assert result["evidence_count"] == len(result["evidence_ids"])
    assert (result["stage"], result["stage_revision"], result["attempt"], result["agent"]) == (
        "07", 1, 4, "quant")
    assert result["latency_ms"] >= 0


def test_service_identity_rejects_stale_attempt_provenance():
    assert dict(QUANT_IMPLEMENTATION_IDENTITY) == {
        "stage": "07",
        "stage_revision": 1,
        "attempt": 4,
        "agent": "quant",
        "implementation_version": "stage07-r1-a4",
    }
    assert QUANT_IMPLEMENTATION_VERSION == "stage07-r1-a4"
    result = service().forecast(
        context=context(), market_features=feature_set(), run_id="attempt4-identity"
    )
    assert result["attempt"] == 4
    assert result["attempt"] != 1


def test_service_rejects_model_readiness_mismatch():
    rows = training_rows()
    gate = readiness(rows)
    model = fit_deterministic_model(rows=rows, readiness=gate)
    with pytest.raises(ValueError, match="universe manifest"):
        MarketForecastService(
            model=model, readiness=gate, universe_manifest_hash="wrong",
            universe_constituents=tuple(fixture()["universe"]["constituents"]),
        )


def test_service_rejects_blank_run_id():
    with pytest.raises(ValueError, match="run_id"):
        service().forecast(context=context(), market_features=feature_set(), run_id="")


@pytest.mark.parametrize("changes", [
    {"ticker": "AAPL"}, {"as_of": "2024-05-31T20:00:00Z"},
    {"dataset_version": "wrong-dataset"}, {"feature_version": "wrong-features"},
    {"schema_version": "wrong-schema"}, {"calendar_version": "wrong-calendar"},
])
def test_service_rejects_market_feature_context_identity_mismatch(changes):
    with pytest.raises(ValueError, match="ForecastContext identity"):
        service().forecast(
            context=context(), market_features=replace(feature_set(), **changes),
            run_id="identity-counterexample",
        )


@pytest.mark.parametrize("run_id", [
    "PRIVATE_SENTINEL", "multiline\nrun", " leading", "trailing ", "x" * 129,
])
def test_service_rejects_nonopaque_run_id(run_id):
    with pytest.raises(ValueError, match="run_id"):
        service().forecast(context=context(), market_features=feature_set(), run_id=run_id)
