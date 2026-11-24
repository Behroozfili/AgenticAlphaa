"""Cross-module Stage 07 gates: readiness -> model -> deterministic forecast."""

import json
import logging

import pytest

from tests.unit_tests.market_test import (
    context, feature_set, fixture, market_bundle, training_rows,
)
from tools.market_tools import (
    QUANT_IMPLEMENTATION_IDENTITY,
    QUANT_IMPLEMENTATION_VERSION,
    QUANT_LIFECYCLE_BOUNDARIES,
)
from tools.market_tools.feature_builder import build_market_features
from tools.market_tools.forecast_model import fit_deterministic_model, predict
from tools.market_tools.historical_dataset_builder import (
    build_data_readiness, build_feature_snapshot, materialize_outcome,
)
from tools.market_tools.market_server import MarketForecastService


def test_stage07_readiness_model_ready_sequence_and_reproduction():
    rows = training_rows()
    gate = build_data_readiness(rows=rows, universe_manifest=fixture()["universe"],
        run_id="stage07-e2e", source_hashes={"fixture": "locked-local"})
    assert gate.status == "PASS"
    model = fit_deterministic_model(rows=rows, readiness=gate)
    first = predict(model, feature_set()).to_dict()
    second = predict(model, feature_set()).to_dict()
    assert first == second
    assert model.training_cutoffs == 40
    assert model.readiness_manifest_hash == gate.manifest_hash
    assert all(
        row.direction_positive
        == (1 if row.ticker_total_return - row.benchmark_total_return > 0.0 else 0)
        for row in rows
    )


def test_stage07_missing_is_never_encoded_as_observed_zero():
    features = feature_set()
    assert features.values["financial_earnings_growth"] is None
    assert features.numeric_vector()["financial_earnings_growth"] is None
    assert features.numeric_vector()["financial_earnings_growth__missing"] == 1.0


def _lifecycle_events(caplog):
    prefix = "QUANT_LIFECYCLE "
    return [json.loads(record.getMessage()[len(prefix):]) for record in caplog.records
            if record.name == "quant-lifecycle" and record.getMessage().startswith(prefix)]


def test_stage07_runtime_identity_matches_active_attempt4():
    assert dict(QUANT_IMPLEMENTATION_IDENTITY) == {
        "stage": "07",
        "stage_revision": 1,
        "attempt": 4,
        "agent": "quant",
        "implementation_version": "stage07-r1-a4",
    }
    assert QUANT_IMPLEMENTATION_VERSION == "stage07-r1-a4"


def test_stage07_measured_compact_lifecycle_matrix(caplog):
    caplog.set_level(logging.INFO, logger="quant-lifecycle")
    run_id = "lifecycle-stage07"
    ctx = context()
    market = market_bundle()
    features = feature_set()
    snapshot = build_feature_snapshot(
        context=ctx, features=features, finalized_at=ctx.as_of, run_id=run_id
    )
    materialize_outcome(
        snapshot=snapshot, market=market,
        outcome_materialized_at=f"{ctx.target_date.isoformat()}T23:00:00Z", run_id=run_id,
    )
    rows = training_rows()
    gate = build_data_readiness(
        rows=rows, universe_manifest=fixture()["universe"], run_id=run_id,
        source_hashes={"fixture": "locked-local"},
    )
    model = fit_deterministic_model(rows=rows, readiness=gate, run_id=run_id)
    predict(model, features, context=ctx, run_id=run_id)
    service = MarketForecastService(
        model=model, readiness=gate,
        universe_manifest_hash=gate.manifest["universe_manifest_hash"],
        universe_constituents=tuple(fixture()["universe"]["constituents"]),
    )
    service.forecast(context=ctx, market_features=features, run_id=run_id)

    events = _lifecycle_events(caplog)
    assert QUANT_LIFECYCLE_BOUNDARIES <= {event["boundary"] for event in events}
    forbidden_keys = {
        "values", "bars", "labels", "evidence_ids", "coefficients", "exception",
        "input", "output", "payload",
    }
    for event in events:
        assert (event["stage"], event["stage_revision"], event["attempt"], event["agent"]) == (
            "07", 1, 4, "quant"
        )
        assert event["implementation_version"] == "stage07-r1-a4"
        assert event["latency_ms"] >= 0
        assert forbidden_keys.isdisjoint(event)
        assert "PRIVATE_SENTINEL" not in json.dumps(event, sort_keys=True)


def test_failure_lifecycle_does_not_emit_private_evidence_or_exception(caplog):
    market = market_bundle()
    caplog.clear()
    caplog.set_level(logging.INFO, logger="quant-lifecycle")
    upstream = {
        "values": {}, "evidence_ids": ["PRIVATE_SENTINEL raw article body"],
        "missing": False,
    }
    with pytest.raises(ValueError, match="evidence"):
        build_market_features(
            context=context(), market=market, research_features=upstream,
            run_id="privacy-failure",
        )
    serialized = json.dumps(_lifecycle_events(caplog), sort_keys=True)
    assert "PRIVATE_SENTINEL" not in serialized
    assert "raw article body" not in serialized
    assert "QUANT_FEATURE_REJECTED" in serialized
