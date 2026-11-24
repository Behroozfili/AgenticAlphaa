from __future__ import annotations

import copy

import pytest

from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.historical_dataset_builder import build_data_readiness
from tools.market_tools.market_data import canonical_hash
from tools.market_tools.quant_artifacts import (
    QuantArtifactError,
    build_quant_artifact_bundle,
    load_quant_artifacts,
    save_quant_artifact_bundle,
)
from tests.unit_tests.market_test import fixture, training_rows


def _bundle():
    rows = training_rows()
    universe = fixture()["universe"]
    readiness = build_data_readiness(
        rows=rows, universe_manifest=universe, run_id="artifact-test-run",
    )
    model = fit_deterministic_model(
        rows=rows, readiness=readiness, run_id="artifact-test-run",
    )
    return build_quant_artifact_bundle(
        model=model, readiness=readiness, universe_manifest=universe,
    )


def _rehash(payload):
    payload.pop("artifact_hash", None)
    payload["artifact_hash"] = canonical_hash(payload)
    return payload


def test_artifact_round_trip_builds_valid_runtime_service(tmp_path):
    payload = _bundle()
    path = tmp_path / "quant-runtime-v1.json"
    save_quant_artifact_bundle(path, payload)

    loaded = load_quant_artifacts(
        path,
        expected_dataset_version=payload["model"]["dataset_version"],
        expected_feature_version=payload["model"]["feature_version"],
        expected_model_version=payload["model"]["model_version"],
    )

    assert loaded.artifact_hash == payload["artifact_hash"]
    assert loaded.model.artifact() == payload["model"]
    assert loaded.readiness.status == "PASS"
    assert loaded.service().universe_manifest_hash == payload["universe_manifest"]["manifest_hash"]


def test_artifact_tampering_fails_even_when_outer_hash_is_recomputed():
    payload = copy.deepcopy(_bundle())
    payload["model"]["logistic"]["coefficients_with_intercept"][0] += 0.1
    _rehash(payload)

    with pytest.raises(QuantArtifactError, match="model version") as raised:
        load_quant_artifacts(payload)
    assert raised.value.error_code == "QUANT_ARTIFACT_REJECTED"


def test_readiness_hash_and_configured_version_mismatches_fail_closed():
    payload = _bundle()
    broken = copy.deepcopy(payload)
    broken["readiness"]["ticker_count"] += 1
    _rehash(broken)
    with pytest.raises(QuantArtifactError, match="readiness manifest hash"):
        load_quant_artifacts(broken)

    with pytest.raises(QuantArtifactError, match="configured model version"):
        load_quant_artifacts(payload, expected_model_version="wrong-model-v1")


def test_published_artifact_path_is_immutable(tmp_path):
    path = tmp_path / "quant-runtime-v1.json"
    payload = _bundle()
    save_quant_artifact_bundle(path, payload)
    with pytest.raises(QuantArtifactError, match="already exists"):
        save_quant_artifact_bundle(path, payload)

