from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import api.main as main_module
from api.config import Settings
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.historical_dataset_builder import build_data_readiness
from tools.market_tools.quant_artifacts import build_quant_artifact_bundle, save_quant_artifact_bundle
from tests.unit_tests.market_test import fixture, training_rows


def _artifact(path):
    rows = training_rows()
    universe = fixture()["universe"]
    readiness = build_data_readiness(
        rows=rows, universe_manifest=universe, run_id="runtime-wiring-test",
    )
    model = fit_deterministic_model(
        rows=rows, readiness=readiness, run_id="runtime-wiring-test",
    )
    payload = build_quant_artifact_bundle(
        model=model, readiness=readiness, universe_manifest=universe,
    )
    save_quant_artifact_bundle(path, payload)
    return payload


def _settings(**overrides):
    values = {
        "ANTHROPIC_API_KEY": "test-key",
        "SUPABASE_URL": "https://example.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "service-key",
        "NEO4J_URI": "bolt://example",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "APP_ENV": "test",
        "QUANT_ARTIFACT_PATH": "",
        "QUANT_RUNTIME_REQUIRED": False,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_startup_loads_real_artifact_and_injects_quant_dependencies(tmp_path):
    artifact_path = tmp_path / "quant-runtime-v1.json"
    payload = _artifact(artifact_path)
    runtime = _settings(
        QUANT_ARTIFACT_PATH=str(artifact_path),
        MARKET_DATA_CACHE_DIR=str(tmp_path / "market-cache"),
        QUANT_RUNTIME_REQUIRED=True,
    )
    application = MagicMock()
    application.state.runtime_settings = runtime
    application.state.telemetry_sink = None

    with patch("api.main.create_client", return_value=MagicMock()), patch(
        "api.main.GraphDatabase.driver", return_value=MagicMock()
    ), patch("api.main.init_sentry", return_value=False), patch(
        "api.main.init_langsmith", return_value=False
    ):
        async with main_module.lifespan(application):
            factory = application.state.manager_factory
            assert factory.market_forecast_service is not None
            assert factory.market_data_provider is not None
            assert factory.market_forecast_service.model.model_version == payload["model"]["model_version"]
            assert application.state.runtime_settings.FORECAST_MODEL_VERSION == payload["model"]["model_version"]
            assert application.state.runtime_settings.FORECAST_FEATURE_VERSION == payload["model"]["feature_version"]
            assert application.state.runtime_settings.LIVE_DATASET_VERSION == payload["model"]["dataset_version"]


def test_required_quant_runtime_without_artifact_fails_closed():
    with pytest.raises(Exception, match="QUANT_ARTIFACT_PATH is required"):
        main_module._configure_quant_runtime(_settings(QUANT_RUNTIME_REQUIRED=True))

