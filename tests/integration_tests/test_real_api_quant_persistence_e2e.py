"""Offline API/Quant integration with synthetic inputs and mocked persistence; no live providers or database."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agents.manager_agent import ManagerAgent
from agents.monthly_forecast_agent import MonthlyForecastAgent
from api.config import Settings
from api.dependencies import get_manager_memory
from api.routes.analyze import router
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.historical_dataset_builder import build_data_readiness
from tools.market_tools.market_provider import YahooMarketDataProvider
from tools.market_tools.market_server import MarketForecastService
from tests.integration_tests.test_real_monthly_quant_smoke import _Specialist, _history
from tests.unit_tests.market_test import fixture, training_rows


def _client(tmp_path):
    rows = training_rows()
    universe = fixture()["universe"]
    readiness = build_data_readiness(
        rows=rows, universe_manifest=universe, run_id="real-api-training",
    )
    model = fit_deterministic_model(
        rows=rows, readiness=readiness, run_id="real-api-training",
    )
    service = MarketForecastService(
        model=model, readiness=readiness,
        universe_manifest_hash=readiness.manifest["universe_manifest_hash"],
        universe_constituents=tuple(universe["constituents"]),
    )
    calls: list[str] = []

    def fetch(symbol, start, end):
        calls.append(symbol)
        return _history(start, end, 100 if symbol == "MSFT" else 200), []

    provider = YahooMarketDataProvider(
        cache_dir=tmp_path / "market-cache", fetcher=fetch,
        clock=lambda: datetime(2024, 6, 29, tzinfo=UTC),
    )
    manager = ManagerAgent(
        _Specialist("research"), _Specialist("financial"), _Specialist("sentiment"),
        MagicMock(), llm_client=AsyncMock(),
        monthly_forecast_agent=MonthlyForecastAgent(),
        market_forecast_service=service, market_data_provider=provider,
    )
    tables = {name: MagicMock() for name in ("analyses", "monthly_forecasts", "feature_snapshots")}
    database = MagicMock()
    database.table.side_effect = lambda name: tables[name]
    def commit(name, args):
        assert name == "persist_monthly_bundle"
        for table, key in (("analyses", "p_analysis"), ("feature_snapshots", "p_snapshot"), ("monthly_forecasts", "p_forecast")):
            tables[table].upsert(args[key])  # Capture the single RPC bundle for existing assertions.
        return MagicMock(execute=MagicMock(return_value=MagicMock(data={"status": "SAVED", "run_id": args["p_forecast"]["run_id"]})))
    database.rpc.side_effect = commit
    runtime = Settings(
        _env_file=None,
        FORECAST_SCHEMA_VERSION="forecast-v1",
        LIVE_DATASET_VERSION=model.dataset_version,
        HISTORICAL_DATASET_VERSION=model.dataset_version,
        FORECAST_FEATURE_VERSION=model.feature_version,
        FORECAST_MODEL_VERSION=model.model_version,
    )
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")
    application.state.manager_agent = manager
    application.state.supabase = database
    application.state.runtime_settings = runtime
    memory = MagicMock()
    memory.long.get_all_preferences.return_value = {}
    application.dependency_overrides[get_manager_memory] = lambda: memory
    return TestClient(application, raise_server_exceptions=False), tables, calls


def test_api_to_real_quant_and_supabase_row_is_repeatable(tmp_path):
    client, tables, calls = _client(tmp_path)
    payload = {
        "query": "Forecast MSFT for the next 21 trading sessions",
        "ticker": "MSFT", "as_of": "2024-06-28", "mode": "LIVE",
        "cutoff_policy": "MONTH_END", "horizon": 21,
        "run_id": "a6f60e38-7726-4a62-bfea-1e60d2b90767",
    }

    first = client.post("/api/v1/analyze", json=payload)
    first_row = dict(tables["monthly_forecasts"].upsert.call_args.args[0])
    second = client.post("/api/v1/analyze", json=payload)
    second_row = dict(tables["monthly_forecasts"].upsert.call_args.args[0])

    assert first.status_code == second.status_code == 200
    left, right = first.json()["monthly_forecast"], second.json()["monthly_forecast"]
    assert left["status"] == "READY"
    assert left["quality_gate_passed"] is True
    assert first.json()["publication_status"] == "PUBLISHED"
    assert left["probability_positive"] == right["probability_positive"]
    assert left["expected_return_21d"] == right["expected_return_21d"]
    assert left["versions"] == right["versions"]
    assert calls == ["MSFT", "SPY"]
    assert first_row == second_row
    assert first_row["feature_snapshot_id"] == tables["feature_snapshots"].upsert.call_args.args[0]["snapshot_id"]
    assert first.json()["persistence_status"] == "SAVED"
    assert first_row["cutoff_policy"] == "MONTH_END"
    assert first_row["predicted_ticker_return"] is not None
    assert first_row["forecast_payload"]["universe_manifest_hash"]
    assert tables["monthly_forecasts"].upsert.call_count == 2



def test_frontend_sends_mtd_policy_and_renders_monthly_forecast_contract():
    html = (Path(__file__).parents[2] / "frontend" / "alpha-agent-app.html").read_text(encoding="utf-8")
    assert 'id="cutoff-policy-input"' in html
    assert '<option value="MONTH_TO_DATE">' in html
    assert 'cutoff_policy:cutoffPolicy' in html
    assert '21-session forecast' in html

