"""Offline integration of real Quant code with synthetic prices and stub specialists; not a historical multi-agent evaluation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
from agents.monthly_forecast_agent import MonthlyForecastAgent
from core.forecast_contract import AnalysisMode, ForecastContext, is_us_regular_trading_session
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.historical_dataset_builder import build_data_readiness
from tools.market_tools.market_provider import YahooMarketDataProvider
from tools.market_tools.market_server import MarketForecastService
from tests.unit_tests.market_test import fixture, training_rows


def _slot(kind: str, context: dict, run_id: str) -> dict:
    values: dict = {"dataset_version": context["dataset_version"], "run_id": run_id}
    if kind == "research":
        values.update({
            "timestamps": [context["as_of"]], "graph_coverage": True,
            "vector_coverage": True, "source_diversity": 2,
            "catalysts": ["research:msft"], "risks": [],
        })
    return {
        "version": f"{kind}-v1", "values": values,
        "evidence_ids": [f"{kind}:msft"],
        "missing": False, "degraded": False, "errors": [],
    }


class _Specialist:
    def __init__(self, kind: str):
        self.kind = kind

    async def run(self, state):
        context = state["forecast_context"]
        run_id = state["manager_directives"]["run_id"]
        return {**state, f"{self.kind}_features": _slot(self.kind, context, run_id)}


def _history(start: date, end: date, base: float):
    rows = []
    cursor = start
    index = 0
    while cursor <= end:
        if is_us_regular_trading_session(cursor):
            close = base + 0.2 * index + 0.01 * (index % 7)
            rows.append({
                "date": cursor.isoformat(), "open": close - 0.1,
                "high": close + 0.3, "low": close - 0.3, "close": close,
                "adjusted_close": close, "volume": 1_000_000 + 100 * index,
            })
            index += 1
        cursor += timedelta(days=1)
    return rows


@pytest.mark.asyncio
async def test_real_manager_quant_monthly_chain_is_numeric_and_repeatable(tmp_path):
    rows = training_rows()
    universe = fixture()["universe"]
    readiness = build_data_readiness(
        rows=rows, universe_manifest=universe, run_id="real-smoke-training",
    )
    model = fit_deterministic_model(
        rows=rows, readiness=readiness, run_id="real-smoke-training",
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
    context = ForecastContext(
        ticker="MSFT", as_of="2024-06-28T20:00:00Z", benchmark="SPY",
        analysis_mode=AnalysisMode.LIVE, schema_version="forecast-v1",
        dataset_version=model.dataset_version, feature_version=model.feature_version,
        model_version=model.model_version,
    )
    manager = ManagerAgent(
        _Specialist("research"), _Specialist("financial"), _Specialist("sentiment"),
        MagicMock(), llm_client=AsyncMock(),
        monthly_forecast_agent=MonthlyForecastAgent(),
        market_forecast_service=service, market_data_provider=provider,
    )
    directives = {
        "ticker": "MSFT", "run_id": "real-smoke-msft",
        "forecast_context": context.to_dict(),
    }

    first = await manager.run("Forecast MSFT", directives)
    second = await manager.run("Forecast MSFT", directives)

    left, right = first["monthly_forecast"], second["monthly_forecast"]
    assert left["status"] in {"READY", "DEGRADED"}
    assert left["action"] in {"BUY", "HOLD", "AVOID"}
    assert isinstance(left["probability_positive"], float)
    assert isinstance(left["expected_return_21d"], float)
    assert isinstance(left["expected_excess_return"], float)
    for field in (
        "probability_positive", "expected_return_21d", "expected_excess_return",
        "confidence_interval", "action", "versions",
    ):
        assert left[field] == right[field]
    assert calls == ["MSFT", "SPY"]
    assert "QUANT_SERVICE_NOT_CONFIGURED" not in str(first)

