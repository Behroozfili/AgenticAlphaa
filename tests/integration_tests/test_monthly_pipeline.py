"""Offline Stage 10B API-to-Manager monthly concurrency integration test."""

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from api.dependencies import get_manager_memory
from api.routes.analyze import router


class EchoMonthlyManager:
    async def run(self, **kwargs):
        directives = kwargs["manager_directives"]
        context = directives["forecast_context"]
        await asyncio.sleep(0)
        return {
            "execution_status": "completed",
            "publication_status": "PUBLISHED",
            "final_publication_validation": {"passed": True},
            "monthly_forecast": {
                "status": "READY",
                "ticker": context["ticker"],
                "as_of": context["as_of"],
                "target_date": context["target"],
                "horizon_trading_days": context["horizon_trading_days"],
                "analysis_mode": context["analysis_mode"],
                "benchmark": context["benchmark"],
                "run_id": directives["run_id"],
                "universe_manifest_hash": "a" * 64,
                "model_variant": "FULL_MODEL",
                "feature_materialized_at": context["as_of"],
                "action": "HOLD",
                "probability_positive": 0.5,
                "expected_return_21d": 0.01,
                "expected_excess_return": 0.0,
                "confidence_interval": [-0.02, 0.04],
                "confidence": 0.6,
                "evidence_ids": [f"evidence-{context['ticker']}"],
                "warnings": [], "missing": False, "degraded": False,
                "graph_coverage": True,
                "versions": {
                    "schema_version": context["schema_version"],
                    "dataset_version": context["dataset_version"],
                    "feature_version": context["feature_version"],
                    "model_version": context["model_version"],
                    "action_policy_version": "action_policy_v1",
                },
            },
        }


@pytest.mark.asyncio
async def test_two_monthly_api_requests_have_no_ticker_as_of_or_run_cross_talk():
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")
    application.state.manager_agent = EchoMonthlyManager()
    application.state.supabase = MagicMock()
    telemetry_events = []
    application.state.telemetry_sink = telemetry_events.append

    def memory_for_request():
        memory = MagicMock()
        memory.long.get_all_preferences.return_value = {}
        return memory

    application.dependency_overrides[get_manager_memory] = memory_for_request
    payloads = [
        {
            "query": "Generate the first isolated monthly forecast",
            "ticker": "MSFT", "as_of": "2024-01-31T21:00:00Z",
            "mode": "HISTORICAL_BACKTEST", "horizon": 21,
            "run_id": "d180d30b-7f78-4c75-8e2b-52bca8466bf1",
        },
        {
            "query": "Generate the second isolated monthly forecast",
            "ticker": "AAPL", "as_of": "2024-02-29T21:00:00Z",
            "mode": "HISTORICAL_BACKTEST", "horizon": 21,
            "run_id": "95fb1c27-242c-4bd0-83af-41d0c39e7db2",
        },
    ]
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(*(
            client.post("/api/v1/analyze", json=payload) for payload in payloads
        ))
    assert [response.status_code for response in responses] == [200, 200]
    bodies = [response.json() for response in responses]
    assert [(body["forecast_context"]["ticker"], body["forecast_run_id"]) for body in bodies] == [
        ("MSFT", payloads[0]["run_id"]), ("AAPL", payloads[1]["run_id"]),
    ]
    assert bodies[0]["forecast_context"]["as_of"] != bodies[1]["forecast_context"]["as_of"]
    assert bodies[0]["monthly_forecast"]["evidence_ids"] == ["evidence-MSFT"]
    assert bodies[1]["monthly_forecast"]["evidence_ids"] == ["evidence-AAPL"]
    terminal_events = [
        event for event in telemetry_events
        if event["event"] == "stage10b.analysis.completed"
    ]
    assert {
        (event["run_id"], event["ticker"], event["as_of"])
        for event in terminal_events
    } == {
        (payloads[0]["run_id"], "MSFT", payloads[0]["as_of"]),
        (payloads[1]["run_id"], "AAPL", payloads[1]["as_of"]),
    }
    assert all(event["latency_ms"] >= 0 for event in terminal_events)
