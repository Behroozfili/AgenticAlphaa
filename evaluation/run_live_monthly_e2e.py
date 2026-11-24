"""Run one real MSFT monthly request through API -> all agents -> Manager."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv


async def run(*, ticker: str, as_of: str, query: str) -> dict:
    load_dotenv()
    from api.config import Settings
    from api.main import create_app

    application = create_app(runtime_settings=Settings())
    payload = {
        "query": query,
        "ticker": ticker.upper(),
        "as_of": as_of,
        "mode": "LIVE",
        "cutoff_policy": "MONTH_TO_DATE",
        "horizon": 21,
        "days_back": 30,
        "search_depth": "advanced",
        "include_sentiment": True,
        "run_id": str(uuid.uuid4()),
    }
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://agenticalpha.local", timeout=900.0,
        ) as client:
            response = await client.post("/api/v1/analyze", json=payload)
    response.raise_for_status()
    body = response.json()
    forecast = body.get("monthly_forecast") or {}
    result = {
        "http_status": response.status_code,
        "execution_status": body.get("status"),
        "publication_status": body.get("publication_status"),
        "run_id": body.get("forecast_run_id"),
        "ticker": forecast.get("ticker"),
        "as_of": forecast.get("as_of"),
        "target_date": forecast.get("target_date"),
        "forecast_status": forecast.get("status"),
        "quality_gate_passed": forecast.get("quality_gate_passed"),
        "quality_gate_failures": forecast.get("quality_gate_failures"),
        "action": forecast.get("action"),
        "probability_positive": forecast.get("probability_positive"),
        "expected_return_21d": forecast.get("expected_return_21d"),
        "expected_excess_return": forecast.get("expected_excess_return"),
        "model_variant": forecast.get("model_variant"),
        "graph_coverage": forecast.get("graph_coverage"),
        "evidence_count": len(forecast.get("evidence_ids") or []),
        "warnings": forecast.get("warnings"),
        "specialist_contributions": forecast.get("specialist_contributions"),
        "decision_drivers": forecast.get("decision_drivers"),
        "narrative": forecast.get("narrative"),
        "agent_execution_count": len(body.get("agent_execution_history") or []),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--as-of", default="2026-09-21T20:00:00Z")
    parser.add_argument(
        "--query",
        default="Produce a coordinated 21-trading-day monthly forecast using research, RAG, graph, finance, and sentiment.",
    )
    args = parser.parse_args()
    asyncio.run(run(ticker=args.ticker, as_of=args.as_of, query=args.query))


if __name__ == "__main__":
    main()
