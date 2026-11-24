"""Stage 10B run-correlated progress route tests."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from api.routes.progress import router, stream_progress
from core.progress_bus import get_queue, publish


@pytest.mark.asyncio
async def test_backtest_progress_carries_run_id_and_closes_on_terminal_event():
    run_id = "9ff36e62-cec5-4629-8175-1c80338db441"
    original_queue = get_queue(run_id)
    publish(
        run_id, "backtest_start", agent="backtest", message="BACKTEST_STARTED",
        detail={"stage": "10B"}, run_id=run_id,
    )
    publish(
        run_id, "backtest_complete", agent="backtest", message="BACKTEST_COMPLETED",
        detail={"stage": "10B"}, run_id=run_id,
    )
    request = MagicMock()
    request.is_disconnected = AsyncMock(return_value=False)
    response = await stream_progress(run_id, request)
    frames = []
    async for frame in response.body_iterator:
        frames.append(frame)
    assert len(frames) == 2
    payloads = [json.loads(frame.removeprefix("data: ").strip()) for frame in frames]
    assert [item["type"] for item in payloads] == ["backtest_start", "backtest_complete"]
    assert all(item["run_id"] == run_id for item in payloads)
    assert get_queue(run_id) is not original_queue


def test_backtest_routes_are_registered_through_existing_progress_router():
    application = FastAPI()
    application.include_router(router, prefix="/api/v1")
    paths = set(application.openapi()["paths"])
    assert "/api/v1/backtests" in paths
    assert "/api/v1/backtests/{run_id}" in paths
    assert "/api/v1/backtests/{run_id}/artifact" in paths
