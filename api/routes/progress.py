"""
api/routes/progress.py
-----------------------
Server-Sent Events (SSE) endpoint that streams live pipeline-progress
events for a given analysis session_id, produced by core/progress_bus.py.

Implemented with a plain StreamingResponse (text/event-stream) so no new
dependency (e.g. sse_starlette) is required beyond FastAPI/Starlette,
which the project already uses.

The client must open this connection BEFORE (or at the same instant as)
firing the POST /api/v1/analyze request that carries the same session_id,
so no early events are missed. See frontend/alpha-agent-app.html for the
client-side wiring: it opens the EventSource first, then POSTs.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from core.progress_bus import get_queue, close_session

log = logging.getLogger("api.routes.progress")

router = APIRouter()

_HEARTBEAT_INTERVAL_S = 15  # keeps proxies/load balancers from closing an idle connection


@router.get("/analyze/stream/{session_id}")
async def stream_progress(session_id: str, request: Request):
    """
    Stream progress events for `session_id` as Server-Sent Events.

    Each event from core/progress_bus is forwarded as a standard SSE
    frame: ``data: <json>\\n\\n``. The stream closes itself once a
    "pipeline_complete" or "pipeline_error" event is seen, or when the
    client disconnects. A periodic comment-only heartbeat keeps
    intermediary proxies from timing out the connection during long gaps
    between events (e.g. while a specialist agent's MCP tool call is
    still in flight).
    """

    async def event_generator():
        queue = get_queue(session_id)
        try:
            while True:
                if await request.is_disconnected():
                    log.info("SSE client disconnected — session=%s", session_id)
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_INTERVAL_S)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] in ("pipeline_complete", "pipeline_error"):
                    break
        except asyncio.CancelledError:
            log.info("SSE stream cancelled — session=%s", session_id)
            raise
        finally:
            close_session(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )