"""
core/progress_bus.py
---------------------
Lightweight in-process pub/sub used to stream live pipeline-progress events
(Manager -> specialist agent -> tool call -> back to Manager, etc.) to the
frontend via Server-Sent Events, without the orchestration graph needing
any reference to the actual HTTP connection.

One asyncio.Queue per active analysis session, keyed by session_id. Any
node/agent anywhere in the call stack calls publish(session_id, ...); the
SSE route (api/routes/progress.py) subscribes to the same session_id's
queue and forwards whatever arrives to the client.

Scaling note: this lives entirely in the API process's memory. If
Alpha-Agent Node is ever run across multiple worker processes, this must
be swapped for a real broker (Redis pub/sub, etc.) -- the publish()/
subscribe() call sites elsewhere would not need to change.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

_QUEUES: dict[str, asyncio.Queue] = {}
_MAX_QUEUE_SIZE = 300  # backpressure guard so a stalled/absent SSE client can't leak memory


def _get_queue(session_id: str) -> asyncio.Queue:
    q = _QUEUES.get(session_id)
    if q is None:
        q = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        _QUEUES[session_id] = q
    return q


def publish(
    session_id: str | None,
    event_type: str,
    agent: str | None = None,
    message: str = "",
    detail: dict[str, Any] | None = None,
) -> None:
    """
    Fire-and-forget: push a progress event onto session_id's queue.

    Silently no-ops if session_id is falsy (e.g. a code path invoked
    outside a live-tracked request, such as a CLI run or a unit test) so
    instrumenting the orchestration graph with publish() calls never
    requires every caller to guard against "no active SSE session".

    Uses put_nowait so a slow/absent consumer can never stall the actual
    analysis pipeline; if the queue is full, the event is dropped.

    Parameters
    ----------
    session_id : str | None
        The analysis session this event belongs to.
    event_type : str
        One of: "pipeline_start", "hydrate", "route", "dispatch_start",
        "dispatch_end", "agent_brain", "agent_tool_call",
        "agent_tool_result", "agent_checker", "evaluate", "finalise",
        "pipeline_complete", "pipeline_error".
    agent : str | None
        "manager" | "research" | "financial" | "sentiment" | None.
    message : str
        Short human-readable description for display.
    detail : dict | None
        Optional structured payload (tool name, score, reasoning, etc).
    """
    if not session_id:
        return
    q = _get_queue(session_id)
    event = {
        "ts":      time.time(),
        "type":    event_type,
        "agent":   agent,
        "message": message,
        "detail":  detail or {},
    }
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass


async def subscribe(session_id: str) -> AsyncIterator[dict[str, Any]]:
    """
    Async-iterate events for session_id as they're published.

    NOTE: do not wrap this generator's __anext__() in asyncio.wait_for() —
    when the timeout cancels the underlying await, the cancellation can
    surface as an invalid StopAsyncIteration escaping the generator (a
    known CPython/asyncio interaction with PEP 479), which asyncio then
    reports as "RuntimeError: async generator raised StopAsyncIteration".
    Callers that need a timeout (e.g. to emit periodic SSE heartbeats)
    should use get_queue(session_id) and call asyncio.wait_for(queue.get(),
    ...) directly instead — see api/routes/progress.py.
    """
    q = _get_queue(session_id)
    while True:
        event = await q.get()
        yield event


def get_queue(session_id: str) -> asyncio.Queue:
    """
    Public accessor for session_id's queue, for callers (like the SSE
    route) that need to apply their own timeout/cancellation handling via
    asyncio.wait_for(queue.get(), ...) rather than iterating subscribe().
    """
    return _get_queue(session_id)


def session_from_shared(shared_manager_ref: dict[str, Any] | None) -> str | None:
    """
    Convenience helper for specialist agents: extract the progress
    session_id that ManagerAgent.run() stashed into manager_directives,
    from the shared_manager_ref every agent-private state already carries.
    Returns None (and publish() then silently no-ops) if absent, e.g. when
    an agent is invoked directly outside the Manager's graph (tests, CLI).
    """
    if not shared_manager_ref:
        return None
    return shared_manager_ref.get("manager_directives", {}).get("_progress_session_id")


def close_session(session_id: str) -> None:
    """Drop the queue for a finished session so memory doesn't grow unbounded."""
    _QUEUES.pop(session_id, None)