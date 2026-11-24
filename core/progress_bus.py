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
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, AsyncIterator, TypedDict

_QUEUES: dict[str, asyncio.Queue] = {}
_EXPIRY_HANDLES: dict[str, asyncio.TimerHandle] = {}
_CLOSED: OrderedDict[str, None] = OrderedDict()
_MAX_SESSIONS = 512
_IDLE_TTL_SECONDS = 1800
_TERMINAL_GRACE_SECONDS = 30
_TERMINAL_EVENTS = {"pipeline_complete", "pipeline_error", "backtest_complete", "forecast_error"}


def _schedule_cleanup(session_id: str, delay: float) -> None:
    previous = _EXPIRY_HANDLES.pop(session_id, None)
    if previous is not None:
        previous.cancel()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # Synchronous callers are still bounded by _MAX_SESSIONS.
    _EXPIRY_HANDLES[session_id] = loop.call_later(delay, close_session, session_id)


_MAX_QUEUE_SIZE = 300  # backpressure guard so a stalled/absent SSE client can't leak memory


class ForecastProgressEvent(TypedDict):
    """Stable wire shape for opt-in monthly forecast progress events."""

    ts: float
    type: str
    run_id: str
    agent: str | None
    message: str
    detail: dict[str, Any]


FORECAST_PROGRESS_EVENT_TYPES = frozenset({
    "forecast_context",
    "feature_start",
    "feature_complete",
    "monthly_forecast_start",
    "monthly_forecast_complete",
    "backtest_start",
    "backtest_complete",
    "forecast_degraded",
    "forecast_error",
})


def _get_queue(session_id: str) -> asyncio.Queue:
    _CLOSED.pop(session_id, None)
    q = _QUEUES.get(session_id)
    if q is None:
        if len(_QUEUES) >= _MAX_SESSIONS:
            close_session(next(iter(_QUEUES)))
        q = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        _QUEUES[session_id] = q
    _schedule_cleanup(session_id, _IDLE_TTL_SECONDS)
    return q


def publish(
    session_id: str | None,
    event_type: str,
    agent: str | None = None,
    message: str = "",
    detail: dict[str, Any] | None = None,
    *,
    run_id: str | None = None,
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
        Optional structured payload. New forecast callers must use
        publish_forecast() so only compact allowlisted metadata is emitted.
    run_id : str | None
        Optional stable run identifier for versioned forecast events. Legacy
        callers may omit it; publish_forecast() requires and validates it.
    """
    if not session_id or session_id in _CLOSED:
        return
    q = _get_queue(session_id)
    event = {
        "ts":      time.time(),
        "type":    event_type,
        "agent":   agent,
        "message": message,
        "detail":  detail or {},
    }
    if run_id is not None:
        event["run_id"] = run_id
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        if event_type in _TERMINAL_EVENTS:
            q.get_nowait()
            q.put_nowait(event)
    if event_type in _TERMINAL_EVENTS:
        _schedule_cleanup(session_id, _TERMINAL_GRACE_SECONDS)


def publish_forecast(
    session_id: str | None,
    event_type: str,
    *,
    run_id: str,
    context: Mapping[str, Any] | object,
    stage: str,
    stage_revision: int,
    attempt: int,
    agent: str,
    message: str = "",
    started_at: float | None = None,
    evidence_count: int | None = None,
    degraded_flags: Sequence[str] | None = None,
) -> None:
    """Publish an allowlisted forecast event without raw State or evidence.

    This additive API leaves the legacy publish() path unchanged. It accepts a
    ForecastContext (or its compact mapping), derives only approved metadata,
    and records measured latency only when a monotonic start time is supplied.
    """
    if event_type not in FORECAST_PROGRESS_EVENT_TYPES:
        raise ValueError(f"unsupported forecast progress event: {event_type!r}")

    # Local import keeps progress_bus lightweight for existing legacy callers.
    from core.observability import elapsed_latency_ms, forecast_trace_metadata

    latency_ms = elapsed_latency_ms(started_at) if started_at is not None else None
    detail = forecast_trace_metadata(
        context,
        run_id=run_id,
        stage=stage,
        stage_revision=stage_revision,
        attempt=attempt,
        agent=agent,
        latency_ms=latency_ms,
        evidence_count=evidence_count,
        degraded_flags=degraded_flags,
    )
    event_run_id = detail.pop("run_id")
    publish(
        session_id,
        event_type,
        agent=agent,
        message=message,
        detail=dict(detail),
        run_id=event_run_id,
    )


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
    handle = _EXPIRY_HANDLES.pop(session_id, None)
    if handle is not None:
        handle.cancel()
    _CLOSED[session_id] = None
    _CLOSED.move_to_end(session_id)
    while len(_CLOSED) > _MAX_SESSIONS:
        _CLOSED.popitem(last=False)
    
def session_from_directives(directives: dict[str, Any] | None) -> str | None:
    """Extract the progress session_id directly from manager_directives."""
    if not directives:
        return None
    return directives.get("_progress_session_id")
