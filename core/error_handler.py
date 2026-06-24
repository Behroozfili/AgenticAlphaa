"""
core/error_handler.py
---------------------
Reusable decorator / context manager that adds Sentry breadcrumbs and
captures exceptions without altering the wrapped function's control flow.

Both async and sync functions are supported.

Usage
-----
    from core.error_handler import with_error_reporting

    # as a decorator
    @with_error_reporting(component="research.executor")
    async def _executor_node(self, state):
        ...

    # or inline for a sync block
    with with_error_reporting.context(component="rag.vector_search"):
        results = vector_store.search(query)

    # or inline for an async block
    async with with_error_reporting.async_context(component="rag.vector_search"):
        results = await vector_store.search(query)
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import inspect
from typing import Any, AsyncIterator, Callable, Iterator

from core.observability import sentry_enabled

log = logging.getLogger("core.error_handler")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_breadcrumb(component: str, function_name: str, extra: dict | None = None) -> None:
    """Add a Sentry breadcrumb if Sentry is enabled; silently no-op otherwise."""
    if not sentry_enabled():
        return
    try:
        import sentry_sdk  # noqa: PLC0415
        sentry_sdk.add_breadcrumb(
            category=component,
            message=function_name,
            data=extra or {},
            level="info",
        )
    except Exception:
        pass  # never let observability code crash the caller


def _capture(component: str, exc: BaseException) -> None:
    """Send exception to Sentry tagged with component; silently no-op otherwise."""
    if not sentry_enabled():
        return
    try:
        import sentry_sdk  # noqa: PLC0415
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("component", component)
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def _safe_extra(kwargs: dict) -> dict:
    """
    Build a breadcrumb data dict from call kwargs, keeping only
    JSON-serialisable scalar values to avoid Sentry payload issues.
    """
    safe: dict = {}
    for k, v in kwargs.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# Public decorator
# ─────────────────────────────────────────────────────────────────────────────

def with_error_reporting(component: str) -> Callable:
    """
    Decorator factory that wraps async **or** sync callables.

    On entry  → adds a Sentry breadcrumb (category=component, message=fn name).
    On exception → captures to Sentry tagged with component, then re-raises.

    The decorator never suppresses exceptions — existing control flow
    (including manager_agent._dispatch()'s deliberate swallowing) is unchanged.

    Args:
        component: A dot-namespaced string identifying the subsystem,
                   e.g. "research.executor", "rag.graph_traverse",
                   "financial.brain", "sentiment.pipeline".
    """
    def decorator(fn: Callable) -> Callable:

        if  inspect.iscoroutinefunction(fn): 
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _add_breadcrumb(component, fn.__name__, _safe_extra(kwargs))
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    log.debug(
                        "Capturing exception in component=%s fn=%s: %s",
                        component, fn.__name__, exc,
                    )
                    _capture(component, exc)
                    raise
            return async_wrapper

        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _add_breadcrumb(component, fn.__name__, _safe_extra(kwargs))
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    log.debug(
                        "Capturing exception in component=%s fn=%s: %s",
                        component, fn.__name__, exc,
                    )
                    _capture(component, exc)
                    raise
            return sync_wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Inline context managers  (with_error_reporting.context / .async_context)
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _sync_context(component: str) -> Iterator[None]:
    """
    Sync context manager variant.

    Usage::

        with with_error_reporting.context(component="rag.vector_search"):
            results = vector_store.search(query)
    """
    _add_breadcrumb(component, "<block>")
    try:
        yield
    except Exception as exc:
        log.debug("Capturing exception in component=%s <block>: %s", component, exc)
        _capture(component, exc)
        raise


@contextlib.asynccontextmanager
async def _async_context(component: str) -> AsyncIterator[None]:
    """
    Async context manager variant.

    Usage::

        async with with_error_reporting.async_context(component="rag.vector_search"):
            results = await vector_store.search(query)
    """
    _add_breadcrumb(component, "<block>")
    try:
        yield
    except Exception as exc:
        log.debug("Capturing exception in component=%s <block>: %s", component, exc)
        _capture(component, exc)
        raise


# Attach as static attributes so callers can use with_error_reporting.context(...)
with_error_reporting.context = _sync_context          # type: ignore[attr-defined]
with_error_reporting.async_context = _async_context   # type: ignore[attr-defined]
