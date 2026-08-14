"""
core/error_handler.py
"""

from __future__ import annotations
import asyncio

import contextlib 
import functools
import inspect
import logging
from typing import Any, AsyncIterator, Callable, Iterator, TypeVar

import sentry_sdk
from core.observability import sentry_enabled

log = logging.getLogger("core.error_handler")

T = TypeVar("T")

_SCALARS = (str, int, float, bool, type(None))
_MAX_ITEMS = 10
_MAX_LEN = 100


def add_breadcrumb(component: str, message: str, data: dict | None = None) -> None:
    """Record an info-level breadcrumb; no-op when Sentry is disabled."""
    if not sentry_enabled():
        return
    try:
        sentry_sdk.add_breadcrumb(
            category=component,
            message=message,
            data=data or {},
            level="info",
        )
    except Exception:
        pass


def capture_exception(
    component: str,
    exc: BaseException,
    *,
    tags: dict[str, str] | None = None,
) -> None:
    """Send an exception to Sentry tagged with component (+ optional extra tags)."""
    if not sentry_enabled():
        return
    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("component", component)
            for k, v in (tags or {}).items():
                scope.set_tag(k, v)
            sentry_sdk.capture_exception(exc)
    except Exception:
        pass


def capture_message(
    component: str,
    message: str,
    *,
    level: str = "warning",
    tags: dict[str, str] | None = None,
) -> None:
    """Send a plain text message to Sentry tagged with component; no-op when disabled."""
    if not sentry_enabled():
        return
    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("component", component)
            for k, v in (tags or {}).items():
                scope.set_tag(k, v)
            sentry_sdk.capture_message(message, level=level)
    except Exception:
        pass


_SCALARS = (str, int, float, bool, type(None))
_MAX_ITEMS = 10
_MAX_LEN = 100


def _safe_extra(kwargs: dict) -> dict:
    safe = {}
    for k, v in kwargs.items():
        try:
            if isinstance(v, _SCALARS):
                safe[k] = v
            elif isinstance(v, (list, tuple, set)):
                items = [str(item)[:_MAX_LEN] for item in list(v)[:_MAX_ITEMS]]
                if len(v) > _MAX_ITEMS:
                    items.append(f"...(+{len(v) - _MAX_ITEMS} more)")
                safe[k] = items
            elif isinstance(v, dict):
                out = {}
                for i, (dk, dv) in enumerate(v.items()):
                    if i >= _MAX_ITEMS:
                        out["..."] = f"(+{len(v) - _MAX_ITEMS} more)"
                        break
                    out[str(dk)] = str(dv)[:_MAX_LEN]
                safe[k] = out
            else:
                safe[k] = f"<{type(v).__name__}: {str(v)[:_MAX_LEN]}>"
        except Exception:
            safe[k] = f"<{type(v).__name__} (unstringifiable)>"
    return safe


def with_error_reporting(component: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
  
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(fn)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                add_breadcrumb(component, fn.__name__, _safe_extra(bound.arguments))
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    log.debug("component=%s fn=%s raised: %s", component, fn.__name__, exc)
                    capture_exception(component, exc)
                    raise
            return async_wrapper  

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            add_breadcrumb(component, fn.__name__, _safe_extra(bound.arguments))
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                log.debug("component=%s fn=%s raised: %s", component, fn.__name__, exc)
                capture_exception(component, exc)
                raise
        return sync_wrapper  

    return decorator


@contextlib.contextmanager
def _sync_context(component: str) -> Iterator[None]:
    """
    Sync context-manager variant. Reports and re-raises.

    Usage::

        with with_error_reporting.context("rag.vector_search"):
            results = vector_store.search(query)
    """
    add_breadcrumb(component, "<block>")
    try:
        yield
    except Exception as exc:
        log.debug("component=%s <block> raised: %s", component, exc)
        capture_exception(component, exc)
        raise


@contextlib.asynccontextmanager
async def _async_context(component: str) -> AsyncIterator[None]:
    
    add_breadcrumb(component, "<block>")
    try:
        yield
    except Exception as exc:
        log.debug("component=%s <block> raised: %s", component, exc)
        capture_exception(component, exc)
        raise


with_error_reporting.context = _sync_context          
with_error_reporting.async_context = _async_context