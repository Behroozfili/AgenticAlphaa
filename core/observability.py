"""
core/observability.py

"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("core.observability")
import sentry_sdk
from langsmith import traceable


_sentry_enabled: bool = False
_langsmith_enabled: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#                                        Sentry
# ─────────────────────────────────────────────────────────────────────────────

def init_sentry(app_env: str | None = None) -> bool:
    
    global _sentry_enabled

    if _sentry_enabled:
        return True

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        log.info("Sentry disabled — SENTRY_DSN is not set.")
        return False

    
    env = app_env or os.environ.get("APP_ENV", "development")

    try:
        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=1.0,
            environment=env,
            send_default_pii=False,
        )
        _sentry_enabled = True
        log.info("Sentry enabled — environment=%s", env)
        return True

    except ImportError:
        log.warning(
            "Sentry disabled — sentry-sdk package is not installed. "
            "Add sentry-sdk>=2.0.0 to requirements.txt."
        )
        return False

    except Exception as exc: 
        log.warning("Sentry disabled — initialisation failed: %s", exc)
        return False


def sentry_enabled() -> bool:
    """Return True if Sentry was successfully initialised in this process."""
    return _sentry_enabled


# ─────────────────────────────────────────────────────────────────────────────
#                                        langsmith
# ────────────────────────────────────────────────────────────────


def init_langsmith() -> bool:
   
    global _langsmith_enabled

    if _langsmith_enabled:
        return True

    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        log.info("LangSmith tracing disabled — LANGSMITH_API_KEY is not set.")
        return False

    try:
        project = os.environ.get("LANGSMITH_PROJECT", "alpha-agent-node")

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project

        _langsmith_enabled = True
        log.info("LangSmith tracing enabled — project=%s", project)
        return True

    except Exception as exc:
        log.warning("LangSmith tracing disabled — setup failed: %s", exc)
        return False


def langsmith_enabled() -> bool:
   
    return _langsmith_enabled

# ─────────────────────────────────────────────────────────────────────────────
#                            MCP tool-call tracing
# ─────────────────────────────────────────────────────────────────────────────

def traced_call_tool(session, tool_name, arguments):
    """
    Record a single MCP tool call as an independent LangSmith span
    with run_type="tool".

    A raw session.call_tool is not traced by LangSmith: it runs inside the
    mcp library layer and creates no span. This wrapper runs inside the agent
    process (which has an active trace context), so the span is correctly
    attached as a child of the parent node.

    The span name is set to the actual tool name (e.g. tool_get_cik) so the
    validation script can locate it in actual_tool_calls.

    arguments is passed positionally so this works with both call patterns
    used across the agents (positional and keyword arguments=).
    """

    @traceable(run_type="tool", name=tool_name)
    async def _inner():
        return await session.call_tool(tool_name, arguments)

    return _inner()