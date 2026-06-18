"""
core/observability.py
---------------------
Centralised bootstrap for Sentry (error tracking) and LangSmith (LLM tracing).

Both integrations are **optional**: if the relevant env var is absent the
function logs once at INFO level and returns False — nothing else in the
codebase needs to branch on this; every Sentry / LangSmith call already
degrades gracefully when the SDK was never initialised.

Usage
-----
    from core.observability import init_sentry, init_langsmith, langsmith_enabled

    # call once per process (api/main.py lifespan, each MCP server __main__)
    init_sentry()
    init_langsmith()

    # optional guard before importing @traceable in subprocess-launched servers
    if langsmith_enabled():
        from langsmith import traceable
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("core.observability")

# Module-level flags set by the init functions so callers can query state
# without re-reading env vars.
_sentry_enabled: bool = False
_langsmith_enabled: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Sentry
# ─────────────────────────────────────────────────────────────────────────────

def init_sentry(app_env: str | None = None) -> bool:
    """
    Initialise Sentry SDK if SENTRY_DSN is set in the environment.

    Idempotent — safe to call multiple times (subsequent calls are no-ops
    because the module-level flag is checked first).

    Args:
        app_env: The application environment string (e.g. "production",
                 "development"). If None, falls back to the APP_ENV env var,
                 then to "development". Pass this explicitly from the entry
                 point (api/main.py lifespan, MCP server __main__) so that
                 core/ has zero dependency on api/.

    Returns True if Sentry was successfully initialised, False otherwise.
    """
    global _sentry_enabled

    if _sentry_enabled:
        return True

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        log.info("Sentry disabled — SENTRY_DSN is not set.")
        return False

    # Resolve environment — no import from api/ needed
    env = app_env or os.environ.get("APP_ENV", "development")

    try:
        import sentry_sdk  # noqa: PLC0415  (local import keeps it optional)

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=1.0,
            environment=env,
            # Attach request data automatically where available
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

    except Exception as exc:  # malformed DSN, network error during handshake …
        log.warning("Sentry disabled — initialisation failed: %s", exc)
        return False


def sentry_enabled() -> bool:
    """Return True if Sentry was successfully initialised in this process."""
    return _sentry_enabled


# ─────────────────────────────────────────────────────────────────────────────
# LangSmith
# ─────────────────────────────────────────────────────────────────────────────

def init_langsmith() -> bool:
    """
    Enable LangSmith tracing if LANGSMITH_API_KEY is set in the environment.

    LangSmith's Python SDK activates automatically when the env vars
    LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY are present; this function
    sets them so the rest of the codebase only needs to apply @traceable.

    Idempotent — safe to call multiple times.

    Returns True if LangSmith tracing was enabled, False otherwise.
    """
    global _langsmith_enabled

    if _langsmith_enabled:
        return True

    api_key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        log.info("LangSmith tracing disabled — LANGSMITH_API_KEY is not set.")
        return False

    try:
        project = os.environ.get("LANGSMITH_PROJECT", "alpha-agent-node")

        # LangSmith SDK reads these standard env vars automatically.
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
    """
    Return True if LangSmith tracing was successfully enabled in this process.

    MCP server scripts that are launched as subprocesses can call this before
    importing @traceable to avoid an unnecessary import when no API key is set.
    """
    return _langsmith_enabled