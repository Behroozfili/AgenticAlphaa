"""
core/observability.py

"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("core.observability")
import sentry_sdk 


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