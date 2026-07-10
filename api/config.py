"""
api/config.py
-------------
Centralised settings for Alpha-Agent Node API.

All values are loaded from environment variables (or .env file).
Import the singleton `settings` anywhere in the codebase:

    from api.config import settings
    print(settings.SUPABASE_URL)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
from api.core.exceptions import ConfigurationError

log = logging.getLogger("api.config")


# ─────────────────────────────────────────────────────────────
# Settings model
# ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Application settings — loaded from env vars / .env file.

    Priority:
      1. Real environment variables
      2. .env file
      3. Hardcoded defaults below
    """

    # ── Anthropic ────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    MAX_ROUTING_LOOPS: int = 8

    # ── Supabase ─────────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = Field(
        default="", validation_alias="SUPABASE_SERVICE_ROLE_KEY"
    )

    # ── App ──────────────────────────────────────────────────
    APP_ENV: Literal["development", "production"] = "development"
    DEFAULT_USER_ID: str = "anonymous"
    REQUEST_TIMEOUT_S: int = 300    # max seconds an /analyze call may run

    # ── Logging ──────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: list[str] = []
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True,extra="ignore"
    )

    


# ─────────────────────────────────────────────────────────────
# Settings factory — lazy, cached, testable
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the single Settings instance, created on first call.

    lru_cache ensures Settings() is evaluated lazily (not at import time)
    and only once per process. In tests, call get_settings.cache_clear()
    after patching env vars so the next call picks up the new values::

        monkeypatch.setenv("APP_ENV", "production")
        get_settings.cache_clear()
        assert get_settings().APP_ENV == "production"
    """
    return Settings()


# Backward-compatible alias — callers using `from api.config import settings`
# continue to work without change. New code should prefer get_settings().
settings: Settings = get_settings()


# ─────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────

def validate_settings() -> None:
    """
    Assert that required environment variables are set.
    Called once during FastAPI lifespan startup.

    Raises ConfigurationError (instead of calling sys.exit directly) so
    that tests can assert on the exception without killing the process.
    The lifespan in api/main.py catches this and calls sys.exit(1).
    """
    s = get_settings()
    required = {
        "ANTHROPIC_API_KEY": s.ANTHROPIC_API_KEY,
        "SUPABASE_URL":      s.SUPABASE_URL,
        "SUPABASE_KEY":      s.SUPABASE_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)} — set them in .env"
        )

    
    log.info(f"Settings validated — env={s.APP_ENV} model={s.ANTHROPIC_MODEL}")