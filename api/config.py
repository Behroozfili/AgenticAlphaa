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

import os
import sys
import logging
from typing import Literal

from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field

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
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
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
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

    


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

settings = Settings()


# ─────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────

def validate_settings() -> None:
    """
    Assert that required environment variables are set.
    Called once during FastAPI lifespan startup.
    Exits the process with a clear message if anything is missing.
    """
    required = {
        "ANTHROPIC_API_KEY": settings.ANTHROPIC_API_KEY,
        "SUPABASE_URL":      settings.SUPABASE_URL,
        "SUPABASE_KEY":      settings.SUPABASE_KEY,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log.critical(
            "Missing required environment variables: %s — set them in .env",
            ", ".join(missing),
        )
        sys.exit(1)

    log.info(
        "Settings validated — env=%s model=%s",
        settings.APP_ENV,
        settings.ANTHROPIC_MODEL,
    )
