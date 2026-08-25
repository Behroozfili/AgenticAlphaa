#api/config.py

from __future__ import annotations
import logging
from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
from api.core.exceptions import ConfigurationError
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("api.config")


class Settings(BaseSettings):
 
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    MAX_ROUTING_LOOPS: int = 8
    SUPABASE_URL: str = ""
    NEO4J_URI : str = ""
    NEO4J_USER : str = ""
    NEO4J_PASSWORD : str = ""
    SENTRY_DSN : str = ""
    LANGSMITH_API_KEY : str = ""
    
    SUPABASE_KEY: str = Field(
        default="", validation_alias="SUPABASE_SERVICE_ROLE_KEY"
    )

    
    APP_ENV: Literal["development", "production"] = "development"
    DEFAULT_USER_ID: str = "anonymous"
    REQUEST_TIMEOUT_S: int = 300    
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: list[str] = []
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True,extra="ignore"
    )



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings: Settings = get_settings()



def validate_settings() -> None:
    s = get_settings()
    required = {
        "ANTHROPIC_API_KEY": s.ANTHROPIC_API_KEY,
        "SUPABASE_URL":      s.SUPABASE_URL,
        "SUPABASE_KEY":      s.SUPABASE_KEY,
        "NEO4J_URI":         s.NEO4J_URI,
        "NEO4J_USER":        s.NEO4J_USER,
        "NEO4J_PASSWORD":    s.NEO4J_PASSWORD,  
        "SENTRY_DSN":        s.SENTRY_DSN,
        "LANGSMITH_API_KEY": s.LANGSMITH_API_KEY
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {', '.join(missing)} — set them in .env"
        )

    
    log.info(f"Settings validated — env={s.APP_ENV} model={s.ANTHROPIC_MODEL}")