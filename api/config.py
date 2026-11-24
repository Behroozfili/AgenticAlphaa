"""Versioned, side-effect-free settings for the API runtime."""

from __future__ import annotations

import logging
from datetime import datetime
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from api.core.exceptions import ConfigurationError
from core.forecast_contract import (
    AnalysisMode, ForecastContext, ForecastCutoffPolicy, HORIZON_TRADING_DAYS,
)

log = logging.getLogger("api.config")


class Settings(BaseSettings):
    """Environment-backed configuration with explicit runtime versions.

    Observability credentials are optional and do not enable an exporter by
    themselves. Local development, tests, and offline backtests therefore stay
    free of external telemetry unless both a flag and credentials are supplied.
    """

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"
    MAX_ROUTING_LOOPS: int = 8

    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = Field(
        default="",
        validation_alias=AliasChoices("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY"),
    )
    NEO4J_URI: str = ""
    NEO4J_USER: str = ""
    NEO4J_PASSWORD: str = ""

    APP_ENV: Literal["development", "test", "production"] = "development"
    DEFAULT_USER_ID: str = "anonymous"
    AUTH_REQUIRED: bool = False
    SUPABASE_JWT_SECRET: str = ""
    JWT_AUDIENCE: str = "authenticated"
    JWT_ISSUER: str = ""
    RAG_VECTOR_SCHEMA: Literal["legacy", "v2"] = "legacy"
    HF_TOKEN: str = ""
    REQUEST_TIMEOUT_S: int = 300
    LOG_LEVEL: str = "INFO"
    ALLOWED_ORIGINS: list[str] = []

    RUNTIME_VERSION: str = "stage10a-runtime-v1"
    FORECAST_SCHEMA_VERSION: str = "forecast-schema-v2"
    FORECAST_FEATURE_VERSION: str = "monthly-features-v1"
    FORECAST_MODEL_VERSION: str = "monthly-model-v1"
    LIVE_DATASET_VERSION: str = "live-dataset-v1"
    HISTORICAL_DATASET_VERSION: str = "historical-dataset-v1"
    BACKTEST_CONTRACT_VERSION: str = "monthly-backtest-v1"
    FORECAST_HORIZON_TRADING_DAYS: int = HORIZON_TRADING_DAYS
    FORECAST_LOOKBACK_DAYS: int = 365
    FORECAST_BENCHMARK: str = "SPY"
    FORECAST_CUTOFF_POLICY: Literal["MONTH_END", "MONTH_TO_DATE"] = "MONTH_TO_DATE"

    QUANT_ARTIFACT_PATH: str = ""
    MARKET_DATA_CACHE_DIR: str = ".cache/market-data"
    MARKET_SECTOR_BENCHMARK: str = ""
    QUANT_RUNTIME_REQUIRED: bool = False

    OBSERVABILITY_ENABLED: bool = False
    SENTRY_ENABLED: bool = False
    LANGSMITH_ENABLED: bool = False
    SENTRY_DSN: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "alpha-agent-node"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator(
        "RUNTIME_VERSION",
        "FORECAST_SCHEMA_VERSION",
        "FORECAST_FEATURE_VERSION",
        "FORECAST_MODEL_VERSION",
        "LIVE_DATASET_VERSION",
        "HISTORICAL_DATASET_VERSION",
        "BACKTEST_CONTRACT_VERSION",
        "FORECAST_BENCHMARK",
        "MARKET_DATA_CACHE_DIR",
    )
    @classmethod
    def _non_empty_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("runtime version fields must not be empty")
        return value

    @field_validator("FORECAST_HORIZON_TRADING_DAYS")
    @classmethod
    def _fixed_monthly_horizon(cls, value: int) -> int:
        if value != HORIZON_TRADING_DAYS:
            raise ValueError("forecast horizon must be exactly 21 trading sessions")
        return value

    @field_validator("FORECAST_LOOKBACK_DAYS", "REQUEST_TIMEOUT_S", "MAX_ROUTING_LOOPS")
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if isinstance(value, bool) or value <= 0:
            raise ValueError("value must be a positive integer")
        return value

    @property
    def sentry_active(self) -> bool:
        return bool(self.OBSERVABILITY_ENABLED and self.SENTRY_ENABLED and self.SENTRY_DSN.strip())

    @property
    def langsmith_active(self) -> bool:
        return bool(
            self.OBSERVABILITY_ENABLED
            and self.LANGSMITH_ENABLED
            and self.LANGSMITH_API_KEY.strip()
        )

    def dataset_version_for(self, mode: AnalysisMode | str) -> str:
        parsed_mode = AnalysisMode(mode)
        if parsed_mode is AnalysisMode.HISTORICAL_BACKTEST:
            return self.HISTORICAL_DATASET_VERSION
        return self.LIVE_DATASET_VERSION

    def forecast_context(
        self,
        *,
        ticker: str,
        as_of: datetime | str,
        analysis_mode: AnalysisMode | str = AnalysisMode.LIVE,
        cutoff_policy: ForecastCutoffPolicy | str | None = None,
    ) -> ForecastContext:
        """Build the canonical Stage-01 context without duplicating calendar logic."""
        mode = AnalysisMode(analysis_mode)
        policy = (
            ForecastCutoffPolicy.MONTH_END
            if cutoff_policy is None and mode is AnalysisMode.HISTORICAL_BACKTEST
            else ForecastCutoffPolicy(cutoff_policy or self.FORECAST_CUTOFF_POLICY)
        )
        return ForecastContext(
            ticker=ticker,
            as_of=as_of,
            horizon_trading_days=self.FORECAST_HORIZON_TRADING_DAYS,
            lookback_days=self.FORECAST_LOOKBACK_DAYS,
            benchmark=self.FORECAST_BENCHMARK,
            analysis_mode=mode,
            cutoff_policy=policy,
            schema_version=self.FORECAST_SCHEMA_VERSION,
            dataset_version=self.dataset_version_for(mode),
            feature_version=self.FORECAST_FEATURE_VERSION,
            model_version=self.FORECAST_MODEL_VERSION,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Backward-compatible import for existing modules. New runtime code should call
# get_settings() at the lifecycle/request boundary so test overrides are fresh.
settings: Settings = get_settings()


def validate_settings(candidate: Settings | None = None) -> None:
    """Validate required service settings; telemetry remains optional."""
    current = candidate or get_settings()
    required = {
        "ANTHROPIC_API_KEY": current.ANTHROPIC_API_KEY,
        "SUPABASE_URL": current.SUPABASE_URL,
        "SUPABASE_KEY": current.SUPABASE_KEY,
        "NEO4J_URI": current.NEO4J_URI,
        "NEO4J_USER": current.NEO4J_USER,
        "NEO4J_PASSWORD": current.NEO4J_PASSWORD,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if current.AUTH_REQUIRED and not current.SUPABASE_JWT_SECRET.strip():
        missing.append("SUPABASE_JWT_SECRET")
    if current.APP_ENV == "production":
        if not current.AUTH_REQUIRED:
            missing.append("AUTH_REQUIRED=true")
        if not current.QUANT_RUNTIME_REQUIRED or not current.QUANT_ARTIFACT_PATH.strip():
            missing.append("QUANT_RUNTIME_REQUIRED/QUANT_ARTIFACT_PATH")
        if current.RAG_VECTOR_SCHEMA != "v2":
            missing.append("RAG_VECTOR_SCHEMA=v2")
    if missing:
        raise ConfigurationError(
            f"Missing required runtime settings: {', '.join(missing)}",
        )
    if not current.HF_TOKEN.strip():
        log.warning("huggingface_token_missing code=HF_TOKEN_MISSING")
    log.info("Settings validated: env=%s runtime=%s", current.APP_ENV, current.RUNTIME_VERSION)
