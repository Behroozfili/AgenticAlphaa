"""FastAPI application and request-safe Stage 10A runtime lifecycle."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from neo4j import GraphDatabase
from supabase import create_client

from api.auth import jwt_auth_middleware
from api.config import Settings, get_settings, settings, validate_settings
from api.core.exceptions import (
    AlphaAgentError,
    CORE_EXCEPTION_TYPES,
    ConfigurationError,
    map_core_exception,
)
from api.dependencies import ManagerAgentFactory
from api.routes.analyze import router as analyze_router
from api.routes.progress import router as progress_router
from core.observability import init_langsmith, init_sentry
from langsmith_trace import emit_stage10a_event, new_stage10a_run_id

log = logging.getLogger("api.main")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SENSITIVE_DEPENDENCY_LOGGERS = (
    "httpcore", "httpx", "hpack", "urllib3", "supabase", "postgrest",
    "anthropic", "yfinance", "peewee", "neo4j", "langsmith",
)


def _harden_dependency_logging() -> None:
    """Prevent HTTP dependencies from logging authentication headers."""
    for logger_name in _SENSITIVE_DEPENDENCY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _runtime_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _configure_quant_runtime(runtime: Settings):
    """Load one validated artifact and make it the runtime version authority."""
    configured_path = runtime.QUANT_ARTIFACT_PATH.strip()
    if not configured_path:
        if runtime.QUANT_RUNTIME_REQUIRED:
            raise ConfigurationError("QUANT_ARTIFACT_PATH is required")
        return runtime, None, None

    from tools.market_tools.market_provider import YahooMarketDataProvider
    from tools.market_tools.quant_artifacts import load_quant_artifacts

    artifacts = load_quant_artifacts(_runtime_path(configured_path))
    resolved = runtime.model_copy(update={
        "LIVE_DATASET_VERSION": artifacts.model.dataset_version,
        "HISTORICAL_DATASET_VERSION": artifacts.model.dataset_version,
        "FORECAST_FEATURE_VERSION": artifacts.model.feature_version,
        "FORECAST_MODEL_VERSION": artifacts.model.model_version,
    })
    provider = YahooMarketDataProvider(
        cache_dir=_runtime_path(runtime.MARKET_DATA_CACHE_DIR),
        sector_benchmark=runtime.MARKET_SECTOR_BENCHMARK.strip() or None,
    )
    return resolved, artifacts.service(), provider


def _configure_optional_observability(runtime: Settings) -> tuple[bool, bool]:
    """Configure explicit exporters with outbound payload filtering."""
    sentry_ok = False
    langsmith_ok = False
    try:
        sentry_ok = init_sentry(
            app_env=runtime.APP_ENV,
            observability_enabled=runtime.OBSERVABILITY_ENABLED,
            exporter_enabled=runtime.SENTRY_ENABLED,
            dsn=runtime.SENTRY_DSN,
        ) is True
    except Exception:
        log.warning("observability_initializer_failed code=SENTRY_INIT_FAILED")
    try:
        langsmith_ok = init_langsmith(
            observability_enabled=runtime.OBSERVABILITY_ENABLED,
            exporter_enabled=runtime.LANGSMITH_ENABLED,
            api_key=runtime.LANGSMITH_API_KEY,
            project=runtime.LANGSMITH_PROJECT,
        ) is True
    except Exception:
        log.warning("observability_initializer_failed code=LANGSMITH_INIT_FAILED")
    return sentry_ok, langsmith_ok


def _telemetry_sink(application: FastAPI):
    sink = getattr(application.state, "telemetry_sink", None)
    return sink if callable(sink) else None


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Own connection clients for one application lifecycle, not request state."""
    _harden_dependency_logging()
    lifecycle_run_id = new_stage10a_run_id()
    lifecycle_started_at = time.perf_counter()
    sink = _telemetry_sink(application)
    configured = getattr(application.state, "runtime_settings", None)
    runtime = configured if isinstance(configured, Settings) else get_settings()
    startup_metadata = {
        "run_id": lifecycle_run_id,
        "component": "api_runtime",
        "runtime_version": runtime.RUNTIME_VERSION,
    }
    emit_stage10a_event(
        "api.startup.started",
        metadata={**startup_metadata, "status": "STARTED"},
        sink=sink,
    )
    neo4j_driver = None
    try:
        validate_settings(runtime)
        runtime, market_forecast_service, market_data_provider = _configure_quant_runtime(runtime)
        application.state.runtime_settings = runtime
        application.state.sentry_ok, application.state.langsmith_ok = (
            _configure_optional_observability(runtime)
        )

        supabase_client = create_client(runtime.SUPABASE_URL, runtime.SUPABASE_KEY)
        neo4j_driver = GraphDatabase.driver(
            runtime.NEO4J_URI,
            auth=(runtime.NEO4J_USER, runtime.NEO4J_PASSWORD),
        )
        application.state.supabase = supabase_client
        application.state.neo4j_driver = neo4j_driver

        # The object stored on app.state is stateless. It creates the mutable
        # ManagerAgent and ManagerMemory per request/invocation.
        factory = ManagerAgentFactory(
            supabase_client=supabase_client,
            neo4j_driver=neo4j_driver,
            runtime_settings=runtime,
            market_forecast_service=market_forecast_service,
            market_data_provider=market_data_provider,
        )
        # Development/production pays the heavy agent import cost before the UI
        # becomes available, so the first analysis does not appear idle for ~20s.
        # Tests keep imports isolated and side-effect free.
        if runtime.APP_ENV != "test":
            preload_started_at = time.perf_counter()
            factory.preload()
            log.info(
                "agent_modules_preloaded latency_ms=%.3f",
                (time.perf_counter() - preload_started_at) * 1000.0,
            )
        application.state.manager_factory = factory
        application.state.manager_agent = factory  # Stage-10B route compatibility
    except Exception:
        emit_stage10a_event(
            "api.startup.failed",
            metadata={**startup_metadata, "status": "FAILED", "error_code": "API_STARTUP_FAILED"},
            started_at=lifecycle_started_at,
            sink=sink,
        )
        log.error("api_startup_failed code=API_STARTUP_FAILED")
        raise
    emit_stage10a_event(
        "api.startup.completed",
        metadata={**startup_metadata, "status": "READY"},
        started_at=lifecycle_started_at,
        sink=sink,
    )
    log.info("api_runtime_ready")

    try:
        yield
    finally:
        from core.observability import flush_observability
        shutdown_started_at = time.perf_counter()
        emit_stage10a_event(
            "api.shutdown.started",
            metadata={**startup_metadata, "status": "STARTED"},
            sink=sink,
        )
        try:
            close = getattr(neo4j_driver, "close", None)
            if callable(close):
                close()
        except Exception:
            emit_stage10a_event(
                "api.shutdown.failed",
                metadata={**startup_metadata, "status": "FAILED", "error_code": "API_SHUTDOWN_FAILED"},
                started_at=shutdown_started_at,
                sink=sink,
            )
            log.error("api_shutdown_failed code=API_SHUTDOWN_FAILED")
        else:
            emit_stage10a_event(
                "api.shutdown.completed",
                metadata={**startup_metadata, "status": "COMPLETED"},
                started_at=shutdown_started_at,
                sink=sink,
            )
            log.info("api_runtime_stopped")
        flush_observability()


async def alpha_agent_exception_handler(
    request: Request,
    exc: AlphaAgentError,
) -> JSONResponse:
    log.error("api_request_error")
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


async def core_contract_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    return await alpha_agent_exception_handler(request, map_core_exception(exc))


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a stable error envelope without leaking exception text."""
    error = AlphaAgentError()
    log.error("api_request_error code=INTERNAL_ERROR")
    return JSONResponse(status_code=500, content=error.to_dict())


async def request_timing_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    run_id = new_stage10a_run_id()
    request.state.stage10a_run_id = run_id
    sink = _telemetry_sink(request.app)
    metadata = {
        "run_id": run_id,
        "component": "api_request",
        "runtime_version": request.app.state.runtime_settings.RUNTIME_VERSION,
        "request_method": request.method,
    }
    emit_stage10a_event(
        "api.request.started",
        metadata={**metadata, "status": "STARTED"},
        sink=sink,
    )
    try:
        response = await call_next(request)
    except Exception:
        emit_stage10a_event(
            "api.request.failed",
            metadata={**metadata, "status": "FAILED", "error_code": "REQUEST_EXECUTION_FAILED"},
            started_at=started_at,
            sink=sink,
        )
        log.error("api_request_failed code=REQUEST_EXECUTION_FAILED")
        raise
    event = "api.request.failed" if response.status_code >= 500 else "api.request.completed"
    terminal = {
        **metadata,
        "status": "FAILED" if response.status_code >= 500 else "COMPLETED",
        "http_status": response.status_code,
    }
    if response.status_code >= 500:
        terminal["error_code"] = "REQUEST_FAILED_RESPONSE"
    emit_stage10a_event(event, metadata=terminal, started_at=started_at, sink=sink)
    log.info("api_request_terminal status=%s", terminal["status"])
    return response


def register_route_skeletons(application: FastAPI) -> None:
    """Register existing routes; Stage 10B will extend their business contract."""
    application.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])
    application.include_router(progress_router, prefix="/api/v1", tags=["Progress"])


def create_app(*, runtime_settings: Settings | None = None, telemetry_sink=None) -> FastAPI:
    runtime = runtime_settings or get_settings()
    is_production = runtime.APP_ENV == "production"
    application = FastAPI(
        title="AgenticAlpha API",
        description="Versioned monthly forecast runtime foundation.",
        version=runtime.RUNTIME_VERSION,
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
    )
    application.state.runtime_settings = runtime
    application.state.telemetry_sink = telemetry_sink
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime.ALLOWED_ORIGINS if is_production else ["*"],
        allow_credentials=is_production,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.middleware("http")(request_timing_middleware)
    application.middleware("http")(jwt_auth_middleware)
    application.add_exception_handler(AlphaAgentError, alpha_agent_exception_handler)
    for exception_type in CORE_EXCEPTION_TYPES:
        application.add_exception_handler(exception_type, core_contract_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)
    register_route_skeletons(application)

    @application.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "runtime_version": runtime.RUNTIME_VERSION}

    @application.get("/readiness", tags=["Health"])
    async def readiness(request: Request) -> JSONResponse:
        try:
            request.app.state.supabase.table("long_term_memory").select("user_id").limit(1).execute()
            ready = True
        except Exception:
            ready = False
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "not_ready",
                "supabase": "ok" if ready else "unreachable",
                "runtime_version": runtime.RUNTIME_VERSION,
            },
        )

    @application.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "frontend" / "alpha-agent-app.html")

    return application


_is_prod = settings.APP_ENV == "production"  # compatibility for legacy tests/imports
app = create_app(runtime_settings=settings)


logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
_harden_dependency_logging()
