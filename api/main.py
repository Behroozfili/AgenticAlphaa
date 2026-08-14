"""
api/main.py
-----------
Run locally
-----------
    uvicorn api.main:app --reload --port 3000
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from neo4j import GraphDatabase

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from supabase import create_client

from agents.financial_agent import FinancialAnalystAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from agents.manager_agent import ManagerAgent
from memory.manager_memory import ManagerMemory

from api.config import settings, validate_settings
from api.core.exceptions import AlphaAgentError, ConfigurationError
from api.routes.analyze import router as analyze_router
from api.routes.progress import router as progress_router
from core.observability import init_sentry, init_langsmith
import os
from fastapi.responses import FileResponse


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("api.main")


# ─────────────────────────────────────────────────────────────
# Lifespan — startup & shutdown
# ─────────────────────────────────────────────────────────────
script_path = os.path.join(os.getcwd(), "tools", "sentiment_tools", "sentiment_server.py")
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    try:
        validate_settings()
    except ConfigurationError as exc:
        log.critical("Startup aborted — %s", exc)
        sys.exit(1)

    
    app.state.sentry_ok  = init_sentry(app_env=settings.APP_ENV)
    langsmith_ok         = init_langsmith()
    log.info("Sentry enabled: %s, LangSmith enabled: %s", app.state.sentry_ok, langsmith_ok)

    
    log.info("Connecting to Supabase...")
    try:
        app.state.supabase = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY,
        )
        log.info("Supabase client ready.")
    except Exception as exc:
        log.critical("Failed to connect to Supabase: %s", exc)
        sys.exit(1)
    log.info("Connecting to Neo4j...")
    try:
        app.state.neo4j_driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        log.info("Neo4j driver ready.")
    except Exception as exc:
        log.critical("Failed to connect to Neo4j: %s", exc)
        sys.exit(1)

    
    log.info("Initialising specialist agents...")
    research_agent  = ResearchAgent()
    financial_agent = FinancialAnalystAgent()
    sentiment_agent = SentimentAgent(server_script_path=script_path)

    # ── 4. Initialise ManagerMemory with a system-level user ─
    #       Per-request memory (user-scoped) is created inside
    #       analyze.py using the user_id from the request.
    #       This instance is only used for agent warm-up.
    system_memory = ManagerMemory(
        user_id="system",
        supabase_client=app.state.supabase,
    )

    # ── 5. Compile ManagerAgent (builds LangGraph once) ──────
    log.info("Compiling ManagerAgent graph...")
    app.state.manager_agent = ManagerAgent(
        research_agent=research_agent,
        financial_agent=financial_agent,
        sentiment_agent=sentiment_agent,
        memory=system_memory,
        model=settings.ANTHROPIC_MODEL,
        max_routing_loops=settings.MAX_ROUTING_LOOPS,
    )
    log.info("ManagerAgent ready.")

    log.info("Alpha-Agent Node API started — env=%s", settings.APP_ENV)

    yield  # ← app is running, handle requests

    # ── Shutdown ─────────────────────────────────────────────
    log.info("Alpha-Agent Node API shutting down.")


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
# is_prod is evaluated inside lifespan (not at import time) so that
# patching APP_ENV in tests takes effect without reloading the module.
_is_prod: bool = settings.APP_ENV == "production"

app = FastAPI(
    title="Alpha-Agent Node API",
    description=(
        "Multi-agent market intelligence system. "
        "Dispatches Research, Financial, and Sentiment agents "
        "to produce structured investment analysis reports."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
)


# ─────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────

# CORS — W3C spec forbids allow_credentials=True with allow_origins=["*"].
# In development: wildcard origins, credentials disabled (safe for local use).
# In production:  explicit ALLOWED_ORIGINS list, credentials enabled.
_dev_mode = settings.APP_ENV == "development"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _dev_mode else settings.ALLOWED_ORIGINS,
    allow_credentials=False if _dev_mode else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Log method, path, status, and duration for every request."""
    start = time.monotonic()
    response = await call_next(request)
    duration = round(time.monotonic() - start, 3)
    log.info(
        "%s %s → %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ─────────────────────────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────────────────────────

@app.exception_handler(AlphaAgentError)
async def alpha_agent_exception_handler(
    request: Request,
    exc: AlphaAgentError,
) -> JSONResponse:
    """
    Convert any AlphaAgentError subclass into a structured JSON response.

    All errors share the same shape:
        { "error": "ERROR_CODE", "message": "...", "detail": "...", "trace_id": "..." }

    trace_id appears both in the log and the response — grep it instantly:
        grep "trace_id=a3f2c1b9" logs/api.log
    """
    log.error(
        "AlphaAgentError [%s] trace_id=%s — %s | detail: %s",
        exc.code, exc.trace_id, exc.message, exc.detail,
    )
    if request.app.state.sentry_ok:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("trace_id", exc.trace_id)
            scope.set_tag("error_code", exc.code)
            sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for any unhandled exception — return 500."""
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    if request.app.state.sentry_ok:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    # In production, never leak internal exception details to the client.
    # The trace_id lets engineers grep the server logs for the full traceback.
    import uuid
    trace_id = str(uuid.uuid4())[:8]
    if settings.APP_ENV == "production":
        content = {
            "error":    "INTERNAL_ERROR",
            "message":  "An unexpected error occurred.",
            "trace_id": trace_id,
        }
    else:
        content = {
            "error":    "INTERNAL_ERROR",
            "message":  "An unexpected error occurred.",
            "detail":   str(exc),
            "trace_id": trace_id,
        }
    return JSONResponse(status_code=500, content=content)


# ─────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────

app.include_router(analyze_router, prefix="/api/v1", tags=["Analysis"])
app.include_router(progress_router, prefix="/api/v1", tags=["Progress"])


# ─────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    """
    Liveness probe — Railway/Fly.io calls this to check the app is alive.
    Returns 200 as long as the process is running.
    """
    return {"status": "ok"}


@app.get("/readiness", tags=["Health"])
async def readiness(request: Request):
    """
    Readiness probe — checks that Supabase is reachable.
    Returns 200 only when all dependencies are healthy.
    """
    try:
        # Lightweight query to verify Supabase connection
        request.app.state.supabase.table("long_term_memory").select("user_id").limit(1).execute()
        supabase_ok = True
    except Exception:
        supabase_ok = False

    status = "ready" if supabase_ok else "not_ready"
    http_code = 200 if supabase_ok else 503

    return JSONResponse(
        status_code=http_code,
        content={
            "status":   status,
            "supabase": "ok" if supabase_ok else "unreachable",
        },
    )
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("frontend/alpha-agent-app.html")