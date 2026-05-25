"""
api/routes/analyze.py
---------------------
POST /api/v1/analyze  — Core analysis endpoint for Alpha-Agent Node.

Request flow
------------
  1. Validate request body via Pydantic (AnalyzeRequest)
  2. Build task_query and manager_directives from request fields
  3. Receive ManagerMemory via Depends() — injected per-request
  4. Call ManagerAgent.run() — async, may take 15-60 seconds
  5. Persist result to Supabase `analyses` table
  6. Return structured AnalyzeResponse

Authentication
--------------
  Not implemented yet. user_id is taken directly from the request body.
  When JWT is added: remove user_id from AnalyzeRequest and read from
  request.state.user_id (set by JWT middleware). The DI layer in
  dependencies.py already supports this via get_user_id().

Dependencies
------------
  pip install fastapi supabase uuid
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from memory.manager_memory import ManagerMemory
from api.dependencies import get_manager_memory

log = logging.getLogger("api.analyze")

router = APIRouter()


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Incoming analysis request.

    Fields
    ------
    query : str
        Natural-language analysis objective.
        Example: "Is NVIDIA a good buy for Q1 2025?"
    ticker : str | None
        Uppercase ticker symbol. Injected into manager_directives.
        Example: "NVDA"
    user_id : str
        Caller identifier. Used for per-user long-term memory in Supabase.
        Will be replaced by JWT token extraction once auth is added.
    search_depth : str
        Passed to ResearchAgent via manager_directives. "basic" | "advanced".
    days_back : int
        How many days of historical data to consider. 1–365.
    include_sentiment : bool
        Whether SentimentAgent should run. Passed via manager_directives.
    peers : list[str]
        Peer tickers for competitor comparison. Example: ["AMD", "INTC"]
    """

    query: str = Field(..., min_length=10, max_length=500)
    ticker: str | None = Field(default=None)
    user_id: str = Field(default="anonymous")
    search_depth: str = Field(default="advanced")
    days_back: int = Field(default=14, ge=1, le=365)
    include_sentiment: bool = Field(default=True)
    peers: list[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str | None) -> str | None:
        """Force ticker to uppercase and validate format."""
        if v is None:
            return v
        v = v.upper().strip()
        if not v.isalpha() or not (1 <= len(v) <= 5):
            raise ValueError("ticker must be 1-5 uppercase letters, e.g. 'NVDA'")
        return v

    @field_validator("search_depth")
    @classmethod
    def valid_search_depth(cls, v: str) -> str:
        if v not in ("basic", "advanced"):
            raise ValueError("search_depth must be 'basic' or 'advanced'")
        return v

    @field_validator("peers")
    @classmethod
    def peers_uppercase(cls, v: list[str]) -> list[str]:
        return [p.upper().strip() for p in v]


class AnalyzeResponse(BaseModel):
    """
    Structured response returned after analysis completes.

    Maps directly from SharedManagerState fields returned by ManagerAgent.run().
    """

    analysis_id: str
    user_id: str
    status: str                          # "completed" | "failed"
    created_at: str
    completed_at: str
    duration_s: float

    # Core report
    final_report: str

    # Agent outputs (pass-through from SharedManagerState)
    financial_metrics: dict[str, Any]
    sentiment_analysis: dict[str, Any]
    research_context_chunks: int

    # Observability
    agent_execution_history: list[Any]
    orchestrator_logs: list[Any]


# ─────────────────────────────────────────────
# Helper — persist result to Supabase
# ─────────────────────────────────────────────

async def _persist_analysis(
    request: Request,
    analysis_id: str,
    user_id: str,
    req: AnalyzeRequest,
    result: dict[str, Any],
    status: str,
    error_message: str | None,
    created_at: str,
    completed_at: str,
    duration_s: float,
) -> None:
    """
    Write the analysis result to the `analyses` Supabase table.

    Called after ManagerAgent.run() returns (success or failure).
    Errors here are logged but do NOT raise — the client already has
    the response by this point (fire-and-forget pattern).

    Required Supabase table
    -----------------------
    Run once in the Supabase SQL Editor::

        CREATE TABLE analyses (
            analysis_id   TEXT PRIMARY KEY,
            user_id       TEXT,
            query         TEXT,
            ticker        TEXT,
            status        TEXT,
            final_report  TEXT,
            financial_metrics    JSONB DEFAULT '{}',
            sentiment_analysis   JSONB DEFAULT '{}',
            agent_execution_history JSONB DEFAULT '[]',
            orchestrator_logs       JSONB DEFAULT '[]',
            error_message TEXT,
            duration_s    FLOAT,
            created_at    TIMESTAMPTZ,
            completed_at  TIMESTAMPTZ
        );
    """
    try:
        db = request.app.state.supabase
        db.table("analyses").insert({
            "analysis_id":              analysis_id,
            "user_id":                  user_id,
            "query":                    req.query,
            "ticker":                   req.ticker,
            "status":                   status,
            "final_report":             result.get("final_report", ""),
            "financial_metrics":        result.get("financial_metrics_summary", {}),
            "sentiment_analysis":       result.get("sentiment_analysis_summary", {}),
            "agent_execution_history":  result.get("agent_execution_history", []),
            "orchestrator_logs":        result.get("orchestrator_logs", []),
            "error_message":            error_message,
            "duration_s":               duration_s,
            "created_at":               created_at,
            "completed_at":             completed_at,
        }).execute()
        log.info("[analyze] Persisted analysis_id=%s to Supabase.", analysis_id)
    except Exception as exc:
        log.error("[analyze] Failed to persist analysis_id=%s: %s", analysis_id, exc)


# ─────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Run multi-agent market analysis",
    description=(
        "Dispatches a coordinated team of specialist agents (Research, Financial, Sentiment) "
        "to produce a structured investment analysis report. Typical response time: 15–60 seconds."
    ),
)
async def analyze(
    req: AnalyzeRequest,
    request: Request,
    memory: ManagerMemory = Depends(get_manager_memory),
) -> AnalyzeResponse:
    """
    POST /api/v1/analyze

    Orchestrates the full Alpha-Agent Node pipeline for a single query.
    ManagerMemory is injected via Depends() — not created manually.
    """
    analysis_id = str(uuid.uuid4())
    created_at  = datetime.now(timezone.utc).isoformat()
    started_at  = time.monotonic()

    log.info(
        "[analyze] START analysis_id=%s user=%s ticker=%s query='%s'",
        analysis_id, req.user_id, req.ticker, req.query[:80],
    )

    # ── 1. Build manager_directives from request fields ──────────────────────
    manager_directives: dict[str, Any] = {
        "ticker":             req.ticker,
        "search_depth":       req.search_depth,
        "days_back":          req.days_back,
        "include_sentiment":  req.include_sentiment,
        "peers":              req.peers,
    }

    # ── 2. user_preferences from injected memory ──────────────────────────────
    user_preferences = memory.long.get_all_preferences()

    # ── 3. Run the ManagerAgent pipeline ─────────────────────────────────────
    manager_agent = request.app.state.manager_agent
    result: dict[str, Any] = {}
    status        = "completed"
    error_message = None

    try:
        result = await manager_agent.run(
            task_query=req.query,
            manager_directives=manager_directives,
            user_preferences=user_preferences,
        )
    except Exception as exc:
        log.exception("[analyze] ManagerAgent.run() failed: %s", exc)
        status        = "failed"
        error_message = str(exc)

    # ── 4. Timing ─────────────────────────────────────────────────────────────
    duration_s   = round(time.monotonic() - started_at, 2)
    completed_at = datetime.now(timezone.utc).isoformat()

    log.info(
        "[analyze] END analysis_id=%s status=%s duration=%.1fs",
        analysis_id, status, duration_s,
    )

    # ── 5. Persist to Supabase ────────────────────────────────────────────────
    await _persist_analysis(
        request=request,
        analysis_id=analysis_id,
        user_id=req.user_id,
        req=req,
        result=result,
        status=status,
        error_message=error_message,
        created_at=created_at,
        completed_at=completed_at,
        duration_s=duration_s,
    )

    # ── 6. If the agent failed, raise HTTP 500 ────────────────────────────────
    if status == "failed":
        raise HTTPException(
            status_code=500,
            detail={
                "error":       "Agent pipeline failed",
                "code":        "AGENT_ERROR",
                "analysis_id": analysis_id,
                "message":     error_message,
            },
        )

    # ── 7. Build and return response ──────────────────────────────────────────
    return AnalyzeResponse(
        analysis_id=analysis_id,
        user_id=req.user_id,
        status=status,
        created_at=created_at,
        completed_at=completed_at,
        duration_s=duration_s,
        final_report=result.get("final_report", ""),
        financial_metrics=result.get("financial_metrics_summary", {}),
        sentiment_analysis=result.get("sentiment_analysis_summary", {}),
        research_context_chunks=len(result.get("aggregated_research_context", [])),
        agent_execution_history=result.get("agent_execution_history", []),
        orchestrator_logs=result.get("orchestrator_logs", []),
    )
