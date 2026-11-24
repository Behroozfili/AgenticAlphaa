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
  JWT middleware verifies the identity; get_user_id() reads that identity.
  Client-provided identity headers and body fields are not trusted.

Dependencies
------------
  pip install fastapi supabase uuid
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from api.config import Settings, get_settings
from memory.manager_memory import ManagerMemory
from core.forecast_storage import feature_snapshot, write_with_retry
from api.dependencies import get_manager_memory, get_user_id
from api.core.exceptions import AgentError
from core.forecast_contract import (
    AnalysisMode, ForecastCutoffPolicy, HORIZON_TRADING_DAYS, regular_session_close_utc,
)
from evaluation.publication_policy import (
    assemble_publication,
    execution_status_from_publication,
)
from langsmith_trace import emit_stage10b_event
from core.progress_bus import publish as _publish_progress

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
    search_depth: str = Field(default="advanced")
    days_back: int = Field(default=14, ge=1, le=365)
    include_sentiment: bool = Field(default=True)
    as_of: str | None = Field(
        default=None,
        description="UTC close of the last regular US trading session of the month.",
    )
    horizon: int = Field(default=HORIZON_TRADING_DAYS)
    mode: AnalysisMode | None = Field(default=None)
    cutoff_policy: ForecastCutoffPolicy | None = Field(default=None)
    run_id: uuid.UUID | None = Field(
        default=None,
        description="Stable forecast run identifier; reuse it for idempotent retries.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional client-generated id. If provided, ManagerAgent uses it "
            "verbatim as the analysis/session id, and it doubles as the "
            "channel for the live progress stream at "
            "GET /api/v1/analyze/stream/{session_id} — the frontend should "
            "open that stream BEFORE posting here, using this same id, so no "
            "early progress events are missed. If omitted, a server-side id "
            "is generated and no live stream is available for this request."
        ),
    )

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

    @field_validator("horizon")
    @classmethod
    def fixed_monthly_horizon(cls, value: int) -> int:
        if value != HORIZON_TRADING_DAYS:
            raise ValueError("horizon must be exactly 21 trading sessions")
        return value

    @field_validator("mode", "cutoff_policy", mode="before")
    @classmethod
    def normalize_mode(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_monthly_identity(self) -> "AnalyzeRequest":
        monthly_requested = self.as_of is not None or self.mode is not None or self.run_id is not None
        if monthly_requested and (self.ticker is None or self.as_of is None):
            raise ValueError("monthly requests require both ticker and as_of")
        return self



class MonthlyForecastResponse(BaseModel):
    status: str
    ticker: str
    as_of: str
    target_date: str
    horizon_trading_days: int
    analysis_mode: str
    cutoff_policy: str = "MONTH_END"
    benchmark: str
    run_id: str
    universe_manifest_hash: str | None
    model_variant: str | None
    feature_materialized_at: str | None
    action: str
    probability_positive: float | None
    expected_return_21d: float | None
    expected_excess_return: float | None
    confidence_interval: list[float] | None
    confidence: float | None
    evidence_ids: list[str]
    warnings: list[str]
    missing: bool
    degraded: bool
    graph_coverage: bool | None
    quality_gate_passed: bool = True
    quality_gate_failures: list[str] = Field(default_factory=list)
    specialist_contributions: dict[str, Any] = Field(default_factory=dict)
    decision_drivers: list[dict[str, Any]] = Field(default_factory=list)
    versions: dict[str, str | None]
    narrative: str = ""
    implementation: dict[str, Any] = Field(default_factory=dict)


class ForecastDefaultsResponse(BaseModel):
    """Safe frontend defaults derived from the active artifact and immutable caches."""

    ticker: str | None
    active_tickers: list[str]
    as_of: str | None
    horizon: int = HORIZON_TRADING_DAYS
    mode: AnalysisMode = AnalysisMode.LIVE
    cutoff_policy: ForecastCutoffPolicy = ForecastCutoffPolicy.MONTH_TO_DATE
    days_back: int = 30
    source: str


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
    publication_status: str = "PUBLISHED"
    publication_warnings: list[Any] = Field(default_factory=list)
    withheld_sections: list[str] = Field(default_factory=list)
    section_reliability: list[dict[str, Any]] = Field(default_factory=list)

    # Agent outputs (pass-through from SharedManagerState)
    financial_metrics: dict[str, Any]
    sentiment_analysis: dict[str, Any]
    research_context_chunks: int

    # Observability
    agent_execution_history: list[Any]
    orchestrator_logs: list[Any]
    forecast_context: dict[str, Any] | None = None
    monthly_forecast: MonthlyForecastResponse | None = None
    forecast_run_id: str | None = None
    persistence_status: str = "UNKNOWN"
    persistence_details: dict[str, str] = Field(default_factory=dict)


def _runtime_settings(request: Request) -> Settings:
    configured = getattr(request.app.state, "runtime_settings", None)
    return configured if isinstance(configured, Settings) else get_settings()



def _runtime_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _active_artifact_tickers(settings: Settings) -> list[str]:
    configured = settings.QUANT_ARTIFACT_PATH.strip()
    if not configured:
        return []
    try:
        payload = json.loads(_runtime_path(configured).read_text(encoding="utf-8"))
        constituents = payload.get("universe_manifest", {}).get("constituents", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    tickers: list[str] = []
    for row in constituents if isinstance(constituents, list) else []:
        ticker = str(row.get("ticker") or "").strip().upper() if isinstance(row, dict) else ""
        if ticker.isalpha() and 1 <= len(ticker) <= 5 and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _cached_cutoff_dates(settings: Settings, symbol: str) -> set[date]:
    folder = (
        _runtime_path(settings.MARKET_DATA_CACHE_DIR)
        / settings.LIVE_DATASET_VERSION
        / symbol
    )
    dates: set[date] = set()
    if not folder.is_dir():
        return dates
    for path in folder.glob("*.json"):
        parts = path.stem.split("_")
        if len(parts) != 2:
            continue
        try:
            date.fromisoformat(parts[0])
            dates.add(date.fromisoformat(parts[1]))
        except ValueError:
            continue
    return dates


def _latest_cached_cutoff(settings: Settings, *, ticker: str) -> date | None:
    required_symbols = {ticker, settings.FORECAST_BENCHMARK.strip().upper()}
    if settings.MARKET_SECTOR_BENCHMARK.strip():
        required_symbols.add(settings.MARKET_SECTOR_BENCHMARK.strip().upper())
    coverage = [_cached_cutoff_dates(settings, symbol) for symbol in sorted(required_symbols)]
    common = set.intersection(*coverage) if coverage and all(coverage) else set()
    return max(common) if common else None


@router.get("/forecast/defaults", response_model=ForecastDefaultsResponse)
async def forecast_defaults(request: Request) -> ForecastDefaultsResponse:
    """Return a cutoff that the active model and cached market inputs can serve."""

    settings = _runtime_settings(request)
    tickers = _active_artifact_tickers(settings)
    ticker = tickers[0] if tickers else None
    cutoff = _latest_cached_cutoff(settings, ticker=ticker) if ticker else None
    return ForecastDefaultsResponse(
        ticker=ticker,
        active_tickers=tickers,
        as_of=cutoff.isoformat() if cutoff else None,
        source="active_artifact_common_market_cache" if cutoff else "active_artifact_only",
    )


def _monthly_request_context(
    request: Request,
    req: AnalyzeRequest,
) -> tuple[dict[str, Any] | None, str | None]:
    if req.as_of is None:
        return None, None
    as_of: str = req.as_of
    if len(as_of) == 10:
        try:
            as_of = regular_session_close_utc(date.fromisoformat(as_of)).isoformat()
        except ValueError:
            pass
    context = _runtime_settings(request).forecast_context(
        ticker=req.ticker or "",
        as_of=as_of,
        analysis_mode=req.mode or AnalysisMode.LIVE,
        cutoff_policy=req.cutoff_policy,
    )
    if context.analysis_mode is AnalysisMode.LIVE and context.as_of > datetime.now(timezone.utc):
        raise ValueError("LIVE as_of must be a completed market close")
    return context.to_dict(), str(req.run_id or uuid.uuid4())


def _compact_monthly_forecast(result: dict[str, Any]) -> dict[str, Any] | None:
    forecast = result.get("monthly_forecast")
    if not isinstance(forecast, dict):
        return None
    allowed = (
        "status", "ticker", "as_of", "target_date", "horizon_trading_days",
        "analysis_mode", "cutoff_policy", "benchmark", "run_id", "universe_manifest_hash",
        "model_variant", "feature_materialized_at", "action",
        "probability_positive", "expected_return_21d", "expected_excess_return",
        "confidence_interval", "confidence", "evidence_ids", "warnings",
        "missing", "degraded", "graph_coverage", "quality_gate_passed",
        "quality_gate_failures", "specialist_contributions", "decision_drivers", "versions", "narrative", "implementation",
    )
    compact = {key: forecast.get(key) for key in allowed}
    compact["cutoff_policy"] = forecast.get("cutoff_policy") or "MONTH_END"
    compact["quality_gate_passed"] = forecast.get("quality_gate_passed", True)
    compact["quality_gate_failures"] = [
        str(value)[:160] for value in (forecast.get("quality_gate_failures") or [])[:16]
    ]
    compact["narrative"] = str(forecast.get("narrative") or "")[:4000]
    compact["implementation"] = dict(forecast.get("implementation") or {})
    compact["specialist_contributions"] = dict(
        forecast.get("specialist_contributions") or {}
    )
    compact["decision_drivers"] = [
        dict(value) for value in (forecast.get("decision_drivers") or [])[:16]
        if isinstance(value, dict)
    ]

    compact["evidence_ids"] = [
        str(value)[:160] for value in (forecast.get("evidence_ids") or [])[:256]
    ]
    compact["warnings"] = [
        str(value)[:200] for value in (forecast.get("warnings") or [])[:32]
    ]
    return compact


def _monthly_analysis_id(context: dict[str, Any]) -> str:
    identity = "|".join(str(context.get(key, "MONTH_END" if key == "cutoff_policy" else "")) for key in (
        "ticker", "as_of", "horizon_trading_days", "analysis_mode", "cutoff_policy", "schema_version",
    ))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agenticalpha:analysis:{identity}"))


def _forecast_row(forecast: dict[str, Any]) -> dict[str, Any] | None:
    versions = forecast.get("versions")
    versions = versions if isinstance(versions, dict) else {}
    required = {
        "ticker": forecast.get("ticker"),
        "as_of": forecast.get("as_of"),
        "target_date": forecast.get("target_date"),
        "benchmark": forecast.get("benchmark"),
        "cutoff_policy": forecast.get("cutoff_policy") or "MONTH_END",
        "model_variant": forecast.get("model_variant"),
        "model_version": versions.get("model_version"),
        "feature_version": versions.get("feature_version"),
        "dataset_version": versions.get("dataset_version"),
        "schema_version": versions.get("schema_version"),
        "action_policy_version": versions.get("action_policy_version"),
        "run_id": forecast.get("run_id"),
    }
    provenance = {
        "universe_manifest_hash": forecast.get("universe_manifest_hash"),
        "feature_materialized_at": forecast.get("feature_materialized_at"),
    }
    if any(
        value is None or value == ""
        for value in (*required.values(), *provenance.values())
    ):
        return None
    interval = forecast.get("confidence_interval")
    lower = interval[0] if isinstance(interval, (list, tuple)) and len(interval) == 2 else None
    upper = interval[1] if isinstance(interval, (list, tuple)) and len(interval) == 2 else None
    ticker_return = forecast.get("expected_return_21d")
    excess_return = forecast.get("expected_excess_return")
    benchmark_return = (
        ticker_return - excess_return
        if isinstance(ticker_return, (int, float)) and isinstance(excess_return, (int, float))
        else None
    )
    return {
        **required,
        "horizon_trading_days": forecast.get("horizon_trading_days"),
        "feature_snapshot_id": None,
        "predicted_excess_return": excess_return,
        "predicted_ticker_return": ticker_return,
        "predicted_benchmark_return": benchmark_return,
        "uncertainty_lower": lower,
        "uncertainty_upper": upper,
        "action": forecast.get("action"),
        "evidence_ids": list(forecast.get("evidence_ids") or []),
        "forecast_payload": {
            key: forecast.get(key) for key in (
                "status", "cutoff_policy", "probability_positive", "confidence", "missing",
                "degraded", "graph_coverage", "quality_gate_passed",
                "quality_gate_failures", "warnings", "versions",
                "specialist_contributions", "decision_drivers",
            )
        } | provenance,
    }


def _outcome_row(
    result: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    outcome = result.get("forecast_outcome")
    if not isinstance(outcome, dict):
        return None
    required_names = (
        "price_policy_version", "ticker_start_adjusted_close",
        "ticker_end_adjusted_close", "benchmark_start_adjusted_close",
        "benchmark_end_adjusted_close", "ticker_total_return",
        "benchmark_total_return", "excess_return", "observable_at",
    )
    if any(outcome.get(name) is None for name in required_names):
        return None
    if any(outcome.get(name) != context.get(name) for name in (
        "ticker", "as_of", "horizon_trading_days", "benchmark",
    )):
        return None
    if outcome.get("target_date") != context.get("target"):
        return None
    try:
        observable_at = datetime.fromisoformat(
            str(outcome["observable_at"]).replace("Z", "+00:00")
        )
        if observable_at.tzinfo is None or observable_at > datetime.now(timezone.utc):
            return None
    except (TypeError, ValueError):
        return None
    return {
        "ticker": context["ticker"],
        "as_of": context["as_of"],
        "target_date": context["target"],
        "horizon_trading_days": context["horizon_trading_days"],
        "price_policy_version": outcome["price_policy_version"],
        "benchmark": context["benchmark"],
        "dataset_version": context["dataset_version"],
        "ticker_start_adjusted_close": outcome["ticker_start_adjusted_close"],
        "ticker_end_adjusted_close": outcome["ticker_end_adjusted_close"],
        "benchmark_start_adjusted_close": outcome["benchmark_start_adjusted_close"],
        "benchmark_end_adjusted_close": outcome["benchmark_end_adjusted_close"],
        "ticker_total_return": outcome["ticker_total_return"],
        "benchmark_total_return": outcome["benchmark_total_return"],
        "excess_return": outcome["excess_return"],
        "source_snapshot_ids": [
            str(value)[:160] for value in (outcome.get("source_snapshot_ids") or [])[:256]
        ],
        "observable_at": outcome["observable_at"],
    }


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
    forecast_context: dict[str, Any] | None = None,
    forecast_run_id: str | None = None,
    persistence_details: dict[str, str] | None = None,
) -> bool:
    """
    Write the analysis result to the `analyses` Supabase table.

    Called after ManagerAgent.run() returns (success or failure).
    Errors are nonfatal; the awaited outcome is exposed in persistence_status.
    Monthly bundles use one transaction. A transport failure means commit is unconfirmed.

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
    details = persistence_details if persistence_details is not None else {}
    details.update(snapshot="NOT_APPLICABLE" if forecast_context is None else "UNAVAILABLE", analysis="NOT_ATTEMPTED", forecast="NOT_APPLICABLE" if forecast_context is None else "UNAVAILABLE", outcome="NOT_AVAILABLE")
    current_component = "analysis"
    opaque_user_id = "sha256:" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    try:
        db = request.app.state.supabase
        payload: dict[str, Any] = {
            "analysis_id":              analysis_id,
            "user_id":                  opaque_user_id,
            "query":                    "[redacted]",
            "ticker":                   req.ticker,
            "status":                   status,
            "final_report":             "",
            "financial_metrics":        {},
            "sentiment_analysis":       {},
            "agent_execution_history":  [],
            "orchestrator_logs":        [],
            "error_message":            error_message,
            "duration_s":               duration_s,
            "created_at":               created_at,
            "completed_at":             completed_at,
        }
        if forecast_context is None:
            db.table("analyses").insert(payload).execute()
            details["analysis"] = "SAVED"
        else:
            versions = {
                key: forecast_context.get(key) for key in (
                    "schema_version", "dataset_version", "feature_version", "model_version",
                )
            }
            compact = _compact_monthly_forecast(result)
            payload.update({
                "analysis_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "agenticalpha:run:" + str(forecast_run_id))),
                "as_of": forecast_context.get("as_of"),
                "target_date": forecast_context.get("target"),
                "horizon_trading_days": forecast_context.get("horizon_trading_days"),
                "analysis_mode": forecast_context.get("analysis_mode"),
                **versions,
                "prompt_version": "prompt-manager-monthly-v1",
                "run_id": forecast_run_id,
                "evidence_ids": list((compact or {}).get("evidence_ids") or []),
            })
            row = _forecast_row(compact or {})
            if row is None:
                # No partial monthly record: a valid bundle is required.
                raise ValueError("FORECAST_BUNDLE_UNAVAILABLE")
            current_component = "bundle"
            if row["run_id"] != forecast_run_id:
                raise ValueError("FORECAST_RUN_ID_MISMATCH")
            snapshot = feature_snapshot(result, forecast_context, compact)
            row["feature_snapshot_id"] = snapshot["snapshot_id"]
            row["forecast_payload"].update({
                "analysis_mode": forecast_context["analysis_mode"],
                "forecast_context": forecast_context,
                "monthly_insight": {k: v for k, v in compact.items() if k != "narrative"},
            })
            outcome = _outcome_row(result, forecast_context)
            request_identity = req.model_dump(mode="json", exclude={"session_id", "run_id", "as_of", "mode", "cutoff_policy"})
            request_identity.update(context=forecast_context, user_id=opaque_user_id)
            request_hash = hashlib.sha256(json.dumps(
                request_identity, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest()
            args = {"p_analysis": payload, "p_snapshot": snapshot,
                    "p_forecast": row, "p_outcome": outcome, "p_request_hash": request_hash}
            response = await write_with_retry(lambda: db.rpc("persist_monthly_bundle", args).execute())
            acknowledgement = response.data
            if not isinstance(acknowledgement, dict) or acknowledgement.get("status") != "SAVED" or acknowledgement.get("run_id") != forecast_run_id:
                raise ValueError("FORECAST_COMMIT_UNCONFIRMED")
            details.update(analysis="SAVED", snapshot="SAVED", forecast="SAVED", bundle="SAVED")
            if outcome is not None:
                details["outcome"] = "SAVED"
        log.info("analysis_persist_completed")
        return True
    except Exception as exc:
        if str(getattr(exc, "code", "")) == "PAA01":
            raise HTTPException(status_code=409, detail={"code": "FORECAST_RUN_CONFLICT"}) from None
        details[current_component] = "FAILED"
        log.error("analysis_persist_failed code=ANALYSIS_PERSIST_FAILED")
        return False


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
    # Single-user correlation id shared by the request and progress stream.
    analysis_id = req.session_id or str(uuid.uuid4())
    created_at  = datetime.now(timezone.utc).isoformat()
    started_at  = time.perf_counter()

    log.info("analysis_request_started")
    user_id = get_user_id(request)
    _publish_progress(
        req.session_id,
        "pipeline_start",
        agent="manager",
        message="Request accepted — preparing the agent runtime",
        detail={"ticker": req.ticker, "phase": "runtime_prepare"},
    )
    # Let the already-open SSE response flush the accepted event before any
    # synchronous provider/module setup begins.
    await asyncio.sleep(0)

    try:
        forecast_context, forecast_run_id = _monthly_request_context(request, req)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_FORECAST_CONTEXT"},
        ) from None
    telemetry_sink = getattr(request.app.state, "telemetry_sink", None)
    telemetry_sink = telemetry_sink if callable(telemetry_sink) else None
    if forecast_context is not None and forecast_run_id is not None:
        emit_stage10b_event(
            "stage10b.analysis.started",
            run_id=forecast_run_id,
            status="STARTED",
            context=forecast_context,
            sink=telemetry_sink,
        )

    # ── 1. Build manager_directives from request fields ──────────────────────
    manager_directives: dict[str, Any] = {
        "ticker":             req.ticker,
        "search_depth":       req.search_depth,
        "days_back":          req.days_back,
        "include_sentiment":  req.include_sentiment,
        
    }
    if forecast_context is not None:
        manager_directives.update({
            "forecast_context": forecast_context,
            "run_id": forecast_run_id,
            "horizon": req.horizon,
            "mode": forecast_context["analysis_mode"],
        })

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
            client_session_id=analysis_id,
        )
        _publish_progress(
            req.session_id,
            "pipeline_complete",
            agent="manager",
            message="All agents completed — preparing the result",
            detail={"phase": "result_prepare"},
        )
    except Exception:
        log.error("analysis_execution_failed code=ANALYSIS_EXECUTION_FAILED")
        _publish_progress(
            req.session_id,
            "pipeline_error",
            agent="manager",
            message="The coordinated analysis failed",
            detail={"error_code": "ANALYSIS_EXECUTION_FAILED"},
        )
        status        = "failed"
        error_message = "ANALYSIS_EXECUTION_FAILED"
        result = assemble_publication("", {"passed":False,"critical_findings":[],"warnings":[]}, {})

    if status != "failed":
        validation = result.get("final_publication_validation", {}) or {}
        status = result.get("execution_status") or execution_status_from_publication(
            result.get("publication_status"),
            validation_passed=validation.get("passed"),
        )
        if status == "failed" and error_message is None:
            error_message = "ANALYSIS_PUBLICATION_FAILED"

    # ── 4. Timing ─────────────────────────────────────────────────────────────
    duration_s   = round(time.perf_counter() - started_at, 2)
    completed_at = datetime.now(timezone.utc).isoformat()

    log.info("analysis_request_completed")

    # ── 5. Persist to Supabase ────────────────────────────────────────────────
    persistence_started_at = time.perf_counter()
    persistence_details: dict[str, str] = {}
    persisted = await _persist_analysis(
        request=request,
        analysis_id=analysis_id,
        user_id=user_id,
        req=req,
        result=result,
        status=status,
        error_message=error_message,
        created_at=created_at,
        completed_at=completed_at,
        duration_s=duration_s,
        forecast_context=forecast_context,
        forecast_run_id=forecast_run_id,
        persistence_details=persistence_details,
    )
    persistence_status = (
        ("PARTIAL" if "SAVED" in persistence_details.values() else "FAILED") if not persisted else
        "PARTIAL" if forecast_context is not None and persistence_details.get("forecast") != "SAVED" else
        "SAVED"
    )
    compact_forecast = _compact_monthly_forecast(result)
    if forecast_context is not None and forecast_run_id is not None:
        emit_stage10b_event(
            (
                "stage10b.persistence.completed"
                if persisted else "stage10b.persistence.failed"
            ),
            run_id=forecast_run_id,
            status="COMPLETED" if persisted else "FAILED",
            context=forecast_context,
            error_code=None if persisted else "ANALYSIS_PERSIST_FAILED",
            started_at=persistence_started_at,
            sink=telemetry_sink,
        )
        forecast_status = (compact_forecast or {}).get("status")
        if forecast_status == "INSUFFICIENT_DATA":
            terminal_status = "INSUFFICIENT_DATA"
        elif status == "failed":
            terminal_status = "FAILED"
        elif (compact_forecast or {}).get("degraded"):
            terminal_status = "DEGRADED"
        else:
            terminal_status = "COMPLETED"
        evidence = (compact_forecast or {}).get("evidence_ids")
        emit_stage10b_event(
            (
                "stage10b.analysis.failed"
                if terminal_status == "FAILED"
                else "stage10b.analysis.completed"
            ),
            run_id=forecast_run_id,
            status=terminal_status,
            context=forecast_context,
            error_code=error_message if terminal_status == "FAILED" else None,
            started_at=started_at,
            evidence_count=len(evidence) if isinstance(evidence, list) else 0,
            degraded_flags=(
                ["monthly_degraded"]
                if (compact_forecast or {}).get("degraded")
                else []
            ),
            sink=telemetry_sink,
        )

    # ── 6. If the agent failed, raise AgentError (→ Sentry via main.py handler)
    # ── 7. Build and return response ──────────────────────────────────────────
    return AnalyzeResponse(
        analysis_id=analysis_id,
        user_id=user_id,
        status=status,
        persistence_status=persistence_status,
        persistence_details=persistence_details,
        created_at=created_at,
        completed_at=completed_at,
        duration_s=duration_s,
        final_report=result.get("final_report", ""),
        publication_status=result.get("publication_status", "DATA_INSUFFICIENT"),
        publication_warnings=result.get("publication_warnings", []),
        withheld_sections=result.get("withheld_sections", []),
        section_reliability=result.get("section_reliability", []),
        financial_metrics=result.get("financial_metrics_summary", {}),
        sentiment_analysis=result.get("sentiment_analysis_summary", {}),
        research_context_chunks=len(result.get("aggregated_research_context", [])),
        agent_execution_history=result.get("agent_execution_history", []),
        orchestrator_logs=result.get("orchestrator_logs", []),
        forecast_context=forecast_context,
        monthly_forecast=_compact_monthly_forecast(result),
        forecast_run_id=forecast_run_id,
    )
