"""Asynchronous, app-scoped API jobs for the closed Stage 09 backtest engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.progress_bus import publish
from evaluation.backtest_cli import build_report
from langsmith_trace import emit_stage10b_event


router = APIRouter()
BACKTEST_MODEL_VERSION = "monthly-model-v1"
BACKTEST_CODE_VERSION = "stage09-r1-a2"
BACKTEST_CONFIG_VERSION = "stage10b-r1-a2"
_FIXTURES = {
    "stage9_walk_forward": Path(__file__).resolve().parents[2]
    / "tests" / "fixtures" / "stage9_walk_forward.json",
}


class BacktestRequest(BaseModel):
    fixture: Literal["stage9_walk_forward"] = "stage9_walk_forward"
    run_id: uuid.UUID | None = Field(
        default=None,
        description="Reuse a run_id to receive the existing idempotent job.",
    )


class BacktestJobResponse(BaseModel):
    run_id: str
    status: Literal["PLANNED", "RUNNING", "COMPLETED", "FAILED"]
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    progress_url: str
    artifact_url: str
    persistence_status: str = "UNKNOWN"
    error_code: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_store(application: Any) -> tuple[dict[str, dict[str, Any]], set[asyncio.Task]]:
    jobs = getattr(application.state, "stage10b_backtest_jobs", None)
    if not isinstance(jobs, dict):
        jobs = {}
        application.state.stage10b_backtest_jobs = jobs
    tasks = getattr(application.state, "stage10b_backtest_tasks", None)
    if not isinstance(tasks, set):
        tasks = set()
        application.state.stage10b_backtest_tasks = tasks
    return jobs, tasks


def _public_job(job: dict[str, Any]) -> BacktestJobResponse:
    return BacktestJobResponse(**{
        key: job.get(key) for key in BacktestJobResponse.model_fields
    })


def _telemetry_sink(application: Any):
    sink = getattr(application.state, "telemetry_sink", None)
    return sink if callable(sink) else None


def _backtest_trace_context(report: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(report.get("manifest") or {})
    return {
        key: manifest[key]
        for key in (
            "dataset_version", "feature_version", "model_version", "schema_version",
        )
        if manifest.get(key) is not None
    }


def _compact_artifact(report: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(report.get("manifest") or {})
    return {
        "manifest": manifest,
        "metrics": dict(report.get("metrics") or {}),
        "leakage_audit": dict(report.get("leakage_audit") or {}),
        "fold_count": len(report.get("folds") or []),
        "record_count": len(report.get("records") or []),
        "report_hash": manifest.get("report_hash"),
    }


def _canonical_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reproducibility_identity(report: dict[str, Any]) -> dict[str, Any]:
    manifest = dict(report.get("manifest") or {})
    return {
        "backtest_version": manifest.get("backtest_version"),
        "universe": {
            "version": manifest.get("universe_version"),
            "manifest_hash": manifest.get("universe_manifest_hash"),
        },
        "cutoff_schedule": manifest.get("cutoff_schedule"),
        "horizon_trading_days": manifest.get("horizon_trading_days"),
        "fold_method": manifest.get("fold_method"),
        "minimum_training_cutoffs": manifest.get("minimum_training_cutoffs"),
        "folds": [dict(item) for item in (report.get("folds") or [])],
        "transaction_cost_policy": {
            "one_way_bps": manifest.get("transaction_cost_bps_one_way"),
            "gross_results_reported": manifest.get("gross_results_reported"),
        },
        "dataset": {
            "version": manifest.get("dataset_version"),
            "manifest_hash": manifest.get("dataset_manifest_hash"),
            "eligible_rows_sha256": manifest.get("eligible_rows_sha256"),
        },
        "feature_version": manifest.get("feature_version"),
        "model_variants": list(manifest.get("model_variants") or []),
        "model_version": BACKTEST_MODEL_VERSION,
        "action_policy": {
            "version": manifest.get("action_policy_version"),
            "thresholds": manifest.get("action_thresholds"),
        },
        "code_version": BACKTEST_CODE_VERSION,
        "config_version": BACKTEST_CONFIG_VERSION,
    }


def _reproducibility_hash(report: dict[str, Any]) -> str:
    return _canonical_hash(_reproducibility_identity(report))


def _persist_backtest(application: Any, job: dict[str, Any], report: dict[str, Any]) -> bool:
    database = getattr(application.state, "supabase", None)
    if database is None:
        return False
    manifest = dict(report.get("manifest") or {})
    reproduction = _reproducibility_identity(report)
    schedule = manifest.get("cutoff_schedule")
    schedule = schedule if isinstance(schedule, dict) else {}
    payload = {
        "run_id": job["run_id"],
        "reproducibility_hash": _reproducibility_hash(report),
        "universe_version": str(manifest.get("universe_version") or "unknown"),
        "cutoff_schedule_version": str(schedule.get("version") or "monthly-cutoff-v1"),
        "fold_definition": {
            "method": manifest.get("fold_method"),
            "minimum_training_cutoffs": manifest.get("minimum_training_cutoffs"),
            "fold_count": manifest.get("fold_count"),
            "definition_sha256": _canonical_hash(reproduction["folds"]),
        },
        "transaction_cost_policy": {
            "one_way_bps": manifest.get("transaction_cost_bps_one_way"),
            "gross_results_reported": manifest.get("gross_results_reported"),
        },
        "dataset_version": str(manifest.get("dataset_version") or "unknown"),
        "feature_version": str(manifest.get("feature_version") or "unknown"),
        "model_variant": "ALL_LOCKED_VARIANTS",
        "model_version": BACKTEST_MODEL_VERSION,
        "action_policy_version": str(manifest.get("action_policy_version") or "unknown"),
        "code_version": BACKTEST_CODE_VERSION,
        "config_version": BACKTEST_CONFIG_VERSION,
        "status": "COMPLETED",
        "metrics": dict(report.get("metrics") or {}),
        "artifact_refs": {
            "api": f"/api/v1/backtests/{job['run_id']}/artifact",
            "report_hash": manifest.get("report_hash"),
        },
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
    }
    try:
        database.table("backtest_runs").upsert(
            payload, on_conflict="run_id", ignore_duplicates=True,
        ).execute()
    except Exception:
        return False
    return True


async def _execute_job(application: Any, job: dict[str, Any], fixture: Path) -> None:
    job["status"] = "RUNNING"
    job["started_at"] = _utc_now()
    started_at = time.perf_counter()
    sink = _telemetry_sink(application)
    emit_stage10b_event(
        "stage10b.backtest.started",
        run_id=job["run_id"],
        status="RUNNING",
        sink=sink,
    )
    publish(
        job["run_id"], "backtest_start", agent="backtest",
        message="BACKTEST_STARTED", detail={"stage": "10B", "attempt": 1},
        run_id=job["run_id"],
    )
    try:
        report = await asyncio.to_thread(build_report, fixture, run_id=job["run_id"])
        job["artifact"] = _compact_artifact(report)
        job["status"] = "COMPLETED"
        job["completed_at"] = _utc_now()
        job["persistence_status"] = "SAVED" if _persist_backtest(application, job, report) else "FAILED"
        emit_stage10b_event(
            "stage10b.backtest.completed",
            run_id=job["run_id"],
            status="COMPLETED",
            context=_backtest_trace_context(report),
            started_at=started_at,
            sink=sink,
        )
        publish(
            job["run_id"], "backtest_complete", agent="backtest",
            message="BACKTEST_COMPLETED",
            detail={"stage": "10B", "attempt": 1, "latency_ms": round((time.perf_counter() - started_at) * 1000, 3)},
            run_id=job["run_id"],
        )
    except Exception:
        job["status"] = "FAILED"
        job["completed_at"] = _utc_now()
        job["error_code"] = "BACKTEST_EXECUTION_FAILED"
        emit_stage10b_event(
            "stage10b.backtest.failed",
            run_id=job["run_id"],
            status="FAILED",
            error_code=job["error_code"],
            started_at=started_at,
            sink=sink,
        )
        publish(
            job["run_id"], "forecast_error", agent="backtest",
            message="BACKTEST_FAILED",
            detail={"stage": "10B", "attempt": 1, "error_code": job["error_code"]},
            run_id=job["run_id"],
        )


@router.post("/backtests", response_model=BacktestJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_backtest(req: BacktestRequest, request: Request) -> BacktestJobResponse:
    jobs, tasks = _job_store(request.app)
    run_id = str(req.run_id or uuid.uuid4())
    existing = jobs.get(run_id)
    if existing is not None:
        return _public_job(existing)
    job = {
        "persistence_status": "PENDING",
        "run_id": run_id,
        "status": "PLANNED",
        "submitted_at": _utc_now(),
        "started_at": None,
        "completed_at": None,
        "progress_url": f"/api/v1/analyze/stream/{run_id}",
        "artifact_url": f"/api/v1/backtests/{run_id}/artifact",
        "error_code": None,
        "artifact": None,
    }
    jobs[run_id] = job
    task = asyncio.create_task(_execute_job(request.app, job, _FIXTURES[req.fixture]))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return _public_job(job)


@router.get("/backtests/{run_id}", response_model=BacktestJobResponse)
async def get_backtest(run_id: uuid.UUID, request: Request) -> BacktestJobResponse:
    jobs, _ = _job_store(request.app)
    job = jobs.get(str(run_id))
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "BACKTEST_NOT_FOUND"})
    return _public_job(job)


@router.get("/backtests/{run_id}/artifact")
async def get_backtest_artifact(run_id: uuid.UUID, request: Request) -> dict[str, Any]:
    jobs, _ = _job_store(request.app)
    job = jobs.get(str(run_id))
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "BACKTEST_NOT_FOUND"})
    if job["status"] != "COMPLETED":
        raise HTTPException(status_code=409, detail={"code": "BACKTEST_NOT_READY"})
    return dict(job["artifact"])


__all__ = [
    "BacktestJobResponse", "BacktestRequest", "create_backtest",
    "get_backtest", "get_backtest_artifact", "router",
]
