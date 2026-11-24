"""Versioned point-in-time contracts shared by the Stage 05 financial tools."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Mapping
from datetime import date, datetime, timezone
from types import MappingProxyType
from typing import Any

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError
from core.forecast_contract import ForecastContext
from core.observability import elapsed_latency_ms, forecast_trace_metadata

FINANCIAL_CONTRACT_VERSION = "monthly-financial-features-v1"
FINANCIAL_IMPLEMENTATION_VERSION = "stage05-r1-a1"
FINANCIAL_IMPLEMENTATION_IDENTITY: Mapping[str, Any] = MappingProxyType({
    "stage": "05", "stage_revision": 1, "attempt": 1,
    "agent": "financial", "implementation_version": FINANCIAL_IMPLEMENTATION_VERSION,
})

FINANCIAL_LIFECYCLE_BOUNDARIES = frozenset({
    "sec_submissions", "sec_companyfacts", "sec_filing_selection_and_text",
    "sec_restatement_gate", "yahoo_adjusted_history",
    "yahoo_current_only_fail_closed", "financial_mcp_ratio_dispatcher",
    "monthly_financial_adapter", "agent_extraction", "agent_calculation",
    "agent_checker", "pre_shared_state_gate",
})

_LIFECYCLE_ROUTE: Mapping[str, tuple[str, str]] = MappingProxyType({
    "sec_submissions": ("sec_edgar", "submissions_snapshot"),
    "sec_companyfacts": ("sec_edgar", "companyfacts_snapshot"),
    "sec_filing_selection_and_text": ("sec_edgar", "filing_selection_and_text"),
    "sec_restatement_gate": ("sec_edgar", "restatement_gate"),
    "yahoo_adjusted_history": ("yahoo_finance", "adjusted_history"),
    "yahoo_current_only_fail_closed": ("yahoo_finance", "current_only_fail_closed"),
    "financial_mcp_ratio_dispatcher": ("financial_mcp", "dispatch"),
    "monthly_financial_adapter": ("financial_adapter", "build_monthly_features"),
    "agent_extraction": ("financial_agent", "extract"),
    "agent_calculation": ("financial_agent", "calculate"),
    "agent_checker": ("financial_agent", "check"),
    "pre_shared_state_gate": ("financial_agent", "pre_shared_state"),
})
_LIFECYCLE_COUNT_KEYS = frozenset({
    "input_count", "result_count", "evidence_count", "accession_count",
    "eligible_count", "rejected_count", "missing_count", "tool_count",
    "error_count", "loop_count", "candidate_count",
})
_LIFECYCLE_OUTCOMES = frozenset({"success", "unavailable", "rejected", "degraded", "fail_closed"})
_LIFECYCLE_ERROR_CODES = frozenset({
    "CURRENT_ONLY_PROVIDER_REJECTED", "FINANCIAL_CALCULATION_FAIL_CLOSED",
    "FINANCIAL_CHECKER_API_ERROR", "FINANCIAL_CHECKER_PARSE_ERROR",
    "FINANCIAL_OPERATION_FAIL_CLOSED", "FINANCIAL_TOOL_DATA_UNAVAILABLE",
    "FINANCIAL_TOOL_FAIL_CLOSED", "SEC_COMPANYFACTS_UNAVAILABLE",
    "SEC_FILING_UNAVAILABLE", "SEC_SUBMISSIONS_UNAVAILABLE",
    "YAHOO_ADJUSTED_HISTORY_UNAVAILABLE",
})
_lifecycle_log = logging.getLogger("financial-lifecycle")


class FinancialDataUnavailable(DataUnavailableAsOfError):
    """A required financial observation is unavailable at the cutoff."""


def financial_implementation_identity() -> dict[str, Any]:
    return dict(FINANCIAL_IMPLEMENTATION_IDENTITY)


def require_financial_context(
    forecast_context: ForecastContext | Mapping[str, Any] | None,
    run_id: str | None,
    *, ticker: str | None = None,
) -> tuple[ForecastContext, str]:
    """Validate execution identity before any provider or calculation call."""
    if forecast_context is None:
        raise ValueError("forecast_context is required for every financial execution")
    context = (
        forecast_context if isinstance(forecast_context, ForecastContext)
        else ForecastContext.from_dict(dict(forecast_context))
    )
    identity = str(run_id or "").strip()
    if not identity or len(identity) > 160 or "\n" in identity or "\r" in identity:
        raise ValueError("run_id must be compact non-empty text")
    if ticker is not None and str(ticker).strip().upper() != context.ticker:
        raise ValueError("ticker does not match ForecastContext.ticker")
    return context, identity


def financial_trace_metadata(
    context: ForecastContext | Mapping[str, Any], run_id: str, **measurements: Any,
) -> dict[str, Any]:
    return dict(forecast_trace_metadata(
        context, run_id=run_id, stage="05", stage_revision=1, attempt=1,
        agent="financial", **measurements,
    ), implementation_version=FINANCIAL_IMPLEMENTATION_VERSION)


def _bounded_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not math.isfinite(numeric):
        return 0
    return max(0, min(int(numeric), 1_000_000))


def financial_lifecycle_event(
    *, context: ForecastContext | Mapping[str, Any], run_id: str,
    boundary: str, started_at: float, outcome: str,
    counts: Mapping[str, Any] | None = None, missing: bool = False,
    degraded: bool = False, error_code: str | None = None,
) -> dict[str, Any]:
    """Emit one measured, compact and payload-free Stage05 lifecycle event."""
    if boundary not in FINANCIAL_LIFECYCLE_BOUNDARIES:
        raise ValueError("unknown financial lifecycle boundary")
    validated_context, identity = require_financial_context(context, run_id)
    provider, operation = _LIFECYCLE_ROUTE[boundary]
    safe_outcome = outcome if outcome in _LIFECYCLE_OUTCOMES else "fail_closed"
    event = financial_trace_metadata(validated_context, identity)
    event.update({
        "boundary": boundary, "provider": provider, "operation": operation,
        "latency_ms": elapsed_latency_ms(started_at), "outcome": safe_outcome,
        "missing": bool(missing), "degraded": bool(degraded),
    })
    for key, value in (counts or {}).items():
        if key in _LIFECYCLE_COUNT_KEYS:
            event[key] = _bounded_count(value)
    if error_code:
        event["error_code"] = (
            error_code if error_code in _LIFECYCLE_ERROR_CODES
            else "FINANCIAL_OPERATION_FAIL_CLOSED"
        )
    _lifecycle_log.info(
        "FINANCIAL_LIFECYCLE %s",
        json.dumps(event, sort_keys=True, separators=(",", ":")),
    )
    return event


def as_of_date(context: ForecastContext) -> date:
    return context.as_of.astimezone(timezone.utc).date()  # type: ignore[union-attr]


def require_eligible_filed_at(value: Any, context: ForecastContext) -> str:
    """Return canonical filed date, rejecting missing/future availability."""
    if value is None or not str(value).strip():
        raise FinancialDataUnavailable("filed_at is required for point-in-time financial data")
    raw = str(value).strip()
    try:
        filed = date.fromisoformat(raw[:10])
    except ValueError as exc:
        raise FinancialDataUnavailable("filed_at must be an ISO date") from exc
    if filed > as_of_date(context):
        raise TemporalLeakageError("filed_at is after ForecastContext.as_of")
    return filed.isoformat()


def deterministic_financial_reference(
    *, provider: str, ticker: str, accession_number: str | None = None,
    metric: str | None = None, period_end: str | None = None,
    filed_at: str | None = None, dataset_version: str,
) -> str:
    payload = {
        "provider": provider, "ticker": ticker, "accession_number": accession_number,
        "metric": metric, "period_end": period_end, "filed_at": filed_at,
        "dataset_version": dataset_version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"financial:{digest}"


def historical_current_data_unavailable(
    *, context: ForecastContext, run_id: str, provider: str, fields: list[str] | None = None,
) -> dict[str, Any]:
    """Explicit missing result; historical execution never falls back to current snapshots."""
    started_at = time.perf_counter()
    result = {
        "ticker": context.ticker, "status": "DATA_UNAVAILABLE",
        "error_code": "CURRENT_ONLY_PROVIDER_REJECTED",
        "missing_fields": sorted(fields or []), "degraded": True,
        "provider": provider, "as_of": context.to_dict()["as_of"],
        "dataset_version": context.dataset_version, "run_id": run_id,
        "provenance": {"provider": provider, "current_fallback_used": False},
        "error": None,
    }
    financial_lifecycle_event(
        context=context, run_id=run_id, boundary="yahoo_current_only_fail_closed",
        started_at=started_at, outcome="rejected", counts={"missing_count": len(fields or [])},
        missing=True, degraded=True, error_code="CURRENT_ONLY_PROVIDER_REJECTED",
    )
    return result


def canonical_filed_timestamp(value: str) -> str:
    return datetime.combine(date.fromisoformat(value[:10]), datetime.min.time(), tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "FINANCIAL_CONTRACT_VERSION", "FINANCIAL_IMPLEMENTATION_IDENTITY",
    "FINANCIAL_LIFECYCLE_BOUNDARIES",
    "FINANCIAL_IMPLEMENTATION_VERSION", "FinancialDataUnavailable",
    "as_of_date", "canonical_filed_timestamp", "deterministic_financial_reference",
    "financial_implementation_identity", "financial_lifecycle_event",
    "financial_trace_metadata",
    "historical_current_data_unavailable", "require_eligible_filed_at",
    "require_financial_context",
]
