"""
core/observability.py

"""

from __future__ import annotations

import math
import logging
import os
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any, TypedDict

log = logging.getLogger("core.observability")
import sentry_sdk
from langsmith import traceable


_sentry_enabled: bool = False
_langsmith_enabled: bool = False
EXTERNAL_EXPORTERS_REQUEST_SAFE: bool = True
EXTERNAL_EXPORTER_BLOCK_CODE = "EXPORTER_NOT_CONFIGURED"
_langsmith_client = None


class ForecastTraceMetadata(TypedDict, total=False):
    """Compact allowlisted metadata for monthly-forecast spans."""

    run_id: str
    ticker: str
    as_of: str
    target: str
    horizon: int
    analysis_mode: str
    dataset_version: str
    feature_version: str
    model_version: str
    schema_version: str
    calendar_version: str
    stage: str
    stage_revision: int
    attempt: int
    agent: str
    latency_ms: float
    evidence_count: int
    degraded_flags: list[str]


_FORECAST_CONTEXT_FIELDS = {
    "ticker": "ticker",
    "as_of": "as_of",
    "target": "target",
    "horizon_trading_days": "horizon",
    "analysis_mode": "analysis_mode",
    "dataset_version": "dataset_version",
    "feature_version": "feature_version",
    "model_version": "model_version",
    "schema_version": "schema_version",
    "calendar_version": "calendar_version",
}
_MAX_METADATA_TEXT = 160
_MAX_DEGRADED_FLAGS = 20


def _context_fields(context: Mapping[str, Any] | object) -> Mapping[str, Any]:
    """Read only known context fields; never serialize a full State object."""
    if isinstance(context, Mapping):
        return context
    serializer = getattr(context, "to_dict", None)
    if callable(serializer):
        payload = serializer()
        if isinstance(payload, Mapping):
            return payload
    return {
        field: getattr(context, field)
        for field in _FORECAST_CONTEXT_FIELDS
        if hasattr(context, field)
    }


def _safe_metadata_text(value: Any, *, field_name: str) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (date, datetime)):
        value = value.isoformat()
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text, a date, datetime, or enum")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > _MAX_METADATA_TEXT or "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} is not compact trace metadata")
    return value


def elapsed_latency_ms(started_at: float) -> float:
    """Return non-negative monotonic elapsed time without emitting a span."""
    if isinstance(started_at, bool) or not isinstance(started_at, (int, float)):
        raise TypeError("started_at must be a time.perf_counter() value")
    elapsed = (time.perf_counter() - float(started_at)) * 1000.0
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("started_at must be a valid earlier monotonic timestamp")
    return round(elapsed, 3)


def forecast_trace_metadata(
    context: Mapping[str, Any] | object,
    *,
    run_id: str | None = None,
    stage: str | None = None,
    stage_revision: int | None = None,
    attempt: int | None = None,
    agent: str | None = None,
    latency_ms: float | None = None,
    evidence_count: int | None = None,
    degraded_flags: Sequence[str] | None = None,
) -> ForecastTraceMetadata:
    """Build safe forecast metadata from an allowlist, never from full State.

    Optional operational fields are omitted when unavailable. Callers provide
    already measured latency; this helper does not fabricate measurements or
    emit data to an external tracing service.
    """
    source = _context_fields(context)
    metadata: ForecastTraceMetadata = {}
    for source_name, output_name in _FORECAST_CONTEXT_FIELDS.items():
        value = source.get(source_name)
        if value is None:
            continue
        if source_name == "horizon_trading_days":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("horizon_trading_days must be a positive integer")
            metadata[output_name] = value  # type: ignore[literal-required]
        else:
            metadata[output_name] = _safe_metadata_text(  # type: ignore[literal-required]
                value, field_name=source_name
            )

    for field_name, value in (
        ("run_id", run_id), ("stage", stage), ("agent", agent),
    ):
        if value is not None:
            metadata[field_name] = _safe_metadata_text(  # type: ignore[literal-required]
                value, field_name=field_name
            )

    for field_name, value in (
        ("stage_revision", stage_revision), ("attempt", attempt),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
            metadata[field_name] = value  # type: ignore[literal-required]

    if latency_ms is not None:
        if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
            raise TypeError("latency_ms must be numeric")
        if not math.isfinite(float(latency_ms)) or latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")
        metadata["latency_ms"] = round(float(latency_ms), 3)

    if evidence_count is not None:
        if isinstance(evidence_count, bool) or not isinstance(evidence_count, int) or evidence_count < 0:
            raise ValueError("evidence_count must be a non-negative integer")
        metadata["evidence_count"] = evidence_count

    if degraded_flags is not None:
        if isinstance(degraded_flags, (str, bytes)):
            raise TypeError("degraded_flags must be a sequence of compact labels")
        flags = [
            _safe_metadata_text(flag, field_name="degraded_flag")
            for flag in degraded_flags
        ]
        if len(flags) > _MAX_DEGRADED_FLAGS:
            raise ValueError(f"degraded_flags supports at most {_MAX_DEGRADED_FLAGS} labels")
        metadata["degraded_flags"] = flags

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
#                                        Sentry
# ─────────────────────────────────────────────────────────────────────────────

def init_sentry(
    app_env: str | None = None,
    *,
    observability_enabled: bool = False,
    exporter_enabled: bool = False,
    dsn: str | None = None,
) -> bool:
    """Enable explicit error export behind configuration and a payload filter."""
    global _sentry_enabled
    _sentry_enabled = False
    configured_dsn = (dsn if dsn is not None else os.getenv("SENTRY_DSN", "")).strip()
    if not (observability_enabled and exporter_enabled and configured_dsn):
        return False
    try:
        from core.telemetry_export import sentry_event
        sentry_sdk.init(
            dsn=configured_dsn, environment=app_env or os.getenv("APP_ENV", "development"),
            send_default_pii=False, include_local_variables=False,
            default_integrations=False, auto_enabling_integrations=False,
            auto_session_tracking=False, send_client_reports=False,
            traces_sample_rate=0.0, before_send=sentry_event,
            before_breadcrumb=lambda event, hint: None,
        )
        _sentry_enabled = True
        log.info("Sentry export enabled with operational payload filtering")
    except Exception:
        log.warning("observability_initializer_failed code=SENTRY_INIT_FAILED")
    return _sentry_enabled


def sentry_enabled() -> bool:
    """Return True if Sentry was successfully initialised in this process."""
    return _sentry_enabled


# ─────────────────────────────────────────────────────────────────────────────
#                                        langsmith
# ────────────────────────────────────────────────────────────────


def init_langsmith(
    *,
    observability_enabled: bool = False,
    exporter_enabled: bool = False,
    api_key: str | None = None,
    project: str | None = None,
) -> bool:
    """Enable tracing through the filtered client; credentials never enter traces."""
    global _langsmith_enabled, _langsmith_client
    _langsmith_enabled = False
    from langsmith.run_trees import configure
    previous = _langsmith_client
    _langsmith_client = None
    configure(enabled=False, client=None)
    try:
        if previous is not None:
            previous.flush(timeout=5)
        # Global SDK configuration enables this process only. Child processes
        # must initialize their own filtered client instead of inheriting raw tracing.
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ.pop("LANGCHAIN_API_KEY", None)
        key = (api_key if api_key is not None else os.getenv("LANGSMITH_API_KEY", "")).strip()
        if not (observability_enabled and exporter_enabled and key):
            return False
        from core.telemetry_export import OperationalClient
        _langsmith_client = OperationalClient(
            api_key=key, omit_traced_runtime_info=True, timeout_ms=10000,
        )
        configure(client=_langsmith_client, enabled=True,
                  project_name=project or os.getenv("LANGSMITH_PROJECT", "alpha-agent-node"))
        _langsmith_enabled = True
        log.info("LangSmith export enabled with operational payload filtering")
    except Exception:
        configure(enabled=False, client=None)
        _langsmith_client = None
        log.warning("observability_initializer_failed code=LANGSMITH_INIT_FAILED")
    return _langsmith_enabled


def telemetry_client():
    if not _langsmith_enabled or _langsmith_client is None:
        raise RuntimeError("LangSmith export is not configured or initialization failed.")
    return _langsmith_client


def flush_observability():
    """Drain pending telemetry without failing application shutdown."""
    try:
        if _sentry_enabled:
            sentry_sdk.flush(timeout=5)
        if _langsmith_enabled and _langsmith_client is not None:
            _langsmith_client.flush(timeout=5)
    except Exception:
        log.warning("observability_flush_failed")


def configure_process_observability():
    """Apply the same explicit settings to standalone MCP tool processes."""
    from api.config import Settings
    settings = Settings()
    init_sentry(app_env=settings.APP_ENV, observability_enabled=settings.OBSERVABILITY_ENABLED,
                exporter_enabled=settings.SENTRY_ENABLED, dsn=settings.SENTRY_DSN)
    init_langsmith(observability_enabled=settings.OBSERVABILITY_ENABLED,
                  exporter_enabled=settings.LANGSMITH_ENABLED,
                  api_key=settings.LANGSMITH_API_KEY, project=settings.LANGSMITH_PROJECT)


def langsmith_enabled() -> bool:
   
    return _langsmith_enabled

# ─────────────────────────────────────────────────────────────────────────────
#                            MCP tool-call tracing
# ─────────────────────────────────────────────────────────────────────────────

def traced_call_tool(session, tool_name, arguments):
    """
    Record a single MCP tool call as an independent LangSmith span
    with run_type="tool".

    A raw session.call_tool is not traced by LangSmith: it runs inside the
    mcp library layer and creates no span. This wrapper runs inside the agent
    process (which has an active trace context), so the span is correctly
    attached as a child of the parent node.

    The span name is set to the actual tool name (e.g. tool_get_cik) so the
    validation script can locate it in actual_tool_calls.

    arguments is passed positionally so this works with both call patterns
    used across the agents (positional and keyword arguments=).
    """

    @traceable(run_type="tool", name=tool_name)
    async def _inner():
        return await session.call_tool(tool_name, arguments)

    return _inner()
