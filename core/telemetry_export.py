"""Outbound telemetry boundary: operational traces without raw user payloads."""
from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any

from langsmith import Client


def label(value: Any, fallback: str = "redacted") -> str:
    return value if isinstance(value, str) and re.fullmatch(r"[\w.:-]{1,160}", value) else fallback


def sentry_event(event, hint):
    """Rebuild events so request bodies, locals and exception messages never leave."""
    from core.observability import sentry_enabled
    if not sentry_enabled():
        return None
    result = {k: event[k] for k in ("event_id", "timestamp", "platform", "level", "environment", "release") if k in event}
    values = []
    for exc in event.get("exception", {}).get("values", []):
        frames = []
        for frame in exc.get("stacktrace", {}).get("frames", []):
            frames.append({"filename": PurePath(frame.get("filename", "unknown")).name,
                           "function": label(frame.get("function")),
                           "lineno": frame.get("lineno"), "in_app": frame.get("in_app", False)})
        values.append({"type": label(exc.get("type"), "Exception"),
                       "value": "Exception details omitted", "stacktrace": {"frames": frames}})
    if values:
        result["exception"] = {"values": values}
    else:
        message = event.get("message") or event.get("logentry", {}).get("message", "")
        result["message"] = message if isinstance(message, str) and re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", message) else "APPLICATION_EVENT"
    result["tags"] = {"component": label(event.get("tags", {}).get("component", "application"))}
    return result


_META = {"run_id", "ticker", "as_of", "target", "horizon", "analysis_mode", "dataset_version",
         "feature_version", "model_version", "schema_version", "calendar_version", "stage", "stage_revision",
         "attempt", "agent", "latency_ms", "evidence_count", "status", "error_code", "implementation_version"}


def compact_metadata(value):
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in _META and
            (isinstance(v, (int, float, bool)) or isinstance(v, str) and len(v) <= 160 and "\n" not in v)}


def trace_fields(fields):
    result = {k: fields[k] for k in ("id", "trace_id", "parent_run_id", "dotted_order", "start_time", "end_time",
                                   "session_name", "reference_example_id") if k in fields}
    if "name" in fields:
        result["name"] = label(fields["name"], "operation")
    if "run_type" in fields:
        result["run_type"] = fields["run_type"]
    result["extra"] = {"metadata": compact_metadata((fields.get("extra") or {}).get("metadata", {}))}
    if fields.get("error"):
        result["error"] = "OPERATION_FAILED"
    if "outputs" in fields:
        result["outputs"] = compact_metadata(fields.get("outputs"))
    return result


class OperationalClient(Client):
    """Preserve span identity, timing and compact metadata at the SDK boundary."""

    def create_run(self, name, inputs, run_type, *, project_name=None, **kwargs):
        from core.observability import langsmith_enabled
        if not langsmith_enabled():
            return None
        return super().create_run(label(name, "operation"), compact_metadata(inputs), run_type,
                                  project_name=project_name, **trace_fields(kwargs))

    def update_run(self, run_id, **kwargs):
        from core.observability import langsmith_enabled
        if not langsmith_enabled():
            return None
        return super().update_run(run_id, **trace_fields(kwargs))
