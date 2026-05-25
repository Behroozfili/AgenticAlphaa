"""
api/core/exceptions.py
----------------------
Custom exception hierarchy for Alpha-Agent Node.

Every exception maps to a specific HTTP status code and carries
a machine-readable `code` field so the frontend can handle errors
programmatically — not just by reading human text.

Usage
-----
    from api.core.exceptions import AgentError, MemoryError

    raise AgentError(
        message="ManagerAgent timed out",
        detail="Graph execution exceeded 300 s",
    )

The ExceptionMiddleware in main.py catches all subclasses of
AlphaAgentError and converts them to structured JSON responses.
"""

from __future__ import annotations

import uuid


# ─────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────

class AlphaAgentError(Exception):
    """
    Base class for all Alpha-Agent Node exceptions.

    Attributes
    ----------
    message : str
        Human-readable description (shown to the caller).
    code : str
        Machine-readable error code (e.g. "AGENT_ERROR").
    http_status : int
        HTTP status code to return.
    detail : str | None
        Optional extra context (logged server-side, may be hidden from users).
    trace_id : str
        Auto-generated 8-char unique ID per exception instance.
        Appears in logs AND in the JSON response so the caller can
        send it to support and you can grep it instantly in production.

        Example log grep:
            grep "trace_id=a3f2c1b9" logs/api.log
    """

    http_status: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An unexpected error occurred.",
        detail:  str | None = None,
    ) -> None:
        super().__init__(message)
        self.message  = message
        self.detail   = detail
        self.trace_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict:
        return {
            "error":    self.code,
            "message":  self.message,
            "detail":   self.detail,
            "trace_id": self.trace_id,
        }


# ─────────────────────────────────────────────────────────────
# Validation errors  (400)
# ─────────────────────────────────────────────────────────────

class ValidationError(AlphaAgentError):
    """
    Request payload failed validation.
    Raised when a field value is invalid beyond Pydantic's built-in checks.

    Example: ticker format is wrong, query is too vague.
    """
    http_status = 400
    code = "VALIDATION_ERROR"


# ─────────────────────────────────────────────────────────────
# Agent errors  (500)
# ─────────────────────────────────────────────────────────────

class AgentError(AlphaAgentError):
    """
    ManagerAgent.run() raised an unhandled exception.

    Example: LangGraph graph execution failed mid-pipeline.
    """
    http_status = 500
    code = "AGENT_ERROR"


class AgentTimeoutError(AlphaAgentError):
    """
    ManagerAgent.run() exceeded REQUEST_TIMEOUT_S.

    Example: A specialist agent hung waiting for an external API.
    """
    http_status = 504
    code = "AGENT_TIMEOUT"


# ─────────────────────────────────────────────────────────────
# Memory errors  (500)
# ─────────────────────────────────────────────────────────────

class MemoryError(AlphaAgentError):
    """
    ManagerMemory failed to load or persist.

    Example: Supabase returned an error during SELECT or UPSERT.
    """
    http_status = 500
    code = "MEMORY_ERROR"


# ─────────────────────────────────────────────────────────────
# External service errors  (503)
# ─────────────────────────────────────────────────────────────

class ExternalServiceError(AlphaAgentError):
    """
    A downstream service (Supabase, Tavily, Yahoo Finance…) is unreachable.

    Example: Supabase connection refused at startup.
    """
    http_status = 503
    code = "EXTERNAL_SERVICE_ERROR"
