

from __future__ import annotations

import uuid

from core.error_handler import (
    DataUnavailableAsOfError,
    TemporalLeakageError,
    VersionMismatchError,
)


# ─────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────

class AlphaAgentError(Exception):
   
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
   
    http_status = 400
    code = "VALIDATION_ERROR"


# ─────────────────────────────────────────────────────────────
# Agent errors  (500)
# ─────────────────────────────────────────────────────────────

class AgentError(AlphaAgentError):
   
    http_status = 500
    code = "AGENT_ERROR"


class AgentTimeoutError(AlphaAgentError):
   
    http_status = 504
    code = "AGENT_TIMEOUT"


# ─────────────────────────────────────────────────────────────
# Memory errors  (500)
# ─────────────────────────────────────────────────────────────

class MemoryError(AlphaAgentError):
    
    http_status = 500
    code = "MEMORY_ERROR"


# ─────────────────────────────────────────────────────────────
# External service errors  (503)
# ─────────────────────────────────────────────────────────────

class ExternalServiceError(AlphaAgentError):
    
    http_status = 503
    code = "EXTERNAL_SERVICE_ERROR"


class ConfigurationError(AlphaAgentError):
    
    http_status = 500
    code = "CONFIGURATION_ERROR"


class TemporalBoundaryError(AlphaAgentError):
    """Public, sanitized representation of a point-in-time violation."""

    http_status = 422
    code = "TEMPORAL_BOUNDARY_ERROR"


class DataUnavailableError(AlphaAgentError):
    """Public representation of unavailable as-of data."""

    http_status = 422
    code = "DATA_UNAVAILABLE_AS_OF"


class RuntimeVersionError(AlphaAgentError):
    """Public representation of an incompatible versioned input."""

    http_status = 409
    code = "VERSION_MISMATCH"


def map_core_exception(exc: BaseException) -> AlphaAgentError:
    """Map core contract failures without exposing timestamps or versions.

    The original exception is intentionally retained only in server logs.  Its
    text can contain provider payloads, exact cutoffs, or internal versions and
    must never become an API response detail.
    """
    if isinstance(exc, TemporalLeakageError):
        return TemporalBoundaryError(
            "The request violates the point-in-time data boundary."
        )
    if isinstance(exc, DataUnavailableAsOfError):
        return DataUnavailableError(
            "Required data is unavailable at the requested cutoff."
        )
    if isinstance(exc, VersionMismatchError):
        return RuntimeVersionError(
            "The request is incompatible with the active runtime versions."
        )
    raise TypeError("unsupported core exception type")


CORE_EXCEPTION_TYPES = (
    TemporalLeakageError,
    DataUnavailableAsOfError,
    VersionMismatchError,
)
