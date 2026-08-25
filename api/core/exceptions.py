

from __future__ import annotations

import uuid


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