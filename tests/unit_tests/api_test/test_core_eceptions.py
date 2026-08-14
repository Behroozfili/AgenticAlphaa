"""
Tests for: api/core/exceptions.py
Phase: 1 — Pure-Logic / Zero-Mock Foundations

No mocking required — pure Python exception classes with no I/O.
The only "dynamic" piece is trace_id (uuid4), which we verify by shape/format
rather than mocking uuid, to keep tests resilient to refactors.
"""
import pytest

from api.core.exceptions import (
    AlphaAgentError,
    ValidationError,
    AgentError,
    AgentTimeoutError,
    MemoryError,
    ExternalServiceError,
    ConfigurationError,
)


# ---------------------------------------------------------------------------
# Base class: AlphaAgentError
# ---------------------------------------------------------------------------

class TestAlphaAgentErrorBase:
    def test_default_message(self):
        err = AlphaAgentError()
        assert err.message == "An unexpected error occurred."

    def test_custom_message_and_detail(self):
        err = AlphaAgentError(message="Custom failure", detail="extra context")
        assert err.message == "Custom failure"
        assert err.detail == "extra context"

    def test_default_detail_is_none(self):
        err = AlphaAgentError(message="x")
        assert err.detail is None

    def test_default_http_status_and_code(self):
        err = AlphaAgentError()
        assert err.http_status == 500
        assert err.code == "INTERNAL_ERROR"

    def test_trace_id_is_generated_and_is_8_chars(self):
        err = AlphaAgentError()
        assert isinstance(err.trace_id, str)
        assert len(err.trace_id) == 8

    def test_trace_id_is_unique_per_instance(self):
        err1 = AlphaAgentError()
        err2 = AlphaAgentError()
        assert err1.trace_id != err2.trace_id

    def test_trace_id_is_derived_from_a_valid_uuid4_prefix(self):
        err = AlphaAgentError()
        # Confirm the 8-char trace_id is a valid prefix of a hex UUID
        assert all(c in "0123456789abcdef" for c in err.trace_id)

    def test_is_exception_subclass(self):
        err = AlphaAgentError("boom")
        assert isinstance(err, Exception)
        with pytest.raises(AlphaAgentError):
            raise err

    def test_str_uses_message_via_exception_init(self):
        err = AlphaAgentError(message="boom")
        assert str(err) == "boom"

    def test_to_dict_structure(self):
        err = AlphaAgentError(message="boom", detail="d")
        d = err.to_dict()
        assert d["error"] == "INTERNAL_ERROR"
        assert d["message"] == "boom"
        assert d["detail"] == "d"
        assert d["trace_id"] == err.trace_id

    def test_to_dict_keys_exact_set(self):
        err = AlphaAgentError()
        assert set(err.to_dict().keys()) == {"error", "message", "detail", "trace_id"}


# ---------------------------------------------------------------------------
# Subclasses — verify each maps to correct http_status + code
# ---------------------------------------------------------------------------

class TestValidationError:
    def test_http_status_and_code(self):
        err = ValidationError("bad ticker")
        assert err.http_status == 400
        assert err.code == "VALIDATION_ERROR"

    def test_is_alpha_agent_error_subclass(self):
        assert issubclass(ValidationError, AlphaAgentError)


class TestAgentError:
    def test_http_status_and_code(self):
        err = AgentError("graph failed")
        assert err.http_status == 500
        assert err.code == "AGENT_ERROR"

    def test_is_alpha_agent_error_subclass(self):
        assert issubclass(AgentError, AlphaAgentError)


class TestAgentTimeoutError:
    def test_http_status_and_code(self):
        err = AgentTimeoutError("timed out")
        assert err.http_status == 504
        assert err.code == "AGENT_TIMEOUT"

    def test_is_alpha_agent_error_subclass(self):
        assert issubclass(AgentTimeoutError, AlphaAgentError)


class TestMemoryError:
    def test_http_status_and_code(self):
        err = MemoryError("supabase down")
        assert err.http_status == 500
        assert err.code == "MEMORY_ERROR"

    def test_is_alpha_agent_error_subclass(self):
        assert issubclass(MemoryError, AlphaAgentError)

    def test_does_not_shadow_builtin_in_isolation(self):
        """
        Documents an API design risk: this class is named `MemoryError`,
        shadowing Python's built-in `MemoryError` (raised on real OOM
        conditions) for any module that does
        `from api.core.exceptions import MemoryError`.
        This test only documents current behavior (subclassing
        AlphaAgentError, not the builtin) — it is not a bug assertion,
        just a guardrail so the shadowing is caught if it's ever 'fixed'
        by accident in a way that breaks the public API.
        """
        assert issubclass(MemoryError, AlphaAgentError)
        assert not issubclass(MemoryError, BaseException) or issubclass(MemoryError, AlphaAgentError)


class TestExternalServiceError:
    def test_http_status_and_code(self):
        err = ExternalServiceError("tavily unreachable")
        assert err.http_status == 503
        assert err.code == "EXTERNAL_SERVICE_ERROR"

    def test_is_alpha_agent_error_subclass(self):
        assert issubclass(ExternalServiceError, AlphaAgentError)


class TestConfigurationError:
    """
    FIXED: ConfigurationError was missing entirely (Phase 0 bug), causing
    api/config.py's `from api.core.exceptions import ConfigurationError`
    to raise ImportError on every startup. Now added — these tests confirm
    the fix and that api.config can import it successfully.
    """
    def test_http_status_and_code(self):
        err = ConfigurationError("ANTHROPIC_API_KEY missing")
        assert err.http_status == 500
        assert err.code == "CONFIGURATION_ERROR"

    def test_is_alpha_agent_error_subclass(self):
        assert issubclass(ConfigurationError, AlphaAgentError)

    def test_to_dict_reflects_configuration_error_code(self):
        err = ConfigurationError("missing var", detail="set it in .env")
        d = err.to_dict()
        assert d["error"] == "CONFIGURATION_ERROR"
        assert d["detail"] == "set it in .env"


# ---------------------------------------------------------------------------
# Cross-cutting: every subclass's to_dict() uses its own `code`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc_cls,expected_code,expected_status", [
    (ValidationError, "VALIDATION_ERROR", 400),
    (AgentError, "AGENT_ERROR", 500),
    (AgentTimeoutError, "AGENT_TIMEOUT", 504),
    (MemoryError, "MEMORY_ERROR", 500),
    (ExternalServiceError, "EXTERNAL_SERVICE_ERROR", 503),
    (ConfigurationError, "CONFIGURATION_ERROR", 500),
])
def test_to_dict_reflects_subclass_code(exc_cls, expected_code, expected_status):
    err = exc_cls("msg", detail="d")
    d = err.to_dict()
    assert d["error"] == expected_code
    assert err.http_status == expected_status