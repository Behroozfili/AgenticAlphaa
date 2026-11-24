"""
Tests for: api/config.py
Phase: 1 — Pure-Logic / Zero-Mock Foundations

KNOWN BLOCKING BUG (see Phase 0): api/config.py does
    from api.core.exceptions import ConfigurationError
but api/core/exceptions.py does NOT define a `ConfigurationError` class
(only AlphaAgentError, ValidationError, AgentError, AgentTimeoutError,
MemoryError, ExternalServiceError exist). As written, `import api.config`
raises ImportError immediately — the whole module is currently unimportable.

TC-CFG0 below documents and proves this bug. ALL OTHER tests in this file
require the bug to be fixed first (i.e. ConfigurationError must exist in
exceptions.py) — they are written against the intended/expected behavior
and will fail to even collect until then.

Mocking strategy: we use monkeypatch.setenv/delenv to control environment
variables, and get_settings.cache_clear() (per the function's own docstring)
to force re-evaluation of the lru_cache'd singleton between tests.
"""
import importlib
import sys

import pytest


# ---------------------------------------------------------------------------
# TC-CFG0 — Documents the blocking import bug (Phase 0)
# ---------------------------------------------------------------------------

def test_import_now_succeeds_bug_fixed():
    """
    FIXED: ConfigurationError was added to api/core/exceptions.py (Phase 0
    bug). api.config now imports successfully. This test confirms the fix;
    it replaces the old test_import_currently_fails_... test that documented
    the bug while it was still present.
    """
    sys.modules.pop("api.config", None)
    module = importlib.import_module("api.config")
    assert hasattr(module, "ConfigurationError")
    assert hasattr(module, "get_settings")


# ---------------------------------------------------------------------------
# Everything below now runs for real — Phase 0's ConfigurationError bug
# has been fixed in api/core/exceptions.py, so api.config imports cleanly.
# ---------------------------------------------------------------------------

@pytest.fixture
def config_module(monkeypatch):
    """
    Import (or re-import) the config module fresh, with required env vars
    set, so Settings() construction and validate_settings() succeed by
    default. Each test can override specific env vars before requesting
    this fixture's `get_settings`/`validate_settings`/`Settings` exports.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-supabase-key")

    sys.modules.pop("api.config", None)
    module = importlib.import_module("api.config")
    module.get_settings.cache_clear()
    yield module
    module.get_settings.cache_clear()
    sys.modules.pop("api.config", None)


class TestSettingsDefaults:
    def test_default_anthropic_model(self, config_module):
        config_module.get_settings.cache_clear()
        s = config_module.get_settings()
        assert s.ANTHROPIC_MODEL == "claude-haiku-4-5"

    def test_default_max_routing_loops(self, config_module):
        s = config_module.get_settings()
        assert s.MAX_ROUTING_LOOPS == 8

    def test_default_app_env_is_development(self, config_module):
        s = config_module.get_settings()
        assert s.APP_ENV == "development"

    def test_default_request_timeout(self, config_module):
        s = config_module.get_settings()
        assert s.REQUEST_TIMEOUT_S == 300

    def test_default_user_id(self, config_module):
        s = config_module.get_settings()
        assert s.DEFAULT_USER_ID == "anonymous"

    def test_app_env_rejects_invalid_literal(self, monkeypatch, config_module):
        monkeypatch.setenv("APP_ENV", "staging")  # not in Literal["development","production"]
        config_module.get_settings.cache_clear()
        with pytest.raises(Exception):  # pydantic ValidationError
            config_module.get_settings()

    def test_monthly_runtime_versions_are_explicit(self, config_module):
        s = config_module.get_settings()
        assert s.RUNTIME_VERSION
        assert s.FORECAST_SCHEMA_VERSION
        assert s.LIVE_DATASET_VERSION != s.HISTORICAL_DATASET_VERSION
        assert s.FORECAST_HORIZON_TRADING_DAYS == 21

    def test_observability_is_disabled_by_default(self, config_module):
        s = config_module.Settings(
            _env_file=None,
            OBSERVABILITY_ENABLED=False,
            SENTRY_ENABLED=True,
            LANGSMITH_ENABLED=True,
            SENTRY_DSN="https://example.invalid/1",
            LANGSMITH_API_KEY="secret",
        )
        assert s.sentry_active is False
        assert s.langsmith_active is False

    def test_observability_requires_flags_and_credentials(self, config_module):
        s = config_module.Settings(
            _env_file=None,
            OBSERVABILITY_ENABLED=True,
            SENTRY_ENABLED=True,
            LANGSMITH_ENABLED=True,
            SENTRY_DSN="https://example.invalid/1",
            LANGSMITH_API_KEY="secret",
        )
        assert s.sentry_active is True
        assert s.langsmith_active is True

    def test_live_and_historical_contexts_use_separate_dataset_versions(self, config_module):
        s = config_module.Settings(_env_file=None)
        live = s.forecast_context(
            ticker="msft", as_of="2024-05-31T20:00:00Z", analysis_mode="LIVE"
        )
        historical = s.forecast_context(
            ticker="MSFT",
            as_of="2024-05-31T20:00:00Z",
            analysis_mode="HISTORICAL_BACKTEST",
        )
        assert live.dataset_version == s.LIVE_DATASET_VERSION
        assert historical.dataset_version == s.HISTORICAL_DATASET_VERSION
        assert live.target == historical.target
        assert live.horizon_trading_days == 21

    def test_context_reuses_holiday_weekend_and_21_session_contract(self, config_module):
        context = config_module.Settings(_env_file=None).forecast_context(
            ticker="MSFT", as_of="2024-11-29T18:00:00Z"
        )
        # November month-end is the early-close Friday after Thanksgiving;
        # weekends and Christmas are excluded from the 21-session target.
        assert context.target.isoformat() == "2024-12-31"


class TestSupabaseKeyAlias:
    def test_supabase_key_reads_from_service_role_alias(self, monkeypatch, config_module):
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "alias-value")
        config_module.get_settings.cache_clear()
        s = config_module.get_settings()
        assert s.SUPABASE_KEY == "alias-value"


class TestGetSettingsCaching:
    def test_get_settings_returns_same_instance_across_calls(self, config_module):
        s1 = config_module.get_settings()
        s2 = config_module.get_settings()
        assert s1 is s2

    def test_cache_clear_allows_picking_up_new_env_vars(self, monkeypatch, config_module):
        monkeypatch.setenv("APP_ENV", "production")
        config_module.get_settings.cache_clear()
        s = config_module.get_settings()
        assert s.APP_ENV == "production"

    def test_settings_alias_matches_get_settings_at_import_time(self, config_module):
        # module-level `settings` was created once at import time
        assert config_module.settings.ANTHROPIC_MODEL == config_module.get_settings().ANTHROPIC_MODEL


class TestValidateSettings:
    def test_passes_when_all_required_vars_present(self, config_module):
        config_module.get_settings.cache_clear()
        config_module.validate_settings()  # should not raise

    def test_raises_configuration_error_when_anthropic_key_missing(self, monkeypatch, config_module):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        config_module.get_settings.cache_clear()
        with pytest.raises(config_module.ConfigurationError) as exc_info:
            config_module.validate_settings()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_raises_configuration_error_when_supabase_url_missing(self, monkeypatch, config_module):
        monkeypatch.setenv("SUPABASE_URL", "")
        config_module.get_settings.cache_clear()
        with pytest.raises(config_module.ConfigurationError) as exc_info:
            config_module.validate_settings()
        assert "SUPABASE_URL" in str(exc_info.value)

    def test_error_message_lists_all_missing_vars_together(self, monkeypatch, config_module):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setenv("SUPABASE_URL", "")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
        config_module.get_settings.cache_clear()
        with pytest.raises(config_module.ConfigurationError) as exc_info:
            config_module.validate_settings()
        msg = str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "SUPABASE_URL" in msg
        assert "SUPABASE_KEY" in msg
