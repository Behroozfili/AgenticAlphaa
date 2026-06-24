"""
Tests for: tools/financial_tools/financial_server.py
Phase: 2a — Financial Tools (MCP server delegation layer)

Scope decision: this file is a thin FastMCP wrapper around functions already
fully tested in test_yahoo_finance.py / test_sec_edgar_financial.py /
test_financial_ratio_calculator.py. We do NOT re-test the underlying ratio
math or HTTP calls here. We test only:
  1. That each @mcp.tool() function correctly delegates to the right
     underlying function with the right arguments (sample across all 3
     tool categories: Yahoo Finance, SEC EDGAR, Ratio Calculator).
  2. The `_sentry_tool` / `_sentry_capture` error-handling wrapper behavior,
     since that logic is unique to this file and not covered elsewhere.

IMPORTANT — accessing the underlying function: @mcp.tool() (FastMCP) returns
the original function unchanged (it registers it as a side effect), so
`tool_get_price_history` etc. remain directly callable Python functions in
tests — no special MCP client/transport setup is needed.
"""
from unittest.mock import patch, MagicMock
import pytest

from tools.financial_tools.financial_server import (
    tool_get_price_history,
    tool_get_financial_ratios,
    tool_get_cik,
    tool_calc_pe,
    tool_calc_composite_score,
    _sentry_tool,
    _sentry_capture,
)


# ---------------------------------------------------------------------------
# Delegation — Yahoo Finance category (sample)
# ---------------------------------------------------------------------------

class TestYahooFinanceDelegation:
    @patch("tools.financial_tools.financial_server.get_price_history")
    def test_tool_get_price_history_delegates_with_same_args(self, mock_fn):
        mock_fn.return_value = {"ticker": "NVDA", "records": [], "error": None}
        result = tool_get_price_history("NVDA", period="6mo")
        mock_fn.assert_called_once_with("NVDA", "6mo")
        assert result["ticker"] == "NVDA"

    @patch("tools.financial_tools.financial_server.get_financial_ratios")
    def test_tool_get_financial_ratios_uses_sentry_wrapper(self, mock_fn):
        mock_fn.return_value = {"ticker": "NVDA", "error": None}
        result = tool_get_financial_ratios("NVDA")
        mock_fn.assert_called_once_with("NVDA")
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Delegation — SEC EDGAR category (sample)
# ---------------------------------------------------------------------------

class TestSecEdgarDelegation:
    @patch("tools.financial_tools.financial_server.get_cik")
    def test_tool_get_cik_delegates(self, mock_fn):
        mock_fn.return_value = {"ticker": "NVDA", "cik": "0001045810", "error": None}
        result = tool_get_cik("NVDA")
        mock_fn.assert_called_once_with("NVDA")
        assert result["cik"] == "0001045810"


# ---------------------------------------------------------------------------
# Delegation — Ratio Calculator category (sample, incl. param remapping)
# ---------------------------------------------------------------------------

class TestRatioCalculatorDelegation:
    @patch("tools.financial_tools.financial_server.price_to_earnings")
    def test_tool_calc_pe_delegates(self, mock_fn):
        mock_fn.return_value = {"pe_ratio": 20.0, "interpretation": "fairly_valued"}
        result = tool_calc_pe(price=200, eps=10)
        mock_fn.assert_called_once_with(200, 10)
        assert result["pe_ratio"] == 20.0

    @patch("tools.financial_tools.financial_server.composite_financial_score")
    def test_tool_calc_composite_score_remaps_current_ratio_val_param(self, mock_fn):
        """
        IMPORTANT WIRING DETAIL: the MCP tool's parameter is named
        `current_ratio_val` (to avoid shadowing the imported `current_ratio`
        function) but it must be passed to composite_financial_score's
        `current_ratio` keyword. This test locks in that remapping so a
        future refactor doesn't silently break it.
        """
        mock_fn.return_value = {"score": 70.0, "grade": "B", "sub_scores": {}, "missing_inputs": []}
        tool_calc_composite_score(current_ratio_val=2.0)
        _, kwargs = mock_fn.call_args
        assert kwargs["current_ratio"] == 2.0


# ---------------------------------------------------------------------------
# _sentry_tool / _sentry_capture — error handling wrapper
# ---------------------------------------------------------------------------

class TestSentryToolWrapper:
    def test_successful_call_returns_function_result_unchanged(self):
        fn = MagicMock(return_value={"ok": True})
        result = _sentry_tool("some_tool", fn, "arg1", kw=2)
        fn.assert_called_once_with("arg1", kw=2)
        assert result == {"ok": True}

    def test_exception_is_caught_and_returns_error_dict(self):
        fn = MagicMock(side_effect=RuntimeError("boom"))
        result = _sentry_tool("some_tool", fn)
        assert result == {"error": "boom", "tool": "some_tool"}

    @patch("tools.financial_tools.financial_server.sentry_enabled", return_value=True)
    @patch("tools.financial_tools.financial_server._sentry_capture")
    def test_exception_triggers_sentry_capture_when_enabled(
        self, mock_capture, mock_enabled
    ):
        fn = MagicMock(side_effect=ValueError("bad input"))
        _sentry_tool("some_tool", fn)
        mock_capture.assert_called_once()
        assert mock_capture.call_args[0][0] == "some_tool"

    @patch("tools.financial_tools.financial_server.sentry_enabled", return_value=False)
    def test_sentry_capture_skipped_when_sentry_disabled(self, mock_enabled):
        # _sentry_capture itself should be a no-op when sentry_enabled() is False
        _sentry_capture("some_tool", RuntimeError("boom"))  # should not raise
        mock_enabled.assert_called_once()

    @patch("tools.financial_tools.financial_server.sentry_enabled", return_value=True)
    @patch("sentry_sdk.push_scope")
    @patch("sentry_sdk.capture_exception")
    def test_sentry_capture_tags_tool_and_server(
        self, mock_capture_exc, mock_push_scope, mock_enabled
    ):
        scope = MagicMock()
        mock_push_scope.return_value.__enter__.return_value = scope
        exc = RuntimeError("boom")

        _sentry_capture("tool_get_cik", exc)

        scope.set_tag.assert_any_call("tool", "tool_get_cik")
        scope.set_tag.assert_any_call("server", "financial-agent-mcp")
        mock_capture_exc.assert_called_once_with(exc)