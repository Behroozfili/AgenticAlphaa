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
import logging
import re
import sentry_sdk

from financial_test import STAGE5_RUN_ID, stage5_context
from core.forecast_contract import ForecastContext

from tools.financial_tools.financial_server import (
    tool_get_price_history,
    tool_get_financial_ratios,
    tool_get_cik,
    tool_calc_pe,
    tool_calc_gross_margin,
    tool_calc_net_debt_to_ebitda,
    tool_calc_dcf_scenarios,
    tool_calc_reverse_dcf,
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
        result = tool_get_price_history("NVDA", stage5_context(), STAGE5_RUN_ID, period="6mo")
        args, kwargs = mock_fn.call_args
        assert args[:2] == ("NVDA", "6mo")
        assert isinstance(kwargs["forecast_context"], ForecastContext)
        assert kwargs["run_id"] == STAGE5_RUN_ID
        assert result["ticker"] == "NVDA"

    @patch("tools.financial_tools.financial_server.get_financial_ratios")
    def test_tool_get_financial_ratios_uses_sentry_wrapper(self, mock_fn):
        mock_fn.return_value = {"ticker": "NVDA", "error": None}
        result = tool_get_financial_ratios("NVDA", stage5_context(), STAGE5_RUN_ID)
        args = mock_fn.call_args.args
        assert args[0] == "NVDA" and isinstance(args[1], ForecastContext) and args[2] == STAGE5_RUN_ID
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Delegation — SEC EDGAR category (sample)
# ---------------------------------------------------------------------------

class TestSecEdgarDelegation:
    @patch("tools.financial_tools.financial_server.get_cik")
    def test_tool_get_cik_delegates(self, mock_fn):
        mock_fn.return_value = {"ticker": "NVDA", "cik": "0001045810", "error": None}
        result = tool_get_cik("NVDA", stage5_context(), STAGE5_RUN_ID)
        args = mock_fn.call_args.args
        assert args[0] == "NVDA" and isinstance(args[1], ForecastContext) and args[2] == STAGE5_RUN_ID
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
        result = _sentry_tool("tool_calc_pe", fn, "arg1", kw=2)
        fn.assert_called_once_with("arg1", kw=2)
        assert result == {"ok": True}

    def test_exception_is_caught_and_returns_error_dict(self):
        fn = MagicMock(side_effect=RuntimeError("boom"))
        result = _sentry_tool("tool_calc_pe", fn)
        assert result == {
            "error_code": "FINANCIAL_TOOL_FAIL_CLOSED",
            "error": "Financial tool failed closed.", "tool": "tool_calc_pe",
        }

    @patch("tools.financial_tools.financial_server.sentry_enabled", return_value=True)
    @patch("tools.financial_tools.financial_server._sentry_capture")
    def test_exception_triggers_sentry_capture_when_enabled(
        self, mock_capture, mock_enabled
    ):
        fn = MagicMock(side_effect=ValueError("bad input"))
        _sentry_tool("tool_calc_pe", fn)
        mock_capture.assert_called_once()
        assert mock_capture.call_args[0][0] == "tool_calc_pe"

    @patch("tools.financial_tools.financial_server.sentry_enabled", return_value=False)
    def test_sentry_capture_skipped_when_sentry_disabled(self, mock_enabled):
        # _sentry_capture itself should be a no-op when sentry_enabled() is False
        _sentry_capture("tool_calc_pe", RuntimeError("boom"))  # should not raise
        mock_enabled.assert_called_once()

    @patch("tools.financial_tools.financial_server.sentry_enabled", return_value=True)
    @patch("sentry_sdk.push_scope")
    @patch("sentry_sdk.capture_message")
    def test_sentry_capture_tags_tool_and_server(
        self, mock_capture_message, mock_push_scope, mock_enabled
    ):
        scope = MagicMock()
        mock_push_scope.return_value.__enter__.return_value = scope
        exc = RuntimeError("boom")

        _sentry_capture("tool_get_cik", exc)

        scope.set_tag.assert_any_call("tool", "tool_get_cik")
        scope.set_tag.assert_any_call("server", "financial-agent-mcp")
        mock_capture_message.assert_called_once_with("FINANCIAL_TOOL_FAIL_CLOSED", level="error")


def test_financial_server_logs_never_persist_raw_scalars_or_private_failures(caplog):
    marker = "PRIVATE_SENTINEL"
    raw_values = (987654321.125, 876543210.25, 765432109.375, 654321098.5)
    with caplog.at_level(logging.INFO, logger="financial-mcp-server"):
        tool_calc_pe(raw_values[0], raw_values[1])
        tool_calc_gross_margin(raw_values[2], raw_values[3])
        tool_calc_net_debt_to_ebitda(raw_values[0], raw_values[1], raw_values[2])
        tool_calc_pe(float("inf"), float("nan"))
        tool_calc_dcf_scenarios(
            fcf_base=raw_values[0], beta=1.2, base_growth_rate_pct=8.0,
            shares_outstanding=raw_values[1],
        )
        tool_calc_reverse_dcf(
            fcf_base=raw_values[2], beta=1.1,
            target_enterprise_value=raw_values[3],
        )
        failure = _sentry_tool(
            marker, MagicMock(side_effect=RuntimeError(marker)), marker,
        )
    rendered = caplog.text
    assert marker not in rendered
    for value in raw_values:
        assert str(value) not in rendered
    assert re.search(r"\b(?:inf|nan)\b", rendered.lower()) is None
    assert failure["error_code"] == "FINANCIAL_TOOL_FAIL_CLOSED"
    assert failure["tool"] == "financial_tool_unknown"
    assert "FINANCIAL_TOOL_START" in rendered
    assert "input_count=" in rendered
