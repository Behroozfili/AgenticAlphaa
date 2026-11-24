"""
Tests for: tools/financial_tools/yahoo_finance.py
Phase: 2a — Financial Tools

Mocking strategy: yfinance.Ticker is the only external dependency (it wraps
Yahoo Finance HTTP calls internally). We mock yf.Ticker entirely so no real
network calls are made. We use unittest.mock.patch + MagicMock with
pandas-like DataFrame stand-ins for .history()/.financials/.quarterly_financials.
"""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from tools.financial_tools.yahoo_finance import (
    _safe_get,
    get_price_history,
    get_financial_ratios,
    get_balance_sheet_snapshot,
    get_revenue_growth,
    get_peer_comparison,
)
import tools.financial_tools.yahoo_finance as yahoo_module
from financial_test import STAGE5_RUN_ID, stage5_context

_real_get_price_history = get_price_history
_real_get_financial_ratios = get_financial_ratios
_real_get_balance_sheet_snapshot = get_balance_sheet_snapshot
_real_get_revenue_growth = get_revenue_growth
_real_get_peer_comparison = get_peer_comparison


@pytest.fixture(autouse=True)
def inject_stage5_identity(monkeypatch):
    module = __import__(__name__, fromlist=["x"])
    monkeypatch.setattr(module, "get_price_history", lambda ticker, *args, **kwargs: _real_get_price_history(
        ticker, *args, start=kwargs.pop("start", "2020-01-01"),
        forecast_context=kwargs.pop("forecast_context", stage5_context(str(ticker).upper())),
        run_id=kwargs.pop("run_id", STAGE5_RUN_ID), **kwargs,
    ))
    monkeypatch.setattr(module, "get_financial_ratios", lambda ticker, *args, **kwargs: _real_get_financial_ratios(
        ticker, *(args or (stage5_context(str(ticker).upper()), STAGE5_RUN_ID)), **kwargs,
    ))
    monkeypatch.setattr(module, "get_balance_sheet_snapshot", lambda ticker, *args, **kwargs: _real_get_balance_sheet_snapshot(
        ticker, *(args or (stage5_context(str(ticker).upper()), STAGE5_RUN_ID)), **kwargs,
    ))
    monkeypatch.setattr(module, "get_revenue_growth", lambda ticker, *args, **kwargs: _real_get_revenue_growth(
        ticker, *(args or (stage5_context(str(ticker).upper()), STAGE5_RUN_ID)), **kwargs,
    ))
    monkeypatch.setattr(module, "get_peer_comparison", lambda ticker, *args, **kwargs: _real_get_peer_comparison(
        ticker, *args, forecast_context=kwargs.pop("forecast_context", stage5_context(str(ticker).upper())),
        run_id=kwargs.pop("run_id", STAGE5_RUN_ID), **kwargs,
    ))


# ---------------------------------------------------------------------------
# _safe_get
# ---------------------------------------------------------------------------

class TestSafeGet:
    def test_returns_value_when_present(self):
        assert _safe_get({"a": 1}, "a") == 1

    def test_returns_default_when_missing(self):
        assert _safe_get({}, "a", default="x") == "x"

    def test_returns_default_when_value_is_none(self):
        assert _safe_get({"a": None}, "a", default="x") == "x"


# ---------------------------------------------------------------------------
# get_price_history
# ---------------------------------------------------------------------------

class TestGetPriceHistory:
    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_stage5_exact_adjusted_boundary(self, mock_ticker_cls):
        frame = pd.DataFrame({
            "Open": [149.0, 151.0], "High": [151.0, 153.0],
            "Low": [148.0, 150.0], "Close": [150.0, 152.0],
            "Volume": [10, 20],
        }, index=pd.to_datetime(["2026-06-29", "2026-06-30"]))
        mock_ticker_cls.return_value.history.return_value = frame
        result = _real_get_price_history(
            "NVDA", start="2026-06-29", end="2026-06-30",
            forecast_context=stage5_context(), run_id=STAGE5_RUN_ID,
        )
        mock_ticker_cls.return_value.history.assert_called_once_with(
            start="2026-06-29", end="2026-07-01", auto_adjust=True, actions=False,
        )
        assert [row["adjusted_close"] for row in result["records"]] == [150.0, 152.0]
        assert result["adjustment_mode"] == "auto_adjust_total_return"

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_historical_current_snapshot_fails_before_provider(self, mock_ticker_cls):
        result = _real_get_financial_ratios(
            "NVDA", stage5_context("NVDA", "HISTORICAL_BACKTEST"), STAGE5_RUN_ID,
        )
        assert result["status"] == "DATA_UNAVAILABLE"
        assert result["error_code"] == "CURRENT_ONLY_PROVIDER_REJECTED"
        assert result["provenance"]["current_fallback_used"] is False
        mock_ticker_cls.assert_not_called()
    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_happy_path_returns_records(self, mock_ticker_cls):
        df = pd.DataFrame({
            "Open":   [100.0, 101.0],
            "High":   [105.0, 106.0],
            "Low":    [99.0, 100.0],
            "Close":  [104.0, 105.0],
            "Volume": [1000, 2000],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        mock_ticker_cls.return_value.history.return_value = df

        result = get_price_history("nvda", period="1y")

        assert result["ticker"] == "NVDA"
        assert result["error"] is None
        assert len(result["records"]) == 2
        assert result["records"][0]["close"] == 104.0
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-01-02"

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_empty_history_returns_error(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        result = get_price_history("BADTICKER")

        assert result["records"] == []
        assert result["error"] == "No data returned for this ticker/period."

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_exception_is_caught_and_returned_as_error(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = ConnectionError("network down")

        result = get_price_history("NVDA")

        assert result["records"] == []
        assert result["error_code"] == "YAHOO_ADJUSTED_HISTORY_UNAVAILABLE"
        assert "network down" not in result["error"]

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_ticker_uppercased_in_output(self, mock_ticker_cls):
        df = pd.DataFrame({
            "Open": [1.0], "High": [1.0], "Low": [1.0],
            "Close": [1.0], "Volume": [1],
        }, index=pd.to_datetime(["2024-01-01"]))
        mock_ticker_cls.return_value.history.return_value = df

        result = get_price_history("aapl")
        assert result["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# get_financial_ratios
# ---------------------------------------------------------------------------

class TestGetFinancialRatios:
    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    @patch("tools.financial_tools.yahoo_finance.get_balance_sheet_snapshot")
    def test_lease_liabilities_falls_back_to_balance_sheet(self, mock_snapshot, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {}
        mock_snapshot.return_value = {"capital_lease_obligations": 123.0}

        result = get_financial_ratios("NVDA")

        assert result["lease_liabilities"] == 123.0

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    @patch("tools.financial_tools.yahoo_finance.get_balance_sheet_snapshot")
    def test_info_lease_liabilities_takes_precedence(self, mock_snapshot, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {"capitalLeaseObligations": 456.0}
        mock_snapshot.return_value = {"capital_lease_obligations": 123.0}

        result = get_financial_ratios("NVDA")

        assert result["lease_liabilities"] == 456.0
        mock_snapshot.assert_not_called()

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_happy_path_maps_all_fields(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "longName": "NVIDIA Corp",
            "sector": "Technology",
            "industry": "Semiconductors",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 65.0,
            "forwardPE": 45.0,
            "pegRatio": 1.5,
            "priceToBook": 50.0,
            "priceToSalesTrailing12Months": 30.0,
            "enterpriseValue": 3_100_000_000_000,
            "enterpriseToEbitda": 50.0,
            "trailingEps": 1.5,
            "forwardEps": 2.0,
            "dividendYield": 0.0003,
            "beta": 1.7,
            "fiftyTwoWeekHigh": 150.0,
            "fiftyTwoWeekLow": 60.0,
            "currentPrice": 130.0,
        }

        result = get_financial_ratios("nvda")

        assert result["ticker"] == "NVDA"
        assert result["company_name"] == "NVIDIA Corp"
        assert result["pe_ratio"] == 65.0
        assert result["error"] is None

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_recovery_fields_are_normalized_with_period_metadata(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {
            "debtToEquity": 66.0,
            "mostRecentQuarter": 1_767_139_200,
            "freeCashflow": 500.0,
            "operatingCashflow": 700.0,
            "totalRevenue": 2_000.0,
        }

        result = get_financial_ratios("MSFT")

        assert result["de_ratio"] == 0.66
        assert result["financial_data_period_end"] == "2025-12-31"
        assert result["free_cash_flow"] == 500.0
        assert result["operating_cash_flow"] == 700.0
        assert result["revenue_ttm"] == 2_000.0

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_missing_fields_default_to_none_or_na(self, mock_ticker_cls):
        mock_ticker_cls.return_value.info = {}

        result = get_financial_ratios("XYZ")

        assert result["company_name"] == "N/A"
        assert result["sector"] == "N/A"
        assert result["market_cap"] is None
        assert result["pe_ratio"] is None

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_exception_returns_error_dict(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("rate limited")

        result = get_financial_ratios("NVDA")

        assert result["error_code"] == "YAHOO_CURRENT_SNAPSHOT_UNAVAILABLE"
        assert "rate limited" not in result["error"]
        # NOTE: on the error path this function only returns {"ticker", "error"} —
        # no other keys — which differs from the happy-path schema. Callers
        # must check `error` before reading other fields.
        assert set(result.keys()) == {"ticker", "error_code", "error"}


class TestGetBalanceSheetSnapshot:
    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_extracts_capital_lease_obligations(self, mock_ticker_cls):
        period = pd.Timestamp("2025-01-31")
        mock_ticker_cls.return_value.balance_sheet = pd.DataFrame(
            {period: {"Capital Lease Obligations": 123.0}}
        )
        mock_ticker_cls.return_value.financials = pd.DataFrame()

        result = get_balance_sheet_snapshot("NVDA")

        assert result["capital_lease_obligations"] == 123.0


# ---------------------------------------------------------------------------
# get_revenue_growth
# ---------------------------------------------------------------------------

class TestGetRevenueGrowth:
    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_annual_revenue_growth_calculated(self, mock_ticker_cls):
        # Columns are most-recent-first per the function's own comment.
        col_2024 = pd.Timestamp("2024-12-31")
        col_2023 = pd.Timestamp("2023-12-31")
        financials = pd.DataFrame(
            {col_2024: [120.0, 20.0], col_2023: [100.0, 10.0]},
            index=["Total Revenue", "Net Income"],
        )
        mock_instance = mock_ticker_cls.return_value
        mock_instance.info = {"revenueGrowth": 0.2}
        mock_instance.financials = financials
        mock_instance.quarterly_financials = pd.DataFrame()

        result = get_revenue_growth("NVDA")

        assert result["error"] is None
        assert result["annual_revenue"][0]["revenue"] == 120.0
        assert result["annual_revenue"][0]["yoy_growth"] == 0.2  # (120-100)/100
        assert result["annual_net_income"][0]["yoy_growth"] == 1.0  # (20-10)/10

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_oldest_period_has_no_prior_year_growth_is_none(self, mock_ticker_cls):
        col_only = pd.Timestamp("2024-12-31")
        financials = pd.DataFrame(
            {col_only: [100.0, 10.0]}, index=["Total Revenue", "Net Income"]
        )
        mock_instance = mock_ticker_cls.return_value
        mock_instance.info = {}
        mock_instance.financials = financials
        mock_instance.quarterly_financials = pd.DataFrame()

        result = get_revenue_growth("NVDA")
        assert result["annual_revenue"][0]["yoy_growth"] is None

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_empty_financials_returns_empty_lists_not_crash(self, mock_ticker_cls):
        mock_instance = mock_ticker_cls.return_value
        mock_instance.info = {}
        mock_instance.financials = pd.DataFrame()
        mock_instance.quarterly_financials = pd.DataFrame()

        result = get_revenue_growth("NVDA")

        assert result["annual_revenue"] == []
        assert result["annual_net_income"] == []
        assert result["error"] is None

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    def test_exception_is_caught(self, mock_ticker_cls):
        mock_ticker_cls.side_effect = RuntimeError("boom")
        result = get_revenue_growth("NVDA")
        assert result["error_code"] == "YAHOO_GROWTH_DATA_UNAVAILABLE"
        assert "boom" not in result["error"]


# ---------------------------------------------------------------------------
# get_peer_comparison
# ---------------------------------------------------------------------------

class TestGetPeerComparison:
    @patch("tools.financial_tools.yahoo_finance.get_financial_ratios")
    def test_computes_peer_averages(self, mock_get_ratios):
        def fake_ratios(ticker, *args, **kwargs):
            data = {
                "NVDA": {"pe_ratio": 60, "forward_pe": 40, "price_to_book": 50,
                         "price_to_sales": 30, "ev_to_ebitda": 50, "beta": 1.7, "error": None},
                "AMD":  {"pe_ratio": 40, "forward_pe": 30, "price_to_book": 10,
                         "price_to_sales": 10, "ev_to_ebitda": 20, "beta": 1.9, "error": None},
            }
            return data[ticker]
        mock_get_ratios.side_effect = fake_ratios

        result = get_peer_comparison("NVDA", peers=["AMD"])

        assert result["error"] is None
        assert result["summary"]["avg_pe_ratio"] == 40.0
        assert len(result["peers"]) == 1

    @patch("tools.financial_tools.yahoo_finance.get_financial_ratios")
    def test_primary_ticker_error_short_circuits(self, mock_get_ratios):
        mock_get_ratios.return_value = {"ticker": "BAD", "error": "not found"}

        result = get_peer_comparison("BAD", peers=["AMD"])

        assert result["error"] == "not found"
        assert result["peers"] == []
        mock_get_ratios.assert_called_once()  # peers should never be fetched

    @patch("tools.financial_tools.yahoo_finance.yf.Ticker")
    @patch("tools.financial_tools.yahoo_finance.get_financial_ratios")
    def test_no_peers_provided_falls_back_to_empty_list(self, mock_get_ratios, mock_ticker_cls):
        mock_get_ratios.return_value = {"ticker": "NVDA", "pe_ratio": 60, "error": None}
        mock_ticker_cls.return_value.recommendations = MagicMock()

        result = get_peer_comparison("NVDA", peers=None)

        assert result["peers"] == []
        assert result["error"] is None

    @patch("tools.financial_tools.yahoo_finance.get_financial_ratios")
    def test_peer_fields_with_none_excluded_from_average(self, mock_get_ratios):
        def fake_ratios(ticker, *args, **kwargs):
            if ticker == "NVDA":
                return {"pe_ratio": 60, "forward_pe": None, "price_to_book": None,
                         "price_to_sales": None, "ev_to_ebitda": None, "beta": None, "error": None}
            return {"pe_ratio": None, "forward_pe": None, "price_to_book": None,
                    "price_to_sales": None, "ev_to_ebitda": None, "beta": None, "error": None}
        mock_get_ratios.side_effect = fake_ratios

        result = get_peer_comparison("NVDA", peers=["AMD"])
        assert result["summary"]["avg_pe_ratio"] is None  # no peer had a value

    def test_exception_returns_error_dict(self):
        with patch(
            "tools.financial_tools.yahoo_finance.get_financial_ratios",
            side_effect=RuntimeError("boom"),
        ):
            result = get_peer_comparison("NVDA", peers=["AMD"])
            assert result["error_code"] == "YAHOO_PEER_DATA_UNAVAILABLE"
            assert "boom" not in result["error"]
            assert result["primary"] == {}
