"""
yahoo_finance.py
----------------
Tool for extracting quantitative market data using the yfinance library.

This module provides functions to retrieve:
- Historical price data
- Key financial ratios (P/E, EPS, etc.)
- Revenue and earnings growth metrics
- Peer comparison data

All functions return structured dictionaries suitable for downstream
consumption by the Financial Analyst Agent or MCP server.
"""

import math
import time
from datetime import date, datetime, timedelta, timezone
from collections.abc import Mapping

from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
from typing import Any

from core.forecast_contract import AnalysisMode, ForecastContext
from tools.financial_tools import (
    deterministic_financial_reference, financial_lifecycle_event,
    historical_current_data_unavailable,
    require_financial_context,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """
    Convert a pandas/numpy cell value to a plain Python float, collapsing
    NaN/Inf to None instead of a real (but JSON-illegal) NaN float.

    Root cause this guards against: pandas represents a *missing* cell
    within an otherwise-present row/column as ``numpy.nan`` — NOT as
    ``None``. ``float(numpy.nan)`` succeeds and returns a real Python
    ``nan``, so a bare ``float(cell) if row is not None else None`` check
    (which only verifies the row exists, not that this specific cell has
    data) silently lets NaN through as if it were a valid number.

    Two concrete failures this caused downstream, both traced from a real
    run:
      1. A NaN "oldest" revenue value passed every truthiness/None check
         (``nan is not None`` is True, ``bool(nan)`` is True) but failed
         every ordering comparison (``nan > 0`` is False), so the CAGR
         guard's ``oldest > 0`` check silently evaluated False and CAGR
         was reported as "unavailable" even though 4 valid years of
         revenue history were actually present.
      2. The raw NaN value survived into financial_metrics_summary and
         broke JSON persistence to Supabase with
         "Out of range float values are not JSON compliant: nan", since
         strict JSON (RFC 8259) has no NaN token.

    Returns
    -------
    float | None
        The value as a plain float, or None if it is missing, NaN, or
        +/-Infinity (all of which are equally "no data" from a financial-
        reporting standpoint and equally illegal in strict JSON).
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _safe_get(info: dict, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from the yfinance info dictionary.

    Parameters
    ----------
    info : dict
        The raw info dict returned by yf.Ticker().info.
    key : str
        The key to look up.
    default : Any
        Value returned when the key is missing or its value is None.

    Returns
    -------
    Any
        The retrieved value, or *default* if absent.
    """
    value = info.get(key)
    return value if value is not None else default


def _safe_period_end(value: Any) -> str | None:
    """Normalize Yahoo's mostRecentQuarter value to an ISO period-end date."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()
        except (OSError, OverflowError, ValueError):
            return None
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except (AttributeError, ValueError):
            return None
    text = str(value).strip()
    return text[:10] if len(text) >= 10 else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_price_history(
    ticker: str, period: str = "1y", *, start: str | None = None,
    end: str | None = None, as_of: str | None = None,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Retrieve historical OHLCV price data for a given ticker.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. "NVDA", "AAPL").
    period : str
        Time period for historical data. Valid values: "1d", "5d", "1mo",
        "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".
        Defaults to "1y".

    Returns
    -------
    dict
        A dictionary with the following keys:
        - "ticker"      (str)  : The requested ticker symbol.
        - "period"      (str)  : The requested period.
        - "records"     (list) : List of OHLCV dicts keyed by date string.
        - "start_date"  (str)  : Earliest date in the result set (ISO-8601).
        - "end_date"    (str)  : Latest date in the result set (ISO-8601).
        - "error"       (str | None) : Error message if the request failed.

    Examples
    --------
    >>> result = get_price_history("NVDA", period="6mo")
    >>> result["ticker"]
    'NVDA'
    """
    context, identity = require_financial_context(forecast_context, run_id, ticker=ticker)
    started_at = time.perf_counter()

    def finish(result: dict[str, Any], outcome: str, error_code: str | None = None) -> dict[str, Any]:
        records = result.get("records") if isinstance(result.get("records"), list) else []
        missing = not bool(records)
        financial_lifecycle_event(
            context=context, run_id=identity, boundary="yahoo_adjusted_history",
            started_at=started_at, outcome=outcome,
            counts={
                "result_count": len(records), "eligible_count": len(records),
                "missing_count": 1 if missing else 0,
            },
            missing=missing, degraded=missing, error_code=error_code,
        )
        return result

    cutoff = context.as_of.date()  # type: ignore[union-attr]
    requested_as_of = date.fromisoformat(as_of[:10]) if as_of else cutoff
    if requested_as_of != cutoff:
        raise ValueError("as_of must equal ForecastContext.as_of")
    requested_start = date.fromisoformat(start[:10]) if start else cutoff - timedelta(days=context.lookback_days)
    requested_end = date.fromisoformat(end[:10]) if end else cutoff
    if requested_start > requested_end:
        raise ValueError("start must be on or before end")
    if requested_end > cutoff:
        raise ValueError("end must be on or before ForecastContext.as_of")
    try:
        stock = yf.Ticker(ticker)
        # yfinance's end is exclusive. Request one extra calendar day, while
        # validating every returned row against the inclusive exact boundary.
        hist = stock.history(
            start=requested_start.isoformat(),
            end=(requested_end + timedelta(days=1)).isoformat(),
            auto_adjust=True, actions=False,
        )

        if hist.empty:
            return finish({"ticker": ticker, "period": period, "records": [],
                           "start_date": None, "end_date": None,
                           "error": "No data returned for this ticker/period."}, "unavailable")

        records = []
        for observed_date, row in hist.iterrows():
            records.append({
                "date":   observed_date.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "adjusted_close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })

        records = [row for row in records if requested_start.isoformat() <= row["date"] <= requested_end.isoformat()]
        if not records:
            return finish({
                "ticker": ticker.upper(), "period": period, "records": [],
                "start_date": requested_start.isoformat(), "end_date": requested_end.isoformat(),
                "as_of": context.to_dict()["as_of"], "adjustment_mode": "auto_adjust_total_return",
                "dataset_version": context.dataset_version, "run_id": identity,
                "status": "DATA_UNAVAILABLE", "missing": True,
                "error": "No adjusted data returned inside the exact requested boundary.",
            }, "unavailable")

        return finish({
            "ticker":     ticker.upper(),
            "period":     period,
            "records":    records,
            "start_date": records[0]["date"],
            "end_date":   records[-1]["date"],
            "requested_start": requested_start.isoformat(),
            "requested_end": requested_end.isoformat(),
            "as_of": context.to_dict()["as_of"],
            "adjustment_mode": "auto_adjust_total_return",
            "provider": "yahoo_finance", "dataset_version": context.dataset_version,
            "run_id": identity, "missing": False,
            "evidence_id": deterministic_financial_reference(
                provider="yahoo_adjusted_history", ticker=context.ticker,
                period_end=requested_end.isoformat(), filed_at=requested_end.isoformat(),
                dataset_version=context.dataset_version,
            ),
            "error":      None,
        }, "success")

    except Exception:
        return finish({
            "ticker": ticker, "period": period, "records": [],
            "start_date": None, "end_date": None,
            "error_code": "YAHOO_ADJUSTED_HISTORY_UNAVAILABLE",
            "error": "Yahoo adjusted history is unavailable.",
        }, "fail_closed", "YAHOO_ADJUSTED_HISTORY_UNAVAILABLE")


def get_financial_ratios(
    ticker: str,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Retrieve key financial ratios and valuation metrics for a ticker.

    Fetches data directly from yfinance's info endpoint and normalises
    the most commonly used ratios into a clean output structure.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. "MSFT", "TSLA").

    Returns
    -------
    dict
        A dictionary containing:
        - "ticker"              (str)
        - "company_name"        (str)
        - "sector"              (str)
        - "industry"            (str)
        - "market_cap"          (float | None)  : In USD.
        - "pe_ratio"            (float | None)  : Trailing P/E.
        - "forward_pe"          (float | None)  : Forward P/E.
        - "peg_ratio"           (float | None)  : PEG ratio.
        - "price_to_book"       (float | None)  : P/B ratio.
        - "price_to_sales"      (float | None)  : P/S (TTM).
        - "enterprise_value"    (float | None)  : EV in USD.
        - "ev_to_ebitda"        (float | None)
        - "eps_trailing"        (float | None)  : EPS (TTM).
        - "eps_forward"         (float | None)  : Forward EPS estimate.
        - "dividend_yield"      (float | None)  : As a decimal (e.g. 0.012).
        - "beta"                (float | None)
        - "current_ratio"       (float | None)  : Current assets / current liabilities.
        - "quick_ratio"         (float | None)  : (Current assets − inventory) / current liabilities.
        - "52w_high"            (float | None)
        - "52w_low"             (float | None)
        - "current_price"       (float | None)
        - "total_debt"          (float | None)  : Short+long-term debt, USD.
        - "total_cash"          (float | None)  : Cash & equivalents, USD.
        - "ebitda"              (float | None)  : USD (TTM).
        - "free_cash_flow"      (float | None)  : Operating CF − capex, USD (TTM).
        - "operating_cash_flow" (float | None)  : USD (TTM).
        - "diluted_shares"       (float | None)  : Best available current diluted-share proxy.
        - "error"               (str | None)

    Examples
    --------
    >>> ratios = get_financial_ratios("AAPL")
    >>> ratios["pe_ratio"]
    28.5
    """
    context, identity = require_financial_context(forecast_context, run_id, ticker=ticker)
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        return historical_current_data_unavailable(
            context=context, run_id=identity, provider="yahoo_info",
            fields=["current_price", "market_cap", "valuation_ratios", "ttm_metrics"],
        )
    try:
        info = yf.Ticker(ticker).info
        yahoo_de_pct = _safe_float(info.get("debtToEquity"))
        lease_liabilities = info.get("capitalLeaseObligations")
        if not lease_liabilities:
            balance_sheet_snapshot = get_balance_sheet_snapshot(ticker, context, identity)
            lease_liabilities = (
                info.get("capitalLeaseObligations")
                or balance_sheet_snapshot.get("capital_lease_obligations")
            )

        return {
            "ticker":           ticker.upper(),
            "company_name":     _safe_get(info, "longName", "N/A"),
            "sector":           _safe_get(info, "sector", "N/A"),
            "industry":         _safe_get(info, "industry", "N/A"),
            "market_cap":       _safe_get(info, "marketCap"),
            "pe_ratio":         _safe_get(info, "trailingPE"),
            "forward_pe":       _safe_get(info, "forwardPE"),
            "peg_ratio":        _safe_get(info, "pegRatio"),
            "price_to_book":    _safe_get(info, "priceToBook"),
            "price_to_sales":   _safe_get(info, "priceToSalesTrailing12Months"),
            "enterprise_value": _safe_get(info, "enterpriseValue"),
            "ev_to_ebitda":     _safe_get(info, "enterpriseToEbitda"),
            "eps_trailing":     _safe_get(info, "trailingEps"),
            "eps_forward":      _safe_get(info, "forwardEps"),
            "dividend_yield":   _safe_get(info, "dividendYield"),
            "beta":             _safe_get(info, "beta"),
            # Liquidity ratios — present in yfinance's financialData module.
            # Without these the composite score can never receive a
            # current_ratio input and always flags it as missing.
            "current_ratio":    _safe_get(info, "currentRatio"),
            "quick_ratio":      _safe_get(info, "quickRatio"),
            # Yahoo reports debtToEquity in percentage points (e.g. 66.0
            # means a 0.66 ratio). Normalize it to the canonical ratio.
            "de_ratio":         yahoo_de_pct / 100.0 if yahoo_de_pct is not None else None,
            "financial_data_period_end": _safe_period_end(info.get("mostRecentQuarter")),
            "52w_high":         _safe_get(info, "fiftyTwoWeekHigh"),
            "52w_low":          _safe_get(info, "fiftyTwoWeekLow"),
            "current_price":    _safe_get(info, "currentPrice"),
            # Added for net_debt_to_ebitda / fcf_margin / fcf_yield /
            # operating_cash_flow_ratio (financial_ratio_calculator.py) --
            # all already present in yfinance's .info, just weren't being
            # extracted before these ratios existed.
            "total_debt":            _safe_get(info, "totalDebt"),
            "total_cash":            _safe_get(info, "totalCash"),
            "ebitda":                _safe_get(info, "ebitda"),
            "free_cash_flow":        _safe_get(info, "freeCashflow"),
            "operating_cash_flow":   _safe_get(info, "operatingCashflow"),
            "revenue_ttm":           _safe_get(info, "totalRevenue"),
            "diluted_shares":        _safe_get(info, "sharesOutstanding"),
            "lease_liabilities":     lease_liabilities,
            "minority_interest":     _safe_get(info, "minorityInterest"),
            "interest_expense":      _safe_get(info, "interestExpense"),
            "effective_tax_rate":    _safe_get(info, "effectiveTaxRate"),
            "as_of": context.to_dict()["as_of"], "dataset_version": context.dataset_version,
            "run_id": identity, "provider": "yahoo_info", "current_fallback_used": False,
            "error":            None,
        }

    except Exception:
        return {"ticker": ticker, "error_code": "YAHOO_CURRENT_SNAPSHOT_UNAVAILABLE", "error": "Yahoo current snapshot is unavailable."}


def _first_available_row(df, candidates: list[str]) -> float | None:
    """
    Try each row-label candidate in order against a yfinance financial-
    statement dataframe (balance_sheet / financials / cashflow) and return
    the most recent column's value for the first one that exists.

    Why candidates, not one fixed name: yfinance has renamed several
    standard row labels across versions (e.g. "Total Liab" ->
    "Total Liabilities Net Minority Interest"), and which exact label a
    given ticker's data uses can still vary. Trying several known aliases
    in priority order is far more robust than hard-coding one string that
    silently returns nothing the moment yfinance's naming shifts again.
    """
    if df is None or df.empty:
        return None
    most_recent_col = df.columns[0]  # yfinance orders newest-first
    for name in candidates:
        if name in df.index:
            return _safe_float(df.loc[name, most_recent_col])
    return None


def get_balance_sheet_snapshot(
    ticker: str,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Retrieve a single most-recent-period snapshot of balance-sheet and
    EBIT figures needed for the "Phase 2" ratios in
    financial_ratio_calculator.py: return_on_invested_capital,
    return_on_capital_employed, debt_ratio, equity_ratio, altman_z_score.

    Unlike get_financial_ratios() (which reads yfinance's flat `.info`
    dict), this reads the structured `.balance_sheet` / `.financials`
    dataframes, because total_assets / total_liabilities / total_equity /
    current_assets / current_liabilities / retained_earnings / EBIT are
    NOT present in `.info` at all -- they only exist in the full
    financial-statement tables.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol (e.g. "MSFT", "TSLA").

    Returns
    -------
    dict
        - "ticker"               (str)
        - "period_end"           (str | None) : ISO date of the most
          recent balance-sheet column used, e.g. "2026-03-31". Always
          check this is recent enough for your purpose -- yfinance may
          only have annual data for smaller/less-covered tickers, in
          which case this could be up to a year stale.
        - "total_assets"         (float | None)
        - "total_liabilities"    (float | None)
        - "total_equity"         (float | None)
        - "current_assets"       (float | None)
        - "current_liabilities"  (float | None)
        - "working_capital"      (float | None) : current_assets − current_liabilities.
        - "retained_earnings"    (float | None)
        - "capital_lease_obligations" (float | None)
        - "ebit"                 (float | None) : from the income
          statement (most recent period, may be a different (annual vs.
          quarterly-summed) period than the balance-sheet snapshot above
          -- both are yfinance's "most recent available", not guaranteed
          to be the exact same fiscal period; fine for the ratios this
          feeds, which mix a flow (EBIT) with a point-in-time balance
          either way, same as how these ratios are computed everywhere).
        - "error"                (str | None)

    Examples
    --------
    >>> snap = get_balance_sheet_snapshot("MSFT")
    >>> snap["total_assets"]
    620000000000.0
    """
    context, identity = require_financial_context(forecast_context, run_id, ticker=ticker)
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        return historical_current_data_unavailable(
            context=context, run_id=identity, provider="yahoo_balance_sheet_current",
            fields=["balance_sheet_snapshot", "ebit_snapshot"],
        )
    try:
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
        fin = stock.financials

        period_end = None
        if bs is not None and not bs.empty:
            col = bs.columns[0]
            period_end = col.date().isoformat() if hasattr(col, "date") else str(col)

        total_assets = _first_available_row(bs, ["Total Assets"])
        total_liabilities = _first_available_row(
            bs, ["Total Liabilities Net Minority Interest", "Total Liab", "Total Liabilities"]
        )
        total_equity = _first_available_row(
            bs, ["Total Equity Gross Minority Interest", "Stockholders Equity",
                 "Total Stockholder Equity", "Common Stock Equity"]
        )
        current_assets = _first_available_row(bs, ["Current Assets"])
        current_liabilities = _first_available_row(bs, ["Current Liabilities"])
        retained_earnings = _first_available_row(bs, ["Retained Earnings"])
        capital_lease_obligations = _first_available_row(
            bs, ["Capital Lease Obligations"]
        )

        ebit = _first_available_row(fin, ["EBIT", "Operating Income"])

        working_capital = (
            current_assets - current_liabilities
            if current_assets is not None and current_liabilities is not None
            else None
        )

        return {
            "ticker":              ticker.upper(),
            "period_end":          period_end,
            "total_assets":        total_assets,
            "total_liabilities":   total_liabilities,
            "total_equity":        total_equity,
            "current_assets":      current_assets,
            "current_liabilities": current_liabilities,
            "working_capital":     working_capital,
            "retained_earnings":   retained_earnings,
            "capital_lease_obligations": capital_lease_obligations,
            "ebit":                ebit,
            "as_of": context.to_dict()["as_of"], "dataset_version": context.dataset_version,
            "run_id": identity, "provider": "yahoo_balance_sheet_current",
            "current_fallback_used": False, "error": None,
        }

    except Exception:
        return {"ticker": ticker, "error_code": "YAHOO_BALANCE_SHEET_UNAVAILABLE", "error": "Yahoo balance-sheet snapshot is unavailable."}


def _fiscal_quarter_label(period_end_year: int, period_end_month: int, fye_month: int | None) -> str:
    """
    Build a company-accurate "FYyyyy-Qn" label for a quarterly period, using
    the company's OWN fiscal-year-end month (fye_month) rather than assuming
    a calendar-year quarter (Jan-Mar=Q1, ...).

    Naive calendar-quarter labeling silently mislabels any company whose
    fiscal year doesn't end in December — e.g. it called Microsoft's
    quarter ended March 31 "2026-Q1" when Microsoft itself calls that
    quarter "Q3 FY2026" (Microsoft's fiscal year starts in July). That
    produced a real, visible bug: a generated report cited "Q1 FY2026" net
    margin in one paragraph and "Q3 FY2026" Azure revenue (quoted directly
    from the 10-Q, which naturally uses the company's own fiscal label) in
    another — two mentions of the SAME quarter that looked like two
    different quarters to a reader.

    Falls back to a plain calendar-quarter label if fye_month is
    unavailable (e.g. annual `financials` was empty), which is the same
    fallback behavior as before this fix — no worse than the prior
    behavior, just no longer the default for the common case.
    """
    if fye_month is None:
        q_num = (period_end_month - 1) // 3 + 1
        return f"{period_end_year}-Q{q_num}"
    fy_start_month = (fye_month % 12) + 1
    months_since_fy_start = (period_end_month - fy_start_month) % 12
    fiscal_quarter = months_since_fy_start // 3 + 1
    fiscal_year = (
        period_end_year if period_end_month <= fye_month else period_end_year + 1
    )
    return f"FY{fiscal_year}-Q{fiscal_quarter}"


def get_revenue_growth(
    ticker: str,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Retrieve annual and quarterly revenue / earnings growth metrics.

    Pulls income statement data from yfinance and calculates year-over-year
    (YoY) growth rates for revenue and net income.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.

    Returns
    -------
    dict
        A dictionary with:
        - "ticker"              (str)
        - "annual_revenue"      (list[dict]) : [{year, revenue, yoy_growth}]
        - "quarterly_revenue"   (list[dict]) : [{quarter, revenue, yoy_growth}]
        - "annual_net_income"   (list[dict]) : [{year, net_income, yoy_growth}]
        - "quarterly_net_income" (list[dict]) : [{quarter, net_income, yoy_growth}]
        - "revenue_growth_ttm"  (float | None) : YoY growth for trailing 12 months.
        - "error"               (str | None)

    Notes
    -----
    Growth values are expressed as decimals (e.g. 1.22 = +122% growth).
    None is returned for periods where prior-year data is unavailable.
    """
    context, identity = require_financial_context(forecast_context, run_id, ticker=ticker)
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        return historical_current_data_unavailable(
            context=context, run_id=identity, provider="yahoo_statements_current",
            fields=["annual_revenue", "quarterly_revenue", "annual_net_income", "quarterly_net_income"],
        )
    try:
        stock    = yf.Ticker(ticker)
        info     = stock.info

        # --- Annual income statement ---
        financials = stock.financials  # columns = fiscal years (most recent first)
        annual_revenue    = []
        annual_net_income = []
        year_labeling_warnings: list[str] = []
        fye_month: int | None = None

        if financials is not None and not financials.empty:
            rev_row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
            inc_row = financials.loc["Net Income"]    if "Net Income"    in financials.index else None

            cols = list(financials.columns)  # most-recent first
            # Fiscal-year-end month, derived from the company's OWN annual
            # data — NOT hardcoded per-ticker. Used below to build correct
            # "FYyyyy-Qn" quarter labels (see _fiscal_quarter_label()).
            # Works for any fiscal calendar: a Jan-end company (NVIDIA), a
            # June-end company (Microsoft), a Sept-end company (Apple), or
            # a standard Dec-end company all derive correctly from this.
            if cols and hasattr(cols[0], "month"):
                fye_month = cols[0].month
            for i, col in enumerate(cols):
                year = col.year if hasattr(col, "year") else str(col)

                # Sanity check: annual columns should be ~1 fiscal year
                # apart. yfinance labels each column by the *calendar* year
                # of its period-end date, which usually (but not always,
                # depending on how a company names its own fiscal years —
                # e.g. a Jan-end company vs. one with a differently-named FY
                # convention) matches the company's own "Fiscal Year N"
                # label. This can't fully verify the label is *correct*
                # (there's no independent source here), but it does catch
                # gross mislabeling like a duplicated or skipped year,
                # which would otherwise silently corrupt every "annual:{year}"
                # period tag downstream (see financial_agent.py).
                if i + 1 < len(cols) and hasattr(col, "year") and hasattr(cols[i + 1], "year"):
                    gap = col.year - cols[i + 1].year
                    if gap not in (0, 1):
                        year_labeling_warnings.append(
                            f"columns[{i}]={col.date()} (FY{col.year}) and "
                            f"columns[{i+1}]={cols[i+1].date()} (FY{cols[i+1].year}) "
                            f"are {gap} calendar years apart, not the expected ~1 — "
                            f"annual_revenue/annual_net_income year labels may not "
                            f"line up with the company's own fiscal-year numbering."
                        )

                # Revenue — _safe_float() collapses pandas NaN cells (a
                # present row with a missing value for THIS specific column,
                # e.g. the oldest fiscal year often isn't fully populated)
                # to None. A bare `float(rev_row[col])` would instead return
                # a real `nan`, which passes every `is not None` check
                # downstream and breaks both CAGR calculation (nan fails
                # numeric comparisons) and JSON persistence (nan is not
                # valid JSON).
                rev = _safe_float(rev_row[col]) if rev_row is not None else None
                prev_rev = _safe_float(rev_row[cols[i + 1]]) if (rev_row is not None and i + 1 < len(cols)) else None
                # `is not None` (not truthiness) so a legitimate revenue of
                # exactly 0 isn't silently treated as "missing".
                rev_growth = (
                    round((rev - prev_rev) / abs(prev_rev), 4)
                    if (rev is not None and prev_rev is not None and prev_rev != 0)
                    else None
                )
                annual_revenue.append({"year": year, "revenue": rev, "yoy_growth": rev_growth})

                # Net Income
                ni = _safe_float(inc_row[col]) if inc_row is not None else None
                prev_ni = _safe_float(inc_row[cols[i + 1]]) if (inc_row is not None and i + 1 < len(cols)) else None
                ni_growth = (
                    round((ni - prev_ni) / abs(prev_ni), 4)
                    if (ni is not None and prev_ni is not None and prev_ni != 0)
                    else None
                )
                annual_net_income.append({"year": year, "net_income": ni, "yoy_growth": ni_growth})

        # --- Quarterly revenue ---
        quarterly = stock.quarterly_financials
        quarterly_revenue = []
        quarterly_net_income = []
        if quarterly is not None and not quarterly.empty and "Total Revenue" in quarterly.index:
            q_rev_row = quarterly.loc["Total Revenue"]
            q_cols = list(quarterly.columns)
            for i, col in enumerate(q_cols):
                # NOTE: "%q" is NOT a valid strftime directive in Python
                # (ValueError: Invalid format string) — quarter must be
                # computed manually from the month instead.
                if hasattr(col, "year") and hasattr(col, "month"):
                    quarter = _fiscal_quarter_label(col.year, col.month, fye_month)
                else:
                    quarter = str(col)
                rev = _safe_float(q_rev_row[col])
                # YoY: compare with same quarter last year (4 periods back)
                prev_rev = _safe_float(q_rev_row[q_cols[i + 4]]) if i + 4 < len(q_cols) else None
                growth = (
                    round((rev - prev_rev) / abs(prev_rev), 4)
                    if (rev is not None and prev_rev is not None and prev_rev != 0)
                    else None
                )
                quarterly_revenue.append({"quarter": quarter, "revenue": rev, "yoy_growth": growth})

        # --- Quarterly net income ---
        # Mirrors the quarterly_revenue block above. Previously MISSING —
        # only quarterly_revenue was ever extracted, even though
        # `quarterly_financials` carries a "Net Income" row exactly like the
        # annual `financials` frame does. Its absence forced
        # financial_agent.py's net_margin calculation to fall back to
        # annual_net_income[0]/annual_revenue[0] (a full fiscal year) even
        # when narrating a specific quarter (e.g. Q1 FY2027) alongside it —
        # the root cause of the NVDA net_margin bug (55.6% FY-annual vs.
        # 71.5% actual Q1 FY2027, per the 10-Q's own percentage-of-revenue
        # table). With this present, callers can compute a genuinely
        # same-quarter net margin instead of just labeling the mismatch.
        if quarterly is not None and not quarterly.empty and "Net Income" in quarterly.index:
            q_inc_row = quarterly.loc["Net Income"]
            q_cols = list(quarterly.columns)
            for i, col in enumerate(q_cols):
                if hasattr(col, "year") and hasattr(col, "month"):
                    quarter = _fiscal_quarter_label(col.year, col.month, fye_month)
                else:
                    quarter = str(col)
                ni = _safe_float(q_inc_row[col])
                prev_ni = _safe_float(q_inc_row[q_cols[i + 4]]) if i + 4 < len(q_cols) else None
                growth = (
                    round((ni - prev_ni) / abs(prev_ni), 4)
                    if (ni is not None and prev_ni is not None and prev_ni != 0)
                    else None
                )
                quarterly_net_income.append({"quarter": quarter, "net_income": ni, "yoy_growth": growth})

        # NOTE: despite the key name "revenue_growth_ttm" (kept for backward
        # compatibility with existing consumers), Yahoo Finance's
        # info["revenueGrowth"] is NOT a true trailing-twelve-month
        # calculation — it's the YoY growth of the most recently reported
        # quarter only. This is a known Yahoo/yfinance field-naming quirk,
        # not something we can fix at the source. Concretely: for AAPL this
        # returned 0.166 (16.6%) while the full FY2025 annual yoy_growth in
        # annual_revenue was 0.0643 (6.43%) — a single strong quarter can
        # diverge a lot from the full-year average. Treat this as
        # "most recent quarter's YoY growth", NOT as annual/TTM growth, and
        # do not directly compare it to annual_revenue's yoy_growth values
        # without noting the period mismatch.
        revenue_growth_ttm = _safe_get(info, "revenueGrowth")

        return {
            "ticker":                    ticker.upper(),
            "annual_revenue":            annual_revenue,
            "quarterly_revenue":         quarterly_revenue,
            "annual_net_income":         annual_net_income,
            "quarterly_net_income":      quarterly_net_income,
            "revenue_growth_ttm":        revenue_growth_ttm,
            # Explicit period label so downstream consumers (LLM narrative
            # generation, reports) don't conflate this with true annual/TTM
            # growth — see note above.
            "revenue_growth_ttm_period": "most_recent_quarter_yoy",
            # Non-fatal sanity-check output (see the gap check in the annual
            # loop above). Empty list = no gross mislabeling detected; does
            # NOT guarantee the year labels are correct, only that they're
            # internally consistent (~1 year apart, no gaps/duplicates).
            "year_labeling_warnings":    year_labeling_warnings,
            "as_of": context.to_dict()["as_of"], "dataset_version": context.dataset_version,
            "run_id": identity, "provider": "yahoo_statements_current",
            "current_fallback_used": False, "error": None,
        }

    except Exception:
        return {"ticker": ticker, "error_code": "YAHOO_GROWTH_DATA_UNAVAILABLE", "error": "Yahoo growth data is unavailable."}


def get_peer_comparison(
    ticker: str, peers: list[str] | None = None,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict:
    """
    Compare a stock's key financial ratios against a list of peer companies.

    If no peers are provided, the function attempts to infer them from the
    ticker's industry via yfinance (best-effort; accuracy varies).

    Parameters
    ----------
    ticker : str
        The primary stock ticker symbol.
    peers : list[str] | None
        Optional list of peer ticker symbols to compare against.
        Example: ["AMD", "INTC", "QCOM"]

    Returns
    -------
    dict
        A dictionary with:
        - "primary"     (dict) : Ratios for the primary ticker.
        - "peers"       (list) : List of ratio dicts for each peer.
        - "summary"     (dict) : Peer-average values for each ratio.
        - "growth_adjusted_comparison" (dict) : PEG-based comparison —
          primary vs peer-average PEG ratio, since raw P/E or EV/EBITDA
          comparisons don't account for differing growth rates between
          the primary ticker and its peers.
        - "error"       (str | None)

    Notes
    -----
    Rate limits from Yahoo Finance may cause failures for large peer lists.
    Recommended maximum: 10 peers per call.
    """
    context, identity = require_financial_context(forecast_context, run_id, ticker=ticker)
    if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
        return historical_current_data_unavailable(
            context=context, run_id=identity, provider="yahoo_peer_current",
            fields=["peer_comparison"],
        )
    try:
        # Fetch primary ticker ratios
        primary_ratios = get_financial_ratios(ticker, context, identity)
        if primary_ratios.get("error"):
            return {"primary": primary_ratios, "peers": [], "summary": {}, "error": primary_ratios["error"]}

        # NOTE: auto-discovering peers from yfinance's `.recommendations` was
        # attempted previously but never actually implemented (the response
        # was fetched and discarded) -- removed rather than left as dead code
        # that looks functional but silently does nothing. If this feature
        # is wanted, it needs to parse `.recommendations` into ticker symbols
        # and be covered by a test before being reintroduced.
        if not peers:
            peers = []

        # Peers are fetched concurrently: each get_financial_ratios() call is
        # an independent network round-trip to Yahoo, so running them serially
        # made this function's latency scale linearly with the peer count.
        # ThreadPoolExecutor.map preserves input order, so peer_ratios[i] still
        # corresponds to peers[i]. The pool is capped at 5 to stay well inside
        # Yahoo's rate limits (see the 10-peer recommendation above).
        if peers:
            with ThreadPoolExecutor(max_workers=min(5, len(peers))) as pool:
                peer_ratios = list(pool.map(
                    lambda peer: get_financial_ratios(peer, {
                        **context.to_dict(), "ticker": str(peer).upper(),
                    }, identity), peers,
                ))
        else:
            peer_ratios = []

        # Compute peer averages for numeric fields.
        # NOTE: peg_ratio is included specifically to give a growth-
        # ADJUSTED comparison, not just raw multiples. Raw P/E or
        # EV/EBITDA comparisons treat a 5%-growth company and a 30%-
        # growth company as directly comparable, which is misleading —
        # PEG (P/E ÷ expected earnings growth) already normalises for
        # growth, so "primary PEG vs peer-average PEG" answers "is this
        # stock expensive RELATIVE TO ITS OWN GROWTH RATE compared to
        # peers" rather than just "is its P/E higher".
        numeric_fields = ["pe_ratio", "forward_pe", "price_to_book",
                          "price_to_sales", "ev_to_ebitda", "beta",
                          "peg_ratio"]
        summary = {}
        for field in numeric_fields:
            values = [r[field] for r in peer_ratios if r.get(field) is not None]
            summary[f"avg_{field}"] = round(sum(values) / len(values), 4) if values else None

        # -- Growth-adjusted comparison (PEG-based) --------------------------
        primary_peg = primary_ratios.get("peg_ratio")
        peer_peg_values = [r["peg_ratio"] for r in peer_ratios if r.get("peg_ratio") is not None]
        growth_adjusted_comparison: dict = {
            "primary_peg_ratio": primary_peg,
            "peer_avg_peg_ratio": summary.get("avg_peg_ratio"),
            "interpretation": "insufficient_data",
            "note": (
                "PEG (P/E ÷ expected earnings growth) normalises P/E for "
                "growth differences between companies, unlike a raw P/E "
                "comparison. A stock can have a higher P/E than peers but "
                "still be CHEAPER on a growth-adjusted basis if it's "
                "growing faster — and vice versa."
            ),
        }
        if primary_peg is not None and peer_peg_values:
            peer_avg_peg = summary["avg_peg_ratio"]
            if peer_avg_peg and peer_avg_peg > 0:
                relative_pct = round((primary_peg - peer_avg_peg) / peer_avg_peg * 100, 1)
                growth_adjusted_comparison["relative_to_peers_pct"] = relative_pct
                if relative_pct <= -15:
                    growth_adjusted_comparison["interpretation"] = "cheap_relative_to_growth"
                elif relative_pct >= 15:
                    growth_adjusted_comparison["interpretation"] = "expensive_relative_to_growth"
                else:
                    growth_adjusted_comparison["interpretation"] = "in_line_with_peers"
        elif primary_peg is None:
            growth_adjusted_comparison["note"] += (
                " Primary ticker's PEG ratio was unavailable from Yahoo "
                "Finance — growth-adjusted comparison could not be computed."
            )
        elif not peer_peg_values:
            growth_adjusted_comparison["note"] += (
                " None of the peer tickers had a PEG ratio available — "
                "growth-adjusted comparison could not be computed."
            )

        return {
            "primary": primary_ratios,
            "peers":   peer_ratios,
            "summary": summary,
            "growth_adjusted_comparison": growth_adjusted_comparison,
            "as_of": context.to_dict()["as_of"], "dataset_version": context.dataset_version,
            "run_id": identity, "provider": "yahoo_peer_current",
            "current_fallback_used": False, "error": None,
        }

    except Exception:
        return {"primary": {}, "peers": [], "summary": {}, "error_code": "YAHOO_PEER_DATA_UNAVAILABLE", "error": "Yahoo peer data is unavailable."}
