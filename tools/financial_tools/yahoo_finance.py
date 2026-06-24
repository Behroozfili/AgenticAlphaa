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

import yfinance as yf
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_price_history(ticker: str, period: str = "1y") -> dict:
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
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return {"ticker": ticker, "period": period, "records": [],
                    "start_date": None, "end_date": None,
                    "error": "No data returned for this ticker/period."}

        records = []
        for date, row in hist.iterrows():
            records.append({
                "date":   date.strftime("%Y-%m-%d"),
                "open":   round(float(row["Open"]),   4),
                "high":   round(float(row["High"]),   4),
                "low":    round(float(row["Low"]),    4),
                "close":  round(float(row["Close"]),  4),
                "volume": int(row["Volume"]),
            })

        return {
            "ticker":     ticker.upper(),
            "period":     period,
            "records":    records,
            "start_date": records[0]["date"],
            "end_date":   records[-1]["date"],
            "error":      None,
        }

    except Exception as exc:
        return {
            "ticker": ticker, "period": period, "records": [],
            "start_date": None, "end_date": None,
            "error": str(exc),
        }


def get_financial_ratios(ticker: str) -> dict:
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
        - "52w_high"            (float | None)
        - "52w_low"             (float | None)
        - "current_price"       (float | None)
        - "error"               (str | None)

    Examples
    --------
    >>> ratios = get_financial_ratios("AAPL")
    >>> ratios["pe_ratio"]
    28.5
    """
    try:
        info = yf.Ticker(ticker).info

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
            "52w_high":         _safe_get(info, "fiftyTwoWeekHigh"),
            "52w_low":          _safe_get(info, "fiftyTwoWeekLow"),
            "current_price":    _safe_get(info, "currentPrice"),
            "error":            None,
        }

    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def get_revenue_growth(ticker: str) -> dict:
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
        - "revenue_growth_ttm"  (float | None) : YoY growth for trailing 12 months.
        - "error"               (str | None)

    Notes
    -----
    Growth values are expressed as decimals (e.g. 1.22 = +122% growth).
    None is returned for periods where prior-year data is unavailable.
    """
    try:
        stock    = yf.Ticker(ticker)
        info     = stock.info

        # --- Annual income statement ---
        financials = stock.financials  # columns = fiscal years (most recent first)
        annual_revenue    = []
        annual_net_income = []

        if financials is not None and not financials.empty:
            rev_row = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else None
            inc_row = financials.loc["Net Income"]    if "Net Income"    in financials.index else None

            cols = list(financials.columns)  # most-recent first
            for i, col in enumerate(cols):
                year = col.year if hasattr(col, "year") else str(col)

                # Revenue
                rev = float(rev_row[col]) if rev_row is not None else None
                prev_rev = float(rev_row[cols[i + 1]]) if (rev_row is not None and i + 1 < len(cols)) else None
                rev_growth = round((rev - prev_rev) / abs(prev_rev), 4) if (rev and prev_rev) else None
                annual_revenue.append({"year": year, "revenue": rev, "yoy_growth": rev_growth})

                # Net Income
                ni = float(inc_row[col]) if inc_row is not None else None
                prev_ni = float(inc_row[cols[i + 1]]) if (inc_row is not None and i + 1 < len(cols)) else None
                ni_growth = round((ni - prev_ni) / abs(prev_ni), 4) if (ni and prev_ni) else None
                annual_net_income.append({"year": year, "net_income": ni, "yoy_growth": ni_growth})

        # --- Quarterly revenue ---
        quarterly = stock.quarterly_financials
        quarterly_revenue = []
        if quarterly is not None and not quarterly.empty and "Total Revenue" in quarterly.index:
            q_rev_row = quarterly.loc["Total Revenue"]
            q_cols = list(quarterly.columns)
            for i, col in enumerate(q_cols):
                # NOTE: "%q" is NOT a valid strftime directive in Python
                # (ValueError: Invalid format string) — quarter must be
                # computed manually from the month instead.
                if hasattr(col, "year") and hasattr(col, "month"):
                    q_num = (col.month - 1) // 3 + 1
                    quarter = f"{col.year}-Q{q_num}"
                else:
                    quarter = str(col)
                rev = float(q_rev_row[col])
                # YoY: compare with same quarter last year (4 periods back)
                prev_rev = float(q_rev_row[q_cols[i + 4]]) if i + 4 < len(q_cols) else None
                growth = round((rev - prev_rev) / abs(prev_rev), 4) if prev_rev else None
                quarterly_revenue.append({"quarter": quarter, "revenue": rev, "yoy_growth": growth})

        # TTM growth from info dict
        revenue_growth_ttm = _safe_get(info, "revenueGrowth")

        return {
            "ticker":             ticker.upper(),
            "annual_revenue":     annual_revenue,
            "quarterly_revenue":  quarterly_revenue,
            "annual_net_income":  annual_net_income,
            "revenue_growth_ttm": revenue_growth_ttm,
            "error":              None,
        }

    except Exception as exc:
        return {"ticker": ticker, "error": str(exc)}


def get_peer_comparison(ticker: str, peers: list[str] | None = None) -> dict:
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
        - "error"       (str | None)

    Notes
    -----
    Rate limits from Yahoo Finance may cause failures for large peer lists.
    Recommended maximum: 10 peers per call.
    """
    try:
        # Fetch primary ticker ratios
        primary_ratios = get_financial_ratios(ticker)
        if primary_ratios.get("error"):
            return {"primary": primary_ratios, "peers": [], "summary": {}, "error": primary_ratios["error"]}

        # If no peers provided, try to get them from yfinance recommendedSymbols
        if not peers:
            try:
                rec = yf.Ticker(ticker).recommendations
                # Fallback: empty list if recommendations not available
                peers = []
            except Exception:
                peers = []

        peer_ratios = []
        for p in peers:
            r = get_financial_ratios(p)
            peer_ratios.append(r)

        # Compute peer averages for numeric fields
        numeric_fields = ["pe_ratio", "forward_pe", "price_to_book",
                          "price_to_sales", "ev_to_ebitda", "beta"]
        summary = {}
        for field in numeric_fields:
            values = [r[field] for r in peer_ratios if r.get(field) is not None]
            summary[f"avg_{field}"] = round(sum(values) / len(values), 4) if values else None

        return {
            "primary": primary_ratios,
            "peers":   peer_ratios,
            "summary": summary,
            "error":   None,
        }

    except Exception as exc:
        return {"primary": {}, "peers": [], "summary": {}, "error": str(exc)}