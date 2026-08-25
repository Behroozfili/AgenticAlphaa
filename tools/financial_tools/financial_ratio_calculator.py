"""
financial_ratio_calculator.py
------------------------------
Custom module for computing financial ratios directly from raw financial
statement inputs (income statement, balance sheet, cash flow statement).

This module is intentionally decoupled from any external data source so
that it can be used with figures obtained from yfinance, SEC EDGAR XBRL,
or any other upstream provider.

All ratio functions accept plain numeric arguments and return a flat
dictionary containing the computed value and an interpretation label
(e.g. "strong", "moderate", "weak") where applicable.

Ratio categories
----------------
1. Valuation          : P/E, P/B, EV/EBITDA, PEG
2. Profitability      : Gross Margin, Operating Margin, Net Margin, ROE, ROA, ROIC
3. Liquidity          : Current Ratio, Quick Ratio, Cash Ratio
4. Leverage / Solvency: Debt-to-Equity, Debt-to-Assets, Interest Coverage
5. Efficiency         : Asset Turnover, Inventory Turnover, Receivables Turnover
6. Growth             : Revenue CAGR, Earnings CAGR
7. Composite Score    : Weighted summary score across all categories
"""

from __future__ import annotations
import random
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float | None, denominator: float | None,
              default: Any = None) -> float | None:
    """
    Safely divide two numbers, returning *default* when division is impossible.

    Parameters
    ----------
    numerator : float | None
    denominator : float | None
    default : Any
        Value returned when the denominator is zero, None, or numerator is None.

    Returns
    -------
    float | None
    """
    if numerator is None or denominator is None or denominator == 0:
        return default
    return numerator / denominator


def _label(value: float | None, thresholds: list[tuple[float, str]],
           higher_is_better: bool = True) -> str:
    """
    Map a numeric ratio to a qualitative label using threshold rules.

    Parameters
    ----------
    value : float | None
        The computed ratio value.
    thresholds : list[tuple[float, str]]
        Sorted list of (threshold, label) pairs.
        If *higher_is_better*, labels are assigned for value >= threshold.
        If not, labels are assigned for value <= threshold.
    higher_is_better : bool
        Direction of the evaluation. Defaults to True.

    Returns
    -------
    str
        A qualitative label such as "strong", "moderate", or "weak".
    """
    if value is None:
        return "unavailable"
    for threshold, label in thresholds:
        if higher_is_better and value >= threshold:
            return label
        if not higher_is_better and value <= threshold:
            return label
    return thresholds[-1][1]


# ---------------------------------------------------------------------------
# 1. Valuation Ratios
# ---------------------------------------------------------------------------

def price_to_earnings(price: float, eps: float) -> dict:
    """
    Compute the Price-to-Earnings (P/E) ratio.

    Parameters
    ----------
    price : float
        Current market price per share (USD).
    eps : float
        Earnings Per Share — trailing twelve months (TTM).

    Returns
    -------
    dict
        - "pe_ratio"       (float | None)
        - "interpretation" (str) : "undervalued", "fairly_valued", "overvalued"
        - "formula"        (str) : Human-readable formula used.
    """
    pe = _safe_div(price, eps)
    # DC-2: removed dead _label() call — result was immediately overwritten
    # by the if/elif chain below which is the actual interpretation logic.
    interp = "unavailable"
    
    if pe is not None:
        if pe < 0:
            interp = "negative_earnings"
        elif pe < 15:
            interp = "undervalued"
        elif pe <= 30:
            interp = "fairly_valued"
        else:
            interp = "overvalued"

    return {
        "pe_ratio":       round(pe, 2) if pe is not None else None,
        "interpretation": interp,
        "formula":        "Price / EPS (TTM)",
    }


def price_to_book(price: float, book_value_per_share: float) -> dict:
    """
    Compute the Price-to-Book (P/B) ratio.

    Parameters
    ----------
    price : float
        Current market price per share (USD).
    book_value_per_share : float
        Book value (total equity / shares outstanding) per share.

    Returns
    -------
    dict
        - "pb_ratio"       (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    pb = _safe_div(price, book_value_per_share)
    if pb is None:
        interp = "unavailable"
    elif pb < 1:
        interp = "trading_below_book"
    elif pb <= 3:
        interp = "fairly_valued"
    else:
        interp = "premium_to_book"

    return {
        "pb_ratio":       round(pb, 2) if pb is not None else None,
        "interpretation": interp,
        "formula":        "Price / Book Value Per Share",
    }


def ev_to_ebitda(enterprise_value: float, ebitda: float) -> dict:
    """
    Compute the Enterprise Value to EBITDA multiple.

    Parameters
    ----------
    enterprise_value : float
        Total enterprise value (market cap + debt - cash) in USD.
    ebitda : float
        Earnings Before Interest, Taxes, Depreciation & Amortisation (TTM) in USD.

    Returns
    -------
    dict
        - "ev_ebitda"      (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    ev_eb = _safe_div(enterprise_value, ebitda)
    if ev_eb is None:
        interp = "unavailable"
    elif ev_eb < 10:
        interp = "undervalued"
    elif ev_eb <= 20:
        interp = "fairly_valued"
    else:
        interp = "expensive"

    return {
        "ev_ebitda":      round(ev_eb, 2) if ev_eb is not None else None,
        "interpretation": interp,
        "formula":        "Enterprise Value / EBITDA",
    }


def peg_ratio(pe: float, earnings_growth_rate_pct: float) -> dict:
    """
    Compute the PEG (Price/Earnings-to-Growth) ratio.

    Parameters
    ----------
    pe : float
        The trailing or forward P/E ratio.
    earnings_growth_rate_pct : float
        Expected annual EPS growth rate as a percentage (e.g. 25 for 25%).

    Returns
    -------
    dict
        - "peg"            (float | None)
        - "interpretation" (str) : "undervalued" if PEG < 1, else "overvalued"
        - "formula"        (str)
    """
    peg = _safe_div(pe, earnings_growth_rate_pct)
    if peg is None:
        interp = "unavailable"
    elif peg < 1:
        interp = "undervalued"
    elif peg <= 2:
        interp = "fairly_valued"
    else:
        interp = "overvalued"

    return {
        "peg":            round(peg, 2) if peg is not None else None,
        "interpretation": interp,
        "formula":        "P/E Ratio / EPS Growth Rate (%)",
    }


# ---------------------------------------------------------------------------
# 2. Profitability Ratios
# ---------------------------------------------------------------------------

def gross_margin(revenue: float, cogs: float) -> dict:
    """
    Compute the Gross Profit Margin.

    Parameters
    ----------
    revenue : float   Total revenue (TTM) in USD.
    cogs    : float   Cost of Goods Sold (TTM) in USD.

    Returns
    -------
    dict
        - "gross_margin_pct" (float | None) : Percentage (e.g. 62.5 for 62.5%).
        - "interpretation"   (str)
        - "formula"          (str)
    """
    gp  = revenue - cogs
    gm  = _safe_div(gp, revenue)
    pct = round(gm * 100, 2) if gm is not None else None
    interp = _label(pct, [(60, "excellent"), (40, "good"), (20, "moderate"), (0, "low")])

    return {
        "gross_margin_pct": pct,
        "interpretation":   interp,
        "formula":          "(Revenue - COGS) / Revenue × 100",
    }


def operating_margin(operating_income: float, revenue: float) -> dict:
    """
    Compute the Operating Profit Margin.

    Parameters
    ----------
    operating_income : float   EBIT in USD.
    revenue          : float   Total revenue in USD.

    Returns
    -------
    dict
        - "operating_margin_pct" (float | None)
        - "interpretation"       (str)
        - "formula"              (str)
    """
    om  = _safe_div(operating_income, revenue)
    pct = round(om * 100, 2) if om is not None else None
    interp = _label(pct, [(25, "excellent"), (15, "good"), (5, "moderate"), (0, "low")])

    return {
        "operating_margin_pct": pct,
        "interpretation":       interp,
        "formula":              "Operating Income / Revenue × 100",
    }


def net_margin(net_income: float, revenue: float) -> dict:
    """
    Compute the Net Profit Margin.

    Parameters
    ----------
    net_income : float   Net income (TTM) in USD.
    revenue    : float   Total revenue (TTM) in USD.

    Returns
    -------
    dict
        - "net_margin_pct" (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    nm  = _safe_div(net_income, revenue)
    pct = round(nm * 100, 2) if nm is not None else None
    interp = _label(pct, [(20, "excellent"), (10, "good"), (5, "moderate"), (0, "low")])

    return {
        "net_margin_pct": pct,
        "interpretation": interp,
        "formula":        "Net Income / Revenue × 100",
    }


def return_on_equity(net_income: float, shareholders_equity: float) -> dict:
    """
    Compute Return on Equity (ROE).

    Parameters
    ----------
    net_income          : float   Net income (TTM) in USD.
    shareholders_equity : float   Average total shareholders' equity in USD.

    Returns
    -------
    dict
        - "roe_pct"        (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    roe = _safe_div(net_income, shareholders_equity)
    pct = round(roe * 100, 2) if roe is not None else None
    interp = _label(pct, [(20, "excellent"), (15, "good"), (10, "moderate"), (0, "low")])

    return {
        "roe_pct":        pct,
        "interpretation": interp,
        "formula":        "Net Income / Shareholders' Equity × 100",
    }


def return_on_assets(net_income: float, total_assets: float) -> dict:
    """
    Compute Return on Assets (ROA).

    Parameters
    ----------
    net_income   : float   Net income (TTM) in USD.
    total_assets : float   Average total assets in USD.

    Returns
    -------
    dict
        - "roa_pct"        (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    roa = _safe_div(net_income, total_assets)
    pct = round(roa * 100, 2) if roa is not None else None
    interp = _label(pct, [(10, "excellent"), (5, "good"), (2, "moderate"), (0, "low")])

    return {
        "roa_pct":        pct,
        "interpretation": interp,
        "formula":        "Net Income / Total Assets × 100",
    }


# ---------------------------------------------------------------------------
# 3. Liquidity Ratios
# ---------------------------------------------------------------------------

def current_ratio(current_assets: float, current_liabilities: float) -> dict:
    """
    Compute the Current Ratio (short-term liquidity indicator).

    Parameters
    ----------
    current_assets      : float   Total current assets in USD.
    current_liabilities : float   Total current liabilities in USD.

    Returns
    -------
    dict
        - "current_ratio"  (float | None)
        - "interpretation" (str) : "strong" (>2), "adequate" (1–2), "weak" (<1)
        - "formula"        (str)
    """
    cr = _safe_div(current_assets, current_liabilities)
    if cr is None:
        interp = "unavailable"
    elif cr >= 2:
        interp = "strong"
    elif cr >= 1:
        interp = "adequate"
    else:
        interp = "weak"

    return {
        "current_ratio":  round(cr, 2) if cr is not None else None,
        "interpretation": interp,
        "formula":        "Current Assets / Current Liabilities",
    }


def quick_ratio(cash: float, short_term_investments: float,
                receivables: float, current_liabilities: float) -> dict:
    """
    Compute the Quick Ratio (acid-test ratio).

    Excludes inventories and other less-liquid current assets.

    Parameters
    ----------
    cash                   : float   Cash and cash equivalents in USD.
    short_term_investments : float   Short-term marketable securities in USD.
    receivables            : float   Net accounts receivable in USD.
    current_liabilities    : float   Total current liabilities in USD.

    Returns
    -------
    dict
        - "quick_ratio"    (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    liquid_assets = cash + short_term_investments + receivables
    qr = _safe_div(liquid_assets, current_liabilities)
    if qr is None:
        interp = "unavailable"
    elif qr >= 1:
        interp = "strong"
    elif qr >= 0.5:
        interp = "moderate"
    else:
        interp = "weak"

    return {
        "quick_ratio":    round(qr, 2) if qr is not None else None,
        "interpretation": interp,
        "formula":        "(Cash + Short-term Investments + Receivables) / Current Liabilities",
    }


# ---------------------------------------------------------------------------
# 4. Leverage / Solvency Ratios
# ---------------------------------------------------------------------------

def debt_to_equity(total_debt: float, shareholders_equity: float) -> dict:
    """
    Compute the Debt-to-Equity (D/E) ratio.

    Parameters
    ----------
    total_debt          : float   Total long-term + short-term debt in USD.
    shareholders_equity : float   Total shareholders' equity in USD.

    Returns
    -------
    dict
        - "de_ratio"       (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    de = _safe_div(total_debt, shareholders_equity)
    if de is None:
        interp = "unavailable"
    elif de < 0.5:
        interp = "low_leverage"
    elif de <= 1.5:
        interp = "moderate_leverage"
    else:
        interp = "high_leverage"

    return {
        "de_ratio":       round(de, 2) if de is not None else None,
        "interpretation": interp,
        "formula":        "Total Debt / Shareholders' Equity",
    }


def interest_coverage(ebit: float, interest_expense: float) -> dict:
    """
    Compute the Interest Coverage Ratio (times-interest-earned).

    Parameters
    ----------
    ebit             : float   Earnings Before Interest and Taxes in USD.
    interest_expense : float   Total interest expense in USD.

    Returns
    -------
    dict
        - "interest_coverage" (float | None)
        - "interpretation"    (str)
        - "formula"           (str)
    """
    ic = _safe_div(ebit, interest_expense)
    if ic is None:
        interp = "unavailable"
    elif ic >= 5:
        interp = "strong"
    elif ic >= 2:
        interp = "adequate"
    else:
        interp = "at_risk"

    return {
        "interest_coverage": round(ic, 2) if ic is not None else None,
        "interpretation":    interp,
        "formula":           "EBIT / Interest Expense",
    }


# ---------------------------------------------------------------------------
# 5. Efficiency Ratios
# ---------------------------------------------------------------------------

def asset_turnover(revenue: float, avg_total_assets: float) -> dict:
    """
    Compute the Asset Turnover Ratio.

    Parameters
    ----------
    revenue          : float   Total annual revenue in USD.
    avg_total_assets : float   Average total assets ((start + end) / 2) in USD.

    Returns
    -------
    dict
        - "asset_turnover" (float | None)
        - "interpretation" (str)
        - "formula"        (str)
    """
    at = _safe_div(revenue, avg_total_assets)
    if at is None:
        interp = "unavailable"
    elif at >= 1:
        interp = "efficient"
    elif at >= 0.5:
        interp = "moderate"
    else:
        interp = "low_efficiency"

    return {
        "asset_turnover": round(at, 2) if at is not None else None,
        "interpretation": interp,
        "formula":        "Revenue / Average Total Assets",
    }


# ---------------------------------------------------------------------------
# 6. Growth Ratios
# ---------------------------------------------------------------------------

def cagr(start_value: float, end_value: float, years: float) -> dict:
    """
    Compute the Compound Annual Growth Rate (CAGR).

    Parameters
    ----------
    start_value : float   Value at the beginning of the period.
    end_value   : float   Value at the end of the period.
    years       : float   Number of years in the period.

    Returns
    -------
    dict
        - "cagr_pct"       (float | None) : CAGR as a percentage.
        - "interpretation" (str)
        - "formula"        (str)
    """
    try:
        if start_value <= 0 or years <= 0:
            return {"cagr_pct": None, "interpretation": "unavailable",
                    "formula": "(End / Start)^(1/years) - 1"}
        rate = (end_value / start_value) ** (1 / years) - 1
        pct  = round(rate * 100, 2)
        interp = _label(pct, [(30, "hypergrowth"), (15, "strong"), (5, "moderate"), (0, "slow")])
        return {
            "cagr_pct":       pct,
            "interpretation": interp,
            "formula":        "(End Value / Start Value)^(1 / Years) − 1",
        }
    except Exception as exc:
        return {"cagr_pct": None, "interpretation": "error", "formula": str(exc)}


def compute_revenue_cagr_from_growth(annual_revenue: list[dict]) -> dict:
    """
    Single-purpose convenience wrapper around cagr() for the exact shape
    returned by yahoo_finance.get_revenue_growth()["annual_revenue"]:
        [{"year": 2025, "revenue": ..., "yoy_growth": ...}, ...]

    Saves the calling agent from having to manually pick out the start/end
    values and the year count — just pass the raw annual_revenue list.

    IMPORTANT — ordering: yahoo_finance.get_revenue_growth() returns this
    list MOST-RECENT YEAR FIRST (financials.columns from yfinance are
    newest-first). This function does NOT assume any particular input
    order; it sorts internally by "year" to be safe:
        - start_value = oldest year's revenue
        - end_value   = newest year's revenue
        - years       = newest_year - oldest_year

    Parameters
    ----------
    annual_revenue : list[dict]
        Each dict must have "year" (int) and "revenue" (float | None).
        Entries with revenue=None are dropped before computing CAGR.
        Needs at least 2 valid entries spanning at least 1 year.

    Returns
    -------
    dict
        Same shape as cagr(), plus traceability fields:
        - "cagr_pct"       (float | None)
        - "interpretation" (str)
        - "formula"        (str)
        - "start_year"     (int | None)
        - "end_year"       (int | None)
        - "years_used"     (int | None)
        - "error"          (str | None) : Set when input is insufficient.

    Examples
    --------
    >>> compute_revenue_cagr_from_growth([
    ...     {"year": 2025, "revenue": 391_000_000_000, "yoy_growth": 0.02},
    ...     {"year": 2024, "revenue": 383_000_000_000, "yoy_growth": 0.01},
    ...     {"year": 2023, "revenue": 379_000_000_000, "yoy_growth": -0.03},
    ...     {"year": 2022, "revenue": 391_000_000_000, "yoy_growth": 0.08},
    ...     {"year": 2021, "revenue": 365_000_000_000, "yoy_growth": None},
    ... ])
    {'cagr_pct': 1.71, 'interpretation': 'slow', ..., 'start_year': 2021, 'end_year': 2025, 'years_used': 4, 'error': None}
    """
    valid = [
        e for e in (annual_revenue or [])
        if e.get("year") is not None and e.get("revenue") is not None
    ]
    if len(valid) < 2:
        return {
            "cagr_pct": None, "interpretation": "unavailable",
            "formula": "(End Value / Start Value)^(1 / Years) − 1",
            "start_year": None, "end_year": None, "years_used": None,
            "error": f"Need at least 2 years with revenue data; got {len(valid)}.",
        }

    # Sort oldest -> newest regardless of input order
    valid.sort(key=lambda e: e["year"])
    start_entry = valid[0]
    end_entry   = valid[-1]
    years       = end_entry["year"] - start_entry["year"]

    if years <= 0:
        return {
            "cagr_pct": None, "interpretation": "unavailable",
            "formula": "(End Value / Start Value)^(1 / Years) − 1",
            "start_year": start_entry["year"], "end_year": end_entry["year"],
            "years_used": years,
            "error": "Start and end years are the same or invalid; cannot compute CAGR.",
        }

    result = cagr(
        start_value=start_entry["revenue"],
        end_value=end_entry["revenue"],
        years=years,
    )
    result["start_year"] = start_entry["year"]
    result["end_year"]   = end_entry["year"]
    result["years_used"] = years
    result["error"]      = (
        None if result.get("cagr_pct") is not None
        else "cagr() returned no value (see interpretation)."
    )
    return result


# ---------------------------------------------------------------------------
# 6b. Cash Flow Ratios  (Phase 1 -- data already available via yfinance .info:
#     freeCashflow, operatingCashflow)
# ---------------------------------------------------------------------------

def fcf_margin(free_cash_flow: float, revenue: float) -> dict:
    """
    Compute Free Cash Flow Margin: how much of every revenue dollar
    actually converts to free cash, after capex. A company can show
    healthy net-income margins while burning cash if capex or working
    capital needs are heavy (common in AI-infrastructure-heavy years) --
    this ratio is what catches that gap.

    Parameters
    ----------
    free_cash_flow : float   Operating cash flow − capex, in USD (TTM).
    revenue        : float   Total revenue in USD (TTM).

    Returns
    -------
    dict
        - "fcf_margin_pct"  (float | None)
        - "interpretation"  (str)
        - "formula"         (str)
    """
    m = _safe_div(free_cash_flow, revenue)
    pct = round(m * 100, 2) if m is not None else None
    if pct is None:
        interp = "unavailable"
    elif pct >= 20:
        interp = "excellent"
    elif pct >= 10:
        interp = "strong"
    elif pct >= 0:
        interp = "modest"
    else:
        interp = "cash_burning"

    return {
        "fcf_margin_pct":  pct,
        "interpretation":  interp,
        "formula":         "Free Cash Flow / Revenue",
    }


def fcf_yield(free_cash_flow: float, market_cap: float) -> dict:
    """
    Compute Free Cash Flow Yield: FCF as a percentage of what the market
    is paying for the whole company. A cheap-but-real cash-flow-based
    counterpart to earnings yield / P·E -- less distorted by non-cash
    accounting items (depreciation, stock comp, one-offs) than earnings.

    Parameters
    ----------
    free_cash_flow : float   Free cash flow in USD (TTM).
    market_cap     : float   Market capitalization in USD.

    Returns
    -------
    dict
        - "fcf_yield_pct"   (float | None)
        - "interpretation"  (str)
        - "formula"         (str)
    """
    y = _safe_div(free_cash_flow, market_cap)
    pct = round(y * 100, 2) if y is not None else None
    if pct is None:
        interp = "unavailable"
    elif pct >= 8:
        interp = "cheap"
    elif pct >= 4:
        interp = "fair"
    elif pct >= 0:
        interp = "expensive"
    else:
        interp = "cash_burning"

    return {
        "fcf_yield_pct":   pct,
        "interpretation":  interp,
        "formula":         "Free Cash Flow / Market Capitalization",
    }


def operating_cash_flow_ratio(operating_cash_flow: float, current_liabilities: float) -> dict:
    """
    Compute the Operating Cash Flow Ratio: can the company cover its
    near-term obligations from cash actually generated by operations,
    rather than from the accrual-accounting current ratio (which counts
    inventory/receivables that may not convert to cash in time)?

    Parameters
    ----------
    operating_cash_flow : float   Cash from operating activities in USD (TTM).
    current_liabilities  : float   Total current liabilities in USD.

    Returns
    -------
    dict
        - "ocf_ratio"       (float | None)
        - "interpretation"  (str)
        - "formula"         (str)
    """
    r = _safe_div(operating_cash_flow, current_liabilities)
    if r is None:
        interp = "unavailable"
    elif r >= 1:
        interp = "strong"
    elif r >= 0.4:
        interp = "adequate"
    else:
        interp = "weak"

    return {
        "ocf_ratio":       round(r, 2) if r is not None else None,
        "interpretation":  interp,
        "formula":         "Operating Cash Flow / Current Liabilities",
    }


# ---------------------------------------------------------------------------
# 6c. Advanced Profitability  (Phase 2 -- needs balance-sheet fields beyond
#     yfinance .info: total_assets, total_equity; see
#     yahoo_finance.get_balance_sheet_snapshot())
# ---------------------------------------------------------------------------

def return_on_invested_capital(
    ebit: float,
    tax_rate_pct: float,
    total_debt: float,
    total_equity: float,
    cash: float,
) -> dict:
    """
    Compute Return on Invested Capital (ROIC): NOPAT / Invested Capital.
    Widely considered a better cross-company comparison than ROE, because
    ROE can be inflated by leverage alone (a highly-indebted company can
    show a high ROE on a thin equity base without actually running the
    business better) -- ROIC looks at returns on ALL capital deployed
    (debt + equity), independent of financing mix.

    Parameters
    ----------
    ebit         : float   Earnings Before Interest and Taxes, in USD.
    tax_rate_pct : float   Effective tax rate, as a percentage (e.g. 21.0
                           for 21%). Use the company's own effective rate
                           (provision_for_income_taxes / pretax_income)
                           where available, not just the statutory rate.
    total_debt   : float   Total debt (short+long-term), in USD.
    total_equity : float   Total shareholders' equity, in USD.
    cash         : float   Cash and cash equivalents, in USD (netted out
                           of invested capital -- cash sitting on the
                           balance sheet isn't capital "invested" in the
                           operating business).

    Returns
    -------
    dict
        - "roic_pct"        (float | None)
        - "nopat"           (float | None)
        - "invested_capital" (float | None)
        - "interpretation"  (str)
        - "formula"         (str)
    """
    if ebit is None or tax_rate_pct is None:
        nopat = None
    else:
        nopat = ebit * (1 - tax_rate_pct / 100.0)

    invested_capital = None
    if total_debt is not None and total_equity is not None and cash is not None:
        invested_capital = total_debt + total_equity - cash

    roic = _safe_div(nopat, invested_capital)
    pct = round(roic * 100, 2) if roic is not None else None
    if pct is None:
        interp = "unavailable"
    elif pct >= 20:
        interp = "excellent"
    elif pct >= 10:
        interp = "strong"
    elif pct >= 5:
        interp = "moderate"
    else:
        interp = "weak"

    return {
        "roic_pct":          pct,
        "nopat":              round(nopat, 2) if nopat is not None else None,
        "invested_capital":   round(invested_capital, 2) if invested_capital is not None else None,
        "interpretation":     interp,
        "formula":            "NOPAT / (Total Debt + Total Equity − Cash), where NOPAT = EBIT × (1 − Tax Rate)",
    }


def return_on_capital_employed(ebit: float, total_assets: float, current_liabilities: float) -> dict:
    """
    Compute Return on Capital Employed (ROCE): EBIT / Capital Employed,
    where Capital Employed = Total Assets − Current Liabilities. A
    pre-tax, financing-structure-agnostic profitability measure commonly
    used alongside ROIC; the Magic Formula (Greenblatt) uses a close
    variant of this as its "quality" leg.

    Parameters
    ----------
    ebit                 : float   Earnings Before Interest and Taxes, in USD.
    total_assets         : float   Total assets, in USD.
    current_liabilities  : float   Total current liabilities, in USD.

    Returns
    -------
    dict
        - "roce_pct"          (float | None)
        - "capital_employed"  (float | None)
        - "interpretation"    (str)
        - "formula"           (str)
    """
    capital_employed = None
    if total_assets is not None and current_liabilities is not None:
        capital_employed = total_assets - current_liabilities

    roce = _safe_div(ebit, capital_employed)
    pct = round(roce * 100, 2) if roce is not None else None
    if pct is None:
        interp = "unavailable"
    elif pct >= 20:
        interp = "excellent"
    elif pct >= 12:
        interp = "strong"
    elif pct >= 6:
        interp = "moderate"
    else:
        interp = "weak"

    return {
        "roce_pct":          pct,
        "capital_employed":  round(capital_employed, 2) if capital_employed is not None else None,
        "interpretation":    interp,
        "formula":           "EBIT / (Total Assets − Current Liabilities)",
    }


# ---------------------------------------------------------------------------
# 6d. Additional Leverage / Solvency Ratios
# ---------------------------------------------------------------------------

def net_debt_to_ebitda(total_debt: float, total_cash: float, ebitda: float) -> dict:
    """
    Compute Net Debt / EBITDA -- the standard credit-analysis leverage
    multiple: roughly, how many years of current EBITDA would it take to
    pay off all debt net of cash on hand. Used almost universally in
    credit ratings and loan covenants.

    Parameters
    ----------
    total_debt : float   Total debt (short+long-term), in USD.
    total_cash : float   Cash and cash equivalents, in USD.
    ebitda     : float   EBITDA, in USD (TTM).

    Returns
    -------
    dict
        - "net_debt"            (float | None)
        - "net_debt_to_ebitda"  (float | None)
        - "interpretation"      (str)
        - "formula"             (str)
    """
    net_debt = None
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash

    ratio = _safe_div(net_debt, ebitda)
    if ratio is None:
        interp = "unavailable"
    elif ratio < 0:
        interp = "net_cash_position"  # more cash than debt
    elif ratio <= 1.5:
        interp = "low_leverage"
    elif ratio <= 3.0:
        interp = "moderate_leverage"
    else:
        interp = "high_leverage"

    return {
        "net_debt":            round(net_debt, 2) if net_debt is not None else None,
        "net_debt_to_ebitda":  round(ratio, 2) if ratio is not None else None,
        "interpretation":      interp,
        "formula":             "(Total Debt − Cash) / EBITDA",
    }


def debt_ratio(total_liabilities: float, total_assets: float) -> dict:
    """
    Compute the Debt Ratio: what fraction of total assets is financed by
    liabilities (of any kind, not just interest-bearing debt) rather than
    equity.

    Parameters
    ----------
    total_liabilities : float   Total liabilities, in USD.
    total_assets       : float   Total assets, in USD.

    Returns
    -------
    dict
        - "debt_ratio"      (float | None)
        - "interpretation"  (str)
        - "formula"         (str)
    """
    r = _safe_div(total_liabilities, total_assets)
    if r is None:
        interp = "unavailable"
    elif r < 0.4:
        interp = "low_leverage"
    elif r <= 0.6:
        interp = "moderate_leverage"
    else:
        interp = "high_leverage"

    return {
        "debt_ratio":      round(r, 2) if r is not None else None,
        "interpretation":  interp,
        "formula":         "Total Liabilities / Total Assets",
    }


def equity_ratio(total_equity: float, total_assets: float) -> dict:
    """
    Compute the Equity Ratio: what fraction of total assets is financed
    by shareholders' equity. The complement of debt_ratio (equity_ratio +
    debt_ratio ≈ 1, modulo minority interest / other equity items).

    Parameters
    ----------
    total_equity : float   Total shareholders' equity, in USD.
    total_assets : float   Total assets, in USD.

    Returns
    -------
    dict
        - "equity_ratio"    (float | None)
        - "interpretation"  (str)
        - "formula"         (str)
    """
    r = _safe_div(total_equity, total_assets)
    if r is None:
        interp = "unavailable"
    elif r >= 0.6:
        interp = "well_capitalized"
    elif r >= 0.4:
        interp = "moderate"
    else:
        interp = "thin_equity_base"

    return {
        "equity_ratio":    round(r, 2) if r is not None else None,
        "interpretation":  interp,
        "formula":         "Total Equity / Total Assets",
    }


# ---------------------------------------------------------------------------
# 6e. Bankruptcy Risk -- Altman Z-Score
# ---------------------------------------------------------------------------

def altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    total_assets: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> dict:
    """
    Compute the Altman Z-Score (1968 original, for public manufacturing/
    industrial firms -- the most widely cited variant). A composite
    bankruptcy-risk indicator combining five weighted ratios.

    IMPORTANT — model scope: the ORIGINAL Z-Score was built and validated
    on public manufacturing companies. It has known accuracy issues for
    financial firms (banks/insurers, whose balance sheets don't fit this
    template at all), and for asset-light service/software/tech firms
    (small total_assets denominators can produce extreme, less meaningful
    Z-scores). Report the number, but flag this caveat alongside it for
    non-manufacturing tickers -- do not present it as sector-agnostic.

    Parameters
    ----------
    working_capital   : float   Current Assets − Current Liabilities, in USD.
    retained_earnings : float   Retained earnings, in USD.
    ebit               : float   Earnings Before Interest and Taxes, in USD.
    total_assets       : float   Total assets, in USD.
    market_cap         : float   Market value of equity (≈ market cap), in USD.
    total_liabilities  : float   Total liabilities, in USD.
    revenue            : float   Total revenue (Sales), in USD.

    Returns
    -------
    dict
        - "z_score"         (float | None)
        - "zone"            (str) : "safe" (>2.99), "grey" (1.81–2.99),
                              "distress" (<1.81), "unavailable"
        - "interpretation"  (str)
        - "formula"         (str)
        - "caveat"          (str)
    """
    x1 = _safe_div(working_capital, total_assets)
    x2 = _safe_div(retained_earnings, total_assets)
    x3 = _safe_div(ebit, total_assets)
    x4 = _safe_div(market_cap, total_liabilities)
    x5 = _safe_div(revenue, total_assets)

    if any(v is None for v in (x1, x2, x3, x4, x5)):
        return {
            "z_score": None, "zone": "unavailable", "interpretation": "unavailable",
            "formula": "1.2·(WC/TA) + 1.4·(RE/TA) + 3.3·(EBIT/TA) + 0.6·(MVE/TL) + 1.0·(Sales/TA)",
            "caveat": (
                "Built and validated on public manufacturing/industrial firms; "
                "less reliable for financial firms and asset-light tech/software "
                "companies."
            ),
        }

    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    z = round(z, 2)

    if z > 2.99:
        zone, interp = "safe", "Low bankruptcy risk by this model."
    elif z >= 1.81:
        zone, interp = "grey", "Ambiguous zone -- neither clearly safe nor clearly distressed."
    else:
        zone, interp = "distress", "Elevated bankruptcy risk by this model."

    return {
        "z_score":         z,
        "zone":             zone,
        "interpretation":   interp,
        "formula":          "1.2·(WC/TA) + 1.4·(RE/TA) + 3.3·(EBIT/TA) + 0.6·(MVE/TL) + 1.0·(Sales/TA)",
        "caveat": (
            "Built and validated on public manufacturing/industrial firms; "
            "less reliable for financial firms and asset-light tech/software "
            "companies -- treat as one directional signal, not a verdict."
        ),
    }


# ---------------------------------------------------------------------------
# 6f. Reverse DCF
# ---------------------------------------------------------------------------

def reverse_dcf(
    fcf_base: float,
    beta: float,
    target_enterprise_value: float,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
    growth_rate_search_bounds: tuple[float, float] = (-20.0, 60.0),
    tolerance_pct: float = 0.1,
    max_iterations: int = 100,
    market_cap: float | None = None,
    total_debt: float | None = None,
    pre_tax_cost_of_debt_pct: float | None = None,
    effective_tax_rate_pct: float | None = None,
) -> dict:
    """
    Solve discounted_cash_flow() IN REVERSE: instead of assuming a growth
    rate and computing a fair value, take the value the market is
    currently implying (target_enterprise_value) and solve for the
    constant FCF growth rate that would justify it. Answers the question
    analysts actually ask more often than "what's fair value": "what
    growth rate is priced in right now, and is that realistic?"

    Method: binary search over fcf_growth_rate_pct (the discount rate and
    every other input stay fixed), since discounted_cash_flow's enterprise
    value is monotonically increasing in the growth rate for a fixed
    discount rate above the terminal growth rate -- there's exactly one
    growth rate that reproduces the target value (within the search
    bounds), so bisection converges reliably without needing calculus or
    an external optimizer dependency.

    IMPORTANT — target_enterprise_value must be an ENTERPRISE value (to
    match what discounted_cash_flow returns), not raw market cap. A
    reasonable proxy: market_cap + total_debt − total_cash. Passing raw
    market cap will silently back out a growth rate that's biased by
    however large net debt is -- see discounted_cash_flow's own docstring
    simplification #3 for the same enterprise-vs-equity-value caveat.

    Parameters
    ----------
    fcf_base, beta, terminal_growth_rate_pct, risk_free_rate_pct,
    market_risk_premium_pct, projection_years : same meaning as in
        discounted_cash_flow() -- everything EXCEPT fcf_growth_rate_pct,
        which is the unknown being solved for here.
    target_enterprise_value : float
        The enterprise value to solve for (see IMPORTANT note above).
    growth_rate_search_bounds : tuple[float, float]
        (min, max) growth rate percentages to search between. Default
        -20% to 60% covers essentially any real-world case; narrow it if
        you have a strong prior (e.g. a mature large-cap won't be priced
        for 60% growth) to converge faster / avoid absurd solutions at
        the boundary.
    tolerance_pct : float
        Stop when the computed enterprise value is within this percentage
        of target_enterprise_value. Default 0.1%.
    max_iterations : int
        Safety cap on bisection iterations. Default 100 (each iteration
        halves the search interval, so 100 is vastly more than needed for
        any realistic tolerance -- this just prevents an infinite loop if
        something pathological happens).

    Returns
    -------
    dict
        - "implied_growth_rate_pct" (float | None)
        - "resulting_enterprise_value" (float | None) : what the solved
          growth rate actually produces (should be ≈ target within tolerance).
        - "target_enterprise_value" (float)
        - "converged"        (bool)
        - "iterations_used"  (int)
        - "discount_rate_pct" (float | None)
        - "assumptions"      (dict) : every fixed input echoed back.
        - "interpretation"   (str)
        - "error"            (str | None)
    """
    assumptions = {
        "terminal_growth_rate_pct": terminal_growth_rate_pct,
        "risk_free_rate_pct":       risk_free_rate_pct,
        "market_risk_premium_pct":  market_risk_premium_pct,
        "projection_years":         projection_years,
        "market_cap": market_cap,
        "total_debt": total_debt,
        "pre_tax_cost_of_debt_pct": pre_tax_cost_of_debt_pct,
        "effective_tax_rate_pct": effective_tax_rate_pct,
    }

    def _ev_at(growth_pct: float) -> tuple[float | None, str | None, float | None]:
        r = discounted_cash_flow(
            fcf_base=fcf_base, beta=beta, shares_outstanding=None,
            fcf_growth_rate_pct=growth_pct,
            terminal_growth_rate_pct=terminal_growth_rate_pct,
            risk_free_rate_pct=risk_free_rate_pct,
            market_risk_premium_pct=market_risk_premium_pct,
            projection_years=projection_years,
            market_cap=market_cap,
            total_debt=total_debt,
            pre_tax_cost_of_debt_pct=pre_tax_cost_of_debt_pct,
            effective_tax_rate_pct=effective_tax_rate_pct,
        )
        return r["enterprise_value"], r["error"], r["discount_rate_pct"]

    if target_enterprise_value is None or target_enterprise_value <= 0:
        return {
            "implied_growth_rate_pct": None, "resulting_enterprise_value": None,
            "target_enterprise_value": target_enterprise_value, "converged": False,
            "iterations_used": 0, "discount_rate_pct": None, "assumptions": assumptions,
            "interpretation": "unavailable",
            "error": "target_enterprise_value is missing, zero, or negative.",
        }

    lo, hi = growth_rate_search_bounds
    ev_lo, err_lo, discount_rate_pct = _ev_at(lo)
    ev_hi, err_hi, _ = _ev_at(hi)

    if err_lo or err_hi or ev_lo is None or ev_hi is None:
        return {
            "implied_growth_rate_pct": None, "resulting_enterprise_value": None,
            "target_enterprise_value": target_enterprise_value, "converged": False,
            "iterations_used": 0, "discount_rate_pct": discount_rate_pct,
            "assumptions": assumptions, "interpretation": "unavailable",
            "error": err_lo or err_hi or "DCF could not be evaluated at the search bounds.",
        }

    if not (ev_lo <= target_enterprise_value <= ev_hi):
        return {
            "implied_growth_rate_pct": None, "resulting_enterprise_value": None,
            "target_enterprise_value": target_enterprise_value, "converged": False,
            "iterations_used": 0, "discount_rate_pct": discount_rate_pct,
            "assumptions": assumptions, "interpretation": "unavailable",
            "error": (
                f"target_enterprise_value ({target_enterprise_value:,.0f}) is outside "
                f"the value range achievable within growth_rate_search_bounds "
                f"{growth_rate_search_bounds} (EV range: {ev_lo:,.0f} to {ev_hi:,.0f}). "
                f"Widen growth_rate_search_bounds, or the target itself may be "
                f"unreachable with this model's other assumptions."
            ),
        }

    mid, ev_mid = lo, ev_lo
    converged = False
    iterations_used = 0
    for i in range(max_iterations):
        iterations_used = i + 1
        mid = (lo + hi) / 2
        ev_mid, err_mid, _ = _ev_at(mid)
        if err_mid or ev_mid is None:
            return {
                "implied_growth_rate_pct": None, "resulting_enterprise_value": None,
                "target_enterprise_value": target_enterprise_value, "converged": False,
                "iterations_used": iterations_used, "discount_rate_pct": discount_rate_pct,
                "assumptions": assumptions, "interpretation": "unavailable",
                "error": err_mid or "DCF evaluation failed mid-search.",
            }
        pct_off = abs(ev_mid - target_enterprise_value) / target_enterprise_value * 100
        if pct_off <= tolerance_pct:
            converged = True
            break
        if ev_mid < target_enterprise_value:
            lo = mid
        else:
            hi = mid

    return {
        "implied_growth_rate_pct":     round(mid, 2),
        "resulting_enterprise_value":  round(ev_mid, 2),
        "target_enterprise_value":     target_enterprise_value,
        "converged":                   converged,
        "iterations_used":             iterations_used,
        "discount_rate_pct":           discount_rate_pct,
        "assumptions":                 assumptions,
        "interpretation": (
            "This is the constant annual FCF growth rate (over "
            f"{projection_years} years, then {terminal_growth_rate_pct}% "
            "terminal growth) that reproduces the target enterprise value "
            "under this model's assumptions. It is a model-dependent market-implied "
            "growth estimate, not an absolute market truth. Compare it to the company's "
            "historical revenue/FCF CAGR and to plausible long-run growth "
            "for its sector to judge whether that's optimistic or "
            "conservative; this function only solves the equation, it "
            "doesn't judge plausibility." + ("" if converged else
            " NOTE: did not converge within max_iterations -- treat "
            "resulting_enterprise_value as an approximation, not exact."
            )
        ),
        "error": None,
    }


# ---------------------------------------------------------------------------
# 7. Composite Scoring
# ---------------------------------------------------------------------------

def composite_financial_score(
    pe:           float | None = None,
    pb:           float | None = None,
    roe_pct:      float | None = None,
    net_margin_pct: float | None = None,
    current_ratio:float | None = None,
    de_ratio:     float | None = None,
    revenue_cagr_pct: float | None = None,
) -> dict:
    """
    Compute a weighted composite financial health score (0–100).

    Each sub-score is normalised to a 0–10 scale and then weighted to produce
    a final score. The weighting scheme is:

    +-----------------------+--------+
    | Metric                | Weight |
    +-----------------------+--------+
    | ROE                   | 25 %   |
    | Net Margin            | 20 %   |
    | Revenue CAGR          | 20 %   |
    | P/E (valuation)       | 15 %   |
    | Current Ratio         | 10 %   |
    | D/E (solvency)        | 10 %   |
    +-----------------------+--------+

    Parameters
    ----------
    pe              : float | None   Trailing P/E ratio.
    pb              : float | None   Price-to-Book ratio (unused in weighting).
    roe_pct         : float | None   Return on Equity (%).
    net_margin_pct  : float | None   Net Profit Margin (%).
    current_ratio   : float | None   Current ratio.
    de_ratio        : float | None   Debt-to-Equity ratio.
    revenue_cagr_pct: float | None   Revenue CAGR (%).

    Returns
    -------
    dict
        - "score"          (float | None) : Composite score 0–100.
        - "grade"          (str)          : "A", "B", "C", "D", or "F".
        - "sub_scores"     (dict)         : Normalised 0–10 score per metric,
                                            keyed as "<metric>_normalised" to
                                            avoid being confused with the raw
                                            metric value under the same base
                                            name elsewhere in the state
                                            (e.g. "current_ratio_normalised"
                                            vs. the raw "current_ratio").
        - "missing_inputs" (list[str])    : Metrics that could not be scored.
    """
    sub_scores     = {}
    missing_inputs = []
    weighted_total = 0.0
    weight_used    = 0.0

    def _add(name: str, raw: float | None, min_val: float, max_val: float, weight: float,
             invert: bool = False) -> None:
        """Normalise *raw* to 0–10 and accumulate the weighted total."""
        nonlocal weighted_total, weight_used
        if raw is None:
            missing_inputs.append(name)
            return
        # Clamp and normalise
        clamped  = max(min_val, min(raw, max_val))
        norm     = (clamped - min_val) / (max_val - min_val) * 10
        if invert:
            norm = 10 - norm
        # Suffix intentionally distinguishes this normalised 0-10 score from
        # the raw metric of the same base name (e.g. "de_ratio_normalised"
        # vs. the raw "de_ratio" reported elsewhere in
        # financial_metrics_summary). See DC-4 postmortem: a prior version
        # stored this under the bare metric name, which let a downstream LLM
        # synthesis step cite the normalised score as if it were the actual
        # ratio (e.g. reporting "current ratio of 3.57" instead of 1.07).
        sub_scores[f"{name}_normalised"] = round(norm, 2)
        weighted_total  += norm * weight
        weight_used     += weight

    # --- Add each metric (invert=True means lower raw value is better) ---
    _add("roe",           roe_pct,           0, 40,  0.25)
    _add("net_margin",    net_margin_pct,    0, 30,  0.20)
    _add("revenue_cagr",  revenue_cagr_pct,  0, 50,  0.20)
    _add("pe",            pe,                5, 60,  0.15, invert=True)
    _add("current_ratio", current_ratio,     0,  3,  0.10)
    _add("de_ratio",      de_ratio,          0,  3,  0.10, invert=True)

    if weight_used == 0:
        return {"score": None, "grade": "N/A", "sub_scores": {}, "missing_inputs": missing_inputs}

    # Scale to 0–100 using only the weights for which data was available
    score = round((weighted_total / weight_used) * 10, 1)

    grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"

    return {
        "score":          score,
        "grade":          grade,
        "sub_scores":     sub_scores,
        "missing_inputs": missing_inputs,
    }

# ---------------------------------------------------------------------------
# 5. Discounted Cash Flow (DCF) Valuation
# ---------------------------------------------------------------------------

class ForecastModel:
    """Build auditable five-year operating forecasts before valuation.

    Historical revenue CAGR is deliberately only one bounded feature.  The
    model blends it with the current trajectory and then fades growth toward
    a maturity anchor.  It never passes historical CAGR through as a constant
    FCF growth rate.
    """

    def __init__(self, projection_years: int = 5) -> None:
        if projection_years < 2:
            raise ValueError("projection_years must be at least 2 to model growth fade")
        self.projection_years = projection_years

    @staticmethod
    def _bounded(value: float, low: float, high: float) -> float:
        return max(low, min(value, high))

    def build_scenarios(
        self,
        *,
        revenue_base: float,
        fcf_base: float,
        historical_revenue_cagr_pct: float | None,
        current_revenue_growth_pct: float | None,
        maturity_growth_anchor_pct: float = 5.0,
        operating_margin_pct: float | None = None,
        tax_rate_pct: float = 21.0,
        depreciation_amortization_pct_revenue: float = 3.0,
        capex_pct_revenue: float | None = None,
        working_capital_change_pct_revenue: float = 1.0,
        source_period: str = "annual_latest",
    ) -> dict[str, Any]:
        if revenue_base <= 0 or fcf_base <= 0:
            raise ValueError("revenue_base and fcf_base must be positive")

        anchor = self._bounded(float(maturity_growth_anchor_pct), 0.0, 10.0)
        historical = None if historical_revenue_cagr_pct is None else float(historical_revenue_cagr_pct)
        trajectory = current_revenue_growth_pct
        if trajectory is None:
            trajectory = historical if historical is not None else anchor

        # Current trajectory has the larger weight.  Historical CAGR is
        # winsorised and contributes only 25%, preventing a 100% historical
        # CAGR from becoming a 100% forecast.
        bounded_trajectory = self._bounded(float(trajectory), -20.0, 40.0)
        bounded_historical = self._bounded(
            historical if historical is not None else bounded_trajectory,
            -20.0,
            40.0,
        )
        year1_base = self._bounded(
            0.75 * bounded_trajectory + 0.25 * bounded_historical,
            -15.0,
            35.0,
        )

        fcf_margin = self._bounded(fcf_base / revenue_base * 100.0, 1.0, 60.0)
        op_margin = self._bounded(
            operating_margin_pct if operating_margin_pct is not None else fcf_margin + 5.0,
            1.0,
            70.0,
        )
        capex_pct = self._bounded(
            capex_pct_revenue if capex_pct_revenue is not None else 5.0,
            0.0,
            40.0,
        )

        scenario_specs = {
            "bear": {"year1_delta": -max(5.0, abs(year1_base - anchor) * 0.35), "anchor_delta": -2.0, "margin_factor": 0.85},
            "base": {"year1_delta": 0.0, "anchor_delta": 0.0, "margin_factor": 1.0},
            "bull": {"year1_delta": max(5.0, abs(year1_base - anchor) * 0.25), "anchor_delta": 2.0, "margin_factor": 1.10},
        }
        scenarios: dict[str, Any] = {}
        for name, spec in scenario_specs.items():
            start_growth = self._bounded(year1_base + spec["year1_delta"], -25.0, 45.0)
            end_growth = self._bounded(anchor + spec["anchor_delta"], -2.0, 12.0)
            revenue = float(revenue_base)
            yearly: list[dict[str, Any]] = []
            scenario_fcf_margin = self._bounded(fcf_margin * spec["margin_factor"], 0.5, 65.0)
            for index in range(self.projection_years):
                progress = index / (self.projection_years - 1)
                growth = start_growth + (end_growth - start_growth) * progress
                revenue *= 1.0 + growth / 100.0
                fcf = revenue * scenario_fcf_margin / 100.0
                yearly.append({
                    "year": index + 1,
                    "revenue_growth_pct": round(growth, 4),
                    "revenue": round(revenue, 2),
                    "operating_margin_pct": round(op_margin * spec["margin_factor"], 4),
                    "tax_rate_pct": round(tax_rate_pct, 4),
                    "depreciation_amortization": round(revenue * depreciation_amortization_pct_revenue / 100.0, 2),
                    "capex": round(revenue * capex_pct / 100.0, 2),
                    "working_capital_change": round(revenue * working_capital_change_pct_revenue / 100.0, 2),
                    "fcf_margin_pct": round(scenario_fcf_margin, 4),
                    "fcf": round(fcf, 2),
                })
            scenarios[name] = {
                "yearly_forecast": yearly,
                "fcf_by_year": [row["fcf"] for row in yearly],
            }

        assumptions = {
            "definition": "Explicit yearly revenue and FCF forecast with linear growth fade",
            "derivation": "75% bounded current trajectory + 25% bounded historical revenue CAGR; fades to maturity anchor",
            "historical_revenue_cagr_pct": historical,
            "current_revenue_growth_pct": current_revenue_growth_pct,
            "bounded_year1_base_growth_pct": round(year1_base, 4),
            "maturity_growth_anchor_pct": anchor,
            "source_period": source_period,
            "scenario": "bear/base/bull",
            "confidence": "medium" if current_revenue_growth_pct is not None else "low",
            "validation_status": "pending",
            "historical_cagr_used_as_direct_fcf_growth": False,
        }
        validation = ScenarioValidator.validate(scenarios, year1_base)
        assumptions["validation_status"] = "valid" if validation["is_valid"] else "invalid"
        return {"scenarios": scenarios, "assumptions": assumptions, "validation": validation}


class ScenarioValidator:
    """Deterministic economic and semantic validation for forecast scenarios."""

    @staticmethod
    def validate(scenarios: dict[str, Any], baseline_year1_growth_pct: float) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        required = ("bear", "base", "bull")
        if any(name not in scenarios for name in required):
            return {"is_valid": False, "errors": ["bear/base/bull scenarios are required"], "warnings": []}

        paths = {name: scenarios[name].get("yearly_forecast", []) for name in required}
        lengths = {len(path) for path in paths.values()}
        if len(lengths) != 1 or not lengths or 0 in lengths:
            errors.append("scenario forecast paths must be non-empty and equal length")
        else:
            for index in range(next(iter(lengths))):
                values = [paths[name][index].get("fcf") for name in required]
                if any(value is None for value in values) or not (values[0] < values[1] < values[2]):
                    errors.append(f"year {index + 1}: expected bear FCF < base FCF < bull FCF")
            bear_growth = paths["bear"][0].get("revenue_growth_pct")
            base_growth = paths["base"][0].get("revenue_growth_pct")
            if bear_growth is None or bear_growth >= base_growth or bear_growth >= baseline_year1_growth_pct:
                errors.append("bear scenario must deteriorate relative to base and baseline trajectory")
            for name, path in paths.items():
                growths = [row.get("revenue_growth_pct") for row in path]
                if len(growths) > 1 and all(g == growths[0] for g in growths[1:]) and abs(growths[0]) > 15:
                    errors.append(f"{name}: exceptional constant growth without fade")
                elif len(growths) > 1 and abs(growths[-1]) >= abs(growths[0]) and abs(growths[0]) > 15:
                    warnings.append(f"{name}: growth does not visibly fade")
        return {"is_valid": not errors, "errors": errors, "warnings": warnings}

def discounted_cash_flow(
    fcf_base: float,
    beta: float,
    shares_outstanding: float | None = None,
    fcf_growth_rate_pct: float = 8.0,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
    fcf_by_year: list[float] | None = None,
    market_cap: float | None = None,
    total_debt: float | None = None,
    cash: float | None = None,
    marketable_securities: float | None = None,
    lease_liabilities: float | None = None,
    minority_interest: float | None = None,
    diluted_shares: float | None = None,
    pre_tax_cost_of_debt_pct: float | None = None,
    effective_tax_rate_pct: float | None = None,
    input_assessments: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """
    Compute a simplified Discounted Cash Flow (DCF) valuation.

    IMPORTANT — simplifications and required disclosure:
      1. When market-value capital structure and debt-cost inputs are
         supplied, the discount rate is WACC: CAPM cost of equity plus
         after-tax cost of debt. Legacy callers without those inputs are
         explicitly labelled as cost-of-equity-only.
      2. Preferred callers supply an explicit yearly unlevered-FCF path.
         Scalar growth remains only as a backward-compatible adapter.
      3. Enterprise value is bridged to equity value using cash,
         marketable securities, debt, lease liabilities, and minority
         interest. Per-share value is withheld if this bridge is incomplete.
      4. fcf_growth_rate_pct, terminal_growth_rate_pct, risk_free_rate_pct,
         and market_risk_premium_pct are ASSUMPTIONS with sensible
         defaults (terminal growth ≈ long-run GDP growth, risk-free rate
         ≈ recent 10Y Treasury yield, equity risk premium ≈ long-run
         historical average) — not live market data. Callers should
         override them with current values where precision matters, and
         the output always echoes back exactly which assumptions were
         used so nothing is hidden.

    Parameters
    ----------
    fcf_base : float
        Most recent annual Free Cash Flow (operating cash flow − capex),
        in USD. The starting point for the projection.
    beta : float
        Equity beta (from Yahoo Finance / yahoo_ratios), used for CAPM.
    shares_outstanding : float | None
        Legacy share-count alias. Prefer diluted_shares; per-share output
        also requires a complete enterprise-to-equity bridge.
    fcf_growth_rate_pct : float
        Assumed annual FCF growth rate during the projection window, as
        a percentage (e.g. 8.0 for 8%). Default is a generic placeholder
        — callers should pass something grounded in the company's own
        revenue_cagr or analyst estimates where available.
    terminal_growth_rate_pct : float
        Perpetual growth rate applied after the projection window.
        Default 2.5% (rough long-run nominal GDP growth proxy).
    risk_free_rate_pct : float
        Risk-free rate for CAPM, as a percentage. Default 4.3%
        (approximate recent 10-year US Treasury yield — update this to
        the current value for precision; it is NOT fetched live).
    market_risk_premium_pct : float
        Equity market risk premium for CAPM, as a percentage. Default
        5.0% (commonly cited long-run historical average).
    projection_years : int
        Number of explicit projection years before the terminal value.
        Default 5.

    Returns
    -------
    dict
        - "enterprise_value"      (float | None)
        - "intrinsic_value_per_share" (float | None) — only if
          shares_outstanding was provided; see simplification #3 above.
        - "discount_rate_pct"     (float) — the CAPM cost-of-equity used.
        - "terminal_value"        (float)
        - "pv_of_terminal_value"  (float)
        - "projected_fcf"         (list[float]) — undiscounted, by year.
        - "assumptions"           (dict) — every input assumption used,
          echoed back for transparency.
        - "interpretation"        (str)
        - "error"                 (str | None)
    """
    assumptions = {
        "fcf_growth_rate_pct":      fcf_growth_rate_pct,
        "terminal_growth_rate_pct": terminal_growth_rate_pct,
        "risk_free_rate_pct":       risk_free_rate_pct,
        "market_risk_premium_pct":  market_risk_premium_pct,
        "projection_years":         projection_years,
        "forecast_mode":            "explicit_fcf_by_year" if fcf_by_year is not None else "legacy_constant_growth",
        "market_cap": market_cap,
        "total_debt": total_debt,
        "cash": cash,
        "marketable_securities": marketable_securities,
        "lease_liabilities": lease_liabilities,
        "minority_interest": minority_interest,
        "diluted_shares": diluted_shares,
        "pre_tax_cost_of_debt_pct": pre_tax_cost_of_debt_pct,
        "effective_tax_rate_pct": effective_tax_rate_pct,
        "input_assessments": input_assessments or {},
    }

    if fcf_base is None or fcf_base <= 0:
        return {
            "enterprise_value": None,
            "intrinsic_value_per_share": None,
            "discount_rate_pct": None,
            "terminal_value": None,
            "pv_of_terminal_value": None,
            "projected_fcf": [],
            "assumptions": assumptions,
            "interpretation": "unavailable",
            "error": (
                "fcf_base is missing, zero, or negative — cannot run a "
                "growth-based DCF off a negative or absent starting FCF."
            ),
        }

    if beta is None:
        return {
            "enterprise_value": None,
            "intrinsic_value_per_share": None,
            "discount_rate_pct": None,
            "terminal_value": None,
            "pv_of_terminal_value": None,
            "projected_fcf": [],
            "assumptions": assumptions,
            "interpretation": "unavailable",
            "error": "beta is required for the CAPM discount rate and was not provided.",
        }

    if not 0.0 <= beta <= 3.0:
        return {"enterprise_value": None, "intrinsic_value_per_share": None,
                "discount_rate_pct": None, "terminal_value": None,
                "pv_of_terminal_value": None, "projected_fcf": [],
                "assumptions": assumptions, "interpretation": "unavailable",
                "error": "beta must be between 0.0 and 3.0 for this valuation model."}
    if not 0.0 <= risk_free_rate_pct <= 15.0:
        return {"enterprise_value": None, "intrinsic_value_per_share": None,
                "discount_rate_pct": None, "terminal_value": None,
                "pv_of_terminal_value": None, "projected_fcf": [],
                "assumptions": assumptions, "interpretation": "unavailable",
                "error": "risk_free_rate_pct must be between 0% and 15%."}
    if not 1.0 <= market_risk_premium_pct <= 12.0:
        return {"enterprise_value": None, "intrinsic_value_per_share": None,
                "discount_rate_pct": None, "terminal_value": None,
                "pv_of_terminal_value": None, "projected_fcf": [],
                "assumptions": assumptions, "interpretation": "unavailable",
                "error": "market_risk_premium_pct must be between 1% and 12%."}
    if not -2.0 <= terminal_growth_rate_pct <= 5.0:
        return {"enterprise_value": None, "intrinsic_value_per_share": None,
                "discount_rate_pct": None, "terminal_value": None,
                "pv_of_terminal_value": None, "projected_fcf": [],
                "assumptions": assumptions, "interpretation": "unavailable",
                "error": "terminal_growth_rate_pct must be between -2% and 5%."}

    cost_of_equity_pct = risk_free_rate_pct + beta * market_risk_premium_pct
    complete_wacc = all(value is not None for value in
                        (market_cap, total_debt, pre_tax_cost_of_debt_pct, effective_tax_rate_pct))
    if complete_wacc:
        if market_cap <= 0 or total_debt < 0:
            return {"enterprise_value": None, "intrinsic_value_per_share": None,
                    "discount_rate_pct": None, "terminal_value": None,
                    "pv_of_terminal_value": None, "projected_fcf": [],
                    "assumptions": assumptions, "interpretation": "unavailable",
                    "error": "market_cap must be positive and total_debt cannot be negative."}
        if not 0.0 <= pre_tax_cost_of_debt_pct <= 25.0:
            return {"enterprise_value": None, "intrinsic_value_per_share": None,
                    "discount_rate_pct": None, "terminal_value": None,
                    "pv_of_terminal_value": None, "projected_fcf": [],
                    "assumptions": assumptions, "interpretation": "unavailable",
                    "error": "pre_tax_cost_of_debt_pct must be between 0% and 25%."}
        if not 0.0 <= effective_tax_rate_pct <= 50.0:
            return {"enterprise_value": None, "intrinsic_value_per_share": None,
                    "discount_rate_pct": None, "terminal_value": None,
                    "pv_of_terminal_value": None, "projected_fcf": [],
                    "assumptions": assumptions, "interpretation": "unavailable",
                    "error": "effective_tax_rate_pct must be between 0% and 50%."}
        capital = market_cap + total_debt
        equity_weight = market_cap / capital
        debt_weight = total_debt / capital
        after_tax_debt_pct = pre_tax_cost_of_debt_pct * (1 - effective_tax_rate_pct / 100.0)
        wacc_pct = equity_weight * cost_of_equity_pct + debt_weight * after_tax_debt_pct
        wacc_status = "COMPLETE"
    else:
        equity_weight, debt_weight, after_tax_debt_pct = 1.0, 0.0, None
        wacc_pct = cost_of_equity_pct
        wacc_status = "LEGACY_COST_OF_EQUITY_ONLY"
    wacc_components = {
        "status": wacc_status, "cost_of_equity_pct": round(cost_of_equity_pct, 2),
        "pre_tax_cost_of_debt_pct": pre_tax_cost_of_debt_pct,
        "after_tax_cost_of_debt_pct": round(after_tax_debt_pct, 2) if after_tax_debt_pct is not None else None,
        "effective_tax_rate_pct": effective_tax_rate_pct,
        "equity_weight_pct": round(equity_weight * 100, 2),
        "debt_weight_pct": round(debt_weight * 100, 2), "wacc_pct": round(wacc_pct, 2),
    }
    assumptions["wacc_components"] = wacc_components
    assumptions["valuation_assumption_metadata"] = {
        "wacc": {
            "definition": "Market-value weighted cost of equity and after-tax cost of debt.",
            "source_or_derivation": "CAPM plus debt cost weighted by market capitalization and total debt",
            "period": "current_market_and_ttm", "scenario": "all",
            "confidence": "high" if complete_wacc else "low",
            "validation_status": wacc_status,
        },
        "equity_bridge": {
            "definition": "EV + cash + marketable securities - debt - leases - minority interest",
            "source_or_derivation": "caller-supplied balance-sheet and debt-like claims",
            "period": "latest_available", "scenario": "all",
            "confidence": "pending_bridge_validation", "validation_status": "PENDING",
        },
        "terminal_growth": {
            "definition": "Perpetual unlevered FCF growth after the explicit forecast",
            "source_or_derivation": "caller assumption", "period": "terminal",
            "scenario": "all", "confidence": "model_assumption",
            "validation_status": "VALID" if wacc_pct > terminal_growth_rate_pct else "INVALID",
        },
    }
    discount_rate = wacc_pct / 100.0
    terminal_growth = terminal_growth_rate_pct / 100.0
    fcf_growth = fcf_growth_rate_pct / 100.0

    if discount_rate <= terminal_growth:
        return {
            "enterprise_value": None,
            "intrinsic_value_per_share": None,
            "discount_rate_pct": round(discount_rate * 100, 2),
            "terminal_value": None,
            "pv_of_terminal_value": None,
            "projected_fcf": [],
            "assumptions": assumptions,
            "interpretation": "unavailable",
            "error": (
                f"Discount rate ({discount_rate*100:.2f}%) must exceed terminal "
                f"growth rate ({terminal_growth_rate_pct:.2f}%) — Gordon Growth "
                "terminal value is undefined/negative otherwise. This usually "
                "means beta is unrealistically low for the assumptions given."
            ),
        }

    if fcf_by_year is not None:
        if not fcf_by_year or any(value is None or value <= 0 for value in fcf_by_year):
            return {
                "enterprise_value": None, "intrinsic_value_per_share": None,
                "discount_rate_pct": round(discount_rate * 100, 2), "terminal_value": None,
                "pv_of_terminal_value": None, "projected_fcf": [], "assumptions": assumptions,
                "interpretation": "unavailable", "error": "fcf_by_year must contain positive yearly FCF values.",
            }
        projection_years = len(fcf_by_year)
        assumptions["projection_years"] = projection_years

    # -- Project FCF and discount each year -----------------------------
    projected_fcf: list[float] = []
    pv_fcf_sum = 0.0
    fcf = fcf_base
    explicit_path = fcf_by_year if fcf_by_year is not None else []
    for year in range(1, projection_years + 1):
        fcf = explicit_path[year - 1] if explicit_path else fcf * (1 + fcf_growth)
        projected_fcf.append(round(fcf, 2))
        pv_fcf_sum += fcf / ((1 + discount_rate) ** year)

    # -- Terminal value (Gordon Growth from the final projected year) ---
    terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

    enterprise_value = pv_fcf_sum + pv_terminal_value

    bridge_inputs = {
        "cash": cash, "marketable_securities": marketable_securities,
        "total_debt": total_debt, "lease_liabilities": lease_liabilities,
        "minority_interest": minority_interest,
    }
    assessments = input_assessments or {}
    calculation_adjustments: dict[str, float] = {}
    blocking_classifications = {"CRITICAL_MISSING", "MATERIAL_MISSING"}
    equity_bridge_complete = True
    resolved_bridge: dict[str, float] = {}
    for name, value in bridge_inputs.items():
        if value is not None:
            resolved_bridge[name] = value
            continue
        finding = assessments.get(name, {})
        classification = finding.get("classification")
        if classification == "ESTIMATED_WITH_DISCLOSURE" and finding.get("assumed_value") is not None:
            resolved_bridge[name] = float(finding["assumed_value"])
            calculation_adjustments[name] = resolved_bridge[name]
        elif classification in {"NOT_APPLICABLE", "IMMATERIAL_MISSING"}:
            # Keep the source field as None; zero is only a disclosed calculation
            # adjustment after applicability/materiality assessment.
            resolved_bridge[name] = 0.0
            calculation_adjustments[name] = 0.0
        else:
            equity_bridge_complete = False
        if classification in blocking_classifications:
            equity_bridge_complete = False
    equity_value = None
    if equity_bridge_complete:
        equity_value = (enterprise_value + resolved_bridge["cash"] + resolved_bridge["marketable_securities"]
                        - resolved_bridge["total_debt"] - resolved_bridge["lease_liabilities"]
                        - resolved_bridge["minority_interest"])
    share_count = diluted_shares if diluted_shares is not None else shares_outstanding
    share_count_source = "diluted_shares" if diluted_shares is not None else (
        "legacy_shares_outstanding_proxy" if shares_outstanding is not None else "missing")
    intrinsic_value_per_share = (
        _safe_div(equity_value, share_count)
        if equity_value is not None and share_count is not None and share_count > 0 else None
    )
    terminal_share = _safe_div(pv_terminal_value, enterprise_value)
    terminal_share_pct = terminal_share * 100 if terminal_share is not None else None
    terminal_warning = (
        "Terminal value exceeds 75% of enterprise value; valuation is highly sensitive to WACC and terminal growth."
        if terminal_share_pct is not None and terminal_share_pct > 75 else None
    )
    equity_bridge = {
        "enterprise_value": round(enterprise_value), **bridge_inputs,
        "equity_value": round(equity_value) if equity_value is not None else None,
        "complete": equity_bridge_complete,
        "missing_inputs": [name for name, value in bridge_inputs.items() if value is None],
        "calculation_adjustments": calculation_adjustments,
        "input_assessments": {name: assessments.get(name) for name in bridge_inputs},
        "share_count": share_count, "share_count_source": share_count_source,
    }
    assessment_classes = {item.get("classification") for item in assessments.values()}
    valuation_confidence_cap = (
        "LOW" if assessment_classes & {"CRITICAL_MISSING", "MATERIAL_MISSING"}
        else "MEDIUM" if "ESTIMATED_WITH_DISCLOSURE" in assessment_classes
        else "HIGH"
    )
    assumptions["valuation_assumption_metadata"]["equity_bridge"]["confidence"] = (
        "medium" if equity_bridge_complete else "low"
    )
    assumptions["valuation_assumption_metadata"]["equity_bridge"]["validation_status"] = (
        "COMPLETE" if equity_bridge_complete else "INCOMPLETE"
    )

    return {
        "enterprise_value":           round(enterprise_value),
        "equity_value":               round(equity_value) if equity_value is not None else None,
        "intrinsic_value_per_share":  (
            round(intrinsic_value_per_share)
            if intrinsic_value_per_share is not None else None
        ),
        "discount_rate_pct":          round(discount_rate * 100, 2),
        "wacc_pct":                   round(wacc_pct, 2),
        "wacc_components":            wacc_components,
        "terminal_value":             round(terminal_value),
        "pv_of_terminal_value":       round(pv_terminal_value),
        "terminal_value_share_of_ev_pct": round(terminal_share_pct, 1),
        "terminal_value_warning":     terminal_warning,
        "projected_fcf":              projected_fcf,
        "equity_bridge":              equity_bridge,
        "equity_bridge_complete":     equity_bridge_complete,
        "input_assessments":          assessments,
        "valuation_precision":        "whole_dollars",
        "valuation_confidence_cap":   valuation_confidence_cap,
        "assumptions":                assumptions,
        "interpretation": (
            "Under this model's assumptions, unlevered FCF produces enterprise value. "
            "Equity and per-share values are emitted only when the equity bridge is complete."
        ),
        "error": None,
    }


def dcf_scenario_range(
    fcf_base: float,
    beta: float,
    base_growth_rate_pct: float | None = None,
    shares_outstanding: float | None = None,
    bear_growth_rate_pct: float | None = None,
    bull_growth_rate_pct: float | None = None,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
    forecast_scenarios: dict[str, Any] | None = None,
    market_cap: float | None = None,
    total_debt: float | None = None,
    cash: float | None = None,
    marketable_securities: float | None = None,
    lease_liabilities: float | None = None,
    minority_interest: float | None = None,
    diluted_shares: float | None = None,
    pre_tax_cost_of_debt_pct: float | None = None,
    effective_tax_rate_pct: float | None = None,
    input_assessments: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """
    Run discounted_cash_flow() three times at different FCF growth
    assumptions (bear / base / bull) instead of returning a single point
    estimate.

    WHY THIS EXISTS: a single-point DCF using trailing growth (e.g. a
    company's own historical revenue CAGR) routinely lands far below the
    current market price for a stock the market is pricing on much higher
    forward growth expectations (e.g. an AI-infrastructure supercycle
    narrative). That gap is not necessarily a modelling error — it can
    correctly show how much "growth premium" is embedded in the price —
    but presenting only ONE number invites exactly that misreading ("the
    model must be wrong, or the market must be about to crash"). Showing
    the value across a growth-rate range makes the sensitivity explicit
    instead of hiding it behind a single figure, and gives the Bull/Base/
    Bear narrative scenarios an actual numeric anchor (still not a
    probability — see the caller's guidance on that).

    Parameters
    ----------
    fcf_base, beta, shares_outstanding, terminal_growth_rate_pct,
    risk_free_rate_pct, market_risk_premium_pct, projection_years :
        Same as discounted_cash_flow().
    base_growth_rate_pct : float
        The "Base case" FCF growth assumption — typically the company's
        own trailing revenue/FCF CAGR.
    bear_growth_rate_pct : float | None
        Bear case growth. Defaults to half of base_growth_rate_pct
        (floored at terminal_growth_rate_pct + 0.5 to stay valid) if
        not provided.
    bull_growth_rate_pct : float | None
        Bull case growth. Defaults to 1.75x base_growth_rate_pct if not
        provided — a rough stand-in for "the market's more optimistic
        growth case", not a specific forecast.

    Returns
    -------
    dict
        - "bear", "base", "bull" (dict each) — full discounted_cash_flow()
          output for that growth assumption.
        - "note" (str) — explains what the range does and doesn't mean.
    """
    forecast_package: dict[str, Any] | None = None
    if forecast_scenarios is None:
        if base_growth_rate_pct is None:
            return {
                "bear": {}, "base": {}, "bull": {},
                "validation": {"is_valid": False, "errors": ["forecast_scenarios are required"], "warnings": []},
                "error": "Provide explicit forecast_scenarios (preferred) or legacy base_growth_rate_pct.",
            }
        # Backward-compatible adapter.  Unlike the old implementation this
        # creates a bounded, fading path and never multiplies one scalar by
        # 0.5/1.0/1.75 for five constant-growth years.
        forecast_package = ForecastModel(projection_years).build_scenarios(
            revenue_base=fcf_base / 0.20,
            fcf_base=fcf_base,
            historical_revenue_cagr_pct=base_growth_rate_pct,
            current_revenue_growth_pct=base_growth_rate_pct,
            maturity_growth_anchor_pct=max(3.0, min(7.0, terminal_growth_rate_pct + 2.0)),
            source_period="legacy_adapter",
        )
        forecast_scenarios = forecast_package["scenarios"]

    validation = ScenarioValidator.validate(
        forecast_scenarios,
        (
            forecast_scenarios.get("base", {}).get("yearly_forecast", [{}])[0]
            .get("revenue_growth_pct", 0.0)
        ),
    )
    if not validation["is_valid"]:
        return {
            "bear": {}, "base": {}, "bull": {}, "validation": validation,
            "error": "Scenario validation failed; valuation was not run.",
        }

    def _run(name: str) -> dict:
        scenario = forecast_scenarios[name]
        return discounted_cash_flow(
            fcf_base=fcf_base,
            beta=beta,
            shares_outstanding=shares_outstanding,
            terminal_growth_rate_pct=terminal_growth_rate_pct,
            risk_free_rate_pct=risk_free_rate_pct,
            market_risk_premium_pct=market_risk_premium_pct,
            projection_years=projection_years,
            fcf_by_year=scenario["fcf_by_year"],
            market_cap=market_cap,
            total_debt=total_debt,
            cash=cash,
            marketable_securities=marketable_securities,
            lease_liabilities=lease_liabilities,
            minority_interest=minority_interest,
            diluted_shares=diluted_shares,
            pre_tax_cost_of_debt_pct=pre_tax_cost_of_debt_pct,
            effective_tax_rate_pct=effective_tax_rate_pct,
            input_assessments=input_assessments,
        )

    return {
        "bear": _run("bear"),
        "base": _run("base"),
        "bull": _run("bull"),
        "forecast_scenarios": forecast_scenarios,
        "forecast_assumptions": forecast_package["assumptions"] if forecast_package else None,
        "validation": validation,
        "error": None,
        "note": (
            "Three DCF runs at different FCF growth assumptions (bear/base/"
            "bull), NOT weighted or averaged into a single 'expected value' "
            "— no probability is assigned to any of the three. A wide gap "
            "between the bull case and current market price indicates the "
            "market is pricing in growth beyond even the optimistic case "
            "modelled here; a narrow gap suggests current price is closer "
            "to what a strong-growth scenario would justify."
        ),
    }


def dcf_monte_carlo(
    fcf_base: float,
    beta: float,
    base_growth_rate_pct: float,
    shares_outstanding: float | None = None,
    growth_std_pct: float | None = None,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
    n_simulations: int = 2000,
    seed: int | None = None,
) -> dict:
    """
    Probabilistic DCF via Monte Carlo simulation on the FCF growth-rate
    assumption, producing a distribution of enterprise values (P10/P50/P90)
    instead of a single point or three fixed scenarios.

    SCOPE — what this does and does NOT model:
      - Randomises FCF growth_rate_pct only, sampled from a normal
        distribution centred on base_growth_rate_pct. Growth rate is the
        single biggest driver of variance in this DCF's output (see the
        bear/base/bull spread in dcf_scenario_range), so it's the highest-
        value variable to randomise first.
      - Does NOT randomise margin, discount rate (beta), or terminal
        growth — those are held fixed at the values given. A fuller
        multi-factor Monte Carlo (margin, discount rate, terminal growth
        all varying jointly, ideally with realistic correlations between
        them) is a larger extension than this function provides.
      - Growth samples are clamped to stay above
        (terminal_growth_rate_pct + 0.5) so every simulation run is valid
        (the underlying DCF formula is undefined when growth >= discount
        rate); simulations that would violate this are floored rather
        than discarded, which means the resulting distribution is not a
        pure unclamped normal — this is disclosed in the output, not
        hidden.
      - This is a genuinely wider, more honest picture of valuation
        uncertainty than a single point estimate, but it is NOT a
        calibrated statistical model — the growth_std_pct is a
        judgment-call input (default derived from the same spread used
        in dcf_scenario_range), not fit to historical forecast errors.

    Parameters
    ----------
    fcf_base, beta, shares_outstanding, terminal_growth_rate_pct,
    risk_free_rate_pct, market_risk_premium_pct, projection_years :
        Same as discounted_cash_flow().
    base_growth_rate_pct : float
        Mean of the growth-rate distribution — typically the company's
        own trailing revenue/FCF CAGR.
    growth_std_pct : float | None
        Standard deviation of the growth-rate distribution, in
        percentage points. If None, defaults to
        max(base_growth_rate_pct * 0.4, 2.0) — a moderate spread
        proportional to the base growth rate, floored at 2 percentage
        points so low-growth companies still get meaningful dispersion.
    n_simulations : int
        Number of Monte Carlo draws. Default 2000 (enough for stable
        percentile estimates without being slow — this runs in pure
        Python, no external dependencies).
    seed : int | None
        Optional RNG seed for reproducible results (e.g. in tests).

    Returns
    -------
    dict
        - "n_simulations"    (int) — successful simulation count.
        - "enterprise_value_p10/p50/p90" (float)
        - "intrinsic_value_per_share_p10/p50/p90" (float | None) — only
          if shares_outstanding was provided.
        - "growth_rate_mean_pct", "growth_rate_std_pct" (float) — echoed
          back for transparency.
        - "discount_rate_pct" (float) — fixed CAPM rate used throughout.
        - "note" (str) — scope disclosure, see above.
        - "error" (str | None)
    """
    if fcf_base is None or fcf_base <= 0:
        return {
            "n_simulations": 0,
            "enterprise_value_p10": None, "enterprise_value_p50": None, "enterprise_value_p90": None,
            "intrinsic_value_per_share_p10": None, "intrinsic_value_per_share_p50": None, "intrinsic_value_per_share_p90": None,
            "growth_rate_mean_pct": base_growth_rate_pct,
            "growth_rate_std_pct": growth_std_pct,
            "discount_rate_pct": None,
            "note": "",
            "error": "fcf_base is missing, zero, or negative — cannot run Monte Carlo DCF.",
        }
    if beta is None:
        return {
            "n_simulations": 0,
            "enterprise_value_p10": None, "enterprise_value_p50": None, "enterprise_value_p90": None,
            "intrinsic_value_per_share_p10": None, "intrinsic_value_per_share_p50": None, "intrinsic_value_per_share_p90": None,
            "growth_rate_mean_pct": base_growth_rate_pct,
            "growth_rate_std_pct": growth_std_pct,
            "discount_rate_pct": None,
            "note": "",
            "error": "beta is required for the CAPM discount rate and was not provided.",
        }

    if growth_std_pct is None:
        growth_std_pct = max(base_growth_rate_pct * 0.4, 2.0)

    rng = random.Random(seed)
    ev_samples: list[float] = []
    per_share_samples: list[float] = []
    discount_rate_pct = None
    clamped_count = 0

    floor_growth = terminal_growth_rate_pct + 0.5

    for _ in range(n_simulations):
        g = rng.gauss(base_growth_rate_pct, growth_std_pct)
        if g < floor_growth:
            g = floor_growth
            clamped_count += 1

        result = discounted_cash_flow(
            fcf_base=fcf_base,
            beta=beta,
            shares_outstanding=shares_outstanding,
            fcf_growth_rate_pct=g,
            terminal_growth_rate_pct=terminal_growth_rate_pct,
            risk_free_rate_pct=risk_free_rate_pct,
            market_risk_premium_pct=market_risk_premium_pct,
            projection_years=projection_years,
        )
        if result.get("error") is None:
            ev_samples.append(result["enterprise_value"])
            if result.get("intrinsic_value_per_share") is not None:
                per_share_samples.append(result["intrinsic_value_per_share"])
            if discount_rate_pct is None:
                discount_rate_pct = result["discount_rate_pct"]

    if not ev_samples:
        return {
            "n_simulations": 0,
            "enterprise_value_p10": None, "enterprise_value_p50": None, "enterprise_value_p90": None,
            "intrinsic_value_per_share_p10": None, "intrinsic_value_per_share_p50": None, "intrinsic_value_per_share_p90": None,
            "growth_rate_mean_pct": base_growth_rate_pct,
            "growth_rate_std_pct": growth_std_pct,
            "discount_rate_pct": None,
            "note": "",
            "error": "All simulations failed — beta may be unrealistically low relative to the assumptions given.",
        }

    def _percentile(sorted_vals: list[float], p: float) -> float:
        idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
        return round(sorted_vals[idx], 2)

    ev_samples.sort()
    result_dict = {
        "n_simulations": len(ev_samples),
        "enterprise_value_p10": _percentile(ev_samples, 0.10),
        "enterprise_value_p50": _percentile(ev_samples, 0.50),
        "enterprise_value_p90": _percentile(ev_samples, 0.90),
        "growth_rate_mean_pct": base_growth_rate_pct,
        "growth_rate_std_pct": growth_std_pct,
        "discount_rate_pct": discount_rate_pct,
        "note": (
            f"Monte Carlo over FCF growth rate only (mean={base_growth_rate_pct}%, "
            f"std={growth_std_pct}%), {n_simulations} simulations "
            f"({clamped_count} clamped to the minimum valid growth rate). "
            "Margin, discount rate, and terminal growth were held fixed — "
            "see function docstring for full scope disclosure. This shows "
            "the range of outcomes from growth-rate uncertainty alone, not "
            "a fully calibrated probabilistic valuation."
        ),
        "error": None,
    }
    result_dict["model_classification"] = "LIMITED"
    if per_share_samples:
        per_share_samples.sort()
        result_dict["intrinsic_value_per_share_p10"] = _percentile(per_share_samples, 0.10)
        result_dict["intrinsic_value_per_share_p50"] = _percentile(per_share_samples, 0.50)
        result_dict["intrinsic_value_per_share_p90"] = _percentile(per_share_samples, 0.90)
    else:
        result_dict["intrinsic_value_per_share_p10"] = None
        result_dict["intrinsic_value_per_share_p50"] = None
        result_dict["intrinsic_value_per_share_p90"] = None
    return result_dict


def dcf_multi_factor_monte_carlo(
    fcf_by_year: list[float], wacc_pct: float, terminal_growth_rate_pct: float,
    *, cash: float = 0.0, total_debt: float = 0.0,
    lease_liabilities: float = 0.0, minority_interest: float = 0.0,
    diluted_shares: float | None = None, n_simulations: int = 2000,
    seed: int | None = None,
) -> dict:
    """Multi-factor uncertainty over demand, conversion margin, WACC and terminal growth."""
    import random
    if not fcf_by_year or any(v is None or v <= 0 for v in fcf_by_year):
        return {"model_classification": "MULTI_FACTOR", "error": "positive explicit fcf_by_year is required"}
    if n_simulations < 100:
        return {"model_classification": "MULTI_FACTOR", "error": "n_simulations must be at least 100"}
    rng = random.Random(seed)
    evs: list[float] = []
    per_share: list[float] = []
    rejected = 0
    for _ in range(n_simulations):
        demand_z = rng.gauss(0, 1)
        independent_margin = rng.gauss(0, 1)
        margin_z = 0.60 * demand_z + (1 - 0.60 ** 2) ** 0.5 * independent_margin
        rate_z = rng.gauss(0, 1)
        terminal_z = -0.25 * rate_z + (1 - 0.25 ** 2) ** 0.5 * rng.gauss(0, 1)
        sampled_wacc = min(max(wacc_pct + 1.25 * rate_z, 5.0), 18.0)
        sampled_terminal = min(max(terminal_growth_rate_pct + 0.65 * terminal_z, -1.0), 5.0, sampled_wacc - 0.5)
        margin_multiplier = min(max(1.0 + 0.08 * margin_z, 0.70), 1.30)
        sampled_fcf = [max(value * margin_multiplier * (1.0 + 0.06 * demand_z * (1.0 - i / (2 * len(fcf_by_year)))), value * 0.50) for i, value in enumerate(fcf_by_year)]
        rate = sampled_wacc / 100.0
        terminal_rate = sampled_terminal / 100.0
        if rate <= terminal_rate:
            rejected += 1
            continue
        pv = sum(value / ((1 + rate) ** (i + 1)) for i, value in enumerate(sampled_fcf))
        terminal = sampled_fcf[-1] * (1 + terminal_rate) / (rate - terminal_rate)
        ev = pv + terminal / ((1 + rate) ** len(sampled_fcf))
        evs.append(ev)
        if diluted_shares and diluted_shares > 0:
            equity = ev + cash - total_debt - lease_liabilities - minority_interest
            per_share.append(equity / diluted_shares)
    evs.sort(); per_share.sort()
    def pct(values: list[float], p: float) -> float | None:
        if not values: return None
        idx = (len(values) - 1) * p; lo = int(idx); hi = min(lo + 1, len(values));
        return round(values[lo] + (values[hi] - values[lo]) * (idx - lo), 2)
    return {
        "model_classification": "MULTI_FACTOR", "n_simulations": len(evs),
        "enterprise_value_p10": pct(evs, .10), "enterprise_value_p50": pct(evs, .50), "enterprise_value_p90": pct(evs, .90),
        "intrinsic_value_per_share_p10": pct(per_share, .10), "intrinsic_value_per_share_p50": pct(per_share, .50), "intrinsic_value_per_share_p90": pct(per_share, .90),
        "stochastic_drivers": ["demand/growth", "cash-conversion margin", "WACC", "terminal growth"],
        "correlations": {"demand_margin": 0.60, "wacc_terminal_growth": -0.25},
        "bounds": {"wacc_pct": [5.0, 18.0], "terminal_growth_pct": [-1.0, 5.0], "margin_multiplier": [0.70, 1.30]},
        "rejected_draws": rejected, "seed": seed, "error": None,
        "note": "Multi-factor scenario distribution, not a calibrated forecast probability; correlations and volatilities are disclosed model assumptions.",
    }