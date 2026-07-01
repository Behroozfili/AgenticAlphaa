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
import math
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