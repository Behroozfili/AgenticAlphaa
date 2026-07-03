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

def discounted_cash_flow(
    fcf_base: float,
    beta: float,
    shares_outstanding: float | None = None,
    fcf_growth_rate_pct: float = 8.0,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
) -> dict:
    """
    Compute a simplified Discounted Cash Flow (DCF) valuation.

    IMPORTANT — simplifications and required disclosure:
      1. Discount rate is cost-of-equity via CAPM (risk_free_rate +
         beta * market_risk_premium), used as a stand-in for WACC. The
         cost of debt / capital-structure weighting component of a true
         WACC is NOT included — this pipeline does not have a reliable
         cost-of-debt or debt/equity weighting input (see the de_ratio
         approximation note in financial_agent.py). For a low-leverage
         company this is a reasonable approximation; for a highly
         levered one it will overstate the discount rate and understate
         value.
      2. FCF is projected at a CONSTANT growth rate for projection_years,
         then a terminal value is computed via Gordon Growth from the
         final projected year — no fade-down to terminal growth is
         modelled (a real DCF often tapers growth year by year; this is
         a simplified single-stage-then-terminal model).
      3. The result is an ENTERPRISE value, not equity value — no net
         debt adjustment (total debt − cash) is applied, because this
         pipeline does not have a reliable standalone cash figure (only
         total_assets, which includes far more than cash). Do not
         present "intrinsic_value_per_share" as a directly actionable
         target price without accounting for net debt separately.
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
        If provided, also computes per-share values. If None, only
        aggregate enterprise value is returned.
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

    discount_rate = (risk_free_rate_pct + beta * market_risk_premium_pct) / 100.0
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

    # -- Project FCF and discount each year -----------------------------
    projected_fcf: list[float] = []
    pv_fcf_sum = 0.0
    fcf = fcf_base
    for year in range(1, projection_years + 1):
        fcf = fcf * (1 + fcf_growth)
        projected_fcf.append(round(fcf, 2))
        pv_fcf_sum += fcf / ((1 + discount_rate) ** year)

    # -- Terminal value (Gordon Growth from the final projected year) ---
    terminal_value = (projected_fcf[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + discount_rate) ** projection_years)

    enterprise_value = pv_fcf_sum + pv_terminal_value

    intrinsic_value_per_share = (
        _safe_div(enterprise_value, shares_outstanding)
        if shares_outstanding else None
    )

    return {
        "enterprise_value":           round(enterprise_value, 2),
        "intrinsic_value_per_share":  (
            round(intrinsic_value_per_share, 2)
            if intrinsic_value_per_share is not None else None
        ),
        "discount_rate_pct":          round(discount_rate * 100, 2),
        "terminal_value":             round(terminal_value, 2),
        "pv_of_terminal_value":       round(pv_terminal_value, 2),
        "projected_fcf":              projected_fcf,
        "assumptions":                assumptions,
        "interpretation": (
            "This is an enterprise value estimate with simplified WACC "
            "(CAPM cost-of-equity only, no debt-cost component) and no "
            "net-debt adjustment to equity value — see function docstring "
            "for full disclosure of simplifications. Treat as a rough "
            "sanity-check range, not a precise target price."
        ),
        "error": None,
    }


def dcf_scenario_range(
    fcf_base: float,
    beta: float,
    base_growth_rate_pct: float,
    shares_outstanding: float | None = None,
    bear_growth_rate_pct: float | None = None,
    bull_growth_rate_pct: float | None = None,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
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
    if bear_growth_rate_pct is None:
        bear_growth_rate_pct = max(
            base_growth_rate_pct * 0.5, terminal_growth_rate_pct + 0.5
        )
    if bull_growth_rate_pct is None:
        bull_growth_rate_pct = base_growth_rate_pct * 1.75

    def _run(growth_pct: float) -> dict:
        return discounted_cash_flow(
            fcf_base=fcf_base,
            beta=beta,
            shares_outstanding=shares_outstanding,
            fcf_growth_rate_pct=growth_pct,
            terminal_growth_rate_pct=terminal_growth_rate_pct,
            risk_free_rate_pct=risk_free_rate_pct,
            market_risk_premium_pct=market_risk_premium_pct,
            projection_years=projection_years,
        )

    return {
        "bear": _run(bear_growth_rate_pct),
        "base": _run(base_growth_rate_pct),
        "bull": _run(bull_growth_rate_pct),
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