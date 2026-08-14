"""
server.py
---------
MCP (Model Context Protocol) server for the Financial Analyst Agent.

Built with FastMCP — the modern, decorator-based MCP server API.
Transport: stdio only (stdin → JSON-RPC requests, stdout → JSON-RPC responses).

Architecture
------------
The agent (client) launches this script as a subprocess and communicates
with it exclusively via stdin/stdout using the MCP JSON-RPC protocol.
No HTTP, no FastAPI, no web framework of any kind.

                ┌──────────────────────┐
                │   Financial Agent    │  ← LangGraph / Claude Desktop
                │      (client)        │
                └────────┬─────────────┘
                         │  stdin / stdout  (MCP JSON-RPC over stdio)
                ┌────────▼─────────────┐
                │     server.py        │
                │  FastMCP stdio server│
                └────────┬─────────────┘
                         │ Python function calls
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   yahoo_finance.py  sec_edgar.py  financial_ratio_calculator.py

Tool catalogue (39 tools)
--------------------------
Yahoo Finance  : get_price_history, get_financial_ratios,
                 get_revenue_growth, get_peer_comparison,
                 get_balance_sheet_snapshot
SEC EDGAR      : get_cik, list_filings, get_filing_text, get_xbrl_financials
Ratio Calc     : calc_pe, calc_pb, calc_ev_ebitda, calc_peg,
                 calc_gross_margin, calc_operating_margin, calc_net_margin,
                 calc_roe, calc_roa, calc_current_ratio, calc_quick_ratio,
                 calc_debt_to_equity, calc_interest_coverage,
                 calc_asset_turnover, calc_cagr,
                 calc_revenue_cagr_from_growth, calc_composite_score,
                 calc_dcf, calc_dcf_scenarios, calc_dcf_monte_carlo,
                 calc_net_debt_to_ebitda, calc_fcf_margin, calc_fcf_yield,
                 calc_ocf_ratio, calc_roic, calc_roce, calc_debt_ratio,
                 calc_equity_ratio, calc_altman_z_score, calc_reverse_dcf

Usage
-----
    python server.py

Dependencies
------------
    pip install "mcp[cli]" yfinance requests
"""

import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from core.observability import init_sentry, sentry_enabled

# ---------------------------------------------------------------------------
# FastMCP import — the modern MCP server API (no FastAPI, no HTTP)
# ---------------------------------------------------------------------------
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "ERROR: 'mcp' package not found.\n"
        "Install it with:  pip install 'mcp[cli]'\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Local tool imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from yahoo_finance import (
    get_price_history,
    get_financial_ratios,
    get_revenue_growth,
    get_peer_comparison,
    get_balance_sheet_snapshot,
)
from sec_edgar import (
    get_cik,
    list_filings,
    get_filing_text,
    get_xbrl_financials,
)
from financial_ratio_calculator import (
    price_to_earnings,
    price_to_book,
    ev_to_ebitda,
    peg_ratio,
    gross_margin,
    operating_margin,
    net_margin,
    return_on_equity,
    return_on_assets,
    current_ratio,
    quick_ratio,
    debt_to_equity,
    interest_coverage,
    asset_turnover,
    cagr,
    compute_revenue_cagr_from_growth,
    composite_financial_score,
    ForecastModel,
    discounted_cash_flow,
    dcf_scenario_range,
    dcf_monte_carlo,
    dcf_multi_factor_monte_carlo,
    net_debt_to_ebitda,
    fcf_margin,
    fcf_yield,
    operating_cash_flow_ratio,
    return_on_invested_capital,
    return_on_capital_employed,
    debt_ratio,
    equity_ratio,
    altman_z_score,
    reverse_dcf,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    # Log to stderr so it does NOT pollute the stdio MCP channel
    stream=sys.stderr,
)
log = logging.getLogger("financial-mcp-server")

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="financial-analyst-agent",
    instructions="MCP server exposing financial analysis tools for the Financial Analyst Agent.",
    
)


@mcp.tool()
def tool_calc_dcf_multi_factor_monte_carlo(
    fcf_by_year: list[float], wacc_pct: float, terminal_growth_rate_pct: float,
    cash: float = 0.0, total_debt: float = 0.0, lease_liabilities: float = 0.0,
    minority_interest: float = 0.0, diluted_shares: float | None = None,
    n_simulations: int = 2000, seed: int | None = None,
) -> dict:
    return dcf_multi_factor_monte_carlo(
        fcf_by_year, wacc_pct, terminal_growth_rate_pct, cash=cash,
        total_debt=total_debt, lease_liabilities=lease_liabilities,
        minority_interest=minority_interest, diluted_shares=diluted_shares,
        n_simulations=n_simulations, seed=seed,
    )


def _sentry_capture(tool_name: str, exc: Exception) -> None:
    """Capture an exception to Sentry tagged with the tool name."""
    if sentry_enabled():
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("tool", tool_name)
            scope.set_tag("server", "financial-agent-mcp")
            sentry_sdk.capture_exception(exc)


def _sentry_tool(tool_name: str, fn, *args, **kwargs):
    """
    Call a tool function and capture any exception to Sentry before re-raising.
    Returns {"error": str(exc)} on failure so FastMCP can serialize it.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.exception("Tool %s failed: %s", tool_name, exc)
        _sentry_capture(tool_name, exc)
        return {"error": str(exc), "tool": tool_name}


# ===========================================================================
# Yahoo Finance tools
# ===========================================================================

@mcp.tool()
def tool_get_price_history(ticker: str, period: str = "1y") -> dict:
    """
    Retrieve historical OHLCV price data for a stock ticker.

    Fetches Open, High, Low, Close, and Volume data for the requested period
    using the Yahoo Finance API via the yfinance library.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA', 'AAPL', 'MSFT'.
        period: Time period for the data. Valid values:
                '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y',
                '10y', 'ytd', 'max'. Defaults to '1y'.

    Returns:
        dict with keys:
          - ticker      (str)        Normalised ticker symbol.
          - period      (str)        Requested period.
          - records     (list[dict]) OHLCV rows keyed by ISO-8601 date.
          - start_date  (str | None) Earliest date in the result set.
          - end_date    (str | None) Latest date in the result set.
          - error       (str | None) Error message if the request failed.
    """
    log.info("tool_get_price_history called: ticker=%s period=%s", ticker, period)
    return get_price_history(ticker, period)


@mcp.tool()
def tool_get_financial_ratios(ticker: str) -> dict:
    """
    Retrieve key financial ratios and valuation metrics for a stock ticker.

    Fetches a comprehensive set of ratios including P/E, P/B, EV/EBITDA,
    EPS (trailing and forward), dividend yield, beta, 52-week range,
    market cap, sector, and industry classification.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA', 'AAPL'.

    Returns:
        dict with keys:
          - ticker, company_name, sector, industry
          - market_cap, pe_ratio, forward_pe, peg_ratio
          - price_to_book, price_to_sales, enterprise_value, ev_to_ebitda
          - eps_trailing, eps_forward, dividend_yield, beta
          - 52w_high, 52w_low, current_price
          - error (str | None)
    """
    log.info("tool_get_financial_ratios called: ticker=%s", ticker)
    return _sentry_tool("tool_get_financial_ratios", get_financial_ratios, ticker)


@mcp.tool()
def tool_get_revenue_growth(ticker: str) -> dict:
    """
    Retrieve annual and quarterly revenue and net income growth metrics.

    Pulls income statement data and computes year-over-year (YoY) growth
    rates for revenue and net income across annual and quarterly periods.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        dict with keys:
          - ticker                     (str)
          - annual_revenue             (list[dict]) [{year, revenue, yoy_growth}]
          - quarterly_revenue          (list[dict]) [{quarter, revenue, yoy_growth}]
          - annual_net_income          (list[dict]) [{year, net_income, yoy_growth}]
          - revenue_growth_ttm         (float | None) NOT a true trailing-12-
                                        month figure — this is Yahoo Finance's
                                        info["revenueGrowth"], which is the
                                        YoY growth of the MOST RECENT QUARTER
                                        only (a known Yahoo/yfinance naming
                                        quirk). Do not treat this as
                                        comparable to annual_revenue's
                                        yoy_growth values without accounting
                                        for the period mismatch.
          - revenue_growth_ttm_period  (str) Always "most_recent_quarter_yoy" —
                                        explicit label for the above caveat.
          - error                      (str | None)

        Growth values are decimals (e.g. 1.22 = +122% growth).
    """
    log.info("tool_get_revenue_growth called: ticker=%s", ticker)
    return _sentry_tool("tool_get_revenue_growth", get_revenue_growth, ticker)


@mcp.tool()
def tool_get_peer_comparison(ticker: str, peers: list[str] | None = None) -> dict:
    """
    Compare a stock's financial ratios against a list of peer companies.

    Retrieves key ratios for the primary ticker and each peer, then
    computes peer-average benchmarks for numeric ratio fields.

    Args:
        ticker: Primary stock ticker symbol.
        peers:  List of peer ticker symbols, e.g. ['AMD', 'INTC', 'QCOM'].
                Maximum 10 peers per call to avoid Yahoo Finance rate limits.

    Returns:
        dict with keys:
          - primary  (dict)        Ratios for the primary ticker.
          - peers    (list[dict])  Ratio dicts for each peer.
          - summary  (dict)        Peer-average values per ratio field.
          - error    (str | None)
    """
    log.info("tool_get_peer_comparison called: ticker=%s peers=%s", ticker, peers)
    return get_peer_comparison(ticker, peers)


@mcp.tool()
def tool_get_balance_sheet_snapshot(ticker: str) -> dict:
    """
    Retrieve a most-recent-period balance-sheet + EBIT snapshot for a
    ticker: total_assets, total_liabilities, total_equity, current_assets,
    current_liabilities, working_capital, retained_earnings, ebit.

    Feeds the "Phase 2" ratios that need balance-sheet fields NOT present
    in tool_get_financial_ratios' output (which only reads yfinance's flat
    .info dict): tool_calc_roic, tool_calc_roce, tool_calc_debt_ratio,
    tool_calc_equity_ratio, tool_calc_altman_z_score.

    Args:
        ticker: Stock ticker symbol, e.g. 'NVDA', 'AAPL'.

    Returns:
        dict with keys:
          - ticker, period_end (str | None — ISO date of the balance-sheet
            column used; check this is recent enough for your purpose)
          - total_assets, total_liabilities, total_equity,
            current_assets, current_liabilities, working_capital,
            retained_earnings, ebit  (float | None each)
          - error (str | None)
    """
    log.info("tool_get_balance_sheet_snapshot called: ticker=%s", ticker)
    return _sentry_tool("tool_get_balance_sheet_snapshot", get_balance_sheet_snapshot, ticker)


# ===========================================================================
# SEC EDGAR tools
# ===========================================================================

@mcp.tool()
def tool_get_cik(ticker: str) -> dict:
    """
    Resolve a stock ticker symbol to its SEC Central Index Key (CIK).

    The CIK is required by all other SEC EDGAR tools. Uses the EDGAR
    company-tickers index endpoint (no API key required).

    Args:
        ticker: Stock ticker symbol (case-insensitive).

    Returns:
        dict with keys:
          - ticker        (str)        Normalised ticker (upper-case).
          - cik           (str)        10-digit zero-padded CIK.
          - company_name  (str)        Official company name in EDGAR.
          - error         (str | None)

    Example:
        tool_get_cik('NVDA') → {'cik': '0001045810', 'company_name': 'NVIDIA CORP', ...}
    """
    log.info("tool_get_cik called: ticker=%s", ticker)
    return get_cik(ticker)


@mcp.tool()
def tool_list_filings(ticker: str, form_type: str = "10-K", limit: int = 5) -> dict:
    """
    List the most recent SEC EDGAR filings of a given form type for a ticker.

    Searches the EDGAR submissions API for filings matching the requested
    form type and returns metadata including dates and document URLs.

    Args:
        ticker:    Stock ticker symbol.
        form_type: SEC form type to filter by. Common values: '10-K'
                   (annual report), '10-Q' (quarterly report), '8-K'
                   (current/material event). Defaults to '10-K'.
        limit:     Maximum number of filings to return. Defaults to 5.

    Returns:
        dict with keys:
          - ticker    (str)
          - cik       (str)
          - form_type (str)
          - filings   (list[dict]) Each entry contains:
              - accession_number (str) EDGAR accession number (with dashes).
              - filing_date      (str) Date filed (YYYY-MM-DD).
              - report_date      (str) Period of report (YYYY-MM-DD).
              - document_url     (str) URL to the filing index page.
          - error     (str | None)
    """
    log.info("tool_list_filings called: ticker=%s form_type=%s limit=%d", ticker, form_type, limit)
    return list_filings(ticker, form_type, limit)


@mcp.tool()
def tool_get_filing_text(accession_number: str, cik: str) -> dict:
    """
    Download and extract the plain-text content of a specific SEC filing.

    Fetches the primary document for the given accession number, strips
    HTML/XBRL tags, and returns clean text suitable for chunking and
    embedding into a vector store.

    Args:
        accession_number: EDGAR accession number with dashes,
                          e.g. '0001045810-24-000010'.
                          Obtain this from tool_list_filings.
        cik:              10-digit zero-padded CIK of the filer.
                          Obtain this from tool_get_cik.

    Returns:
        dict with keys:
          - accession_number  (str)
          - cik               (str)
          - text              (str | None) Clean plain text of the filing.
          - word_count        (int)        Approximate word count.
          - error             (str | None)

    Note:
        10-K filings can be very large (100k–500k words). Chunk the
        returned text before passing it to an embedding model.
        Parsing fails for ~8% of non-standard PDF-based filings.
    """
    log.info("tool_get_filing_text called: accession=%s cik=%s", accession_number, cik)
    return get_filing_text(accession_number, cik)


@mcp.tool()
def tool_get_xbrl_financials(ticker: str) -> dict:
    """
    Retrieve structured financial statement data from the SEC EDGAR XBRL API.

    Fetches machine-readable XBRL company facts including annual revenue,
    net income, total assets, and total liabilities across all reported periods.
    No parsing of PDF documents is involved.

    Args:
        ticker: Stock ticker symbol.

    Returns:
        dict with keys:
          - ticker                (str)
          - cik                   (str)
          - revenue_annual        (list[dict]) [{period_end, value, unit}]
          - net_income_annual     (list[dict])
          - total_assets          (list[dict])
          - total_liabilities     (list[dict])
          - error                 (str | None)

        Values are in USD. Only the 10 most recent annual data points
        are returned per metric.
    """
    log.info("tool_get_xbrl_financials called: ticker=%s", ticker)
    return _sentry_tool("tool_get_xbrl_financials", get_xbrl_financials, ticker)


# ===========================================================================
# Financial Ratio Calculator tools
# ===========================================================================

@mcp.tool()
def tool_calc_pe(price: float, eps: float) -> dict:
    """
    Calculate the Price-to-Earnings (P/E) ratio and classify its valuation.

    Args:
        price: Current market price per share (USD).
        eps:   Earnings Per Share — trailing twelve months (TTM).

    Returns:
        dict with keys:
          - pe_ratio       (float | None)
          - interpretation (str) 'undervalued' | 'fairly_valued' | 'overvalued'
                                 | 'negative_earnings'
          - formula        (str) Human-readable formula.
    """
    log.info("tool_calc_pe called: price=%s eps=%s", price, eps)
    return price_to_earnings(price, eps)


@mcp.tool()
def tool_calc_pb(price: float, book_value_per_share: float) -> dict:
    """
    Calculate the Price-to-Book (P/B) ratio.

    Args:
        price:                Current market price per share (USD).
        book_value_per_share: Book value (total equity / shares outstanding)
                              per share (USD).

    Returns:
        dict with keys:
          - pb_ratio       (float | None)
          - interpretation (str) 'trading_below_book' | 'fairly_valued'
                                 | 'premium_to_book'
          - formula        (str)
    """
    log.info("tool_calc_pb called: price=%s bvps=%s", price, book_value_per_share)
    return price_to_book(price, book_value_per_share)


@mcp.tool()
def tool_calc_ev_ebitda(enterprise_value: float, ebitda: float) -> dict:
    """
    Calculate the Enterprise Value to EBITDA multiple.

    Args:
        enterprise_value: Total EV (market cap + debt − cash) in USD.
        ebitda:           Earnings Before Interest, Taxes, Depreciation
                          & Amortisation (TTM) in USD.

    Returns:
        dict with keys:
          - ev_ebitda      (float | None)
          - interpretation (str) 'undervalued' | 'fairly_valued' | 'expensive'
          - formula        (str)
    """
    log.info("tool_calc_ev_ebitda called: ev=%s ebitda=%s", enterprise_value, ebitda)
    return ev_to_ebitda(enterprise_value, ebitda)


@mcp.tool()
def tool_calc_peg(pe: float, earnings_growth_rate_pct: float) -> dict:
    """
    Calculate the PEG (Price/Earnings-to-Growth) ratio.

    Args:
        pe:                       The trailing or forward P/E ratio.
        earnings_growth_rate_pct: Expected annual EPS growth rate as a
                                  percentage (e.g. 25 for 25%).

    Returns:
        dict with keys:
          - peg            (float | None)
          - interpretation (str) 'undervalued' | 'fairly_valued' | 'overvalued'
          - formula        (str)
    """
    log.info("tool_calc_peg called: pe=%s growth=%s", pe, earnings_growth_rate_pct)
    return peg_ratio(pe, earnings_growth_rate_pct)


@mcp.tool()
def tool_calc_gross_margin(revenue: float, cogs: float) -> dict:
    """
    Calculate the Gross Profit Margin.

    Args:
        revenue: Total revenue — TTM (USD).
        cogs:    Cost of Goods Sold — TTM (USD).

    Returns:
        dict with keys:
          - gross_margin_pct (float | None) Percentage, e.g. 62.5 for 62.5%.
          - interpretation   (str) 'excellent' | 'good' | 'moderate' | 'low'
          - formula          (str)
    """
    log.info("tool_calc_gross_margin called: revenue=%s cogs=%s", revenue, cogs)
    return gross_margin(revenue, cogs)


@mcp.tool()
def tool_calc_operating_margin(operating_income: float, revenue: float) -> dict:
    """
    Calculate the Operating Profit Margin (EBIT margin).

    Args:
        operating_income: EBIT — Earnings Before Interest & Taxes (USD).
        revenue:          Total revenue (USD).

    Returns:
        dict with keys:
          - operating_margin_pct (float | None)
          - interpretation       (str)
          - formula              (str)
    """
    log.info("tool_calc_operating_margin called: ebit=%s revenue=%s", operating_income, revenue)
    return operating_margin(operating_income, revenue)


@mcp.tool()
def tool_calc_net_margin(net_income: float, revenue: float) -> dict:
    """
    Calculate the Net Profit Margin.

    Args:
        net_income: Net income — TTM (USD).
        revenue:    Total revenue — TTM (USD).

    Returns:
        dict with keys:
          - net_margin_pct (float | None)
          - interpretation (str)
          - formula        (str)
    """
    log.info("tool_calc_net_margin called: net_income=%s revenue=%s", net_income, revenue)
    return net_margin(net_income, revenue)


@mcp.tool()
def tool_calc_roe(net_income: float, shareholders_equity: float) -> dict:
    """
    Calculate Return on Equity (ROE).

    Args:
        net_income:          Net income — TTM (USD).
        shareholders_equity: Average total shareholders' equity (USD).

    Returns:
        dict with keys:
          - roe_pct        (float | None)
          - interpretation (str) 'excellent' | 'good' | 'moderate' | 'low'
          - formula        (str)
    """
    log.info("tool_calc_roe called: net_income=%s equity=%s", net_income, shareholders_equity)
    return return_on_equity(net_income, shareholders_equity)


@mcp.tool()
def tool_calc_roa(net_income: float, total_assets: float) -> dict:
    """
    Calculate Return on Assets (ROA).

    Args:
        net_income:   Net income — TTM (USD).
        total_assets: Average total assets (USD).

    Returns:
        dict with keys:
          - roa_pct        (float | None)
          - interpretation (str)
          - formula        (str)
    """
    log.info("tool_calc_roa called: net_income=%s assets=%s", net_income, total_assets)
    return return_on_assets(net_income, total_assets)


@mcp.tool()
def tool_calc_current_ratio(current_assets: float, current_liabilities: float) -> dict:
    """
    Calculate the Current Ratio (short-term liquidity indicator).

    Args:
        current_assets:      Total current assets (USD).
        current_liabilities: Total current liabilities (USD).

    Returns:
        dict with keys:
          - current_ratio  (float | None)
          - interpretation (str) 'strong' (>=2) | 'adequate' (1-2) | 'weak' (<1)
          - formula        (str)
    """
    log.info("tool_calc_current_ratio called: assets=%s liabilities=%s",
             current_assets, current_liabilities)
    return current_ratio(current_assets, current_liabilities)


@mcp.tool()
def tool_calc_quick_ratio(
    cash: float,
    short_term_investments: float,
    receivables: float,
    current_liabilities: float,
) -> dict:
    """
    Calculate the Quick Ratio (acid-test ratio), excluding inventory.

    Args:
        cash:                   Cash and cash equivalents (USD).
        short_term_investments: Short-term marketable securities (USD).
        receivables:            Net accounts receivable (USD).
        current_liabilities:    Total current liabilities (USD).

    Returns:
        dict with keys:
          - quick_ratio    (float | None)
          - interpretation (str) 'strong' | 'moderate' | 'weak'
          - formula        (str)
    """
    log.info("tool_calc_quick_ratio called")
    return quick_ratio(cash, short_term_investments, receivables, current_liabilities)


@mcp.tool()
def tool_calc_debt_to_equity(total_debt: float, shareholders_equity: float) -> dict:
    """
    Calculate the Debt-to-Equity (D/E) ratio.

    Args:
        total_debt:          Total long-term + short-term debt (USD).
        shareholders_equity: Total shareholders' equity (USD).

    Returns:
        dict with keys:
          - de_ratio       (float | None)
          - interpretation (str) 'low_leverage' | 'moderate_leverage'
                                 | 'high_leverage'
          - formula        (str)
    """
    log.info("tool_calc_debt_to_equity called: debt=%s equity=%s", total_debt, shareholders_equity)
    return debt_to_equity(total_debt, shareholders_equity)


@mcp.tool()
def tool_calc_dcf(
    fcf_base: float,
    beta: float,
    shares_outstanding: float | None = None,
    fcf_growth_rate_pct: float = 8.0,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
    market_cap: float | None = None,
    total_debt: float | None = None,
    cash: float | None = None,
    marketable_securities: float | None = None,
    lease_liabilities: float | None = None,
    minority_interest: float | None = None,
    diluted_shares: float | None = None,
    pre_tax_cost_of_debt_pct: float | None = None,
    effective_tax_rate_pct: float | None = None,
    input_assessments: dict | None = None,
) -> dict:
    """
    Compute a simplified Discounted Cash Flow (DCF) valuation.

    Uses market-value WACC when capital-structure inputs are supplied and
    bridges enterprise value to equity value only when every bridge input is
    explicit. Legacy callers are retained but labelled as incomplete.

    Args:
        fcf_base:                  Most recent annual Free Cash Flow
                                    (operating cash flow − capex), USD.
                                    Get both from tool_get_xbrl_financials'
                                    operating_cash_flow_annual / capex_annual.
        beta:                      Equity beta (from tool_get_financial_ratios).
        shares_outstanding:        Optional — enables per-share output.
        fcf_growth_rate_pct:       Assumed FCF growth during projection window.
                                    Consider grounding this in the company's
                                    own revenue_cagr rather than the default.
        terminal_growth_rate_pct:  Perpetual growth after projection window.
        risk_free_rate_pct:        CAPM risk-free rate — update to current
                                    10-year Treasury yield for precision.
        market_risk_premium_pct:   CAPM equity risk premium.
        projection_years:          Number of explicit projection years.

    Returns:
        dict with keys:
          - enterprise_value, intrinsic_value_per_share (float | None)
          - discount_rate_pct, terminal_value, pv_of_terminal_value
          - projected_fcf (list[float])
          - assumptions (dict) — every assumption used, echoed back
          - interpretation (str), error (str | None)
    """
    log.info(
        "tool_calc_dcf called: fcf_base=%s beta=%s shares_outstanding=%s",
        fcf_base, beta, shares_outstanding,
    )
    return discounted_cash_flow(
        fcf_base=fcf_base,
        beta=beta,
        shares_outstanding=shares_outstanding,
        fcf_growth_rate_pct=fcf_growth_rate_pct,
        terminal_growth_rate_pct=terminal_growth_rate_pct,
        risk_free_rate_pct=risk_free_rate_pct,
        market_risk_premium_pct=market_risk_premium_pct,
        projection_years=projection_years,
        market_cap=market_cap, total_debt=total_debt, cash=cash,
        marketable_securities=marketable_securities,
        lease_liabilities=lease_liabilities, minority_interest=minority_interest,
        diluted_shares=diluted_shares,
        pre_tax_cost_of_debt_pct=pre_tax_cost_of_debt_pct,
        effective_tax_rate_pct=effective_tax_rate_pct,
        input_assessments=input_assessments,
    )


@mcp.tool()
def tool_build_forecast(
    revenue_base: float,
    fcf_base: float,
    historical_revenue_cagr_pct: float | None = None,
    current_revenue_growth_pct: float | None = None,
    maturity_growth_anchor_pct: float = 5.0,
    operating_margin_pct: float | None = None,
    tax_rate_pct: float = 21.0,
    capex_pct_revenue: float | None = None,
    projection_years: int = 5,
    source_period: str = "annual_latest",
) -> dict:
    """Build explicit bear/base/bull yearly operating and FCF forecasts."""
    return ForecastModel(projection_years=projection_years).build_scenarios(
        revenue_base=revenue_base,
        fcf_base=fcf_base,
        historical_revenue_cagr_pct=historical_revenue_cagr_pct,
        current_revenue_growth_pct=current_revenue_growth_pct,
        maturity_growth_anchor_pct=maturity_growth_anchor_pct,
        operating_margin_pct=operating_margin_pct,
        tax_rate_pct=tax_rate_pct,
        capex_pct_revenue=capex_pct_revenue,
        source_period=source_period,
    )


@mcp.tool()
def tool_calc_dcf_scenarios(
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
    forecast_scenarios: dict | None = None,
    market_cap: float | None = None,
    total_debt: float | None = None,
    cash: float | None = None,
    marketable_securities: float | None = None,
    lease_liabilities: float | None = None,
    minority_interest: float | None = None,
    diluted_shares: float | None = None,
    pre_tax_cost_of_debt_pct: float | None = None,
    effective_tax_rate_pct: float | None = None,
    input_assessments: dict | None = None,
) -> dict:
    """
    Run DCF at three FCF growth assumptions (bear/base/bull) instead of a
    single point estimate. Prefer this over tool_calc_dcf for report
    generation — a single-point DCF using trailing growth routinely lands
    far below market price for a high-growth stock, which reads as "the
    model is broken" unless the growth-rate sensitivity is shown
    explicitly. See dcf_scenario_range()'s docstring for full rationale.

    Args:
        fcf_base, beta, shares_outstanding, terminal_growth_rate_pct,
        risk_free_rate_pct, market_risk_premium_pct, projection_years:
            Same as tool_calc_dcf.
        base_growth_rate_pct: Base case growth — typically the company's
            own trailing revenue/FCF CAGR (e.g. from tool_calc_cagr).
        bear_growth_rate_pct: Optional override; defaults to ~half of base.
        bull_growth_rate_pct: Optional override; defaults to ~1.75x base.

    Returns:
        dict with keys "bear", "base", "bull" (each a full DCF result
        dict) and "note" explaining how to read the range.
    """
    log.info(
        "tool_calc_dcf_scenarios called: fcf_base=%s beta=%s base_growth=%s",
        fcf_base, beta, base_growth_rate_pct,
    )
    return dcf_scenario_range(
        fcf_base=fcf_base,
        beta=beta,
        base_growth_rate_pct=base_growth_rate_pct,
        shares_outstanding=shares_outstanding,
        bear_growth_rate_pct=bear_growth_rate_pct,
        bull_growth_rate_pct=bull_growth_rate_pct,
        terminal_growth_rate_pct=terminal_growth_rate_pct,
        risk_free_rate_pct=risk_free_rate_pct,
        market_risk_premium_pct=market_risk_premium_pct,
        projection_years=projection_years,
        forecast_scenarios=forecast_scenarios,
        market_cap=market_cap, total_debt=total_debt, cash=cash,
        marketable_securities=marketable_securities,
        lease_liabilities=lease_liabilities, minority_interest=minority_interest,
        diluted_shares=diluted_shares,
        pre_tax_cost_of_debt_pct=pre_tax_cost_of_debt_pct,
        effective_tax_rate_pct=effective_tax_rate_pct,
        input_assessments=input_assessments,
    )


@mcp.tool()
def tool_calc_dcf_monte_carlo(
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
) -> dict:
    """
    Probabilistic DCF via Monte Carlo simulation on the FCF growth rate,
    returning P10/P50/P90 percentiles instead of a single point or three
    fixed scenarios.

    SCOPE: randomises growth rate ONLY (the biggest driver of variance in
    this model) — margin, discount rate, and terminal growth are held
    fixed. This is a wider, more honest picture of valuation uncertainty
    than a single number, but is NOT a fully calibrated multi-factor
    probabilistic model. See dcf_monte_carlo()'s docstring in
    financial_ratio_calculator.py for complete scope disclosure.

    Args:
        fcf_base, beta, shares_outstanding, terminal_growth_rate_pct,
        risk_free_rate_pct, market_risk_premium_pct, projection_years:
            Same as tool_calc_dcf.
        base_growth_rate_pct: Mean of the growth-rate distribution —
            typically the company's own trailing revenue/FCF CAGR.
        growth_std_pct: Std dev of the growth-rate distribution in
            percentage points. Defaults to max(base*0.4, 2.0) if omitted.
        n_simulations: Number of Monte Carlo draws (default 2000).

    Returns:
        dict with enterprise_value_p10/p50/p90,
        intrinsic_value_per_share_p10/p50/p90 (if shares_outstanding
        given), growth_rate_mean_pct/std_pct, discount_rate_pct, note,
        and error.
    """
    log.info(
        "tool_calc_dcf_monte_carlo called: fcf_base=%s beta=%s base_growth=%s n_sim=%s",
        fcf_base, beta, base_growth_rate_pct, n_simulations,
    )
    return dcf_monte_carlo(
        fcf_base=fcf_base,
        beta=beta,
        base_growth_rate_pct=base_growth_rate_pct,
        shares_outstanding=shares_outstanding,
        growth_std_pct=growth_std_pct,
        terminal_growth_rate_pct=terminal_growth_rate_pct,
        risk_free_rate_pct=risk_free_rate_pct,
        market_risk_premium_pct=market_risk_premium_pct,
        projection_years=projection_years,
        n_simulations=n_simulations,
    )


@mcp.tool()
def tool_calc_interest_coverage(ebit: float, interest_expense: float) -> dict:
    """
    Calculate the Interest Coverage Ratio (times-interest-earned).

    Args:
        ebit:             Earnings Before Interest & Taxes (USD).
        interest_expense: Total interest expense (USD).

    Returns:
        dict with keys:
          - interest_coverage (float | None)
          - interpretation    (str) 'strong' | 'adequate' | 'at_risk'
          - formula           (str)
    """
    log.info("tool_calc_interest_coverage called: ebit=%s interest=%s", ebit, interest_expense)
    return interest_coverage(ebit, interest_expense)


@mcp.tool()
def tool_calc_asset_turnover(revenue: float, avg_total_assets: float) -> dict:
    """
    Calculate the Asset Turnover Ratio (revenue generation efficiency).

    Args:
        revenue:          Total annual revenue (USD).
        avg_total_assets: Average total assets — (start + end) / 2 (USD).

    Returns:
        dict with keys:
          - asset_turnover (float | None)
          - interpretation (str) 'efficient' | 'moderate' | 'low_efficiency'
          - formula        (str)
    """
    log.info("tool_calc_asset_turnover called: revenue=%s avg_assets=%s", revenue, avg_total_assets)
    return asset_turnover(revenue, avg_total_assets)


@mcp.tool()
def tool_calc_cagr(start_value: float, end_value: float, years: float) -> dict:
    """
    Calculate the Compound Annual Growth Rate (CAGR) for any metric.

    Args:
        start_value: Value at the beginning of the period (must be > 0).
        end_value:   Value at the end of the period.
        years:       Number of years in the period (must be > 0).

    Returns:
        dict with keys:
          - cagr_pct       (float | None) CAGR as a percentage.
          - interpretation (str) 'hypergrowth' | 'strong' | 'moderate' | 'slow'
          - formula        (str)
    """
    log.info("tool_calc_cagr called: start=%s end=%s years=%s", start_value, end_value, years)
    return cagr(start_value, end_value, years)


@mcp.tool()
def tool_calc_revenue_cagr_from_growth(annual_revenue: list[dict]) -> dict:
    """
    Calculate revenue CAGR directly from the annual_revenue list returned
    by tool_get_revenue_growth — no need to manually pick start/end values
    or count years.

    Args:
        annual_revenue: The "annual_revenue" list from tool_get_revenue_growth,
            e.g. [{"year": 2025, "revenue": 391000000000, "yoy_growth": 0.02}, ...].
            Order does not matter; sorted internally by year.

    Returns:
        dict with keys:
          - cagr_pct       (float | None) CAGR as a percentage.
          - interpretation (str) 'hypergrowth' | 'strong' | 'moderate' | 'slow' | 'unavailable'
          - formula        (str)
          - start_year     (int | None)
          - end_year       (int | None)
          - years_used     (int | None)
          - error          (str | None) Set when fewer than 2 valid years were supplied.
    """
    log.info("tool_calc_revenue_cagr_from_growth called: %d entries", len(annual_revenue or []))
    return compute_revenue_cagr_from_growth(annual_revenue)


@mcp.tool()
def tool_calc_composite_score(
    pe:                float | None = None,
    pb:                float | None = None,
    roe_pct:           float | None = None,
    net_margin_pct:    float | None = None,
    current_ratio_val: float | None = None,
    de_ratio:          float | None = None,
    revenue_cagr_pct:  float | None = None,
) -> dict:
    """
    Compute a weighted composite financial health score (0-100) and letter grade.

    All parameters are optional. The score is calculated proportionally
    from whichever metrics are supplied. Missing metrics are listed in
    'missing_inputs' but do not invalidate the score.

    Weighting scheme:
        ROE             25%
        Net Margin      20%
        Revenue CAGR    20%
        P/E ratio       15%  (lower is better)
        Current Ratio   10%
        D/E ratio       10%  (lower is better)

    Args:
        pe:                Trailing P/E ratio.
        pb:                Price-to-Book ratio (noted but not weighted).
        roe_pct:           Return on Equity (%).
        net_margin_pct:    Net Profit Margin (%).
        current_ratio_val: Current ratio.
        de_ratio:          Debt-to-Equity ratio.
        revenue_cagr_pct:  Revenue CAGR (%).

    Returns:
        dict with keys:
          - score          (float | None) Composite score 0-100.
          - grade          (str)          Letter grade: A | B | C | D | F.
          - sub_scores     (dict)         Normalised 0-10 score per metric.
          - missing_inputs (list[str])    Metrics absent from the calculation.
    """
    log.info("tool_calc_composite_score called")
    return composite_financial_score(
        pe=pe,
        pb=pb,
        roe_pct=roe_pct,
        net_margin_pct=net_margin_pct,
        current_ratio=current_ratio_val,
        de_ratio=de_ratio,
        revenue_cagr_pct=revenue_cagr_pct,
    )


# ---------------------------------------------------------------------------
# Phase 1 — Cash Flow Ratios & Reverse DCF
# (data already available from tool_get_financial_ratios / tool_calc_dcf)
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_calc_net_debt_to_ebitda(total_debt: float, total_cash: float, ebitda: float) -> dict:
    """
    Calculate Net Debt / EBITDA — the standard credit-analysis leverage
    multiple (years of current EBITDA needed to pay off debt net of cash).

    Args:
        total_debt: Total short+long-term debt (USD). From
                    tool_get_financial_ratios' total_debt.
        total_cash: Cash and cash equivalents (USD). From
                    tool_get_financial_ratios' total_cash.
        ebitda:     EBITDA, TTM (USD). From tool_get_financial_ratios' ebitda.

    Returns:
        dict with keys:
          - net_debt, net_debt_to_ebitda (float | None)
          - interpretation (str) 'net_cash_position' | 'low_leverage' |
            'moderate_leverage' | 'high_leverage'
          - formula (str)
    """
    log.info("tool_calc_net_debt_to_ebitda called: debt=%s cash=%s ebitda=%s", total_debt, total_cash, ebitda)
    return net_debt_to_ebitda(total_debt, total_cash, ebitda)


@mcp.tool()
def tool_calc_fcf_margin(free_cash_flow: float, revenue: float) -> dict:
    """
    Calculate Free Cash Flow Margin: what fraction of revenue converts to
    actual free cash after capex. Catches capex-heavy periods (e.g. AI
    infrastructure buildouts) where net-income margins look fine but cash
    generation is being consumed by investment.

    Args:
        free_cash_flow: Operating cash flow − capex, TTM (USD). From
                        tool_get_financial_ratios' free_cash_flow.
        revenue:        Total revenue, TTM (USD).

    Returns:
        dict with keys:
          - fcf_margin_pct (float | None)
          - interpretation (str) 'excellent' | 'strong' | 'modest' | 'cash_burning'
          - formula (str)
    """
    log.info("tool_calc_fcf_margin called: fcf=%s revenue=%s", free_cash_flow, revenue)
    return fcf_margin(free_cash_flow, revenue)


@mcp.tool()
def tool_calc_fcf_yield(free_cash_flow: float, market_cap: float) -> dict:
    """
    Calculate Free Cash Flow Yield: FCF as a percentage of market cap —
    a cash-based, less-distorted-by-accounting counterpart to earnings
    yield / P·E.

    Args:
        free_cash_flow: Free cash flow, TTM (USD). From
                        tool_get_financial_ratios' free_cash_flow.
        market_cap:     Market capitalization (USD). From
                        tool_get_financial_ratios' market_cap.

    Returns:
        dict with keys:
          - fcf_yield_pct (float | None)
          - interpretation (str) 'cheap' | 'fair' | 'expensive' | 'cash_burning'
          - formula (str)
    """
    log.info("tool_calc_fcf_yield called: fcf=%s market_cap=%s", free_cash_flow, market_cap)
    return fcf_yield(free_cash_flow, market_cap)


@mcp.tool()
def tool_calc_ocf_ratio(operating_cash_flow: float, current_liabilities: float) -> dict:
    """
    Calculate the Operating Cash Flow Ratio: can near-term obligations be
    covered by cash actually generated from operations (a stricter test
    than the accrual-based current ratio)?

    Args:
        operating_cash_flow: Cash from operations, TTM (USD). From
                              tool_get_financial_ratios' operating_cash_flow.
        current_liabilities: Total current liabilities (USD). From
                              tool_get_balance_sheet_snapshot's current_liabilities.

    Returns:
        dict with keys:
          - ocf_ratio (float | None)
          - interpretation (str) 'strong' | 'adequate' | 'weak'
          - formula (str)
    """
    log.info("tool_calc_ocf_ratio called: ocf=%s current_liab=%s", operating_cash_flow, current_liabilities)
    return operating_cash_flow_ratio(operating_cash_flow, current_liabilities)


@mcp.tool()
def tool_calc_reverse_dcf(
    fcf_base: float,
    beta: float,
    target_enterprise_value: float,
    terminal_growth_rate_pct: float = 2.5,
    risk_free_rate_pct: float = 4.3,
    market_risk_premium_pct: float = 5.0,
    projection_years: int = 5,
    growth_rate_search_bounds: tuple[float, float] = (-20.0, 60.0),
    market_cap: float | None = None,
    total_debt: float | None = None,
    pre_tax_cost_of_debt_pct: float | None = None,
    effective_tax_rate_pct: float | None = None,
) -> dict:
    """
    Solve DCF in reverse: given the enterprise value the market is
    currently implying, find the constant FCF growth rate that would
    justify it. Answers "what growth rate is priced in right now?"
    instead of "what's fair value at an assumed growth rate?".

    IMPORTANT: target_enterprise_value must be an ENTERPRISE value, not
    raw market cap — a reasonable proxy is market_cap + total_debt −
    total_cash (both from tool_get_financial_ratios). Passing raw market
    cap biases the result by however large net debt is.

    Args:
        fcf_base:                  Most recent annual Free Cash Flow (USD).
        beta:                      Equity beta (from tool_get_financial_ratios).
        target_enterprise_value:   The enterprise value to solve for — see
                                    IMPORTANT note above for how to build it.
        terminal_growth_rate_pct:  Perpetual growth after projection window.
        risk_free_rate_pct:        CAPM risk-free rate.
        market_risk_premium_pct:   CAPM equity risk premium.
        projection_years:          Number of explicit projection years.
        growth_rate_search_bounds: (min, max) growth % to search between.
                                    Narrow this if you have a strong prior
                                    to converge faster / avoid boundary
                                    solutions.

    Returns:
        dict with keys:
          - implied_growth_rate_pct (float | None) — the answer.
          - resulting_enterprise_value (float | None) — should ≈ target.
          - target_enterprise_value (float), converged (bool),
            iterations_used (int), discount_rate_pct (float | None),
            assumptions (dict), interpretation (str), error (str | None)
    """
    log.info(
        "tool_calc_reverse_dcf called: fcf_base=%s beta=%s target_ev=%s",
        fcf_base, beta, target_enterprise_value,
    )
    return reverse_dcf(
        fcf_base=fcf_base,
        beta=beta,
        target_enterprise_value=target_enterprise_value,
        terminal_growth_rate_pct=terminal_growth_rate_pct,
        risk_free_rate_pct=risk_free_rate_pct,
        market_risk_premium_pct=market_risk_premium_pct,
        projection_years=projection_years,
        growth_rate_search_bounds=growth_rate_search_bounds,
        market_cap=market_cap,
        total_debt=total_debt,
        pre_tax_cost_of_debt_pct=pre_tax_cost_of_debt_pct,
        effective_tax_rate_pct=effective_tax_rate_pct,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Advanced Profitability, Leverage & Bankruptcy Risk
# (needs tool_get_balance_sheet_snapshot's output)
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_calc_roic(
    ebit: float,
    tax_rate_pct: float,
    total_debt: float,
    total_equity: float,
    cash: float,
) -> dict:
    """
    Calculate Return on Invested Capital (ROIC): NOPAT / Invested Capital.
    Considered a better cross-company comparison than ROE, since ROE can
    be inflated by leverage alone; ROIC measures returns on ALL capital
    (debt + equity) regardless of financing mix.

    Args:
        ebit:         Earnings Before Interest and Taxes (USD).
        tax_rate_pct: Effective tax rate as a percentage (e.g. 21.0).
                      Use provision_for_income_taxes / pretax_income where
                      available, not just the statutory rate.
        total_debt:   Total debt (USD). From tool_get_financial_ratios.
        total_equity: Total shareholders' equity (USD). From
                      tool_get_balance_sheet_snapshot.
        cash:         Cash and cash equivalents (USD). From
                      tool_get_financial_ratios' total_cash.

    Returns:
        dict with keys:
          - roic_pct, nopat, invested_capital (float | None)
          - interpretation (str) 'excellent' | 'strong' | 'moderate' | 'weak'
          - formula (str)
    """
    log.info("tool_calc_roic called: ebit=%s tax_rate=%s", ebit, tax_rate_pct)
    return return_on_invested_capital(ebit, tax_rate_pct, total_debt, total_equity, cash)


@mcp.tool()
def tool_calc_roce(ebit: float, total_assets: float, current_liabilities: float) -> dict:
    """
    Calculate Return on Capital Employed (ROCE): EBIT / (Total Assets −
    Current Liabilities). A pre-tax, financing-agnostic profitability
    measure; a close variant is the "quality" leg of Greenblatt's Magic
    Formula.

    Args:
        ebit:                 Earnings Before Interest and Taxes (USD).
        total_assets:         Total assets (USD). From
                               tool_get_balance_sheet_snapshot.
        current_liabilities:  Total current liabilities (USD). From
                               tool_get_balance_sheet_snapshot.

    Returns:
        dict with keys:
          - roce_pct, capital_employed (float | None)
          - interpretation (str) 'excellent' | 'strong' | 'moderate' | 'weak'
          - formula (str)
    """
    log.info("tool_calc_roce called: ebit=%s total_assets=%s", ebit, total_assets)
    return return_on_capital_employed(ebit, total_assets, current_liabilities)


@mcp.tool()
def tool_calc_debt_ratio(total_liabilities: float, total_assets: float) -> dict:
    """
    Calculate the Debt Ratio: fraction of total assets financed by
    liabilities of any kind (not just interest-bearing debt).

    Args:
        total_liabilities: Total liabilities (USD). From
                            tool_get_balance_sheet_snapshot.
        total_assets:       Total assets (USD). From
                            tool_get_balance_sheet_snapshot.

    Returns:
        dict with keys:
          - debt_ratio (float | None)
          - interpretation (str) 'low_leverage' | 'moderate_leverage' | 'high_leverage'
          - formula (str)
    """
    log.info("tool_calc_debt_ratio called: total_liab=%s total_assets=%s", total_liabilities, total_assets)
    return debt_ratio(total_liabilities, total_assets)


@mcp.tool()
def tool_calc_equity_ratio(total_equity: float, total_assets: float) -> dict:
    """
    Calculate the Equity Ratio: fraction of total assets financed by
    shareholders' equity. Complement of the Debt Ratio.

    Args:
        total_equity: Total shareholders' equity (USD). From
                      tool_get_balance_sheet_snapshot.
        total_assets: Total assets (USD). From
                      tool_get_balance_sheet_snapshot.

    Returns:
        dict with keys:
          - equity_ratio (float | None)
          - interpretation (str) 'well_capitalized' | 'moderate' | 'thin_equity_base'
          - formula (str)
    """
    log.info("tool_calc_equity_ratio called: total_equity=%s total_assets=%s", total_equity, total_assets)
    return equity_ratio(total_equity, total_assets)


@mcp.tool()
def tool_calc_altman_z_score(
    working_capital: float,
    retained_earnings: float,
    ebit: float,
    total_assets: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> dict:
    """
    Calculate the Altman Z-Score (1968 original) — a composite
    bankruptcy-risk indicator.

    IMPORTANT — model scope: built and validated on public manufacturing/
    industrial firms. Less reliable for financial firms (banks/insurers)
    and asset-light tech/software companies (small total_assets can
    produce extreme, less meaningful scores). Present the caveat
    alongside the number for non-manufacturing tickers.

    Args:
        working_capital:    Current Assets − Current Liabilities (USD).
                             From tool_get_balance_sheet_snapshot.
        retained_earnings:  Retained earnings (USD). From
                             tool_get_balance_sheet_snapshot.
        ebit:                Earnings Before Interest and Taxes (USD).
        total_assets:        Total assets (USD). From
                             tool_get_balance_sheet_snapshot.
        market_cap:          Market value of equity (USD). From
                             tool_get_financial_ratios.
        total_liabilities:   Total liabilities (USD). From
                             tool_get_balance_sheet_snapshot.
        revenue:             Total revenue / Sales (USD).

    Returns:
        dict with keys:
          - z_score (float | None)
          - zone (str) 'safe' (>2.99) | 'grey' (1.81–2.99) | 'distress' (<1.81)
          - interpretation (str), formula (str), caveat (str)
    """
    log.info("tool_calc_altman_z_score called: total_assets=%s", total_assets)
    return altman_z_score(
        working_capital, retained_earnings, ebit,
        total_assets, market_cap, total_liabilities, revenue,
    )


# ===========================================================================
# Entry point — stdio transport only
# ===========================================================================

if __name__ == "__main__":
    # mcp.run(transport="stdio") starts the MCP JSON-RPC loop over
    # stdin/stdout. All logging is directed to stderr so it does NOT
    # interfere with the MCP protocol byte stream on stdout.
    init_sentry()
    log.info("Financial Analyst MCP server starting (stdio transport)")
    mcp.run(transport="stdio")
