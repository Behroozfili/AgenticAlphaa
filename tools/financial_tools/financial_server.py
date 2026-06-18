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

Tool catalogue (16 tools)
--------------------------
Yahoo Finance  : get_price_history, get_financial_ratios,
                 get_revenue_growth, get_peer_comparison
SEC EDGAR      : get_cik, list_filings, get_filing_text, get_xbrl_financials
Ratio Calc     : calc_pe, calc_pb, calc_ev_ebitda, calc_peg,
                 calc_gross_margin, calc_operating_margin, calc_net_margin,
                 calc_roe, calc_roa, calc_current_ratio, calc_quick_ratio,
                 calc_debt_to_equity, calc_interest_coverage,
                 calc_asset_turnover, calc_cagr, calc_composite_score

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
    composite_financial_score,
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
    version="1.0.0",
    description=(
        "Financial Analyst Agent tools: real-time market data via Yahoo Finance, "
        "SEC EDGAR filing access, and a comprehensive financial ratio calculator."
    ),
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
          - ticker              (str)
          - annual_revenue      (list[dict]) [{year, revenue, yoy_growth}]
          - quarterly_revenue   (list[dict]) [{quarter, revenue, yoy_growth}]
          - annual_net_income   (list[dict]) [{year, net_income, yoy_growth}]
          - revenue_growth_ttm  (float | None) Trailing 12-month YoY growth.
          - error               (str | None)

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