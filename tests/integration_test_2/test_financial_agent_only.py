"""
test_financial_agent_only.py

Direct, isolated run of ONLY the FinancialAnalystAgent -- no ManagerAgent,
no ResearchAgent, no SentimentAgent. None of them are "woken up".

This script calls the FinancialAnalystAgent class directly with a minimal
SharedManagerState (task_query + manager_directives["ticker"]) -- exactly
what FinancialAnalystAgent.run() needs to operate.

Where to run it:
    Place this file next to your project's agents/, tools/, core/ folders
    (the same root that financial_agent.py imports from via
    `from agents.state import ...`). If your project structure differs,
    adjust the sys.path below manually.

Prerequisites (env vars):
    ANTHROPIC_API_KEY   -- for the Checker LLM inside FinancialAnalystAgent
    (yfinance and SEC EDGAR need no key -- both are public/keyless)

Run:
    python3 test_financial_agent_only.py MSFT
    or without an argument for the default ticker below (CONFIG).

Output: financial_agent_test_report_<ticker>_<timestamp>.md in this folder.
"""
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env BEFORE anything checks os.environ -- see the identical note in
# test_research_agent_only.py for why this must happen here, not rely on
# financial_agent.py's own load_dotenv() (which only runs on import, after
# _check_env() would already have run).
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_TICKER = "MSFT"
MAX_LOOPS      = 3


def _check_env() -> list[str]:
    missing = []
    for key in ("ANTHROPIC_API_KEY",):
        if not os.environ.get(key):
            missing.append(key)
    return missing


def _md_escape_fence(text: str) -> str:
    return text.replace("```", "``\u200b`")


def _fmt(val) -> str:
    """Render a value for the markdown table: dicts/lists as compact JSON,
    everything else as-is."""
    if isinstance(val, (dict, list)):
        return f"`{json.dumps(val, ensure_ascii=False)}`"
    return str(val)


async def main():
    print("[1/5] Checking env vars ...", flush=True)
    missing = _check_env()
    if missing:
        print(f"These env vars are not set: {', '.join(missing)}", flush=True)
        sys.exit(1)
    print("    OK", flush=True)

    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("[2/5] Importing FinancialAnalystAgent/SharedManagerState ...", flush=True)
    try:
        from agents.financial_agent import FinancialAnalystAgent  # noqa: E402
        from agents.state import SharedManagerState                # noqa: E402
    except Exception:
        print("    IMPORT FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK", flush=True)

    print(f"[3/5] ticker = {ticker!r}", flush=True)

    print("[4/5] Creating FinancialAnalystAgent instance ...", flush=True)
    try:
        agent = FinancialAnalystAgent(max_loops=MAX_LOOPS)
    except Exception:
        print("    INIT FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK", flush=True)

    # ─────────────────────────────────────────────
    # Structural graph check (no network, no API) --
    # Confirm the graph conversion compiled correctly and the node/edges
    # are wired as expected, BEFORE the real run (which needs network/key)
    # begins.
    # ─────────────────────────────────────────────
    print("[4.5/5] Checking graph structure ...", flush=True)
    try:
        graph = getattr(agent, "_graph", None)
        if graph is None:
            print("    WARNING: agent._graph is missing -- is this the graph version?", flush=True)
        else:
            node_names = set(getattr(graph, "nodes", {}).keys())
            expected = {"brain", "extract", "compute", "checker"}
            missing_nodes = expected - node_names
            if missing_nodes:
                print(f"    WARNING: missing nodes: {missing_nodes}", flush=True)
            else:
                visible = sorted(n for n in node_names if n not in ("__start__", "__end__"))
                print(f"    OK -- nodes: {visible}", flush=True)
            # Confirm the session slot exists and is None before the run (approach B)
            if hasattr(agent, "_session"):
                print(f"    OK -- self._session slot present (initial value: {agent._session})", flush=True)
            else:
                print("    WARNING: self._session is missing", flush=True)
    except Exception:
        print("    GRAPH CHECK FAILED:", flush=True)
        traceback.print_exc()
        # Continue -- this is only a diagnostic check, not a fatal error.
    print("    OK", flush=True)

    shared_state: SharedManagerState = {
        "task_query":                  f"Analyze {ticker} financial health and valuation",
        "manager_directives":          {"ticker": ticker},
        "aggregated_research_context": [],
        "financial_metrics_summary":   {},
        "sentiment_analysis_summary":  {},
        "agent_execution_history":     [],
        "orchestrator_logs":           [],
        "final_report":                "",
    }

    print(
        "[5/5] Running agent.run() -- the MCP server (financial_server.py) "
        "starts up, real yfinance/SEC EDGAR calls are made, this may take a "
        "few tens of seconds, it is not hanging, please wait ...",
        flush=True,
    )
    try:
        result = await agent.run(shared_state)
    except Exception:
        print("    agent.run() FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK -- agent.run() returned.", flush=True)

    fm = result.get("financial_metrics_summary", {})
    print(f"Done. {len(fm)} keys in financial_metrics_summary.", flush=True)

    # ─────────────────────────────────────────────
    # Post-run graph-behavior checks --
    # These depend directly on the correctness of the graph conversion:
    #   - loop_iterations_used must be between 1 and max_loops
    #     (0 means the brain node never incremented loop_counter -- a
    #      conversion bug; > max_loops means the _should_continue guardrail
    #      did not fire).
    #   - self._session must be None again after the run (the finally block
    #     in run()).
    # ─────────────────────────────────────────────
    loops_used = fm.get("loop_iterations_used")
    print("-- Graph behavior checks --", flush=True)
    if loops_used is None:
        print("    WARNING: loop_iterations_used is missing.", flush=True)
    elif not (1 <= loops_used <= MAX_LOOPS):
        print(f"    WARNING: loop_iterations_used={loops_used} outside [1, {MAX_LOOPS}] -- "
              "possible bug in loop counting or the guardrail.", flush=True)
    else:
        print(f"    OK -- loop_iterations_used={loops_used} (within valid range).", flush=True)

    sess_after = getattr(agent, "_session", "N/A")
    if sess_after is None:
        print("    OK -- self._session cleared after the run (None).", flush=True)
    else:
        print(f"    WARNING: self._session is not None after the run: {sess_after} -- "
              "the finally block in run() may not have executed.", flush=True)

    now = datetime.now(timezone.utc)
    lines = [
        f"# FinancialAnalystAgent-only Test Report — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"- ticker: `{ticker}`",
        f"- max_loops: `{MAX_LOOPS}`",
        f"- **No ManagerAgent/ResearchAgent/SentimentAgent was run -- only FinancialAnalystAgent.run()**",
        f"- validation_passed: `{fm.get('validation_passed')}`",
        f"- loop_iterations_used: `{fm.get('loop_iterations_used')}`",
        f"- extraction_errors: `{fm.get('extraction_errors')}`",
        "",
        "## Company",
        "",
        f"- ticker: {fm.get('ticker')}",
        f"- company_name: {fm.get('company_name')}",
        f"- sector: {fm.get('sector')} | industry: {fm.get('industry')}",
        f"- current_price: {fm.get('current_price')} | market_cap: {fm.get('market_cap')}",
        f"- beta: {fm.get('beta')} | 52w: {fm.get('52w_low')} – {fm.get('52w_high')}",
        "",
        "## Composite Score",
        "",
        f"- composite_score: {_fmt(fm.get('composite_score'))}",
        "",
        "## Core Ratios (pre-existing)",
        "",
        "| Metric | Value | Period |",
        "|---|---|---|",
    ]

    existing = [
        ("P/E", "pe_ratio", "pe_ratio"), ("ROE %", "roe", "roe_pct"),
        ("Net Margin %", "net_margin", "net_margin_pct"),
        ("D/E", "de_ratio", "de_ratio"), ("Revenue CAGR %", "revenue_cagr", "cagr_pct"),
    ]
    for label, key, subkey in existing:
        entry = fm.get(key, {})
        val = entry.get(subkey) if isinstance(entry, dict) else entry
        period = entry.get("_period") if isinstance(entry, dict) else None
        lines.append(f"| {label} | {val} | {period} |")
    lines.append(f"| Current Ratio | {fm.get('current_ratio')} | — |")
    lines.append(f"| Forward P/E | {fm.get('forward_pe')} | — |")
    lines.append(f"| PEG | {fm.get('peg_ratio')} | — |")
    lines.append("")

    lines.append("## New Ratios (Phase 1 & 2)")
    lines.append("")
    lines.append("| Metric | Value | Period | Note |")
    lines.append("|---|---|---|---|")

    new_ratios = [
        ("Net Debt/EBITDA", "net_debt_to_ebitda", "net_debt_to_ebitda"),
        ("FCF Margin %", "fcf_margin", "fcf_margin_pct"),
        ("FCF Yield %", "fcf_yield", "fcf_yield_pct"),
        ("OCF Ratio", "ocf_ratio", "ocf_ratio"),
        ("ROIC %", "roic", "roic_pct"),
        ("ROCE %", "roce", "roce_pct"),
        ("Debt Ratio", "debt_ratio", "debt_ratio"),
        ("Equity Ratio", "equity_ratio", "equity_ratio"),
    ]
    for label, key, subkey in new_ratios:
        entry = fm.get(key, {})
        val = entry.get(subkey) if isinstance(entry, dict) else entry
        period = entry.get("_period") if isinstance(entry, dict) else None
        note = entry.get("note", "") if isinstance(entry, dict) else ""
        lines.append(f"| {label} | {val} | {period} | {note} |")

    rdcf = fm.get("reverse_dcf", {})
    lines.append(
        f"| Reverse DCF implied growth % | {rdcf.get('implied_growth_rate_pct')} | "
        f"{rdcf.get('_period', rdcf.get('assumptions'))} | converged={rdcf.get('converged')} |"
    )
    az = fm.get("altman_z_score", {})
    lines.append(
        f"| Altman Z-Score | {az.get('z_score')} | {az.get('_period')} | "
        f"zone={az.get('zone')}; {az.get('caveat', '')[:80]} |"
    )
    lines.append("")

    lines.append("## Full raw financial_metrics_summary")
    lines.append("")
    lines.append("```json")
    lines.append(_md_escape_fence(json.dumps(fm, indent=2, ensure_ascii=False)))
    lines.append("```")

    report_md = "\n".join(lines)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    out_path = f"financial_agent_test_report_{ticker}_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print()
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())