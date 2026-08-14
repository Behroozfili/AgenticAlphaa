"""
test_sentiment_agent_only.py

Direct, isolated run of ONLY the SentimentAgent -- no ManagerAgent, no
ResearchAgent, no FinancialAnalystAgent. None of them are "woken up".

This script calls the SentimentAgent class directly with a minimal
SharedManagerState. Because the Sentiment Agent normally consumes the
Financial and Research agents' outputs, and neither runs here, those two
inputs are passed EMPTY -- the agent must degrade gracefully (it still
computes sentiment from whatever it can retrieve via its own MCP tools).

Where to run it:
    Place this file next to your project's agents/, tools/, core/ folders
    (the same root that sentiment_agent.py imports from via
    `from agents.state import ...`). If your project structure differs,
    adjust the sys.path below manually.

Prerequisites (env vars):
    ANTHROPIC_API_KEY   -- for the Brain LLM inside SentimentAgent

Run:
    python3 test_sentiment_agent_only.py MSFT
    or without an argument for the default ticker below (CONFIG).

Output: sentiment_agent_test_report_<ticker>_<timestamp>.md in this folder.
"""
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env BEFORE anything checks os.environ.
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_TICKER = "MSFT"
MAX_LOOPS      = 2
# Path to the sentiment MCP server, relative to this file. Adjust if yours
# lives elsewhere (e.g. "tools/sentiment_server.py").
SENTIMENT_SERVER_PATH = "tools/sentiment_tools/sentiment_server.py"


def _check_env() -> list[str]:
    missing = []
    for key in ("ANTHROPIC_API_KEY",):
        if not os.environ.get(key):
            missing.append(key)
    return missing


def _md_escape_fence(text: str) -> str:
    return text.replace("```", "``\u200b`")


def _fmt(val) -> str:
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

    print("[2/5] Importing SentimentAgent/SharedManagerState ...", flush=True)
    try:
        from agents.sentiment_agent import SentimentAgent  # noqa: E402
        from agents.state import SharedManagerState         # noqa: E402
    except Exception:
        print("    IMPORT FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK", flush=True)

    print(f"[3/5] ticker = {ticker!r}", flush=True)

    # Resolve the sentiment server path (absolute).
    server_path = SENTIMENT_SERVER_PATH
    if not os.path.isabs(server_path):
        server_path = os.path.join(project_root, server_path)
    if not os.path.exists(server_path):
        print(f"    WARNING: sentiment server not found at {server_path!r} -- "
              "adjust SENTIMENT_SERVER_PATH at the top of this script.", flush=True)

    print("[4/5] Creating SentimentAgent instance ...", flush=True)
    try:
        agent = SentimentAgent(server_script_path=server_path, max_loops=MAX_LOOPS)
    except Exception:
        print("    INIT FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK", flush=True)

    # ─────────────────────────────────────────────
    # Structural graph check (no network, no API) --
    # Confirm the graph conversion compiled correctly and the node/edges
    # are wired as expected, BEFORE the real run begins.
    # ─────────────────────────────────────────────
    print("[4.5/5] Checking graph structure ...", flush=True)
    try:
        graph = getattr(agent, "_graph", None)
        if graph is None:
            print("    WARNING: agent._graph is missing -- is this the graph version?", flush=True)
        else:
            node_names = set(getattr(graph, "nodes", {}).keys())
            expected = {"brain_plan", "execute", "analyze"}
            missing_nodes = expected - node_names
            if missing_nodes:
                print(f"    WARNING: missing nodes: {missing_nodes}", flush=True)
            else:
                visible = sorted(n for n in node_names if n not in ("__start__", "__end__"))
                print(f"    OK -- nodes: {visible}", flush=True)
            if hasattr(agent, "_session"):
                print(f"    OK -- self._session slot present (initial value: {agent._session})", flush=True)
            else:
                print("    WARNING: self._session is missing", flush=True)
    except Exception:
        print("    GRAPH CHECK FAILED:", flush=True)
        traceback.print_exc()
    print("    OK", flush=True)

    # Minimal shared state. financial_metrics_summary and
    # aggregated_research_context are intentionally EMPTY -- neither upstream
    # agent runs in this isolated test.
    shared_state: SharedManagerState = {
        "task_query":                  f"Analyze market sentiment for {ticker}",
        "manager_directives":          {"ticker": ticker},
        "aggregated_research_context": [],
        "financial_metrics_summary":   {},
        "sentiment_analysis_summary":  {},
        "agent_execution_history":     [],
        "orchestrator_logs":           [],
        "final_report":                "",
    }

    print(
        "[5/5] Running agent.run() -- the MCP server (sentiment_server.py) "
        "starts up, real sentiment tools are called, this may take a few tens "
        "of seconds, it is not hanging, please wait ...",
        flush=True,
    )
    try:
        result = await agent.run(shared_state)
    except Exception:
        print("    agent.run() FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK -- agent.run() returned.", flush=True)

    sm = result.get("sentiment_analysis_summary", {})
    print(f"Done. {len(sm)} keys in sentiment_analysis_summary.", flush=True)

    # ─────────────────────────────────────────────
    # Post-run graph-behavior checks --
    #   - loop_iterations_used must be between 1 and max_loops (0 means the
    #     brain_plan node never incremented loop_counter -- a conversion bug).
    #   - self._session must be None again after the run (finally block).
    # ─────────────────────────────────────────────
    loops_used = sm.get("loop_iterations_used")
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
        f"# SentimentAgent-only Test Report — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"- ticker: `{ticker}`",
        f"- max_loops: `{MAX_LOOPS}`",
        f"- **No ManagerAgent/ResearchAgent/FinancialAgent was run -- only SentimentAgent.run()**",
        f"- **financial_metrics_summary and aggregated_research_context were EMPTY inputs**",
        f"- loop_iterations_used: `{sm.get('loop_iterations_used')}`",
        f"- total_chunks_analyzed: `{sm.get('total_chunks_analyzed')}`",
        f"- extraction_errors: `{sm.get('extraction_errors')}`",
        "",
        "## Fear / Greed",
        "",
        f"- ticker: {sm.get('ticker')}",
        f"- fear_greed_score: {sm.get('fear_greed_score')}",
        f"- fear_greed_label: {sm.get('fear_greed_label')}",
        f"- fear_greed_confidence: {sm.get('fear_greed_confidence')}",
        "",
        "## Model Signals",
        "",
        "| Model | Value |",
        "|---|---|",
        f"| FinBERT label | {sm.get('finbert_label')} |",
        f"| FinBERT bullish prob | {sm.get('finbert_bullish_prob')} |",
        f"| FinBERT bearish prob | {sm.get('finbert_bearish_prob')} |",
        f"| FinBERT neutral prob | {sm.get('finbert_neutral_prob')} |",
        f"| VADER compound | {sm.get('vader_compound')} |",
        f"| VADER label | {sm.get('vader_label')} |",
        "",
        "## Brain Interpretation",
        "",
        f"- overall_sentiment: {sm.get('overall_sentiment')}",
        f"- conviction_level: {sm.get('conviction_level')}",
        f"- model_agreement: {sm.get('model_agreement')}",
        f"- key_signals: {_fmt(sm.get('key_signals'))}",
        f"- risk_flags: {_fmt(sm.get('risk_flags'))}",
        "",
        "### Narrative",
        "",
        str(sm.get("narrative", "")),
        "",
        "## Full raw sentiment_analysis_summary",
        "",
        "```json",
        _md_escape_fence(json.dumps(sm, indent=2, ensure_ascii=False)),
        "```",
    ]

    report_md = "\n".join(lines)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    out_path = f"sentiment_agent_test_report_{ticker}_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print()
    print(f"Full report: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())