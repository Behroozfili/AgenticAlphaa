"""
test_research_agent_only.py

اجرای مستقیم و ایزوله‌ی فقط ResearchAgent -- بدون ManagerAgent، بدون
FinancialAnalystAgent، بدون SentimentAgent. هیچ‌کدوم "بیدار" نمی‌شن.

این اسکریپت مستقیماً کلاس ResearchAgent را با یک SharedManagerState حداقلی
صدا می‌زند -- دقیقاً همون چیزی که ResearchAgent.run() برای کارکردن لازم
دارد (task_query + manager_directives + aggregated_research_context خالی)،
نه از طریق ManagerAgent.

کجا اجرا کنی:
    این فایل را کنار پوشه‌های agents/, tools/, core/, rag/ پروژه‌ات بذار
    (یعنی همون root ای که research_agent.py از آن‌جا import می‌شه با
    `from agents.state import ...`, `from tools.research_tools... import ...`).
    اگر ساختار پروژه‌ات فرق دارد، مسیر sys.path پایین را دستی تنظیم کن.

پیش‌نیازها (env vars):
    ANTHROPIC_API_KEY   -- برای Brain/Checker (داخل ResearchAgent) و
                           context_synthesizer (Haiku call)
    TAVILY_API_KEY      -- برای tavily_search
    NEWSAPI_KEY         -- برای news_search
    (sec_edgar_search/sec_edgar_filing کلید نمی‌خوان -- SEC EDGAR عمومی است)

اجرا:
    python3 test_research_agent_only.py "Analyze NVIDIA competitive position in AI chips" NVDA
    یا بدون آرگومان برای مقادیر پیش‌فرض پایین (CONFIG).

خروجی: research_agent_test_report_<ticker>_<timestamp>.md در همین پوشه.
"""
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env BEFORE anything checks os.environ. research_agent.py also calls
# load_dotenv() itself, but only as a side effect of importing it -- and
# that import happens further down, AFTER _check_env() would already have
# run. Without this explicit call here, _check_env() sees an empty
# environment and reports keys as "missing" even when they're correctly
# set in .env.
load_dotenv()  # auto-search: CWD and parent directories
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))  # explicit fallback next to this file

# ─────────────────────────────────────────────
# CONFIG -- می‌تونی این‌ها رو هم مستقیم عوض کنی، هم با آرگومان خط فرمان بدی
# ─────────────────────────────────────────────
DEFAULT_TASK_QUERY = "Analyze Microsoft's AI infrastructure spending and competitive risks"
DEFAULT_TICKER      = "MSFT"
MAX_LOOPS            = 3   # همون چیزی که ResearchAgent پیش‌فرض داره


def _check_env() -> list[str]:
    """چک می‌کنه کدوم env varهای لازم غایبن، قبل از این‌که وسط اجرا fail کنیم."""
    missing = []
    for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "NEWSAPI_KEY"):
        if not os.environ.get(key):
            missing.append(key)
    return missing


def _md_escape_fence(text: str) -> str:
    return text.replace("```", "``\u200b`")


def _describe_chunk(chunk, idx: int) -> dict:
    """
    یک chunk خام (رشته‌ی [TOOL: ...] | [QUERY: ...] \\n {json}) را باز می‌کند:
    tool name، آرگومان‌ها، و خودِ payload واقعی (نه یک placeholder از نوع/طول --
    محتوای واقعی، چون کل هدف این اسکریپت خوندن اطلاعات واقعیه، نه شکل داده).
    """
    text = chunk if isinstance(chunk, str) else chunk.get("text", str(chunk))
    is_synthesis = text.startswith("[SYNTHESIZED RESEARCH SUMMARY")

    if is_synthesis:
        return {
            "index": idx, "tool": "SYNTHESIZED_SUMMARY", "query": None,
            "raw_chars": len(text), "summary": text, "payload": None,
        }

    # Parse "[TOOL: name] | [QUERY: q]" header if present
    tool, query = None, None
    if "[TOOL:" in text[:200]:
        try:
            head = text.split("\n", 3)[1]  # line 2 is the [TOOL: ...] | [QUERY: ...] line
            tool = head.split("[TOOL:")[1].split("]")[0].strip()
            if "[QUERY:" in head:
                query = head.split("[QUERY:")[1].split("]")[0].strip()
        except (IndexError, ValueError):
            pass

    idx_brace = text.find("{")
    payload = None
    if idx_brace != -1:
        try:
            payload = json.loads(text[idx_brace:])
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "index": idx, "tool": tool or "UNKNOWN", "query": query,
        "raw_chars": len(text), "summary": None, "payload": payload,
    }


async def main():
    print("[1/5] چک env varها ...", flush=True)
    missing = _check_env()
    if missing:
        print(f"این env varها ست نشدن: {', '.join(missing)}", flush=True)
        print("بدون این‌ها ResearchAgent (یا ابزارهاش) fail می‌کنن. یا export کن، یا .env رو چک کن.", flush=True)
        sys.exit(1)
    print("    OK — همه‌ی env varها ست هستن.", flush=True)

    task_query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK_QUERY
    ticker     = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TICKER

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("[2/5] import کردن ResearchAgent/SharedManagerState ...", flush=True)
    try:
        from agents.research_agent import ResearchAgent  # noqa: E402
        from agents.state import SharedManagerState       # noqa: E402
        from core.forecast_contract import ForecastContext  # noqa: E402
    except Exception:
        print("    IMPORT FAILED -- این یعنی مسیر یا پکیج‌بندی پروژه درست نیست:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK", flush=True)

    print(f"[3/5] task_query = {task_query!r} | ticker = {ticker!r}", flush=True)

    print("[4/5] ساخت ResearchAgent instance (این یعنی MCP server هنوز اجرا نشده، فقط class ساخته می‌شه) ...", flush=True)
    try:
        agent = ResearchAgent(max_loops=MAX_LOOPS)
    except Exception:
        print("    ResearchAgent() INIT FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK", flush=True)

    forecast_context = ForecastContext(
        ticker=ticker, as_of="2026-08-31T20:00:00Z",
        schema_version="v1", dataset_version="integration-live-v1",
        feature_version="research-features-v1", model_version="model-v1",
    )
    shared_state: SharedManagerState = {
        "task_query":                  task_query,
        "manager_directives":          {"ticker": ticker, "run_id": "stage04b-agent-only"},
        "forecast_context":            forecast_context.to_dict(),
        "aggregated_research_context": [],
        "financial_metrics_summary":   {},
        "sentiment_analysis_summary":  {},
        "agent_execution_history":     [],
        "orchestrator_logs":           [],
        "final_report":                "",
    }

    print(
        "[5/5] اجرای agent.run() -- این‌جا یک subprocess برای research_server.py "
        "بالا می‌آید و چند تا API call واقعی (Claude + Tavily/NewsAPI/SEC) می‌زند؛ "
        "ممکنه چند ده ثانیه طول بکشه، هنگ نیست، منتظر بمون ...",
        flush=True,
    )
    try:
        result = await agent.run(shared_state)
    except Exception:
        print("    agent.run() FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print("    OK -- agent.run() برگشت.", flush=True)

    chunks = result.get("aggregated_research_context", [])

    print(f"تمام شد. {len(chunks)} chunk برگشت.")
    for i, c in enumerate(chunks, 1):
        d = _describe_chunk(c, i)
        print(f"  [{i}] tool={d['tool']:<20} query={d['query']!r:<30} chars={d['raw_chars']}")

    # ── گزارش کامل ──────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    lines = [
        f"# ResearchAgent-only Test Report — {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"- task_query: `{task_query}`",
        f"- ticker: `{ticker}`",
        f"- max_loops: `{MAX_LOOPS}`",
        f"- **هیچ ManagerAgent/FinancialAnalystAgent/SentimentAgent اجرا نشد -- فقط ResearchAgent.run()**",
        "",
        f"## خلاصه: {len(chunks)} chunk در aggregated_research_context",
        "",
        "| # | tool | query | chars |",
        "|---|---|---|---|",
    ]
    total_chars = 0
    for i, c in enumerate(chunks, 1):
        d = _describe_chunk(c, i)
        total_chars += d["raw_chars"]
        lines.append(f"| {i} | `{d['tool']}` | {d['query']} | {d['raw_chars']} |")
    lines.append("")
    lines.append(f"**مجموع کاراکتر همه‌ی chunk ها: {total_chars}** (این چیزیه که اگه Manager بیدار بشه، به‌عنوان context ازش تغذیه می‌شه)")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, c in enumerate(chunks, 1):
        d = _describe_chunk(c, i)
        lines.append(f"## Chunk {i}: `{d['tool']}`" + (f" (query={d['query']})" if d["query"] else ""))
        lines.append("")
        if d["summary"] is not None:
            lines.append("این chunk خلاصه‌ی synthesized (LLM-compressed) نهایی است:")
            lines.append("")
            lines.append("```text")
            lines.append(_md_escape_fence(d["summary"]))
            lines.append("```")
        elif d["payload"] is not None:
            lines.append(f"raw_chars: **{d['raw_chars']}**")
            lines.append("")
            if d["tool"] == "sec_edgar_filing" and isinstance(d["payload"].get("sections"), dict):
                # Render readably (digest/key_figures/important_excerpts), not raw JSON.
                p = d["payload"]
                lines.append(f"company={p.get('company')} form_type={p.get('form_type')} "
                              f"filed_at={p.get('filed_at')} mode={p.get('mode')}")
                lines.append("")
                for sec_name, entry in p["sections"].items():
                    lines.append(f"### section: `{sec_name}` (total_chars in source doc: {entry.get('total_chars')})")
                    lines.append("")
                    if entry.get("key_figures"):
                        lines.append("| line item | مقادیر |")
                        lines.append("|---|---|")
                        for label, nums in entry["key_figures"].items():
                            lines.append(f"| {label} | {', '.join(nums)} |")
                        lines.append("")
                    if entry.get("digest"):
                        for dd in entry["digest"]:
                            lines.append(f"- **{dd.get('heading','')}** — {dd.get('lead','')}")
                        lines.append("")
                    if entry.get("important_excerpts"):
                        for ex in entry["important_excerpts"]:
                            lines.append(f"- ★ {ex}")
                        lines.append("")
            else:
                # Full real payload, not a shape placeholder.
                lines.append("```json")
                lines.append(_md_escape_fence(json.dumps(d["payload"], indent=2, ensure_ascii=False)))
                lines.append("```")
        else:
            text = c if isinstance(c, str) else c.get("text", str(c))
            lines.append(f"raw_chars: **{d['raw_chars']}** (payload JSON parse نشد، متن خام کامل:)")
            lines.append("")
            lines.append("```text")
            lines.append(_md_escape_fence(text))
            lines.append("```")
        lines.append("")

    report_md = "\n".join(lines)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    out_path = f"research_agent_test_report_{ticker}_{ts}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print()
    print(f"گزارش کامل: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
