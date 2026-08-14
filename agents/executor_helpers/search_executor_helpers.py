from __future__ import annotations
import json
from typing import Any
import logging

logger = logging.getLogger(__name__)

TICKER_ARG_BY_TOOL: dict[str, str] = {
    "rag_vector_search":  "ticker_filter",
    "rag_graph_traverse": "entity",
    "rag_hybrid_query":   "entity",
    "sec_edgar_search":   "ticker",
    "sec_edgar_filing":   "ticker",
}


def ensure_ticker_argument(
    tool_name: str,
    arguments: dict[str, Any],
    directives: dict[str, Any],
) -> dict[str, Any]:
    """
    Defensive safety net, independent of LLM Brain behaviour: if this tool
    is ticker-scoped and the Brain's plan omitted the ticker argument (or
    supplied it under the wrong key), inject it from manager_directives.

    Root cause this guards against: rag_hybrid_query's built-in fallback
    (_extract_ticker_from_query in rag/hybrid_rag.py) only works when the
    literal ticker symbol happens to appear inside the free-text query
    (e.g. "Apple AAPL earnings" -> "AAPL" extracted by luck). Queries
    phrased using the company name instead ("Microsoft AI infrastructure
    spending") have no extractable ticker, silently degrading
    rag_hybrid_query to vector-only mode AND removing the ticker filter
    from the underlying vector search entirely — letting chunks from
    unrelated tickers leak into the results. The system prompt now
    instructs the Brain to always pass the ticker explicitly, but a prompt
    instruction can be missed; this function makes correctness NOT depend
    on the LLM remembering.

    A ticker already present in `arguments` (whatever the LLM supplied) is
    always preserved — this only fills in a MISSING value, never overrides
    an explicit one.
    """
    arg_key = TICKER_ARG_BY_TOOL.get(tool_name)
    if not arg_key:
        return arguments  # not a ticker-scoped tool — nothing to do

    if arguments.get(arg_key):
        return arguments  # Brain already supplied it

    ticker = directives.get("ticker")
    if not ticker:
        return arguments  # nothing to inject (no ticker known for this task)

    patched = dict(arguments)
    patched[arg_key] = ticker
    logger.info(
        "[Executor] Injected missing ticker argument for '%s': %s='%s'",
        tool_name, arg_key, ticker,
    )
    return patched


# ── Duplicate tool-call detection ───────────────────────────────────
#
# Real trace (confirmed via _tag_chunk_with_source-formatted chunks): the
# Brain re-issued sec_edgar_filing(ticker="MSFT", ...) in a later loop
# iteration with the SAME or a SUBSET of sections it had already fetched
# in an earlier iteration -- because nothing told it "you already have
# this." That padded context_chunks with a byte-for-byte or near-duplicate
# repeat of an earlier chunk for zero new information, at the cost of a
# full tool round-trip and several thousand extra characters of context.
#
# Two checks, both keyed off `called_tool_signatures` (see
# ResearchAgentState's docstring):
#   1. Exact match -- same tool, byte-identical arguments. Generic, applies
#      to every tool.
#   2. sec_edgar_filing subset match -- same ticker/form_type/mode (i.e.
#      the same underlying document), and every section in the new call
#      was already covered by a previous call to that same document. This
#      catches the partial-overlap case exact-match alone would miss (e.g.
#      an earlier call fetched ["financial_statements","mda"], a later one
#      asks for just ["financial_statements"]).

def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}|{json.dumps(arguments, sort_keys=True, default=str)}"


def sec_filing_doc_key(arguments: dict[str, Any]) -> str | None:
    """Identifies WHICH document a sec_edgar_filing call targets (ticker +
    form_type + mode), independent of which sections were requested."""
    ticker = arguments.get("ticker")
    if not ticker:
        return None
    form_type = arguments.get("form_type", "10-K")
    mode = arguments.get("mode", "preview")
    return f"{ticker}|{form_type}|{mode}"


def sec_filing_sections(arguments: dict[str, Any]) -> set[str]:
    sections = arguments.get("sections") or []
    if isinstance(sections, str):
        sections = [sections]
    return set(sections)


def build_sec_filing_coverage(signatures: set[str]) -> dict[str, set[str]]:
    """Reconstruct "which sections have already been fetched for which
    document" from the flat list of prior call signatures, so a new
    sec_edgar_filing call can be checked against everything gathered so
    far, not just the immediately preceding call."""
    coverage: dict[str, set[str]] = {}
    for sig in signatures:
        if not sig.startswith("sec_edgar_filing|"):
            continue
        try:
            _, args_json = sig.split("|", 1)
            args = json.loads(args_json)
        except (ValueError, json.JSONDecodeError):
            continue
        doc_key = sec_filing_doc_key(args)
        if doc_key:
            coverage.setdefault(doc_key, set()).update(sec_filing_sections(args))
    return coverage


def format_skip_notice(tool_name: str, query_hint: str, reason_tag: str, message: str) -> str:
    """
    Build a skipped-call notice chunk in the SAME separator/[TOOL:]/
    separator/body shape _format_tool_result produces for a real call --
    not an ad-hoc one-liner. Every downstream consumer of these chunks
    (this test harness's tool-name parser, manager_agent.py's
    filing_chunks/news_chunks detection which look for "[TOOL: x]" in the
    first ~150 chars, etc.) was built against that exact shape; a
    differently-structured skip message would show up mislabeled as
    tool=UNKNOWN in tooling even though the skip itself worked correctly.
    """
    separator = "─" * 60
    return (
        f"{separator}\n"
        f"[TOOL: {tool_name}] | [QUERY: {query_hint}] | [SKIPPED: {reason_tag}]\n"
        f"{separator}\n"
        f"{message}\n"
    )