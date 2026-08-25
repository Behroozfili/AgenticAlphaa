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
    
    arg_key = TICKER_ARG_BY_TOOL.get(tool_name)
    if not arg_key:
        return arguments  

    if arguments.get(arg_key):
        return arguments  

    ticker = directives.get("ticker")
    if not ticker:
        return arguments 

    patched = dict(arguments)
    patched[arg_key] = ticker
    logger.info(
        "[Executor] Injected missing ticker argument for '%s': %s='%s'",
        tool_name, arg_key, ticker,
    )
    return patched




def tool_call_signature(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}|{json.dumps(arguments, sort_keys=True, default=str)}"


def sec_filing_doc_key(arguments: dict[str, Any]) -> str | None:
    
    ticker = arguments.get("ticker")
    if not ticker:
        return None
    form_type = arguments.get("form_type", "10-K")
    mode = arguments.get("mode", "preview")
    accession_number = arguments.get("accession_number")
    if accession_number:
        normalized_accession = str(accession_number).strip().replace("-", "")
        return f"{ticker}|{form_type}|{normalized_accession}|{mode}"
    return f"{ticker}|{form_type}|{mode}"


def sec_filing_sections(arguments: dict[str, Any]) -> set[str]:
    sections = arguments.get("sections") or []
    if isinstance(sections, str):
        sections = [sections]
    return set(sections)


def build_sec_filing_coverage(signatures: set[str]) -> dict[str, set[str]]:
 
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
   
    separator = "─" * 60
    return (
        f"{separator}\n"
        f"[TOOL: {tool_name}] | [QUERY: {query_hint}] | [SKIPPED: {reason_tag}]\n"
        f"{separator}\n"
        f"{message}\n"
    )
