"""Privacy-safe point-in-time synthesis for Stage 04B research evidence."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Mapping

from langsmith import traceable

from core.forecast_contract import ForecastContext
from core.observability import elapsed_latency_ms
from tools.research_tools import (
    compact_lifecycle, eligible_evidence, research_implementation_identity,
    research_trace_metadata, require_research_context,
)

logger = logging.getLogger(__name__)
_SOURCE_PRIORITY = {
    "sec_edgar_filing": 1, "sec_edgar_search": 1, "company_filing": 1,
    "regulatory": 2, "rag_vector_search": 3, "rag_graph_traverse": 3,
    "rag_hybrid_query": 3, "news_search": 4, "tavily_search": 5,
}
_DEFAULT_PRIORITY = 3
_MAX_SYNTHESIS_INPUT_CHARS = 24_000
_MAX_SYNTHESIS_SUMMARY_CHARS = 6_000
_SYNTHESIS_MARKER = '[SYNTHESIZED RESEARCH SUMMARY]'
_SYNTHESIS_SYSTEM_PROMPT = (
    "Summarize the supplied financial evidence into a dense factual executive summary. "
    "Preserve exact figures and dates, group by theme, identify conflicts, and add no unsupported claims."
)
_SYNTHESIS_SYSTEM_PROMPT += (
    ' Every factual claim must include one or more inline [evidence_id] citations '
    'using only the supplied IDs.'
)


def _normalize_chunk(chunk: Any) -> tuple[str | None, str]:
    if isinstance(chunk, Mapping):
        compact = {key: chunk.get(key) for key in (
            "evidence_id", "source", "provider", "ticker", "timestamp",
            "dataset_version", "citation", "title", "kind",
        )}
        return chunk.get("source"), json.dumps(compact, sort_keys=True, default=str)
    return None, str(chunk)


def _order_and_label_chunks(chunks: list[Any]) -> str:
    normalized = [_normalize_chunk(chunk) for chunk in chunks]
    ordered = sorted(enumerate(normalized), key=lambda item: (
        _SOURCE_PRIORITY.get(item[1][0], _DEFAULT_PRIORITY), item[0]
    ))
    return "\n\n---\n\n".join(
        f"[SOURCE: {source}]\n{text}" if source else text
        for _, (source, text) in ordered
    )


def _synthesis_trace_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        context, run_id = require_research_context(inputs.get("forecast_context"), inputs.get("run_id"))
        metadata = research_trace_metadata(context, run_id)
    except (TypeError, ValueError):
        metadata = research_implementation_identity()
    chunks = inputs.get("chunks")
    return {**metadata, "operation": "research_synthesis", "input_reference_count": len(chunks) if isinstance(chunks, list) else 0}


def _synthesis_trace_outputs(outputs: Any) -> dict[str, Any]:
    payload = outputs if isinstance(outputs, Mapping) else {}
    telemetry = payload.get("telemetry") if isinstance(payload.get("telemetry"), Mapping) else {}
    return {key: telemetry.get(key) for key in (
        "stage", "stage_revision", "attempt", "agent", "implementation_version",
        "run_id", "ticker", "as_of",
        "dataset_version", "analysis_mode", "operation", "provider", "outcome",
        "latency_ms", "input_count", "result_count", "evidence_count", "citation_count",
        "quarantine_count", "rejection_counts", "degraded",
    ) if telemetry.get(key) is not None}


@traceable(
    name="research.context_synthesizer", run_type="chain",
    metadata=research_implementation_identity(),
    process_inputs=_synthesis_trace_inputs, process_outputs=_synthesis_trace_outputs,
)
async def synthesize_research_context(
    chunks: list[Any], task_query: str, llm_client, model: str = "claude-haiku-4-5",
    max_tokens: int = 2048,
    forecast_context: ForecastContext | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    context, run_identity = require_research_context(forecast_context, run_id)
    candidates = [chunk for chunk in chunks if isinstance(chunk, Mapping) and chunk.get("evidence_id")]
    eligible, rejected = eligible_evidence(
        candidates, context=context, run_id=run_identity,
        source="research_synthesis", historical=False,
    )
    payload: dict[str, Any] = {
        "kind": "research_synthesis", "summary_generated": False,
        "evidence_ids": [item["evidence_id"] for item in eligible],
        "quarantined": rejected, "missing": not eligible,
        "degraded": bool(rejected) or not eligible,
        "dataset_version": context.dataset_version, "run_id": run_identity,
        "status": "OK" if eligible else "DATA_UNAVAILABLE",
    }
    if eligible:
        digest_parts = []
        for item in eligible:
            excerpt = str(item.get('content_excerpt') or '').strip()
            if excerpt:
                digest_parts.append(
                    f"[{item['evidence_id']}] {item.get('source')} "
                    f"{item.get('timestamp')}\n{excerpt}"
                )
        if digest_parts:
            evidence_line = 'EVIDENCE_IDS: ' + ', '.join(payload['evidence_ids'])
            prefix = f'{_SYNTHESIS_MARKER}\n{evidence_line}\n'
            digest = '\n\n'.join(digest_parts)
            payload['summary'] = prefix + digest[:max(
                0, _MAX_SYNTHESIS_SUMMARY_CHARS - len(prefix)
            )]
            payload['summary_generated'] = True
    payload["telemetry"] = compact_lifecycle(
        context=context, run_id=run_identity, operation="research_synthesis",
        latency_ms=elapsed_latency_ms(started_at), payload=payload, input_count=len(candidates),
    )
    return payload
