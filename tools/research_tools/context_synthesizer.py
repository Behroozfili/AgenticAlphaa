"""
tools/research_tools/context_synthesizer.py

"""
import logging

from langsmith import traceable
import asyncio

logger = logging.getLogger(__name__)

_SOURCE_PRIORITY = {
    "sec_edgar_filing":  1,
    "sec_edgar_search":  1,
    "company_filing":    1,
    "regulatory":        2,
    "rag_vector_search": 3,
    "rag_graph_traverse": 3,
    "rag_hybrid_query":  3,
    "news_search":       4,
    "tavily_search":     5,
}
_DEFAULT_PRIORITY = 3  # untagged chunks: treat as unknown-but-not-lowest

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a Senior Financial Intelligence Analyst. Summarize the raw "
    "research chunks below (SEC filings, news, search results) into a "
    "dense, factual executive summary for a downstream Manager agent.\n\n"
    "Source priority -- SEMANTIC, not absolute. Chunks are labeled with "
    "their source and grouped with higher-priority sources first:\n"
    "  1. SEC filings (10-K, 10-Q, 8-K) / official company filings\n"
    "  2. Regulatory sources\n"
    "  3. RAG / knowledge-base retrieval\n"
    "  4. News\n"
    "  5. General web search (Tavily)\n"
    "This ordering reflects evidentiary weight for FINANCIAL FUNDAMENTALS "
    "(revenue, margins, debt, legally-disclosed risk factors, litigation) "
    "-- SEC filings are audited/legally mandated, so they lead there. It "
    "does NOT mean SEC always overrides other sources: filings lag real "
    "events by weeks, so for recent events, announcements, leadership "
    "changes, or new deals/contracts, NEWS is the authoritative source "
    "even though it's ranked lower above -- a filing's silence on "
    "something recent means \"not yet filed,\" not \"didn't happen.\"\n\n"
    "Rules:\n"
    "1. Preserve every specific number, date, and named entity you see — "
    "do not round, approximate, or drop a figure to save space. If you "
    "are not sure a number carries over exactly, quote it verbatim rather "
    "than paraphrasing it.\n"
    "2. Group by theme (financial results, risk factors, competitive "
    "position, sentiment/market reaction) rather than by source.\n"
    "3. When sources conflict or give different figures for the same "
    "thing, do not silently pick one — state both and label what kind of "
    "claim each is (e.g. \"SEC-reported revenue: $220B\" vs \"Analyst "
    "(news) estimate: $230B\"). A reported/audited figure and a forecast/"
    "expectation are different kinds of claims, not competing answers to "
    "the same question — never merge them into one number.\n"
    "4. Do not add analysis, opinions, or conclusions not present in the "
    "source chunks — this is a compression of what's there, not a new "
    "interpretation.\n"
    "5. No conversational preamble, no meta-commentary about the task "
    "itself — output only the summary content."
)


def _normalize_chunk(chunk) -> tuple[str | None, str]:
    if isinstance(chunk, dict):
        return chunk.get("source"), str(chunk.get("text", ""))
    return None, str(chunk)


def _order_and_label_chunks(chunks: list) -> str:
    normalized = [_normalize_chunk(c) for c in chunks]
    ordered = sorted(
        enumerate(normalized),
        key=lambda item: (_SOURCE_PRIORITY.get(item[1][0], _DEFAULT_PRIORITY), item[0]),
    )
    parts = []
    for _, (source, text) in ordered:
        label = f"[SOURCE: {source}]\n" if source else ""
        parts.append(f"{label}{text}")
    return "\n\n---\n\n".join(parts)


@traceable(name="research.context_synthesizer", run_type="llm")
async def synthesize_research_context(
    chunks: list,
    task_query: str,
    llm_client,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 2048,
) -> str | None:
   
    if not chunks:
        return None

    combined_raw_text = _order_and_label_chunks(chunks)
    user_content = (
        f"ORIGINAL RESEARCH QUERY:\n{task_query}\n\n"
        f"RAW RETRIEVED CHUNKS (ordered by source priority, see system "
        f"prompt):\n{combined_raw_text}\n\n"
        "Generate the synthesized executive research summary now."
    )

    try:
        
        response = await asyncio.to_thread(
            llm_client.messages.create,
            model=model,
            max_tokens=max_tokens,
            temperature=0,  # deterministic — see module docstring point 2
            system=_SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        summary_text = response.content[0].text.strip()
        return f"[SYNTHESIZED RESEARCH SUMMARY — see raw chunks above for exact source figures]\n{summary_text}"
    except Exception as exc:
        logger.warning(
            "synthesize_research_context: synthesis failed (%s) — skipping, "
            "raw chunks remain available so no data is lost.", exc,
        )
        return None