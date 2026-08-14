"""
tools/research_tools/context_synthesizer.py

Standalone tool: compresses a list of raw research context chunks into one
dense executive summary, WITHOUT touching ResearchAgent's own Brain/
Checker/Executor loop — this is called once, at the tail of
ResearchAgent.run(), after the loop has already finished gathering data.

Design choices, and why (see also the concerns raised before building this):

1. ADDITIVE, not a replacement. The synthesized summary is APPENDED to
   aggregated_research_context as one more chunk — the raw chunks (with
   exact numbers: JPMorgan's 227% price-target raise, 10-Q dollar figures,
   etc.) are still passed through untouched. This preserves exact
   traceability for numeric faithfulness validation (see
   validate_numeric_faithfulness.py) while still giving the Manager a
   quick-read dense summary to reduce how much raw text it has to parse.
   Replacing the raw chunks outright would let this summarizing LLM's own
   paraphrasing/rounding/omissions become a NEW, harder-to-detect source
   of numeric drift — exactly the class of bug this project spent an
   entire session finding and fixing at the sec_edgar/news_search/
   Checker layer. Don't reintroduce it one level up.

2. temperature=0, not 0.1. This project's core, repeatedly-relearned
   lesson (see the SentimentAgent stability work) is that ANY LLM call in
   this pipeline whose output feeds an analytical report needs
   temperature=0 for reproducibility. A "low but nonzero" temperature
   still introduces run-to-run drift.

3. Never raises on failure. If synthesis fails (bad JSON, API error,
   timeout), returns None so the caller can just skip appending it — the
   raw chunks are ALREADY safely captured regardless, so a synthesis
   failure should never be able to lose data or crash the pipeline.

4. This is a plain importable async function, not something registered
   in ResearchAgent's own Brain tool list — the LLM inside ResearchAgent
   never decides whether to call this; ResearchAgent.run() calls it
   unconditionally once, after its internal loop completes.

5. Source priority is SEMANTIC, not a fixed weight. SEC filings are the
   audited, legally-mandated primary source for financial fundamentals
   (revenue, risk factors, obligations) — but they're not universally
   "more true" than news: an 8-K/10-Q can lag a real event by weeks, so
   for "did X happen recently" questions, news is the right authority.
   Hard-coding "SEC always wins" would make the synthesizer confidently
   silent on recent events just because the filing hasn't caught up yet.
   So: chunks are tagged with a source and grouped by priority (SEC
   first) as a soft ordering signal, but the actual judgment of which
   source answers a given question is left to the system prompt's
   instructions, not to code that reorders/drops chunks.
"""
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

# Soft ordering only (see design note 5 above) -- chunks from
# higher-priority (lower-number) sources are placed earlier in the
# combined text, which is a real but soft attention signal, not a rule
# that any source's claims override another's. Unrecognized/untagged
# sources sit in the middle rather than last, since "we don't know what
# this is" isn't the same as "this is low-value."
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
    """Accept either a plain string (source unknown -- the original,
    still-supported input shape) or a {"source": ..., "text": ...} dict.
    Returns (source_or_None, text)."""
    if isinstance(chunk, dict):
        return chunk.get("source"), str(chunk.get("text", ""))
    return None, str(chunk)


def _order_and_label_chunks(chunks: list) -> str:
    """Sort chunks by source priority (stable -- same-priority chunks keep
    their original relative order) and render each with a [SOURCE: ...]
    label when known, so the model can apply the priority rules in the
    system prompt instead of guessing where each chunk came from."""
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
    """
    Synthesize raw research chunks into one dense executive summary.

    Parameters
    ----------
    chunks : list[str] | list[dict]
        The raw context chunks gathered by ResearchAgent's own loop
        (untouched — this function only READS them, the caller is
        responsible for keeping the originals in aggregated_research_context).
        Each entry can be either a plain string (source unknown -- fully
        backward compatible with the original signature) or a dict
        {"source": "sec_edgar_filing" | "news_search" | "tavily_search" |
        "rag_vector_search" | ..., "text": "..."} to enable source-priority
        ordering and labeling (see module docstring point 5). Tagging is
        optional per-chunk; a mix of tagged and untagged chunks is fine.
    task_query : str
        The original research task, for framing what the summary should
        emphasize.
    llm_client : anthropic.Anthropic (or compatible)
        Caller-supplied client — this tool doesn't construct its own,
        so it's trivially mockable in tests.
    model, max_tokens : passed straight through to the API call.

    Returns
    -------
    str | None
        The synthesized summary text (prefixed with a clear marker so
        downstream code and human readers can tell it's a compressed
        summary, not a raw tool chunk), or None if synthesis failed for
        any reason — callers should treat None as "skip appending, the
        raw chunks are enough on their own", never as an error to raise.
    """
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
        import asyncio
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