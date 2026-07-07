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
"""
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a Senior Financial Intelligence Analyst. Summarize the raw "
    "research chunks below (SEC filings, news, search results) into a "
    "dense, factual executive summary for a downstream Manager agent.\n\n"
    "Rules:\n"
    "1. Preserve every specific number, date, and named entity you see — "
    "do not round, approximate, or drop a figure to save space. If you "
    "are not sure a number carries over exactly, quote it verbatim rather "
    "than paraphrasing it.\n"
    "2. Group by theme (financial results, risk factors, competitive "
    "position, sentiment/market reaction) rather than by source.\n"
    "3. Do not add analysis, opinions, or conclusions not present in the "
    "source chunks — this is a compression of what's there, not a new "
    "interpretation.\n"
    "4. No conversational preamble, no meta-commentary about the task "
    "itself — output only the summary content."
)


@traceable(name="research.context_synthesizer", run_type="llm")
async def synthesize_research_context(
    chunks: list[str],
    task_query: str,
    llm_client,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 2048,
) -> str | None:
    """
    Synthesize raw research chunks into one dense executive summary.

    Parameters
    ----------
    chunks : list[str]
        The raw context chunks gathered by ResearchAgent's own loop
        (untouched — this function only READS them, the caller is
        responsible for keeping the originals in aggregated_research_context).
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

    combined_raw_text = "\n\n---\n\n".join(chunks)
    user_content = (
        f"ORIGINAL RESEARCH QUERY:\n{task_query}\n\n"
        f"RAW RETRIEVED CHUNKS:\n{combined_raw_text}\n\n"
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