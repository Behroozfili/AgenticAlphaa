
import json
from typing import Any, Union
from agents.state import EvaluationSnapshot
from memory.manager_memory import EvaluationFeedback
from agents.state import SharedManagerState
def format_news_chunk_fairly(chunk_text: str, max_articles: int = 5, per_article_budget: int = 220) -> str:
    """
    Format a news_search chunk so multiple articles survive, instead of
    the flat 200-char truncation cutting off partway through the FIRST
    article's title alone.

    Regression this guards against: news_search's raw JSON puts
    query/effective_query/total_results BEFORE the "articles" list, and
    a flat [:200] truncation on the whole serialized blob rarely reaches
    past the first article's title — meaning a chunk with 8 real,
    specific, often highly-relevant articles (e.g. "JPMorgan reversed
    course on Tesla, price target +227%") contributed effectively ZERO
    information to the finalizer, even though the tool call succeeded and
    fetched all 8. This surfaces title + a short snippet of description
    for up to `max_articles`, so the finalizer actually sees the specific
    events, not just a truncated fragment of the first headline.

    Falls back to a flat truncation if the embedded JSON can't be parsed.
    """
    idx = chunk_text.rfind("─")
    json_start = chunk_text.find("{", idx) if idx != -1 else chunk_text.find("{")
    if json_start == -1:
        return chunk_text[:per_article_budget * 2]

    try:
        payload = json.loads(chunk_text[json_start:])
    except (json.JSONDecodeError, ValueError):
        return chunk_text[:per_article_budget * 2]

    articles = payload.get("articles")
    if not isinstance(articles, list) or not articles:
        return chunk_text[:per_article_budget * 2]

    header = f"[NEWS: query={payload.get('effective_query', payload.get('query', '?'))!r}, {payload.get('total_results', '?')} total results]"
    lines = [header]
    for a in articles[:max_articles]:
        title = a.get("title", "")
        desc = a.get("description", "")
        published = a.get("published_at", "")
        lines.append(f"  • [{published}] {title} — {desc[:per_article_budget]}")
    return "\n".join(lines)


def extract_chunk_text(chunk: Union[str, dict]) -> str:
    """
    Safely extract plain text from a research context chunk.

    DC-3: The isinstance(chunk, str) branch is kept for safety but is
    currently unreachable — all callers in the pipeline produce dict chunks
    with a "text" key. If the pipeline schema changes, this guard prevents
    a silent crash.
    """
    if isinstance(chunk, str):
        # NOTE: unreachable in the current pipeline (all chunks are dicts).
        # Kept as a defensive fallback in case the schema changes upstream.
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("text", str(chunk))
    return str(chunk)


def render_section_entry(entry, per_section_budget: int) -> str:
    """
    Render one section's value from sec_edgar_filing's response into text.

    sec_edgar_filing now returns each section as a dict, not a plain
    string (see tools/research_tools/sec_edgar.py's mode="preview"/"full").
    Shape depends on mode:
      - preview (default): {total_chars, preview, headings,
        digest: [{heading, lead}, ...]} OR key_figures: {label: [nums]}
        (financial_statements only), optionally important_excerpts: [...]
      - full: {total_chars, text, offset, returned_chars}

    Falls back to str(entry) if given something unexpected (e.g. the
    section really is still a plain string somewhere) rather than raising.
    """
    if isinstance(entry, str):
        return entry[:per_section_budget]
    if not isinstance(entry, dict):
        return str(entry)[:per_section_budget]

    lines = []
    total = entry.get("total_chars")
    if total is not None:
        lines.append(f"(total_chars in source document: {total})")

    if "key_figures" in entry and entry["key_figures"]:
        for label, nums in entry["key_figures"].items():
            lines.append(f"  {label}: {', '.join(nums)}")

    if "digest" in entry and entry["digest"]:
        for d in entry["digest"]:
            lines.append(f"  • {d.get('heading', '')}: {d.get('lead', '')}")

    if "important_excerpts" in entry and entry["important_excerpts"]:
        for ex in entry["important_excerpts"]:
            lines.append(f"  ★ {ex}")

    if "text" in entry:  # mode="full"
        lines.append(entry["text"])

    if not lines and "preview" in entry:  # nothing structured came back; fall back to raw preview
        lines.append(entry["preview"])

    rendered = "\n".join(lines) if lines else str(entry)
    return rendered[:per_section_budget]


def format_filing_chunk_fairly(chunk_text: str, per_section_budget: int = 3000) -> str:
    """
    Format a sec_edgar_filing chunk so every requested section gets a FAIR
    character budget, instead of naively truncating the whole serialized
    JSON blob.

    Regression this guards against: sec_edgar_filing's "sections" dict
    preserves document order — for a 10-Q that's typically
    financial_statements, then mda, then risk_factors. financial_statements
    is routinely tens of thousands of characters (full balance sheet,
    income statement, cash flow statement, and notes). When the Brain
    requests sections=["financial_statements", "mda"] together (observed
    routinely in real traces) and the WHOLE chunk is truncated to a single
    flat budget, financial_statements alone consumes the entire budget —
    mda is fully present in the fetched data (confirmed via direct tool
    output inspection) but never reaches the finalizer prompt at all,
    because the flat truncation cuts the string before mda's text even
    starts. Parsing the JSON and budgeting PER SECTION fixes this: each
    section gets its own space regardless of how large its neighbors are.

    NOTE: sec_edgar_filing's per-section VALUE is no longer a plain string
    (see _render_section_entry) -- it's now a dict of
    total_chars/digest/key_figures/important_excerpts (mode="preview") or
    total_chars/text (mode="full"). This function budgets/truncates the
    RENDERED text of each section, not the raw dict.

    Falls back to a flat truncation of the raw text if the embedded JSON
    can't be parsed (e.g. unexpected format) — never raises, never returns
    less information than the old behavior would have.
    """
    # The chunk's raw text looks like:
    #   ──────────────
    #   [TOOL: sec_edgar_filing] | [QUERY: TSLA]
    #   ──────────────
    #   {...json...}
    # Find the JSON payload by locating the last line of dashes and taking
    # everything after it.
    idx = chunk_text.rfind("─")
    json_start = chunk_text.find("{", idx) if idx != -1 else chunk_text.find("{")
    if json_start == -1:
        return chunk_text[:per_section_budget * 2]  # not JSON at all; flat fallback

    try:
        payload = json.loads(chunk_text[json_start:])
    except (json.JSONDecodeError, ValueError):
        return chunk_text[:per_section_budget * 2]  # unparseable; flat fallback

    sections = payload.get("sections")
    if not isinstance(sections, dict) or not sections:
        return chunk_text[:per_section_budget * 2]  # no sections dict; flat fallback

    header = (
        f"[SEC FILING: {payload.get('ticker', '?')} {payload.get('form_type', '?')}, "
        f"filed {payload.get('filed_at', '?')}, mode={payload.get('mode', '?')}]"
    )
    parts = [header]
    for name, entry in sections.items():
        parts.append(f"--- {name} ---\n{render_section_entry(entry, per_section_budget)}")
    return "\n".join(parts)


def feedback_to_snapshot(fb: EvaluationFeedback) -> EvaluationSnapshot:
    """Convert an EvaluationFeedback dataclass to a plain EvaluationSnapshot dict."""
    return EvaluationSnapshot(
        step=        fb.step,
        passed=      fb.passed,
        score=       fb.score,
        next_action= fb.next_action,
        issues=      list(fb.issues),
    )


def infer_result_keys(agent_key: str, state: SharedManagerState) -> list[str]:
    """
    Infer which SharedManagerState keys an agent populated.

    Parameters
    ----------
    agent_key : str
        One of ``"research"`` | ``"financial"`` | ``"sentiment"``.
    state : SharedManagerState
        State after the agent's ``run()`` returned.

    Returns
    -------
    list[str]
        List of state keys that appear non-empty after the agent ran.
    """
    key_map = {
        "research":  "aggregated_research_context",
        "financial": "financial_metrics_summary",
        "sentiment": "sentiment_analysis_summary",
    }
    result_key = key_map.get(agent_key)
    if result_key and state.get(result_key):
        return [result_key]
    return []