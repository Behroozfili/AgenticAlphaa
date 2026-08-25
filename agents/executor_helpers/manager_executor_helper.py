
import json
from typing import Any, Union
from agents.state import EvaluationSnapshot
from memory.manager_memory import EvaluationFeedback
from agents.state import SharedManagerState
def format_news_chunk_fairly(chunk_text: str, max_articles: int = 5, per_article_budget: int = 220) -> str:
    
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
   
    if isinstance(chunk, str):
        # NOTE: unreachable in the current pipeline (all chunks are dicts).
        # Kept as a defensive fallback in case the schema changes upstream.
        return chunk
    if isinstance(chunk, dict):
        return chunk.get("text", str(chunk))
    return str(chunk)


def render_section_entry(entry, per_section_budget: int) -> str:
    
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
    
    key_map = {
        "research":  "aggregated_research_context",
        "financial": "financial_metrics_summary",
        "sentiment": "sentiment_analysis_summary",
    }
    result_key = key_map.get(agent_key)
    if result_key and state.get(result_key):
        return [result_key]
    return []