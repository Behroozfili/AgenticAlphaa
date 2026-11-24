
import asyncio
import copy
import inspect
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


async def maybe_await(value):
    """Await a provider result only when the injected boundary is asynchronous."""
    return await value if inspect.isawaitable(value) else value


async def run_monthly_specialists_concurrently(
    agents: dict[str, Any], state: SharedManagerState,
) -> dict[str, SharedManagerState | BaseException]:
    """Run specialists after one common context, on isolated state copies."""
    names = ("research", "financial", "sentiment")
    calls = [agents[name].run(copy.deepcopy(state)) for name in names]
    results = await asyncio.gather(*calls, return_exceptions=True)
    return dict(zip(names, results, strict=True))


def controlled_monthly_join(
    base: SharedManagerState,
    results: dict[str, SharedManagerState | BaseException],
) -> tuple[SharedManagerState, list[str]]:
    """Join only each specialist's owned typed slot and compact legacy summary."""
    joined: SharedManagerState = copy.deepcopy(base)
    order = ["context", "research", "financial", "sentiment", "controlled_join"]
    contracts = {
        "research": ("research_features", "aggregated_research_context"),
        "financial": ("financial_features", "financial_metrics_summary"),
        "sentiment": ("sentiment_features", "sentiment_analysis_summary"),
    }
    history = list(joined.get("agent_execution_history", []))
    for name, (slot_name, legacy_name) in contracts.items():
        result = results[name]
        if isinstance(result, BaseException):
            joined[slot_name] = {
                "version": "unavailable", "values": {}, "evidence_ids": [],
                "missing": True, "degraded": True,
                "errors": [f"{name.upper()}_EXECUTION_FAILED"],
            }
            history.append({
                "agent_name": name, "outcome": "degraded",
                "error_code": f"{name.upper()}_EXECUTION_FAILED",
            })
            continue
        if not isinstance(result, dict):
            raise TypeError(f"{name} agent must return SharedManagerState")
        slot = result.get(slot_name)
        if not isinstance(slot, dict):
            slot = {
                "version": "unavailable", "values": {}, "evidence_ids": [],
                "missing": True, "degraded": True,
                "errors": [f"{name.upper()}_FEATURES_MISSING"],
            }
        joined[slot_name] = copy.deepcopy(slot)
        legacy = result.get(legacy_name)
        if isinstance(legacy, (dict, list)):
            joined[legacy_name] = copy.deepcopy(legacy)
        history.append({
            "agent_name": name,
            "outcome": "degraded" if slot.get("degraded") else "success",
            "result_keys": [slot_name],
        })
    joined["agent_execution_history"] = history
    return joined, order
