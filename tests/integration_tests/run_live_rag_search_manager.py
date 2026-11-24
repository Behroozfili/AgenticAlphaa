"""Focused live smoke test for ResearchAgent -> RAG/web -> Manager handoff."""

import asyncio
import json
from collections import Counter

from agents.executor_helpers.manager_executor_helper import controlled_monthly_join
from agents.manager_agent import agent_result_is_valid
from agents.research_agent import ResearchAgent
from core.forecast_contract import ForecastContext


async def main() -> None:
    context = ForecastContext(
        ticker="MSFT",
        as_of="2026-09-30T20:00:00Z",
        schema_version="forecast-schema-v2",
        dataset_version="live-dataset-v1",
        feature_version="monthly-features-v1",
        model_version="monthly-model-v1",
        lookback_days=31,
    )
    run_id = "live-msft-rag-search-manager-20260920"
    query = (
        "Prepare evidence for MSFT's October 2026 monthly stock forecast. "
        "You MUST use rag_hybrid_query to retrieve current-September company "
        "catalysts and risks from the project RAG, and also use tavily_search "
        "to corroborate them with current web news. Return temporally valid "
        "evidence available no later than 2026-09-30."
    )
    base = {
        "task_query": query,
        "manager_directives": {"ticker": "MSFT", "run_id": run_id},
        "forecast_context": context.to_dict(),
        "aggregated_research_context": [],
        "financial_metrics_summary": {},
        "sentiment_analysis_summary": {},
        "agent_execution_history": [],
        "orchestrator_logs": [],
        "final_report": "",
    }

    result = await ResearchAgent(max_loops=4).run(base)
    contexts = result.get("aggregated_research_context", [])
    references = [item for item in contexts if isinstance(item, dict)]
    summaries = [item for item in contexts if isinstance(item, str)]
    sources = Counter(str(item.get("source")) for item in references)
    providers = Counter(str(item.get("provider")) for item in references)
    timestamps = [item.get("timestamp") for item in references if item.get("timestamp")]
    features = result.get("research_features", {})
    values = features.get("values", {}) if isinstance(features, dict) else {}

    unavailable = {
        "version": "test-placeholder",
        "values": {},
        "evidence_ids": [],
        "missing": True,
        "degraded": True,
        "errors": ["NOT_RUN_IN_THIS_FOCUSED_TEST"],
    }
    financial = {
        **base,
        "financial_features": unavailable,
        "financial_metrics_summary": {},
    }
    sentiment = {
        **base,
        "sentiment_features": unavailable,
        "sentiment_analysis_summary": {},
    }
    joined, order = controlled_monthly_join(
        base,
        {"research": result, "financial": financial, "sentiment": sentiment},
    )

    report = {
        "query": query,
        "reference_count": len(references),
        "summary_count": len(summaries),
        "sources": dict(sources),
        "providers": dict(providers),
        "earliest_timestamp": min(timestamps) if timestamps else None,
        "latest_timestamp": max(timestamps) if timestamps else None,
        "research_feature_evidence_count": len(features.get("evidence_ids", [])),
        "research_missing": features.get("missing"),
        "research_degraded": features.get("degraded"),
        "research_errors": features.get("errors"),
        "rejection_counts": values.get("rejection_counts"),
        "lifecycle": values.get("lifecycle"),
        "manager_join_order": order,
        "manager_received_reference_count": len(
            joined.get("aggregated_research_context", [])
        ),
        "manager_received_research_features": isinstance(
            joined.get("research_features"), dict
        ),
        "manager_research_result_valid": agent_result_is_valid("research", joined),
        "manager_history_research": [
            item
            for item in joined.get("agent_execution_history", [])
            if item.get("agent_name") == "research"
        ],
        "has_rag": any(source.startswith("rag_") for source in sources),
        "has_tavily": "tavily_search" in sources,
    }
    print("E2E_RESULT=" + json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
