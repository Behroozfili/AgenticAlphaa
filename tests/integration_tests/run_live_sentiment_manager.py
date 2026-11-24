"""Focused live smoke test for monthly SentimentAgent -> Manager handoff."""

import asyncio
import json
from pathlib import Path

from agents.executor_helpers.manager_executor_helper import controlled_monthly_join
from agents.manager_agent import agent_result_is_valid
from agents.sentiment_agent import SentimentAgent
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
    run_id = "live-msft-sentiment-manager-20260920"
    query = (
        "Prepare a monthly stock forecast for MSFT for October 2026 using "
        "only information available by the September 2026 month-end cutoff."
    )
    base = {
        "task_query": query,
        "manager_directives": {
            "ticker": "MSFT",
            "run_id": run_id,
            "max_loops": 3,
        },
        "forecast_context": context.to_dict(),
        "aggregated_research_context": [],
        "financial_metrics_summary": {},
        "sentiment_analysis_summary": {},
        "agent_execution_history": [],
        "orchestrator_logs": [],
        "final_report": "",
    }

    server_path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "sentiment_tools"
        / "sentiment_server.py"
    )
    result = await SentimentAgent(
        server_script_path=str(server_path), max_loops=3
    ).run(base.copy())
    summary = result.get("sentiment_analysis_summary", {})
    features = result.get("sentiment_features", {})
    values = features.get("values", {}) if isinstance(features, dict) else {}

    unavailable = {
        "version": "test-placeholder",
        "values": {},
        "evidence_ids": [],
        "missing": True,
        "degraded": True,
        "errors": ["NOT_RUN_IN_THIS_FOCUSED_TEST"],
    }
    research = {
        **base,
        "research_features": unavailable,
        "aggregated_research_context": [],
    }
    financial = {
        **base,
        "financial_features": unavailable,
        "financial_metrics_summary": {},
    }
    joined, order = controlled_monthly_join(
        base,
        {"research": research, "financial": financial, "sentiment": result},
    )

    report = {
        "ticker": summary.get("ticker"),
        "query_sent_to_sentiment": query,
        "query_preserved_after_run": result.get("task_query") == query,
        "as_of": context.as_of.isoformat(),
        "target": context.target.isoformat(),
        "horizon_trading_days": context.horizon_trading_days,
        "status": summary.get("status"),
        "quality_passed": summary.get("quality_passed"),
        "quality_failures": summary.get("quality_failures"),
        "pipeline_execution_success": summary.get("pipeline_execution_success"),
        "loop_iterations_used": summary.get("loop_iterations_used"),
        "total_chunks_analyzed": summary.get("total_chunks_analyzed"),
        "unique_document_count": summary.get("unique_document_count"),
        "unique_domain_count": summary.get("unique_domain_count"),
        "fear_greed_score": summary.get("fear_greed_score"),
        "fear_greed_label": summary.get("fear_greed_label"),
        "overall_sentiment": summary.get("overall_sentiment"),
        "conviction_level": summary.get("conviction_level"),
        "model_agreement": summary.get("model_agreement"),
        "extraction_errors": summary.get("extraction_errors"),
        "causal_mismatch_detected": summary.get("causal_mismatch_detected"),
        "feature_version": features.get("version"),
        "feature_missing": features.get("missing"),
        "feature_degraded": features.get("degraded"),
        "feature_missing_flags": features.get("missing_flags"),
        "feature_degraded_flags": features.get("degraded_flags"),
        "feature_evidence_count": len(features.get("evidence_ids", [])),
        "feature_values": values,
        "manager_join_order": order,
        "manager_received_sentiment_features": isinstance(
            joined.get("sentiment_features"), dict
        ),
        "manager_received_sentiment_summary": bool(
            joined.get("sentiment_analysis_summary")
        ),
        "manager_sentiment_result_valid": agent_result_is_valid("sentiment", joined),
        "manager_history_sentiment": [
            item
            for item in joined.get("agent_execution_history", [])
            if item.get("agent_name") == "sentiment"
        ],
    }
    print("SENTIMENT_E2E_RESULT=" + json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
