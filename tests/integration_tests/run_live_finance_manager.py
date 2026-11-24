"""Focused live smoke test for FinanceAgent -> Manager handoff."""

import asyncio
import json

from agents.executor_helpers.manager_executor_helper import controlled_monthly_join
from agents.financial_agent import FinancialAnalystAgent
from agents.manager_agent import agent_result_is_valid
from core.forecast_contract import ForecastContext
from evaluation.financial_sanity import evaluate_financial_sanity


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
    run_id = "live-msft-finance-manager-20260920"
    base = {
        "task_query": (
            "Prepare a comprehensive financial assessment of MSFT for its "
            "October 2026 monthly stock forecast, including valuation, growth, "
            "profitability, leverage, liquidity, cash flow, and peer comparison."
        ),
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

    result = await FinancialAnalystAgent(max_loops=3).run(base.copy())
    features = result.get("financial_features", {})
    values = features.get("values", {}) if isinstance(features, dict) else {}
    summary = result.get("financial_metrics_summary", {})

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
    sentiment = {
        **base,
        "sentiment_features": unavailable,
        "sentiment_analysis_summary": {},
    }
    joined, order = controlled_monthly_join(
        base,
        {"research": research, "financial": result, "sentiment": sentiment},
    )

    legacy_manager_keys = (
        "company_name", "sector", "current_price", "market_cap",
        "composite_score", "pe_ratio", "roe", "net_margin",
        "current_ratio", "de_ratio", "revenue_cagr", "forward_pe",
        "peg_ratio", "peer_comparison", "dcf_valuation", "dcf_monte_carlo",
        "capital_allocation", "earnings_quality", "forecast",
        "financial_sanity", "net_debt_to_ebitda", "fcf_margin", "fcf_yield",
        "ocf_ratio", "reverse_dcf", "roic", "roce", "debt_ratio",
        "equity_ratio", "altman_z_score", "data_quality",
    )
    non_null_values = sorted(
        key for key, value in values.items()
        if value is not None and key not in {"availability", "accession_references"}
    )
    manager_sanity = evaluate_financial_sanity(summary)

    report = {
        "ticker": summary.get("ticker"),
        "summary_keys": sorted(summary),
        "validation_passed": summary.get("validation_passed"),
        "loop_iterations_used": summary.get("loop_iterations_used"),
        "analysis_intents": summary.get("analysis_intents"),
        "feature_version": features.get("version"),
        "feature_missing": features.get("missing"),
        "feature_degraded": features.get("degraded"),
        "feature_missing_flags": features.get("missing_flags"),
        "feature_degraded_flags": features.get("degraded_flags"),
        "feature_evidence_count": len(features.get("evidence_ids", [])),
        "feature_availability": values.get("availability"),
        "feature_non_null_values": non_null_values,
        "feature_values": {
            key: values.get(key)
            for key in (
                "valuation_gap", "dcf_long_term_value_per_share",
                "revenue_growth", "earnings_growth", "roe_pct",
                "net_margin_pct", "roic_pct", "net_debt_to_ebitda",
                "debt_ratio", "current_ratio", "quick_ratio",
                "fcf_margin_pct", "fcf_yield_pct", "ocf_ratio",
                "filing_age_days",
            )
        },
        "manager_legacy_keys_missing": [
            key for key in legacy_manager_keys if key not in summary
        ],
        "manager_sanity_passed_on_received_summary": manager_sanity.get("passed"),
        "manager_sanity_score_on_received_summary": manager_sanity.get("score"),
        "manager_sanity_critical_failures": manager_sanity.get("critical_failures"),
        "manager_join_order": order,
        "manager_received_financial_features": isinstance(
            joined.get("financial_features"), dict
        ),
        "manager_received_financial_summary": bool(
            joined.get("financial_metrics_summary")
        ),
        "manager_financial_result_valid": agent_result_is_valid("financial", joined),
        "manager_history_financial": [
            item
            for item in joined.get("agent_execution_history", [])
            if item.get("agent_name") == "financial"
        ],
    }
    print("FINANCE_E2E_RESULT=" + json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
