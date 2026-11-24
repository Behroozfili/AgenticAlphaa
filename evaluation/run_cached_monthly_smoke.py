"""Offline smoke test for cached market -> Quant -> monthly forecast wiring."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agents.executor_helpers.monthly_forecast_helper import assemble_monthly_forecast
from agents.executor_helpers.monthly_narrative_validator import deterministic_narrative
from core.forecast_contract import ForecastContext
from tools.market_tools.market_provider import YahooMarketDataProvider
from tools.market_tools.quant_artifacts import load_quant_artifacts


def _no_network(*_args, **_kwargs):
    raise RuntimeError("offline smoke test attempted network acquisition")


def main() -> None:
    load_dotenv()
    artifact_path = Path(os.environ.get(
        "QUANT_ARTIFACT_PATH", "artifacts/quant/msft_market_only_v1.json",
    ))
    artifacts = load_quant_artifacts(artifact_path)
    model = artifacts.model
    context = ForecastContext(
        ticker="MSFT",
        as_of="2026-09-18T20:00:00Z",
        target="2026-10-19",
        lookback_days=365,
        benchmark="SPY",
        analysis_mode="LIVE",
        cutoff_policy="MONTH_TO_DATE",
        schema_version="forecast-schema-v2",
        dataset_version=model.dataset_version,
        feature_version=model.feature_version,
        model_version=model.model_version,
    )
    run_id = "cached-msft-monthly-smoke"
    provider = YahooMarketDataProvider(
        cache_dir=Path(".cache/market-data"), fetcher=_no_network,
    )
    market = provider(context)
    research = {
        "version": "research-features-v2",
        "values": {
            "dataset_version": context.dataset_version,
            "run_id": run_id,
            "source_diversity": 3,
            "vector_coverage": True,
            "graph_coverage": True,
            "directional_score": 0.45,
            "claim_confidence": 0.78,
            "classification_coverage": 0.28,
            "classified_claim_count": 2,
            "catalysts": ["research:r0", "research:r1"],
            "claims": [{
                "evidence_id": "research:r0",
                "citation_reference_id": "citation:research-r0",
                "source": "rag_hybrid_query",
                "timestamp": context.as_of.isoformat().replace("+00:00", "Z"),
                "direction": "POSITIVE", "strength": 0.8,
                "confidence": 0.78, "recency_weight": 1.0,
                "event_signals": ["raises guidance", "record revenue"],
            }],
            "risks": [],
        },
        "evidence_ids": [f"research:r{i}" for i in range(7)],
        "missing": False, "degraded": False, "errors": [],
    }
    financial = {
        "version": "financial-v1",
        "values": {
            "dataset_version": context.dataset_version,
            "run_id": run_id,
            "valuation_gap": 0.05,
            "revenue_growth": 0.1,
            "roe_pct": 20.0,
            "current_ratio": 1.5,
            "fcf_margin_pct": 15.0,
        },
        "evidence_ids": [f"financial:f{i}" for i in range(64)],
        "missing": False, "degraded": False, "errors": [],
    }
    sentiment = {
        "version": "sentiment-v1",
        "values": {
            "dataset_version": context.dataset_version,
            "run_id": run_id,
            "aggregate_7d": 0.24,
            "aggregate_14d": 0.20,
            "aggregate_30d": 0.18,
            "momentum_7d_vs_30d": 0.06,
            "volume_30d": 11,
            "confidence": 0.4,
        },
        "evidence_ids": [f"sentiment:s{i}" for i in range(11)],
        "missing": False, "degraded": False, "errors": [],
    }
    service = artifacts.service()
    features = service.build_features(
        context=context,
        market=market,
        research_features=research,
        financial_features=financial,
        sentiment_features=sentiment,
    )
    prediction = service.forecast(
        context=context, market_features=features, run_id=run_id,
    )
    market_slot = features.to_state_slot()
    market_slot["values"].update({
        "run_id": run_id,
        "quant_prediction": prediction,
    })
    assembled = assemble_monthly_forecast({
        "forecast_context": context.to_dict(),
        "manager_directives": {"ticker": "MSFT", "run_id": run_id},
        "research_features": research,
        "financial_features": financial,
        "sentiment_features": sentiment,
        "market_features": market_slot,
    })
    result = assembled.with_narrative(deterministic_narrative(assembled)).to_dict()
    print(json.dumps({
        "status": result["status"],
        "action": result["action"],
        "probability_positive": result["probability_positive"],
        "expected_return_21d": result["expected_return_21d"],
        "expected_excess_return": result["expected_excess_return"],
        "target_date": result["target_date"],
        "quality_gate_passed": result["quality_gate_passed"],
        "warnings": result["warnings"],
        "market_evidence_count": len(features.evidence_ids),
        "model_variant": result["model_variant"],
        "specialist_contributions": result["specialist_contributions"],
        "decision_drivers": result["decision_drivers"],
        "narrative": result["narrative"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
