from __future__ import annotations

import logging

from agents.executor_helpers.finance_executor_helper import infer_peers
from agents.executor_helpers.monthly_financial_features import build_monthly_financial_features
from agents.executor_helpers.monthly_forecast_helper import assemble_monthly_forecast
from agents.executor_helpers.sentiment_helper import build_monthly_sentiment_features
from api.main import _harden_dependency_logging
from tests.unit_tests.test_agents.test_financial_agent import STAGE5_RUN_ID, stage5_context
from tests.unit_tests.test_stage8_manager_integration import complete_state


def test_peer_lookup_normalizes_yahoo_dash_variants():
    assert infer_peers("Technology", "Software - Infrastructure", "MSFT") == ["ORCL", "CRM"]
    assert infer_peers("Technology", "Software—Infrastructure", "MSFT") == ["ORCL", "CRM"]


def test_financial_adapter_caps_shared_evidence_but_records_total():
    facts = [{
        "filed_at": "2026-02-01",
        "accession_number": f"acc-{index}",
        "xbrl_concept": "Revenue",
        "period_end": "2025-12-31",
        "evidence_id": f"financial:e{index}",
    } for index in range(70)]
    features = build_monthly_financial_features(
        raw={
            "yahoo_ratios": {"current_price": 100.0},
            "revenue_growth": {"revenue_growth_ttm": 0.1},
            "xbrl_financials": {"revenue": facts},
        },
        calculated={
            "dcf": {"base": {"intrinsic_value_per_share": 110.0}},
            "roe": {"roe_pct": 20.0},
            "net_debt_to_ebitda": {"net_debt_to_ebitda": 1.0},
            "current_ratio_calc": {"current_ratio": 1.5},
            "fcf_margin": {"fcf_margin_pct": 15.0},
        },
        forecast_context=stage5_context(), run_id=STAGE5_RUN_ID,
    )
    slot = features.to_state_slot()
    assert len(slot["evidence_ids"]) == 64
    assert slot["values"]["evidence_reference_count"] == 70
    assert slot["values"]["evidence_references_truncated"] is True


def test_monthly_sentiment_accepts_valid_30d_when_7d_and_14d_are_empty():
    texts = ["older positive", "older neutral", "older negative"]
    metadata = [{
        "ticker": "AMD",
        "published_at": f"2026-06-{day:02d}T20:00:00Z",
        "dataset_version": "sentiment-fixture-v1",
        "url": f"https://source{index}.example/article",
    } for index, day in enumerate((2, 3, 4), start=1)]
    features = build_monthly_sentiment_features(
        texts=texts,
        sources_metadata=metadata,
        finbert_result={"status": "VALID", "chunk_scores": [
            {"bullish_prob": 0.7, "bearish_prob": 0.2} for _ in texts
        ]},
        vader_result={"status": "VALID", "chunk_scores": [
            {"compound": 0.2} for _ in texts
        ]},
        fear_greed_result={"status": "VALID", "score": 0.2, "confidence": 0.7, "polarity": "NEUTRAL"},
        selected_attempt={
            "status": "VALID", "quality_passed": True,
            "retrieval_stats": {"unique_documents": 3, "unique_domains": 3},
        },
        forecast_context={
            **stage5_context("AMD"),
            "as_of": "2026-06-30T20:00:00Z",
            "target": "2026-07-30",
            "dataset_version": "sentiment-fixture-v1",
        },
        run_id="sentiment-optional-windows",
    )
    slot = features.to_state_slot()
    assert slot["values"]["availability"]["window_7d"] is False
    assert slot["values"]["availability"]["window_14d"] is False
    assert slot["values"]["availability"]["window_30d"] is True
    assert slot["missing"] is False


def test_oversized_specialist_evidence_fails_closed_without_crashing():
    state = complete_state()
    state["financial_features"]["evidence_ids"] = [f"financial:e{index}" for index in range(65)]
    result = assemble_monthly_forecast(state).to_dict()
    assert result["status"] == "INSUFFICIENT_DATA"
    assert "FINANCIAL_FEATURES_EVIDENCE_REJECTED" in result["warnings"]


def test_sensitive_http_loggers_cannot_remain_at_debug():
    for name in ("httpcore", "httpx", "hpack", "supabase", "postgrest"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    _harden_dependency_logging()
    assert all(
        logging.getLogger(name).level >= logging.WARNING
        for name in ("httpcore", "httpx", "hpack", "supabase", "postgrest")
    )
