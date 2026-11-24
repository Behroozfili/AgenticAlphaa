import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
from agents.sentiment_agent import SentimentAgent
from core.forecast_contract import ForecastContext
from tools.sentiment_tools.finbert_analyzer import FinBertSentimentAnalyzer
from tools.sentiment_tools.vader_scorer import VaderLexiconScorer
from tools.sentiment_tools.fear_greed_calculator import FearGreedIndexCalculator
from agents.executor_helpers.sentiment_helper import (
    build_monthly_sentiment_features,
    compact_sentiment_summary,
    select_point_in_time_evidence,
)


TEST_FORECAST_CONTEXT = ForecastContext(
    ticker="AMD", as_of="2026-06-30T20:00:00Z",
    schema_version="2.1", dataset_version="sentiment-fixture-v1",
    feature_version="monthly-sentiment-features-v1", model_version="test-model-v1",
).to_dict()


def tool_result(payload):
    result = MagicMock()
    result.content = [MagicMock(text=json.dumps(payload))]
    return result


def agent_state():
    return {
        "messages": [], "retrieved_chunks": [], "sources_metadata": [],
        "finbert_result": {}, "vader_result": {}, "fear_greed_result": {},
        "brain_reasoning": "", "loop_counter": 0, "extraction_errors": [],
        "prior_attempt_errors": [], "attempts": [], "sentiment_status": "NO_DATA",
        "sentiment_polarity": None, "task_query": "AMD sentiment",
        "manager_directives": {
            "forecast_context": TEST_FORECAST_CONTEXT,
            "run_id": "sentiment-phase4-test", "ticker": "AMD",
        }, "financial_metrics_summary": {},
        "aggregated_research_context": [], "current_plan": {},
        "forecast_context": TEST_FORECAST_CONTEXT, "run_id": "sentiment-phase4-test",
    }


def sentiment_agent():
    return SentimentAgent("dummy.py", llm_client=MagicMock())


def test_empty_analyzers_return_no_data_with_null_scores_not_neutral():
    finbert = FinBertSentimentAnalyzer.__new__(FinBertSentimentAnalyzer)._empty_result(2)
    vader = VaderLexiconScorer.__new__(VaderLexiconScorer)._empty_result(2)
    combined = FearGreedIndexCalculator().calculate(finbert, vader)
    for result in (finbert, vader, combined):
        assert result.status == "NO_DATA"
        assert result.polarity is None
        assert result.label is None
    assert finbert.bullish_prob is None
    assert vader.compound is None
    assert combined.score is None


def test_stage06_fixed_windows_reject_future_and_duplicate_evidence():
    fixture = json.loads(
        Path("tests/fixtures/stage6_sentiment_point_in_time.json").read_text(encoding="utf-8")
    )
    eligible, errors = select_point_in_time_evidence(
        texts=fixture["texts"], metadata=fixture["sources_metadata"],
        forecast_context=fixture["forecast_context"], run_id=fixture["run_id"],
    )
    assert len(eligible) == 4
    assert "FUTURE_EVIDENCE_REJECTED" in errors
    assert "DUPLICATE_EVIDENCE_REJECTED" in errors

    features = build_monthly_sentiment_features(
        texts=fixture["texts"], sources_metadata=fixture["sources_metadata"],
        finbert_result=fixture["finbert_result"], vader_result=fixture["vader_result"],
        fear_greed_result=fixture["fear_greed_result"],
        forecast_context=fixture["forecast_context"], run_id=fixture["run_id"],
    )
    assert (features.volume_7d, features.volume_14d, features.volume_30d) == (2, 3, 4)
    assert features.aggregate_7d is not None
    assert features.aggregate_14d is not None
    assert features.aggregate_30d is not None
    assert len(features.evidence_ids) == 4
    assert "4556d13015211d73dccd3fdd39d39232506f3e43" in features.model_versions["finbert"]


def test_stage06_missing_timestamps_are_missing_not_neutral():
    features = build_monthly_sentiment_features(
        texts=["unknown time"], sources_metadata=[{"ticker": "AMD"}],
        finbert_result={}, vader_result={}, fear_greed_result={},
        forecast_context=TEST_FORECAST_CONTEXT, run_id="missing-time-run",
    )
    slot = features.to_state_slot()
    assert slot["missing"] is True
    assert slot["values"]["aggregate_30d"] is None
    assert "EVIDENCE_TIMESTAMP_MISSING" in slot["degraded_flags"]


def _typed_feature_inputs():
    texts = ["positive one", "positive two", "positive three"]
    metadata = [
        {
            "ticker": "AMD",
            "published_at": f"2026-06-{day:02d}T20:00:00Z",
            "dataset_version": "sentiment-fixture-v1",
            "url": f"https://source{index}.example/article",
        }
        for index, day in enumerate((30, 29, 28), start=1)
    ]
    finbert = {
        "status": "VALID",
        "total_chunks": 3,
        "chunk_scores": [
            {"bullish_prob": 0.8, "bearish_prob": 0.1},
            {"bullish_prob": 0.7, "bearish_prob": 0.2},
            {"bullish_prob": 0.6, "bearish_prob": 0.2},
        ],
    }
    vader = {
        "status": "VALID",
        "total_chunks": 3,
        "chunk_scores": [{"compound": 0.4}, {"compound": 0.3}, {"compound": 0.2}],
    }
    fusion = {
        "status": "VALID", "score": 0.5, "confidence": 0.8,
        "polarity": "BULLISH",
    }
    selected = {
        "status": "VALID", "quality_passed": True,
        "retrieval_stats": {"unique_documents": 3, "unique_domains": 3},
    }
    return texts, metadata, finbert, vader, fusion, selected


def test_typed_features_keep_partial_aggregate_but_expose_one_model_failure():
    texts, metadata, finbert, _, fusion, selected = _typed_feature_inputs()
    vader = {
        "status": "ERROR", "total_chunks": 0,
        "chunk_scores": [{"compound": 0.9}, {"compound": 0.9}, {"compound": 0.9}],
    }
    selected.update(status="ERROR", quality_passed=False)

    features = build_monthly_sentiment_features(
        texts=texts, sources_metadata=metadata,
        finbert_result=finbert, vader_result=vader,
        fear_greed_result=fusion, selected_attempt=selected,
        forecast_context=TEST_FORECAST_CONTEXT, run_id="one-model-failure",
    )
    slot = features.to_state_slot()

    assert slot["values"]["aggregate_7d"] == 0.533333
    assert slot["degraded"] is True
    assert slot["values"]["availability"]["finbert"] is True
    assert slot["values"]["availability"]["vader"] is False
    assert "vader" in slot["missing_flags"]
    assert "VADER_ERROR" in slot["degraded_flags"]
    assert "PARTIAL_MODEL_COVERAGE" in slot["degraded_flags"]
    assert "vader" not in slot["values"]["model_versions"]


def test_typed_features_keep_model_aggregates_but_expose_fusion_failure():
    texts, metadata, finbert, vader, _, selected = _typed_feature_inputs()
    fusion = {"status": "ERROR", "score": None, "confidence": 0.0}
    selected.update(status="ERROR", quality_passed=False)

    features = build_monthly_sentiment_features(
        texts=texts, sources_metadata=metadata,
        finbert_result=finbert, vader_result=vader,
        fear_greed_result=fusion, selected_attempt=selected,
        forecast_context=TEST_FORECAST_CONTEXT, run_id="fusion-failure",
    )
    slot = features.to_state_slot()

    assert slot["values"]["aggregate_7d"] is not None
    assert slot["degraded"] is True
    assert slot["values"]["availability"]["finbert"] is True
    assert slot["values"]["availability"]["vader"] is True
    assert slot["values"]["availability"]["fusion"] is False
    assert "fusion" in slot["missing_flags"]
    assert "FUSION_ERROR" in slot["degraded_flags"]
    assert "fusion" not in slot["values"]["model_versions"]


def test_typed_features_propagate_selected_attempt_low_coverage():
    texts, metadata, finbert, vader, fusion, selected = _typed_feature_inputs()
    selected.update(
        status="LOW_COVERAGE", quality_passed=False,
        retrieval_stats={"unique_documents": 1, "unique_domains": 1},
    )

    features = build_monthly_sentiment_features(
        texts=texts, sources_metadata=metadata,
        finbert_result=finbert, vader_result=vader,
        fear_greed_result=fusion, selected_attempt=selected,
        forecast_context=TEST_FORECAST_CONTEXT, run_id="low-coverage",
    )
    slot = features.to_state_slot()

    assert slot["values"]["aggregate_7d"] is not None
    assert slot["missing"] is False
    assert slot["degraded"] is True
    assert slot["values"]["availability"]["coverage_quality"] is False
    assert slot["values"]["availability"]["selected_attempt_quality"] is False
    assert "LOW_COVERAGE" in slot["degraded_flags"]
    assert "SELECTED_ATTEMPT_QUALITY_FAILED" in slot["degraded_flags"]


def test_typed_features_fail_closed_for_private_statuses_and_polarity():
    private = "PRIVATE_SENTINEL"
    texts, metadata, finbert, vader, fusion, selected = _typed_feature_inputs()
    finbert["status"] = private
    vader["status"] = private * 8
    fusion.update(status=f"{private}!", polarity=private)
    selected.update(status=f"{private}\nSELECTED", quality_passed=True)

    features = build_monthly_sentiment_features(
        texts=texts, sources_metadata=metadata,
        finbert_result=finbert, vader_result=vader,
        fear_greed_result=fusion, selected_attempt=selected,
        forecast_context=TEST_FORECAST_CONTEXT, run_id="private-statuses",
    )
    slot = features.to_state_slot()
    compact = compact_sentiment_summary({
        "status": "ERROR", "polarity": features.polarity,
        "model_versions": dict(features.model_versions),
        "missing_flags": list(features.missing_flags),
        "degraded_flags": list(features.degraded_flags),
    })
    serialized = json.dumps({"slot": slot, "compact": compact}, sort_keys=True)

    assert private not in serialized
    assert slot["degraded"] is True
    assert slot["values"]["aggregate_7d"] is None
    assert slot["values"]["polarity"] is None
    assert slot["values"]["model_statuses"] == {
        "finbert": "UNKNOWN", "vader": "UNKNOWN",
        "fusion": "UNKNOWN", "selected_attempt": "UNKNOWN",
    }
    for key in (
        "finbert", "vader", "fusion", "selected_attempt_quality", "polarity",
    ):
        assert slot["values"]["availability"][key] is False
    assert slot["values"]["model_versions"] == {}
    assert {
        "FINBERT_STATUS_UNKNOWN", "VADER_STATUS_UNKNOWN",
        "FUSION_STATUS_UNKNOWN", "SELECTED_ATTEMPT_STATUS_UNKNOWN",
        "POLARITY_UNKNOWN",
    } <= set(slot["degraded_flags"])


def test_stage06_rejects_future_effective_from_even_with_past_publication():
    eligible, errors = select_point_in_time_evidence(
        texts=["must never be scored"],
        metadata=[{
            "ticker": "AMD",
            "published_at": "2026-06-20T12:00:00Z",
            "effective_from": "2026-07-01T00:00:00Z",
            "url": "https://example.com/future-effective",
        }],
        forecast_context=TEST_FORECAST_CONTEXT,
        run_id="future-effective-run",
    )
    assert eligible == []
    assert "FUTURE_EVIDENCE_REJECTED" in errors


def test_stage06_compact_summary_drops_private_payloads():
    compact = compact_sentiment_summary({
        "status": "VALID", "fear_greed_score": 0.2, "evidence_ids": ["e1"],
        "brain_reasoning": "private", "attempts": [{"raw": "private"}],
        "sources_metadata": [{"text": "raw"}], "raw_chunks": ["raw"],
    })
    assert compact == {"status": "VALID", "fear_greed_score": 0.2, "evidence_ids": ["e1"]}


@pytest.mark.asyncio
async def test_retry_attempts_are_isolated_and_prior_errors_do_not_contaminate_success():
    agent = sentiment_agent()
    state = agent_state()
    first = AsyncMock()
    first.call_tool.return_value = tool_result({"error": "temporary retrieval failure"})
    await agent._execute_sentiment_pipeline(first, state, {"retrieval_query": "narrow", "days_back": 7})
    assert state["attempts"][0]["status"] == "ERROR"

    second = AsyncMock()
    second.call_tool.side_effect = [
        tool_result({
            "chunks": ["a", "b", "c"],
            "sources_metadata": [
                {"url": "https://one.example/a"}, {"url": "https://two.example/b"},
                {"url": "https://three.example/c"},
            ],
        }),
        tool_result({"status": "VALID", "bullish_prob": .7, "bearish_prob": .1,
                     "neutral_prob": .2, "label": "Bullish", "total_chunks": 3}),
        tool_result({"status": "VALID", "compound": .4, "label": "Bullish", "total_chunks": 3}),
        tool_result({"status": "VALID", "score": .5, "label": "Greed",
                     "polarity": "BULLISH", "confidence": .5}),
    ]
    await agent._execute_sentiment_pipeline(second, state, {"retrieval_query": "broad", "days_back": 14})
    assert len(state["attempts"]) == 2
    assert state["sentiment_status"] == "VALID"
    assert state["extraction_errors"] == []
    assert state["prior_attempt_errors"] == [
        "retrieve_social_data: SENTIMENT_TOOL_DATA_UNAVAILABLE"
    ]
    assert state["attempts"][1]["errors"] == []
    assert state["attempts"][1]["retrieval"]["chunks"] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_low_source_independence_produces_low_coverage_and_reduces_confidence():
    agent = sentiment_agent()
    state = agent_state()
    session = AsyncMock()
    session.call_tool.side_effect = [
        tool_result({"chunks": ["a", "b", "c"], "sources_metadata": [
            {"url": "https://same.example/a"}, {"url": "https://same.example/b"},
            {"url": "https://same.example/c"},
        ]}),
        tool_result({"bullish_prob": .7, "bearish_prob": .1, "neutral_prob": .2, "total_chunks": 3}),
        tool_result({"compound": .4, "total_chunks": 3}),
        tool_result({"score": .5, "label": "Greed", "confidence": .9}),
    ]
    await agent._execute_sentiment_pipeline(session, state, {"retrieval_query": "q", "days_back": 7})
    assert state["sentiment_status"] == "LOW_COVERAGE"
    assert state["fear_greed_result"]["confidence"] < .9
    assert state["fear_greed_result"]["coverage"]["unique_domains"] == 1


@pytest.mark.asyncio
async def test_manager_never_accepts_no_data_as_success():
    memory = MagicMock()
    manager = ManagerAgent(
        research_agent=MagicMock(), financial_agent=MagicMock(), sentiment_agent=MagicMock(),
        memory=memory, llm_client=AsyncMock(),
    )
    feedback = await manager._brain_evaluate(
        "SentimentAgent", {"sentiment_analysis_summary": {
            "status": "NO_DATA", "polarity": None, "fear_greed_score": None,
            "total_chunks_analyzed": 0,
        }}, {"short_term": {}},
    )
    assert feedback.passed is False
    assert feedback.next_action == "rerun_sentiment"
