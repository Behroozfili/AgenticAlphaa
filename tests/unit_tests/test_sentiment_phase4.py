import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
from agents.sentiment_agent import SentimentAgent
from tools.sentiment_tools.finbert_analyzer import FinBertSentimentAnalyzer
from tools.sentiment_tools.vader_scorer import VaderLexiconScorer
from tools.sentiment_tools.fear_greed_calculator import FearGreedIndexCalculator


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
        "manager_directives": {}, "financial_metrics_summary": {},
        "aggregated_research_context": [], "current_plan": {},
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
    assert state["prior_attempt_errors"] == ["retrieve_social_data error: temporary retrieval failure"]
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
