from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.monthly_forecast_agent import MonthlyForecastAgent
from tests.unit_tests.test_stage8_manager_integration import complete_state
from tools.market_tools.specialist_fusion import (
    SPECIALIST_FUSION_VARIANT_TOKEN,
    SPECIALIST_FUSION_VERSION,
)


async def _narrative(text: str) -> str:
    llm = AsyncMock()
    llm.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=text)]
    )
    result = await MonthlyForecastAgent(llm_client=llm).run(complete_state())
    return result["monthly_forecast"]["narrative"]


@pytest.mark.asyncio
async def test_compliant_action_and_rounded_quant_numbers_are_kept():
    text = "Action BUY with probability 61% and expected return 4% over 21 sessions."
    assert await _narrative(text) == text


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "SELL because the expected return is -99%.",
    "Action BUY with probability 99% over 21 sessions.",
    "BUY is not warranted despite the model output.",
    "Prefer HOLD instead of taking the structured BUY action.",
])
async def test_action_or_number_contradictions_use_deterministic_fallback(text):
    narrative = await _narrative(text)
    assert text not in narrative
    assert "action BUY" in narrative
    assert "SELL" not in narrative
    assert "99%" not in narrative



def _fusion_state() -> dict:
    state = deepcopy(complete_state())
    prediction = state["market_features"]["values"]["quant_prediction"]
    prediction.update({
        "model_variant": f"full_model_v1-msft+{SPECIALIST_FUSION_VARIANT_TOKEN}",
        "specialist_fusion_version": SPECIALIST_FUSION_VERSION,
        "specialist_scores": {"research": 0.4, "financial": 0.5, "sentiment": 0.2},
        "specialist_return_adjustments": {"research": 0.001, "financial": 0.003, "sentiment": 0.001},
        "specialist_logit_adjustments": {"research": 0.08, "financial": 0.175, "sentiment": 0.06},
        "specialist_total_return_adjustment": 0.005,
        "base_model_probability_positive": prediction["probability_positive"],
        "base_model_expected_return_21d": prediction["expected_return_21d"] - 0.005,
        "base_model_expected_excess_return": prediction["expected_excess_return"] - 0.005,
    })
    research_id = state["research_features"]["evidence_ids"][0]
    state["research_features"]["values"]["claims"] = [{
        "evidence_id": research_id,
        "citation_reference_id": "citation:test",
        "source": "tavily_search",
        "timestamp": state["forecast_context"]["as_of"],
        "direction": "POSITIVE",
        "strength": 0.8,
        "confidence": 0.75,
        "recency_weight": 1.0,
        "event_signals": ["raises guidance", "record revenue"],
    }]
    state["financial_features"]["values"].update({"revenue_growth": 0.12, "roe_pct": 0.25})
    state["sentiment_features"]["values"].update({
        "aggregate_7d": 0.2, "aggregate_14d": 0.1,
        "aggregate_30d": 0.05, "momentum_7d_vs_30d": 0.15,
        "confidence": 0.8,
    })
    return state


@pytest.mark.asyncio
async def test_deterministic_explanation_names_each_specialist_with_human_reasons():
    result = await MonthlyForecastAgent().run(_fusion_state())
    forecast = result["monthly_forecast"]
    research_id = result["research_features"]["evidence_ids"][0]

    assert {driver["specialist"] for driver in forecast["decision_drivers"]} == {
        "research", "financial", "sentiment",
    }
    assert all(driver.get("reason") for driver in forecast["decision_drivers"])
    assert research_id not in forecast["narrative"]
    assert "raises guidance" in forecast["narrative"]
    assert "Research contributed" in forecast["narrative"]
    assert "Finance contributed" in forecast["narrative"]
    assert "Sentiment contributed" in forecast["narrative"]


@pytest.mark.asyncio
async def test_unknown_evidence_citation_is_replaced_by_grounded_fallback():
    llm = AsyncMock()
    llm.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="Action BUY because of research [unknown-evidence].")]
    )
    result = await MonthlyForecastAgent(llm_client=llm).run(_fusion_state())
    narrative = result["monthly_forecast"]["narrative"]

    assert "[unknown-evidence]" not in narrative
    assert "[research:msft]" not in narrative
    assert "raises guidance" in narrative