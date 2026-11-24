from __future__ import annotations

import pytest

from agents.executor_helpers.monthly_forecast_helper import assemble_monthly_forecast
from tests.unit_tests.test_stage8_manager_integration import (
    FIXTURE,
    complete_state,
    feature_slot,
    manager,
)


@pytest.mark.parametrize("kind", ["research", "financial", "sentiment"])
@pytest.mark.parametrize("condition", ["missing", "degraded", "no_evidence"])
def test_each_specialist_below_minimum_fails_quality_gate(kind, condition):
    state = complete_state()
    slot = state[f"{kind}_features"]
    if condition == "missing":
        slot["missing"] = True
    elif condition == "degraded":
        slot["degraded"] = True
    else:
        slot["evidence_ids"] = []

    result = assemble_monthly_forecast(state).to_dict()

    assert result["quality_gate_passed"] is False
    assert result["quality_gate_failures"]
    # Diagnostic numeric output remains available, but publication is decided
    # separately by Manager and must be blocked.
    assert result["probability_positive"] is not None


class _BelowMinimumSpecialist:
    def __init__(self, kind: str):
        self.kind = kind

    async def run(self, state):
        context = state["forecast_context"]
        run_id = state["manager_directives"]["run_id"]
        slot = feature_slot(self.kind, context, run_id, degraded=True)
        return {**state, f"{self.kind}_features": slot}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["research", "financial", "sentiment"])
async def test_manager_never_publishes_when_a_required_specialist_is_degraded(kind):
    subject = manager()
    subject._agents[kind] = _BelowMinimumSpecialist(kind)
    context = FIXTURE["contexts"]["MSFT"]

    result = await subject.run(
        "Analyze MSFT",
        {"ticker": "MSFT", "run_id": "quality-gate-msft", "forecast_context": context},
    )

    assert result["monthly_forecast"]["quality_gate_passed"] is False
    assert result["publication_status"] == "DATA_QUALITY_BLOCKED"
    assert result["execution_status"] == "failed"

