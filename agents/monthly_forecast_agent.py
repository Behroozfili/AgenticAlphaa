"""Explanation-only agent over the deterministic Stage 07 forecast."""

from __future__ import annotations

import json
from typing import Any, Mapping

from agents.executor_helpers.monthly_forecast_helper import (
    MonthlyForecastResult, assemble_monthly_forecast,
)
from agents.executor_helpers.monthly_narrative_validator import (
    deterministic_narrative, narrative_matches_quant,
)
from agents.prompts.prompt_monthly_forecast import MONTHLY_FORECAST_SYSTEM_PROMPT
from agents.state import SharedManagerState


class MonthlyForecastAgent:
    """Reads Shared State only; it never dispatches another agent or alters Quant."""

    def __init__(self, *, llm_client: Any | None = None, model: str = "claude-haiku-4-5") -> None:
        self._llm = llm_client
        self._model = model

    async def _explain(self, result: MonthlyForecastResult) -> str:
        if self._llm is None:
            return deterministic_narrative(result)
        payload = result.to_dict()
        payload.pop("narrative", None)
        try:
            response = await self._llm.messages.create(
                model=self._model, max_tokens=320, temperature=0.0,
                system=MONTHLY_FORECAST_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload, sort_keys=True)}],
            )
            text = response.content[0].text
            candidate = str(text).strip()[:4000]
            return (
                candidate
                if narrative_matches_quant(candidate, result)
                else deterministic_narrative(result)
            )
        except Exception:
            return deterministic_narrative(result)

    async def run(self, shared_state: SharedManagerState) -> SharedManagerState:
        result = assemble_monthly_forecast(shared_state)
        narrative = await self._explain(result)
        final = result.with_narrative(narrative)
        return {**shared_state, "monthly_forecast": final.to_state_slot()}


__all__ = ["MonthlyForecastAgent", "deterministic_narrative", "narrative_matches_quant"]
