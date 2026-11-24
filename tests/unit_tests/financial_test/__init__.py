"""Shared deterministic offline identity for Stage 05 financial tests."""

from core.forecast_contract import ForecastContext

STAGE5_RUN_ID = "stage05-offline-test"


def stage5_context(ticker: str = "NVDA", mode: str = "LIVE") -> dict:
    return ForecastContext(
        ticker=ticker, as_of="2026-06-30T20:00:00Z",
        schema_version="forecast-v1", dataset_version="stage5-fixture-v1",
        feature_version="monthly-financial-features-v1", model_version="model-v1",
        analysis_mode=mode,
    ).to_dict()


__all__ = ["STAGE5_RUN_ID", "stage5_context"]
