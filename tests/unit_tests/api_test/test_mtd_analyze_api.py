from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from api.config import Settings
from api.routes.analyze import (
    AnalyzeRequest,
    _active_artifact_tickers,
    _latest_cached_cutoff,
    _monthly_request_context,
)
from core.forecast_contract import is_us_regular_trading_session, regular_session_close_utc


def _request():
    settings = Settings(_env_file=None)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_settings=settings)))


def test_api_accepts_mid_month_completed_close_for_live_mtd():
    body = AnalyzeRequest(
        query="Forecast MSFT using current month data",
        ticker="MSFT", as_of="2024-05-30", mode="LIVE",
        cutoff_policy="MONTH_TO_DATE", days_back=30,
    )
    context, run_id = _monthly_request_context(_request(), body)

    assert context["as_of"] == "2024-05-30T20:00:00Z"
    assert context["cutoff_policy"] == "MONTH_TO_DATE"
    assert context["lookback_days"] == Settings(_env_file=None).FORECAST_LOOKBACK_DAYS
    assert context["target"] == "2024-07-01"
    assert run_id


def test_api_rejects_live_cutoff_whose_close_has_not_happened():
    now = datetime.now(timezone.utc)
    day = now.date()
    while not is_us_regular_trading_session(day) or regular_session_close_utc(day) <= now:
        day += timedelta(days=1)
    body = AnalyzeRequest(
        query="Forecast MSFT using future unavailable data",
        ticker="MSFT", as_of=day.isoformat(), mode="LIVE",
        cutoff_policy="MONTH_TO_DATE", days_back=30,
    )
    with pytest.raises(ValueError, match="completed market close"):
        _monthly_request_context(_request(), body)



def test_forecast_defaults_use_active_artifact_and_latest_common_cache(tmp_path):
    artifact = tmp_path / "quant.json"
    artifact.write_text(json.dumps({
        "universe_manifest": {
            "constituents": [{"ticker": "MSFT", "benchmark": "SPY"}],
        },
    }), encoding="utf-8")
    cache = tmp_path / "market"
    for symbol, ends in {
        "MSFT": ("2026-09-21", "2026-09-22"),
        "SPY": ("2026-09-21",),
    }.items():
        folder = cache / "adjusted-ohlcv-cache-v1" / symbol
        folder.mkdir(parents=True)
        for end in ends:
            (folder / f"2025-09-21_{end}.json").write_text("{}", encoding="utf-8")

    settings = Settings(_env_file=None).model_copy(update={
        "QUANT_ARTIFACT_PATH": str(artifact),
        "MARKET_DATA_CACHE_DIR": str(cache),
        "LIVE_DATASET_VERSION": "adjusted-ohlcv-cache-v1",
        "FORECAST_BENCHMARK": "SPY",
    })

    assert _active_artifact_tickers(settings) == ["MSFT"]
    assert _latest_cached_cutoff(settings, ticker="MSFT") == date(2026, 9, 21)