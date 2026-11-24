from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from core.error_handler import DataUnavailableAsOfError
from core.forecast_contract import AnalysisMode, ForecastContext, is_us_regular_trading_session
from tools.market_tools.market_provider import YahooMarketDataProvider


def _context(*, mode: AnalysisMode = AnalysisMode.LIVE) -> ForecastContext:
    return ForecastContext(
        ticker="MSFT", as_of="2024-06-28T20:00:00Z", benchmark="SPY",
        analysis_mode=mode, schema_version="forecast-v1",
        dataset_version="adjusted-ohlcv-cache-v1",
        feature_version="monthly-market-features-v1", model_version="monthly-model-v1",
    )


def _rows(start: date, end: date, base: float):
    output = []
    cursor = start
    index = 0
    while cursor <= end:
        if is_us_regular_trading_session(cursor):
            close = base + index * 0.1
            output.append({
                "date": cursor.isoformat(), "open": close - 0.1, "high": close + 0.2,
                "low": close - 0.2, "close": close, "adjusted_close": close,
                "volume": 1_000_000 + index,
            })
            index += 1
        cursor += timedelta(days=1)
    return output


def test_live_provider_populates_then_reuses_immutable_exact_range_cache(tmp_path):
    calls: list[str] = []

    def fetch(symbol, start, end):
        calls.append(symbol)
        rows = _rows(start, end, 100 if symbol == "MSFT" else 200)
        # Simulate an upstream client that over-fetches future data/actions.
        rows.append({"date": "2024-07-01", "open": 1, "high": 1, "low": 1,
                     "close": 1, "adjusted_close": 1, "volume": 1})
        return rows, [
            {"effective_date": end.isoformat(), "type": "dividend", "value": 0.5},
            {"effective_date": "2024-07-01", "type": "split", "value": 2},
        ]

    provider = YahooMarketDataProvider(
        cache_dir=tmp_path, fetcher=fetch,
        clock=lambda: datetime(2024, 6, 29, tzinfo=UTC),
    )
    first = provider(_context())
    second = provider(_context())

    assert calls == ["MSFT", "SPY"]
    assert first.cache_hashes == second.cache_hashes
    assert first.ticker.bars[-1].session == date(2024, 6, 28)
    assert all(bar.session <= date(2024, 6, 28) for bar in first.ticker.bars)
    assert [action.action_type for action in first.ticker.corporate_actions] == ["dividend"]
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_historical_provider_is_cache_only_and_never_calls_fetcher(tmp_path):
    called = False

    def fetch(symbol, start, end):
        nonlocal called
        called = True
        return [], []

    provider = YahooMarketDataProvider(cache_dir=tmp_path, fetcher=fetch)
    with pytest.raises(DataUnavailableAsOfError, match="network fallback is forbidden"):
        provider(_context(mode=AnalysisMode.HISTORICAL_BACKTEST))
    assert called is False


def test_existing_corrupt_cache_fails_closed_without_refetch(tmp_path):
    context = _context()
    provider = YahooMarketDataProvider(cache_dir=tmp_path, fetcher=lambda *_: pytest.fail("must not fetch"))
    end = context.as_of.date()
    start = end - timedelta(days=context.lookback_days)
    path = provider._path(context=context, symbol="MSFT", start=start, end=end)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"not": "a valid cache"}), encoding="utf-8")

    with pytest.raises((ValueError, DataUnavailableAsOfError)):
        provider(context)

