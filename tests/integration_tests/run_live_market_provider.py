"""Manual Yahoo smoke: python -m tests.integration_tests.run_live_market_provider"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.forecast_contract import (
    ForecastContext,
    ForecastCutoffPolicy,
    is_us_regular_trading_session,
    regular_session_close_utc,
)
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION
from tools.market_tools.market_data import MARKET_DATA_VERSION
from tools.market_tools.market_provider import YahooMarketDataProvider


def _last_completed_session(now: datetime):
    day = now.date()
    while not is_us_regular_trading_session(day) or regular_session_close_utc(day) > now:
        day -= timedelta(days=1)
    return day


def main() -> None:
    now = datetime.now(UTC)
    day = _last_completed_session(now)
    context = ForecastContext(
        ticker="MSFT", as_of=regular_session_close_utc(day),
        schema_version="forecast-v1", dataset_version=MARKET_DATA_VERSION,
        feature_version=MARKET_FEATURE_VERSION, model_version="provider-smoke-v1",
        cutoff_policy=ForecastCutoffPolicy.MONTH_TO_DATE,
    )
    with tempfile.TemporaryDirectory(prefix="agenticalpha-market-smoke-") as directory:
        provider = YahooMarketDataProvider(cache_dir=Path(directory))
        first = provider(context)
        second = provider(context)
        print(json.dumps({
            "status": "PASS",
            "ticker": first.ticker.ticker,
            "benchmark": first.benchmark.ticker,
            "as_of": context.to_dict()["as_of"],
            "ticker_rows": len(first.ticker.bars),
            "benchmark_rows": len(first.benchmark.bars),
            "last_ticker_session": first.ticker.bars[-1].session.isoformat(),
            "cache_reused": first.cache_hashes == second.cache_hashes,
            "cache_hashes": dict(first.cache_hashes),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
