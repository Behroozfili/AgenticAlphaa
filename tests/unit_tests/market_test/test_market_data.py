from copy import deepcopy
from datetime import date

import pytest

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError
from tests.unit_tests.market_test import context, fixture, market_bundle
from tools.market_tools.market_data import build_market_bundle, load_adjusted_ohlcv_cache


def test_local_cache_bundle_is_exact_adjusted_and_versioned():
    bundle = market_bundle(include_sector=True)
    assert bundle.ticker.ticker == "MSFT"
    assert bundle.benchmark.ticker == "SPY"
    assert bundle.sector.ticker == "XLK"
    assert bundle.ticker.adjustment_mode == "adjusted"
    assert set(bundle.cache_hashes) == {"MSFT", "SPY", "XLK"}


def test_cache_hash_tampering_is_rejected():
    bundle = market_bundle()
    raw = {"ticker": bundle.ticker.ticker, "provider": bundle.ticker.provider,
           "dataset_version": "adjusted-ohlcv-cache-v1", "adjustment_mode": "adjusted",
           "start": bundle.ticker.start.isoformat(), "end": bundle.ticker.end.isoformat(),
           "retrieved_at": bundle.ticker.retrieved_at.isoformat(), "bars": [],
           "corporate_actions": [], "content_hash": "bad"}
    with pytest.raises(DataUnavailableAsOfError):
        load_adjusted_ohlcv_cache(raw, ticker="MSFT", start=bundle.ticker.start, end=bundle.ticker.end)


def test_exact_requested_range_is_enforced():
    bundle = market_bundle()
    with pytest.raises(DataUnavailableAsOfError, match="exact requested"):
        load_adjusted_ohlcv_cache({
            "ticker": "MSFT", "provider": "yahoo_finance", "dataset_version": "adjusted-ohlcv-cache-v1",
            "adjustment_mode": "adjusted", "start": "2024-02-02", "end": bundle.ticker.end.isoformat(),
            "retrieved_at": "2024-08-01T00:00:00Z", "bars": [{"date": "2024-02-02", "open": 1,
            "high": 1, "low": 1, "close": 1, "adjusted_close": 1, "volume": 1}], "corporate_actions": []},
            ticker="MSFT", start=date(2024, 2, 1), end=bundle.ticker.end)


def test_missing_required_cache_fails_closed():
    with pytest.raises(DataUnavailableAsOfError, match="immutable cache"):
        build_market_bundle(context=context(), caches={}, start=date(2024, 2, 1), end=context().target_date)


def test_no_network_acquisition_api_is_exposed():
    import tools.market_tools.market_data as module
    assert not hasattr(module, "download")
    assert not hasattr(module, "fetch")
