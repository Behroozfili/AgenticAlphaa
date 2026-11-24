"""Deterministic point-in-time market data loaded from immutable local caches."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError
from core.forecast_contract import (
    ForecastContext,
    is_us_regular_trading_session,
    parse_utc_datetime,
)
from tools.market_tools import quant_lifecycle_event, validate_run_id


MARKET_DATA_VERSION = "adjusted-ohlcv-cache-v1"
AUTHORIZED_PROVIDER = "yahoo_finance"
ADJUSTMENT_MODE = "adjusted"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _day(value: Any, *, field: str) -> date:
    try:
        parsed = date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 date") from exc
    return parsed


def _finite_positive(value: Any, *, field: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    minimum_ok = number >= 0 if allow_zero else number > 0
    if not math.isfinite(number) or not minimum_ok:
        raise ValueError(f"{field} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return number


@dataclass(frozen=True, slots=True)
class PriceBar:
    session: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: float


@dataclass(frozen=True, slots=True)
class CorporateAction:
    effective_date: date
    action_type: str
    value: float


@dataclass(frozen=True, slots=True)
class MarketSeries:
    ticker: str
    start: date
    end: date
    provider: str
    retrieved_at: datetime
    adjustment_mode: str
    bars: tuple[PriceBar, ...]
    corporate_actions: tuple[CorporateAction, ...]
    content_hash: str

    def adjusted_close_on(self, session: date) -> float:
        for bar in self.bars:
            if bar.session == session:
                return bar.adjusted_close
        raise DataUnavailableAsOfError(f"missing adjusted close for {self.ticker} on {session}")

    def through(self, session: date) -> tuple[PriceBar, ...]:
        return tuple(bar for bar in self.bars if bar.session <= session)


@dataclass(frozen=True, slots=True)
class MarketDataBundle:
    ticker: MarketSeries
    benchmark: MarketSeries
    sector: MarketSeries | None
    dataset_version: str
    cache_hashes: Mapping[str, str]


def _load_adjusted_ohlcv_cache(
    payload: Mapping[str, Any],
    *,
    ticker: str,
    start: date,
    end: date,
) -> MarketSeries:
    """Validate one exact-range immutable Yahoo cache; never performs acquisition."""
    symbol = str(payload.get("ticker") or "").strip().upper()
    if symbol != ticker.strip().upper():
        raise ValueError("cache ticker does not match requested ticker")
    if str(payload.get("provider")) != AUTHORIZED_PROVIDER:
        raise ValueError("historical cache provider must be yahoo_finance")
    if str(payload.get("adjustment_mode")) != ADJUSTMENT_MODE:
        raise ValueError("market cache must use adjusted history consistently")
    if str(payload.get("dataset_version")) != MARKET_DATA_VERSION:
        raise ValueError("market cache dataset version mismatch")
    cached_start = _day(payload.get("start"), field="start")
    cached_end = _day(payload.get("end"), field="end")
    if cached_start != start or cached_end != end:
        raise DataUnavailableAsOfError("cache does not satisfy the exact requested start/end range")
    retrieved_at = parse_utc_datetime(payload.get("retrieved_at"), field_name="retrieved_at")

    raw_rows = payload.get("bars")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise DataUnavailableAsOfError("cache contains no OHLCV rows")
    bars: list[PriceBar] = []
    previous: date | None = None
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise TypeError("cache bars must be objects")
        session = _day(raw.get("date"), field=f"bars[{index}].date")
        if session < start or session > end:
            raise TemporalLeakageError("cache bar falls outside the exact requested range")
        if not is_us_regular_trading_session(session):
            raise ValueError(f"{session} is not a regular US trading session")
        if previous is not None and session <= previous:
            raise ValueError("cache bars must be unique and strictly chronological")
        previous = session
        open_ = _finite_positive(raw.get("open"), field="open")
        high = _finite_positive(raw.get("high"), field="high")
        low = _finite_positive(raw.get("low"), field="low")
        close = _finite_positive(raw.get("close"), field="close")
        adjusted = _finite_positive(raw.get("adjusted_close"), field="adjusted_close")
        volume = _finite_positive(raw.get("volume"), field="volume", allow_zero=True)
        if low > min(open_, close) or high < max(open_, close) or high < low:
            raise ValueError("invalid OHLC price ordering")
        bars.append(PriceBar(session, open_, high, low, close, adjusted, volume))

    raw_actions = payload.get("corporate_actions", [])
    if not isinstance(raw_actions, list):
        raise TypeError("corporate_actions must be a list")
    actions: list[CorporateAction] = []
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise TypeError("corporate actions must be objects")
        effective = _day(raw.get("effective_date"), field="effective_date")
        if effective > end:
            raise TemporalLeakageError("future corporate action is outside requested range")
        action_type = str(raw.get("type") or "").strip().lower()
        if action_type not in {"split", "dividend"}:
            raise ValueError("corporate action type must be split or dividend")
        actions.append(CorporateAction(effective, action_type, _finite_positive(raw.get("value"), field="action value")))

    hash_payload = dict(payload)
    claimed_hash = str(hash_payload.pop("content_hash", "") or "")
    computed_hash = canonical_hash(hash_payload)
    if claimed_hash and claimed_hash != computed_hash:
        raise ValueError("market cache content hash mismatch")
    return MarketSeries(
        ticker=symbol, start=start, end=end, provider=AUTHORIZED_PROVIDER,
        retrieved_at=retrieved_at, adjustment_mode=ADJUSTMENT_MODE,
        bars=tuple(bars), corporate_actions=tuple(actions), content_hash=computed_hash,
    )


def load_adjusted_ohlcv_cache(
    payload: Mapping[str, Any], *, ticker: str, start: date, end: date,
    context: ForecastContext | None = None, run_id: str | None = None,
) -> MarketSeries:
    started_at = time.perf_counter()
    identity = validate_run_id(run_id) if run_id is not None else None
    metadata_context: Mapping[str, Any] | object = context or {
        "ticker": ticker.strip().upper(), "dataset_version": MARKET_DATA_VERSION,
    }
    try:
        result = _load_adjusted_ohlcv_cache(payload, ticker=ticker, start=start, end=end)
    except Exception:
        quant_lifecycle_event(context=metadata_context, boundary="cache_validation",
            started_at=started_at, outcome="fail_closed", run_id=identity,
            error_code="QUANT_CACHE_REJECTED")
        raise
    quant_lifecycle_event(context=metadata_context, boundary="cache_validation",
        started_at=started_at, outcome="success", run_id=identity,
        counts={"result_count": len(result.bars), "action_count": len(result.corporate_actions)})
    return result


def _build_market_bundle(
    *,
    context: ForecastContext,
    caches: Mapping[str, Mapping[str, Any]],
    start: date,
    end: date,
    sector_benchmark: str | None = None,
    run_id: str | None = None,
) -> MarketDataBundle:
    """Load ticker, SPY/context benchmark and optional sector from exact local caches."""
    symbols = [context.ticker, context.benchmark]
    sector = str(sector_benchmark or "").strip().upper() or None
    if sector and sector not in symbols:
        symbols.append(sector)
    loaded: dict[str, MarketSeries] = {}
    for symbol in symbols:
        raw = caches.get(symbol)
        if not isinstance(raw, Mapping):
            raise DataUnavailableAsOfError(f"authorized immutable cache is missing for {symbol}")
        loaded[symbol] = load_adjusted_ohlcv_cache(raw, ticker=symbol, start=start, end=end,
                                                   context=context, run_id=run_id)
    hashes = MappingProxyType({symbol: loaded[symbol].content_hash for symbol in sorted(loaded)})
    return MarketDataBundle(
        ticker=loaded[context.ticker], benchmark=loaded[context.benchmark],
        sector=loaded.get(sector) if sector else None,
        dataset_version=context.dataset_version, cache_hashes=hashes,
    )


def build_market_bundle(
    *, context: ForecastContext, caches: Mapping[str, Mapping[str, Any]],
    start: date, end: date, sector_benchmark: str | None = None,
    run_id: str | None = None,
) -> MarketDataBundle:
    started_at = time.perf_counter()
    identity = validate_run_id(run_id) if run_id is not None else None
    try:
        result = _build_market_bundle(context=context, caches=caches, start=start, end=end,
            sector_benchmark=sector_benchmark, run_id=identity)
    except Exception:
        quant_lifecycle_event(context=context, boundary="market_bundle", started_at=started_at,
            outcome="fail_closed", run_id=identity, error_code="QUANT_BUNDLE_REJECTED")
        raise
    quant_lifecycle_event(context=context, boundary="market_bundle", started_at=started_at,
        outcome="success", run_id=identity, counts={"cache_count": len(result.cache_hashes)})
    return result


def rows_to_cache(
    *, ticker: str, start: date, end: date, retrieved_at: str,
    bars: Iterable[Mapping[str, Any]], corporate_actions: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a hash-complete cache payload for deterministic fixtures/importers."""
    payload: dict[str, Any] = {
        "ticker": ticker.strip().upper(), "provider": AUTHORIZED_PROVIDER,
        "dataset_version": MARKET_DATA_VERSION, "adjustment_mode": ADJUSTMENT_MODE,
        "start": start.isoformat(), "end": end.isoformat(), "retrieved_at": retrieved_at,
        "bars": [dict(row) for row in bars],
        "corporate_actions": [dict(item) for item in corporate_actions],
    }
    payload["content_hash"] = canonical_hash(payload)
    return payload


__all__ = [
    "ADJUSTMENT_MODE", "AUTHORIZED_PROVIDER", "MARKET_DATA_VERSION",
    "CorporateAction", "MarketDataBundle", "MarketSeries", "PriceBar",
    "build_market_bundle", "canonical_hash", "load_adjusted_ohlcv_cache", "rows_to_cache",
]
