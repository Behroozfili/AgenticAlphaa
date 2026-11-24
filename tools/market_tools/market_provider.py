"""Yahoo acquisition backed by immutable point-in-time market caches."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from core.error_handler import DataUnavailableAsOfError
from core.forecast_contract import AnalysisMode, ForecastContext
from tools.market_tools.market_data import MarketDataBundle, build_market_bundle, load_adjusted_ohlcv_cache, rows_to_cache


class MarketHistoryFetcher(Protocol):
    def __call__(
        self, symbol: str, start: date, end: date
    ) -> tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]: ...


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _component(value: str, *, field_name: str) -> str:
    candidate = str(value).strip()
    if not _SAFE_COMPONENT.fullmatch(candidate):
        raise ValueError(f"{field_name} is not safe for an immutable cache path")
    return candidate


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("provider clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _number(value: Any, *, field_name: str, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataUnavailableAsOfError(f"Yahoo returned invalid {field_name}") from exc
    if not math.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        raise DataUnavailableAsOfError(f"Yahoo returned invalid {field_name}")
    return result


def _default_yahoo_fetch(
    symbol: str, start: date, end: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch adjusted OHLCV; Yahoo's exclusive end is extended by one day."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required for live market acquisition") from exc
    frame = yf.Ticker(symbol).history(
        start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True, actions=True,
    )
    if frame is None or frame.empty:
        raise DataUnavailableAsOfError(f"Yahoo returned no market history for {symbol}")
    bars: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for timestamp, row in frame.iterrows():
        session = timestamp.date()
        if session < start or session > end:
            continue
        open_ = _number(row.get("Open"), field_name="open")
        close = _number(row.get("Close"), field_name="close")
        # Adjusted Yahoo floats can differ by a few ULPs (for example High is
        # 3e-14 below Close). Canonicalize the OHLC envelope at acquisition.
        high = max(_number(row.get("High"), field_name="high"), open_, close)
        low = min(_number(row.get("Low"), field_name="low"), open_, close)
        bars.append({
            "date": session.isoformat(), "open": open_,
            "high": high, "low": low, "close": close,
            "adjusted_close": close,
            "volume": _number(row.get("Volume", 0), field_name="volume", allow_zero=True),
        })
        dividend = float(row.get("Dividends", 0) or 0)
        split = float(row.get("Stock Splits", 0) or 0)
        if math.isfinite(dividend) and dividend > 0:
            actions.append({"effective_date": session.isoformat(), "type": "dividend", "value": dividend})
        if math.isfinite(split) and split > 0:
            actions.append({"effective_date": session.isoformat(), "type": "split", "value": split})
    if not bars:
        raise DataUnavailableAsOfError(f"Yahoo returned no in-range market history for {symbol}")
    return bars, actions


def _bounded_rows(
    bars: Sequence[Mapping[str, Any]], actions: Sequence[Mapping[str, Any]], *, start: date, end: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop provider over-fetch before any bytes reach the persistent cache."""
    bounded_bars = [dict(row) for row in bars if start <= date.fromisoformat(str(row.get("date"))) <= end]
    bounded_actions = [dict(row) for row in actions
                       if start <= date.fromisoformat(str(row.get("effective_date"))) <= end]
    bounded_bars.sort(key=lambda row: str(row["date"]))
    bounded_actions.sort(key=lambda row: (str(row["effective_date"]), str(row.get("type", ""))))
    return bounded_bars, bounded_actions


@dataclass(slots=True)
class YahooMarketDataProvider:
    """LIVE populates caches; HISTORICAL_BACKTEST is strictly cache-only."""

    cache_dir: Path | str
    sector_benchmark: str | None = None
    fetcher: MarketHistoryFetcher = field(default=_default_yahoo_fetch)
    clock: Callable[[], datetime] = field(default=_utc_now)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir).resolve()
        if self.sector_benchmark is not None:
            self.sector_benchmark = _component(
                self.sector_benchmark.strip().upper(), field_name="sector_benchmark"
            )

    def _path(self, *, context: ForecastContext, symbol: str, start: date, end: date) -> Path:
        dataset = _component(context.dataset_version, field_name="dataset_version")
        canonical_symbol = _component(symbol.strip().upper(), field_name="ticker")
        return self.cache_dir / dataset / canonical_symbol / f"{start.isoformat()}_{end.isoformat()}.json"

    @staticmethod
    def _read(path: Path) -> Mapping[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataUnavailableAsOfError(f"market cache cannot be read: {path.name}") from exc
        if not isinstance(payload, Mapping):
            raise DataUnavailableAsOfError("market cache root must be an object")
        return payload

    @staticmethod
    def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def _payload(self, *, context: ForecastContext, symbol: str, start: date, end: date) -> Mapping[str, Any]:
        path = self._path(context=context, symbol=symbol, start=start, end=end)
        if path.exists():
            payload = self._read(path)
            load_adjusted_ohlcv_cache(
                payload, ticker=symbol, start=start, end=end, context=context,
            )
            return payload
        if context.analysis_mode is AnalysisMode.HISTORICAL_BACKTEST:
            raise DataUnavailableAsOfError(
                f"historical market cache is missing for {symbol}; network fallback is forbidden"
            )
        bars, actions = self.fetcher(symbol, start, end)
        bounded_bars, bounded_actions = _bounded_rows(bars, actions, start=start, end=end)
        if not bounded_bars or str(bounded_bars[-1].get("date")) != end.isoformat():
            raise DataUnavailableAsOfError(f"market data for {symbol} is not available through as_of {end}")
        payload = rows_to_cache(
            ticker=symbol, start=start, end=end, retrieved_at=_as_utc_iso(self.clock()),
            bars=bounded_bars, corporate_actions=bounded_actions,
        )
        self._write_once(path, payload)
        persisted = self._read(path)
        load_adjusted_ohlcv_cache(
            persisted, ticker=symbol, start=start, end=end, context=context,
        )
        return persisted

    def __call__(self, context: ForecastContext) -> MarketDataBundle:
        # target_date is the future label date, never an input-data cutoff.
        end = context.as_of.date()
        start = end - timedelta(days=context.lookback_days)
        symbols = list(dict.fromkeys(filter(None, (
            context.ticker, context.benchmark, self.sector_benchmark,
        ))))
        caches = {symbol: self._payload(context=context, symbol=symbol, start=start, end=end)
                  for symbol in symbols}
        return build_market_bundle(
            context=context, caches=caches, start=start, end=end,
            sector_benchmark=self.sector_benchmark,
        )


__all__ = ["MarketHistoryFetcher", "YahooMarketDataProvider"]
