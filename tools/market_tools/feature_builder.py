"""Trading-session-aware point-in-time market and cross-domain features."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from core.error_handler import DataUnavailableAsOfError
from core.forecast_contract import CALENDAR_VERSION, ForecastContext, parse_utc_datetime
from tools.market_tools import (quant_lifecycle_event, validate_evidence_references,
                                validate_run_id)
from tools.market_tools.market_data import MarketDataBundle, PriceBar


MARKET_FEATURE_VERSION = "monthly-market-features-v1"
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,159}$")
_MAX_COMBINED_EVIDENCE = 64


def _merge_evidence(*groups: Sequence[str]) -> tuple[str, ...]:
    """Fair deterministic merge so one specialist cannot consume the cap."""
    ordered = [tuple(sorted(set(group))) for group in groups]
    merged: list[str] = []
    seen: set[str] = set()
    for index in range(max((len(group) for group in ordered), default=0)):
        for group in ordered:
            if index >= len(group) or group[index] in seen:
                continue
            seen.add(group[index])
            merged.append(group[index])
            if len(merged) == _MAX_COMBINED_EVIDENCE:
                return tuple(merged)
    return tuple(merged)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _slot(value: Any) -> tuple[Mapping[str, Any], bool, bool, tuple[str, ...]]:
    if value is None:
        return {}, True, True, ()
    if hasattr(value, "to_state_slot"):
        value = value.to_state_slot()
    if not isinstance(value, Mapping):
        raise TypeError("typed upstream features must be a dataclass or state-slot mapping")
    values = value.get("values", value)
    if not isinstance(values, Mapping):
        raise TypeError("feature slot values must be a mapping")
    evidence = value.get("evidence_ids", ())
    evidence_ids = validate_evidence_references(evidence)
    return values, bool(value.get("missing", False)), bool(value.get("degraded", False)), evidence_ids


def _momentum(bars: Sequence[PriceBar], sessions: int) -> float | None:
    if len(bars) <= sessions:
        return None
    return bars[-1].adjusted_close / bars[-1 - sessions].adjusted_close - 1.0


def _log_returns(bars: Sequence[PriceBar], sessions: int) -> np.ndarray | None:
    if len(bars) <= sessions:
        return None
    prices = np.asarray([bar.adjusted_close for bar in bars[-(sessions + 1):]], dtype=float)
    return np.diff(np.log(prices))


def _annualized_volatility(bars: Sequence[PriceBar], sessions: int) -> float | None:
    returns = _log_returns(bars, sessions)
    return float(np.std(returns, ddof=1) * math.sqrt(252.0)) if returns is not None and len(returns) > 1 else None


def _beta(ticker: Sequence[PriceBar], benchmark: Sequence[PriceBar], sessions: int = 63) -> float | None:
    ticker_by_day = {bar.session: bar.adjusted_close for bar in ticker}
    benchmark_by_day = {bar.session: bar.adjusted_close for bar in benchmark}
    common = sorted(set(ticker_by_day) & set(benchmark_by_day))
    if len(common) <= sessions:
        return None
    common = common[-(sessions + 1):]
    asset = np.diff(np.log([ticker_by_day[day] for day in common]))
    market = np.diff(np.log([benchmark_by_day[day] for day in common]))
    variance = float(np.var(market, ddof=1))
    if variance <= 0:
        return None
    return float(np.cov(asset, market, ddof=1)[0, 1] / variance)


def _put(values: dict[str, float | None], missing: dict[str, bool], name: str, raw: Any) -> None:
    value = _finite(raw)
    values[name] = value
    missing[f"{name}__missing"] = value is None


@dataclass(frozen=True, slots=True)
class MarketFeatures:
    ticker: str
    as_of: str
    values: Mapping[str, float | None]
    missing_indicators: Mapping[str, bool]
    evidence_ids: tuple[str, ...]
    missing: bool
    degraded: bool
    dataset_version: str
    feature_version: str = MARKET_FEATURE_VERSION
    materialized_at: str = ""
    schema_version: str = "forecast-v1"
    calendar_version: str = CALENDAR_VERSION
    benchmark: str = "SPY"
    analysis_mode: str = "LIVE"
    cutoff_policy: str = "MONTH_END"

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", validate_evidence_references(self.evidence_ids))
        for field in (
            "ticker", "dataset_version", "feature_version", "schema_version",
            "calendar_version", "benchmark", "analysis_mode", "cutoff_policy",
        ):
            value = getattr(self, field)
            if (not isinstance(value, str) or value != value.strip()
                    or not _IDENTITY.fullmatch(value)
                    or "PRIVATE_SENTINEL" in value.upper()):
                raise ValueError(f"{field} must be compact canonical identity")
        parsed_as_of = parse_utc_datetime(self.as_of, field_name="as_of")
        parsed_materialized = parse_utc_datetime(self.materialized_at, field_name="materialized_at")
        object.__setattr__(self, "as_of", parsed_as_of.isoformat().replace("+00:00", "Z"))
        object.__setattr__(self, "materialized_at", parsed_materialized.isoformat().replace("+00:00", "Z"))

    def to_state_slot(self) -> dict[str, Any]:
        evidence_ids = validate_evidence_references(self.evidence_ids)
        payload = dict(self.values)
        payload.update(self.missing_indicators)
        payload.update({
            "ticker": self.ticker, "as_of": self.as_of,
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "calendar_version": self.calendar_version,
            "feature_materialized_at": self.materialized_at,
            "cutoff_policy": self.cutoff_policy,
        })
        return {
            "version": self.feature_version, "values": payload,
            "evidence_ids": list(evidence_ids), "missing": self.missing,
            "degraded": self.degraded, "errors": [],
        }

    def numeric_vector(self) -> dict[str, float | None]:
        vector = dict(self.values)
        vector.update({name: 1.0 if flag else 0.0 for name, flag in self.missing_indicators.items()})
        return vector


def _build_market_features(
    *,
    context: ForecastContext,
    market: MarketDataBundle,
    research_features: Any = None,
    financial_features: Any = None,
    sentiment_features: Any = None,
    materialized_at: str,
) -> MarketFeatures:
    """Build a fixed vector using only sessions and typed features available at as_of."""
    cutoff = context.as_of.date()
    ticker = market.ticker.through(cutoff)
    benchmark = market.benchmark.through(cutoff)
    if not ticker or ticker[-1].session != cutoff or not benchmark or benchmark[-1].session != cutoff:
        raise DataUnavailableAsOfError("ticker and benchmark must both close exactly at as_of")

    values: dict[str, float | None] = {}
    missing: dict[str, bool] = {}
    for window in (5, 21, 63):
        _put(values, missing, f"momentum_{window}", _momentum(ticker, window))
    for window in (21, 63):
        _put(values, missing, f"volatility_{window}", _annualized_volatility(ticker, window))
    trailing_volume = [bar.volume for bar in ticker[-22:-1]]
    volume_surprise = ticker[-1].volume / float(np.mean(trailing_volume)) - 1.0 if len(trailing_volume) == 21 and np.mean(trailing_volume) > 0 else None
    _put(values, missing, "volume_surprise_21", volume_surprise)
    trailing_prices = [bar.adjusted_close for bar in ticker[-63:]]
    _put(values, missing, "drawdown_63", ticker[-1].adjusted_close / max(trailing_prices) - 1.0 if len(trailing_prices) == 63 else None)
    _put(values, missing, "beta_63", _beta(ticker, benchmark))
    for window in (21, 63):
        asset_momentum = _momentum(ticker, window)
        benchmark_momentum = _momentum(benchmark, window)
        _put(values, missing, f"relative_strength_{window}", asset_momentum - benchmark_momentum if asset_momentum is not None and benchmark_momentum is not None else None)
    if market.sector is not None:
        sector = market.sector.through(cutoff)
        sector_momentum = _momentum(sector, 21)
        asset_momentum = _momentum(ticker, 21)
        _put(values, missing, "sector_relative_strength_21", asset_momentum - sector_momentum if asset_momentum is not None and sector_momentum is not None else None)
    else:
        _put(values, missing, "sector_relative_strength_21", None)

    research, research_missing, research_degraded, research_evidence = _slot(research_features)
    financial, financial_missing, financial_degraded, financial_evidence = _slot(financial_features)
    sentiment, sentiment_missing, sentiment_degraded, sentiment_evidence = _slot(sentiment_features)
    research_fields = {
        "research_source_diversity": research.get("source_diversity"),
        "research_directional_score": research.get("directional_score"),
        "research_claim_confidence": research.get("claim_confidence"),
        "research_classification_coverage": research.get("classification_coverage"),
        "research_classified_claim_count": research.get("classified_claim_count"),
        "research_vector_coverage": 1.0 if research.get("vector_coverage") is True else (0.0 if research.get("vector_coverage") is False else None),
        "research_graph_coverage": 1.0 if research.get("graph_coverage") is True else (0.0 if research.get("graph_coverage") is False else None),
        "research_catalyst_count": len(research.get("catalysts", [])) if isinstance(research.get("catalysts"), list) else None,
        "research_risk_count": len(research.get("risks", [])) if isinstance(research.get("risks"), list) else None,
    }
    financial_names = (
        "valuation_gap", "revenue_growth", "earnings_growth", "roe_pct", "net_margin_pct",
        "roic_pct", "net_debt_to_ebitda", "debt_ratio", "current_ratio", "quick_ratio",
        "fcf_margin_pct", "fcf_yield_pct", "ocf_ratio", "filing_age_days",
    )
    sentiment_names = (
        "aggregate_7d", "aggregate_14d", "aggregate_30d", "momentum_7d_vs_30d",
        "dispersion_30d", "volume_7d", "volume_14d", "volume_30d",
        "source_diversity_30d", "confidence",
    )
    for name, raw in research_fields.items():
        _put(values, missing, name, raw)
    for name in financial_names:
        _put(values, missing, f"financial_{name}", financial.get(name))
    for name in sentiment_names:
        _put(values, missing, f"sentiment_{name}", sentiment.get(name))

    missing.update({
        "research_features__missing": research_missing,
        "financial_features__missing": financial_missing,
        "sentiment_features__missing": sentiment_missing,
    })
    evidence = _merge_evidence(research_evidence, financial_evidence, sentiment_evidence)
    required_missing = any(missing[name] for name in (
        "momentum_21__missing", "volatility_21__missing", "beta_63__missing",
        "relative_strength_21__missing",
    ))
    return MarketFeatures(
        ticker=context.ticker, as_of=context.as_of.isoformat().replace("+00:00", "Z"),
        values=MappingProxyType(dict(sorted(values.items()))),
        missing_indicators=MappingProxyType(dict(sorted(missing.items()))),
        evidence_ids=evidence, missing=required_missing,
        degraded=required_missing or research_degraded or financial_degraded or sentiment_degraded,
        dataset_version=context.dataset_version,
        materialized_at=materialized_at,
        schema_version=context.schema_version,
        calendar_version=CALENDAR_VERSION,
        benchmark=context.benchmark,
        analysis_mode=context.analysis_mode.value,
        cutoff_policy=context.cutoff_policy.value,
    )


def build_market_features(
    *, context: ForecastContext, market: MarketDataBundle,
    research_features: Any = None, financial_features: Any = None,
    sentiment_features: Any = None, run_id: str | None = None,
    materialized_at: datetime | str | None = None,
) -> MarketFeatures:
    started_at = time.perf_counter()
    identity = validate_run_id(run_id) if run_id is not None else None
    if materialized_at is None:
        materialized = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    elif isinstance(materialized_at, datetime):
        if materialized_at.tzinfo is None:
            raise ValueError("materialized_at must be timezone-aware")
        materialized = materialized_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    else:
        materialized = parse_utc_datetime(
            materialized_at, field_name="materialized_at"
        ).isoformat().replace("+00:00", "Z")
    try:
        result = _build_market_features(context=context, market=market,
            research_features=research_features, financial_features=financial_features,
            sentiment_features=sentiment_features, materialized_at=materialized)
    except Exception:
        quant_lifecycle_event(context=context, boundary="feature_build", started_at=started_at,
            outcome="fail_closed", run_id=identity, error_code="QUANT_FEATURE_REJECTED")
        raise
    flags = ["optional_feature_degradation"] if result.degraded else []
    quant_lifecycle_event(context=context, boundary="feature_build", started_at=started_at,
        outcome="degraded" if result.degraded else "success", run_id=identity,
        evidence_count=len(result.evidence_ids), counts={"feature_count": len(result.values),
        "missing_count": sum(result.missing_indicators.values())}, degraded_flags=flags)
    return result


__all__ = ["MARKET_FEATURE_VERSION", "MarketFeatures", "build_market_features"]
