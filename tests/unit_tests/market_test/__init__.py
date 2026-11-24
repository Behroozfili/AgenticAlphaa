"""Shared deterministic builders for the locked Stage 07 focused tests."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType

from core.forecast_contract import (AnalysisMode, ForecastContext,
    compute_forward_return_labels,
    is_us_regular_trading_session, last_trading_session_of_month,
    nth_trading_session_after, regular_session_close_utc)
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION, build_market_features
from tools.market_tools.historical_dataset_builder import FeatureSnapshot, LabeledSnapshot
from tools.market_tools.market_data import MARKET_DATA_VERSION, build_market_bundle, rows_to_cache


FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "monthly_point_in_time_fixture.json"


def fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def context() -> ForecastContext:
    return ForecastContext(**fixture()["context"])


def sessions(start: date, end: date):
    cursor = start
    result = []
    while cursor <= end:
        if is_us_regular_trading_session(cursor):
            result.append(cursor)
        cursor += timedelta(days=1)
    return result


def market_bundle(*, include_sector: bool = False):
    spec = fixture()["market_spec"]
    ctx = context()
    start = date.fromisoformat(spec["start"])
    end = ctx.target_date
    days = sessions(start, end)

    def bars(base, step, volume_base):
        output = []
        for index, day in enumerate(days):
            close = base + step * index + math.sin(index / 4.0) * 0.2
            output.append({"date": day.isoformat(), "open": close - 0.1,
                "high": close + 0.4, "low": close - 0.4, "close": close,
                "adjusted_close": close, "volume": volume_base + 1000 * (index % 13)})
        return output

    caches = {
        "MSFT": rows_to_cache(ticker="MSFT", start=start, end=end,
            retrieved_at=spec["retrieved_at"], bars=bars(spec["ticker_base"], spec["ticker_daily_step"], spec["volume_base"])),
        "SPY": rows_to_cache(ticker="SPY", start=start, end=end,
            retrieved_at=spec["retrieved_at"], bars=bars(spec["benchmark_base"], spec["benchmark_daily_step"], spec["volume_base"] * 2)),
    }
    if include_sector:
        caches["XLK"] = rows_to_cache(ticker="XLK", start=start, end=end,
            retrieved_at=spec["retrieved_at"], bars=bars(150, 0.22, spec["volume_base"] // 2))
    return build_market_bundle(context=ctx, caches=caches, start=start, end=end,
                               sector_benchmark="XLK" if include_sector else None)


def feature_set():
    research = {"version": "research-features-v2", "values": {
        "catalysts": ["e1"], "risks": ["e2"], "source_diversity": 2,
        "directional_score": 0.1, "claim_confidence": 0.8,
        "classification_coverage": 1.0, "classified_claim_count": 2,
        "vector_coverage": True, "graph_coverage": True},
        "evidence_ids": ["e1", "e2"], "missing": False, "degraded": False}
    financial = {"version": "monthly-financial-features-v1", "values": {
        "valuation_gap": 0.1, "revenue_growth": 0.08, "roe_pct": 25.0,
        "filing_age_days": 40}, "evidence_ids": ["f1"], "missing": False, "degraded": False}
    sentiment = {"version": "monthly-sentiment-features-v1", "values": {
        "aggregate_7d": 0.2, "aggregate_14d": 0.1, "aggregate_30d": 0.05,
        "momentum_7d_vs_30d": 0.15, "dispersion_30d": 0.2,
        "volume_7d": 4, "volume_14d": 7, "volume_30d": 12,
        "source_diversity_30d": 3, "confidence": 0.8},
        "evidence_ids": ["s1"], "missing": False, "degraded": False}
    return build_market_features(context=context(), market=market_bundle(),
        research_features=research, financial_features=financial, sentiment_features=sentiment)


def training_rows(count: int = 40):
    spec = fixture()["training_spec"]
    year, month = map(int, spec["first_month"].split("-"))
    rows = []
    for index in range(count):
        y = year + (month - 1 + index) // 12
        m = (month - 1 + index) % 12 + 1
        cutoff_day = last_trading_session_of_month(y, m)
        cutoff = regular_session_close_utc(cutoff_day)
        target = nth_trading_session_after(cutoff_day)
        optional = None if index % spec["missing_every_nth_optional_value"] == 0 else float(index % 7) / 7
        values = {"f1": float(index) / 10, "f2": math.sin(index / 3),
                  "optional": optional, "optional__missing": 1.0 if optional is None else 0.0,
                  "research_graph_coverage": 1.0}
        payload_key = f"MSFT|{cutoff.isoformat()}|{index}"
        snapshot = FeatureSnapshot(ticker="MSFT", as_of=cutoff.isoformat().replace("+00:00", "Z"),
            target_date=target.isoformat(), horizon_trading_days=21, benchmark="SPY",
            analysis_mode=AnalysisMode.HISTORICAL_BACKTEST.value,
            dataset_version=MARKET_DATA_VERSION, feature_version=MARKET_FEATURE_VERSION,
            values=MappingProxyType(values), evidence_ids=(f"evidence-{index}",),
            missing_indicators=MappingProxyType({"financial_features__missing": False}),
            degraded=False, finalized_at=cutoff.isoformat().replace("+00:00", "Z"),
            snapshot_hash=hashlib.sha256(payload_key.encode()).hexdigest())
        ticker_start = float(spec["ticker_start_adjusted_close"])
        benchmark_start = float(spec["benchmark_start_adjusted_close"])
        ticker_target = ticker_start * (1.0 + spec["ticker_return_intercept"]
                                        + spec["ticker_return_per_index"] * index)
        benchmark_target = benchmark_start * (1.0 + spec["benchmark_return_intercept"]
                                               + spec["benchmark_return_per_index"] * index)
        labels = compute_forward_return_labels(
            ticker_start_adjusted_close=ticker_start,
            ticker_target_adjusted_close=ticker_target,
            benchmark_start_adjusted_close=benchmark_start,
            benchmark_target_adjusted_close=benchmark_target,
        )
        rows.append(LabeledSnapshot(snapshot=snapshot,
            ticker_total_return=float(labels["ticker_total_return"]),
            benchmark_total_return=float(labels["benchmark_total_return"]),
            forward_21d_excess_return=float(labels["forward_21d_excess_return"]),
            direction_positive=1 if labels["excess_return_direction"] == "POSITIVE" else 0,
            outcome_materialized_at=datetime.combine(target + timedelta(days=1), datetime.min.time(), timezone.utc).isoformat().replace("+00:00", "Z")))
    return tuple(rows)
