"""Train a preliminary immutable Quant artifact from real Yahoo history.

This job intentionally uses market-only point-in-time features.  It never
substitutes the synthetic regression fixtures for a runtime model.
"""

from __future__ import annotations

import argparse
import calendar
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from core.forecast_contract import (
    AnalysisMode,
    ForecastContext,
    ForecastCutoffPolicy,
    is_us_regular_trading_session,
    last_trading_session_of_month,
    nth_trading_session_after,
    regular_session_close_utc,
)
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION, build_market_features
from tools.market_tools.forecast_model import fit_deterministic_model
from tools.market_tools.historical_dataset_builder import (
    MONTH_END_CUTOFF_POLICY_VERSION,
    MONTH_TO_DATE_CUTOFF_POLICY_VERSION,
    build_data_readiness,
    build_feature_snapshot,
    materialize_outcome,
)
from tools.market_tools.market_data import MARKET_DATA_VERSION, build_market_bundle, rows_to_cache
from tools.market_tools.market_provider import _default_yahoo_fetch
from tools.market_tools.quant_artifacts import build_quant_artifact_bundle, save_quant_artifact_bundle


def _monthly_cutoffs(
    start: date, end: date, *, cutoff_policy: ForecastCutoffPolicy,
    month_day: int,
):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        if cutoff_policy is ForecastCutoffPolicy.MONTH_END:
            cutoff = last_trading_session_of_month(year, month)
        else:
            cutoff = date(
                year, month, min(month_day, calendar.monthrange(year, month)[1])
            )
            while not is_us_regular_trading_session(cutoff):
                cutoff -= timedelta(days=1)
        yield cutoff
        month += 1
        if month == 13:
            year, month = year + 1, 1


def train(
    *, ticker: str, benchmark: str, start: date, output: Path,
    cutoff_policy: ForecastCutoffPolicy = ForecastCutoffPolicy.MONTH_END,
    month_day: int = 18,
) -> Path:
    ticker, benchmark = ticker.upper(), benchmark.upper()
    retrieved = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    fetch_start = start - timedelta(days=370)
    fetch_end = datetime.now(UTC).date()
    histories = {
        symbol: _default_yahoo_fetch(symbol, fetch_start, fetch_end)
        for symbol in (ticker, benchmark)
    }
    available = set.intersection(*(
        {date.fromisoformat(str(row["date"])) for row in histories[symbol][0]}
        for symbol in (ticker, benchmark)
    ))
    rows = []
    if not 1 <= month_day <= 28:
        raise ValueError("month_day must be between 1 and 28")
    policy_slug = (
        "mtd" if cutoff_policy is ForecastCutoffPolicy.MONTH_TO_DATE else "month-end"
    )
    run_id = f"train-{ticker.lower()}-market-only-{policy_slug}-v1"
    for cutoff in _monthly_cutoffs(
        start, fetch_end, cutoff_policy=cutoff_policy, month_day=month_day,
    ):
        target = nth_trading_session_after(cutoff)
        if cutoff not in available or target not in available:
            continue
        context = ForecastContext(
            ticker=ticker,
            as_of=regular_session_close_utc(cutoff),
            target=target,
            lookback_days=365,
            benchmark=benchmark,
            analysis_mode=AnalysisMode.HISTORICAL_BACKTEST,
            cutoff_policy=cutoff_policy,
            schema_version="forecast-v1",
            dataset_version=MARKET_DATA_VERSION,
            feature_version=MARKET_FEATURE_VERSION,
            model_version="training-pending",
        )
        range_start = cutoff - timedelta(days=context.lookback_days)
        caches = {}
        for symbol in (ticker, benchmark):
            bars, actions = histories[symbol]
            caches[symbol] = rows_to_cache(
                ticker=symbol,
                start=range_start,
                end=target,
                retrieved_at=retrieved,
                bars=(row for row in bars if range_start <= date.fromisoformat(str(row["date"])) <= target),
                corporate_actions=(
                    item for item in actions
                    if range_start <= date.fromisoformat(str(item["effective_date"])) <= target
                ),
            )
        market = build_market_bundle(
            context=context, caches=caches, start=range_start, end=target, run_id=run_id,
        )
        features = build_market_features(
            context=context, market=market, materialized_at=context.as_of, run_id=run_id,
        )
        snapshot = build_feature_snapshot(
            context=context, features=features, finalized_at=context.as_of, run_id=run_id,
        )
        rows.append(materialize_outcome(
            snapshot=snapshot,
            market=market,
            outcome_materialized_at=regular_session_close_utc(target),
            run_id=run_id,
        ))

    universe = {
        "schema_version": "1.0",
        "universe_version": f"{ticker.lower()}-yahoo-market-only-{policy_slug}-v1",
        "source": "Yahoo Finance adjusted OHLCV",
        "created_at": retrieved,
        "constituents": [{
            "ticker": ticker,
            "benchmark": benchmark,
            "effective_from": start.isoformat(),
            "effective_to": None,
            "source_ref": f"yahoo:{ticker}:{benchmark}",
        }],
        "limitations": [
            "PRELIMINARY single-ticker model",
            "Training features are market-only; historical specialist features are unavailable",
        ],
    }
    readiness = build_data_readiness(
        rows=rows, universe_manifest=universe, run_id=run_id,
        cutoff_policy_version=(
            MONTH_TO_DATE_CUTOFF_POLICY_VERSION
            if cutoff_policy is ForecastCutoffPolicy.MONTH_TO_DATE
            else MONTH_END_CUTOFF_POLICY_VERSION
        ),
    )
    model = fit_deterministic_model(rows=rows, readiness=readiness, run_id=run_id)
    payload = build_quant_artifact_bundle(
        model=model,
        readiness=readiness,
        universe_manifest=universe,
        model_variant=f"preliminary_market_only_{policy_slug}_v1",
    )
    save_quant_artifact_bundle(output, payload)
    print(f"artifact={output} rows={model.training_rows} cutoffs={model.training_cutoffs}")
    print(f"model_version={model.model_version}")
    print(f"readiness={readiness.status} report_level={readiness.manifest['scientific_report_level']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--output", type=Path, default=Path("artifacts/quant/msft_market_only_v1.json"))
    parser.add_argument(
        "--cutoff-policy", choices=[item.value for item in ForecastCutoffPolicy],
        default=ForecastCutoffPolicy.MONTH_END.value,
    )
    parser.add_argument("--month-day", type=int, default=18)
    args = parser.parse_args()
    train(
        ticker=args.ticker, benchmark=args.benchmark, start=args.start, output=args.output,
        cutoff_policy=ForecastCutoffPolicy(args.cutoff_policy), month_day=args.month_day,
    )


if __name__ == "__main__":
    main()
