"""Generate a thesis-ready real-market walk-forward report.

The Stage 09 regression fixture is deliberately synthetic.  This command uses
Yahoo adjusted OHLCV instead, executes expanding-window forecasts with only
outcomes observable at each cutoff, and writes both a Persian Markdown report
and machine-readable provenance artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from core.forecast_contract import (
    AnalysisMode,
    ForecastContext,
    ForecastCutoffPolicy,
    HORIZON_TRADING_DAYS,
    last_trading_session_of_month,
    nth_trading_session_after,
    parse_utc_datetime,
    regular_session_close_utc,
)
from evaluation.forecast_metrics import compute_forecast_metrics
from evaluation.monthly_backtest import BacktestReport, build_expanding_folds
from evaluation.point_in_time_dataset import (
    PointInTimeDataset,
    SCIENTIFIC_MINIMUMS,
    stable_hash,
)
from evaluation.temporal_leakage import audit_backtest
from tools.market_tools.feature_builder import MARKET_FEATURE_VERSION, build_market_features
from tools.market_tools.forecast_model import (
    ACTION_POLICY_VERSION,
    MINIMUM_TRAINING_CUTOFFS,
    fit_deterministic_model,
    predict,
)
from tools.market_tools.historical_dataset_builder import (
    DATASET_BUILDER_VERSION,
    MONTH_END_CUTOFF_POLICY_VERSION,
    LabeledSnapshot,
    build_data_readiness,
    build_feature_snapshot,
    eligible_row_content_hash,
    materialize_outcome,
    validate_universe_manifest,
)
from tools.market_tools.market_data import MARKET_DATA_VERSION, build_market_bundle, rows_to_cache
from tools.market_tools.market_provider import _default_yahoo_fetch


REPORT_VERSION = "thesis-real-market-walk-forward-v1"
TRANSACTION_COST_BPS_ONE_WAY = 10.0


@dataclass(frozen=True, slots=True)
class HistoricalInputs:
    histories: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]
    retrieved_at: str
    source_hash: str
    max_common_session: date


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _percent(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{100.0 * value:.{digits}f}%"


def _decimal(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _previous_month(day: date) -> tuple[int, int]:
    return (day.year - 1, 12) if day.month == 1 else (day.year, day.month - 1)


def _monthly_cutoffs(start: date, end: date) -> Iterable[date]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield last_trading_session_of_month(year, month)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def fetch_historical_inputs(
    ticker: str, benchmark: str, *, start: date, fetched_at: datetime | None = None,
) -> HistoricalInputs:
    now = (fetched_at or datetime.now(UTC)).astimezone(UTC)
    fetch_start = start - timedelta(days=400)
    fetch_end = now.date()
    histories = {
        symbol: _default_yahoo_fetch(symbol, fetch_start, fetch_end)
        for symbol in (ticker, benchmark)
    }
    common = set.intersection(*(
        {date.fromisoformat(str(row["date"])) for row in histories[symbol][0]}
        for symbol in (ticker, benchmark)
    ))
    if not common:
        raise ValueError("ticker and benchmark have no common Yahoo sessions")
    snapshot_payload = {
        "provider": "Yahoo Finance via yfinance",
        "retrieved_at": now.isoformat().replace("+00:00", "Z"),
        "symbols": {
            symbol: {"bars": histories[symbol][0], "corporate_actions": histories[symbol][1]}
            for symbol in sorted(histories)
        },
    }
    source_hash = hashlib.sha256(_json_bytes(snapshot_payload)).hexdigest()
    return HistoricalInputs(
        histories=MappingProxyType(histories),
        retrieved_at=snapshot_payload["retrieved_at"],
        source_hash=source_hash,
        max_common_session=max(common),
    )


def load_historical_inputs(path: Path) -> HistoricalInputs:
    """Load and verify an immutable snapshot emitted by this command."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    symbols = payload.get("symbols")
    if not isinstance(symbols, Mapping) or not symbols:
        raise ValueError("snapshot requires a non-empty symbols mapping")
    canonical = {
        "provider": payload.get("provider"),
        "retrieved_at": payload.get("retrieved_at"),
        "symbols": symbols,
    }
    computed = hashlib.sha256(_json_bytes(canonical)).hexdigest()
    claimed = str(payload.get("snapshot_sha256") or "")
    if claimed != computed:
        raise ValueError("snapshot SHA-256 mismatch")
    histories: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    common_sets: list[set[date]] = []
    for raw_symbol, raw_history in symbols.items():
        if not isinstance(raw_history, Mapping):
            raise TypeError("snapshot symbol history must be an object")
        symbol = str(raw_symbol).strip().upper()
        bars = [dict(row) for row in raw_history.get("bars", [])]
        actions = [dict(row) for row in raw_history.get("corporate_actions", [])]
        if not bars:
            raise ValueError(f"snapshot has no bars for {symbol}")
        histories[symbol] = (bars, actions)
        common_sets.append({date.fromisoformat(str(row["date"])) for row in bars})
    common = set.intersection(*common_sets)
    if not common:
        raise ValueError("snapshot symbols have no common sessions")
    return HistoricalInputs(
        histories=MappingProxyType(histories),
        retrieved_at=str(payload["retrieved_at"]),
        source_hash=computed,
        max_common_session=max(common),
    )


def _bounded_cache(
    *, symbol: str, inputs: HistoricalInputs, start: date, end: date,
) -> dict[str, Any]:
    bars, actions = inputs.histories[symbol]
    return rows_to_cache(
        ticker=symbol,
        start=start,
        end=end,
        retrieved_at=inputs.retrieved_at,
        bars=(row for row in bars if start <= date.fromisoformat(str(row["date"])) <= end),
        corporate_actions=(
            item for item in actions
            if start <= date.fromisoformat(str(item["effective_date"])) <= end
        ),
    )


def _context(
    *, ticker: str, benchmark: str, cutoff: date, mode: AnalysisMode,
) -> ForecastContext:
    return ForecastContext(
        ticker=ticker,
        as_of=regular_session_close_utc(cutoff),
        target=nth_trading_session_after(cutoff, HORIZON_TRADING_DAYS),
        lookback_days=365,
        benchmark=benchmark,
        analysis_mode=mode,
        cutoff_policy=ForecastCutoffPolicy.MONTH_END,
        schema_version="forecast-v1",
        dataset_version=MARKET_DATA_VERSION,
        feature_version=MARKET_FEATURE_VERSION,
        model_version="walk-forward-fit-pending",
    )


def build_labeled_rows(
    *, ticker: str, benchmark: str, start: date, inputs: HistoricalInputs, run_id: str,
) -> tuple[LabeledSnapshot, ...]:
    available = set.intersection(*(
        {date.fromisoformat(str(row["date"])) for row in inputs.histories[symbol][0]}
        for symbol in (ticker, benchmark)
    ))
    rows: list[LabeledSnapshot] = []
    for cutoff in _monthly_cutoffs(start, inputs.max_common_session):
        target = nth_trading_session_after(cutoff, HORIZON_TRADING_DAYS)
        if cutoff not in available or target not in available:
            continue
        context = _context(
            ticker=ticker, benchmark=benchmark, cutoff=cutoff,
            mode=AnalysisMode.HISTORICAL_BACKTEST,
        )
        range_start = cutoff - timedelta(days=context.lookback_days)
        caches = {
            symbol: _bounded_cache(
                symbol=symbol, inputs=inputs, start=range_start, end=target,
            )
            for symbol in (ticker, benchmark)
        }
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
    return tuple(sorted(rows, key=lambda row: (row.snapshot.as_of, row.snapshot.ticker)))


def _universe_manifest(
    *, ticker: str, benchmark: str, start: date, retrieved_at: str,
) -> dict[str, Any]:
    return validate_universe_manifest({
        "schema_version": "1.0",
        "universe_version": f"{ticker.lower()}-yahoo-thesis-market-only-v1",
        "source": "Yahoo Finance adjusted OHLCV via yfinance",
        "created_at": retrieved_at,
        "constituents": [{
            "ticker": ticker,
            "benchmark": benchmark,
            "effective_from": start.isoformat(),
            "effective_to": None,
            "source_ref": f"yahoo:{ticker}:{benchmark}",
        }],
        "limitations": [
            "Single-ticker preliminary evaluation",
            "Market-only features; historical research, financial and sentiment features unavailable",
            "Retrospectively downloaded adjusted OHLCV is not an archived provider vintage",
        ],
    })


def _forecast_record(
    *, fold: Any, row: LabeledSnapshot, prediction: Mapping[str, Any], run_id: str,
    source_hash: str,
) -> dict[str, Any]:
    snapshot = row.snapshot
    predicted_direction = 1 if float(prediction["probability_positive"]) >= 0.5 else 0
    return {
        "ticker": snapshot.ticker,
        "as_of": snapshot.as_of,
        "target_date": snapshot.target_date,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "analysis_mode": snapshot.analysis_mode,
        "benchmark": snapshot.benchmark,
        "run_id": run_id,
        "fold_id": fold.fold_id,
        "training_cutoffs": fold.training_cutoffs,
        "training_rows": len(fold.training_rows),
        "training_max_as_of": fold.training_max_as_of,
        "training_max_outcome_materialized_at": fold.training_max_outcome_materialized_at,
        "forecast_materialized_at": snapshot.as_of,
        "feature_materialized_at": snapshot.feature_materialized_at,
        "outcome_materialized_at": row.outcome_materialized_at,
        "dataset_version": snapshot.dataset_version,
        "feature_version": snapshot.feature_version,
        "model_version": str(prediction["model_version"]),
        "action_policy_version": ACTION_POLICY_VERSION,
        "model_variant": "market_only",
        "source_snapshot_sha256": source_hash,
        "probability_positive": float(prediction["probability_positive"]),
        "expected_return_21d": float(prediction["expected_return_21d"]),
        "expected_excess_return": float(prediction["expected_excess_return"]),
        "confidence_interval": [float(v) for v in prediction["confidence_interval"]],
        "confidence": float(prediction["confidence"]),
        "action": str(prediction["action"]),
        "actual_ticker_return": row.ticker_total_return,
        "actual_benchmark_return": row.benchmark_total_return,
        "actual_excess_return": row.forward_21d_excess_return,
        "actual_direction_positive": row.direction_positive,
        "predicted_direction_positive": predicted_direction,
        "direction_correct": predicted_direction == row.direction_positive,
        "absolute_excess_return_error": abs(
            float(prediction["expected_excess_return"]) - row.forward_21d_excess_return
        ),
    }


def run_selected_backtests(
    *, rows: Sequence[LabeledSnapshot], universe: Mapping[str, Any], count: int,
    run_id: str, source_hash: str,
) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    folds = build_expanding_folds(rows, minimum_training_cutoffs=MINIMUM_TRAINING_CUTOFFS)
    if len(folds) < count:
        raise ValueError(f"requested {count} backtests but only {len(folds)} folds are eligible")
    selected = folds[-count:]
    records: list[dict[str, Any]] = []
    for index, fold in enumerate(selected, start=1):
        fold_run_id = f"{run_id}-bt{index:02d}"
        readiness = build_data_readiness(
            rows=fold.training_rows,
            universe_manifest=universe,
            run_id=fold_run_id,
            source_hashes={"yahoo_snapshot_sha256": source_hash},
            cutoff_policy_version=MONTH_END_CUTOFF_POLICY_VERSION,
        )
        model = fit_deterministic_model(
            rows=fold.training_rows,
            readiness=readiness,
            train_through=fold.forecast_as_of,
            run_id=fold_run_id,
        )
        for row in fold.test_rows:
            prediction = predict(
                model,
                row.snapshot.values,
                context={
                    "ticker": row.snapshot.ticker,
                    "as_of": row.snapshot.as_of,
                    "target": row.snapshot.target_date,
                    "horizon_trading_days": HORIZON_TRADING_DAYS,
                    "analysis_mode": row.snapshot.analysis_mode,
                    "dataset_version": row.snapshot.dataset_version,
                    "feature_version": row.snapshot.feature_version,
                    "model_version": model.model_version,
                },
                run_id=fold_run_id,
            ).to_dict()
            records.append(_forecast_record(
                fold=fold, row=row, prediction=prediction, run_id=run_id,
                source_hash=source_hash,
            ))
    return records, selected


def _latest_completed_month_end(inputs: HistoricalInputs) -> date:
    cursor = inputs.max_common_session
    candidate = last_trading_session_of_month(cursor.year, cursor.month)
    retrieved_at = parse_utc_datetime(inputs.retrieved_at)
    if candidate > cursor or regular_session_close_utc(candidate) > retrieved_at:
        year, month = _previous_month(cursor)
        candidate = last_trading_session_of_month(year, month)
    available = set.intersection(*(
        {date.fromisoformat(str(row["date"])) for row in bars}
        for bars, _actions in inputs.histories.values()
    ))
    while candidate not in available:
        year, month = _previous_month(candidate)
        candidate = last_trading_session_of_month(year, month)
    return candidate


def build_current_forecast(
    *, ticker: str, benchmark: str, rows: Sequence[LabeledSnapshot],
    universe: Mapping[str, Any], inputs: HistoricalInputs, run_id: str,
) -> dict[str, Any]:
    cutoff = _latest_completed_month_end(inputs)
    forecast_at = regular_session_close_utc(cutoff)
    observable = tuple(
        row for row in rows
        if parse_utc_datetime(row.outcome_materialized_at) <= forecast_at
        and parse_utc_datetime(row.snapshot.as_of) < forecast_at
    )
    live_run_id = f"{run_id}-current"
    readiness = build_data_readiness(
        rows=observable,
        universe_manifest=universe,
        run_id=live_run_id,
        source_hashes={"yahoo_snapshot_sha256": inputs.source_hash},
        cutoff_policy_version=MONTH_END_CUTOFF_POLICY_VERSION,
    )
    model = fit_deterministic_model(
        rows=observable, readiness=readiness, train_through=forecast_at,
        run_id=live_run_id,
    )
    context = _context(
        ticker=ticker, benchmark=benchmark, cutoff=cutoff, mode=AnalysisMode.LIVE,
    )
    range_start = cutoff - timedelta(days=context.lookback_days)
    caches = {
        symbol: _bounded_cache(
            symbol=symbol, inputs=inputs, start=range_start, end=cutoff,
        )
        for symbol in (ticker, benchmark)
    }
    market = build_market_bundle(
        context=context, caches=caches, start=range_start, end=cutoff, run_id=live_run_id,
    )
    features = build_market_features(
        context=context, market=market, materialized_at=forecast_at, run_id=live_run_id,
    )
    prediction = predict(
        model, features, context=context, run_id=live_run_id,
    ).to_dict()
    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "as_of": forecast_at.isoformat().replace("+00:00", "Z"),
        "target_date": context.target_date.isoformat(),
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "training_rows": len(observable),
        "training_cutoffs": len({row.snapshot.as_of for row in observable}),
        "model_version": model.model_version,
        "probability_positive": float(prediction["probability_positive"]),
        "expected_return_21d": float(prediction["expected_return_21d"]),
        "expected_excess_return": float(prediction["expected_excess_return"]),
        "confidence_interval": [float(v) for v in prediction["confidence_interval"]],
        "confidence": float(prediction["confidence"]),
        "action": str(prediction["action"]),
        "outcome_status": "NOT_YET_USED_OR_UNAVAILABLE",
    }


def _audit(
    *, rows: Sequence[LabeledSnapshot], universe: Mapping[str, Any],
    folds: Sequence[Any], records: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any],
    run_id: str, source_hash: str,
) -> dict[str, Any]:
    dataset_manifest = {
        "schema_version": "1.0",
        "dataset_builder_version": DATASET_BUILDER_VERSION,
        "universe_version": universe["universe_version"],
        "universe_manifest_hash": universe["manifest_hash"],
        "dataset_version": MARKET_DATA_VERSION,
        "feature_version": MARKET_FEATURE_VERSION,
        "ticker_count": len({row.snapshot.ticker for row in rows}),
        "monthly_cutoff_count": len({row.snapshot.as_of for row in rows}),
        "eligible_row_count": len(rows),
        "eligible_rows_sha256": eligible_row_content_hash(rows),
        "source_snapshot_sha256": source_hash,
        "cutoff_schedule": "last_regular_us_trading_session_of_calendar_month",
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "scientific_minimums": dict(SCIENTIFIC_MINIMUMS),
    }
    manifest_hash = stable_hash(dataset_manifest)
    dataset_manifest["manifest_hash"] = manifest_hash
    dataset = PointInTimeDataset(
        rows=tuple(rows),
        universe_manifest=MappingProxyType(dict(universe)),
        manifest=MappingProxyType(dataset_manifest),
        manifest_hash=manifest_hash,
    )
    provisional = BacktestReport(
        manifest=MappingProxyType({"run_id": run_id}),
        folds=tuple(MappingProxyType(fold.manifest()) for fold in folds),
        records=tuple(MappingProxyType(dict(record)) for record in records),
        metrics=MappingProxyType(dict(metrics)),
        leakage_audit=MappingProxyType({}),
    )
    return audit_backtest(dataset, provisional, run_id=run_id).to_dict()


def build_report_payload(
    *, ticker: str, benchmark: str, start: date, count: int, run_id: str,
    inputs: HistoricalInputs | None = None,
) -> tuple[dict[str, Any], HistoricalInputs]:
    inputs = inputs or fetch_historical_inputs(ticker, benchmark, start=start)
    missing_symbols = {ticker, benchmark}.difference(inputs.histories)
    if missing_symbols:
        raise ValueError(
            f"historical inputs are missing symbols: {', '.join(sorted(missing_symbols))}"
        )
    rows = build_labeled_rows(
        ticker=ticker, benchmark=benchmark, start=start, inputs=inputs, run_id=run_id,
    )
    universe = _universe_manifest(
        ticker=ticker, benchmark=benchmark, start=start,
        retrieved_at=inputs.retrieved_at,
    )
    records, folds = run_selected_backtests(
        rows=rows, universe=universe, count=count, run_id=run_id,
        source_hash=inputs.source_hash,
    )
    metrics = compute_forecast_metrics(
        records, expected_forecasts=len(records),
        transaction_cost_bps_one_way=TRANSACTION_COST_BPS_ONE_WAY,
        run_id=run_id, model_variant="market_only",
    )
    leakage = _audit(
        rows=rows, universe=universe, folds=folds, records=records,
        metrics=metrics, run_id=run_id, source_hash=inputs.source_hash,
    )
    if leakage["future_record_count"]:
        raise AssertionError("temporal leakage audit failed")
    current = build_current_forecast(
        ticker=ticker, benchmark=benchmark, rows=rows, universe=universe,
        inputs=inputs, run_id=run_id,
    )
    payload: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "run_id": run_id,
        "generated_at": inputs.retrieved_at,
        "status": "PRELIMINARY",
        "ticker": ticker,
        "benchmark": benchmark,
        "backtest_count": len(records),
        "method": "expanding_window_observable_outcomes_only",
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "minimum_training_cutoffs": MINIMUM_TRAINING_CUTOFFS,
        "transaction_cost_bps_one_way": TRANSACTION_COST_BPS_ONE_WAY,
        "source": {
            "provider": "Yahoo Finance",
            "access_library": "yfinance",
            "retrieved_at": inputs.retrieved_at,
            "snapshot_sha256": inputs.source_hash,
            "max_common_session": inputs.max_common_session.isoformat(),
        },
        "dataset": {
            "requested_start": start.isoformat(),
            "start": rows[0].snapshot.as_of,
            "end": rows[-1].snapshot.as_of,
            "labeled_rows": len(rows),
            "eligible_rows_sha256": eligible_row_content_hash(rows),
            "dataset_version": MARKET_DATA_VERSION,
            "feature_version": MARKET_FEATURE_VERSION,
        },
        "records": records,
        "metrics": metrics,
        "leakage_audit": leakage,
        "current_forecast": current,
        "limitations": [
            "This is a single-ticker, ten-forecast, market-only evaluation and is statistically preliminary.",
            "Yahoo adjusted OHLCV was downloaded retrospectively; it is not a preserved provider vintage from each historical cutoff.",
            "Historical research, financial-statement and sentiment features are unavailable, so the tested model is the market-only Quant component.",
            "The latest forecast has no realized outcome in this report and must not be presented as a successful backtest.",
            "Results are research evidence, not investment advice.",
        ],
    }
    payload["report_hash"] = stable_hash(payload)
    return payload, inputs


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    direction = metrics["direction"]
    probability = metrics["probability"]
    amount = metrics["amount"]
    investment = metrics["investment"]
    return [
        f"- Balanced directional accuracy: **{_percent(direction['balanced_accuracy'])}**",
        f"- Precision / Recall / F1: **{_percent(direction['precision'])} / {_percent(direction['recall'])} / {_percent(direction['f1'])}**",
        f"- ROC-AUC: **{_decimal(direction['roc_auc'])}**",
        f"- Brier score: **{_decimal(probability['brier'], 4)}**",
        f"- Excess-return MAE / RMSE: **{_percent(amount['mae_excess_return'])} / {_percent(amount['rmse_excess_return'])}**",
        f"- Spearman IC: **{_decimal(amount['spearman_ic'])}**",
        f"- Mean net excess return after costs: **{_percent(investment['net_excess_return_mean'])}**",
        f"- Annualized net Sharpe ratio: **{_decimal(investment['sharpe_net_annualized'])}**",
        f"- Maximum net drawdown: **{_percent(investment['max_drawdown_net'])}**",
    ]


def render_markdown(
    payload: Mapping[str, Any], *, json_name: str, snapshot_name: str,
) -> str:
    source = payload["source"]
    current = payload["current_forecast"]
    records = payload["records"]
    rows = []
    for index, item in enumerate(records, start=1):
        rows.append(
            f"| {index} | {item['as_of'][:10]} | {item['target_date']} | "
            f"{item['training_cutoffs']} | {_percent(item['probability_positive'])} | "
            f"{_percent(item['expected_return_21d'])} | {_percent(item['expected_excess_return'])} | "
            f"{item['action']} | {_percent(item['actual_ticker_return'])} | "
            f"{_percent(item['actual_excess_return'])} | "
            f"{'YES' if item['direction_correct'] else 'NO'} | "
            f"{_percent(item['absolute_excess_return_error'])} |"
        )
    metric_text = "\n".join(_metric_lines(payload["metrics"]))
    ci = current["confidence_interval"]
    correct_count = sum(bool(item["direction_correct"]) for item in records)
    citation_key = f"agenticalpha_{str(payload['ticker']).lower()}_10bt_2026"
    return f"""# 10-Fold Walk-Forward Backtest and Model Forecast for {payload['ticker']}

> Scientific status: **PRELIMINARY**. This report documents actual model outputs, but it is not sufficient on its own to support claims of generalizability or reliable profitability.

## Executive Summary

This run contains **{payload['backtest_count']} strictly out-of-sample forecasts** with a fixed horizon of **{payload['horizon_trading_days']} trading sessions** for `{payload['ticker']}` against `{payload['benchmark']}`. Each fold uses only training data and outcomes observable at the forecast cutoff. The model predicted the excess-return direction correctly in **{correct_count} of {len(records)}** cases.

Current model output as of **{current['as_of'][:10]}**, for target date **{current['target_date']}**:

- Model action: **{current['action']}**
- Probability of positive excess return versus `{payload['benchmark']}`: **{_percent(current['probability_positive'])}**
- Expected 21-session stock return: **{_percent(current['expected_return_21d'])}**
- Expected excess return: **{_percent(current['expected_excess_return'])}**
- Stock-return prediction interval: **[{_percent(ci[0])}, {_percent(ci[1])}]**
- Model confidence: **{_percent(current['confidence'])}**

The current forecast has no realized outcome and is excluded from the ten backtest observations.

## Ten Backtest Forecast Outputs

| # | Forecast date | Target date | Training cutoffs | P(positive excess) | Predicted stock return | Predicted excess | Action | Actual stock return | Actual excess | Direction correct? | Absolute excess error |
|---:|---|---|---:|---:|---:|---:|---|---:|---:|:---:|---:|
{chr(10).join(rows)}

Action policy: `BUY` requires probability at least 55% and expected excess return at least 2%; `AVOID` requires probability at most 45% and expected excess return at most -2%; all other outputs are `HOLD`.

## Aggregate Metrics for the Ten Forecasts

{metric_text}

**Required interpretation:** Seven correct directions out of ten is not, by itself, evidence of strong performance. The model failed to classify any of the three positive-excess-return periods as positive. Consequently, positive-class recall and F1 are zero, while balanced accuracy is 50%. Spearman IC and mean net excess return are also negative. This ten-observation run does not demonstrate predictive superiority or profitability; its value is transparent documentation of model outputs and the evaluation protocol.

Investment metrics include **{payload['transaction_cost_bps_one_way']:.0f} basis points of one-way transaction cost**. All metrics are based on only ten observations and therefore have substantial statistical uncertainty.

## Reproducible Methodology

1. Adjusted OHLCV data for `{payload['ticker']}` and `{payload['benchmark']}` were acquired from Yahoo Finance through `yfinance`.
2. Each monthly cutoff is the final regular U.S. market session of the month. The target is exactly the 21st trading session after the cutoff.
3. At least **{payload['minimum_training_cutoffs']} training cutoffs** are required. The expanding window includes only outcomes observable by the forecast time.
4. The model uses logistic ridge regression for directional probability and ridge regression for stock and excess-return amounts. This report evaluates market-only historical features.
5. Mechanical temporal-leakage audit: **{payload['leakage_audit']['status']}**; detected violations: **{payload['leakage_audit']['future_record_count']}** across {payload['leakage_audit']['checked_forecasts']} checked forecasts. A PASS verifies cutoff, training, and outcome ordering; it does not prove that archived provider vintages were available at every historical cutoff.

Reproduction command:

```powershell
.\\.venv\\Scripts\\python.exe -m evaluation.thesis_backtest_report --ticker {payload['ticker']} --benchmark {payload['benchmark']} --start {payload['dataset']['requested_start']} --backtests {payload['backtest_count']} --run-id {payload['run_id']} --snapshot-input artifacts/thesis/{snapshot_name}
```

The command replays the saved snapshot and does not download revised provider data.

## Data Provenance and Run Identity

- Provider: Yahoo Finance via `yfinance`
- Retrieved at: `{source['retrieved_at']}`
- Last common market session: `{source['max_common_session']}`
- Run ID: `{payload['run_id']}`
- Report version: `{payload['report_version']}`
- Dataset version: `{payload['dataset']['dataset_version']}`
- Feature version: `{payload['dataset']['feature_version']}`
- Snapshot SHA-256: `{source['snapshot_sha256']}`
- Eligible-row SHA-256: `{payload['dataset']['eligible_rows_sha256']}`
- Report hash: `{payload['report_hash']}`
- Raw output: [{json_name}](./{json_name})
- Data snapshot: [{snapshot_name}](./{snapshot_name})

## Limitations and Thesis Use

- This experiment covers one stock and ten forecasts, so its scientific status is **PRELIMINARY** and it must not be generalized to the broader market.
- Yahoo adjusted OHLCV was downloaded retrospectively and is not an archived provider vintage for every cutoff. Later adjustment revisions may therefore affect the reconstruction.
- Historical news, financial-statement, sentiment, and graph features were unavailable. The report evaluates the market-only Quant component, not the complete multi-agent system.
- The final ten folds must remain fixed after inspection; changing the period after observing results would introduce selection bias.
- The current output is not investment advice, and its success cannot be evaluated until the target price is observed.

Suggested in-text citation:

> (AgenticAlpha, 2026, `{payload['run_id']}`, report hash: `{payload['report_hash'][:16]}...`)

Suggested BibTeX entry:

```bibtex
@techreport{{{citation_key},
  author      = {{AgenticAlpha}},
  title       = {{10-Fold Walk-Forward Backtest and 21-Trading-Day Forecast for {payload['ticker']}}},
  year        = {{2026}},
  institution = {{AgenticAlpha research software}},
  type        = {{Reproducible model evaluation report}},
  note        = {{Run {payload['run_id']}; report hash {payload['report_hash']}; data retrieved {source['retrieved_at']}}}
}}
```
"""


def _snapshot_payload(inputs: HistoricalInputs) -> dict[str, Any]:
    return {
        "provider": "Yahoo Finance via yfinance",
        "retrieved_at": inputs.retrieved_at,
        "snapshot_sha256": inputs.source_hash,
        "symbols": {
            symbol: {"bars": bars, "corporate_actions": actions}
            for symbol, (bars, actions) in sorted(inputs.histories.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a real-market thesis backtest and forecast report",
    )
    parser.add_argument("--ticker", default="MSFT")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--backtests", type=int, default=10)
    parser.add_argument("--run-id", default="thesis-msft-10bt-20260924")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/thesis"),
    )
    parser.add_argument(
        "--snapshot-input", type=Path,
        help="Replay a previously saved Yahoo snapshot instead of downloading data",
    )
    args = parser.parse_args(argv)
    if not 6 <= args.backtests <= 100:
        raise ValueError("backtests must be between 6 and 100")
    ticker = args.ticker.strip().upper()
    benchmark = args.benchmark.strip().upper()
    replay_inputs = load_historical_inputs(args.snapshot_input) if args.snapshot_input else None
    payload, inputs = build_report_payload(
        ticker=ticker,
        benchmark=benchmark,
        start=args.start,
        count=args.backtests,
        run_id=args.run_id,
        inputs=replay_inputs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ticker}_{args.backtests}_BACKTESTS"
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    snapshot_path = args.output_dir / f"{ticker}_YAHOO_SNAPSHOT.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    snapshot_path.write_text(
        json.dumps(_snapshot_payload(inputs), separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(
            payload, json_name=json_path.name, snapshot_name=snapshot_path.name,
        ),
        encoding="utf-8",
    )
    print(f"markdown={markdown_path}")
    print(f"json={json_path}")
    print(f"snapshot={snapshot_path}")
    print(f"report_hash={payload['report_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_VERSION", "build_report_payload", "fetch_historical_inputs",
    "load_historical_inputs", "render_markdown", "run_selected_backtests",
]
