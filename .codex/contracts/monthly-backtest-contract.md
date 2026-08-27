# Monthly forecast and backtest contract

This is the default versioned scientific contract for the refactor. A later change requires a new contract version, a new dataset/model version, and invalidation of affected downstream stages.

## Forecast target v1

- `as_of`: the close of the last regular US trading session in each calendar month, stored as timezone-aware UTC and retaining `America/New_York` market-calendar semantics.
- `horizon_trading_days`: exactly 21 trading sessions.
- `target_date`: the 21st regular trading session strictly after `as_of`.
- `start_price`: adjusted close at `as_of`; `target_price`: adjusted close at `target_date`.
- Labels: ticker total return, SPY total return, excess return, and positive/negative direction over that exact interval.
- Corporate actions: use adjusted history consistently; never mix adjusted and raw closes in one label.
- DCF/WACC is a long-term valuation feature only and is never the source of the 21-session target or predicted price.

## Provider policy v1

- Price/benchmark development source: the existing Yahoo Finance adapter, materialized into an immutable cache with provider, retrieval time, adjustment mode, and content hash. Provider fallback is forbidden inside a historical run.
- Filings/fundamentals: SEC EDGAR submissions and CompanyFacts; only facts filed on or before `as_of`. Store accession and filed time. Later restatements are invisible to earlier cutoffs.
- Historical research/news: GDELT or a previously archived project document with an immutable source identity and publication time. Live RSS/current web results are forbidden in historical mode. Inaccessible full text becomes missing evidence.
- Sentiment: computed only from the same eligible point-in-time evidence; no current sentiment API fallback.
- Macro: ALFRED/FRED vintage-aware series are optional features. If a correct vintage is unavailable, mark missing; do not use today's revised value.
- Network acquisition and bulk downloads require explicit user authorization. Until then, implementation and QA use deterministic local fixtures/cache only.

## Universe and data-readiness v1

- Every run must use a committed/versioned universe manifest containing ticker, effective membership interval, benchmark, source, and manifest hash. Start from `.codex/contracts/templates/universe-manifest.json`; the template itself is not an approved universe.
- A single-ticker MSFT run is a demo/regression fixture, not sufficient evidence of general investment performance.
- The initial scientific report is `PRELIMINARY` unless it has at least 10 tickers, 60 monthly cutoffs, and 500 eligible out-of-sample forecasts after coverage rules. Engineering QA can still pass below these counts if the limitation is explicit.
- Before model fitting, Stage 07 must produce a DATA_READINESS manifest with per-source date range, ticker count, cutoff count, feature missingness, evidence coverage, filing coverage, graph coverage, excluded rows/reasons, and hashes/versions. Its shape starts from `.codex/contracts/templates/data-readiness-manifest.json`.
- A row is eligible only when its price label exists and its feature snapshot was finalized before outcome materialization. Missing optional sources remain explicit missing indicators; missing is never zero.

## Model, action, and folds v1

- Forecast numbers come only from deterministic Quant artifacts; an LLM may explain but never alter them.
- Use expanding-window walk-forward validation. Minimum initial training history is 36 monthly cutoffs; hyperparameters, imputers, scalers, selectors, and any calibration use only past training/validation folds.
- Default `action_policy_v1` is fixed before test evaluation: `BUY` requires `probability_positive >= 0.55` and `expected_excess_return >= 0.02`; `AVOID` requires `probability_positive <= 0.45` and `expected_excess_return <= -0.02`; otherwise `HOLD`.
- If required feature coverage or model validity fails, output `INSUFFICIENT_DATA`; do not manufacture confidence or force HOLD as if it were a valid forecast.
- Transaction-cost reporting assumes 10 basis points per one-way position change and must also report gross results.
- Compare class-prior/always-HOLD, momentum-21d, market-only, no-Graph, no-Sentiment, no-Financial, and Full Model under identical folds and eligible rows.
- Engineering QA judges correctness, leakage, reproducibility, and honest reporting. Full Model underperformance is a valid scientific result, not an automatic QA failure.

## Required reproducibility fields

Every forecast/backtest record includes ticker, `as_of`, target date, horizon, analysis mode, universe manifest hash, dataset/feature/model/action-policy versions, model variant, benchmark, run ID, evidence IDs, missing/degraded flags, and materialization timestamps.
