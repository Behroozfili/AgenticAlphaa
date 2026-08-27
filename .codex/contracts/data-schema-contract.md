# Temporal forecast data-schema contract

Stage 02 may create versioned, reversible schema files because the following natural keys are frozen before migration design. Applying any migration remains separately unauthorized.

## Required identities

- `feature_snapshots`: unique `(ticker, as_of, horizon_trading_days, analysis_mode, feature_set_variant, dataset_version, feature_version)`.
- `monthly_forecasts`: unique `(ticker, as_of, target_date, horizon_trading_days, model_variant, model_version, feature_version, dataset_version, run_id)`.
- `forecast_outcomes`: model-independent unique `(ticker, as_of, target_date, horizon_trading_days, price_policy_version, benchmark, dataset_version)`.
- `backtest_runs`: immutable `run_id`, plus unique reproducibility hash over universe, cutoff schedule, folds, costs, dataset, feature, model, action-policy, and code/config versions.
- `analyses`: identity must distinguish `(ticker, as_of, horizon_trading_days, analysis_mode, schema_version)`; legacy and monthly records must not merge on ticker alone.
- Evidence/document identities must be immutable and shared across Vector and Graph using canonical source identity plus content/chunk hash and version.

## Temporal rules

- Preserve `published_at`, `retrieved_at`, `filed_at`, `effective_from`, `effective_to`, and materialization time when applicable.
- Unknown time remains null/quarantined and is ineligible for historical execution; it is never mapped to current time.
- Historical RPC/traversal must enforce ticker, `as_of`, dataset version, and effective interval before ranking.
- Outcomes are written only after target data is observable and never mutate the previously stored forecast.
- Ablation/model variants coexist; they must never overwrite one another.

## Migration sequencing

- Stage 02 creates additive shadow structures using these frozen keys, reconciliation queries, and rollback pointer/view steps.
- Production read/write switching, migration application, backfill, and legacy deletion require explicit user authorization and evidence gates.
- The missing legacy root migration is not reconstructed implicitly. Its disposition belongs to Stage 00 and is tracked in `.codex/baseline-manifest.json`.
