# Stage 09 revision 1 attempt 2 implementation report

- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `160`
- Runtime identity: `stage09-r1-a2` (`stage=09`, `revision=1`, `attempt=2`, `agent=quant`)
- Stage 10A: `PENDING / NOT_STARTED`

## Outcome

Attempt 2 resolves `REVIEW-09-001`: source-row volume can no longer establish scientific readiness. Dataset materialization defers classification with zero eligible OOS forecasts. The executed walk-forward report classifies only after all seven variants prove identical row identity and eligible coverage, and requires all three frozen minimums: at least 10 executed tickers, 60 monthly source cutoffs and 500 common eligible out-of-sample forecasts per variant.

The exact boundary evidence is: 600 source rows with no executed OOS count remain `PRELIMINARY`; 499 OOS forecasts remain `PRELIMINARY`; 500 become `SCIENTIFIC` only with at least 10 tickers and 60 cutoffs; 9 tickers or 59 cutoffs remain `PRELIMINARY` even at 500 OOS forecasts.

The current deterministic fixture remains honestly `PRELIMINARY` with 2 tickers, 48 source cutoffs, 11 folds and 22 common OOS forecasts per variant. Its folds, records, seven variants, metrics, gross/net 10-bps results, Full Model comparison and zero-future audit are unchanged. The exact report hash remains `213ad73328c244c495d0a465ed1bc7b02fb1ab2348a803e6ab807948f3aad779`.

## Supplemental observability resolution

`OBS-09-A2-001` is resolved by advancing the single immutable `STAGE09_RUNTIME_IDENTITY` to attempt 2 and implementation version `stage09-r1-a2`. Every fixture, fold-fit, fold-prediction, metric, leakage, run and progress event continues to derive from that identity. Direct lifecycle assertions now require `09/r1/a2/quant`.

No telemetry field, boundary, event order, count semantics, latency measurement, degraded/fail-closed behavior, privacy allowlist, stable error code or sink-isolation rule changed. The independent probe emitted 231 events, all with current attempt-2 identity. Telemetry-enabled and disabled scientific outputs remained exactly equal.

## Verification

- Exact Stage 09 focused suite: `23 passed in 10.86s`.
- Direct Stage 07 Quant regression: `71 passed in 1.16s`.
- Exact Stage 09 collection: 23 tests collected in 0.73s with zero errors.
- Independent identity/scientific probe: PASS for all 231 events, active state/runtime `r1/a2`, 499/500 boundary, `PRELIMINARY`, exact report hash and zero future records.
- Compilation of the two retry modules and their direct tests: PASS.
- Scoped `git diff --check`: PASS.
- Official control/lock validation: `CONTROL VALID` at control revision 160 before final artifacts.

## Scientific and operational limitations

The fixture is synthetic engineering evidence, not investment-performance evidence. Full Model underperformance disclosed in Attempt 1 remains unchanged. RAGAS remains outside forecast accuracy. No network source, live provider, external trace service, database, migration, seed, deletion, deploy, commit or push was used.

## Rollback

No database or external rollback is required. Archive Attempt 2 files and artifacts with their recorded hashes. To revert only the supplemental identity update, restore `evaluation/point_in_time_dataset.py` and `tests/unit_tests/test_monthly_backtest_stage9.py` to their immutable Attempt-2 draft hashes. To revert the entire retry, restore the four REVIEW-09-001 source/test paths to the hashes recorded in the immutable Attempt-1 implementation while retaining all workflow artifacts. No rollback action was executed.
