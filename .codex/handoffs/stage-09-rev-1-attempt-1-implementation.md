# Stage 09 revision 1 attempt 1 implementation report

- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `155`
- Runtime identity: `stage09-r1-a1` (`stage=09`, `revision=1`, `attempt=1`, `agent=quant`)
- Stage 10A: `PENDING / NOT_STARTED`

## Outcome

Stage 09 delivers deterministic point-in-time walk-forward evaluation over local fixtures. Expanding folds train only on rows whose outcomes were observable by the forecast cutoff. The same 11 folds and 22 out-of-sample rows per variant are used for class-prior/always-HOLD, momentum-21d, market-only, no-Graph, no-Sentiment, no-Financial and Full Model. Direction, probability/calibration, amount, investment and coverage metrics include gross returns and net returns after the fixed 10-bps one-way transaction cost.

The mechanical audit checks 96 source rows and 154 variant forecast records and reports zero future records. The canonical report remains `PRELIMINARY`: its synthetic two-ticker, 48-cutoff fixture is below the contract minimums of 10 tickers, 60 cutoffs and 500 eligible out-of-sample forecasts. This is engineering evidence, not investment-performance evidence.

## OBS-09-001 resolution

`STAGE09_RUNTIME_IDENTITY` is an immutable local identity for `09/r1/a1/quant`. An injectable in-process sink now receives compact allowlisted events for fixture materialization, fold model fitting, fold prediction, metric aggregation, leakage audit, run start/progress/completion and fail-closed errors. Every event has finite nonnegative latency measured with a monotonic clock. Sink failures are isolated from scientific execution.

Events carry only bounded identity, counts, stable outcomes/error codes and missing/degraded flags. They never serialize `BacktestReport`, raw feature vectors, labels, actual or predicted numeric arrays, model coefficients, evidence IDs/content, prompts, chain-of-thought, private payloads, exception messages/repr/tracebacks, secrets, tokens or headers. Failure events are emitted before the original exception is re-raised.

The enabled-versus-disabled counterprobe produced exact equality of the complete scientific output. The canonical report hash remains `213ad73328c244c495d0a465ed1bc7b02fb1ab2348a803e6ab807948f3aad779`; report level remains `PRELIMINARY`; fold, variant, metrics and leakage outputs are unchanged. The successful probe emitted 231 local events across all required boundaries without contacting an external trace service.

## Scientific result and limitations

Full Model achieved balanced accuracy 1.0, Brier 0.009131935171661608, excess-return MAE 0.0025066123075812787, Spearman IC 0.9909655561829477 and mean net excess return 0.015189519508490802 on the synthetic fixture. Full Model trails no-Financial on MAE and trails market-only/no-Graph/no-Sentiment on mean net excess return; this underperformance remains reported and does not invalidate engineering correctness.

These numbers must not be interpreted as real investment performance. No live provider, network source, external trace service or database was exercised. RAGAS remains outside forecast-accuracy evaluation.

## Verification

- Exact Stage 09 focused suite: `21 passed in 28.69s`.
- Direct Stage 07 Quant plus forecast/State/legacy regressions: `137 passed in 32.12s`.
- Enabled-versus-disabled scientific equality/privacy probe: PASS; report hash unchanged; 231 bounded events; zero future records.
- Compilation of the five modified evaluation modules and five exact tests: PASS.
- Scoped `git diff --check`: PASS.
- Official control/lock validation: `CONTROL VALID` at control revision 155 before final artifact creation.

## Contract and schema diff

No scientific dataset, row, label, feature selection, model coefficient, fold, eligibility rule, variant, action threshold, transaction cost, metric, report schema/content/hash, scientific classification or temporal audit rule changed in `PRIMARY_FINAL`. Only the accepted local observability interface and its exact tests were added to the draft implementation. Stage 07, Stage 08, state, dependencies, database schema, migrations, API/runtime and Stage 10A remain unchanged.

## Rollback

No database or external rollback is required. Archive the current Stage 09 files and artifacts with their recorded hashes. To remove only `OBS-09-001`, restore the five modified evaluation modules and five modified tests to the SHA-256 versions recorded in the immutable draft-ready artifact `de058ebaeeb139a2eb07913fdf7e4b201a68d605e2c5ecf11bce605a2fbe11f5`. To roll back all of Stage 09, additionally archive/remove the new Stage 09-only modules, fixture and tests and restore `evaluation/stage3_compare_cli.py` from the pre-stage repository baseline. Retain all workflow artifacts. No rollback action was executed.
