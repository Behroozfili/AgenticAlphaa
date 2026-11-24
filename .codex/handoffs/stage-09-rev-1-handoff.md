# Stage 09 Revision 1 Handoff

Status: **CLOSED**  
Accepted attempt: **2**  
Primary agent: **quant**  
QA: **PASS**  
Review: **APPROVED**

## Delivered contract

- A deterministic point-in-time monthly walk-forward evaluation runs only on information observable at each forecast cutoff and audits every prediction for future evidence.
- Expanding monthly folds produce the exact 21-trading-day target and evaluate seven locked variants, including Full Model, explicit baselines and ablations.
- Directional accuracy, MAE, RMSE, Brier score, calibration, coverage, interval width, turnover and cost-adjusted return are reported deterministically with honest Full Model comparisons and action-policy checks.
- Scientific classification is assigned only after executed common variant coverage is known: at least 10 tickers, 60 monthly cutoffs and 500 eligible out-of-sample forecasts are all required. Exact 499/500 counterexamples are covered.
- The accepted offline fixture remains honestly `PRELIMINARY`: 2 tickers, 48 source cutoffs, 11 folds and 22 eligible OOS forecasts per variant. No investment-performance claim is made.
- Compact payload-safe lifecycle telemetry covers fixture materialization, model fit, prediction, aggregation, leakage audit and run boundaries with immutable identity `09/r1/a2/quant`, measured finite latency and isolated sink failures.
- Telemetry is outside the scientific report: enabled and disabled runs have identical outputs and canonical report hash `213ad73328c244c495d0a465ed1bc7b02fb1ab2348a803e6ab807948f3aad779`.

## Acceptance evidence

- Focused and action-policy tests: 35 passed.
- Stage 07 and related regression tests: 137 passed.
- Full collection: 1455 collected, 0 errors.
- Independent real-engine boundaries: 499 OOS is `PRELIMINARY`; 500 OOS is `SCIENTIFIC`; 9 tickers or 59 cutoffs remain `PRELIMINARY`.
- Canonical fixture: seven variants, 22 OOS forecasts per variant, 0 future records and `PRELIMINARY` classification.
- Lifecycle probe: all events use `09/r1/a2/quant` and `stage09-r1-a2`; latency, ordering, progress, privacy and failing-sink isolation passed.
- Attempt 1 workflow artifacts remain immutable and the accepted Stage 08 dependency hash matches.

## Retry history

- Attempt 1 reached Review and was rejected by `REVIEW-09-001` because scientific readiness used total source rows instead of executed eligible OOS forecasts.
- Attempt 2 corrected the classification boundary. A supplemental observability finding, `OBS-09-A2-001`, then advanced runtime provenance to the active attempt identity. QA passed and Review approved.

## Rollback

Rollback is local and reversible: restore the twelve Stage 09 source/test/fixture paths recorded in the accepted implementation artifact and remove only the Stage 09 workflow artifacts created by this revision. Do not alter accepted Stage 08 artifacts or activate downstream stages during rollback.

## Known limitations and safety

- Evidence is deterministic and offline; live providers, external databases and external trace export were not exercised.
- The sample is deliberately too small for scientific status, so all conclusions remain engineering/preliminary evidence.
- No network acquisition, external write, migration, seed, deletion, deployment, commit or push was performed.
- Stage 10A remains `PENDING / NOT_STARTED` and is not activated by this closure.

Created at: `2026-09-12T08:20:26.3007664Z`
