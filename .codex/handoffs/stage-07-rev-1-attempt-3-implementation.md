# Stage 07 Quant implementation report

- Stage: `07`
- Revision: `1`
- Attempt: `3`
- Primary agent: `quant`
- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `134`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

Stage 07 revision 1 attempt 3 is ready for independent QA. This non-overwriting
attempt resolves blocking review finding `REVIEW-07-A2-B-001`: every labeled row
must now satisfy the canonical adjusted-close identity
`forward_21d_excess_return = ticker_total_return - benchmark_total_return` within
absolute tolerance `1e-12` and relative tolerance `0`, and
`direction_positive` must be `1` exactly when the canonical excess return is
greater than zero. Both the mapping loader and DATA_READINESS fail closed on
inconsistent return/excess/direction tuples.

The synthetic fixture's 40 labels are generated through the canonical
adjusted-close label function and contain zero canonical-label mismatches. The
deterministic model was fitted only after DATA_READINESS passed and remains bound
to the exact eligible-row and training-data hashes. No LLM generated or altered
forecast numbers.

The prior `QA-07-001` and `OBS-07-001..003` fixes are preserved: milestone
bindings reproduce exactly, lifecycle metadata remains compact and allowlisted,
privacy inputs fail closed, feature/context identity is bound, and measured
latency instrumentation remains present. Observability audited the attempt with
status `AUDITED`, zero requested primary changes and zero blocking issues.

## Changed files

All paths below are locked to `quant` for Stage 07 revision 1 attempt 3:

- `tools/market_tools/historical_dataset_builder.py`
- `tests/fixtures/monthly_point_in_time_fixture.json`
- `tests/unit_tests/market_test/__init__.py`
- `tests/unit_tests/market_test/test_historical_dataset_builder.py`
- `tests/unit_tests/test_market_features_stage7.py`
- `.codex/handoffs/stage-07-rev-1-attempt-3-milestone-DATA_READINESS.json`
- `.codex/handoffs/stage-07-rev-1-attempt-3-milestone-MODEL_READY.json`
- `.codex/handoffs/stage-07-rev-1-attempt-3-draft-ready.json`
- `.codex/handoffs/stage-07-rev-1-attempt-3-implementation.md`
- `.codex/handoffs/stage-07-rev-1-attempt-3-implementation.json`

The Observability-owned audit was consumed as immutable evidence and was not
changed in this pass. Attempt 1 and Attempt 2 artifacts remain immutable.

## Milestone and audit bindings

- DATA_READINESS artifact SHA-256:
  `9f69a24d29bf4a3aeb34cb6225d0834b4655c3375ec180ff266bc9c30d1d73c9`
- DATA_READINESS manifest hash:
  `be7f8d048cca7b3083f3811320f38d20e2c03f615ebed7a9262b148cc90e90ed`
- Eligible-row SHA-256:
  `93b980674b6801db52eb08c8e6b5aa652f82bcc22be2887224d88d45454166e4`
- MODEL_READY artifact SHA-256:
  `491e9597e555fb9c5dfd9ba722896ee7e1eec66bcbc38024d896ca80e6a7f400`
- Training-data hash:
  `78af5fc9e5a1fcb7263601e65ead1732e83625cf3604cde46d2b3289f2691890`
- Model version:
  `logistic-ridge-21d-v1+sha256:f7a06c450329b7da743944cd68a3444f2522640bba899de1b00e917f8aa64e89`
- Observability artifact SHA-256:
  `43407a8a7665b56a82215b38c3353c7297517d0640c96be6f834011a28dc5f75`

## Self-tests and evidence

- Fresh final offline suite:
  `.venv\\Scripts\\python.exe -B -m pytest -p no:cacheprovider tests/unit_tests/test_forecast_contract.py tests/unit_tests/test_state_forecast_contract.py tests/unit_tests/test_market_features_stage7.py tests/unit_tests/market_test -q`
  completed with exit code `0`: `119 passed in 41.17s`.
- Fresh control/lock check at control revision 134 completed with exit code `0`
  and reported `CONTROL VALID` for Stage 07 rev1/attempt3 in `PRIMARY_FINAL`.
- Fresh SHA-256 verification found `0` mismatches across all 10 draft-declared
  runtime/test/fixture inputs; both milestone SHA-256 values and the immutable
  Observability SHA-256 matched exactly.
- The Observability audit independently executed the 69-test focused suite and
  the 119-test combined suite, and reproduced 40 rows with zero canonical-label
  mismatches, 40 exact 21-session targets, identical normal/reversed readiness,
  eligible-row, training-data, model and prediction identities, all nine measured
  lifecycle boundaries, and privacy rejection counterprobes.
- Leakage status is `PASS`: rolling inputs stop at `as_of`, target labels use the
  exact 21st following regular trading session, premature outcome materialization
  and post-`train_through` rows fail closed, preprocessing is fitted from the
  training fold only, and model fit is bound to the readiness-approved rows.

## Contract/schema changes

The existing `forward_21d_excess_return-v1`, dataset, feature, model family and
`action_policy_v1` versions are preserved. This attempt enforces the already
canonical target-label semantics at ingestion and DATA_READINESS; it does not
introduce a new forecast contract, threshold, schema migration or downstream
invalidation. DATA_READINESS remains a mandatory pre-fit gate.

## Known limitations and out-of-scope findings

The evidence is `PRELIMINARY`: the locked fixture is synthetic, single-ticker and
40-cutoff, so it proves engineering correctness and reproducibility only. It is
not investment-performance evidence. Research/news, SEC filing and temporal Graph
coverage are reference-only and remain `UNVERIFIED_REFERENCE_ONLY`. Live-provider
and multi-ticker behavior were not exercised. Walk-forward performance metrics,
baselines, transaction costs and ablations remain Stage 09 scope.

No network/provider acquisition, external trace call, database read/write,
migration, seed, deletion, deployment, commit, push or Stage 08 work occurred.

## Rollback

Rollback was documented but not executed. First archive and verify the SHA-256 of
the five attempt-3-modified source/test/fixture paths and the five Quant-owned
attempt-3 artifacts. Move those exact current files into a timestamped workspace
quarantine without deletion. Restore only the five source/test/fixture paths from
the authorized Attempt 2 archival source, verifying the Attempt 2 hashes recorded
in its implementation artifact; never synthesize missing prior bytes. Preserve
the immutable Observability audit, all Attempt 1/2 artifacts, refactor state,
dependency handoffs, migrations and external data. If an authorized prior-byte
source is unavailable, stop and treat rollback as blocked rather than guessing.
