# Stage 10B revision 1 attempt 1 implementation report

- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `185`
- Runtime identity: `stage10b-r1-a1`
- Stage 10A: `CLOSED`
- Stage 11: `PENDING / NOT_STARTED`

## Outcome

Stage 10B now connects the closed monthly forecast pipeline to the API, versioned idempotent persistence, progress streaming, asynchronous backtest jobs, and the frontend. Monthly requests accept canonical `ticker`, `as_of`, fixed 21-session `horizon`, `mode`, and optional `run_id`; responses expose only the compact forecast contract. Two concurrent API requests retain independent ticker, cutoff, run, forecast, evidence, and telemetry identities.

The persistence path uses deterministic analysis identity and natural-key upserts for compact forecasts. Outcomes remain fail-closed until their observation timestamp is available. Persistence failure is nonfatal and produces only a stable local telemetry code. No narrative, raw evidence, private request text, user preferences, Manager state, exception object, or exception text is duplicated into persistence or telemetry.

Backtests run asynchronously from one fixed Stage 09 fixture through an app-scoped job registry. Submission is nonblocking and idempotent by run ID; status and compact artifact endpoints expose aggregate metrics and hashes without raw folds or records. Progress termination covers pipeline and backtest completion/failure.

The frontend exposes monthly cutoff and mode controls, a compact 21-session forecast card, degraded/insufficient presentation, and asynchronous backtest status. DCF remains explicitly labeled as a long-term valuation feature rather than the expected 21-day price.

## Observability findings resolved

- `OBS-10B-001`: monthly analyze start and terminal events now use the canonical context, opaque run ID, measured latency, evidence count, degraded flags, and stable failure code.
- `OBS-10B-002`: monthly persistence completion/failure is recorded locally without changing nonfatal persistence behavior or idempotent keys.
- `OBS-10B-003`: asynchronous backtest start/completion/failure emits current-attempt local telemetry with available version metadata and stable `BACKTEST_EXECUTION_FAILED`.
- `OBS-10B-004`: route, job, privacy, latency, raising-sink, persistence-disposition, and concurrent API isolation tests were added.

All Stage 10B telemetry uses the allowlisted `stage10b-r1-a1` identity and an injected local sink. External Sentry and LangSmith export remains fail-closed under the accepted Stage 10A policy.

## Verification

- Direct telemetry/analyze/backtest/concurrency and accepted analyze-route regressions: `42 passed in 1.67s`.
- Combined Stage 10B suite plus sensitive Stage 08, 09, and 10A regressions: `127 passed in 27.79s`.
- Observability plus prior privacy/fail-closed regressions before callsite wiring: `39 passed in 8.04s`.
- Full forced-offline collection after final tests: `1503 tests collected in 28.70s`, zero collection errors.
- Compilation of changed Python source/test paths: PASS.
- Scoped `git diff --check`: PASS; line-ending notices only.
- Official state/lock validation: `CONTROL VALID` at control revision 185.

All accepted evidence used offline fixtures and mocks. No network provider, external trace service, live database, migration, seed, dependency install, deletion, deployment, commit, or push was used.

## Contract diff

The legacy analyze request remains valid when monthly fields are absent. Monthly behavior is additive and fixed to the canonical 21-session contract. Manager lifecycle isolation from Stage 10A, Stage 08 forecast numerics/action, Stage 09 backtest formulas, database schema, and closed-stage artifacts are unchanged. The final observability wiring adds local payload-safe events only and does not change API response schemas, URLs, UI behavior, persistence keys, or computation.

## Limitations

- Validation is deterministic and offline; external providers and real database writes were intentionally not exercised.
- Backtest jobs are process-local; a durable distributed queue is outside Stage 10B scope.
- Frontend validation is static contract coverage rather than an interactive browser session.

## Rollback

No database or external rollback is required because no external write or migration was executed. Archive the current Stage 10B files and artifacts with their recorded SHA-256 values, then restore only the Stage 10B source/test paths to their pre-stage versions while retaining the draft, observability, change-request, implementation, QA, review, and handoff audit chain. Do not modify closed Stage 10A or earlier artifacts. No rollback action was executed.
