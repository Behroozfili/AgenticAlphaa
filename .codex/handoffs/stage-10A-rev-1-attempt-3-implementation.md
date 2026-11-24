# Stage 10A Revision 1 Attempt 3 - Core Implementation

## Outcome

Stage 10A Attempt 3 is implemented and ready for QA. The API/runtime
configuration boundary now clears hostile legacy exporter state for every
default, incomplete, and complete Settings combination before effective gate
evaluation. Observability advanced the single runtime identity to
`10A/r1/a3/core`; Core updated only the three stale direct assertions during
PRIMARY_FINAL.

- REVIEW-10A-A2-001 is resolved: all 32 configuration combinations end with
  Sentry and LangSmith inactive, trace environment disabled, the credential
  export variable removed, and zero Sentry SDK, LangSmith Client, or socket
  calls.
- REVIEW-10A-001 remains resolved fail-closed: external exporters stay
  unavailable while safe local lifecycle telemetry remains usable.
- REVIEW-10A-002 remains resolved: the accepted analyze route hash and its
  legacy Manager resolution/status behavior are unchanged.
- OBS-10A-A3-001 is resolved: direct tests now require attempt 3 and
  `stage10a-r1-a3`.

## Attempt 3 changed paths

Core-owned:

- `api/main.py`
- `tests/unit_tests/api_test/test_api_main.py`
- `tests/unit_tests/test_stage10a_runtime_foundation.py`

Observability-owned support consumed without further Core modification:

- `core/observability.py`
- `langsmith_trace.py`
- `tests/unit_tests/core_test/test_observability.py`
- `tests/unit_tests/test_stage10a_observability.py`

Workflow artifacts:

- `.codex/handoffs/stage-10A-rev-1-attempt-3-draft-ready.json`
- `.codex/handoffs/stage-10A-rev-1-attempt-3-observability.json`
- `.codex/handoffs/change-request-stage-10A-rev-1-attempt-3-1.json`
- `.codex/handoffs/stage-10A-rev-1-attempt-3-implementation.md`
- `.codex/handoffs/stage-10A-rev-1-attempt-3-implementation.json`

All 14 Attempt 1 and Attempt 2 artifacts remain immutable and hash-matched.

## Final verification

- Direct identity/API tests: 49 passed in 5.14s.
- Observability tests: 33 passed in 2.28s, including the 32-case hostile
  environment/exporter socket guard.
- Complete Stage 10A API/config/dependency/exception/analyze/memory/root/
  runtime/observability suite: 246 passed in 24.91s.
- Direct analyze route behavior/privacy tests: 22 passed in 2.17s.
- Proportional Stage 01/08/09 regressions: 140 passed in 23.68s.
- Forced-offline full collection: 1478 tests collected in 17.15s with zero
  collection errors and no external request warning.
- py_compile, scoped git diff --check, runtime identity/privacy scans,
  control/lock validation, immutable artifact verification, and SHA-256
  reconciliation passed.

All accepted final evidence used `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, disabled tracing
environment, and the local no-proxy guard. Dedicated socket guards observed
zero external attempts. No database, exporter, migration, seed, dependency,
deletion, deployment, commit, or push operation was performed.

## Honest operational note

An earlier non-offline collection during Attempt 2 emitted an unauthenticated
Hugging Face Hub warning while cached weights loaded. Actual external contact
by that earlier process was not verified. It is excluded from acceptance:
every accepted Attempt 3 run was forced offline, the final collection emitted
no such warning, and dedicated socket guards observed zero external attempts.

## Known limitations

- External Sentry and LangSmith export intentionally remains unavailable until
  every request-reachable legacy trace/capture path is separately proven
  payload-safe.
- The accepted route delegates fresh Manager/memory creation to the app-state
  stateless factory compatibility surface and does not add route-level
  dependency-memory rebinding.
- Stage 10B remains pending/not started and owns persistence, forecast route
  fields, authentication-aware identity, progress/backtest jobs, UI, and E2E
  integration.
- Live providers and external services were not exercised.

## Rollback

No database rollback is required. Restore the three Attempt 3 Core source/test
paths to their Attempt 2 workspace contents while preserving immutable workflow
artifacts and accepted Stage 00-09 work. Do not activate or modify Stage 10B,
apply migrations, seed, or mutate external data during rollback.
