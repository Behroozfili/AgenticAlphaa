# Stage 10A Revision 1 Attempt 2 — Core Implementation

## Outcome

Stage 10A Attempt 2 is implemented and ready for QA. The retry resolves both
review blockers and the final Observability identity request without activating
Stage 10B or changing forecast, financial, research, sentiment, RAG, Manager, or
error-handler business behavior.

- REVIEW-10A-001 is resolved fail-closed: Sentry and LangSmith external
  exporters remain disabled even when all flags and credentials are present,
  until every request-reachable legacy trace/capture path is separately proven
  payload-safe. Safe local injectable Stage 10A telemetry remains available.
- REVIEW-10A-002 is resolved: api/routes/analyze.py again selects
  request.app.state.manager_agent, retains the legacy injected-memory use for
  preferences, and forwards computed execution status unchanged to persistence.
  Only payload-free logging, stable error codes, private persistence redaction,
  and safe lifecycle behavior remain from the authorized route sanitation.
- OBS-10A-A2-001 is resolved: the two direct runtime tests now assert
  10A/r1/a2/core and stage10a-r1-a2.

## Attempt 2 changed paths

Core-owned:

- api/routes/analyze.py
- tests/unit_tests/api_test/test_analyze_cor_api.py
- tests/unit_tests/api_test/test_api_main.py
- tests/unit_tests/test_stage10a_runtime_foundation.py

Observability-owned support consumed by Core without further modification:

- core/observability.py
- langsmith_trace.py
- tests/unit_tests/core_test/test_observability.py
- tests/unit_tests/test_stage10a_observability.py

Workflow artifacts:

- .codex/handoffs/stage-10A-rev-1-attempt-2-draft-ready.json
- .codex/handoffs/stage-10A-rev-1-attempt-2-observability.json
- .codex/handoffs/stage-10A-rev-1-attempt-2-implementation.md
- .codex/handoffs/stage-10A-rev-1-attempt-2-implementation.json

All Attempt 1 artifacts remain immutable and hash-matched.

## Final verification

- Direct identity tests: 18 passed in 2.06s.
- Complete Stage 10A API/config/dependency/exception/analyze/memory/root/runtime/
  observability suite: 214 passed in 23.18s.
- Dedicated fully-enabled exporter PRIVATE_SENTINEL socket-guard countertest:
  1 passed in 1.85s, with zero Sentry initialization/capture, LangSmith client,
  or socket calls.
- Direct analyze route behavior/privacy tests: 22 passed in 1.89s.
- Proportional Stage 01/08/09 regressions: 140 passed in 44.93s.
- Forced-offline full collection: 1446 tests collected in 24.07s, zero
  collection errors.
- py_compile, scoped git diff --check, runtime-identity inspection, privacy
  callsite inspection, control/lock validation, and SHA-256 reconciliation pass.

All accepted final test evidence ran with HF_HUB_OFFLINE=1,
TRANSFORMERS_OFFLINE=1, and HF_DATASETS_OFFLINE=1; relevant suites also used
the local no-proxy guard. No external database, exporter, migration, seed,
dependency, deletion, deployment, commit, or push operation was performed.

## Honest operational note

An earlier non-offline collection in the Observability pass emitted an
unauthenticated Hugging Face Hub warning while cached weights loaded. Actual
external contact by that earlier process was not verified. It is not used as
accepted final evidence: the final collection was forced offline and completed
without that warning, while the dedicated enabled-exporter countertest observed
zero socket attempts.

## Known limitations

- External Sentry and LangSmith export is intentionally unavailable even with
  complete configuration until legacy request-reachable tracing and exception
  capture are audited and proven payload-safe.
- The restored route contract delegates fresh Manager/memory creation to the
  app-state stateless factory compatibility surface; the route does not perform
  a new dependency-memory rebinding.
- Stage 10B remains pending and owns versioned forecast persistence, route
  forecast fields, authentication-aware identity, progress/backtest jobs, UI,
  and E2E integration.
- Live providers and external services were not exercised.

## Rollback

No database rollback is required. A source rollback would restore the eight
Attempt 2 source/test paths listed above to their Attempt 1 workspace contents
while preserving all immutable workflow artifacts and accepted Stage 00–09
work. Do not roll back by changing Stage 10B, applying migrations, or mutating
external data.
