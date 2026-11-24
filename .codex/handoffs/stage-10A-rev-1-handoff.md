# Stage 10A Revision 1 Handoff

Status: **CLOSED**  
Accepted attempt: **3**  
Primary agent: **core**  
QA: **PASS**  
Review: **APPROVED**

## Delivered contract

- The API/runtime foundation now has versioned LIVE/HISTORICAL forecast, backtest and runtime settings, safe temporal/version exception mapping, registered route skeletons and request/run correlation.
- Every request resolves a fresh Manager and ManagerMemory; mutable process-wide analysis state is not reused.
- Monthly memory is isolated by the exact key `(ticker, UTC as_of, horizon, analysis_mode, schema_version)` and cannot fall back to legacy/monthly merged results.
- Root startup has no implicit RAG seeding. Seeding is available only through the explicit `python main.py seed-rag` command.
- Optional Sentry and LangSmith exporters fail closed. Even hostile combinations of legacy environment flags and credentials cannot initialize an SDK, create a LangSmith client or open a socket; compact local lifecycle telemetry remains available.
- The accepted runtime provenance is immutable at `10A/r1/a3/core` with implementation version `stage10a-r1-a3`.
- The pre-existing analyze route contract is preserved: Manager resolution, injected memory preferences and status propagation remain unchanged; only privacy-safe logging, error redaction and safe lifecycle events were added.

## Acceptance evidence

- Hostile legacy-environment matrix: 32 of 32 cases passed, with 0 external attempts.
- Focused guarded tests: 246 passed, 0 failed.
- Analyze route tests: 22 passed, 0 failed.
- Stage 01/08/09 regression set: 140 passed, 0 failed.
- Forced-offline full collection: 1478 collected, 0 errors.
- Current and prior workflow/source hashes matched; Attempt 1 and Attempt 2 artifacts remained immutable.
- QA passed and independent Review approved with no blocking, scope or ownership findings.

## Retry history

- Attempt 1 reached Review and was rejected for privacy/error leakage and optional-exporter fail-closed gaps.
- Attempt 2 restored most constraints but Review rejected an out-of-scope analyze-route behavior change and stale runtime provenance.
- Attempt 3 restored the existing route semantics, advanced provenance to the accepted attempt, closed the hostile-environment gap, passed QA and was approved by Review.

## Rollback

Rollback is local and reversible: restore only the nineteen Stage 10A source/test paths recorded by this closure and remove only the Stage 10A workflow artifacts created by revision 1. Do not modify the accepted Stage 09 handoff, do not seed data, and do not activate Stage 10B during rollback.

## Known limitations and safety

- Acceptance is deterministic forced-offline engineering evidence. Live providers, external databases and external trace export were not exercised.
- A non-forced-offline Attempt 2 collection emitted a Hugging Face Hub authentication warning; actual network contact was not verified, so that run is historical only and is excluded from acceptance evidence.
- External exporters intentionally remain fail-closed until their request paths receive a separately authorized privacy audit.
- `OBS-10A-004` remains non-blocking: downstream integration may add richer lifecycle coverage without changing the accepted runtime boundary.
- No network acquisition, external write, migration, seed, deletion, deployment, commit or push was performed.
- Stage 10B remains `PENDING / NOT_STARTED` and is not activated by this closure.

Created at: `2026-09-13T22:55:56.1552567Z`
