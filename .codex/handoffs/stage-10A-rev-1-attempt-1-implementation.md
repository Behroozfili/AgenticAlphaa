# Stage 10A revision 1 attempt 1 implementation report

- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `167`
- Runtime identity: `stage10a-r1-a1` (`stage=10A`, `revision=1`, `attempt=1`, `agent=core`)
- Stage 10B: `PENDING / NOT_STARTED`

## Outcome

Stage 10A now provides the API/runtime foundation without a mutable cross-request Manager or memory singleton. FastAPI retains only lifecycle connection clients, immutable runtime settings and a stateless `ManagerAgentFactory`; the actual `ManagerAgent` and `ManagerMemory` are constructed for each request. The existing `ManagerAgent` constructor/run interface and existing analyze/progress route response contracts remain compatible.

Settings distinguish LIVE and HISTORICAL_BACKTEST dataset versions, reuse the canonical 21-trading-session `ForecastContext`, and carry explicit runtime/schema/dataset/feature/model/backtest versions. Sentry and LangSmith are disabled by default and require global opt-in, exporter opt-in and a credential. Initializer failures are non-fatal and payload-free.

Monthly memory uses exactly `(ticker, as_of, horizon, analysis_mode, schema_version)` with timezone-aware UTC normalization. It is isolated from legacy ticker memory and does not introduce Stage-10B persistence. Normal startup never seeds. Seeding remains an explicit `python main.py seed-rag` command and was not executed.

## Observability and privacy resolution

`OBS-10A-001` is resolved. The observability-owned safe helpers are wired at startup/shutdown, request started/completed/failed, Manager factory created/failed and Manager memory created/failed boundaries. Every event carries immutable `10A/r1/a1/core` and `stage10a-r1-a1` identity. Exactly one opaque 32-hex run ID is assigned per request and reused by request, memory and Manager events; concurrent requests receive distinct IDs. Terminal and creation events use finite non-negative latency measured from `time.perf_counter`. An absent or raising local sink cannot change startup, shutdown, request, Manager or memory behavior.

`OBS-10A-002` is resolved, including the explicitly authorized privacy-only route call-sites. Registered logs no longer contain query, user/session identity, preference/memory values, Manager state or exception text/stack. Raw Sentry exception/breadcrumb capture was removed. API errors and persisted failures use stable payload-free codes. The existing analyses row shape is preserved while raw user ID is replaced with a deterministic opaque digest, query with `[redacted]`, and raw final report/metrics/sentiment/execution history/orchestrator logs with their empty type-compatible representations. Public request/response schemas and user-visible result mapping are unchanged.

`OBS-10A-003` is resolved jointly with the observability-owned initializer repair. API call-sites pass the three explicit gates and Settings-provided credentials to the repaired initializers, isolate unexpected initializer failure, and never put credentials or failure text in logs/events. Tests construct no external telemetry client and contact no exporter.

The non-blocking optional seed selection event from `OBS-10A-004` was not added. The explicit seed command and no-auto-seed behavior remain covered without expanding runtime telemetry scope.

Support-owned files were consumed without modification by Core:

- `langsmith_trace.py`: `e348fae4ac31277fa8c1a2483a225797eb13dc3969b0d47c9dfbfad45043b691`
- `core/observability.py`: `414b6b661c85138c17e81109e5dabeafba083767019fa9117fd68aa8fe6d6e88`
- Observability artifact: `6952d6790995f426f5ace00dd323b2b96c317d7b35bd633cb74bc2b75816353c`
- Change request 1: `5293a8b273d7330ca11af9491cf9402a22a5a56ea630f8e1f5646805447a05f5`
- Change request 2: `fa5a7f59fb8553138e72e691061fbb6418ad7abdd875b65d9424ddbdacca7895`

## Verification

- Final focused Stage 10A plus observability suite: `206 passed in 27.10s`.
- Direct telemetry/privacy call-site suite: `126 passed in 20.87s`.
- Proportional API, Memory, Forecast, Manager/Stage-08, Core observability and injected validation regressions: `355 passed in 10.50s`.
- Full repository collection: `1438 tests collected in 27.53s`, zero collection errors.
- Compilation of all changed source and focused test modules: PASS.
- Static privacy scan found no raw exception capture/logging; the only matched preference log is the fixed literal `long_term_preference_stored`.
- Scoped `git diff --check`: PASS.
- Official control/lock validation: `CONTROL VALID` at control revision 167 before final artifacts.

## Limitations and stage boundary

Stage 10B still owns request forecast fields, business routing expansion, versioned forecast persistence, progress/backtest jobs, UI and E2E behavior. The Stage-10A route edit is limited to request-safe factory selection, payload-free logs/errors and type-compatible redaction of the existing persistence row. No forecast, RAG, Research, Financial, Sentiment or Quant formula changed.

No live provider, database, exporter, network acquisition, migration, seed, deletion, deployment, dependency change, commit or push was executed.

## Rollback

No database or external rollback is required. Retain all immutable attempt artifacts. Restore the fifteen Core source/test paths to their pre-Stage-10A workspace versions, remove only the Stage-10A final implementation MD/JSON after archival, and retain the observability-owned files/artifacts under their recorded hashes. Do not modify the accepted Stage-09 handoff or activate Stage 10B. No rollback action was executed.
