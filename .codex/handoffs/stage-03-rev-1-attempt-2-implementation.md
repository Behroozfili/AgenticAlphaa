# Stage 03 revision 1 attempt 2 — Temporal Vector/RAG implementation

- Primary agent: `data_rag`
- Phase: `PRIMARY_FINAL`
- Status: `IMPLEMENTED`; ready for QA
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`
- Supersedes: `.codex/handoffs/stage-03-rev-1-attempt-1-implementation.json`

## Outcome

Attempt 2 is the cumulative Stage 03 candidate. It preserves Attempt 1's point-in-time-safe loader, revision-preserving chunk identities, versioned embedding cache, legacy/V2 Vector adapter, pre-ranking temporal filters, checkpointed ingestion, sanitized lifecycle telemetry, explicit seed CLI, and removal of normal-startup auto-seeding.

The fixture-only retry resolves `STAGE03-FIXTURE-COLLECTION-001`: `tests/fixtures/stage3_ticker_regressions.json` again contains the exact legacy top-level `cases` array required by narrative-stability consumers while retaining the new temporal `rows`, dataset/as-of metadata, and expected leakage result. Repository collection is restored without modifying the narrative test or any product source in Attempt 2.

## Cumulative changed product and test paths

All paths are locked to `data_rag` for Stage 03:

- `rag/__init__.py`
- `rag/loader.py`
- `rag/processor.py`
- `rag/embedding_manager.py`
- `rag/vector_store.py`
- `rag/retriever.py`
- `rag/ingestion.py`
- `rag/seed.py`
- `main.py`
- `tests/fixtures/stage3_ticker_regressions.json`
- `tests/unit_tests/test_rag/test_embedding_manager.py`
- `tests/unit_tests/test_rag/test_ingestion.py`
- `tests/unit_tests/test_rag/test_loader.py`
- `tests/unit_tests/test_rag/test_processor.py`
- `tests/unit_tests/test_rag/test_retriever.py`
- `tests/unit_tests/test_rag/test_vectore_store.py`
- `tests/unit_tests/test_rag/test_pipline_rag.py`
- `tests/unit_tests/test_root_main.py`

Attempt 2 additionally owns its draft and final implementation artifacts. Attempt 1 artifacts are immutable and their recorded SHA-256 values were revalidated before finalization.

## Contract and behavior

- Historical loading requires `start`, `end`, and `as_of` plus a reproducible archive provider; no live fallback is allowed.
- Invalid or unknown required historical times are quarantined. Publication, retrieval, and effective times remain distinct.
- Canonical source, immutable document/evidence/chunk IDs, dataset versions, and revisions are retained instead of overwritten.
- Embedding cache keys include content/chunk hash and explicit model version; the manifest records model, version, dimension, and cache schema.
- `RAG_VECTOR_SCHEMA` defaults to `legacy`; V2 remains an explicit shadow adapter and is not required at normal startup.
- Historical retrieval requires ticker, `as_of`, and dataset version at the V2 RPC boundary, rechecks eligibility before ranking, and fails closed on legacy.
- Ingestion uses deterministic run IDs and atomic checkpoints with separate Vector/Graph status and resumable, idempotent behavior.
- Lifecycle telemetry is allowlisted and monotonic, and persists/emits only sanitized error codes. `OBS-03-001` and `OBS-03-002` remain resolved.
- Seeding is an explicit CLI action; normal API startup does not seed.
- The shared regression fixture now supports both its legacy narrative `cases` contract and the temporal leakage `rows` contract.
- No database schema, migration, or production route was changed or applied by Stage 03.

## Verification

- Runtime `py_compile`: PASS.
- Focused Stage 03 suite: `203 passed in 158.75s`.
- Narrative stability: `16 passed in 3.98s`.
- Focused Stage 01–02 regression: `109 passed in 2.53s`.
- Full unit/integration collection: `1218 tests collected in 25.21s`; zero errors.
- Exact legacy `cases` equality plus temporal `rows` coverage: PASS.
- Attempt 1 artifact SHA-256 immutability: PASS.
- `git diff --check`: PASS; line-ending notices only.
- Refactor control: `CONTROL VALID`.

Leakage coverage passes for future publication, wrong ticker, wrong dataset version, late retrieval, unknown required historical time, and legacy-backend fail-closed behavior before ranking. No network, external database read/write, migration, reconciliation, seed/backfill, route switch, deployment, commit, or push occurred.

## Observability

Attempt 2 observability is `AUDITED` with no requested primary changes. Its audit verifies that fixture restoration introduced no new privacy surface and that the existing allowlisted lifecycle latency/status metadata and sanitized exception sentinel protections remain intact.

## Limitations

- V2 is validated with deterministic fixtures and mocks because Stage 02 migrations have not been applied or reconciled.
- Historical acquisition requires a caller-provided reproducible archive provider.
- Temporal Graph evidence modeling remains Stage 04A scope.
- Telemetry uses the existing breadcrumb backend and is a no-op when that backend is disabled.
- No external production latency benchmark was run.
- The pre-existing unrelated `agents/state.py` lock-attribution mismatch remains outside Stage 03 and was not modified or claimed.

## Rollback

No operational database rollback is required because no external state or production route changed. The legacy Vector adapter remains the runtime default.

If workspace rollback is separately authorized, restore only the cumulative Stage 03 product/test paths above to their pre-Stage-03 content and remove only Attempt 2 primary artifacts. Preserve Attempt 1 artifacts, both observability artifacts, Stage 00–02 artifacts, migrations, legacy/V2 data, state history, and unrelated shared-workspace edits. Rollback is documented but was not executed or tested.
