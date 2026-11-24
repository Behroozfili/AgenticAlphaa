# Stage 03 revision 1 attempt 1 — Temporal Vector/RAG implementation

- Stage: `03`
- Revision: `1`
- Attempt: `1`
- Primary agent: `data_rag`
- Phase: `PRIMARY_FINAL`
- Status: `IMPLEMENTED`; ready for QA
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

Stage 03 now provides a point-in-time-safe Vector/RAG data path while retaining the usable legacy runtime default. Historical loading requires explicit `start`, `end`, and `as_of` boundaries plus an injected reproducible archive provider; it never falls back to live web sources. Invalid or unknown required historical times are quarantined and excluded. Publication, retrieval, and effective intervals remain separate, and immutable document/evidence/chunk identities preserve revisions instead of overwriting them.

The Vector store has an explicit `legacy`/`v2` adapter boundary. `legacy` remains the default because Stage 02 migrations have not been applied or reconciled. V2 queries require ticker, `as_of`, and dataset version at the RPC boundary and defensively recheck ticker/version/availability/effective interval before any freshness ranking. Historical queries against the legacy backend fail closed.

Embedding cache identity now includes the content/chunk hash and explicit model version, with model, version, dimension, and cache-schema metadata. Ingestion uses deterministic run IDs and atomic local checkpoints with separate Vector/Graph status, allowing Graph retry without repeating a completed Vector write. Seeding is an explicit CLI operation with temporal/version/checkpoint options; normal API startup no longer auto-seeds.

## Observability findings resolved

- `OBS-03-001`: ingestion now emits compact allowlisted start/complete/failure/resume/skip/checkpoint lifecycle metadata. Each event includes monotonic `latency_ms`, Stage 03 revision/attempt/agent identity, run/dataset/ticker/as-of context when valid, Vector adapter identity, available embedding versions, counts, explicit status, and degraded state. Raw documents, chunks, URLs, queries, prompts, provider payloads, credentials, and headers are not accepted telemetry fields.
- `OBS-03-002`: untrusted exception strings are no longer persisted or emitted. Deterministic `INGESTION_<STAGE>_<CATEGORY>` codes are used in manifests, checkpoint JSON, CLI-visible output, breadcrumbs, and sanitized Sentry exceptions. Previously persisted non-code error text is sanitized during checkpoint materialization. Sentinel regressions prove secret/raw-payload/URL fragments do not reach checkpoints, telemetry, or captured exceptions.

The accepted observability artifact is `.codex/handoffs/stage-03-rev-1-attempt-1-observability.json` with status `AUDITED`.

## Changed files

All primary-owned paths below are locked to `data_rag`:

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
- `.codex/handoffs/stage-03-rev-1-attempt-1-draft-ready.json`
- `.codex/handoffs/stage-03-rev-1-attempt-1-implementation.md`
- `.codex/handoffs/stage-03-rev-1-attempt-1-implementation.json`

## Self-tests and evidence

- `python -m py_compile` over all nine locked Stage 03 runtime paths: PASS.
- Final focused offline Stage 03 suite: `203 passed in 51.09s`; no failures.
- Focused ingestion telemetry/privacy run before final hardening: `18 passed in 20.35s`; the final suite includes these tests plus legacy-checkpoint sanitization.
- `git diff --check`: PASS.
- `python .codex/scripts/validate_refactor_control.py --check-control`: `CONTROL VALID`.
- Leakage fixture: PASS for future publication, wrong ticker, wrong dataset version, late retrieval, unknown historical time, and legacy-backend fail-closed behavior before ranking.
- Resume/idempotency: PASS for completed-run skip and Graph failure/retry without repeating completed Vector upsert.
- Privacy sentinel: PASS; sentinel secret, raw-payload text, and private URL fragments were absent from checkpoint JSON, lifecycle telemetry, and captured sanitized exception material.

No network, external database read/write, migration, reconciliation, seed/backfill, production pointer switch, deployment, commit, or push was performed.

## Contract/schema diff

- `RawDocument` adds distinct retrieval/effective times, dataset version, canonical source identity, and revision while retaining legacy constructor compatibility.
- Processed metadata adds shared immutable `document_id`, `evidence_id`, and `chunk_id`, content/chunk hashes, source/document versions, temporal fields, and dataset version.
- Embedding records add explicit model/version/dimension/cache key and expose an embedding manifest.
- `VectorStoreConfig` selects `legacy` or `v2`; default remains `legacy`. V2 maps to the additive Stage 02 table/RPC without requiring that table at default startup.
- Retriever historical mode requires ticker/`as_of`/dataset version and anchors freshness to `as_of`.
- Ingestion returns a versioned manifest with separate Vector/Graph stage status and sanitized error codes; checkpoint format is atomic and resumable.
- Seed becomes an explicit CLI. Root startup starts only the API server.
- No database schema was changed or applied in Stage 03.

## Known limitations and out-of-scope findings

- V2 behavior is validated with deterministic mocks and fixtures because Stage 02 migrations are not applied or reconciled.
- Historical acquisition requires a caller-supplied reproducible archive provider; live fallback is intentionally forbidden.
- Temporal Graph evidence observations remain Stage 04A scope; Stage 03 does not modify `rag/graph_store.py` or `rag/hybrid_rag.py`.
- Lifecycle telemetry is emitted through the existing breadcrumb path and is a no-op when the configured observability backend is disabled; checkpoint state remains available locally when a checkpoint path is supplied.
- No external performance benchmark was run; `latency_ms` uses monotonic in-process timing and tests verify presence/type/non-negativity rather than production latency targets.
- The global lock validator's pre-existing unrelated `agents/state.py` mismatch remains outside Stage 03 and was not modified or attributed here.

## Rollback

No operational database rollback is needed because migrations, seeds, external writes, backfills, and route switches were not executed. The runtime continues to default to the legacy Vector adapter.

If workspace rollback is separately authorized, restore only the Stage 03 product/test paths listed above to their pre-Stage-03 content and remove only the Stage 03 attempt-1 primary artifacts. Preserve Stage 00–02 artifacts, the observability artifact, legacy/V2 data, migrations, state history, and unrelated user/shared-workspace edits. Rollback was documented but not executed or tested.
