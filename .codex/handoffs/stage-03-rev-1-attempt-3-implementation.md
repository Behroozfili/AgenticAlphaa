# Stage 03 revision 1 attempt 3 — Temporal Vector/RAG implementation

- Primary agent: `data_rag`
- Phase: `PRIMARY_FINAL`
- Status: `IMPLEMENTED`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`
- Supersedes: `.codex/handoffs/stage-03-rev-1-attempt-2-implementation.json`
- Draft-ready: `.codex/handoffs/stage-03-rev-1-attempt-3-draft-ready.json`
- Observability: `.codex/handoffs/stage-03-rev-1-attempt-3-observability.json` (`AUDITED`, no requested primary changes)

## Outcome

Attempt 3 is the cumulative Stage 03 implementation candidate. It preserves the accepted historical loader/quarantine behavior, explicit versioned embedding cache, legacy-default/V2-shadow Vector adapter, pre-ranking temporal retrieval filters, fixture compatibility, sanitized lifecycle telemetry, explicit seed-only startup, and all prior Stage 03 leakage protections.

It resolves all Attempt 2 reviewer blockers:

- `STAGE03-HISTORICAL-INGESTION-PROVIDER-002`: public ingestion accepts an explicit reproducible archive provider and stable identity; the CLI supplies a content-addressed local JSON archive. Complete historical bounds and V2 are mandatory, and live fetch remains disabled.
- `STAGE03-CHECKPOINT-SNAPSHOT-IDENTITY-003`: checkpoints bind normalized boundaries, ticker/dataset/mode, provider hash, Vector adapter, embedding manifest, and ordered document/evidence/chunk identities. Resume may reuse completed Vector work or retry Graph only after exact re-materialization; missing or changed identity fails closed.
- `STAGE03-REVISION-EVIDENCE-PRESERVATION-004`: processor deduplication suppresses exact immutable evidence only. Same-content document revisions, dataset snapshots, retrieval observations, and effective-time observations coexist with distinct evidence/chunk identities.
- `STAGE03-OBSERVABILITY-ATTEMPT-IDENTITY-005`: telemetry provenance is explicitly `stage03-r1-a3`, while compact allowlisted lifecycle metadata, monotonic latency, and deterministic sanitized errors remain intact.

`OBS-03-001`, `OBS-03-002`, and `STAGE03-FIXTURE-COLLECTION-001` remain resolved. Observability independently audited the candidate and requested no changes.

## Cumulative Stage 03 paths

Product and runtime:

- `rag/__init__.py`
- `rag/loader.py`
- `rag/processor.py`
- `rag/embedding_manager.py`
- `rag/vector_store.py`
- `rag/retriever.py`
- `rag/ingestion.py`
- `rag/seed.py`
- `main.py`

Fixtures and tests:

- `tests/fixtures/stage3_ticker_regressions.json`
- `tests/unit_tests/test_rag/test_embedding_manager.py`
- `tests/unit_tests/test_rag/test_ingestion.py`
- `tests/unit_tests/test_rag/test_loader.py`
- `tests/unit_tests/test_rag/test_processor.py`
- `tests/unit_tests/test_rag/test_retriever.py`
- `tests/unit_tests/test_rag/test_vectore_store.py`
- `tests/unit_tests/test_rag/test_pipline_rag.py`
- `tests/unit_tests/test_root_main.py`

Attempt 3 artifact chain:

- `.codex/handoffs/stage-03-rev-1-attempt-3-draft-ready.json`
- `.codex/handoffs/stage-03-rev-1-attempt-3-observability.json`
- `.codex/handoffs/stage-03-rev-1-attempt-3-implementation.md`
- `.codex/handoffs/stage-03-rev-1-attempt-3-implementation.json`

## Contract diff

- Historical acquisition requires `start`, `end`, `as_of`, a reproducible archive provider, a stable provider identity, and the explicit V2 adapter. No historical live-web fallback exists.
- The configured seed boundary uses a local JSON archive whose provider identity is derived from the archive bytes. The raw provider identity/path and archive contents are not persisted or emitted.
- The request fingerprint covers normalized bounds, ordered tickers, dataset/mode, provider hash, adapter schema/table/RPC, and the attempt-versioned control identity.
- The materialization fingerprint additionally covers embedding model/version/cache schema and ordered canonical document/evidence/chunk identity hashes.
- Checkpoints persist safe hashes, counts, normalized boundaries, independent Vector/Graph status, and sanitized error codes only. They never persist raw documents, chunks, URLs, provider payloads, secrets, or exception text.
- Graph resume re-materializes and compares the exact fingerprint. Provider, content, model/cache, or adapter change fails closed before completed work can be mixed across Vector and Graph.
- Document identity preserves source/content/document version. Evidence identity additionally preserves dataset and all publication/retrieval/effective/filing observation times; chunk identity is evidence-scoped.
- Legacy Vector remains the normal default. V2 remains opt-in and historical mode fails closed unless the adapter can enforce temporal boundaries.
- Retrieval continues to enforce ticker, dataset version, `as_of`, retrieval/publication availability, and effective interval before ranking.
- Normal API startup does not seed. Ingestion remains an explicit CLI/API operation only.

## Verification

- Cumulative Python compilation: PASS.
- Focused Stage 03 suite: `216 passed in 94.54s`.
- Exact reviewer archive/provider/content/model/adapter counterexample matrix: `8 passed in 13.88s`.
- Narrative stability plus accepted Stage 01–02 regressions: `125 passed in 3.00s` (`16 + 109`).
- Full unit/integration collection: `1231 tests collected in 15.42s`; zero collection errors.
- Draft retry source hashes exactly match the observability audit.
- All recorded Attempt 1 and Attempt 2 artifact hashes match; neither prior attempt was modified.
- Cumulative scoped `git diff --check`: PASS apart from line-ending notices; refactor control: `CONTROL VALID`.
- No network, external database read/write, migration application, reconciliation, seed/backfill, production route switch, commit, push, or deployment occurred.

## Leakage and observability

Historical live fetching is disabled; invalid/unknown historical time remains quarantined; late-retrieved, future-published, wrong-ticker, wrong-dataset, and out-of-effective-interval evidence remains ineligible before ranking; and legacy historical retrieval fails closed.

Telemetry contains only allowlisted stage/revision/attempt, hashed run/provider/materialization identity, normalized version/adapter identity, monotonic latency, compact counts, stage status, and degradation/error categories. Raw evidence and untrusted exception strings are excluded from checkpoints, breadcrumbs, and capture payloads.

## Limitations

- V2 is validated offline through deterministic fixtures and mocks because Stage 02 migrations were not applied or reconciled.
- Historical acquisition currently requires a caller-configured reproducible local archive provider; no remote archive integration was executed.
- Temporal Graph observation modeling remains Stage 04A scope.
- No production latency benchmark or external tracing call was run.
- The pre-existing unrelated `agents/state.py` lock-attribution mismatch remains outside Stage 03 and was not modified or claimed.

## Rollback

No operational database rollback is required because no migration, external write, seed/backfill, reconciliation, legacy deletion, or route switch occurred. Keep the legacy Vector adapter as the runtime default.

If Attempt 3 retry rollback is separately authorized, restore only its six retry product/test paths to the immutable Attempt 2 bytes and remove only Attempt 3 primary draft/final artifacts; preserve the Attempt 3 observability audit and every Attempt 1/2 artifact. If the entire cumulative Stage 03 workspace change is separately rolled back, restore only the cumulative product/test paths listed above to the pre-Stage-03 baseline. Preserve database migrations/data, state history, and unrelated shared-workspace edits. Rollback is documented but was not executed or tested.
