# Stage 04A Revision 1 Attempt 2 — Implementation

Status: `IMPLEMENTED`  
Phase: `PRIMARY_FINAL`  
Owner: `data_rag`

## Outcome

Attempt 2 resolves all three Stage 04A review findings and the accepted Observability finding `OBS-04A-004`. Temporal V2 graph writes now validate and normalize provenance before writing, use Neo4j temporal values, retain canonical `materialized_at`, and fail closed for incomplete or quarantined inputs. Vector, Graph, RRF and Hybrid trace/lifecycle metadata now consistently reports Stage `04A`, revision `1`, attempt `2`, agent `data_rag` through the existing compact allowlist.

## Resolved findings

- `STAGE04A-NEO4J-TEMPORAL-PROPERTY-TYPE-001`: all supplied temporal values are normalized once; Cypher persists null-safe `datetime($parameter)` properties and never substitutes the current time.
- `STAGE04A-INCOMPLETE-V2-WRITE-FAIL-CLOSED-002`: only `write_mode=legacy` with `provenance=legacy_explicit` may use the legacy aggregate path. Incomplete, invalid, unknown-provenance or quarantined temporal inputs make zero entity/relation writes and expose stable compact rejection codes/counts.
- `STAGE04A-EVIDENCE-MATERIALIZED-AT-003`: canonical Stage 03 `materialized_at` is required for temporal V2, validated, persisted on create and unchanged on idempotent retry.
- `OBS-04A-004`: `_TRACE_IDENTITY.attempt` is `2`; an offline sentinel test proves Vector, Graph, RRF/Hybrid lifecycle and Hybrid processed metadata share the active identity without capturing query text, URLs, evidence content or secrets.

The accepted `OBS-04A-001..003` redaction, measured latency/count telemetry and sanitized stable-error behavior remain intact.

## Files

Changed implementation/test scope:

- `rag/graph_store.py`
- `rag/hybrid_rag.py`
- `tests/unit_tests/test_rag/test_graph_store.py`
- `tests/unit_tests/test_rag/test_hybried.py`
- `tests/fixtures/stage4a_temporal_graph_hybrid.json`

Revalidated unchanged:

- `rag/__init__.py`

Final artifacts:

- `.codex/handoffs/stage-04A-rev-1-attempt-2-implementation.md`
- `.codex/handoffs/stage-04A-rev-1-attempt-2-implementation.json`

## Verification

- Attempt 2 review blocker matrix: `27 passed in 16.24s`.
- New `OBS-04A-004` offline assertion: `1 passed in 42.76s`.
- Full Graph/Hybrid suite: `93 passed in 17.90s`.
- Full RAG plus Stage 03 regression: `349 passed in 58.00s`.
- Collection: `1293 tests collected in 20.51s`, zero collection errors.
- Scoped compile, diff, control/lock and privacy checks passed; control reported `CONTROL VALID`; raw-exception/source-URL static scan reported zero matches.

Temporal leakage and privacy checks passed. Future, unknown, invalid, quarantined, wrong-ticker and wrong-dataset observations remain ineligible. The new sentinel never appeared in processed trace or lifecycle output.

## Schema/data and rollback

No schema/migration file was changed or applied. No live database, trace provider, seed, deployment, commit, push or network operation was performed. External row counts remain unavailable because external reads were not authorized.

Rollback is source-only: restore the prior Attempt 1 Graph writer plus the Attempt 2 draft Hybrid identity/test bytes. No database rollback is necessary because this attempt performed no external write. Legacy namespaces/data and the default legacy-compatible path remain intact.

## Limitations

- Neo4j temporal typing is verified by normalized parameters and exact offline Cypher query-shape tests; no live Neo4j execution was authorized.
- Temporal V2 remains deterministic fixture/mock validated until migration and reconciliation are separately authorized.
- Stage 04B remains pending/not started and was not modified.

