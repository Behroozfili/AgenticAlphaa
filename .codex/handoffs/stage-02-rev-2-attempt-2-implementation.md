# Stage 02 revision 2 attempt 2 — V2 evidence identity amendment

- Primary agent: `data_rag`
- Phase: `PRIMARY_FINAL`
- Status: `IMPLEMENTED`
- Implementation ready: `true`
- Milestone: `.codex/handoffs/stage-02-rev-2-attempt-2-milestone-SCHEMA_CONTRACT_FROZEN.json` (`PASS`)
- Supersedes implementation: `.codex/handoffs/stage-02-rev-2-attempt-1-implementation.json`
- Supersedes handoff: `.codex/handoffs/stage-02-rev-1-handoff.json`

## Outcome

This is the cumulative finalized `PRIMARY_FINAL` handoff for Stage 02 revision 2. Attempt 2 corrects only the stale phase metadata that caused `STAGE02-REV2-FINAL-ARTIFACT-PHASE-001`; migration, product, and test bytes are unchanged from the technically accepted Attempt 1 candidate.

The revision resolves `STAGE03-V2-EVIDENCE-COEXISTENCE-006` through the versioned, reversible `database/supabase/004_alpha_documents_v2_evidence_identity.sql` amendment. Accepted migrations 001–003 remain immutable and migration 004 was not applied.

## Schema amendment

Revision 1 made document/chunk uniqueness independent of evidence observation. Same-content rows with distinct dataset, retrieval, or effective observations therefore had distinct evidence-scoped `chunk_id` and `evidence_id` values but collided on the old document identity constraint. The old `(evidence_id, dataset_version)` uniqueness also prevented a single observation from retaining multiple chunks.

Migration 004 preserves the `chunk_id` primary key and replaces the two overly restrictive constraints with:

- `alpha_documents_v2_document_evidence_chunk_identity`, covering canonical source, content hash, document version/ID, evidence ID, dataset, chunk position/hash, and embedding version; and
- `alpha_documents_v2_evidence_chunk_identity`, covering evidence ID, dataset, chunk position/hash, and embedding version.

Distinct dataset, retrieval, filing, or effective observations can coexist without overwrite, while exact evidence chunks remain conflict-safe. The existing V2 adapter continues to upsert on the immutable `chunk_id` primary key.

## Forward and rollback safety

The forward transaction enforces order after migration 001, checks the exact definitions of revision-1 and revision-2 constraints, changes only the two named unique constraints, is idempotent for the expected schema states, and fails on schema drift. It changes no rows, tables, columns, RLS policies, views, RPCs, grants, reconciliation logic, legacy data, or route pointers.

Migration 004 contains explicit separately authorized rollback SQL. Rollback first checks for rows that would collide under each revision-1 identity. It aborts without mutation when coexistence evidence would be lost; only a clean preflight permits restoring the old constraints. No rollback path deletes, rewrites, merges, or silently discards evidence. Rollback is documented and statically tested, but was not executed.

## Preserved contracts

- `chunk_id` remains the primary key and V2 upsert conflict target.
- Canonical source/document/evidence/chunk identities and versions remain explicit.
- Unknown time remains quarantined and is never mapped to the current time.
- The temporal RPC continues to prefilter ticker, dataset, `as_of`, availability/lookback, quarantine eligibility, and effective interval before ranking.
- RLS, security-invoker views, SELECT/EXECUTE-only grants, ordered reconciliation hashes, and legacy-v1 route defaults remain unchanged.
- V2 remains a shadow schema; production remains on legacy-v1.

## Unchanged technical evidence

Attempt 1 QA independently confirmed that the technical amendment passed every authorized offline gate and failed only because its final implementation artifact incorrectly said `PRIMARY_DRAFT`.

- Focused Stage 02 migration contracts: `17 passed`.
- Focused Stage 01 + Stage 02 regressions: `112 passed`.
- Full unit/integration collection: `1234 collected`; zero collection errors.
- Exact amendment acceptance matrix: `3 passed`.
- Migration 004 SHA-256: `fced2fc6fcca6ef6bda11b887746b03a1fa98d7eaaa4b717a5c7dfb7c646f481`.
- Migration-test SHA-256: `3f543f17f26d6ea4cf7afdcc7c069e50256f929491a012a6d3e269a180e43ece`.
- Both hashes were rechecked before finalization; product, migration, and test bytes remain unchanged.

The offline coexistence regression maps four same-content Stage 03 observations differing by dataset, retrieval, or effective time through the real processor, an offline cached embedder, and the V2 row mapper. All revision-2 primary/unique identities are distinct and non-null, while the replaced revision-1 document tuple collapses to one.

## Cumulative Stage 02 paths

- `database/supabase/001_alpha_documents_v2.sql` — preserved
- `database/supabase/002_forecast_tables.sql` — preserved
- `database/supabase/003_analysis_versioning.sql` — preserved
- `database/supabase/004_alpha_documents_v2_evidence_identity.sql` — revision-2 amendment
- `database/neo4j/001_temporal_evidence.cypher` — preserved
- `tests/unit_tests/test_database_v2_migrations.py` — unchanged from Attempt 1
- `.codex/handoffs/stage-02-rev-2-attempt-2-milestone-SCHEMA_CONTRACT_FROZEN.json`
- `.codex/handoffs/stage-02-rev-2-attempt-2-implementation.md`
- `.codex/handoffs/stage-02-rev-2-attempt-2-implementation.json`

Attempt 2 itself writes only its milestone and finalized implementation artifacts. Attempt 1 artifacts remain immutable.

## Limitations and operational safety

- Migration 004 is statically validated offline only. Application and live reconciliation require separate explicit authorization.
- Stage 03 revision 1 remains stale; its ticker/materialization and pre-materialization-resume findings require separate Stage 03 rework.
- The missing legacy Phase 5 migration retains its Stage 00 disposition and is not replaced by migration 004.
- No migration application, external database/network read or write, seed/backfill, reconciliation, route switch, legacy deletion, commit, push, deployment, or rollback execution occurred.

This `PRIMARY_FINAL` artifact is ready for the QA gate.
