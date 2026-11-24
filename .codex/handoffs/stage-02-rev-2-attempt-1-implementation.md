# Stage 02 revision 2 attempt 1 — V2 evidence identity amendment

- Primary agent: `data_rag`
- Phase: `PRIMARY_DRAFT` (implementation complete; stop before QA)
- Status: `IMPLEMENTED`
- Milestone: `SCHEMA_CONTRACT_FROZEN` — `PASS`
- Supersedes implementation: `.codex/handoffs/stage-02-rev-1-attempt-3-implementation.json`
- Supersedes handoff: `.codex/handoffs/stage-02-rev-1-handoff.json`
- Trigger: `STAGE03-V2-EVIDENCE-COEXISTENCE-006`

## Outcome

The Stage 02 V2 contract now has a reversible, versioned constraint amendment at `database/supabase/004_alpha_documents_v2_evidence_identity.sql`. Accepted migrations 001–003 remain untouched and migration 004 was not applied.

Revision 1 made document/chunk uniqueness independent of evidence observation. Same-content rows with distinct dataset, retrieval, or effective observations therefore had distinct `chunk_id` and `evidence_id` values but collided on `alpha_documents_v2_document_identity`. It also treated `(evidence_id, dataset_version)` as row-unique, which prevented one evidence observation from retaining multiple chunks.

Migration 004 safely replaces those constraints with:

- the unchanged `chunk_id` primary key;
- `alpha_documents_v2_document_evidence_chunk_identity`, which preserves canonical source/content/document semantics while adding `document_id`, `evidence_id`, and `dataset_version` to the chunk identity; and
- `alpha_documents_v2_evidence_chunk_identity`, which scopes uniqueness to evidence, dataset, chunk position/hash, and embedding version.

Distinct immutable observations can coexist; exact evidence chunks remain conflict-safe. The V2 adapter continues to upsert on `chunk_id`, which is the immutable evidence-scoped primary identity.

## Forward safety and idempotency

The forward transaction:

- requires `alpha_documents_v2` to exist, enforcing migration order after 001;
- checks the exact definitions of both revision-1 constraints before dropping them;
- uses `DROP CONSTRAINT IF EXISTS` only for those two named constraints;
- adds each revision-2 constraint only when absent and rejects unexpected definitions when present;
- changes no rows, tables, columns, RLS policies, views, RPCs, grants, reconciliation queries, route pointers, or legacy data; and
- is safe to re-run against the expected fully migrated or already-amended schema.

## Rollback

Migration 004 includes explicit rollback SQL as a separately authorized comment block. Rollback first checks for rows that would collide under either revision-1 unique constraint. If any exist, rollback raises and leaves the evidence-preserving schema/data intact. Only after both checks pass does it drop the revision-2 constraints and restore the exact revision-1 definitions.

Rollback never deletes, rewrites, merges, or silently discards evidence. It was documented and statically tested but not executed.

## Preserved contracts

- `chunk_id` remains the V2 primary key and Vector upsert conflict target.
- `document_id`, `evidence_id`, canonical source, content/chunk hashes, document version, dataset version, and embedding version remain explicit identities.
- Unknown time remains quarantined and is never mapped to the current time.
- The V2 RPC still filters ticker, dataset, `as_of`, availability, lookback, quarantine eligibility, and effective interval before ranking.
- RLS, security-invoker views, SELECT/EXECUTE-only grants, ordered SHA-256 reconciliation, and legacy-v1 route defaults are unchanged.
- V2 remains a shadow schema. No production pointer switched and no migration was applied.

## Verification

- Focused Stage 02 migration contracts: `17 passed in 12.67s`.
- Focused Stage 01 + Stage 02 regressions: `112 passed in 15.77s`.
- Full unit/integration collection: `1234 tests collected in 14.73s`; zero collection errors.
- Python compilation: PASS.
- Scoped diff check: PASS; refactor control: `CONTROL VALID`.
- Offline coexistence regression created four same-content Stage 03 observations differing by dataset, retrieval, or effective time, processed them through the real processor, offline embedding cache, and `_to_v2_row`, and proved all revision-2 primary/unique identities are distinct while the revision-1 document tuple collapses to one.

## Cumulative Stage 02 paths

- `database/supabase/001_alpha_documents_v2.sql` — preserved revision-1 base
- `database/supabase/002_forecast_tables.sql` — preserved
- `database/supabase/003_analysis_versioning.sql` — preserved
- `database/supabase/004_alpha_documents_v2_evidence_identity.sql` — revision-2 amendment
- `database/neo4j/001_temporal_evidence.cypher` — preserved
- `tests/unit_tests/test_database_v2_migrations.py`
- `.codex/handoffs/stage-02-rev-2-attempt-1-milestone-SCHEMA_CONTRACT_FROZEN.json`
- `.codex/handoffs/stage-02-rev-2-attempt-1-implementation.md`
- `.codex/handoffs/stage-02-rev-2-attempt-1-implementation.json`

Revision 2 directly modified only migration 004, the migration contract test, and its locked primary artifacts.

## Limitations and safety boundary

- SQL was statically validated offline; no Supabase or Neo4j connection/application/reconciliation was authorized or run.
- Existing production databases require separately authorized migration application and reconciliation before V2 use.
- Stage 03 revision 1 is stale. This revision resolves only its cross-stage evidence-coexistence blocker; the separately reviewed ticker/materialization and pre-materialization-resume blockers remain Stage 03 rework.
- Production remains on legacy-v1.
- The missing legacy Phase 5 migration retains its Stage 00 disposition and is not replaced by migration 004.
- No external network/database read/write, seed/backfill, route switch, commit, push, or deployment occurred.
