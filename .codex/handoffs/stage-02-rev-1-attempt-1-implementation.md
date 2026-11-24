# Stage 02 revision 1 attempt 1 — Database V2 implementation

Status: `IMPLEMENTED` in `PRIMARY_FINAL`; ready for QA. Stage 03 remains `PENDING/NOT_STARTED`.

## Delivered

- `database/supabase/001_alpha_documents_v2.sql` adds the shadow document/chunk store, immutable Vector/Graph identities, separate evidence timestamps, quarantine rules, temporal indexes, RLS, the fail-closed `alpha_hybrid_search_v2` RPC, and reconciliation output.
- `database/supabase/002_forecast_tables.sql` adds immutable feature snapshots, monthly forecasts and outcomes plus reproducible backtest runs using the frozen natural keys.
- `database/supabase/003_analysis_versioning.sql` preserves the legacy `analyses` shape while adding nullable point-in-time/version columns, V2 views, reconciliation, and guarded route activation/rollback functions.
- `database/neo4j/001_temporal_evidence.cypher` adds immutable document/chunk/evidence constraints, temporal indexes, and a route that remains on `legacy-v1`.
- `tests/unit_tests/test_database_v2_migrations.py` statically gates additive-only behavior, frozen keys, temporal filtering before ranking, unknown-time quarantine, RLS, reconciliation, rollback pointers, and shared Vector/Graph identities.

No legacy relation, label, edge, constraint, index, row, or file is removed. Unknown evidence time remains null and quarantined; it is never converted to current time. Production routes remain `legacy-v1`.

## Verification

- Stage 02 focused contract gate: 11 passed.
- Forced-offline Stage 01 + Stage 02 regression: 106 passed.
- Forced-offline collection: 1,198 tests (1,147 unit and 51 integration).
- Python compile, scoped diff check, destructive-statement scan, control and locks: PASS.
- `SCHEMA_CONTRACT_FROZEN` milestone: PASS.

## Safety, limitations, and rollback

No migration was applied to Supabase or Neo4j. No seed, backfill, reconciliation run, route switch, database mutation, external acquisition, deploy, commit, or push occurred. Empty-database and staging execution remain `NOT_RUN_REQUIRES_SEPARATE_AUTHORIZATION`; therefore no live uniqueness/RLS/RPC/count claim is made.

Before any future route switch, an authorized environment must apply the versioned files, execute the reconciliation views, record a SHA-256 reconciliation result, and explicitly authorize the pointer change. Operational rollback restores the previous route pointer and preserves all V2 and legacy rows. Workspace rollback removes only the four new migration files, the Stage 02 test, and Stage 02 artifacts when explicitly authorized; it preserves Stage 00/01 evidence and user data.

