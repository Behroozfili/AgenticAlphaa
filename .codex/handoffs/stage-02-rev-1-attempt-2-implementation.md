# Stage 02 revision 1 attempt 2 — Database V2 implementation

Status: `IMPLEMENTED` in `PRIMARY_FINAL`; ready for QA. This attempt supersedes Attempt 1 and resolves `STAGE02-RLS-VIEW-001`.

## QA finding resolved

The eligible-document and reconciliation views now explicitly use PostgreSQL `security_invoker` semantics. Supabase read paths receive explicit least-privilege `SELECT` grants for `authenticated` and `service_role` as appropriate; no write or broad `ALL` grant is introduced. The temporal RPC remains `SECURITY INVOKER`, row-level policies remain enabled, quarantined documents remain unavailable to authenticated readers, and production routes remain `legacy-v1`.

The frozen keys, separate temporal fields, unknown-time quarantine, pre-ranking ticker/as-of/dataset/effective filtering, immutable forecasts/outcomes, reproducible backtest identity, additive analysis columns, reconciliation views, rollback pointers, and Neo4j temporal identities from Attempt 1 are preserved.

## Verification

- Forced-offline Stage 01 + Stage 02 focused regression: 107 passed.
- Compile, control, ownership, locks, scoped diff and destructive-statement gates: PASS.
- `SCHEMA_CONTRACT_FROZEN` Attempt 2 milestone: PASS.

## Safety and rollback

No Supabase/Neo4j migration, seed, backfill, reconciliation execution, route switch, database mutation, external acquisition, deploy, commit, or push occurred. Live empty-database/staging evidence remains unclaimed because it requires separate authorization.

Operational rollback restores only the previous route pointer and preserves V2/legacy data. Workspace rollback may restore the migration/test paths to immutable Attempt 1 content and remove only Attempt 2 artifacts when explicitly authorized.

