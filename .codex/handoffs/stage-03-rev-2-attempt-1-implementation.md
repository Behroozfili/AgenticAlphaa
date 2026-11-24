# Stage 03 revision 2 attempt 1 — Temporal Vector/RAG

- Primary agent: `data_rag`
- Phase: `PRIMARY_FINAL`
- Status: `IMPLEMENTED`
- Control identity: `stage03-r2-a1`
- Draft-ready: `.codex/handoffs/stage-03-rev-2-attempt-1-draft-ready.json`
- Observability: `.codex/handoffs/stage-03-rev-2-attempt-1-observability.json` (`AUDITED`, no requested changes)

## Outcome

The revision 2 candidate resolves all remaining Stage 03 review blockers while preserving the temporal, privacy and legacy-default behavior from revision 1.

- The accepted Stage 02 revision 2 evidence-scoped schema is used as the V2 upsert conflict target.
- Historical archive rows outside the requested ticker set are independently quarantined by the loader.
- Ticker and complete metadata association participate in immutable evidence identity, preserving multi-ticker evidence.
- Resume fingerprints include full RawDocument and processed-record manifests, so ticker or metadata drift cannot mix completed Vector work with Graph retry.
- Missing-credential and transient load/process checkpoints may retry only when no completed Vector/Graph work exists. Completed work without exact identity remains fail-closed.
- Telemetry provenance is `stage03-r2-a1`; Observability found no privacy, latency, progress or lifecycle blocker.

Stage 04A remains `PENDING / NOT_STARTED` and was not inspected, modified or activated.

## Verification

- Exact revision 2 blocker matrix: `12 passed`.
- Complete Stage 03 owned suite: `228 passed` against final source bytes.
- Accepted Stage 01–02 regression suite: `112 passed` against final source bytes.
- Full unit/integration collection: `1246` tests, zero collection errors.
- Observability: `AUDITED`; corrected focused audit `8 passed`, snapshot fail-closed class `6 passed`.
- Compilation, scoped diff check, artifact/control/lock validation: PASS / `CONTROL VALID`.

## Migration status

The user explicitly authorized applying `database/supabase/004_alpha_documents_v2_evidence_identity.sql` to the configured remote development target. It was not executed because this workspace has no SQL connection URL, database password, Supabase management token, linked CLI session, `supabase` executable or `psql` executable. The REST service credentials cannot safely execute arbitrary migration SQL. No external operation was attempted and no live database claim is made.

The migration remains statically accepted and covered by the 17-test Stage 02 migration contract. Applying it later requires an actual SQL execution channel for the development target; this does not authorize another migration, production deployment, seed/backfill, route switch or Stage 04A activation.

## Leakage and operational safety

Historical live fallback remains disabled. Future, late-retrieved, unknown-time, wrong-ticker, wrong-dataset and out-of-effective-interval evidence is rejected or quarantined before ranking. Checkpoints and telemetry retain compact hashes, normalized identifiers, counts, statuses and sanitized error codes only—no raw evidence, secret, credential, provider payload or chain-of-thought.

No network acquisition, external database read/write, migration execution, reconciliation, seed/backfill, legacy deletion, production route switch, deployment, commit or push occurred.

## Limitations and rollback

V2 live database behavior remains offline fixture/mock validated until migration 004 is applied through a configured SQL channel. Temporal Graph implementation remains Stage 04A scope and inactive. `agents/state.py` remains under its separate core quarantine lock.

No operational database rollback is needed because no external mutation occurred. Keep the runtime on the legacy adapter. A workspace rollback must restore only Stage 03-owned product/test paths while preserving database migrations, control history, artifacts and unrelated edits.
