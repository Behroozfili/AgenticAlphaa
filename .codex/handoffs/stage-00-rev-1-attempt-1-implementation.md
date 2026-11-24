# Stage 00 implementation report

- Stage: `00`
- Revision: `1`
- Attempt: `1`
- Primary agent: `core`
- Status: `IMPLEMENTED`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

The local Python/pytest runner is reproducible and Stage 00 baseline evidence is ready for QA. This does **not** claim a green unit suite: the executed offline baseline has 1,045 passes and 40 precisely classified pre-existing failures. No product behavior, migration, external database, data, seed, deployment, commit, or push was changed.

The obsolete MSFT trace was retired by explicit user decision and replaced with current local evidence. Supabase, Neo4j, backup, checksum, and restore evidence are `NOT_AVAILABLE`/`NOT_RUN`, never `PASS`. The missing legacy Supabase migration has no valid local or Git-history retarget; its four contract-test failures remain visible and the test was not edited.

## Changed files

- `.codex/baseline-manifest.json` — lock owner `core`; fresh baseline, dispositions, inventory, hashes, counts, and known failures.
- `pytest.ini` — lock owner `core`; default local unit path, live deselection, and registered `unit/integration/live/backtest/slow` markers.
- `.codex/handoffs/stage-00-rev-1-attempt-1-implementation.md` — lock owner `core`; this report.
- `.codex/handoffs/stage-00-rev-1-attempt-1-implementation.json` — lock owner `core`; machine-readable implementation evidence.
- `.codex/handoffs/stage-00-rev-1-attempt-1-milestone-BASELINE_READY.json` — lock owner `core`; milestone evidence.

The orchestrator-owned `.codex/refactor-state.json` activation diff and user-owned pre-existing untracked `dfc.md` are present in the working tree but were not changed by `core` and are excluded from core ownership.

## Self-tests and evidence

- `.venv/Scripts/python.exe --version` — exit 0; Python `3.12.13`.
- `.venv/Scripts/python.exe -m pytest --version` — exit 0; pytest `9.0.3`.
- `uv lock --check --offline` — exit 0; 210 packages resolved from the existing lock.
- Forced-offline full collection over `tests/unit_tests tests/integration_tests` — exit 0 in 15.76s; 1,085 unit + 51 integration = 1,136 collected.
- Forced-offline default collection — exit 0; 1,085 unit tests collected.
- Forced-offline unit baseline — exit 1 in 68.29s; 1,045 passed, 40 failed, 35 warnings.
- Focused legacy migration test — exit 1 in 0.90s; four failures, all `FileNotFoundError` for `supabase_phase5_provenance_migration.sql`.
- `.codex/baseline-manifest.json` parsed with `python -m json.tool` — exit 0.

Offline execution set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `DATASETS_OFFLINE=1`, `WANDB_DISABLED=true`, and `LANGCHAIN_TRACING_V2=false`. Integration tests were collected but not executed because the legacy suite contains real Yahoo, SEC, news, Supabase, Neo4j, and LLM calls.

Failure ownership is recorded in the manifest: Stage 01/core (2), Stage 03/data_rag (1), Stage 05/financial (28), Stage 06/sentiment (4), Stage 08/integration (1), and the legacy migration disposition (4). Stage 00 introduced zero product/test-assertion changes, so these are baseline failures rather than Stage 00 regressions.

## Contract/schema changes

No product contract, schema, SQL, Cypher, dependency declaration, or lock file changed. `pyproject.toml` remains the canonical direct-dependency source and `uv.lock` passed its offline consistency check. `requirements.txt` was inventoried but not regenerated.

## Known limitations and out-of-scope findings

- The missing legacy Phase 5 SQL remains a blocker for that legacy migration contract. Exact recovery or an explicit retirement decision is required; Stage 02 may not silently replace it.
- Actual Supabase schema/RPC/index/RLS/counts and Neo4j constraint/index/counts are unknown because no database is available and network is forbidden.
- No backup artifact, checksum, target version, native restore CLI, or restore test exists.
- Several legacy integration tests perform external I/O without a `live` marker. Default `testpaths` now excludes the integration directory; future exact-lock work should classify those tests individually.
- Historical point-in-time data does not exist. Its acquisition/versioning is Stage 07/09 work, not a Stage 00 failure.

## Rollback

Rollback was not executed. A non-destructive rollback is to apply a reverse patch that restores the prior `pytest.ini` testpaths/marker block and prior incomplete `.codex/baseline-manifest.json`, then removes only the three current-attempt core artifact files. Do not touch `.codex/refactor-state.json` or `dfc.md`. No database or external rollback is needed because no external write occurred.
