# Stage 00 implementation report

- Stage: `00`
- Revision: `1`
- Attempt: `2`
- Primary agent: `core`
- Status: `IMPLEMENTED`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`
- Supersedes: `.codex/handoffs/stage-00-rev-1-attempt-1-implementation.json`

## Outcome

QA finding `STAGE00-OFFLINE-RUNNER-001` is resolved. Pytest now disables the auto-loaded LangSmith pytest plugin through `-p no:langsmith_plugin`. A trace-config probe confirmed that plugin was not registered, and full collection plus the full unit baseline completed without a LangSmith network attempt, teardown error, socket error, or HTTP error. No QA socket monkeypatch was used.

The baseline remains 1,045 passing and 40 precisely captured pre-existing failures; no product code or test assertion changed. Attempt 1 artifacts are immutable and were not edited.

## Changed files

- `pytest.ini` — lock owner `core`; added only `-p no:langsmith_plugin` to existing `addopts`.
- `.codex/baseline-manifest.json` — lock owner `core`; updated attempt-2 evidence, plugin control, timings, artifact references, and runner hash.
- `.codex/handoffs/stage-00-rev-1-attempt-2-implementation.md` — lock owner `core`.
- `.codex/handoffs/stage-00-rev-1-attempt-2-implementation.json` — lock owner `core`.
- `.codex/handoffs/stage-00-rev-1-attempt-2-milestone-BASELINE_READY.json` — lock owner `core`.

The orchestrator-owned `.codex/refactor-state.json`, QA attempt-1 artifact, immutable attempt-1 implementation artifacts, and user-owned `dfc.md` were not modified by `core` in attempt 2.

## Self-tests and evidence

- Forced-offline `pytest --trace-config --collect-only` probe on the migration test: exit 0; `langsmith_plugin` absent; no network/socket/connection/HTTP error.
- Forced-offline full collection over unit + integration: exit 0 in 17.55s; 1,136 collected (1,085 unit + 51 integration).
- Forced-offline full unit baseline without socket monkeypatch: exit 1 in 31.74s; 1,045 passed, 40 pre-existing failed, 35 warnings; no teardown/network error.
- Focused migration baseline: four preserved `FileNotFoundError` failures in 0.15s.
- `uv lock --check --offline`: exit 0; 210 packages resolved.

The environment controls remained `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `DATASETS_OFFLINE=1`, `WANDB_DISABLED=true`, and `LANGCHAIN_TRACING_V2=false`. References to `langsmith.run_helpers` inside failure tracebacks are product decorators, not the disabled pytest plugin; they produced no network/teardown error.

## Contract/schema changes

No product, schema, migration, dependency, lock-file, or test-contract change. The runner-only change prevents an installed third-party pytest plugin from auto-starting a network-capable teardown path.

## Known limitations and out-of-scope findings

- The same 40 product/legacy migration baseline failures remain visible and assigned exactly as in the manifest.
- Supabase, Neo4j, backups, checksums, and restore evidence remain `NOT_AVAILABLE`/`NOT_RUN`, never `PASS`.
- The obsolete trace and missing migration dispositions are unchanged.

## Rollback

Rollback was not executed. Reverse only the attempt-2 addition by removing `-p no:langsmith_plugin` from `pytest.ini`, restore the attempt-1 evidence fields in the baseline manifest, and remove only the three attempt-2 core artifacts. Do not alter attempt-1 artifacts, state, QA evidence, product files, or `dfc.md`.
