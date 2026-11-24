# Stage 04B implementation report

- Stage: `04B`
- Revision: `1`
- Attempt: `2`
- Primary agent: `research`
- Phase: `PRIMARY_FINAL`
- Status: `IMPLEMENTED`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

QA finding `QA-04B-001` is resolved without changing product source. The direct Research MCP dispatcher tests now require a valid `ForecastContext` and `run_id`, cover all eight dispatcher paths, prove missing identity fails before provider invocation, and verify privacy-safe fail-closed result, log and Sentry behavior. The attempt is ready for QA with no blockers.

The immutable Attempt 1 implementation artifact contained one recorded-hash defect for `tools/research_tools/tavily_search.py`. Attempt 2 does not rewrite that artifact or source file; it records the independently recomputed hash `3c40e9638aa23f1c1544e715cdb3760c32b83eafffe790cbeff8b130c11350e1`. The remediated `tests/unit_tests/search_test/test_search_server.py` hash is `29434950869c6f1c33cf31163f471af87a6967c32be3130c47661a11b532fabd`.

## Changed files

- `tests/unit_tests/search_test/test_search_server.py` — locked owner `research`
- `.codex/handoffs/stage-04B-rev-1-attempt-2-implementation.json` — locked owner `research`
- `.codex/handoffs/stage-04B-rev-1-attempt-2-implementation.md` — locked owner `research`

No product source, schema, migration or observability source changed in Attempt 2.

## Same-attempt evidence chain

- Draft-ready: `.codex/handoffs/stage-04B-rev-1-attempt-2-draft-ready.json` — SHA-256 `0d159dc3bad94068a22f8a7ec0b0e93dcb89f9e05fed60fcdfafeef6106b7ce3`
- Observability: `.codex/handoffs/stage-04B-rev-1-attempt-2-observability.json` — SHA-256 `ef761d3048f2dfff4ece394f0caddf0be369136ff82d82380b06f0d57e48aff9`, status `AUDITED`, zero requested changes
- All 20 recorded source/test SHA-256 values were recomputed directly and match.
- All five immutable Attempt 1 artifacts retain their recorded hashes.

## Self-tests and evidence

- Direct dispatcher offline suite: `15 passed`; all eight routes exercised with mandatory identity.
- Complete offline Research search suite: `66 passed`.
- Focused Stage 04B privacy/lifecycle/unit gate: `119 passed`.
- Graph/RAG regression: `333 passed`.
- Collection-only gate: `1311 collected`, zero errors; integration tests were not executed.
- Scoped compile, diff-check, privacy scan, old prompt reference scan, direct hash verification, control/lock/artifact validation: `PASS` with line-ending notices only.

No live/provider integration test was run in Attempt 2. `tests/integration_tests/test_integration_search_agent.py` remains explicitly forbidden from execution.

## Contract and observability result

`ForecastContext` and `run_id` remain mandatory for live and historical execution. Historical research remains allowlist-only and fail-closed without a current-web fallback. SEC and all other evidence must be known and on/before `as_of`, with independent rejection before synthesis and before State materialization. Typed `ResearchFeatures` remains compact and reference-only.

`OBS-04B-001` through `OBS-04B-004` remain resolved. Trace processors, logs, progress, Sentry, MCP errors and State retain allowlisted identity, stable codes, measured latency and compact counts only; raw query, plan, argument values, exception text, URLs, evidence, citations, synthesis and rejection payloads are excluded.

## External execution and incident disclosure

Attempt 2 performed no network access, provider call, database read/write, migration, seed, deployment, commit or push. The disclosed Attempt 1 accidental read-only legacy probe contacted News and SEC before being stopped; it made no external write, was not rerun, and is not used as completion evidence.

Stage 05 remains `PENDING / NOT_STARTED`.

## Limitations

Historical behavior is fixture-validated. No external archive or live provider integration was authorized. This does not block the Stage 04B offline acceptance contract.

## Rollback

Restore only the pre-Attempt2 bytes of `tests/unit_tests/search_test/test_search_server.py`; no product source or database rollback is required. Attempt 1 artifacts remain immutable. If the control plane rejects Attempt 2, remove only the Attempt 2 implementation artifacts through the orchestrated rollback process. No data-loss risk exists from this attempt.
