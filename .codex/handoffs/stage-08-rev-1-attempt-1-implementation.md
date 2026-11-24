# Stage 08 Integration implementation report

- Stage: `08`
- Revision: `1`
- Attempt: `1`
- Primary agent: `integration`
- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

Stage 08 domain integration is implemented and ready for independent QA. The
Manager accepts a common typed `ForecastContext`, runs Research, Financial and
Sentiment readiness concurrently on isolated State copies, performs a controlled
join, consumes the Stage 07 market/Quant service, and delegates synthesis to the
new `MonthlyForecastAgent`. Quant remains the immutable source of probability,
expected return, excess return, uncertainty and `action_policy_v1` action. The
LLM is explanation-only and cannot replace numeric or action fields.

The output has explicit `READY`, `DEGRADED` and `INSUFFICIENT_DATA` behavior,
including null confidence when numeric data are insufficient, compact warnings,
evidence references and version metadata. Future evidence and changed Quant
action fail closed. The legacy Manager constructor and `run()` path remain
backward-compatible when no forecast context is supplied.

The Manager prompt typo was renamed atomically from
`promt_manager_agent.py` to `prompt_manager_agent.py`; direct imports and
exports now use the corrected name.

## Observability remediation

- `OBS-08-001` is resolved. `ManagerAgent.run` now has allowlisted LangSmith
  input/output processors and immutable `stage08-r1-a1` metadata. The processors
  retain compact identity, status, counts and booleans while excluding task text,
  arbitrary directives, complete State, ForecastContext payloads, narratives,
  evidence identifiers, prompts, raw upstream values, secrets and exceptions.
- `OBS-08-002` is resolved. The monthly branch emits measured,
  identity-bound, payload-free events for context acceptance, specialist
  start/join, market/Quant start/complete, monthly synthesis completion,
  degradation and fail-closed error/re-raise. Each event carries the validated
  run identity and exact Stage 08 revision/attempt/agent identity. Session queues
  isolate concurrent runs.

Counterexamples prove a `PRIVATE_SENTINEL` is absent recursively from trace
processor results and lifecycle events, success/error events contain finite
non-negative latency, exception text is not emitted, and two concurrent
ticker/as-of/run sessions have no result or progress cross-talk.

## Changed files

- `agents/monthly_forecast_agent.py`
- `agents/executor_helpers/monthly_forecast_helper.py`
- `agents/prompts/prompt_monthly_forecast.py`
- `agents/manager_agent.py`
- `agents/executor_helpers/manager_executor_helper.py`
- `agents/prompts/promt_manager_agent.py` (deleted by atomic rename)
- `agents/prompts/prompt_manager_agent.py`
- `agents/__init__.py`
- `agents/prompts/__init__.py`
- `agents/executor_helpers/__init__.py`
- `tests/fixtures/stage8_manager_integration.json`
- `tests/unit_tests/test_stage8_manager_integration.py`
- `tests/unit_tests/test_agents/test_manager_agent.py`
- `.codex/handoffs/stage-08-rev-1-attempt-1-implementation.md`
- `.codex/handoffs/stage-08-rev-1-attempt-1-implementation.json`

The draft-ready, observability and change-request artifacts are immutable inputs
and were not overwritten. No upstream Research, Financial, Sentiment, Data/RAG,
Quant, API, database, migration, persistence, route or UI file was changed.

## Self-tests and evidence

- Focused Stage 08 integration and OBS counterprobes: `12 passed in 17.72s`.
- Focused plus legacy Manager regressions: `62 passed in 20.37s`.
- Forecast contract, State, Stage 07 Quant, Stage 08 and Manager regressions:
  `183 passed in 22.30s`.
- Broad unit/integration collection: `1427 tests collected in 10.99s`, with
  zero collection errors; integration tests were collected only.
- `py_compile` over all Stage 08 source/export/test paths passed.
- `git diff --check` exited 0; only existing line-ending warnings were printed.
- Corrected prompt import/absence checks passed.
- Official control and lock validation returned `CONTROL VALID` before artifact
  generation.

All tests were offline with deterministic fixtures and mocks. No live provider,
external trace service, network acquisition or database was contacted.

## Contract and schema diff

No central State schema, database schema or migration changed. Stage 08 adds an
optional monthly Manager domain path keyed by an existing typed
`ForecastContext`, typed feature slots, a canonical monthly forecast output,
controlled node-order evidence and compact Stage 08 telemetry. Legacy Manager
behavior is unchanged when `forecast_context` is absent. DCF remains a
long-term valuation feature and is never presented as the 21-trading-day
forecast. No Quant formula, threshold, probability, expected/excess return,
uncertainty or action policy was changed.

## Known limitations

There are no blocking issues. Validation is deterministic and offline; live
provider acquisition and external trace export were not exercised. Stage 08
covers domain integration only. Request lifecycle foundation remains Stage 10A,
while API routes, persistence, progress jobs and UI remain Stage 10B. If the
Stage 07 market service or market-data provider is not injected, the monthly
path truthfully returns `INSUFFICIENT_DATA`.

## Rollback

No database or external rollback is required. Preserve the immutable Stage 08
draft, observability, change request and final handoffs. Archive the current
Integration-owned paths with their recorded SHA-256 values, move new monthly
agent/helper/prompt, corrected prompt, fixture, focused test and final artifacts
to a timestamped workspace quarantine, then restore modified Integration-owned
files and the typo prompt from base Git object
`c3cda2144c63b6e69252ca64b93d90bbe39f24b4`. Do not modify
`.codex/refactor-state.json`, upstream handoffs, QA/review artifacts, migrations
or external data. This is a recoverable filesystem rollback when quarantine is
retained.
