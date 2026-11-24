# Stage 01 implementation report

- Stage: `01`
- Revision: `1`
- Attempt: `1`
- Primary agent: `core`
- Status: `IMPLEMENTED`
- Draft: `.codex/handoffs/stage-01-rev-1-attempt-1-draft-ready.json`
- Observability: `.codex/handoffs/stage-01-rev-1-attempt-1-observability.json`

## Outcome

The central monthly forecast contract is implemented without activating Stage 02 or changing feature formulas. `ForecastContext` normalizes aware timestamps to UTC, requires the actual close of the last regular XNYS session in a month, derives the exact 21st session strictly after that cutoff, carries SPY and all required versions, and supports `LIVE` and `HISTORICAL_BACKTEST`.

Temporal and version guards fail closed. Adjusted-close label helpers produce ticker return, SPY return, `forward_21d_excess_return`, and direction over the same exact interval. State migration is additive: the eight legacy `SharedManagerState` keys remain required and the new context/feature/output slots are optional and JSON-safe.

Observability added allowlisted forecast metadata, monotonic latency helpers, and opt-in progress events. The legacy `publish()` shape remains compatible. No raw evidence, full State, reasoning, headers, prompts, or secrets enter the new forecast metadata path. Observability requested no Primary changes and no change request exists.

## Core-owned changed files

- `core/forecast_contract.py`
- `agents/state.py`
- `core/error_handler.py`
- `core/__init__.py`
- `agents/__init__.py`
- `tests/unit_tests/test_forecast_contract.py`
- `tests/unit_tests/test_state_forecast_contract.py`
- `tests/unit_tests/core_test/test_error_handler.py`
- `.codex/handoffs/stage-01-rev-1-attempt-1-draft-ready.json`
- `.codex/handoffs/stage-01-rev-1-attempt-1-implementation.md`
- `.codex/handoffs/stage-01-rev-1-attempt-1-implementation.json`

## Independently owned Observability files audited read-only

- `core/observability.py`
- `core/progress_bus.py`
- `.codex/handoffs/stage-01-rev-1-attempt-1-observability.json`

## Final self-tests

- Focused Stage 01 plus legacy observability suite: 68 passed in 2.51s.
- Direct allowlist, forecast progress, and legacy `publish()` compatibility assertions: passed.
- Guarded full collection: 1,160 collected (1,109 unit and 51 integration), exit 0 in 70.26s; zero external attempts.
- Guarded full unit baseline: 1,071 passed, 38 pre-existing failed, 6 warnings in 46.55s; zero external attempts.
- Python compilation of Core, State, Observability, and progress modules: passed.
- JSON/schema/control/lock and diff checks are recorded in the matching JSON artifact.

## Known pre-existing failures

The remaining 38 failures are unchanged baseline groups: Financial 28, Sentiment 4, Manager integration 1, RAG/Sentry patch point 1, and missing legacy Supabase migration 4. The two Stage-01-owned `error_handler` failures from Stage 00 are resolved.

## Limitations

- The deterministic `XNYS-regular-v1` calendar supports 1998 through 2100 and includes standard holidays, Black Friday early close, and known exceptional full closures. Unsupported years fail closed. A future calendar-rule change requires a new calendar/contract version.
- Historical datasets, universe manifests, outcomes, and model artifacts do not yet exist; they belong to later authorized stages.
- Supabase, Neo4j, backup, checksum, restore, and the missing legacy migration retain their Stage 00 dispositions.
- Observability contracts are ready but no runtime feature call sites were instrumented and no latency measurement was fabricated in Stage 01.

## Rollback

Rollback was not executed. Remove the new forecast contract and Stage 01 tests; restore the previous versions of `agents/state.py`, `core/error_handler.py`, `core/__init__.py`, and `agents/__init__.py`; restore the Observability-owned files using its independent artifact; and remove only the Stage 01 attempt-1 artifacts. Preserve all accepted Stage 00 artifacts, state history, user files, and product data.
