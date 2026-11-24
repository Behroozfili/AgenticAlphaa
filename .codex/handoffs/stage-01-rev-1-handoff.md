# Stage 01 handoff

- Stage: `01`
- Revision: `1`
- Accepted attempt: `3`
- Primary agent: `core`
- Status: `CLOSED`

## Accepted artifacts

- Draft-ready: `.codex/handoffs/stage-01-rev-1-attempt-3-draft-ready.json`
- Observability: `.codex/handoffs/stage-01-rev-1-attempt-3-observability.json` — `AUDITED`
- Implementation: `.codex/handoffs/stage-01-rev-1-attempt-3-implementation.md` and `.json`
- QA: `.codex/handoffs/stage-01-rev-1-attempt-3-qa.json` — `PASS`
- Review: `.codex/handoffs/stage-01-rev-1-attempt-3-review.json` — `APPROVED`

## Retry chain

- Attempt 1 ended with Review `REJECTED` for XNYS calendar and fail-open context deserialization defects.
- Attempt 2 resolved those defects but ended with QA `FAIL` for nested non-native Mapping materialization.
- Attempt 3 recursively materialized accepted Mapping/list subtrees into fresh plain JSON-native containers and passed QA and Review.
- All failed-attempt artifacts remain immutable evidence and are not overwritten by this handoff.

## Closure evidence

The accepted attempt passed 95 focused tests. Guarded collection found 1,187 tests: 1,136 unit and 51 integration. The guarded unit run passed 1,098 tests with exactly 38 documented downstream or legacy failures, zero Stage 01 regressions, zero external network attempts and zero completed external connections. Rollback is documented in the accepted implementation artifact. No database, migration, seed, deployment, commit or push action occurred.

## Accepted contract outcomes

- XNYS `2021-12-31` remains an open regular session and December cutoff.
- The `2025-01-09` Carter national closure is represented, making the 21st session after `2024-12-31` equal to `2025-02-03`.
- `ForecastContext.from_dict` requires all canonical serialized fields and cannot silently degrade historical mode to LIVE.
- Accepted feature values are recursively copied into fresh plain `dict`/`list` JSON-native structures and serialize with `allow_nan=False`.
- Observability remains additive, allowlisted and non-traversing for full State, raw evidence and reasoning payloads.

## Known limitations

- Thirty-eight documented failures remain assigned downstream or to the Stage 00 legacy-migration disposition.
- The legacy RAG AlphaEmbedder fixture/import-order concern remains owned by `data_rag` for Stage 03; it did not reproduce in the accepted HF-offline run.
- `XNYS-regular-v1` is bounded to 1998–2100. New exceptional closures or rule changes require a calendar/contract version update and downstream invalidation.
- Historical datasets and model artifacts remain later-stage work.
- Database and backup/restore evidence retain the Stage 00 `NOT_AVAILABLE/NOT_RUN` dispositions.

## Supersedes and downstream effect

This is the first Stage 01 handoff and supersedes no prior revision. Stage 02 is the next eligible stage but is not activated; separate explicit user authorization is required.
