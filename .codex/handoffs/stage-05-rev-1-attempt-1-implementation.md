# Stage 05 Revision 1 Attempt 1 — Financial Primary Final

Status: IMPLEMENTED  
Phase: PRIMARY_FINAL  
Owner: financial  
Implementation identity: stage05-r1-a1

## Delivered

- Preserved the Stage 05 point-in-time Financial contract: mandatory ForecastContext/run identity, SEC `filed_at <= as_of`, future-restatement invisibility, exact adjusted Yahoo history, no current-only historical fallback, nine intents and mappers, DCF/WACC/reverse-DCF/scenarios/Monte-Carlo semantics, missing-as-None, and the canonical checker prompt rename.
- Resolved OBS-05-001 by replacing all three reachable raw `traced_call_tool` paths with one Financial-owned safe MCP wrapper. LangSmith processors retain only immutable identity, validated tool/operation, bounded counts, measured latency, outcome, flags, and stable error codes. Raw arguments/results and exception text are excluded; invalid ForecastContext/run identity fails before any session call.
- Resolved OBS-05-002 by replacing payload-bearing Financial server logs with registry-validated fixed events and bounded count-only metadata. No price, EPS, statement, debt/cash, EBITDA, asset, DCF input, provider exception, or private marker is persisted.
- Resolved OBS-05-003 with one allowlisted measured lifecycle builder and callsite coverage for all 12 required boundaries: SEC submissions, CompanyFacts, filing selection/text, restatement gate; Yahoo adjusted history and current-only rejection; MCP/ratio dispatch; monthly adapter; agent extraction, calculation, checker, and pre-Shared-State gate.
- Shared State remains compact and reference-only; raw filing text, XBRL payloads, Yahoo payloads, checker prose, DCF arrays, and attempt messages remain excluded.

## Offline evidence

- Exact adversarial Observability sentinels: 4 passed in 4.06s.
- Consolidated 13-file Financial suite: 281 passed in 30.63s.
- Nine-intent/mapper and DCF/WACC/reverse-DCF/scenario/Monte-Carlo regression: 165 passed in 19.58s.
- Graph/RAG regression: 333 passed in 56.20s.
- Collection-only: 1322 tests collected in 23.54s with zero errors; integration tests were collected only and never executed.
- Scoped compile, diff, privacy, raw-scalar-log, raw-trace, exception-text, old-prompt, SHA, control, lock, draft and observability artifact checks passed. Diff emitted line-ending notices only.

## Safety, limitations, and rollback

All acceptance evidence is offline and deterministic. No live Yahoo/SEC/provider test, network access, database read/write, migration, seed, deletion, deployment, commit, or push occurred. Live-provider behavior remains unverified by design; current-only Yahoo data in historical mode deliberately yields explicit missing data.

Rollback requires restoring the pre-Stage-05 Financial source/test bytes, reversing the checker prompt rename, and removing the Stage-05 adapter/fixture while retaining immutable attempt artifacts. No database rollback is required because this attempt performed no external or database write. Stage 06 remains PENDING/NOT_STARTED.
