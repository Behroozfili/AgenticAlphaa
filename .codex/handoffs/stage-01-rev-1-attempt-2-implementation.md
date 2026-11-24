# Stage 01 revision 1 attempt 2 — Core implementation

Status: `IMPLEMENTED` in `PRIMARY_FINAL`. This artifact supersedes the rejected attempt 1 implementation and stops before QA.

## Review rework resolved

- Corrected XNYS New Year handling: Friday 2021-12-31 is an open session and the December month-end cutoff.
- Added the 2025-01-09 Carter national day-of-mourning closure; the exact 21st XNYS session after 2024-12-31 is 2025-02-03.
- `ForecastContext.from_dict` now requires every canonical serialized field and validates calendar and label identity; historical payloads cannot silently become LIVE.
- `ForecastFeatureSlot.values` accepts compact recursive JSON-safe mappings and rejects excessive depth/size, non-string or empty keys, non-finite floats, and non-JSON objects. Legacy State's eight required keys remain unchanged.

No final code correction was needed after the matching draft and Observability audit. Observability attempt 2 requested no changes and its files remained read-only.

## Verification

- Focused Stage 01 plus Observability: 93 passed.
- Exact regression/privacy/legacy assertions: PASS.
- Guarded collection: 1,185 collected (1,134 unit, 51 integration), no external attempts.
- Guarded unit baseline: 1,096 passed, 38 known/downstream failures, 6 warnings, no external attempts.
- Python compilation: PASS.

The 38 failures remain assigned: financial/Stage 05 (28), sentiment/Stage 06 (4), integration/Stage 08 (1), data_rag/Stage 03 (1), and the missing legacy Supabase migration disposition from Stage 00 (4). They are not Stage 01 regressions.

## Limitations and rollback

Calendar behavior is versioned and bounded to 1998–2100; later exceptional closures require a calendar-version update. Historical datasets and model artifacts are later-stage work. Database/backup evidence and the legacy migration retain their Stage 00 dispositions. A downstream RAG fixture/import-order concern remains owned by Stage 03.

Rollback only attempt 2's Core edits to the immutable rejected attempt 1 content, then remove only attempt 2 draft/final artifacts. Preserve all attempt 1 evidence, Stage 00 artifacts, Observability-owned files, state history, dependencies, and product data.

References: `.codex/handoffs/stage-01-rev-1-attempt-2-draft-ready.json`, `.codex/handoffs/stage-01-rev-1-attempt-2-observability.json`, and `.codex/handoffs/review-stage-01-rev-1-attempt-1.md`.
