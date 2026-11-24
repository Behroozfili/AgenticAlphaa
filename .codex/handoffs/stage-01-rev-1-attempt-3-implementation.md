# Stage 01 revision 1 attempt 3 — Core implementation

Status: `IMPLEMENTED` in `PRIMARY_FINAL`; ready for QA and stopped before that phase.

## Blocking finding resolved

`STAGE01-STATE-JSON-HYGIENE-005` is resolved. `validate_forecast_feature_values` recursively creates fresh plain `dict` and `list` containers for every accepted `Mapping`/list subtree, preserves valid JSON scalars, retains depth/item/key/text/finite-number limits, and rejects cycles. Returned payloads succeed with `json.dumps(..., allow_nan=False)` and do not alias mutable input containers.

Exact regressions cover nested `MappingProxyType`, a custom `Mapping`, repeated shared input containers, source/custom-mapping mutation after return, cycle rejection, container identity, and JSON serialization. Attempt 2's XNYS New Year/Carter closure, exact 21-session target, strict `from_dict`, privacy and legacy State behavior remain passing. No final code correction was required after draft3 and the matching Observability audit; Observability requested no changes and its files remained read-only.

## Verification

- Focused Stage 01 plus Observability: 95 passed.
- Corrected exact counterexample matrix: PASS. An initial assertion harness used obsolete helper import names and stopped before assertions; rerunning with the actual exported helper names passed.
- Guarded collection: 1,187 collected (1,136 unit and 51 integration), zero external attempts.
- Guarded unit baseline: 1,098 passed, 38 known/downstream failures, 6 warnings, zero external attempts.
- Compilation, privacy/legacy checks, control, locks and scoped diff validation: PASS.

The 38 baseline failures remain assigned to financial/Stage 05 (28), sentiment/Stage 06 (4), integration/Stage 08 (1), data_rag/Stage 03 (1), and the missing legacy Supabase migration disposition from Stage 00 (4).

## Limitations and rollback

Calendar support remains versioned and bounded to 1998–2100. Historical datasets/model artifacts are later-stage work. Database/backup evidence and the legacy migration retain their Stage 00 dispositions; the downstream RAG fixture/import-order concern remains Stage 03-owned.

Rollback only Attempt 3 by restoring `agents/state.py` and its State test to immutable Attempt 2 content, then remove only Attempt 3 artifacts. Preserve Attempts 1/2, Observability artifacts/files, Stage 00 evidence, state history, dependencies and product data.

References: `.codex/handoffs/stage-01-rev-1-attempt-3-draft-ready.json`, `.codex/handoffs/stage-01-rev-1-attempt-3-observability.json`, and `.codex/handoffs/stage-01-rev-1-attempt-2-qa.json`.
