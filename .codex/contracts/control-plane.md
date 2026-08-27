# Refactor control-plane contract

This file is executable policy for the AgenticAlpha refactor. Precedence is: current user instruction, `.codex/refactor-state.json`, active agent TOML, `list.md`, contracts in this directory, and finally the explanatory orchestration document.

## Canonical scope and stages

- `list.md` is the only canonical implementation plan. `docs/implementation_plan.md` is not a source of truth.
- Stage order is `00, 01, 02, 03, 04A, 04B, 05, 06, 07, 08, 09, 10A, 10B, 11, FINAL_QA, FINAL_REVIEW`.
- Exactly one implementation stage may be `ACTIVE`. Its `phase` is one of `PRIMARY_DRAFT`, `OBSERVABILITY`, `PRIMARY_FINAL`, `QA`, or `REVIEW`. Observability, QA, and Reviewer are passes for that same stage/revision/attempt, not additional implementation stages.
- Stage 08 owns Manager/domain integration only. Stage 10A owns API/runtime isolation foundations. Stage 10B owns route, persistence, progress, UI, and E2E wiring.
- Runtime prompt typo renames occur in their feature stages (04B, 05, 06, 08). Stage 11 only verifies those legacy references are zero; test/doc/dependency cleanup remains Stage 11.
- `monthly_financial_features.py` is Financial-owned in Stage 05 and read-only to Integration in Stage 08.

## Attempt workflow

The only valid workflow for an implementation attempt is:

1. Orchestrator activates one stage/revision/attempt and records exact path locks.
2. Primary produces a draft implementation. If `observability_required=true`, it writes `draft-ready` JSON and stops.
3. Observability starts only from that matching `draft-ready` artifact, writes its own attempt artifact, and may edit only explicitly locked observability files; feature findings become a named change-request artifact.
4. Primary is reactivated, applies accepted findings, runs final self-tests, and writes final implementation MD/JSON.
5. QA validates that same attempt and writes only its QA artifact.
6. Reviewer reads matching implementation, observability when applicable, and QA artifacts and writes only its review artifact.
7. Orchestrator validates artifacts and locks, selects the accepted attempt, writes revision handoff MD/JSON, and closes the stage.

No artifact may be overwritten. A retry increments `attempt`; an upstream reopen increments `revision` and invalidates dependent closed stages as `STALE`.

## Artifact names

- `stage-XX-rev-N-attempt-M-implementation.md/json`
- `stage-XX-rev-N-attempt-M-draft-ready.json` when observability is required
- `stage-XX-rev-N-attempt-M-observability.json`
- `stage-XX-rev-N-attempt-M-qa.json`
- `stage-XX-rev-N-attempt-M-review.json`
- `stage-XX-rev-N-handoff.md/json` (Orchestrator; points to the accepted attempt)
- `final-qa-rev-N.json`
- `final-review-rev-N.json`
- `change-request-stage-XX-rev-N-attempt-M-K.json`
- `stage-XX-rev-N-attempt-M-milestone-NAME.json` for an explicitly declared milestone

All JSON artifacts must satisfy `.codex/handoffs/schemas/artifact.schema.json`. Templates under `.codex/handoffs/templates/` are examples only and are never execution evidence.

## Locks and authority

- `.codex/refactor-state.json` has permanent bootstrap ownership by Orchestrator. This is the only lock exception, and it permits only state transitions/locks/history—not product implementation.
- Every other changed source, test, migration, dependency, control, and artifact path requires an exact lock in state for its writer.
- Before handoff, `.codex/scripts/validate_refactor_control.py --check-control --check-locks` must derive the complete changed-path set from Git. Artifact `changed_files` are additionally checked against their writer; a manually supplied subset is never sufficient.
- Accepted locks move into immutable `lock_history` with the accepted file SHA-256 (or `DELETED`) when a Stage closes. Historical ownership is valid only while current content still matches that accepted hash; a later modification therefore requires a new current lock. Mid-refactor commits are forbidden unless separately authorized and the workspace baseline is formally advanced.
- Protected shared files must follow `protected_path_schedule` in state. A schedule entry is not itself a lock; Orchestrator must still issue an exact current-attempt lock.
- No migration application, external database mutation, data deletion, bulk/network acquisition, seed, deploy, commit, or push is authorized by this refactor plan. Each requires separate explicit user authorization recorded in state.
- Stage 00 may inventory external databases read-only when access is available. Claims without evidence remain `UNVERIFIED` or `BLOCKED`, never `PASS`.

## Stage 00 close rule

Test collection must run. Existing failures may be recorded as baseline failures and do not have to be repaired in Stage 00, but they must not be hidden. Missing baseline artifacts require an explicit, documented disposition. See `.codex/baseline-manifest.json`.

## Milestone and closure rules

- A milestone is an object with `status` and `artifact`, not a label. Required values are `PENDING`, `PASS`, `FAIL`, or `BLOCKED`.
- A Stage cannot become `CLOSED` while any declared milestone is not `PASS` or lacks a current revision/attempt evidence artifact.
- A closed Stage must have an accepted attempt and matching implementation, QA, review, optional/required observability, rollback, and handoff references.
- `final-qa-rev-N.json` and `final-review-rev-N.json` use the top-level `final_cycle_revision` as `N`.

## Product compatibility between stages

- Every Stage must leave imports and the existing default runtime path collectable. Prompt renames update their direct imports and locked tests atomically in one attempt.
- Stage 08 must preserve the current API-facing `ManagerAgent` constructor/run compatibility until Stage 10A replaces shared Manager/memory lifecycle. It may add an adapter but must not require Stage 10A changes to close Stage 08.
- Stage 03 introduces V2 RAG access behind an explicit configuration/adapter boundary. Until migration application and reconciliation are separately authorized, the default legacy runtime remains usable and V2 behavior is validated with deterministic fixtures; it may never silently point production reads at absent V2 tables.
