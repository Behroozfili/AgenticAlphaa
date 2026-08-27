# Stage implementation report

- Stage: `XX`
- Revision: `N`
- Attempt: `M`
- Primary agent: `AGENT`
- Status: `IMPLEMENTED | REVALIDATION_ONLY | BLOCKED`
- Base Git reference: `REF`

## Outcome

State what was implemented and whether the attempt is ready for QA.

## Changed files

List every changed path exactly as reported by Git and the matching lock owner.

## Self-tests and evidence

Record commands, exit status, fixture/data scope, leakage result, latency/data counts when applicable, and artifact references. Never report an unexecuted check as PASS.

## Contract/schema changes

Describe compatibility, version changes, migrations created but not applied, and any downstream invalidation.

## Known limitations and out-of-scope findings

Separate blockers from non-blocking limitations and create named change-request artifacts where required.

## Rollback

Provide exact, non-destructive rollback instructions. Do not claim rollback was tested without evidence.
