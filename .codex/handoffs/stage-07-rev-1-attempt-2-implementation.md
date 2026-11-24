# Stage 07 Quant revalidation report

- Stage: `07`
- Revision: `1`
- Attempt: `2`
- Primary agent: `quant`
- Status: `REVALIDATION_ONLY`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `129`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

Stage 07 Attempt 2 is ready for independent QA. This retry resolves `QA-07-001`
by regenerating both mandatory milestone artifacts from the current final runtime
and locked fixture. No product, test, shared, state, dependency, prior-attempt or
downstream-stage file changed.

The reproduced DATA_READINESS manifest is
`ac2129fb3ccb08e630ead2e80f8243b1538938c9373ab030ac81a2d294432ca6` and
contains eligible-row content hash
`1a1c07998cc84ba0f4286733055da68b1410c87dce4b58e6181116cf53ee4bf4`.
MODEL_READY is bound to both values and to current runtime model version
`logistic-ridge-21d-v1+sha256:43fc408aaeac86fba888f822a44ffb90452c816487bf06bff917247b86e11126`.
All forecast values in the milestone were emitted by the deterministic runtime;
no LLM created or altered forecast numbers.

## Attempt 2 artifacts

- `.codex/handoffs/stage-07-rev-1-attempt-2-milestone-DATA_READINESS.json`
  (`cd406506c1b9c001ea4f62861796aadf9243e7d9c049b705cb93611b692b8b6e`)
- `.codex/handoffs/stage-07-rev-1-attempt-2-milestone-MODEL_READY.json`
  (`3c52a74ebc9c3f74ae5db54bff131696e6362997d4edd8af860121813d27f6b1`)
- `.codex/handoffs/stage-07-rev-1-attempt-2-draft-ready.json`
  (`4058fb9ed2d19f47900599204b697101bc380dbd4749a21eddb79a6c4a87265f`)
- `.codex/handoffs/stage-07-rev-1-attempt-2-observability.json`
  (`855c3d6079ad9f9b1f250b24a944e266ce646e3185c514661c34496a2ae8bd82`),
  an immutable Observability-owned input with status `AUDITED` and no findings.
- This implementation MD and its matching JSON.

## Verification

- The exact offline Stage 07 suite passed: `64 passed in 3.45s`.
- Normal and reversed row order reproduced identical readiness, eligible-row,
  model-artifact and forecast outputs.
- Observability independently audited semantic integrity, runtime immutability,
  privacy and preserved latency instrumentation with no requested primary changes.
- The official control/lock validator reports `CONTROL VALID` at control revision
  129 with Stage 07 revision 1 attempt 2 active in `PRIMARY_FINAL`.
- Artifact schema, identity, cross-link, byte-hash and runtime-input checks are
  recorded in the matching implementation JSON.

## Scope and limitations

This is a revalidation-only retry. The final runtime remains the byte-preserved
Attempt 1 Stage 07 implementation. The synthetic, single-ticker, 40-cutoff
fixture proves engineering correctness only; its scientific report level remains
`PRELIMINARY`, and upstream reference coverage remains explicitly unverified.
Walk-forward performance metrics, baselines and ablations remain Stage 09 scope.

No network/provider acquisition, external tracing, database read/write, migration,
seed, deletion, deployment, commit, push or Stage 08 work was performed.

## Rollback

No product or data rollback is required. Preserve the audit trail and move only
the five Quant-owned Attempt 2 artifacts (two milestones, draft-ready, this MD and
the matching implementation JSON) into a timestamped workspace quarantine. Do
not delete or modify Attempt 1 artifacts, the Observability artifact, refactor
state, accepted dependency handoffs, product/test files, migrations or external
data.
