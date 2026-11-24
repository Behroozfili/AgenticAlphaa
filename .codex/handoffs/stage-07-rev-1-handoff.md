# Stage 07 Revision 1 Handoff

Status: CLOSED

Accepted attempt: 4

Primary owner: quant

## Outcome

Stage 07 delivers a deterministic point-in-time market layer, adjusted OHLCV and benchmark handling, immutable feature snapshots, exact 21-trading-session labels, DATA_READINESS and MODEL_READY gates, and logistic/ridge numeric forecasts. The Quant model remains the sole source of forecast numbers and `action_policy_v1` remains unchanged.

Attempt 4 is the accepted chain. Runtime constants, MarketForecastService output and every Quant lifecycle event identify `07/r1/a4/quant` with implementation version `stage07-r1-a4`. Canonical excess return is ticker total return minus benchmark total return, direction is positive exactly when canonical excess return is greater than zero, and inconsistent rows fail closed.

## Accepted evidence

- DATA_READINESS and MODEL_READY: PASS with matching readiness, eligible-row, training-data and model hashes.
- Observability: AUDITED with zero blockers or requested changes; nine measured compact lifecycle boundaries use current Attempt 4 identity.
- QA: PASS; 71 focused tests and 121 combined ForecastContext/State/Stage 07 tests passed.
- Review: APPROVED with no blocking, scope or ownership findings.
- All 40 fixture rows have canonical labels and exact 21-session targets; normal and reversed order reproduce the same readiness, model and prediction.
- Attempts 1-3 remain immutable.

## Retry resolution

- Attempt 1 resolved initial Observability findings, then QA rejected stale readiness/model evidence.
- Attempt 2 regenerated deterministic evidence, then Review rejected non-canonical fixture label semantics.
- Attempt 3 enforced canonical label magnitude and direction, then QA found stale runtime/service/lifecycle attempt identity.
- Attempt 4 advanced the single runtime identity to the accepted attempt and added direct stale-attempt counterexamples.

## Known limitations

- Evidence remains PRELIMINARY: the deterministic fixture is synthetic, single-ticker and 40-cutoff, and is not an investment-performance claim.
- Research/news, SEC filing and temporal Graph coverage remain `UNVERIFIED_REFERENCE_ONLY` in this fixture.
- One explanatory DATA_READINESS sentence still says “current Attempt 3 runtime”; artifact identity, hashes and runtime behavior consistently identify Attempt 4, so this is recorded as non-blocking `OBS-07-A4-NB-001`.
- Walk-forward metrics, baselines, transaction costs and ablations remain Stage 09 scope.
- No network/provider access, external trace contact, database read/write, migration, seed, deletion, deployment, commit or push was performed.

## Rollback

No database rollback is required. Preserve the immutable evidence chain, quarantine the accepted Stage 07 runtime/test paths and artifacts without deletion, and restore only authoritative prior bytes after SHA-256 verification. Stop if authoritative prior bytes are unavailable.

Stage 08 is eligible but remains PENDING, NOT_STARTED and inactive until separately activated.
