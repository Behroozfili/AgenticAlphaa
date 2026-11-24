# Stage 07 Quant implementation report

- Stage: `07`
- Revision: `1`
- Attempt: `4`
- Primary agent: `quant`
- Status: `IMPLEMENTED`
- Execution phase: `PRIMARY_FINAL`
- Control revision: `138`
- Base Git reference: `c3cda2144c63b6e69252ca64b93d90bbe39f24b4`

## Outcome

Stage 07 revision 1 attempt 4 is ready for independent QA. This non-overwriting
retry resolves `QA-07-A3-001`: the immutable Quant runtime identity,
implementation version, MarketForecastService output and every Quant lifecycle
event now publish the exact active identity `07/r1/a4/quant` and
`stage07-r1-a4`. Direct literal counterexamples ensure stale attempts 1 and 3
cannot be silently reported.

All earlier correctness fixes remain preserved. Canonical adjusted-close
excess-return and direction invariants pass for all 40 fixture rows; exact
21-session targets, DATA_READINESS-before-fit, readiness/eligible-row/model
bindings, train-only preprocessing, missing-data semantics, privacy rejection,
compact measured lifecycle telemetry and fixed `action_policy_v1` remain green.
No LLM generated or altered forecast numbers.

## Changed files

All paths below are locked to `quant` for Stage 07 revision 1 attempt 4:

- `tools/market_tools/__init__.py`
- `tests/unit_tests/market_test/test_market_server.py`
- `tests/unit_tests/test_market_features_stage7.py`
- `.codex/handoffs/stage-07-rev-1-attempt-4-milestone-DATA_READINESS.json`
- `.codex/handoffs/stage-07-rev-1-attempt-4-milestone-MODEL_READY.json`
- `.codex/handoffs/stage-07-rev-1-attempt-4-draft-ready.json`
- `.codex/handoffs/stage-07-rev-1-attempt-4-implementation.md`
- `.codex/handoffs/stage-07-rev-1-attempt-4-implementation.json`

The Observability-owned audit was consumed as immutable evidence and was not
changed. Attempts 1 through 3 remain immutable.

## Milestone and audit bindings

- Draft-ready SHA-256:
  `3af823794c1eed6fda318b88826e654c9d92ec6d4e21b4f838f6cf1556a4ee4b`
- DATA_READINESS artifact SHA-256:
  `c0c0813e7569158a36026293b12814b1972b819830f833173542ad8ba8d8543b`
- DATA_READINESS manifest hash:
  `c594e6002196e123ca8f6bead7e233f7272376db84250af87c6174cfb93979b7`
- Eligible-row SHA-256:
  `93b980674b6801db52eb08c8e6b5aa652f82bcc22be2887224d88d45454166e4`
- MODEL_READY artifact SHA-256:
  `835eb5449c4ff89ca888112f8397d17224fc6dd93bf9d879a0ac876603fda500`
- Training-data hash:
  `78af5fc9e5a1fcb7263601e65ead1732e83625cf3604cde46d2b3289f2691890`
- Model version:
  `logistic-ridge-21d-v1+sha256:73d858322007097b578ed1aaaab9146c18fd8b2cc729c64fe348a345f386583a`
- Observability artifact SHA-256:
  `573edf099991ee51160d59e0e24a92d533b230405d54a6d56f4236400fd6eed0`

## Self-tests and evidence

- The focused offline suite passed `71/71` in both the Quant draft pass and the
  independent Observability pass.
- A fresh final combined suite completed with exit code `0`:
  `.venv\\Scripts\\python.exe -B -m pytest -p no:cacheprovider tests/unit_tests/test_forecast_contract.py tests/unit_tests/test_state_forecast_contract.py tests/unit_tests/test_market_features_stage7.py tests/unit_tests/market_test -q`;
  result: `121 passed in 80.85s`.
- The official control/lock validator reported `CONTROL VALID` at control
  revision 138 with Stage 07 rev1/attempt4 active in `PRIMARY_FINAL`.
- Fresh SHA verification found zero mismatches across all 11 draft-declared
  runtime/test/fixture inputs; draft, milestone and Observability hashes matched.
- Observability independently observed all nine lifecycle boundaries across 30
  events with finite non-negative latency and exact Attempt 4 identity. Stale
  attempts 1 and 3, forbidden payload keys and private sentinel data were absent.
- Normal and reversed input order reproduced the readiness manifest, eligible
  rows, training data, deterministic model and prediction exactly.
- Leakage status is `PASS`: features stop at `as_of`; labels use the exact
  21st following regular session; premature outcomes and post-`train_through`
  rows fail closed; selection, imputation and scaling use training rows only.

## Contract/schema changes

The runtime provenance identity advances to `stage07-r1-a4`; this is the only
product behavior change in Attempt 4. Dataset, feature, label-contract and model
family versions remain unchanged. `action_policy_v1` thresholds are unchanged.
No schema migration, upstream contract mutation or downstream invalidation was
introduced.

## Known limitations and out-of-scope findings

The evidence remains `PRELIMINARY`: the locked fixture is synthetic,
single-ticker and 40-cutoff and proves engineering correctness only, not
investment performance. Research/news, SEC filing and temporal Graph coverage
remain `UNVERIFIED_REFERENCE_ONLY`. Live-provider and multi-ticker behavior
were not exercised. Walk-forward metrics, baselines, transaction costs and
ablations remain Stage 09 scope.

Observability recorded non-blocking `OBS-07-A4-NB-001`: one explanatory
DATA_READINESS sentence says "current Attempt 3 runtime." Artifact identity,
run_id, hashes, runtime constant, service output and lifecycle probes all
consistently identify Attempt 4, so this is documented as stale prose with no
behavioral, telemetry, readiness or model-binding impact. It is not repaired in
this final pass because doing so would expand the authorized retry scope and
alter an already validated milestone.

No network/provider acquisition, external trace call, database read/write,
migration, seed, deletion, deployment, commit, push or Stage 08 work occurred.

## Rollback

Rollback is documented but was not executed. Archive and verify the three
Attempt-4-modified runtime/test paths and five Quant-owned Attempt 4 artifacts,
then move those exact current files to a timestamped workspace quarantine without
deletion. Restore only the three runtime/test paths from an authorized Attempt 3
archival source and verify prior hashes; never synthesize unavailable bytes.
Preserve the immutable Observability audit, all prior-attempt artifacts, refactor
state, dependency handoffs, migrations and external data. If authoritative prior
bytes are unavailable, stop and treat rollback as blocked rather than guessing.
