# Agent Refactor Orchestration Plan

> Basis: the current AgenticAlpha project plan for converting the system into a monthly 21-trading-day Forecast, point-in-time RAG/Financial/Sentiment, a numeric model, backtest, and then cleanup.
>
> Purpose of this document: precisely define the role of every `.codex/agents/*.toml` file, the Agent activation order, file ownership, Gates, handoffs, and the prohibition on entering the next stage prematurely.

## Official Agent Structure

From this version onward, `integration.toml` is an official, independent Agent; because the Gate-controlling Agent (`orchestrator`) must not simultaneously own implementation for the integration Stages.

```text
.codex/
└── agents/
    ├── orchestrator.toml
    ├── core.toml
    ├── data_rag.toml
    ├── research.toml
    ├── financial.toml
    ├── sentiment.toml
    ├── quant.toml
    ├── integration.toml
    ├── qa.toml
    ├── observability.toml
    ├── cleanup.toml
    └── reviewer.toml
```

Any other Agent added in the future must be registered in this inventory before use, have an explicit Stage and ownership, and be activated by `orchestrator`. An unknown or unregistered Agent is not a writer.

---


## Execution Coverage Addendum — canonical for the execution pack

The root `list.md` is the only canonical plan for *what* must be implemented. This document explains *who may write what and when*. Executable state and contracts live under `.codex/`; when explanatory examples here differ from `.codex/refactor-state.json`, an active Agent TOML, or `.codex/contracts/`, those executable sources win and execution must stop until the prose is corrected.

1. **Stage 00 DB baseline support:** `data_rag` is an allowed read-only support agent in Stage 00 for Supabase schema/RPC/index/RLS/count export, Neo4j constraint/index/count export, and DB backup inventory. It must perform zero DB/schema/data writes. `core` remains Stage 00 primary.
2. **Dependency support:** `core` is an allowed support agent during Stages 02-10 only after an explicit dependency change request and orchestrator locks. It may edit only `pyproject.toml`, `requirements.txt`, and regenerate `uv.lock`; it does not become the stage primary.
3. **Prompt rename timing:** the four runtime prompt typo renames happen in their owning stages: 04B Research, 05 Financial, 06 Sentiment, 08 Integration. Stage 11 only verifies old prompt references are zero.
4. **Stage 11 explicit cleanup ownership:** `cleanup` owns the planned Stage 11 work for `Dockerfile`, `package-lock.json`, `env.example`, `README.md`, `READ.md`, `infra/ecr-stack.yaml`, `infra/ecs-stack.yaml`, `infra/oidc-role-stack.yaml`, and comparison/consolidation of `tests/integration_tests/` vs `tests/integration_test_2/`, subject to the original safety gates.
5. **API isolation split:** Stage 08 is Manager/domain integration and in-process state isolation only. `core` owns API lifecycle/request-isolation foundations in Stage 10A; `integration` owns routes and E2E concurrency in Stage 10B.
6. **Stage 09 existing evaluation paths:** `quant` owns the existing evaluation modules explicitly listed in `list.md` for the Stage 09 forecast/backtest source-of-truth, gates, metrics and RAG-evaluation consolidation, while preserving legacy coverage until replacement is proven.
7. **Financial feature ownership:** `agents/executor_helpers/monthly_financial_features.py` belongs to `financial` in Stage 05. Stage 08 consumes it read-only.
8. **Executable contracts:** artifact workflow/names, monthly target/provider/backtest policy, and storage natural keys are frozen in `.codex/contracts/` and validated by `.codex/scripts/validate_refactor_control.py`.

---

## 1) Core Principle for Executing the Refactor

This refactor must be executed **Stage-Locked**; meaning only the active Stage is allowed to modify code.

Rule:

```text
PRIMARY DRAFT -> OBSERVABILITY (when applicable) -> PRIMARY FINAL -> QA -> REVIEW -> CLOSE STAGE -> ACTIVATE NEXT STAGE
```

No Agent is allowed to start the next stage merely because “the next stage is obvious.”

### Allowed States for Each Stage

Normal path:

```text
BLOCKED
  ↓
READY
  ↓
ACTIVE
  ↓
IMPLEMENTED
  ↓
QA_PASSED
  ↓
REVIEW_APPROVED
  ↓
CLOSED
```

Invalidation path:

```text
CLOSED
  ↓ upstream dependency changed/reopened
STALE
  ↓ orchestrator opens a new revision
ACTIVE
  ↓
IMPLEMENTED or REVALIDATION_ONLY
  ↓
QA_PASSED
  ↓
REVIEW_APPROVED
  ↓
CLOSED
```

`STALE` means the Stage was previously closed, but one of its upstream dependencies changed and the previous handoff is no longer sufficient to complete the refactor.

Only `orchestrator` is allowed to mark a Stage `CLOSED` or `STALE`, reopen it, or move execution to the next Stage.

### Start Conditions for Each Agent

Before making any change, every Agent must verify these five conditions:

1. Its own Stage is `ACTIVE`.
2. The previous Stage is `CLOSED`.
3. The target file is within its ownership.
4. The declared dependencies are ready.
5. No other Agent has locked the same file.

If even one condition is not satisfied:

```text
DO NOT MODIFY CODE
RETURN: BLOCKED + reason + required handoff
```

---

# 2) Proposed Mechanism to Prevent Agents from Jumping Ahead

## Shared State File

Orchestrator must use the version-2 control file:

```text
.codex/refactor-state.json
```

The actual file, not an example copied from this document, is authoritative. It contains a record for every Stage with status, revision, attempt, dependencies, milestones, accepted artifacts, supersedes chain, invalidation state, external-authority flags, and transition history.

Only `orchestrator` may change `active_stage`, `stage_revision`, ownership, `write_locks`, `stale_stages`, and `invalidated_by`.

`write_locks` are exact path-level orchestration locks. Agents must check them before editing, and the changed-file list must pass `.codex/scripts/validate_refactor_control.py` before a handoff is accepted.

---

## Handoff artifact

To preserve the “one writer per file” rule, artifacts are split across Primary, Observability, QA, Reviewer, and Orchestrator responsibilities. No Agent updates or overwrites another Agent’s artifact. The complete schema and examples live under `.codex/handoffs/schemas/` and `.codex/handoffs/templates/`.

```text
.codex/handoffs/stage-XX-rev-N-attempt-M-implementation.md
.codex/handoffs/stage-XX-rev-N-attempt-M-implementation.json
.codex/handoffs/stage-XX-rev-N-attempt-M-draft-ready.json  # required when state says observability_required=true
.codex/handoffs/stage-XX-rev-N-attempt-M-observability.json
.codex/handoffs/stage-XX-rev-N-attempt-M-qa.json
.codex/handoffs/stage-XX-rev-N-attempt-M-review.json
.codex/handoffs/stage-XX-rev-N-handoff.md
.codex/handoffs/stage-XX-rev-N-handoff.json
```

### 1. Implementation artifact — Primary Agent only

After implementation and self-tests are complete, the Primary Agent writes these two files:

```text
stage-XX-rev-N-attempt-M-implementation.md
stage-XX-rev-N-attempt-M-implementation.json
```

The Markdown file is for human review and must include at least:

- Stage
- Stage revision
- Primary Agent
- commit / diff summary
- changed files
- self-tests run
- schema/contract diff
- data count, if applicable
- initial leakage result for temporal Stages
- initial latency delta
- known limitations
- out-of-scope findings
- requested changes
- rollback instruction

Minimum schema for the implementation JSON file:

```json
{
  "schema_version": "1.0",
  "artifact_type": "implementation",
  "agent": "financial",
  "stage": "05",
  "stage_revision": 1,
  "attempt": 1,
  "created_at": "2026-08-27T00:00:00Z",
  "status": "IMPLEMENTED",
  "execution_mode": "IMPLEMENTATION",
  "changed_files": [],
  "self_tests_run": [],
  "blocking_issues": [],
  "out_of_scope_findings": [],
  "requested_changes": [],
  "rollback_documented": true,
  "supersedes": null,
  "implementation_ready": true
}
```

The Primary Agent must not later fabricate or update `qa_status` or `review_status` inside its own artifact.

If a Stage becomes `STALE` only because of an upstream change, but the Primary Agent proves through impact analysis that no code change is required, the new revision must still have a new implementation artifact, but with:

```json
{
  "status": "REVALIDATION_ONLY",
  "execution_mode": "REVALIDATION",
  "changed_files": [],
  "revalidation_reason": "upstream revision changed; no local code change required"
}
```

`REVALIDATION_ONLY` is not a way to bypass a Gate; QA and Reviewer re-check the same new revision, and only Orchestrator may mark it `CLOSED` again.

### 2. Observability artifact — `observability` only

When applicable, Observability runs after the Primary draft and before the Primary final artifact. It writes only `stage-XX-rev-N-attempt-M-observability.json`; business-logic findings are change requests to the Primary. The Primary is then reactivated for final self-tests and its final implementation artifact.

### 3. QA artifact — `qa` only

QA reads the final implementation artifact and diff for the same revision and attempt and writes only this file:

```text
stage-XX-rev-N-attempt-M-qa.json
```

Minimum schema:

```json
{
  "schema_version": "1.0",
  "artifact_type": "qa",
  "agent": "qa",
  "stage": "05",
  "stage_revision": 1,
  "attempt": 1,
  "created_at": "2026-08-27T00:00:00Z",
  "qa_status": "PASS",
  "tests_run": [],
  "leakage_report": null,
  "regression_report": null,
  "blocking_issues": [],
  "environment_blockers": []
}
```

QA may not edit the implementation artifact or the final handoff.

### 4. Review artifact — `reviewer` only

Reviewer is activated only after `QA = PASS` and writes this file:

```text
stage-XX-rev-N-attempt-M-review.json
```

Minimum schema:

```json
{
  "schema_version": "1.0",
  "artifact_type": "review",
  "agent": "reviewer",
  "stage": "05",
  "stage_revision": 1,
  "attempt": 1,
  "created_at": "2026-08-27T00:00:00Z",
  "review_status": "APPROVED",
  "blocking_findings": [],
  "non_blocking_findings": [],
  "ownership_violations": [],
  "scope_violations": [],
  "required_rework": [],
  "review_blockers": []
}
```

Reviewer may not edit the implementation, QA, or final handoff artifacts.

### 5. Final Stage Handoff — `orchestrator` only

After reading the matching implementation, optional observability, QA, and review artifacts, Orchestrator consolidates the accepted attempt for that revision into these two files:

```text
stage-XX-rev-N-handoff.md
stage-XX-rev-N-handoff.json
```

The Markdown file is the human-readable Stage summary. The JSON file is the sole machine-readable source of truth for transitions and Stage closure.

Minimum final handoff schema:

```json
{
  "schema_version": "1.0",
  "artifact_type": "handoff",
  "stage": "05",
  "stage_revision": 1,
  "accepted_attempt": 1,
  "created_at": "2026-08-27T00:00:00Z",
  "primary_agent": "financial",
  "implementation_status": "IMPLEMENTED",
  "execution_mode": "IMPLEMENTATION",
  "qa_status": "PASS",
  "review_status": "APPROVED",
  "rollback_documented": true,
  "status": "CLOSED",
  "artifacts": {
    "draft_ready": "stage-05-rev-1-attempt-1-draft-ready.json",
    "implementation": "stage-05-rev-1-attempt-1-implementation.json",
    "observability": "stage-05-rev-1-attempt-1-observability.json",
    "qa": "stage-05-rev-1-attempt-1-qa.json",
    "review": "stage-05-rev-1-attempt-1-review.json"
  },
  "supersedes": null
}
```

Rule:

- Each artifact has exactly one official writer.
- All attempt artifacts must refer to the same `stage_revision` and `attempt`; the handoff records the accepted attempt.
- QA and Reviewer never mutate the Primary Agent artifact.
- Orchestrator never infers results from prose; it consumes only machine-readable artifacts from the same revision and accepted attempt.
- Only Orchestrator writes the final handoff, and no other Agent may edit it.
- On reopen, artifacts from the previous revision are never overwritten or deleted; the new revision creates a new artifact set.
- `supersedes` in the new revision’s final handoff must point to the previous revision’s final handoff.

The next Stage may open only when:

```text
Implementation Artifact = VALID
QA Artifact = PASS
Review Artifact = APPROVED
Rollback = DOCUMENTED
Final Machine Handoff = CLOSED
```

---

# 3) File Ownership Rule

Each file has exactly **one Agent writer** in each Stage.

Other Agents may read that file, but may not edit it unless orchestrator formally transfers ownership.

### Support Agents

These three Agents typically do not own feature code:

- `qa`
- `observability`
- `reviewer`

They must not rewrite the main implementation without authorization.

---

# 4) Overall Refactor Order

Locked order:

```text
Stage 00  Baseline / Backup / Test Runner
    ↓
Stage 01  Forecast Contract / State
    ↓
Stage 02  Database V2
    ↓
Stage 03  Temporal RAG / Vector
    ↓
Stage 04A Temporal Graph / Hybrid Infrastructure
    ↓
Stage 04B Research Point-in-Time
    ↓
Stage 05  Financial Point-in-Time
    ↓
Stage 06  Sentiment Point-in-Time
    ↓
Stage 07  Market Features / Numeric Forecast Model
    ↓
Stage 08  MonthlyForecastAgent / Manager Integration
    ↓
Stage 09  Walk-forward Backtest / Evaluation
    ↓
Stage 10A API / Runtime / Persistence
    ↓
Stage 10B End-to-End Orchestration Integration
    ↓
Stage 11  Cleanup / Rename / Dependency / Docs
    ↓
Stage 11 QA / Reviewer / CLOSE
    ↓
Final QA  (full-system acceptance)
    ↓
Final Reviewer  (cross-stage acceptance)
    ↓
Orchestrator -> REFACTOR_COMPLETE
```

Stages `04` and `10` are intentionally split into two substages so ownership does not get mixed.

## Separating Stage-Serial Refactoring from Runtime Concurrency

The Stage lock applies only to **code changes and refactoring** and must not be confused with the system’s runtime behavior.

```text
Refactor implementation:
SERIAL + STAGE-LOCKED

Runtime execution after Stage 08:
ForecastContext
    ↓
Research ─────┐
Financial ────┼─ may run concurrently
Sentiment ────┘
    ↓ join when required inputs are ready
Market / Feature assembly
    ↓
MonthlyForecast
    ↓
Canonical Synthesis / Publication
```

Rules:

- No Agent becomes a writer for refactor work concurrently with another Stage.
- Runtime concurrency is allowed only after `ForecastContext` has been created.
- Only components whose implementation Stage has already been `CLOSED` may run concurrently at runtime.
- Runtime concurrency does not grant permission to modify Research/Financial/Sentiment code concurrently.
- Manager/Integration must explicitly control readiness and joins; missing/degraded inputs must not be replaced with fabricated values.
- If runtime concurrency requires an upstream contract change, the upstream Stage must be reopened according to the Reopen Rule; Integration may not directly patch upstream code.

---

# 5) Agent → Stage Matrix

| Agent TOML | Primary Stage | Role |
|---|---:|---|
| `orchestrator.toml` | all | Controls Stage, dependency, ownership, reopen/close, and Gates; no feature implementation |
| `core.toml` | 00, 01, 10A + support 02-10 | contract, State, runtime isolation, API foundation; dependency-file support only with an explicit lock |
| `data_rag.toml` | 00 read-only support, 02, 03, 04A | Stage 00 DB baseline export without writes; then DB V2, temporal vector, graph, hybrid infrastructure |
| `research.toml` | 04B | research tools and point-in-time ResearchAgent |
| `financial.toml` | 05 | Financial point-in-time + preservation of DCF/WACC |
| `sentiment.toml` | 06 | temporal sentiment and 7/14/30-day features |
| `quant.toml` | 07, 09 | market features, numeric model, backtest, and metrics |
| `integration.toml` | 08, 10B | MonthlyForecast/Manager integration and E2E connection to the API |
| `qa.toml` | end of each Stage | testing, leakage, regression, contract validation |
| `observability.toml` | alongside each Stage | instrumentation, trace metadata, latency, and run/version metadata |
| `cleanup.toml` | 11 only | rename, dependency, dead code/docs cleanup |
| `reviewer.toml` | after QA for each Stage | independent approval and prevention of scope creep |

---

# 6) `orchestrator.toml`

## Mission

Coordinates the entire refactor and **does not perform feature implementation**.

It must:

- identify the active Stage.
- enforce dependencies.
- enforce file ownership and `write_locks`.
- increment `stage_revision` whenever a Stage is reopened.
- traverse the dependency graph on reopen and mark affected downstream Stages as `STALE`.
- invalidate Final QA/Reviewer if the current revision changes.
- receive change requests and handoffs.
- close the Stage after QA and Reviewer, or return it for rework.
- activate the next Stage only after the current Stage is `CLOSED`.

## Active Stages

```text
00-11: coordination only
```

Orchestrator is not a feature-code writer in Stages 08 or 10B either; that work is delegated to `integration`.

## Exclusive Responsibilities

```text
active_stage
stage_revision
stage status
primary_agent
allowed_support_agents
write_locks
stale_stages / invalidated_by
reopen / re-close
cascade invalidation
QA/Reviewer routing
final stage closure
```

## Prerequisites for Activating Stage 08

All of these must be `CLOSED`:

```text
Stage 01
Stage 04B
Stage 05
Stage 06
Stage 07
```

And these typed outputs must be ready:

```text
ForecastContext
ResearchFeatures
MonthlyFinancialFeatures
SentimentFeatures
MarketFeatures
GraphCoverage
```

Then Orchestrator activates only `integration` for Stage 08.

## Forbidden

- directly editing feature code
- independently changing RAG/Financial/Sentiment/Quant to fix integration
- changing the DB schema without a formal reopen/transfer to `data_rag`
- bypassing QA or Reviewer
- approving its own work as implementation owner
- running Stage 09 before Stage 08 is closed
- changing the numeric prediction through the LLM or orchestration layer

## Done

- Stage state is valid and revision-aware.
- ownership and write locks are recorded without conflicts.
- QA and Reviewer statuses are final.
- rollback is documented.
- the handoff for that same revision is stored.
- the next Stage is activated only after the current one is `CLOSED`.

---

# 7) `core.toml`

## Mission

Build the foundation that all Agents rely on.

If the core contract is unstable, every downstream Agent will invent its own version of time and State, which is exactly the kind of creativity we do not want in a refactor.

## Stage 00

Owner:

```text
pytest.ini
pyproject.toml
requirements.txt
uv.lock
baseline manifest plumbing
test runner fixes
```

Together with QA for:

```text
pytest --collect-only
offline unit baseline
```

Must not:

- create a new Agent.
- create the V2 schema.
- delete legacy data.

## Stage 01

Owner:

```text
core/forecast_contract.py
agents/state.py
core/error_handler.py
core/__init__.py
```

### Mandatory Contract

```text
ticker
as_of
target
horizon_trading_days = 21
lookback_days
benchmark = SPY
analysis_mode
schema_version
dataset_version
feature_version
model_version
```

### Gate

- UTC parsing
- weekend/holiday handling
- target 21 trading days
- future data rejection
- serialization backward compatibility

## Stage 10A

Owns the runtime foundation:

```text
api/config.py
api/core/exceptions.py
api/dependencies.py
api/main.py
memory/manager_memory.py
main.py
```

Route skeletons and the runtime foundation remain with `core`; business integration in 10B must be handed off to `integration`. `langsmith_trace.py` and instrumentation belong to `observability`.

## 10A Prerequisite

Stage 09 must be closed; the API contract must not be stabilized ahead of the model and backtest.

## Forbidden

- changing RAG behavior
- changing financial/sentiment/quant formulas
- cleaning up names before Stage 11
- deleting the legacy schema
- creating a mutable Manager singleton

## Handoff

Stage 01 → `data_rag`

Stage 10A → `integration` for 10B

---

# 8) `data_rag.toml`

## Mission

Create a temporal, versioned data plane.

This Agent is responsible for ensuring historical execution truly sees only the past.

## Stage 02 — Database V2

Owner:

```text
database/supabase/001_alpha_documents_v2.sql
database/supabase/002_forecast_tables.sql
database/supabase/003_analysis_versioning.sql
database/neo4j/001_temporal_evidence.cypher
```

### Tasks

- shadow schema
- dataset_version
- temporal columns
- feature_snapshots
- monthly_forecasts
- forecast_outcomes
- backtest_runs
- shared evidence/document identity
- reconciliation
- rollback pointer/view

### Forbidden

- DROP legacy
- destructive ALTER
- switching production reads before reconciliation
- timestamp unknown -> now

## Stage 03 — Temporal Vector/RAG

Owner:

```text
rag/loader.py
rag/processor.py
rag/embedding_manager.py
rag/vector_store.py
rag/retriever.py
rag/ingestion.py
rag/seed.py
rag/__init__.py
main.py   # only the auto-seed section, coordinated with core
```

### Rules

- historical loader supports `start/end/as_of`
- fail-closed in historical mode
- freshness relative to `ForecastContext.as_of`
- invalid timestamp -> quarantine
- versioned embedding cache
- ingestion checkpoint/resume
- auto seed is forbidden

## Stage 04A — Graph/Hybrid Infrastructure

Owner:

```text
rag/graph_store.py
rag/hybrid_rag.py
```

### Done

- temporal observations instead of an aggregated edge with no history
- document/evidence provenance
- as-of traversal
- vector/graph shared ID
- explicit degradation
- deterministic hybrid fixture

## Gate Required to Activate `research`

```text
DB V2 PASS
Temporal Vector PASS
Temporal Graph PASS
Hybrid fixture PASS
No future evidence PASS
```

## Forbidden

- changing ResearchAgent
- changing sentiment/financial tools
- using web-current in historical behavior
- deleting the legacy namespace

## Handoff

```text
research -> Stage 04B
```

---

# 9) `research.toml`

## Mission

Convert Research into a standard `ResearchFeatures` producer with a hard temporal boundary.

## Stage

Only:

```text
04B
```

## Prerequisite

Stage 04A must be `CLOSED`.

## Owner

```text
tools/research_tools/research_server.py
tools/research_tools/news_search.py
tools/research_tools/sec_edgar.py
tools/research_tools/tavily_search.py
tools/research_tools/comprehensive_analysis.py
tools/research_tools/context_synthesizer.py
agents/research_agent.py
agents/executor_helpers/search_executor_helpers.py
agents/prompts/propt_search_agent.py -> agents/prompts/prompt_research_agent.py  # git mv at the start of 04B
```

## Standard Output

```text
ResearchFeatures
- catalysts
- risks
- timestamps
- evidence_ids
- source diversity
- graph/vector coverage
- missing/degraded flags
- dataset_version
```

## Rules

- ForecastContext is mandatory
- `as_of` must actually flow through the schema and dispatcher
- current web results are forbidden in historical backtests
- SEC results must satisfy `filed_at <= as_of`
- future evidence must be rejected again before synthesis
- citation/version must not be lost during dedupe

## Forbidden

- touching the DB schema
- changing the graph model
- starting Stage 05
- writing MonthlyForecast
- persisting large raw evidence in State

## Gate

- future rejection
- historical boundary fixture
- vector-only
- graph-only
- hybrid
- deterministic evidence IDs

## Handoff

```text
financial -> Stage 05
```

---

# 10) `financial.toml`

## Mission

Make Financial point-in-time without breaking DCF/WACC or the existing 9 intents.

## Stage

```text
05
```

## Prerequisite

Stage 04B must be closed.

## Owner

```text
tools/financial_tools/sec_edgar.py
tools/financial_tools/yahoo_finance.py
tools/financial_tools/financial_ratio_calculator.py
tools/financial_tools/financial_server.py
agents/financial_agent.py
agents/executor_helpers/finance_executor_helper.py
agents/executor_helpers/financial_helper.py
agents/executor_helpers/monthly_financial_features.py
agents/prompts/promt_finance_checker.py -> agents/prompts/prompt_finance_checker.py  # git mv at the start of 05
```

## Output

```text
MonthlyFinancialFeatures
- valuation_gap
- growth
- quality
- leverage
- liquidity
- filing_age
- availability_flags
- missing_flags
- provenance
- version
```

## Fixed Contract

- DCF/WACC must be preserved.
- The existing 9 intents and mapper must be preserved.
- DCF is a long-term feature, not a 21-day price forecast.
- filed date must be separate from period end.
- historical mode must not retrieve the current snapshot.

## Forbidden

- changing the meaning of DCF
- removing the mapper
- fabricating missing values
- touching the Market Model
- starting Sentiment

## Gate

- filings after as_of are not visible
- 9-mapper regression PASS
- DCF/WACC regression PASS
- provenance complete

## Handoff

```text
sentiment -> Stage 06
```

---

# 11) `sentiment.toml`

## Mission

Build temporal, windowed, reproducible sentiment.

## Stage

```text
06
```

## Prerequisite

Stage 05 must be `CLOSED`.

## Owner

```text
tools/sentiment_tools/finbert_analyzer.py
tools/sentiment_tools/vader_scorer.py
tools/sentiment_tools/fear_greed_calculator.py
tools/sentiment_tools/sentiment_server.py
agents/sentiment_agent.py
agents/executor_helpers/sentiment_helper.py
agents/prompts/promt_sentinemt.py -> agents/prompts/prompt_sentiment.py  # git mv at the start of 06
```

## Output

```text
SentimentFeatures
- window_7d
- window_14d
- window_30d
- aggregate
- momentum
- dispersion
- volume
- source_diversity
- confidence
- evidence_ids
- missing_flags
- model_version
```

## Rules

- remove future news
- dedupe before aggregation
- historical fear/greed, if unavailable => missing
- record the FinBERT model version
- do not persist raw chunks/CoT/attempt dumps

## Forbidden

- fabricating a value for missing
- changing Research ingestion
- changing the Quant model
- starting Stage 07

## Gate

- future news must not change a past result
- duplicate resistance
- model/version determinism
- explainable missing behavior

## Handoff

```text
quant -> Stage 07
```

---

# 12) `quant.toml`

## Mission

Owns the entire numeric forecasting and evaluation layer.

The LLM in this Agent is not allowed to generate the forecast number.

## Stage 07 — Market + Model

Owner:

```text
tools/market_tools/__init__.py
tools/market_tools/market_data.py
tools/market_tools/feature_builder.py
tools/market_tools/forecast_model.py
tools/market_tools/market_server.py
```

## Features

```text
momentum_5
momentum_21
momentum_63
volatility
volume_surprise
drawdown
beta
relative_strength
research features
financial features
sentiment features
graph coverage
missing indicators
```

## Model contract

```text
probability_positive
expected_return_21d
expected_excess_return
confidence_interval
model_version
feature_version
```

### Hard Rules

- scaler only on the train fold
- imputer only on the train fold
- rolling only up to as_of
- label from the next day through target
- missing != zero
- deterministic artifact

## Stage 07 Prerequisites

```text
Stage 04B CLOSED
Stage 05 CLOSED
Stage 06 CLOSED
```

## Gate Stage 07

- fixed feature fixture
- exact 21-trading-day target
- leakage zero
- deterministic model

---

## Stage 09 — Walk-forward Backtest

Only after Stage 08 is closed.

Owner:

```text
evaluation/monthly_backtest.py
evaluation/forecast_metrics.py
evaluation/temporal_leakage.py
evaluation/backtest_cli.py
```

And related changes in existing evaluation paths only within the scope of the plan.

### Baselines

```text
always HOLD / class prior
momentum 21d
market-only
without Graph
without Sentiment
without Financial
Full Model
```

### Metrics

```text
balanced accuracy
precision / recall / F1
ROC-AUC
Brier
calibration / ECE
MAE / RMSE
Spearman / IC
hit rate
excess return
turnover
drawdown
Sharpe
coverage
```

## Forbidden

- training on a future fold
- recording an outcome before the target date
- changing Stage 08 orchestration without a return handoff
- cleaning up old evaluation code in Stage 09

## Handoff Stage 09

```text
core -> Stage 10A
```

---

# 13) `integration.toml`

## Mission

Owns the real pipeline integration without taking Stage-control responsibility away from Orchestrator.

This Agent is a writer only in Stages `08` and `10B`.

## Stage 08 — MonthlyForecastAgent / Manager Integration

### Owner

```text
agents/monthly_forecast_agent.py
agents/executor_helpers/monthly_forecast_helper.py
agents/prompts/prompt_monthly_forecast.py
agents/manager_agent.py
agents/executor_helpers/manager_executor_helper.py
agents/prompts/promt_manager_agent.py -> agents/prompts/prompt_manager_agent.py  # git mv at the start of 08
```

### Contract

- consume only shared State and typed outputs from closed Stages.
- do not directly call Research/Financial/Sentiment/Data-RAG/Quant to modify their code.
- the Quant numeric model must remain the source of truth.
- the LLM may only perform synthesis/explanation and may not change the numeric prediction.
- degraded and `INSUFFICIENT_DATA` must be explicit.
- mutable Manager singleton or cross-request state is forbidden.

### Prerequisite

```text
Stage 01 CLOSED
Stage 04B CLOSED
Stage 05 CLOSED
Stage 06 CLOSED
Stage 07 CLOSED
```

### Gate

- complete and incomplete State
- future evidence rejection
- node order
- output schema
- two concurrent in-process Manager executions with no state cross-talk; API-request isolation is gated in 10A/10B
- numeric prediction unchanged by LLM

### Handoff

```text
quant -> Stage 09
```

## Stage 10B — End-to-End Integration

Activated only after Stage 10A is closed.

### Proposed Owner

```text
api/routes/analyze.py
api/routes/progress.py
api/routes/backtest.py
frontend/alpha-agent-app.html
E2E wiring between API, Manager, persistence and progress
```

`api/config.py`, `api/dependencies.py`, `api/main.py`, and the runtime foundation remain owned by `core`. `langsmith_trace.py` and instrumentation belong to `observability`.

### Gate

- API live/historical contract
- persistence idempotency
- run/progress wiring
- success/degraded/insufficient-data paths
- end-to-end smoke

## Forbidden

- changing formulas or feature math in upstream Agents
- changing the DB schema
- changing DCF/WACC semantics
- bypassing Stage reopen for an upstream defect
- cleanup or rename

## Handoff

```text
Stage 08 -> quant Stage 09
Stage 10B -> cleanup Stage 11 after QA/Reviewer approval
```

---

# 14) `qa.toml`

## Mission

QA is a Gatekeeper, not a Feature Developer.

It is activated after implementation of each Stage.

## Order

```text
Primary Agent IMPLEMENTED
        ↓
QA ACTIVE
        ↓
PASS / FAIL / BLOCKED
```

`BLOCKED` means QA cannot test due to an environment/fixture/dependency problem and has not yet judged implementation correctness.

On `FAIL`:

```text
Stage -> ACTIVE
ownership -> Primary Agent
next Stage remains BLOCKED
```

## Fixed QA Responsibilities

### Stage 00

- collect-only
- offline unit baseline
- test inventory
- backup restore evidence

### Stage 01

- contract tests
- State serialization
- future rejection

### Stage 02

- migration empty DB
- staging snapshot
- uniqueness/RLS/RPC
- reconciliation

### Stage 03

- temporal ingestion
- idempotency
- resume
- future record rejection

### Stage 04

- temporal graph traversal
- vector-only
- graph-only
- hybrid

### Stage 05

- mapper regression
- DCF/WACC regression
- filing boundary

### Stage 06

- window
- dedupe
- future news isolation

### Stage 07

- feature fixture
- deterministic model
- leakage

### Stage 08

- State complete/incomplete
- node order
- concurrency
- schema output

### Stage 09

- walk-forward leakage zero
- metric fixtures
- ablation
- deterministic run

### Stage 10

- API live/historical
- request isolation
- persistence idempotency
- smoke

### Stage 11

- import scan
- full regression related to cleanup
- reproducible build
- Docker smoke

### Final QA Mode

This mode is activated only after `Stage 11 = CLOSED` and is not the Gate for any specific Stage.

- full unit suite
- full integration suite
- historical point-in-time pipeline
- zero-future-record leakage audit
- walk-forward backtest smoke
- API live/historical smoke
- request isolation and runtime concurrency
- persistence idempotency
- reproducible build
- Docker smoke
- backup/restore evidence
- final acceptance checklist

If Final QA finds a defect, it reports only `FAIL + responsible_stage_hint`; only Orchestrator may reopen a Stage.

## Forbidden

- rewriting implementation to “make the test pass”
- deleting a failing test without reviewer approval
- approving a flaky test without a cause
- opening the next Stage

## Output

```text
QA_STATUS = PASS | FAIL | BLOCKED
test_report
leakage_report
regression_report
blocking_issues
environment_blockers
responsible_stage_hint   # only for Final QA on FAIL
candidate_stale_downstream_stages
```

---

# 15) `observability.toml`

## Mission

Observability for the refactor without storing sensitive data or chain-of-thought.

## Activation Timing

It may be activated starting in Stage 00, but it works only on the active Stage and does not perform feature implementation.

## Instrumentation Ownership

```text
core/observability.py
core/progress_bus.py
langsmith_trace.py
trace instrumentation owned explicitly for the active stage
latency measurement
run/version metadata
```

This ownership is intentionally separated from `core` so the same file never has two writers at once. `core` owns contract/runtime; `observability` owns instrumentation.

## Required Metadata

```text
run_id
ticker
as_of
target
horizon
analysis_mode
dataset_version
feature_version
model_version
schema_version
stage
stage_revision
agent
latency
evidence_count
degraded_flags
```

## Must Not Be Stored

```text
secret
token
headers
chain-of-thought
brain_reasoning
raw article chunks
full filing text
duplicated tool payload
```

## Ownership Limitation

Observability may not change business logic or contracts merely to improve tracing.

If instrumentation requires a contract change:

```text
return BLOCKED
create change request for current owner
do not edit owner file directly
```

In Stage 10A, `core` may prepare the config/runtime needed for optional Sentry/LangSmith, but implementation and trace wiring for observability files still have a separate writer.

---

# 16) `cleanup.toml`

## Mission

Stage 11 only.

This Agent is intentionally BLOCKED until the end.

## Start condition

Stage 11 is activated only when the previous Stages are genuinely closed and valid:

```text
00-10B CLOSED
stale_stages = []
Stage 10B QA = PASS
Stage 10B Reviewer = APPROVED
Backtest PASS
API smoke PASS
```

`Final QA` is not run at this point. Final QA is intentionally activated only **after Stage 11 is CLOSED** and performs full-system acceptance.

## Owner

Renames:

```text
tests/unit_tests/test_rag/test_hybried.py
    -> test_hybrid.py

tests/unit_tests/test_rag/test_vectore_store.py
    -> test_vector_store.py

tests/unit_tests/test_rag/test_pipline_rag.py
    -> test_pipeline_rag.py

tests/unit_tests/sentiment_test/test_finbert_anayzer.py
    -> test_finbert_analyzer.py

tests/unit_tests/api_test/test_core_eceptions.py
    -> test_core_exceptions.py

tests/unit_tests/test_supbase/
    -> test_supabase/

tests/unit_tests/test_memory/__ini__.py
    -> __init__.py
```

### Explicit Ownership for Non-Rename Cleanup

```text
Dockerfile
package-lock.json
env.example
README.md
READ.md
infra/ecr-stack.yaml
infra/ecs-stack.yaml
infra/oidc-role-stack.yaml
tests/integration_tests/**
tests/integration_test_2/**
```

The prompt renames for 04B/05/06/08 are not performed in Stage 11; Stage 11 only verifies that references to the old names are zero.

## Rules

- rename with `git mv`
- each rename batch in a separate commit
- after each batch:
  - test collection
  - unit
  - import scan
- delete a duplicate suite only after coverage comparison
- delete the legacy DB only after the migration-plan conditions are met
- cleanup and behavior changes must not be mixed in one commit

## Forbidden

- any cleanup before Stage 11
- deleting legacy code/data based on guesswork
- rewrite feature code
- refactor formula
- changing prediction behavior

## Gate

```text
old import references = 0
full unit = PASS
integration = PASS
build reproducible = PASS
Docker smoke = PASS
```

---

# 17) `reviewer.toml`

## Mission

Reviewer is the final barrier before each Stage is closed.

QA answers: “Did the tests pass or fail?”
Reviewer answers: “Was the correct work actually done, within the correct scope, under the correct contract?”

## Activation Order

```text
Primary IMPLEMENTED
    ↓
QA PASSED
    ↓
Reviewer ACTIVE
    ↓
APPROVED / REJECTED / BLOCKED
    ↓
Orchestrator closes stage only on APPROVED
```

## Reviewer Checklist

### Scope

- Did the Agent modify only what it owns?
- Were any changes made outside the Stage?
- Was cleanup performed prematurely?

### Temporal safety

- Is `as_of` actually enforced?
- Did a current fallback enter a historical run?
- Was an invalid timestamp converted to now?

### Versioning

- dataset_version
- feature_version
- model_version
- schema_version

### Data hygiene

- Was duplicate raw evidence avoided in persistence?
- Was CoT/brain_reasoning not stored?
- Was legacy content not deleted prematurely?

### Regression

- Were previous contracts preserved?
- Were DCF/WACC semantics preserved?
- Were all 9 financial mappers preserved?

### Operational safety

- Is rollback documented?
- Is the migration reversible?
- Was no mutable Manager singleton created?
- Were Stage artifacts stored?

### Final Reviewer Mode

After `Final QA = PASS`, Reviewer cross-checks all Stages in final mode, not just Stage 11.

- the revision chain and `supersedes` for all Stages are valid.
- implementation/observability/QA/review/final-handoff for each Stage have separate writers and matching revision/attempt references.
- no final handoff was written by any Agent other than Orchestrator.
- the final Acceptance Gate is complete.
- no known defect remains between Stages with an ambiguous status.

## Reviewer Output

```text
REVIEW_STATUS = APPROVED | REJECTED | BLOCKED
blocking_findings
non_blocking_findings
ownership_violations
scope_violations
required_rework
review_blockers
```

Reviewer **may not directly edit implementation**.

---

# 18) Exact Agent Order

## Stage 00

```text
orchestrator
    ↓ activate
core
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 01

```text
orchestrator
    ↓
core
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 02

```text
orchestrator
    ↓
data_rag
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 03

```text
orchestrator
    ↓
data_rag
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 04A

```text
orchestrator
    ↓
data_rag
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 04B

```text
orchestrator
    ↓
research
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 05

```text
orchestrator
    ↓
financial
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 06

```text
orchestrator
    ↓
sentiment
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 07

```text
orchestrator
    ↓
quant
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 08

```text
orchestrator
    ↓ activate
integration
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

None of:

```text
research
financial
sentiment
data_rag
quant
```

may modify their own code in Stage 08 unless a Stage is formally reopened because of a defect.

---

## Stage 09

```text
orchestrator
    ↓
quant
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 10A

```text
orchestrator
    ↓
core
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 10B

```text
orchestrator
    ↓ activate
integration
    ↓
observability
    ↓
qa
    ↓
reviewer
    ↓
orchestrator CLOSE
```

---

## Stage 11

```text
orchestrator
    ↓
cleanup
    ↓
qa                 # only the cleanup Gate
    ↓
reviewer           # only the Stage 11 review
    ↓
orchestrator CLOSE Stage 11
```

`Stage 11 QA` is not the same as `Final QA`. Stage 11 QA only checks that cleanup, rename, dependency, and docs changes were completed without regression. After Stage 11 is `CLOSED`, an independent full-system acceptance is run.

## Final QA — Full-System Acceptance

Activated only when **all Stages 00 through 11 are CLOSED**.

```text
orchestrator
    ↓ activate FINAL_QA
qa (FINAL MODE)
    ↓
PASS / FAIL / BLOCKED
```

Final QA does not perform feature implementation and at minimum runs/validates the following:

```text
full unit suite
full integration suite
historical point-in-time pipeline
zero-future-record leakage audit
walk-forward backtest smoke
baseline / ablation availability
API live + historical smoke
request isolation / runtime concurrency
persistence idempotency
reproducible build
Docker smoke
backup / restore evidence
final acceptance checklist
```

If Final QA returns `FAIL`, QA reports only the defect and `responsible_stage_hint` and may not directly patch the implementation.

Orchestrator must:

```text
1. Reopen the responsible Stage with a new revision.
2. Mark all downstream Stages that depend on it directly or transitively as STALE.
3. Invalidate the previous Final QA and Final Reviewer.
4. Re-IMPLEMENT or REVALIDATE STALE Stages in dependency order.
5. After all Stages are closed again, rerun Final QA from the beginning.
```

No `REFACTOR_COMPLETE` is allowed while `stale_stages != []`.

Final QA output:

```text
.codex/handoffs/final-qa-rev-N.json
```

Example:

```json
{
  "schema_version": "1.0",
  "artifact_type": "final_qa",
  "stage": "FINAL_QA",
  "stage_revision": 1,
  "created_at": "2026-08-27T00:00:00Z",
  "qa_status": "PASS",
  "closed_stages_verified": ["00", "01", "02", "03", "04A", "04B", "05", "06", "07", "08", "09", "10A", "10B", "11"],
  "full_unit": "PASS",
  "full_integration": "PASS",
  "leakage_audit": "PASS",
  "backtest_smoke": "PASS",
  "api_smoke": "PASS",
  "runtime_concurrency": "PASS",
  "reproducible_build": "PASS",
  "docker_smoke": "PASS",
  "blocking_issues": []
}
```

## Final Reviewer — Cross-Stage Acceptance

Activated only after `Final QA = PASS`. This review is no longer stage-specific and checks consistency across the entire refactor:

- all final Stage handoffs and the revision chain are valid.
- no closed Stage has incomplete artifacts or an ambiguous supersedes relationship.
- ownership and protected-path policy were respected throughout the full history.
- temporal safety and versioning are consistent across Stages.
- DCF/WACC is not mixed with the 21-day forecast.
- the numeric model remains the source of truth.
- legacy is removed only according to the plan’s conditions.
- the Final Acceptance Gate is complete.

Output:

```text
.codex/handoffs/final-review-rev-N.json
```

```json
{
  "schema_version": "1.0",
  "artifact_type": "final_review",
  "stage": "FINAL_REVIEW",
  "stage_revision": 1,
  "created_at": "2026-08-27T00:00:00Z",
  "review_status": "APPROVED",
  "blocking_findings": [],
  "cross_stage_findings": [],
  "acceptance_gate_verified": true
}
```

Only if all of these conditions hold may Orchestrator change the final status:

```text
stale_stages = []
all current Stage revisions = CLOSED
Final QA = PASS
Final Reviewer = APPROVED
```

Then:

```text
REFACTOR_COMPLETE
```

Final QA and Final Reviewer must have run against the **current revisions**; an artifact for a superseded revision is not valid for completing the refactor.

---

# 19) Dependency Graph

```text
core:forecast_contract
      |
      v
data_rag:db_v2
      |
      v
data_rag:temporal_vector
      |
      v
data_rag:temporal_graph_hybrid
      |
      v
research:point_in_time
      |
      v
financial:point_in_time
      |
      v
sentiment:point_in_time
      |
      v
quant:features_model
      |
      v
integration:monthly_pipeline
      |
      v
quant:walk_forward_backtest
      |
      v
core:api_runtime
      |
      v
integration:e2e_integration
      |
      v
cleanup
```

---

# 20) Rule for Reopen, STALE, and Cascade Invalidation

## Principle

If a closed Stage (`CLOSED`) is reopened and its contract/data/behavior/version output changes, all downstream Stages that depend on it directly or transitively are no longer automatically valid.

Orchestrator must mark them `STALE`.

```text
UPSTREAM CLOSED
    ↓ reopen + new revision
UPSTREAM ACTIVE
    ↓
dependent CLOSED stages
    ↓
STALE
```

`STALE` does not mean the implementation failed; it means the Stage handoff was built against an old dependency revision and must be revalidated.

## Example

In Stage 07, Quant discovers that `SentimentFeatures` does not contain a required field.

Quant **must not** modify `sentiment_agent.py`.

It must:

```text
quant -> BLOCKED + change request
orchestrator -> pause Stage 07
orchestrator -> reopen Stage 06
orchestrator -> increment Stage 06 revision
sentiment -> fix within transferred write locks
qa -> rerun Stage 06 gate
reviewer -> approve
orchestrator -> write superseding Stage 06 handoff
orchestrator -> re-close Stage 06
orchestrator -> reactivate Stage 07
```

If Stage 06 was already `CLOSED` and downstream Stages were also closed, reopening it must traverse the dependency graph and mark affected Stages as `STALE`.

Conceptual example:

```text
Stage 05 Financial rev1 CLOSED
Stage 06 Sentiment rev1 CLOSED
Stage 07 Quant rev1 CLOSED
Stage 08 Integration rev1 CLOSED
Stage 09 Backtest rev1 CLOSED

Stage 05 reopened -> rev2
        ↓
Orchestrator impact analysis
        ↓
affected downstream stages -> STALE
```

## Two Paths for STALE Stages

### A) A Local Change Is Required

```text
STALE
→ new revision
→ ACTIVE
→ Primary Agent modifies code
→ status = IMPLEMENTED
→ QA
→ Reviewer
→ Orchestrator CLOSE
```

### B) No Local Change Is Required

The Primary Agent must record impact analysis for that same revision:

```text
STALE
→ new revision
→ ACTIVE
→ no code change
→ status = REVALIDATION_ONLY
→ changed_files = []
→ QA reruns required gates
→ Reviewer approves compatibility
→ Orchestrator CLOSE
```

A STALE Stage cannot be made CLOSED again merely by changing its status.

## Revalidation Order

STALE Stages must be closed again in dependency order:

```text
nearest downstream dependency
→ next downstream
→ ...
→ final downstream
```

No downstream Stage may reopen/revalidate before its upstream dependency becomes valid again.

## Revision chain

If Stage 06 was previously closed at revision 1, reopening it must create revision 2 and a new final handoff with:

```text
supersedes: stage-06-rev-1-handoff.json
```

included.

Previous artifact:

```text
NEVER DELETE
NEVER OVERWRITE
```

## Final QA / Reviewer invalidation

If a reopen occurs after Final QA or Final Reviewer:

```text
final-qa-rev-N.json     -> SUPERSEDED / no longer current
final-review-rev-N.json -> SUPERSEDED / no longer current
```

Final QA and Final Reviewer must be rerun from the beginning after all STALE Stages are closed again.

## Final Rule

```text
REFACTOR_COMPLETE requires:
stale_stages = []
all current stage revisions = CLOSED
Final QA = PASS against current revisions
Final Reviewer = APPROVED against current revisions
```

The same rule applies to all dependencies.

---

# 21) Rule for File Conflicts

If two Agents need the same file:

```text
1. current owner keeps write access
2. second agent creates change request
3. orchestrator decides:
   A) defer change
   B) transfer ownership
   C) create interface instead
4. never edit concurrently
```

Files that are conflict-prone:

```text
agents/state.py
agents/manager_agent.py
api/main.py
api/dependencies.py
main.py
pyproject.toml
requirements.txt
uv.lock
agents/__init__.py
tools/__init__.py
rag/__init__.py
```

For these paths, orchestrator must record an explicit owner and `write_locks` in `.codex/refactor-state.json`. The second Agent only submits a change request and does not become a writer until ownership is formally transferred.

## Protected Shared Paths

The following shared files are outside normal Agent ownership and **must not become implicitly editable through broad globs** such as `**/*.py` or package-level ownership:

```text
main.py
pyproject.toml
requirements.txt
uv.lock
agents/__init__.py
agents/prompts/__init__.py
tools/__init__.py
rag/__init__.py
api/main.py
api/dependencies.py
```

Rule for these paths:

```text
default_access = READ_ONLY
write_requires = EXPLICIT_ORCHESTRATOR_LOCK
implicit_glob_ownership = FORBIDDEN
concurrent_writers = 0
```

Before any write, the exact path and owner must be recorded in state:

```json
{
  "write_locks": {
    "pyproject.toml": "core"
  }
}
```

An Agent without an explicit lock:

```text
DO NOT EDIT
CREATE CHANGE REQUEST
```

Ownership transfer must be recorded before editing, not after a diff already exists. “It was only one line” is not an exception.

---

# 22) Rule for Dependency Files

Files:

```text
pyproject.toml
requirements.txt
uv.lock
```

must not be freely modified by all Agents.

Rule:

- An Agent reports a dependency need only as a change request.
- `core` owns dependency management through Stage 10, but still requires an explicit lock on the relevant file for every change.
- In Stage 11, `cleanup` may canonicalize them after formal ownership transfer.
- `requirements.txt` and `pyproject.toml` are excluded from implicit ownership by feature Agents.
- `uv.lock` must not be edited manually and must only be regenerated from the canonical source.
- dependency changes must not be mixed with unrelated feature work.
- No Agent that owns a directory gets implicit write access to these three files.

---

# 23) Rule for Database

Only `data_rag` owns schema migration in Stage 02.

Agents:

```text
research
financial
sentiment
quant
orchestrator
```

If they need a new column/table, they must submit a migration request.

None of them may perform ad-hoc migrations.

---

# 24) Rule for Tests

Primary Agent:

- writes/updates unit tests for its own feature.

`qa`:

- cross-module gate
- integration
- regression
- leakage
- concurrency
- contract verification

Reviewer:

- checks for tests being deleted or weakened.
- if QA cannot run because of environment/fixture/dependency issues, status must remain `BLOCKED`; no fabricated PASS or FAIL.

---

# 25) Rule for Commits

Suggested:

```text
stage-01/core-contract
stage-02/db-v2
stage-03/temporal-rag
stage-04a/temporal-graph
stage-04b/research
stage-05/financial
stage-06/sentiment
stage-07/quant-model
stage-08/monthly-orchestration
stage-09/backtest
stage-10a/api-runtime
stage-10b/e2e
stage-11/cleanup
```

Within each Stage:

```text
Commit A: implementation
Commit B: tests/gate fixes
Commit C: optional observability
```

Keep cleanup renames separate.

---

# 26) Shared Start/Done Definition for TOMLs

It is recommended that each TOML contain these logical fields:

```text
name
mission
allowed_stages
owned_paths
read_only_paths
depends_on
start_conditions
required_inputs
required_outputs
forbidden_actions
qa_gate
handoff_to
stop_conditions
stage_revision
write_lock_policy
protected_paths_policy
change_request_policy
out_of_scope_findings_policy
machine_handoff_schema
runtime_concurrency_policy
stale_invalidation_policy
revalidation_policy
```

### Shared start_conditions

```text
active_stage matches allowed_stages
previous_stage == CLOSED
dependencies == READY
dependencies contain no STALE current revision
file_ownership == CONFIRMED
protected_paths == READ_ONLY_UNLESS_EXPLICITLY_LOCKED
write_locks == CONFIRMED
stage_revision == CURRENT
```

### stop_conditions by Role

An Agent must not wait for the next role’s artifact. Each role stops after delivering its own artifact.

#### Primary Agent

```text
implementation or revalidation complete
self-tests complete
out-of-scope findings recorded
implementation MD/JSON written with current revision
STOP
```

#### QA

```text
valid implementation artifact exists for current revision and attempt
required QA gates executed
attempt-specific qa.json written for current revision and attempt
STOP
```

#### Reviewer

```text
QA = PASS for current revision and attempt
attempt-specific review.json written for current revision and attempt
STOP
```

#### Orchestrator

```text
implementation artifact valid
QA artifact = PASS
review artifact = APPROVED
rollback documented
all artifact revisions and accepted-attempt references match
final MD/JSON handoff written
stage status = CLOSED
STOP / activate next allowed stage
```

#### Observability

```text
instrumentation for active stage complete
no business logic modified
observability findings attached/referenced
STOP
```

No Primary Agent should remain running while waiting for `qa.json`, `review.json`, or the final handoff to be completed.

---

# 27) Policy: “No Agent Starts the Next Work Early”

This text should appear in almost every Agent TOML:

```text
You are stage-locked.

Do not begin work assigned to a later stage.
Do not modify files owned by another agent.
Verify current stage_revision and write_locks before editing.
Do not anticipate or pre-implement downstream work.
Do not treat runtime concurrency as permission for concurrent refactor writes.
If your stage is STALE, do not silently reuse the old handoff; wait for a new revision and perform implementation or explicit revalidation.
When your current stage deliverables are complete, stop and produce only the artifact(s) assigned to your role; never update another role's artifact.
Wait for orchestrator activation before any further implementation.

If a downstream dependency appears necessary, report it as a handoff/blocker/change request.
Do not implement it yourself.
```

For `cleanup`, make it even stricter:

```text
Remain blocked until Stage 11 is explicitly activated.
```

---

# 28) Final Acceptance Gate

Before the refactor is complete:

```text
[ ] monthly point-in-time cutoffs exist
[ ] 21-trading-day label is correct
[ ] benchmark = SPY is specified
[ ] leakage audit = zero future records
[ ] Full Model vs baselines
[ ] Full Model vs ablations
[ ] calibration metrics
[ ] amount error metrics
[ ] investment metrics
[ ] coverage metrics
[ ] every forecast versioned
[ ] historical run reproducible by manifest
[ ] DCF/WACC clearly separated from 21d forecast
[ ] two successful shadow runs before legacy cleanup
[ ] backup restore tested
[ ] no cross-request mutable Manager state
[ ] no CoT/raw duplicated evidence persistence
[ ] integration.toml exists and owns only Stage 08/10B integration work
[ ] every closed Stage has implementation + QA + review + final handoff artifacts for current revision
[ ] every final machine handoff was written only by orchestrator
[ ] no QA/Reviewer artifact mutates Primary Agent artifact
[ ] protected shared paths have no implicit/glob writer
[ ] runtime concurrency is tested without relaxing refactor write locks
[ ] stale_stages = []
[ ] every reopened Stage has a revisioned supersedes chain
[ ] every STALE downstream Stage was reimplemented or REVALIDATION_ONLY with fresh QA/Reviewer
[ ] Stage 11 QA is complete and distinct from Final QA
[ ] Final QA = PASS against current Stage revisions
[ ] Final Reviewer = APPROVED against current Stage revisions
[ ] final status = REFACTOR_COMPLETE
```

---

# 29) Practical Summary

Primary order of feature Agents:

```text
core
→ data_rag
→ research
→ financial
→ sentiment
→ quant
→ integration(Stage 08)
→ quant(backtest)
→ core(API foundation)
→ integration(E2E)
→ cleanup
```

At the end of **each stage**:

```text
qa
→ reviewer
→ orchestrator approval
```

`observability` moves only alongside the active Stage and may not jump ahead or change business logic.

`orchestrator` is intentionally not the implementation owner; Stages 08 and 10B are handled by `integration` so the Agent controlling the Gate is not also implementing the same feature.

After Stage 11 is closed, the same `qa` and `reviewer` are activated in **Final Mode**; Final QA tests the entire system and Final Reviewer validates cross-Stage consistency and the Acceptance Gate. Only Orchestrator may then record `REFACTOR_COMPLETE`.

The five final locks in this version:

```text
1. Stage lock          -> no premature implementation
2. Explicit path lock  -> no concurrent writer on a shared file
3. Revisioned handoff  -> no ambiguous reopen or history overwrite
4. MD + JSON artifacts -> human review + machine-readable transition
5. STALE cascade       -> no upstream change passes without downstream revalidation
```

Runtime concurrency is only behavior of the final system and does not loosen any of these five refactor locks.

This structure is intentionally somewhat strict, because a temporal/financial refactor is not a place where “we’ll test it later” is a healthy sentence.
