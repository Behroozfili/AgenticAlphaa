# Stage 04B Revision 1 Attempt 1 — Primary Final

Status: **IMPLEMENTED**. Observability findings `OBS-04B-001..004` are resolved and the current bytes are ready for QA.

## Outcome

- `ForecastContext` and `run_id` are mandatory across live and historical Research execution.
- Historical mode admits only retained, versioned providers and fails with `DATA_UNAVAILABLE` without a current-web fallback.
- SEC and all other evidence pass `timestamp <= as_of` eligibility before synthesis and again before State materialization.
- Deterministic evidence/citation reference identity and typed compact `ResearchFeatures` preserve attribution without storing article, filing, citation or synthesis payloads in Shared State.
- LangSmith processors, MCP failures, logs, progress and Sentry expose allowlisted identity, measured latency, counts, degradation and stable codes only.
- The prompt rename to `prompt_research_agent.py` is atomic and collection-safe.

## Verification

- Stage 04B focused unit/sentinel suite: **104 passed**.
- Stage 03/04A Graph/RAG regression: **333 passed**.
- Full unit/integration collection: **1308 collected, zero errors**.
- Scoped `py_compile`, `git diff --check`, static privacy scan and control/lock validation: **PASS / CONTROL VALID**.

The deterministic sentinel matrix covers provider/search, SEC, Vector, Graph, Hybrid, comprehensive analysis, synthesis, both leakage gates, `DATA_UNAVAILABLE`, future/provider rejection, fail-closed exceptions, LangSmith, logs, progress, Sentry and reference-only State.

## Operational note

No database write, migration, seed, archive download, deployment, commit or push occurred. One legacy live integration probe unexpectedly used configured News/SEC providers for read-only requests before it was stopped; it was not used as a gate and was not rerun. Live-provider validation remains outside this offline completion contract and requires separate authorization.

Stage 05 remains `PENDING / NOT_STARTED`.
