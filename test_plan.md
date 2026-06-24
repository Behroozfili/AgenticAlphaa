# AgenticAlpha — Comprehensive Test Plan
**Date:** 2026-06-19  
**Scope:** All source files under `D:/AgenticAlpha/`  
**Framework:** `pytest` + `unittest.mock` / `pytest-mock`  
**Mode:** READ-ONLY planning document. No test files were created or modified.

---

## Table of Contents

1. [Root](#root)
2. [agents/](#agents)
3. [memory/](#memory)
4. [core/](#core)
5. [api/](#api)
6. [rag/](#rag)
7. [tools/financial_tools/](#toolsfinancial_tools)
8. [tools/research_tools/](#toolsresearch_tools)
9. [tools/sentiment_tools/](#toolssentiment_tools)
10. [evaluation/](#evaluation)
11. [scheduler/](#scheduler)
12. [Summary](#summary)

---

## Root

### `main.py`

**Not Testable**  
**Reason:** Contains only a trivial stub function (`main()`) that prints a hardcoded string and exits. It is dead code — the real application entry point is `api/main.py`. There is no business logic, no branching, and no meaningful state to assert on.

---

## agents/

### `agents/__init__.py`

**Not Testable**  
**Reason:** Pure re-export facade with no logic. Only imports and `__all__` declarations. Testing that imports resolve is covered by any test that imports from `agents.*`.

---

### `agents/state.py`

**Purpose:** Single source of truth for all cross-agent TypedDicts (`SharedManagerState`, `ResearchAgentState`, `FinancialAgentState`, `SentimentAgentState`, `ManagerGraphState`, `EvaluationSnapshot`). Defines the data contract between agents.

**Not Testable (as unit tests)**  
**Reason:** Contains only TypedDict class declarations and no executable logic, functions, or methods. TypedDicts are structural type hints — they carry no runtime behavior to assert on. Schema correctness is validated at the type-checker level (mypy/pyright), not at runtime with pytest.

> **Note for future work:** If the project adds runtime validators or `__post_init__` style checks to these state types, tests should be added at that point.

---

### `agents/research_agent.py`

**Purpose:** LangGraph-based 3-node agent (Brain → Executor → Checker) that collects web, news, SEC, and RAG context chunks for a given investment query, looping until context is complete or `max_loops` is reached.

---

#### Proposed Test Cases

**Test Group 1 — `ResearchAgent.__init__`**

- **TC-R1.1 — Default construction (no injected client)**
  - What: Instantiate `ResearchAgent()` with no arguments; assert `_llm` is a real `anthropic.Anthropic` instance.
  - Why: Ensures the agent boots without configuration when injected client is omitted.
  - Assertions: `isinstance(agent._llm, anthropic.Anthropic)` is `True`.

- **TC-R1.2 — Injected LLM client is used**
  - What: Pass a `MagicMock` as `llm_client`; assert `agent._llm` is that same mock.
  - Why: Verifies the injection path so tests can control API calls without network.
  - Assertions: `agent._llm is mock_client`.

- **TC-R1.3 — Injected MCP server params are used**
  - What: Pass a `StdioServerParameters` mock; assert it is stored as `_mcp_server_params`.
  - Why: Verifies MCP transport injection for hermetic testing.

---

**Test Group 2 — `_brain_node()`**

- **TC-R2.1 — Happy path: Brain returns valid JSON action plan**
  - What: Mock `agent._llm.messages.create` to return a response with JSON containing `[{"tool_name": "tavily_search", "args": {...}}]`. Call `_brain_node()` directly.
  - Why: Core routing logic — verifies the node parses the LLM response and populates `state["brain_action_plan"]`.
  - Assertions: `state["brain_action_plan"]` is a non-empty list; first entry has `tool_name` key.

- **TC-R2.2 — LLM returns malformed JSON**
  - What: Mock LLM to return `"sorry I can't help"` (non-JSON). Call `_brain_node()`.
  - Why: The Brain must degrade gracefully — it should either retry or return a safe default, not raise.
  - Assertions: No exception raised; `state["brain_action_plan"]` is either a default plan or an empty list.

- **TC-R2.3 — LLM raises `anthropic.APIError`**
  - What: Mock `messages.create` to raise `anthropic.APIError`.
  - Why: Network failure resilience — the agent loop must survive transient API errors.
  - Assertions: `_brain_node()` does not propagate the exception; state contains a recoverable fallback.

- **TC-R2.4 — Loop counter is injected into system prompt**
  - What: Verify the `loop_counter` value appears in the system prompt content sent to the LLM.
  - Why: The Brain prompt includes loop count to adjust strategy on reruns.

---

**Test Group 3 — `_executor_node()`**

- **TC-R3.1 — Happy path: tool call succeeds and chunk is appended**
  - What: Mock the MCP `ClientSession` / stdio transport; mock a successful `call_tool` response with content `[TextContent(text="Apple revenue grew...")]`. Call `_executor_node()`.
  - Why: The Executor's main job is to dispatch MCP calls and accumulate chunks.
  - Assertions: `len(state["context_chunks"]) > 0`; chunk text matches mock response.

- **TC-R3.2 — Single tool call fails; execution continues**
  - What: Provide an action plan with 2 tools; make the first `call_tool` raise `Exception`. Call `_executor_node()`.
  - Why: Individual tool failures must not abort the entire loop — the agent should skip and continue.
  - Assertions: No exception propagated; `state["context_chunks"]` contains an error-marker chunk for the failed tool; second tool still attempted.

- **TC-R3.3 — Empty action plan**
  - What: Set `state["brain_action_plan"] = []`. Call `_executor_node()`.
  - Why: Edge case — Brain may return empty plan on first warm-up loop.
  - Assertions: No exception; `state["context_chunks"]` unchanged.

- **TC-R3.4 — Unknown tool name in action plan**
  - What: Provide `{"tool_name": "nonexistent_tool", "args": {}}` in action plan.
  - Why: MCP will raise on unknown tool; must be handled gracefully.
  - Assertions: Error chunk appended; loop does not crash.

---

**Test Group 4 — `_checker_node()`**

- **TC-R4.1 — Checker returns `is_complete: true`**
  - What: Mock LLM to return `{"is_complete": true, "feedback": "sufficient"}`. Call `_checker_node()`.
  - Why: Happy path — verifies state update `state["is_complete"] = True`.

- **TC-R4.2 — Checker returns `is_complete: false`**
  - What: Mock LLM to return `{"is_complete": false, "feedback": "need more data"}`. Call `_checker_node()`.
  - Why: Normal loop continuation — state should reflect incomplete.
  - Assertions: `state["is_complete"]` is `False`; `state["checker_feedback"]` populated.

- **TC-R4.3 — Checker LLM returns malformed JSON**
  - What: Mock LLM to return plain text. Call `_checker_node()`.
  - Why: LLM output is non-deterministic; checker must default to "not complete" to avoid premature termination.
  - Assertions: `state["is_complete"]` is `False` (safe default).

---

**Test Group 5 — `_should_continue()` conditional edge**

- **TC-R5.1 — `loop_counter >= max_loops` → END (guardrail)**
  - What: Set `state["loop_counter"] = 5`, `state["max_loops"] = 5`, `state["is_complete"] = False`.
  - Why: Loop must terminate at guardrail to prevent infinite cycles.
  - Assertions: `_should_continue(state)` returns `"end"` (or equivalent LangGraph literal).

- **TC-R5.2 — `is_complete=True` and loops remaining → END**
  - What: `loop_counter=2`, `max_loops=5`, `is_complete=True`.
  - Assertions: Returns `"end"`.

- **TC-R5.3 — `is_complete=False` and loops remaining → brain (continue)**
  - What: `loop_counter=1`, `max_loops=5`, `is_complete=False`.
  - Assertions: Returns `"brain"`.

---

**Test Group 6 — `run()` integration (end-to-end with mocks)**

- **TC-R6.1 — Full graph invocation maps results to `SharedManagerState`**
  - What: Mock `_graph.ainvoke()` to return a complete `ResearchAgentState`. Call `run()`.
  - Why: Verifies the input/output mapping between public and private state shapes.
  - Assertions: Returned `SharedManagerState["aggregated_research_context"]` is a string; `SharedManagerState["agent_execution_history"]` updated.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `anthropic.Anthropic.messages.create` | Avoid real API calls; control LLM responses |
| `mcp.client.stdio.stdio_client` | Avoid spawning real subprocess; inject mock tool responses |
| `langsmith.traceable` decorator | Replace with identity to avoid tracing side effects in tests |
| `sentry_sdk.capture_exception` | Assert error capture without real Sentry |

---

### `agents/financial_agent.py`

**Purpose:** Financial analyst agent using a manual loop (Brain → Data Extraction → Ratio Computation → Checker) to gather ticker-specific financial metrics via MCP. Writes results to `SharedManagerState["financial_metrics_summary"]`.

---

#### Proposed Test Cases

**Test Group 1 — `_extract_ticker()`**

- **TC-F1.1 — Ticker resolved from `manager_directives`**
  - What: Set `state["manager_directives"]["ticker"] = "AAPL"`. Call `_extract_ticker(state)`.
  - Why: Highest-priority path; directives are explicit user input.
  - Assertions: Returns `"AAPL"`.

- **TC-F1.2 — Ticker resolved from regex scan of `task_query`**
  - What: `manager_directives = {}`, `task_query = "Analyze Microsoft MSFT Q3 performance"`. Call `_extract_ticker(state)`.
  - Why: Fallback path must correctly extract ticker from free-form query.
  - Assertions: Returns `"MSFT"`.

- **TC-F1.3 — Stop-word filter prevents false positive ticker**
  - What: `task_query = "What is the AI sector outlook for Q2?"`. Call `_extract_ticker(state)`.
  - Why: Words like `AI`, `Q2`, `THE`, `I`, `A` are valid ticker patterns but must be filtered.
  - Assertions: Returns `None` or falls through to next resolution level.

- **TC-F1.4 — No ticker found anywhere → returns `None`**
  - What: All resolution paths empty; assert graceful `None` return without raising.
  - Why: Downstream code must handle missing ticker without crash.

- **TC-F1.5 — Ticker extracted from `financial_metrics_summary`**
  - What: `financial_metrics_summary` contains `"ticker": "NVDA"`, no directives/query ticker.
  - Why: Third-level resolution path.

---

**Test Group 2 — `_execute_data_extraction()`**

- **TC-F2.1 — All three MCP tools succeed**
  - What: Mock MCP session with successful `call_tool` for `tool_get_financial_ratios`, `tool_get_revenue_growth`, `tool_get_xbrl_financials`. Call `_execute_data_extraction()`.
  - Why: Happy path — all results stored in `state["raw_data"]`.
  - Assertions: `state["raw_data"]` contains 3 keys with non-empty values.

- **TC-F2.2 — One MCP tool fails; other two succeed**
  - What: Mock `tool_get_xbrl_financials` to raise `Exception("timeout")`. Others succeed.
  - Why: Individual tool failures must not abort the extraction phase.
  - Assertions: `state["raw_data"]` contains 2 successful results; failed key contains error indicator.

- **TC-F2.3 — All tools fail**
  - What: All three `call_tool` calls raise exceptions.
  - Why: Complete extraction failure must produce empty results for Checker to detect.
  - Assertions: `state["raw_data"]` is empty or contains only error markers; no exception propagated.

---

**Test Group 3 — `_execute_ratio_computation()`**

- **TC-F3.1 — D/E ratio uses Yahoo Finance fallback path**
  - What: Mock `state["raw_data"]` to contain `debtToEquity` from Yahoo Finance data. Call `_execute_ratio_computation()`.
  - Why: D/E ratio is documented to skip an extra MCP round-trip by using existing data.
  - Assertions: `tool_calc_debt_to_equity` MCP call is NOT made; D/E value in `state["calculated_ratios"]`.

- **TC-F3.2 — Composite score calculation includes all available ratios**
  - What: Mock partial ratio results (ROE present, current_ratio absent). Call ratio computation.
  - Why: `composite_financial_score` must exclude missing metrics from denominator.
  - Assertions: Score returned is a number 0–100; no `KeyError` on missing ratio.

---

**Test Group 4 — `_check_data_quality()`**

- **TC-F4.1 — Pre-flight guard: empty raw + calculated → skip LLM call**
  - What: `state["raw_data"] = {}`, `state["calculated_ratios"] = {}`. Call `_check_data_quality()`.
  - Why: Avoid wasting an LLM call when there is no data to evaluate.
  - Assertions: LLM mock is NOT called; state reflects "insufficient data" quality flag.

- **TC-F4.2 — Checker passes with high-quality data**
  - What: Mock LLM to return `{"data_quality": "high", "issues": [], "confidence": 0.9}`.
  - Assertions: `state["data_quality_passed"]` is `True`.

- **TC-F4.3 — Checker finds issues, fails**
  - What: Mock LLM to return `{"data_quality": "low", "issues": ["missing income statement"], "confidence": 0.3}`.
  - Assertions: `state["data_quality_passed"]` is `False`.

---

**Test Group 5 — `_brain()`**

- **TC-F5.1 — Brain returns valid JSON plan**
  - What: Mock LLM; assert returned dict has `plan` and `priority_tools` keys.

- **TC-F5.2 — Brain falls back to default tool list on API failure**
  - What: Mock `messages.create` to raise `anthropic.APIError`.
  - Why: Brain failure must not abort the agent — default extraction plan must kick in.
  - Assertions: Returned dict contains non-empty `priority_tools` list; no exception propagated.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `anthropic.Anthropic.messages.create` | Avoid real API calls |
| MCP `ClientSession.call_tool` | Avoid real subprocess stdio; inject financial data fixtures |
| `sentry_sdk.capture_exception` | Verify error capture without real Sentry |

---

### `agents/sentiment_agent.py`

**Purpose:** Two-tier Brain + Executor sentiment agent. Brain Pass 1 plans retrieval; Executor runs 4 MCP tools sequentially (retrieve → FinBERT → VADER → Fear/Greed); Brain Pass 2 synthesizes the signals into a structured sentiment summary.

---

#### Proposed Test Cases

**Test Group 1 — `__init__` and construction**

- **TC-S1.1 — `server_script_path` is required (runtime bug documentation)**
  - What: Call `SentimentAgent()` with no arguments.
  - Why: Documents the known bug in `api/main.py:107` — this call will raise `TypeError`.
  - Assertions: `pytest.raises(TypeError)` — this test is expected to FAIL until the bug is fixed.

- **TC-S1.2 — Construction succeeds with valid `server_script_path`**
  - What: Pass a valid (or mock) path string. Assert construction succeeds.
  - Assertions: `agent._llm` is set; `agent._mcp_server_params` is set.

---

**Test Group 2 — `_extract_ticker()`**

- **TC-S2.1 — Resolved from `manager_directives`**
  - Same logic as `FinancialAnalystAgent._extract_ticker()` TC-F1.1.

- **TC-S2.2 — Resolved from `financial_metrics_summary`**
  - What: No directives; `financial_metrics_summary` contains `"ticker": "TSLA"`.
  - Why: Sentiment agent uses financial data as secondary source (different from research agent).
  - Assertions: Returns `"TSLA"`.

- **TC-S2.3 — Regex scan fallback**
  - Same as TC-F1.2.

---

**Test Group 3 — `_brain_plan()`**

- **TC-S3.1 — Returns valid retrieval plan JSON**
  - What: Mock LLM; assert returned dict has `retrieval_query`, `ticker`, `days_back`, `reasoning` keys.

- **TC-S3.2 — LLM returns malformed JSON**
  - What: Mock LLM to return plain text.
  - Why: Brain Pass 1 failure must produce a safe default retrieval plan.
  - Assertions: Returns a dict with defaults (e.g., `days_back=14`); no exception.

---

**Test Group 4 — `_execute_sentiment_pipeline()`**

- **TC-S4.1 — Happy path: all 4 MCP steps succeed in sequence**
  - What: Mock all 4 `call_tool` responses with valid FinBERT/VADER/Fear-Greed fixtures. Call `_execute_sentiment_pipeline()`.
  - Why: End-to-end pipeline integration.
  - Assertions: `state["finbert_result"]`, `state["vader_result"]`, `state["fear_greed_result"]` all populated.

- **TC-S4.2 — `retrieve_social_data` returns empty → downstream steps skipped**
  - What: Mock `retrieve_social_data` to return `{"chunks": []}`. Call pipeline.
  - Why: Documented defensive behavior — FinBERT/VADER/Fear-Greed should be skipped on empty retrieval.
  - Assertions: `analyze_finbert`, `score_vader`, `calculate_fear_greed` mock calls are NOT made.

- **TC-S4.3 — FinBERT step fails → VADER and Fear/Greed still attempted**
  - What: `analyze_finbert` raises `Exception`. Retrieval succeeded.
  - Why: Partial pipeline results are better than none; downstream steps must still run.
  - Assertions: `score_vader` is called; `fear_greed` result may be computed from VADER alone.

---

**Test Group 5 — `_brain_analyze()` (Brain Pass 2)**

- **TC-S5.1 — Returns structured sentiment JSON with all 8 required keys**
  - What: Mock LLM with fixture JSON containing `overall_sentiment`, `conviction_level`, `key_signals`, `model_agreement`, `narrative`, `risk_flags`, `data_quality_note`.
  - Assertions: Returned dict has all 8 keys.

- **TC-S5.2 — LLM fails → safe default returned**
  - What: Mock LLM to raise `anthropic.APIError`.
  - Assertions: Returns dict with `overall_sentiment = "neutral"` or similar safe default; no exception.

---

**Test Group 6 — `run()` loop behavior**

- **TC-S6.1 — Loop breaks when chunks retrieved > 0**
  - What: Mock pipeline to return non-empty chunks on first iteration.
  - Assertions: Loop runs exactly once; no second iteration.

- **TC-S6.2 — Loop retries on zero chunks, respects `max_loops`**
  - What: Mock pipeline to always return empty chunks.
  - Assertions: Loop runs exactly `max_loops` times; `state["sentiment_analysis_summary"]` reflects data quality issue.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `anthropic.Anthropic.messages.create` | LLM control |
| MCP `ClientSession.call_tool` (4 sentiment tools) | Avoid heavy model loading |
| `AlphaRetriever.retrieve_raw` | Decouple from Supabase/embedder |

---

### `agents/manager_agent.py`

**Purpose:** Central LangGraph orchestrator with 7 nodes: `hydrate`, `brain_route`, `dispatch`, `evaluate`, `persist`, `finalise`, `abort`. Routes specialist agents, evaluates their outputs, persists results, and synthesizes a final report.

---

#### Proposed Test Cases

**Test Group 1 — `_brain_route()`**

- **TC-M1.1 — Routes to `run_research` on first loop**
  - What: Mock LLM to return `{"action": "run_research", "reasoning": "..."}`. Call `_brain_route()`.
  - Assertions: `state["last_action"]` is `"run_research"`.

- **TC-M1.2 — Routes to `finalise` after all agents have run**
  - What: Mock LLM to return `{"action": "finalise"}`.
  - Assertions: `state["last_action"]` is `"finalise"`.

- **TC-M1.3 — LLM returns invalid action string**
  - What: Mock LLM to return `{"action": "run_everything"}`.
  - Why: `_VALID_ACTIONS` frozenset must reject invalid actions.
  - Assertions: `state["last_action"]` falls back to a valid sequenced action; no exception.

- **TC-M1.4 — LLM raises `APIError` → fallback sequential routing**
  - What: Mock `messages.create` to raise `anthropic.APIError`.
  - Why: Router failure must not crash the orchestrator — sequential fallback must kick in.
  - Assertions: `state["last_action"]` is one of the 8 valid actions; no exception.

- **TC-M1.5 — All 8 valid actions are recognized**
  - What: For each of the 8 actions in `_VALID_ACTIONS`, mock LLM to return it; assert it is accepted.
  - Why: Regression guard for the `_VALID_ACTIONS` frozenset.

---

**Test Group 2 — `_brain_evaluate()`**

- **TC-M2.1 — Evaluation passes with high score**
  - What: Mock LLM to return `{"passed": true, "score": 85, "reason": "Complete data"}`.
  - Assertions: `state["last_evaluation"].passed` is `True`; `score == 85`.

- **TC-M2.2 — Evaluation fails with low score**
  - What: Mock LLM to return `{"passed": false, "score": 30, "reason": "Missing ratios"}`.
  - Assertions: `state["last_evaluation"].passed` is `False`.

- **TC-M2.3 — LLM API failure → pass with score=50 (permissive default)**
  - What: Mock LLM to raise `APIError`.
  - Why: Evaluator failure must not block the pipeline — default pass prevents infinite rerun loops.
  - Assertions: `state["last_evaluation"].passed` is `True`; `score == 50`.

---

**Test Group 3 — `_dispatch()`**

- **TC-M3.1 — ResearchAgent dispatched successfully; timing recorded**
  - What: Mock `ResearchAgent.run()` to return a populated `SharedManagerState`. Call `_dispatch()` with `last_action = "run_research"`.
  - Why: Dispatch must route to the correct agent and record execution metadata.
  - Assertions: `state["agent_execution_history"][-1]["agent"]` is `"research"`; `duration_s > 0`.

- **TC-M3.2 — Agent raises exception; execution record captures failure**
  - What: Mock agent `run()` to raise `Exception("connection refused")`.
  - Why: Agent failure must be recorded but must not crash the orchestrator.
  - Assertions: `state["agent_execution_history"][-1]["outcome"]` is `"error"`; orchestrator does not re-raise.

- **TC-M3.3 — Sentry breadcrumb is recorded on dispatch**
  - What: Mock `sentry_sdk.add_breadcrumb`. Assert it is called when `_dispatch()` runs.
  - Why: Observability guarantee — every dispatch must be traced.

---

**Test Group 4 — `_node_persist()`**

- **TC-M4.1 — Evaluation passed → `last_action` preserved**
  - What: `state["last_evaluation"].passed = True`. Call `_node_persist()`.
  - Assertions: `state["last_action"]` unchanged.

- **TC-M4.2 — Evaluation failed → `last_action` overridden to `rerun_*`**
  - What: `state["last_evaluation"].passed = False`, `state["last_action"] = "run_financial"`. Call `_node_persist()`.
  - Assertions: `state["last_action"]` is `"rerun_financial"`.

---

**Test Group 5 — `_should_route()` conditional edge**

- **TC-M5.1 — Max loops reached → `abort`**
  - What: `state["loop_counter"] = state["max_loops"]`.
  - Assertions: Returns `"abort"`.

- **TC-M5.2 — `last_action` is `"finalise"` → `finalise`**
  - What: `state["last_action"] = "finalise"`, loops remaining.
  - Assertions: Returns `"finalise"`.

- **TC-M5.3 — Normal dispatch action → `dispatch`**
  - What: `state["last_action"] = "run_research"`, loops remaining.
  - Assertions: Returns `"dispatch"`.

---

**Test Group 6 — `_should_continue_after_persist()` conditional edge**

- **TC-M6.1 — Evaluation passed → `brain_route`**
  - What: `state["last_evaluation"].passed = True`.
  - Assertions: Returns `"brain_route"`.

- **TC-M6.2 — Evaluation failed → `dispatch` (rerun)**
  - What: `state["last_evaluation"].passed = False`, loops remaining.
  - Assertions: Returns `"dispatch"`.

- **TC-M6.3 — Evaluation failed but max loops reached → `abort`**
  - What: `state["last_evaluation"].passed = False`, `loop_counter >= max_loops`.
  - Assertions: Returns `"abort"`.

---

**Test Group 7 — `_brain_finalise()`**

- **TC-M7.1 — Returns 400–600 word final report**
  - What: Mock LLM to return a ~500-word string. Call `_brain_finalise()`.
  - Assertions: `state["final_report"]` is a non-empty string.

- **TC-M7.2 — LLM failure → graceful fallback report**
  - What: Mock LLM to raise `APIError`.
  - Assertions: `state["final_report"]` is not empty; contains error acknowledgment.

---

**Test Group 8 — `_infer_result_keys()`**

- **TC-M8.1 — Correct key mapping for all agent types**
  - What: Call `_infer_result_keys("research")`, `_infer_result_keys("financial")`, `_infer_result_keys("sentiment")`.
  - Why: Regression guard for the agent → state field mapping.
  - Assertions: Each returns the correct `SharedManagerState` field name string.

- **TC-M8.2 — Unknown agent key raises or returns safe default**
  - What: Call `_infer_result_keys("unknown_agent")`.
  - Assertions: Either raises `KeyError`/`ValueError` (documented behavior) or returns a safe `None`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `anthropic.AsyncAnthropic.messages.create` | Async LLM control |
| `ResearchAgent.run`, `FinancialAnalystAgent.run`, `SentimentAgent.run` | Isolate orchestrator from sub-agents |
| `ManagerMemory.recall`, `ManagerMemory.remember` | Control memory context |
| `sentry_sdk.add_breadcrumb`, `sentry_sdk.capture_exception` | Verify observability without real Sentry |
| LangSmith `@traceable` | Replace with identity decorator |

---

## memory/

### `memory/manager_memory.py`

**Purpose:** Two-level memory system — `ShortTermMemory` (in-process session FIFO, 50-message cap) and `LongTermMemory` (Supabase-backed, FIFO-evicted, 3 JSONB columns). `ManagerMemory` is a facade combining both.

---

#### Proposed Test Cases

**Test Group 1 — `ShortTermMemory`**

- **TC-STM1.1 — `add_message()` appends to buffer**
  - What: Add 3 messages. Assert `get_messages()` returns all 3 in order.

- **TC-STM1.2 — FIFO eviction at max_messages=50**
  - What: Add 51 messages. Assert buffer length is 50; oldest message is evicted.
  - Why: Unbounded growth would cause memory leaks in long sessions.

- **TC-STM1.3 — `reset()` clears all buffers**
  - What: Add messages + dispatches + evaluations. Call `reset()`. Assert all buffers empty.

- **TC-STM1.4 — `log_dispatch()` creates correct `AgentExecutionRecord`**
  - What: Call `log_dispatch(agent="research", duration=1.23, outcome="success", summary="done")`.
  - Assertions: `memory.dispatch_log[-1].agent == "research"` and `.duration_s == 1.23`.

- **TC-STM1.5 — `to_context_dict()` returns correct structure**
  - What: Populate memory with messages and 2 dispatches. Call `to_context_dict()`.
  - Assertions: Dict has `messages` and `dispatch_log` keys with correct content.

---

**Test Group 2 — `LongTermMemory`**

- **TC-LTM2.1 — `__init__` does NOT call `load()` (I/O deferral)**
  - What: Mock `supabase_client.table(...).select(...).execute`. Instantiate `LongTermMemory(client, "user-1")` directly (not via `create()`).
  - Why: Core testability guarantee from C-3 fix — no I/O in constructor.
  - Assertions: Mock is NOT called during `__init__`; called only when `load()` is explicitly invoked.

- **TC-LTM2.2 — `create()` calls `load()` automatically**
  - What: Mock Supabase client. Call `LongTermMemory.create(client, "user-1")`.
  - Assertions: Supabase `select().execute()` IS called during `create()`.

- **TC-LTM2.3 — `load()` populates memory from Supabase response**
  - What: Mock `execute()` to return `{"data": [{"operational_heuristics": [...], "ticker_insights": [...], "user_preferences": [...]}]}`.
  - Assertions: `memory.operational_heuristics` has correct length; values match fixture.

- **TC-LTM2.4 — `load()` handles empty Supabase response (new user)**
  - What: Mock `execute()` to return `{"data": []}`.
  - Assertions: All three lists initialized as empty; no exception.

- **TC-LTM2.5 — FIFO eviction at `operational_heuristics` cap (100)**
  - What: Pre-populate with 100 entries. Call `remember_heuristic("new")`.
  - Assertions: Length stays at 100; oldest entry evicted; new entry at end.

- **TC-LTM2.6 — `persist()` calls Supabase `upsert` with correct payload**
  - What: Mock `upsert().execute()`. Call `persist()`.
  - Assertions: Mock was called once; payload contains `user_id` + all 3 JSONB columns.

- **TC-LTM2.7 — `persist()` handles Supabase error gracefully**
  - What: Mock `upsert().execute()` to raise `Exception("DB error")`.
  - Assertions: No exception propagated; error is logged.

---

**Test Group 3 — `ManagerMemory` facade**

- **TC-MM3.1 — `recall()` merges short-term and long-term into one dict**
  - What: Populate both memories with data. Call `recall()`.
  - Assertions: Returned dict has `short_term` and `long_term` keys.

- **TC-MM3.2 — `remember()` routes correctly by category**
  - What: Call `remember(category="heuristic", content="...")`. Assert `short_term.add_message()` and `long_term.remember_heuristic()` are called.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `supabase.Client.table().select().execute()` | Avoid real Supabase calls |
| `supabase.Client.table().upsert().execute()` | Avoid real writes |

---

## core/

### `core/__init__.py`

**Not Testable**  
**Reason:** Empty package marker file with no logic.

---

### `core/observability.py`

**Purpose:** Initializes Sentry error tracking (`init_sentry`) and LangSmith tracing (`init_langsmith`) with idempotency guards and optional env var configuration. Exposes query flags `sentry_enabled()` and `langsmith_enabled()`.

---

#### Proposed Test Cases

- **TC-O1 — `init_sentry()` is a no-op when `SENTRY_DSN` is unset**
  - What: Unset `SENTRY_DSN` env var. Call `init_sentry()`.
  - Assertions: `sentry_sdk.init` is NOT called; `sentry_enabled()` returns `False`.

- **TC-O2 — `init_sentry()` calls `sentry_sdk.init` with correct params when DSN set**
  - What: Set `SENTRY_DSN = "https://fake@sentry.io/1"`. Mock `sentry_sdk.init`. Call `init_sentry()`.
  - Assertions: `sentry_sdk.init` called once with `dsn`, `traces_sample_rate=1.0`, `send_default_pii=False`.

- **TC-O3 — `init_sentry()` is idempotent (second call is a no-op)**
  - What: Call `init_sentry()` twice. Assert `sentry_sdk.init` called only once.
  - Why: Idempotency prevents double-initialization in module reload scenarios.

- **TC-O4 — `init_langsmith()` sets required env vars**
  - What: Set `LANGSMITH_API_KEY = "key123"`. Call `init_langsmith()`.
  - Assertions: `os.environ["LANGCHAIN_TRACING_V2"] == "true"`.

- **TC-O5 — `init_langsmith()` is a no-op when API key is unset**
  - What: Unset `LANGSMITH_API_KEY`. Call `init_langsmith()`.
  - Assertions: `LANGCHAIN_TRACING_V2` env var is NOT set; `langsmith_enabled()` returns `False`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `sentry_sdk.init` | Avoid real Sentry SDK initialization |
| `os.environ` | Control env var state per test |

---

### `core/error_handler.py`

**Purpose:** `with_error_reporting(component)` decorator factory. Adds Sentry breadcrumb on entry, captures exception on error, and always re-raises. Works on both sync and async functions. Provides `.context` and `.async_context` context manager variants.

---

#### Proposed Test Cases

- **TC-EH1 — Sync function: success path — breadcrumb added, no capture**
  - What: Decorate a sync function with `@with_error_reporting("test")`. Call it successfully.
  - Assertions: `sentry_sdk.add_breadcrumb` called on entry; `sentry_sdk.capture_exception` NOT called.

- **TC-EH2 — Sync function: exception path — exception captured AND re-raised**
  - What: Decorate a function that raises `ValueError`. Call it.
  - Assertions: `sentry_sdk.capture_exception` called with the `ValueError`; `ValueError` propagates to caller.

- **TC-EH3 — Async function: success path**
  - What: Decorate an `async def` function. `await` it.
  - Assertions: Breadcrumb added; no capture; return value preserved.

- **TC-EH4 — Async function: exception path**
  - What: Decorate an `async def` that raises `RuntimeError`. `await` it.
  - Assertions: `capture_exception` called; `RuntimeError` propagates.

- **TC-EH5 — `_safe_extra()` filters non-JSON-serializable kwargs**
  - What: Pass kwargs containing `object()` (non-serializable). Call `_safe_extra(kwargs)`.
  - Assertions: Non-serializable value is dropped or stringified; no `TypeError`.

- **TC-EH6 — Context manager variant: `with with_error_reporting.context("x"):`**
  - What: Use sync context manager around a block that raises.
  - Assertions: Exception captured; re-raised from `with` block.

- **TC-EH7 — Async context manager variant**
  - What: Use `async with with_error_reporting.async_context("x"):` around a block that raises.
  - Assertions: Exception captured; re-raised.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `sentry_sdk.add_breadcrumb` | Verify observability without Sentry |
| `sentry_sdk.capture_exception` | Verify error capture |

---

## api/

### `api/main.py`

**Purpose:** FastAPI application: manages lifespan (validate config → init observability → create agents → compile graph), middleware (request timing), exception handlers (`AlphaAgentError`, catch-all 500), health endpoints, and mounts the analyze router.

---

#### Proposed Test Cases

**Test Group 1 — Lifespan**

- **TC-API1.1 — Startup succeeds when all env vars present**
  - What: Mock `validate_settings()`, `init_sentry()`, `ManagerAgent`, `SentimentAgent`, etc. Use `AsyncClient` with `app` (ASGI test client) to trigger lifespan.
  - Why: Lifespan failure leaves the server in an undefined state.
  - Assertions: `app.state.manager_agent` is set; `app.state.supabase` is set.

- **TC-API1.2 — Startup fails on `ConfigurationError` → app does not start**
  - What: Mock `validate_settings()` to raise `ConfigurationError`.
  - Assertions: Application startup raises; server is not left in a half-initialized state.

- **TC-API1.3 — `SentimentAgent()` instantiation bug is caught (known bug)**
  - What: Run startup without mocking `SentimentAgent`. Assert `TypeError` is raised.
  - Why: Documents the existing bug at `api/main.py:107` where `SentimentAgent()` is called without required `server_script_path`.
  - Assertions: `pytest.raises(TypeError)` — expected to fail until bug is fixed.

---

**Test Group 2 — Middleware**

- **TC-API2.1 — `request_timing_middleware` logs method, path, status, duration**
  - What: Use `TestClient`. Make a `GET /health` request. Capture log output.
  - Assertions: Log contains `GET`, `/health`, `200`, duration in milliseconds.

---

**Test Group 3 — Exception handlers**

- **TC-API3.1 — `AlphaAgentError` → structured JSON 500 response**
  - What: Add a test route that raises `AgentError("something failed")`. Use `TestClient` to call it.
  - Assertions: Response status is 500; body is JSON with `error`, `message`, `trace_id` keys.

- **TC-API3.2 — `ValidationError` (subclass) → 400 response**
  - What: Route raises `ValidationError("bad input")`.
  - Assertions: Response status is 400.

- **TC-API3.3 — Unhandled `Exception` → 500 with hidden detail in prod**
  - What: Set `APP_ENV=prod`; route raises `Exception("internal")`.
  - Assertions: Response body does NOT contain the raw exception message.

- **TC-API3.4 — Unhandled `Exception` → 500 with visible detail in dev**
  - What: Set `APP_ENV=dev`; route raises `Exception("internal")`.
  - Assertions: Response body contains `"internal"` in detail.

---

**Test Group 4 — Health endpoints**

- **TC-API4.1 — `GET /health` → 200 with status, env, version**
  - What: `TestClient.get("/health")`.
  - Assertions: Status 200; body has `status: "ok"`.

- **TC-API4.2 — `GET /readiness` → 200 when Supabase ping succeeds**
  - What: Mock `app.state.supabase` ping. Assert 200.

- **TC-API4.3 — `GET /readiness` → 503 when Supabase ping fails**
  - What: Mock ping to raise. Assert 503.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `validate_settings` | Control startup path |
| `init_sentry`, `init_langsmith` | Avoid side effects |
| `ManagerAgent`, `SentimentAgent`, `ResearchAgent`, `FinancialAnalystAgent` | Avoid real agent construction |
| `create_client` (Supabase) | Avoid real DB connection |

---

### `api/config.py`

**Purpose:** pydantic-settings `Settings` class reading all env vars. `get_settings()` is a cached singleton. `validate_settings()` raises `ConfigurationError` if required fields are absent.

---

#### Proposed Test Cases

- **TC-CFG1 — `validate_settings()` passes with all required keys present**
  - What: Mock `Settings` with all required fields non-empty. Call `validate_settings()`.
  - Assertions: No exception raised.

- **TC-CFG2 — `validate_settings()` raises `ConfigurationError` when `ANTHROPIC_API_KEY` missing**
  - What: Mock `Settings` with `ANTHROPIC_API_KEY = ""`.
  - Assertions: `pytest.raises(ConfigurationError)` with message mentioning the key.

- **TC-CFG3 — `validate_settings()` raises `ConfigurationError` when `SUPABASE_URL` missing**
  - Same as TC-CFG2 but for `SUPABASE_URL`.

- **TC-CFG4 — `get_settings()` returns same instance on repeated calls (lru_cache)**
  - What: Call `get_settings()` twice. Assert `is` identity.
  - Why: Ensures the cache works and no duplicate Settings objects are created.

- **TC-CFG5 — `APP_ENV` defaults to `"dev"` when not set**
  - What: Unset `APP_ENV`. Instantiate `Settings`.
  - Assertions: `settings.APP_ENV == "dev"`.

- **TC-CFG6 — `APP_ENV = "invalid"` raises `ValidationError`**
  - What: Set `APP_ENV = "staging"` (not in `Literal["dev", "prod"]`).
  - Assertions: pydantic `ValidationError` raised on construction.

- **TC-CFG7 — `ALLOWED_ORIGINS` parses correctly from comma-separated string**
  - What: Set `ALLOWED_ORIGINS = "https://a.com,https://b.com"`. Instantiate `Settings`.
  - Assertions: `settings.ALLOWED_ORIGINS` is a list with 2 entries.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `os.environ` | Inject test values without real `.env` file |
| `get_settings.cache_clear()` | Reset `lru_cache` between tests |

---

### `api/dependencies.py`

**Purpose:** FastAPI dependency providers: `get_user_id()` reads `X-User-Id` header with fallback; `get_manager_memory()` creates per-request `ManagerMemory` from `app.state`.

---

#### Proposed Test Cases

- **TC-DEP1 — `get_user_id()` returns header value when present**
  - What: Mock `Request` with header `X-User-Id: user-42`.
  - Assertions: Returns `"user-42"`.

- **TC-DEP2 — `get_user_id()` falls back to `DEFAULT_USER_ID` when header absent**
  - What: Mock `Request` with no `X-User-Id` header. Mock `Settings.DEFAULT_USER_ID = "anon"`.
  - Assertions: Returns `"anon"`.

- **TC-DEP3 — `get_manager_memory()` constructs `ManagerMemory` with correct `user_id`**
  - What: Mock `Request.app.state.supabase`. Call `get_manager_memory(request, "user-7")`.
  - Assertions: `ManagerMemory.user_id == "user-7"`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `fastapi.Request` | Inject headers without real HTTP request |
| `app.state.supabase` | Avoid real Supabase client |

---

### `api/routes/analyze.py`

**Purpose:** `POST /api/v1/analyze` endpoint. Validates request, builds `manager_directives`, runs `ManagerAgent.run()`, fire-and-forgets persistence to Supabase, returns `AnalyzeResponse`.

---

#### Proposed Test Cases

- **TC-AZ1 — Valid request → 200 with full `AnalyzeResponse`**
  - What: Mock `manager_agent.run()` to return a complete `SharedManagerState`. Post valid JSON `{"query": "Analyze AAPL", "ticker": "AAPL", "search_depth": "basic"}`.
  - Assertions: Status 200; body has `final_report`, `status == "completed"`.

- **TC-AZ2 — `query` too short (< 10 chars) → 422 Unprocessable Entity**
  - What: Post `{"query": "Hi"}`.
  - Assertions: Status 422; body references `query` field.

- **TC-AZ3 — `ticker` contains lowercase → uppercased automatically**
  - What: Post `{"ticker": "aapl"}`.
  - Assertions: `manager_directives["ticker"]` is `"AAPL"` in the call to `run()`.

- **TC-AZ4 — `ticker` longer than 5 chars → 422**
  - What: Post `{"ticker": "TOOLONG"}`.
  - Assertions: Status 422.

- **TC-AZ5 — `days_back` out of range (0 or 366) → 422**
  - What: Post `{"days_back": 0}` and `{"days_back": 366}`.
  - Assertions: Status 422 for both.

- **TC-AZ6 — `include_sentiment = false` → sentiment not in `manager_directives`**
  - What: Post with `include_sentiment: false`.
  - Assertions: Sentiment agent is not called (or `include_sentiment = false` in directives).

- **TC-AZ7 — `manager_agent.run()` raises `AgentTimeoutError` → 504**
  - What: Mock `run()` to raise `AgentTimeoutError`.
  - Assertions: Response status 504.

- **TC-AZ8 — `_persist_analysis()` is called as fire-and-forget (non-blocking)**
  - What: Mock `asyncio.create_task`. Assert it is called after `run()` completes.
  - Why: Persistence must not block the HTTP response.
  - Assertions: Response is returned before Supabase write completes.

- **TC-AZ9 — `_persist_analysis()` Supabase failure does not affect response**
  - What: Mock Supabase `upsert` to raise. Assert HTTP response is still 200.
  - Why: Fire-and-forget side effects must not affect client response.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `manager_agent.run` | Avoid full graph execution |
| `supabase.table().upsert().execute()` | Avoid real Supabase writes |
| `asyncio.create_task` | Verify fire-and-forget without side effects |

---

### `api/core/exceptions.py`

**Purpose:** Exception hierarchy with structured error formatting and auto-generated 8-char `trace_id`.

---

#### Proposed Test Cases

- **TC-EXC1 — `AlphaAgentError.trace_id` is auto-generated and unique**
  - What: Create two `AlphaAgentError("msg")` instances.
  - Assertions: Both have `trace_id`; they are different from each other; length is 8.

- **TC-EXC2 — `to_dict()` returns all required keys**
  - What: Create `AgentError("fail")`. Call `.to_dict()`.
  - Assertions: Dict has `error`, `message`, `detail`, `trace_id` keys.

- **TC-EXC3 — Subclass `http_status_code` is correct**
  - What: Assert `ValidationError.http_status_code == 400`, `AgentError == 500`, `AgentTimeoutError == 504`, `ExternalServiceError == 503`.

- **TC-EXC4 — `ConfigurationError` is catchable as `AlphaAgentError`**
  - What: `raise ConfigurationError("missing key")`. Catch as `AlphaAgentError`.
  - Assertions: `except AlphaAgentError` block is entered.

---

## rag/

### `rag/__init__.py`

**Not Testable**  
**Reason:** Pure re-export file with no logic.

---

### `rag/loader.py`

**Purpose:** Fetches `RawDocument` objects from yfinance news and Reddit RSS feeds. Handles both old and new yfinance schemas, applies circuit breakers per ticker/feed, and normalizes timestamps to UTC ISO-8601.

---

#### Proposed Test Cases

**Test Group 1 — `_to_utc_iso8601()`**

- **TC-L1.1 — `datetime` object → ISO-8601 string**
  - What: Pass `datetime(2024, 1, 15, 12, 0, 0)`. Assert returns `"2024-01-15T12:00:00+00:00"`.

- **TC-L1.2 — Unix int timestamp → correct ISO-8601**
  - What: Pass `1705312200` (int).
  - Assertions: Returns correct UTC datetime string.

- **TC-L1.3 — Unix float timestamp → correct ISO-8601**
  - What: Pass `1705312200.5` (float).

- **TC-L1.4 — RFC-2822 string → ISO-8601**
  - What: Pass `"Mon, 15 Jan 2024 12:00:00 +0000"`.
  - Assertions: Correctly parsed.

- **TC-L1.5 — Date-only string `"2024-01-15"` → ISO-8601**
  - What: Pass `"2024-01-15"`.
  - Assertions: Returns datetime at midnight UTC.

- **TC-L1.6 — Invalid string → raises `ValueError`**
  - What: Pass `"not-a-date"`.
  - Assertions: `pytest.raises(ValueError)`.

- **TC-L1.7 — `None` input → raises `ValueError`**

---

**Test Group 2 — `AlphaLoader.load()`**

- **TC-L2.1 — Happy path: returns `RawDocument` list for valid tickers**
  - What: Mock `yf.Ticker("AAPL").news` to return a fixture news list. Mock RSS feed response. Call `load(["AAPL"])`.
  - Assertions: Returns list of `RawDocument`; `source_type` is `"news"` or `"reddit"`.

- **TC-L2.2 — yfinance raises exception → circuit breaker skips ticker**
  - What: Mock `yf.Ticker("FAIL").news` to raise. Assert `"FAIL"` ticker is skipped.
  - Assertions: Returns documents for other tickers; no exception propagated.

- **TC-L2.3 — New yfinance news schema (has `content` key)**
  - What: Fixture uses new schema `{"content": {"title": "...", "link": "..."}}`.
  - Assertions: `RawDocument` fields populated correctly.

- **TC-L2.4 — Old yfinance news schema (has `title`, `link` at top level)**
  - What: Fixture uses old schema `{"title": "...", "link": "..."}`.
  - Assertions: `RawDocument` fields populated correctly.

- **TC-L2.5 — RSS feed fetch fails → feed skipped, other feeds continue**
  - What: Mock `requests.get` (or `feedparser.parse`) to raise for one feed.
  - Assertions: Other feeds still processed; no crash.

- **TC-L2.6 — `max_news_per_ticker` cap is respected**
  - What: Mock yfinance to return 30 news items; `max_news_per_ticker=20`.
  - Assertions: `len([d for d in docs if d.source_type == "news"])` is ≤ 20.

- **TC-L2.7 — `published_at` field is UTC ISO-8601 string for all documents**
  - What: Return mixed timestamp formats from yfinance/RSS.
  - Assertions: All `doc.published_at` values are valid ISO-8601 UTC strings.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `yfinance.Ticker` | Avoid real Yahoo Finance HTTP calls |
| `feedparser.parse` or `requests.get` | Avoid real RSS fetches |

---

### `rag/processor.py`

**Purpose:** Deduplicates raw documents using dual SHA256 hashing (content + URL) with FIFO-evicting in-memory seen cache (100,000 entries). Chunks accepted documents using `RecursiveCharacterTextSplitter`. Tracks `ProcessorMetrics`.

---

#### Proposed Test Cases

- **TC-P1 — New document → chunked and included**
  - What: Process one `RawDocument` with unique content and URL.
  - Assertions: Returns at least one `ProcessedChunk`; chunk metadata has `ticker`, `source_type`, `chunk_index`, `content_hash`.

- **TC-P2 — Exact duplicate (same URL + same content) → skipped**
  - What: Process same document twice.
  - Assertions: Second call returns 0 new chunks; `metrics.duplicates_skipped == 1`.

- **TC-P3 — URL match but content changed → updated (not skipped)**
  - What: Process doc with URL "http://x.com/a" + content A. Then same URL + content B.
  - Assertions: Second processing returns chunks with updated content.

- **TC-P4 — FIFO eviction at 100,000 entries**
  - What: Pre-populate `_seen` dict with 100,000 entries. Add one more.
  - Assertions: Oldest entry evicted; dict size remains ≤ 100,000.

- **TC-P5 — `ProcessorMetrics` counts are correct**
  - What: Process 3 docs: 1 new, 1 duplicate, 1 updated.
  - Assertions: `metrics.new == 1`, `metrics.duplicates_skipped == 1`, `metrics.updated == 1`.

- **TC-P6 — Short document (< chunk_size) → single chunk, no split**
  - What: Document with 50 chars; chunk_size=500.
  - Assertions: Returns exactly 1 chunk.

- **TC-P7 — Long document → multiple chunks with correct metadata**
  - What: Document with 3000 chars; chunk_size=500 with overlap.
  - Assertions: Multiple chunks returned; `chunk_index` increments correctly; no content loss.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `langchain_text_splitters.RecursiveCharacterTextSplitter` | Control chunk boundaries in tests |

---

### `rag/embedding_manager.py`

**Purpose:** Singleton sentence-transformer embedding manager. Thread-safe double-checked locking. Supports `all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`. OOM fallback to CPU. L2 normalization. `reset_embedder()` for test teardown.

---

#### Proposed Test Cases

- **TC-EM1 — `get_embedder()` returns singleton**
  - What: Call `get_embedder()` twice. Assert `is` identity.
  - Mocking: Mock `SentenceTransformer` to avoid model download.

- **TC-EM2 — `reset_embedder()` then `get_embedder()` creates new instance**
  - What: Call `reset_embedder()` then `get_embedder()` again.
  - Assertions: `SentenceTransformer` constructor called again.

- **TC-EM3 — `embed_chunks()` returns L2-normalized vectors (unit length)**
  - What: Mock `model.encode()` to return `[[3.0, 4.0]]`. Call `embed_chunks([...])`.
  - Assertions: Output vector has norm ≈ 1.0 (`math.sqrt(0.36 + 0.64) == 1.0`).

- **TC-EM4 — `embed_chunks()` returns correct number of outputs**
  - What: Pass 5 chunks. Assert 5 embedding dicts returned.

- **TC-EM5 — OOM fallback switches to CPU**
  - What: Mock `model.encode()` to raise `torch.cuda.OutOfMemoryError` on first call, succeed on second (simulating CPU fallback).
  - Assertions: Second `encode()` call made; result returned without exception.

- **TC-EM6 — `embed_query()` returns a single list (not nested)**
  - What: Call `embed_query("AAPL earnings")`.
  - Assertions: Result is a flat list of floats.

- **TC-EM7 — Unsupported model name raises `ValueError`**
  - What: Call `get_embedder(model_name="fake/model")`.
  - Assertions: `pytest.raises(ValueError)`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `sentence_transformers.SentenceTransformer` | Avoid 440MB model download in CI |
| `torch.cuda.is_available` | Control device selection |

---

### `rag/vector_store.py`

**Purpose:** Supabase pgvector wrapper. `upsert()` inserts/updates on `url_hash` conflict. `hybrid_search()` calls `alpha_hybrid_search` Postgres RPC (RRF cosine + FTS) and filters by score threshold.

---

#### Proposed Test Cases

- **TC-VS1 — `upsert()` calls Supabase with correct table and conflict key**
  - What: Mock `supabase.table("alpha_documents").upsert(...).execute()`. Call `upsert([record1])`.
  - Assertions: `upsert` called with `on_conflict="url_hash"`.

- **TC-VS2 — `upsert()` batches multiple records in one call**
  - What: Pass 100 records. Assert single `upsert` call (not 100 individual calls).

- **TC-VS3 — `hybrid_search()` returns results above threshold**
  - What: Mock RPC to return `[{"score": 0.9, "text": "..."}, {"score": 0.1, "text": "..."}]` with threshold=0.5.
  - Assertions: Only the record with score 0.9 is returned.

- **TC-VS4 — `hybrid_search()` returns empty list when no results above threshold**
  - What: All RPC results below threshold.
  - Assertions: Returns `[]`.

- **TC-VS5 — `hybrid_search()` handles Supabase RPC error gracefully**
  - What: Mock `rpc().execute()` to raise.
  - Assertions: Returns `[]` or raises `ExternalServiceError`; does not propagate raw exception.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `supabase.Client.table()` | Avoid real Supabase calls |
| `supabase.Client.rpc()` | Avoid real RPC calls |

---

### `rag/graph_store.py`

**Purpose:** Neo4j knowledge graph store. `extract_batch()` uses Claude to extract entities and relations from documents. `upsert_batch()` MERGEs into Neo4j idempotently. `_ensure_constraints()` creates uniqueness constraints.

---

#### Proposed Test Cases

- **TC-GS1 — `__init__` does NOT connect to Neo4j**
  - What: Mock `neo4j.GraphDatabase.driver`. Instantiate `AlphaGraphStore()`.
  - Assertions: `driver` mock is NOT called during `__init__`; called only during `connect()`.

- **TC-GS2 — `connect()` initializes `_driver` and `_claude`**
  - What: Mock driver. Call `connect()`.
  - Assertions: `store._driver` is not `None`; `store._claude` is not `None`.

- **TC-GS3 — `extract_batch()` skips documents shorter than 80 chars**
  - What: Pass 2 docs: one with 50 chars, one with 200 chars. Mock Claude extraction.
  - Assertions: Claude called only once (for the long doc); short doc skipped.

- **TC-GS4 — `extract_batch()` parses Claude's entity/relation JSON correctly**
  - What: Mock Claude to return `{"entities": [{"name": "Apple", "type": "Company"}], "relations": []}`.
  - Assertions: Returns list with one `Entity` dataclass.

- **TC-GS5 — `extract_batch()` handles Claude returning malformed JSON gracefully**
  - What: Mock Claude to return plain text.
  - Assertions: Returns empty list; no exception.

- **TC-GS6 — `upsert_batch()` calls Neo4j MERGE for each entity**
  - What: Mock Neo4j session. Pass 3 entities. Call `upsert_batch()`.
  - Assertions: MERGE Cypher called 3 times with correct entity names.

- **TC-GS7 — Relationship weight averaged on update (not overwritten)**
  - What: First `upsert_batch` with relation weight 1.0; second with weight 2.0.
  - Assertions: Neo4j MERGE includes averaging logic; final weight is 1.5.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `anthropic.AsyncAnthropic.messages.create` | Control Claude extraction output |
| `neo4j.GraphDatabase.driver` | Avoid real Neo4j connection |
| Neo4j `Session.run()` | Verify Cypher queries without DB |

---

### `rag/retriever.py`

**Purpose:** 5-stage retrieval pipeline (hybrid search → freshness rerank → source diversity → token budget → citation formatting). Stage 3 caps 2 chunks/URL and 3 chunks/source_type; first chunk always included in Stage 4.

---

#### Proposed Test Cases

- **TC-RT1 — Freshness rerank: fresher chunk ranks higher than older with same base score**
  - What: Two chunks, same hybrid score. Chunk A is 1 hour old; Chunk B is 100 hours old.
  - Assertions: Chunk A has higher reranked score than Chunk B.

- **TC-RT2 — Freshness formula: score decays with 72h half-life**
  - What: Chunk with base score 1.0, 72h old.
  - Assertions: Reranked score ≈ `1.0 * exp(-1.0) ≈ 0.368`.

- **TC-RT3 — Source diversity: max 2 chunks per URL**
  - What: 5 chunks from the same URL.
  - Assertions: Stage 3 output contains ≤ 2 chunks from that URL.

- **TC-RT4 — Source diversity: max 3 chunks per `source_type`**
  - What: 5 chunks all with `source_type = "reddit"`.
  - Assertions: Stage 3 output contains ≤ 3 `reddit` chunks.

- **TC-RT5 — Token budget: first chunk always included even if large**
  - What: `token_budget = 100`; first chunk is 500 chars (> budget * 4). Second chunk is 50 chars.
  - Assertions: First chunk included despite exceeding budget; second may be excluded.

- **TC-RT6 — Token budget: greedy inclusion until budget exceeded**
  - What: `token_budget = 100` (400 chars budget). 3 chunks of 150 chars each.
  - Assertions: First 2 chunks included; third excluded.

- **TC-RT7 — Citation formatting includes all required fields**
  - What: One chunk with `source_type`, `ticker`, `published_at`, `score`.
  - Assertions: Output string contains `[1]`, `SOURCE:`, `TICKER:`, `DATE:`, `SCORE:`.

- **TC-RT8 — `retrieve()` with no results returns empty string**
  - What: Mock `AlphaVectorStore.hybrid_search()` to return `[]`.
  - Assertions: Returns `""`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `AlphaVectorStore.hybrid_search` | Control candidate set |
| `AlphaEmbedder.embed_query` | Avoid model load |
| `datetime.datetime.now(UTC)` | Control freshness calculation |

---

### `rag/hybrid_rag.py`

**Purpose:** Module-level lazy-initialized facade for three retrieval coroutines: `rag_vector_search` (Supabase pgvector via `asyncio.to_thread`), `rag_graph_traverse` (Neo4j Cypher with `max_hops ≤ 3`), `rag_hybrid_query` (parallel vector+graph with RRF fusion).

---

#### Proposed Test Cases

- **TC-HRR1 — `rag_vector_search()` returns results dict on success**
  - What: Mock Supabase RPC. Call `await rag_vector_search("AAPL earnings", ticker="AAPL")`.
  - Assertions: Returns `{"query": ..., "results": [...]}`.

- **TC-HRR2 — `rag_vector_search()` returns `{"error": ...}` on Supabase failure**
  - What: Mock Supabase to raise.
  - Assertions: Returns dict with `"error"` key; no exception propagated.

- **TC-HRR3 — `rag_graph_traverse()` rejects `max_hops > 3`**
  - What: Call `await rag_graph_traverse("AAPL", max_hops=4)`.
  - Assertions: Either `max_hops` is clamped to 3, or `ValueError` raised.

- **TC-HRR4 — `rag_hybrid_query()` combines vector + graph with RRF fusion**
  - What: Mock both `rag_vector_search` and `rag_graph_traverse` with different result sets. Call `rag_hybrid_query()`.
  - Assertions: Fused result contains items from both; ordering reflects 0.7/0.3 weighting.

- **TC-HRR5 — `_rrf()` assigns correct reciprocal rank scores**
  - What: Create two ranked lists with overlap. Call `_rrf(list_a, list_b, k=60)`.
  - Assertions: Item present in both lists scores higher than items in only one.

- **TC-HRR6 — `_embed()` falls back to SHA256-based pseudo-embedding when model unavailable**
  - What: Mock `get_embedder()` to raise. Call `_embed("test query")`.
  - Assertions: Returns a list of floats; no exception.

- **TC-HRR7 — `rag_hybrid_query()` handles graph traversal failure gracefully**
  - What: Mock `rag_graph_traverse` to raise. Mock vector search to succeed.
  - Assertions: Returns vector results only; no exception.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `supabase.Client.rpc()` | Avoid real Supabase |
| `neo4j` session | Avoid real Neo4j |
| `get_embedder` | Avoid model loading |

---

### `rag/ingestion.py`

**Purpose:** Full ETL pipeline: Load → Process → Embed → Vector upsert → Graph upsert. Validates env vars before running. Individual stage `try/except` with Sentry capture. Graph stage skipped if vector stage failed or `skip_graph=True`.

---

#### Proposed Test Cases

- **TC-ING1 — Happy path: all 5 stages succeed**
  - What: Mock all RAG components. Call `await run_ingestion_pipeline(["AAPL"])`.
  - Assertions: All 5 stage methods called in order; no exception.

- **TC-ING2 — Missing `SUPABASE_URL` → pipeline aborts before loading**
  - What: Unset `SUPABASE_URL`. Call pipeline.
  - Assertions: `AlphaLoader.load()` is NOT called; `ConfigurationError` or early return.

- **TC-ING3 — Loader stage fails → pipeline aborts; no further stages run**
  - What: Mock `AlphaLoader.load()` to raise.
  - Assertions: Processor, Embedder, VectorStore, GraphStore NOT called.

- **TC-ING4 — Vector upsert fails → graph stage skipped**
  - What: Mock `AlphaVectorStore.upsert()` to raise. Mock other stages to succeed.
  - Assertions: `AlphaGraphStore.upsert_batch()` NOT called; no exception propagated.

- **TC-ING5 — `skip_graph=True` → graph stage skipped even when vector succeeded**
  - What: All stages succeed. Call with `skip_graph=True`.
  - Assertions: `AlphaGraphStore.upsert_batch()` NOT called.

- **TC-ING6 — Sentry breadcrumb fired at each stage**
  - What: Mock `sentry_sdk.add_breadcrumb`. Run successful pipeline.
  - Assertions: `add_breadcrumb` called 5 times (once per stage).

- **TC-ING7 — Empty ticker list → all stages complete with empty results**
  - What: Call `run_ingestion_pipeline([])`.
  - Assertions: No exception; stages complete gracefully with 0 items.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `AlphaLoader`, `AlphaProcessor`, `AlphaEmbedder` | Avoid real data fetching/ML |
| `AlphaVectorStore`, `AlphaGraphStore` | Avoid real DB connections |
| `sentry_sdk.add_breadcrumb`, `capture_exception` | Verify observability |
| `os.environ` | Control env var validation |

---

### `rag/evaluation.py`

**Purpose:** LLM-as-Judge evaluation framework. 4 Claude-powered metrics: `faithfulness`, `context_precision`, `context_recall`, `answer_relevance`. Weighted `overall_score` (0.35/0.25/0.25/0.15). `batch_evaluate()` for multi-sample evaluation.

---

#### Proposed Test Cases

- **TC-EVL1 — `evaluate()` returns `EvaluationReport` with all 4 metric scores**
  - What: Mock Claude to return `{"score": 0.85}` for each metric call.
  - Assertions: Report has 4 `MetricResult` objects; all scores are `0.85`.

- **TC-EVL2 — `overall_score` property uses correct weights**
  - What: Set `faithfulness=1.0`, `context_precision=0.8`, `context_recall=0.6`, `answer_relevance=0.4`.
  - Assertions: `overall_score ≈ 0.35*1.0 + 0.25*0.8 + 0.25*0.6 + 0.15*0.4 = 0.72`.

- **TC-EVL3 — `_parse_json()` strips markdown fences before parsing**
  - What: Call `_parse_json("```json\n{\"score\": 0.7}\n```")`.
  - Assertions: Returns `{"score": 0.7}`.

- **TC-EVL4 — `_parse_json()` falls back to default score on total parse failure**
  - What: Call `_parse_json("This is not JSON at all", default_score=0.5)`.
  - Assertions: Returns `{"score": 0.5}`.

- **TC-EVL5 — `evaluate()` handles Claude API failure gracefully**
  - What: Mock Claude to raise `anthropic.APIError` for one metric.
  - Assertions: That metric gets default score; other metrics succeed; no exception propagated.

- **TC-EVL6 — `batch_evaluate()` returns average `overall_score` for multiple samples**
  - What: Two samples; first gets overall=0.8, second gets overall=0.6. Mock Claude.
  - Assertions: Average overall score logged ≈ 0.7.

- **TC-EVL7 — `latency_seconds` in `EvaluationReport` reflects real elapsed time**
  - What: Mock `time.time()` to return known values. Call `evaluate()`.
  - Assertions: `report.latency_seconds` equals the mocked time delta.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `anthropic.Anthropic.messages.create` | Control evaluation responses |
| `time.time()` | Test latency calculation |

---

### `rag/test_rag_pipeline.py`

**Not Testable (this IS the test file)**  
**Reason:** This file is an existing test suite, not a source file. It should not have tests written for it. It already covers: `TestTimestampNormalization`, `TestAlphaLoaderSchema`, `TestAlphaProcessor`, `TestHashFunctions`, `TestAlphaEmbedder`, `TestAlphaVectorStore`, `TestAlphaRetriever`, `TestAlphaEvaluator`, and `TestPipelineIntegration`.

---

## tools/financial_tools/

### `tools/financial_tools/__init__.py`

**Not Testable**  
**Reason:** Empty package marker with no logic.

---

### `tools/financial_tools/yahoo_finance.py`

**Purpose:** yfinance wrapper for price history, financial ratios (20+ fields), revenue growth, and peer comparison. All functions return `{"error": str}` on any exception — never raise.

---

#### Proposed Test Cases

- **TC-YF1 — `get_price_history()` returns correctly structured OHLCV list**
  - What: Mock `yf.Ticker("AAPL").history()` to return a pandas DataFrame with Date, Open, High, Low, Close, Volume.
  - Assertions: Returns list of dicts; each has `date` (ISO-8601 string), `open`, `high`, `low`, `close`, `volume`.

- **TC-YF2 — `get_price_history()` returns `{"error": ...}` on exception**
  - What: Mock `history()` to raise `Exception("ticker not found")`.
  - Assertions: Returns `{"error": "ticker not found"}`; no exception propagated.

- **TC-YF3 — `get_financial_ratios()` extracts P/E and forward P/E correctly**
  - What: Mock `yf.Ticker.info` to return `{"trailingPE": 25.3, "forwardPE": 21.0, ...}`.
  - Assertions: `result["pe_ratio"] == 25.3`; `result["forward_pe"] == 21.0`.

- **TC-YF4 — `get_financial_ratios()` handles missing `info` fields gracefully**
  - What: Mock `info` to return `{}` (empty dict).
  - Assertions: Returns dict with `None` values for missing fields; no `KeyError`.

- **TC-YF5 — `get_revenue_growth()` computes YoY annual growth correctly**
  - What: Mock income statement with revenue [100M, 120M, 150M] for 3 years.
  - Assertions: YoY growth rates are `20.0%` and `25.0%`.

- **TC-YF6 — `get_revenue_growth()` handles empty income statement**
  - What: Mock `financials` to return empty DataFrame.
  - Assertions: Returns `{"annual_growth": [], "quarterly_growth": [], "ttm_growth": None}`.

- **TC-YF7 — `get_peer_comparison()` computes peer average correctly**
  - What: Mock P/E for `AAPL=25`, `MSFT=30`. Call `get_peer_comparison("AAPL", ["MSFT"])`.
  - Assertions: `result["peer_average"]["pe_ratio"] == 27.5`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `yfinance.Ticker` | Avoid real Yahoo Finance HTTP calls |
| `yf.Ticker.history()` | Avoid real price data fetch |
| `yf.Ticker.info` | Inject fixture financial ratios |

---

### `tools/financial_tools/sec_edgar.py`

**Purpose:** Sync SEC EDGAR client (requests). CIK lookup via `company_tickers.json`. Filing enumeration. Full text extraction with HTML stripping. XBRL financial facts retrieval with dual revenue concept fallback.

---

#### Proposed Test Cases

- **TC-EDGAR1 — `get_cik()` resolves ticker to padded 10-digit CIK**
  - What: Mock `requests.get("https://www.sec.gov/files/company_tickers.json")` to return `{"0": {"ticker": "AAPL", "cik_str": 320193}}`.
  - Assertions: `get_cik("AAPL") == "0000320193"` (zero-padded to 10 digits).

- **TC-EDGAR2 — `get_cik()` raises on unknown ticker**
  - What: Response does not contain `"FAKE"` ticker.
  - Assertions: `pytest.raises(ValueError)` or returns `None`.

- **TC-EDGAR3 — `list_filings()` filters by `form_type`**
  - What: Mock submissions JSON with 3 filings: two `10-K`, one `8-K`. Call `list_filings("AAPL", form_type="10-K")`.
  - Assertions: Returns 2 results; all have `form_type == "10-K"`.

- **TC-EDGAR4 — `get_filing_text()` strips HTML tags**
  - What: Mock document fetch to return `"<b>Revenue grew</b> by <i>20%</i>"`.
  - Assertions: Returned text is `"Revenue grew by 20%"` (no HTML tags).

- **TC-EDGAR5 — `get_xbrl_financials()` falls back to alternate revenue concept**
  - What: Mock XBRL facts to have `RevenueFromContractWithCustomerExcludingAssessedTax` but not `Revenues`.
  - Assertions: Revenue figures are extracted from the fallback concept.

- **TC-EDGAR6 — `get_xbrl_financials()` caps at 10 most recent entries**
  - What: Mock XBRL facts with 15 annual revenue entries.
  - Assertions: Returns exactly 10 entries.

- **TC-EDGAR7 — 150ms polite delay between requests**
  - What: Mock `time.sleep`. Make two sequential API calls.
  - Assertions: `time.sleep` called with value ≈ 0.15.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `requests.get` | Avoid real SEC EDGAR HTTP calls |
| `time.sleep` | Verify polite delay without slowing tests |

---

### `tools/financial_tools/financial_ratio_calculator.py`

**Purpose:** Pure-Python financial ratio library with 16+ calculation functions and `composite_financial_score()` (weighted 0–100 with letter grade). `_safe_div` handles `None` and divide-by-zero.

---

#### Proposed Test Cases

> This file is the most straightforwardly unit-testable in the codebase — pure math with no I/O.

- **TC-FRC1 — `price_to_earnings()` happy path**
  - What: `price_to_earnings(price=150.0, eps=6.0)`. Assertions: Returns `25.0`.

- **TC-FRC2 — `price_to_earnings()` when EPS is zero → `None`**
  - What: `price_to_earnings(price=150.0, eps=0.0)`. Assertions: Returns `None`.

- **TC-FRC3 — `price_to_earnings()` when EPS is negative → `None`**
  - What: `price_to_earnings(price=150.0, eps=-2.0)`. Assertions: Returns `None` (P/E undefined for negative earnings).

- **TC-FRC4 — `gross_margin()` correct percentage**
  - What: `gross_margin(revenue=1000, cogs=600)`. Assertions: Returns `40.0`.

- **TC-FRC5 — `gross_margin()` when revenue is zero → `None`**

- **TC-FRC6 — `return_on_equity()` correct**
  - What: `return_on_equity(net_income=200, equity=1000)`. Assertions: Returns `20.0`.

- **TC-FRC7 — `cagr()` correct over multiple years**
  - What: `cagr(beginning_value=100, ending_value=200, years=7)`. Assertions: ≈ `10.41%`.

- **TC-FRC8 — `cagr()` when `years=0` → `None`**

- **TC-FRC9 — `composite_financial_score()` returns score in [0, 100]**
  - What: Pass full set of valid metrics.
  - Assertions: `0 <= score <= 100`.

- **TC-FRC10 — `composite_financial_score()` assigns correct letter grades**
  - What: Score = 92 → `"A"`, 82 → `"B"`, 72 → `"C"`, 62 → `"D"`, 52 → `"F"`.

- **TC-FRC11 — `composite_financial_score()` with all-`None` inputs → graceful default**
  - What: All metrics are `None`.
  - Assertions: Returns `{"score": 0.0, "grade": "F"}` or similar; no `ZeroDivisionError`.

- **TC-FRC12 — `composite_financial_score()` with partial inputs — missing metrics excluded from denominator**
  - What: Provide only ROE and net_margin (2 of 6 inputs).
  - Assertions: Score is computed from those 2; denominator is 0.45 (25% + 20%), not 1.0.

- **TC-FRC13 — `_safe_div()` returns `None` on divide-by-zero**
  - What: `_safe_div(10, 0)`. Assertions: Returns `None`.

- **TC-FRC14 — `_safe_div()` returns `None` when numerator is `None`**
- **TC-FRC15 — `_safe_div()` returns `None` when denominator is `None`**

- **TC-FRC16 — `debt_to_equity()` correct**
  - What: `debt_to_equity(total_debt=500, equity=1000)`. Assertions: Returns `0.5`.

- **TC-FRC17 — `current_ratio()` correct**
  - What: `current_ratio(current_assets=800, current_liabilities=400)`. Assertions: Returns `2.0`.

- **TC-FRC18 — `peg_ratio()` when P/E is `None` → `None`**

---

**Suggested Mocking Points**

None — this file has no external dependencies. All test cases use direct function calls with primitive float inputs.

---

### `tools/financial_tools/financial_server.py`

**Purpose:** FastMCP stdio server exposing 16 financial tools. All logging redirected to stderr. `_sentry_tool()` wraps each tool call for error capture.

---

#### Proposed Test Cases

- **TC-FS1 — Each tool handler calls the correct underlying function**
  - What: Mock `yahoo_finance.get_financial_ratios`. Call the `tool_get_financial_ratios` MCP handler directly.
  - Assertions: `get_financial_ratios` called with correct ticker argument.

- **TC-FS2 — `tool_calc_pe` calls `financial_ratio_calculator.price_to_earnings`**
  - What: Mock calculator. Call `tool_calc_pe(price=150.0, eps=6.0)`.
  - Assertions: Returns `{"pe_ratio": 25.0}`.

- **TC-FS3 — `_sentry_tool()` captures exception and does NOT re-raise to MCP**
  - What: Mock underlying function to raise. Call via `_sentry_tool`.
  - Assertions: `sentry_sdk.capture_exception` called; returns `{"error": ...}` instead of raising.

- **TC-FS4 — All tools handle `None` ticker gracefully**
  - What: Call each tool without a ticker argument (or with `None`).
  - Assertions: Returns `{"error": ...}` or uses empty default; no `AttributeError`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `yahoo_finance.*` functions | Avoid real Yahoo Finance calls |
| `sec_edgar.*` functions | Avoid real SEC calls |
| `financial_ratio_calculator.*` functions | Unit-isolate MCP routing layer |
| `sentry_sdk.capture_exception` | Verify error handling |

---

## tools/research_tools/

### `tools/research_tools/__init__.py`

**Not Testable**  
**Reason:** Empty package marker.

---

### `tools/research_tools/tavily_search.py`

**Purpose:** Async Tavily API client. Posts to `https://api.tavily.com/search`. Returns structured results with `title, url, snippet, score, published_date`. Raises `KeyError` if `TAVILY_API_KEY` is missing.

---

#### Proposed Test Cases

- **TC-TAV1 — Happy path: returns parsed results dict**
  - What: Mock `httpx.AsyncClient.post()` to return a 200 response with fixture JSON. Call `await tavily_search("AAPL earnings")`.
  - Assertions: Returns `{"query": ..., "answer": ..., "results": [...]}`.

- **TC-TAV2 — `TAVILY_API_KEY` missing → `KeyError` raised at function start**
  - What: Unset `TAVILY_API_KEY` env var. Call `await tavily_search("query")`.
  - Assertions: `pytest.raises(KeyError)`.

- **TC-TAV3 — HTTP 4xx response → raises or returns `{"error": ...}`**
  - What: Mock response with status 429 (rate limit).
  - Assertions: Exception or error dict returned; no unhandled exception.

- **TC-TAV4 — HTTP 5xx response → raises or returns `{"error": ...}`**
  - What: Mock response with status 500.

- **TC-TAV5 — `max_results` parameter is sent in request body**
  - What: Call with `max_results=3`. Inspect mock call args.
  - Assertions: Request body contains `"max_results": 3`.

- **TC-TAV6 — `include_domains` filter is included in request when provided**
  - What: Call with `include_domains=["reuters.com"]`.
  - Assertions: Request body contains domain filter.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `httpx.AsyncClient.post` | Avoid real Tavily API calls |
| `os.environ["TAVILY_API_KEY"]` | Control auth key presence |

---

### `tools/research_tools/news_search.py`

**Purpose:** Async NewsAPI client. Defaults `from_date` to 30 days ago. Filters `[Removed]` articles. Raises `KeyError` if `NEWSAPI_KEY` missing.

---

#### Proposed Test Cases

- **TC-NS1 — Happy path: returns parsed articles list**
  - What: Mock `httpx.AsyncClient.get()` to return 200 with fixture JSON containing 3 articles.
  - Assertions: Returns `{"query": ..., "total_results": 3, "articles": [...]}`.

- **TC-NS2 — `[Removed]` articles filtered out**
  - What: Fixture contains 2 valid articles and 1 with `"[Removed]"` content.
  - Assertions: Returned `articles` list has 2 items.

- **TC-NS3 — `NEWSAPI_KEY` missing → `KeyError`**
  - What: Unset env var.
  - Assertions: `pytest.raises(KeyError)`.

- **TC-NS4 — `from_date` defaults to 30 days ago when not specified**
  - What: Call without `from_date`. Inspect mock request query params.
  - Assertions: `from` param in request is today − 30 days.

- **TC-NS5 — Empty results → returns `{"articles": []}`**
  - What: Mock API to return `{"articles": [], "totalResults": 0}`.
  - Assertions: Returns expected structure without error.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `httpx.AsyncClient.get` | Avoid real NewsAPI calls |
| `datetime.date.today` | Control "30 days ago" calculation |

---

### `tools/research_tools/sec_edgar.py`

**Purpose:** Async SEC EDGAR client (httpx). CIK lookup via `browse-edgar` ATOM XML. Full-text search via `efts.sec.gov`. Section parsing from filing text via regex. In-memory `_CIK_CACHE`.

---

#### Proposed Test Cases

- **TC-ASEC1 — `sec_edgar_search()` returns filing metadata list**
  - What: Mock `httpx.AsyncClient.get` for `efts.sec.gov/LATEST/search-index`. Return fixture JSON with 2 filings.
  - Assertions: Returns list of 2 dicts with `accession_number`, `form_type`, `filing_date`.

- **TC-ASEC2 — `sec_edgar_search()` quotes multi-word query**
  - What: Call `await sec_edgar_search("revenue growth")`. Inspect request URL.
  - Assertions: Query string contains `"revenue+growth"` or `%22revenue+growth%22` (quoted).

- **TC-ASEC3 — `sec_edgar_filing()` extracts `business` section from filing text**
  - What: Mock filing text containing `ITEM 1. BUSINESS\n<content>`. Call with `sections=["business"]`.
  - Assertions: Returns `{"sections": {"business": "<content>"}}`.

- **TC-ASEC4 — `sec_edgar_filing()` caps text at 500,000 chars**
  - What: Mock filing text with 600,000 chars.
  - Assertions: Only first 500,000 chars are processed.

- **TC-ASEC5 — `_CIK_CACHE` prevents duplicate CIK lookups**
  - What: Call `sec_edgar_filing("AAPL", ...)` twice. Mock CIK lookup.
  - Assertions: HTTP request for CIK made only once (second uses cache).

- **TC-ASEC6 — Unknown ticker → appropriate error**
  - What: Mock CIK lookup to return empty ATOM response.
  - Assertions: Raises `ValueError` or returns `{"error": ...}`.

- **TC-ASEC7 — Missing section header → section key absent from result**
  - What: Filing text does not contain `ITEM 1A. RISK FACTORS`. Call with `sections=["risk_factors"]`.
  - Assertions: `result["sections"]["risk_factors"]` is `""` or key absent.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `httpx.AsyncClient.get` | Avoid real EDGAR HTTP calls |
| `_CIK_CACHE` dict | Reset between tests for isolation |

---

### `tools/research_tools/research_server.py`

**Purpose:** MCP stdio server with 7 research tools routed via `match/case`. All tool errors caught and returned as `{"error": ..., "tool": name}` with `isError=True`. Never crashes the MCP loop.

---

#### Proposed Test Cases

- **TC-RS1 — `call_tool("tavily_search", args)` delegates to `tavily_search()`**
  - What: Mock `tavily_search`. Call MCP `call_tool` handler.
  - Assertions: `tavily_search` called with correct args; result returned as `TextContent`.

- **TC-RS2 — `call_tool("rag_hybrid_query", args)` delegates to `rag_hybrid_query()`**
  - What: Mock `rag_hybrid_query`. Call handler.
  - Assertions: Delegated correctly.

- **TC-RS3 — Unknown tool name → `isError=True` response**
  - What: Call `call_tool("nonexistent_tool", {})`.
  - Assertions: Returns `[TextContent]` with `isError=True`; no exception.

- **TC-RS4 — Tool raises exception → `isError=True` response + Sentry capture**
  - What: Mock `news_search` to raise. Call `call_tool("news_search", {})`.
  - Assertions: Returns error response; `sentry_sdk.capture_exception` called.

- **TC-RS5 — `list_tools()` returns all 7 tool definitions**
  - What: Call the `list_tools` handler.
  - Assertions: Returns list of 7 `Tool` objects with non-empty `name` and `description`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `tavily_search`, `news_search`, `sec_edgar_search`, `sec_edgar_filing` | Avoid external HTTP |
| `rag_vector_search`, `rag_graph_traverse`, `rag_hybrid_query` | Avoid DB calls |
| `sentry_sdk.capture_exception` | Verify error reporting |

---

## tools/sentiment_tools/

### `tools/sentiment_tools/__init__.py`

**Not Testable (currently broken)**  
**Reason:** All 4 module-level imports in this file reference non-existent module paths (`tools.local_social_retriever`, `tools.finbert_analyzer`, etc.). Any import of this package raises `ModuleNotFoundError`. The file must be fixed before tests can run against anything in `tools.sentiment_tools`.

---

### `tools/sentiment_tools/finbert_analyzer.py`

**Purpose:** ProsusAI/FinBERT sentiment analyzer. Thread-safe singleton with double-checked locking. Batched inference with GPU OOM fallback to CPU. `reset_finbert()` for test teardown.

---

#### Proposed Test Cases

- **TC-FB1 — `FinBertSentimentAnalyzer` is a singleton**
  - What: Mock `BertTokenizer.from_pretrained` and `BertForSequenceClassification.from_pretrained`. Instantiate twice.
  - Assertions: Both instances are `is`-identical; model loaded only once.

- **TC-FB2 — `reset_finbert()` allows fresh instantiation**
  - What: Instantiate once. Call `reset_finbert()`. Instantiate again.
  - Assertions: Model constructor called twice.

- **TC-FB3 — `analyze()` returns correct `FinBertResult` structure**
  - What: Mock `_infer_batch()` to return `[[0.8, 0.1, 0.1]]` (bullish/bearish/neutral probs). Call `analyze(["Apple beats earnings"])`.
  - Assertions: `result.bullish_prob ≈ 0.8`; `result.label == "Bullish"`.

- **TC-FB4 — `analyze()` filters empty strings before batching**
  - What: Call `analyze(["", "Apple grew 20%", ""])`.
  - Assertions: `_infer_batch` called with only 1 text (non-empty); result reflects 1-text corpus.

- **TC-FB5 — `analyze()` with empty input list → neutral defaults**
  - What: Call `analyze([])`.
  - Assertions: Returns `FinBertResult` with all probs ≈ 0.33 or all 0.0; no exception.

- **TC-FB6 — GPU OOM fallback: switches to CPU mid-batch**
  - What: Mock first `model()` call to raise `torch.cuda.OutOfMemoryError`; second call succeeds.
  - Assertions: Second call made; result returned without exception.

- **TC-FB7 — Label assignment: bearish_prob > bullish_prob → "Bearish"**
  - What: Mock probs `[0.1, 0.8, 0.1]`.
  - Assertions: `result.label == "Bearish"`.

- **TC-FB8 — Label assignment: neutral wins three-way tie**
  - What: Mock probs `[0.33, 0.33, 0.34]`.
  - Assertions: `result.label == "Neutral"`.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `transformers.BertTokenizer.from_pretrained` | Avoid 440MB download |
| `transformers.BertForSequenceClassification.from_pretrained` | Avoid model download |
| `torch.cuda.is_available` | Control device selection |
| The FinBERT model's `forward()` call | Inject known logit tensors |

---

### `tools/sentiment_tools/vader_scorer.py`

**Purpose:** NLTK VADER scorer. Auto-downloads `vader_lexicon` on first use. Thread-safe singleton. Mean corpus aggregation. Label thresholds ±0.05. `reset_vader()` for test teardown. `score_single()` raises `ValueError` on empty input.

---

#### Proposed Test Cases

- **TC-VADER1 — `VaderLexiconScorer` is a singleton**
  - What: Mock `nltk.download` and `SentimentIntensityAnalyzer`. Instantiate twice.
  - Assertions: Same instance returned; `nltk.download` called once.

- **TC-VADER2 — `reset_vader()` allows fresh instantiation**
  - What: Instantiate. `reset_vader()`. Instantiate again.
  - Assertions: `SentimentIntensityAnalyzer` constructor called twice.

- **TC-VADER3 — `score()` returns correct `VaderResult` with mean compound**
  - What: Mock `polarity_scores()` to return `{"compound": 0.8}` for 2 texts.
  - Assertions: `result.compound ≈ 0.8`; `result.label == "Bullish"`.

- **TC-VADER4 — `score()` with compound ≥ 0.05 → "Bullish"**
  - Assertions: `result.label == "Bullish"`.

- **TC-VADER5 — `score()` with compound ≤ -0.05 → "Bearish"**
  - Assertions: `result.label == "Bearish"`.

- **TC-VADER6 — `score()` with compound in (-0.05, 0.05) → "Neutral"**
  - Assertions: `result.label == "Neutral"`.

- **TC-VADER7 — `score_single("")` raises `ValueError`**
  - What: Call `score_single("")`.
  - Assertions: `pytest.raises(ValueError)`.

- **TC-VADER8 — `score()` with empty list → neutral defaults or empty result**
  - What: Call `score([])`.
  - Assertions: No exception; returns `VaderResult` with `compound == 0.0`.

- **TC-VADER9 — Each `VaderLexiconScorer` has its own `SentimentIntensityAnalyzer`**
  - What: Create two separate scorer objects (reset between creation).
  - Assertions: Each has a separate `SentimentIntensityAnalyzer` instance (not thread-safe to share).

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `nltk.download` | Avoid downloading lexicon in CI |
| `nltk.sentiment.SentimentIntensityAnalyzer` | Inject known compound scores |

---

### `tools/sentiment_tools/fear_greed_calculator.py`

**Purpose:** Aggregates FinBERT and VADER signals into a normalized [-1.0, +1.0] Fear/Greed score. `_validate_weights()` requires non-negative values summing to 1.0 ±0.001. Five label bands. `calculate_from_dict()` for JSON/MCP deserialized inputs.

---

#### Proposed Test Cases

> Entirely pure math — no I/O. Most comprehensive unit test coverage opportunity in the codebase.

- **TC-FGI1 — Default weights sum to 1.0**
  - What: Instantiate `FearGreedIndexCalculator()`.
  - Assertions: `W_FINBERT + W_VADER ≈ 1.0` (0.65 + 0.35).

- **TC-FGI2 — `_validate_weights()` raises on weights not summing to 1.0**
  - What: Pass `w_finbert=0.7, w_vader=0.4`.
  - Assertions: `pytest.raises(ValueError)`.

- **TC-FGI3 — `_validate_weights()` raises on negative weight**
  - What: Pass `w_finbert=-0.1, w_vader=1.1`.
  - Assertions: `pytest.raises(ValueError)`.

- **TC-FGI4 — `calculate()` — Extreme Greed: score ≥ 0.60**
  - What: FinBERT bullish_prob=0.9, bearish_prob=0.05. VADER compound=0.9.
  - Assertions: `result.score >= 0.60`; `result.label == "Extreme Greed"`.

- **TC-FGI5 — `calculate()` — Greed: 0.20 ≤ score < 0.60**
  - What: FinBERT bullish_prob=0.6, bearish_prob=0.2. VADER compound=0.4.
  - Assertions: `result.label == "Greed"`.

- **TC-FGI6 — `calculate()` — Neutral: -0.20 < score < 0.20**
  - What: Balanced FinBERT/VADER near zero.
  - Assertions: `result.label == "Neutral"`.

- **TC-FGI7 — `calculate()` — Fear: -0.60 ≤ score ≤ -0.20**
  - Assertions: `result.label == "Fear"`.

- **TC-FGI8 — `calculate()` — Extreme Fear: score ≤ -0.60**
  - What: FinBERT bullish_prob=0.05, bearish_prob=0.90. VADER compound=-0.9.
  - Assertions: `result.label == "Extreme Fear"`.

- **TC-FGI9 — Score is clamped to [-1.0, +1.0]**
  - What: Force raw score > 1.0 (edge case with floating point).
  - Assertions: `result.score <= 1.0`.

- **TC-FGI10 — `confidence` equals `abs(score)`**
  - What: Score = -0.75. Assertions: `result.confidence == 0.75`.

- **TC-FGI11 — `calculate_from_dict()` correctly reconstructs from JSON**
  - What: Pass `finbert_dict = {"bullish_prob": 0.7, "bearish_prob": 0.2, "neutral_prob": 0.1, "label": "Bullish", "chunk_scores": []}`. Pass a valid `vader_dict`.
  - Assertions: Returns same result as `calculate()` with equivalent objects.

- **TC-FGI12 — `calculate_from_dict()` handles missing keys gracefully**
  - What: Pass `finbert_dict = {}`.
  - Assertions: Either raises `KeyError` (documented behavior) or returns neutral default.

- **TC-FGI13 — finbert_score formula: `bullish_prob − bearish_prob`**
  - What: bullish=0.7, bearish=0.2 → finbert_score=0.5. With default weights: raw = 0.5*0.65 + compound*0.35.
  - Assertions: result.score matches formula.

---

**Suggested Mocking Points**

None — pure computation. All tests use direct instantiation with float inputs.

---

### `tools/sentiment_tools/sentiment_server.py`

**Purpose:** MCP stdio server with 4 sentiment tools. `retrieve_social_data` uses `AlphaRetriever`. `analyze_finbert` and `score_vader` run in `asyncio.to_thread`. `calculate_fear_greed` accepts optional per-call weight overrides. Env var weight overrides at startup.

---

#### Proposed Test Cases

- **TC-SS1 — `call_tool("retrieve_social_data")` delegates to `AlphaRetriever.retrieve_raw()`**
  - What: Mock `_get_retriever()` to return mock retriever. Call handler.
  - Assertions: `retrieve_raw` called with correct query/ticker/days_back.

- **TC-SS2 — `call_tool("analyze_finbert")` runs in thread (non-blocking)**
  - What: Mock `asyncio.to_thread`. Call handler.
  - Assertions: `asyncio.to_thread` called (not direct `FinBertSentimentAnalyzer.analyze()`).

- **TC-SS3 — `call_tool("calculate_fear_greed")` uses per-call weight overrides**
  - What: Pass `{"finbert_weight": 0.5, "vader_weight": 0.5}` in args. Mock calculator.
  - Assertions: `FearGreedIndexCalculator` instantiated with those weights.

- **TC-SS4 — `FEAR_GREED_FINBERT_WEIGHT` env var overrides default weight at startup**
  - What: Set `FEAR_GREED_FINBERT_WEIGHT=0.8`. Reload module. Inspect default weights used.
  - Assertions: Default `W_FINBERT == 0.8`.

- **TC-SS5 — Tool error → `isError=True` response**
  - What: Mock `FinBertSentimentAnalyzer.analyze()` to raise. Call `call_tool("analyze_finbert")`.
  - Assertions: Response has `isError=True`; no exception propagated.

- **TC-SS6 — `_to_dict()` converts nested dataclasses to JSON-safe dicts**
  - What: Pass a `FinBertResult` dataclass (with nested list of `ChunkSentiment`). Call `_to_dict()`.
  - Assertions: Result is a plain dict with no dataclass instances.

- **TC-SS7 — `list_tools()` returns exactly 4 tool definitions**
  - What: Call the `list_tools` handler.
  - Assertions: Returns list of 4 `Tool` objects.

---

**Suggested Mocking Points**

| Target | Reason |
|--------|---------|
| `AlphaRetriever` | Avoid Supabase + embedder |
| `FinBertSentimentAnalyzer.analyze` | Avoid model loading |
| `VaderLexiconScorer.score` | Avoid NLTK download |
| `asyncio.to_thread` | Verify threading behavior |

---

## evaluation/

### `evaluation/backtester.py`

**Not Testable**  
**Reason:** Stub file containing only a single comment line. No implementation exists.

---

### `evaluation/metrics.py`

**Not Testable**  
**Reason:** Stub file containing only a single comment line. No implementation exists.

---

## scheduler/

### `scheduler/daily_refresh.py`

**Not Testable**  
**Reason:** Stub file containing only a single comment line. No implementation exists.

---

## Summary

### Files by Testability

| Status | Count | Files |
|--------|-------|-------|
| Fully Testable | 27 | All agent, memory, core, api, rag, and tools source files with logic |
| Not Testable | 10 | `__init__.py` (no logic), stubs, TypedDicts-only, existing test file |

### Not Testable Files

| File | Reason |
|------|--------|
| `main.py` | Dead code stub |
| `agents/__init__.py` | Pure re-exports |
| `agents/state.py` | TypedDicts only — no runtime logic |
| `core/__init__.py` | Empty package marker |
| `rag/__init__.py` | Pure re-exports |
| `tools/financial_tools/__init__.py` | Empty package marker |
| `tools/research_tools/__init__.py` | Empty package marker |
| `tools/sentiment_tools/__init__.py` | Broken imports — untestable until fixed |
| `evaluation/backtester.py` | Stub |
| `evaluation/metrics.py` | Stub |
| `scheduler/daily_refresh.py` | Stub |
| `rag/test_rag_pipeline.py` | This IS the test file |

### Total Proposed Test Cases

| Module | Test Cases |
|--------|-----------|
| `agents/research_agent.py` | 16 |
| `agents/financial_agent.py` | 11 |
| `agents/sentiment_agent.py` | 13 |
| `agents/manager_agent.py` | 18 |
| `memory/manager_memory.py` | 13 |
| `core/observability.py` | 5 |
| `core/error_handler.py` | 7 |
| `api/main.py` | 9 |
| `api/config.py` | 7 |
| `api/dependencies.py` | 3 |
| `api/routes/analyze.py` | 9 |
| `api/core/exceptions.py` | 4 |
| `rag/loader.py` | 13 |
| `rag/processor.py` | 7 |
| `rag/embedding_manager.py` | 7 |
| `rag/vector_store.py` | 5 |
| `rag/graph_store.py` | 7 |
| `rag/retriever.py` | 8 |
| `rag/hybrid_rag.py` | 7 |
| `rag/ingestion.py` | 7 |
| `rag/evaluation.py` | 7 |
| `tools/financial_tools/yahoo_finance.py` | 7 |
| `tools/financial_tools/sec_edgar.py` | 7 |
| `tools/financial_tools/financial_ratio_calculator.py` | 18 |
| `tools/financial_tools/financial_server.py` | 4 |
| `tools/research_tools/tavily_search.py` | 6 |
| `tools/research_tools/news_search.py` | 5 |
| `tools/research_tools/sec_edgar.py` | 7 |
| `tools/research_tools/research_server.py` | 5 |
| `tools/sentiment_tools/finbert_analyzer.py` | 8 |
| `tools/sentiment_tools/vader_scorer.py` | 9 |
| `tools/sentiment_tools/fear_greed_calculator.py` | 13 |
| `tools/sentiment_tools/sentiment_server.py` | 7 |
| **Total** | **~282** |

### Priority Notes for Test Implementation

1. **Highest priority — write first:**
   - `tools/financial_tools/financial_ratio_calculator.py` (pure math, zero mocking needed)
   - `tools/sentiment_tools/fear_greed_calculator.py` (pure math, zero mocking needed)
   - `api/core/exceptions.py` (pure Python, trivial to test)
   - `api/config.py` (critical startup path, env-var controlled)

2. **High priority — fix `tools/sentiment_tools/__init__.py` before writing tests** for any sentiment tool file, otherwise all imports will fail.

3. **High priority — fix the `SentimentAgent()` constructor bug** in `api/main.py:107` before integration tests can run.

4. **Moderate priority — requires heavier mocking:**
   - All four agents (mock LLM + MCP transport)
   - `rag/ingestion.py` (mock all 5 stages)
   - `memory/manager_memory.py` (mock Supabase)

5. **Lower priority — integration/E2E tests for:**
   - `api/routes/analyze.py` (requires full FastAPI test client)
   - `rag/hybrid_rag.py` (requires both Supabase and Neo4j mocks)
