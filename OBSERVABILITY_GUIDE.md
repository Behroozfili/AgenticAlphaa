# Alpha-Agent Node — Observability Implementation Guide

This guide describes how to add **Sentry** (error tracking) and **LangSmith** (LLM/agent tracing) to the Alpha-Agent Node codebase, with graceful degradation when the relevant env vars are unset. It contains no code — only precise, file-by-file instructions referencing real symbols in this repository.

---

## 1. Codebase Summary

Alpha-Agent Node is a multi-agent financial research system built on FastAPI + LangGraph + Anthropic Claude + MCP (Model Context Protocol) + Supabase + Neo4j.

**Entry point**: [api/main.py](api/main.py) boots a FastAPI app via a `lifespan` context manager. It validates settings (`validate_settings()`), creates a shared Supabase client (`app.state.supabase`), instantiates the three specialist agents, builds a `ManagerMemory`, and compiles `ManagerAgent` once (`app.state.manager_agent`). It registers two exception handlers — one for `AlphaAgentError` subclasses, one catch-all — and one router, `analyze_router`, mounted at `/api/v1`.

**API layer**: [api/routes/analyze.py](api/routes/analyze.py) exposes `POST /api/v1/analyze`. It builds `manager_directives`, calls `manager_agent.run(...)`, persists the result to the Supabase `analyses` table via `_persist_analysis()`, and returns `AnalyzeResponse`. Notably it currently raises `HTTPException` directly on failure rather than an `AgentError` from [api/core/exceptions.py](api/core/exceptions.py).

**Exception hierarchy**: [api/core/exceptions.py](api/core/exceptions.py) defines `AlphaAgentError` (base, carries `trace_id`, `code`, `http_status`) and subclasses `ValidationError`, `AgentError`, `AgentTimeoutError`, `MemoryError`, `ExternalServiceError`. This file already exists — see Warning 1 below.

**Orchestration**: [agents/manager_agent.py](agents/manager_agent.py) implements `ManagerAgent`, a LangGraph state machine with nodes `hydrate → brain_route → dispatch → evaluate → persist → finalise|abort`, implemented as `_node_hydrate`, `_node_brain_route`, `_node_dispatch`, `_node_evaluate`, `_node_persist`, `_node_finalise`, `_node_abort`. The `_dispatch()` helper (called from `_node_dispatch`) invokes `agent.run(state)` for whichever specialist agent the Brain routed to, and **swallows exceptions** — it records `outcome="error"` in `agent_execution_history`/`orchestrator_logs` and returns the unmodified state rather than re-raising. `_node_finalise()` and `_node_abort()` call `self._memory.persist_long_term()` unguarded (see Warning 5).

**Specialist agents** (each takes a `SharedManagerState`, returns a mutated copy):
- [agents/research_agent.py](agents/research_agent.py) — `ResearchAgent`, internal LangGraph with nodes `_brain_node → _executor_node → _checker_node`, looping via `_should_continue`. Connects to `tools/research_tools/research_server.py` over stdio MCP.
- [agents/financial_agent.py](agents/financial_agent.py) — `FinancialAnalystAgent`, three-tier (no LangGraph — a plain `while` loop in `run()`): `_brain()` → `_execute_data_extraction()` + `_execute_ratio_computation()` → `_check_data_quality()`. Connects to `tools/financial_tools/financial_server.py`.
- [agents/sentiment_agent.py](agents/sentiment_agent.py) — `SentimentAgent`, two-tier: `_brain_plan()` → `_execute_sentiment_pipeline()` → `_brain_analyze()`, driven by `run()`'s `while` loop. Connects to `tools/sentiment_tools/sentiment_server.py`.

**MCP tool servers**:
- [tools/research_tools/research_server.py](tools/research_tools/research_server.py) — raw `mcp.server.Server`, single `@app.call_tool()` dispatcher (`async def call_tool(name, arguments)`) routing via `match/case` to `tavily_search`, `news_search`, `sec_edgar_search`, `sec_edgar_filing`, `rag_vector_search`, `rag_graph_traverse`, `rag_hybrid_query`.
- [tools/sentiment_tools/sentiment_server.py](tools/sentiment_tools/sentiment_server.py) — same `Server` + single `call_tool()` pattern, routing to `retrieve_social_data`, `analyze_finbert`, `score_vader`, `calculate_fear_greed`.
- [tools/financial_tools/financial_server.py](tools/financial_tools/financial_server.py) — **structurally different**: built on `FastMCP` with one `@mcp.tool()`-decorated function per tool (16 tools: `tool_get_price_history`, `tool_get_financial_ratios`, `tool_get_revenue_growth`, `tool_get_peer_comparison`, `tool_get_cik`, `tool_list_filings`, `tool_get_filing_text`, `tool_get_xbrl_financials`, `tool_calc_pe`, `tool_calc_pb`, `tool_calc_ev_ebitda`, `tool_calc_peg`, `tool_calc_gross_margin`, `tool_calc_operating_margin`, `tool_calc_net_margin`, `tool_calc_roe`, `tool_calc_roa`, `tool_calc_current_ratio`, `tool_calc_quick_ratio`, `tool_calc_debt_to_equity`, `tool_calc_interest_coverage`, `tool_calc_asset_turnover`, `tool_calc_cagr`, `tool_calc_composite_score`). There is **no central `call_tool()` dispatcher** to instrument — see Warning 7.

**RAG pipeline**:
- [rag/hybrid_rag.py](rag/hybrid_rag.py) — `rag_vector_search()`, `rag_graph_traverse()`, `rag_hybrid_query()`. These are the functions exposed as MCP tools by `research_server.py`.
- [rag/ingestion.py](rag/ingestion.py) — `run_ingestion_pipeline(tickers, skip_graph)`, a 5-stage ETL: `AlphaLoader.load()` → `AlphaProcessor.process()` → `embedder.embed_chunks()` → `AlphaVectorStore.upsert()` → `AlphaGraphStore.extract_batch()`/`upsert_batch()`. Each stage already has a `try/except` that logs and `return`s early on failure — this is sequential pipeline short-circuiting, not per-stage isolation.
- [rag/retriever.py](rag/retriever.py) — `AlphaRetriever`, used by `sentiment_server.py`'s `_retrieve_social_data()` via `retrieve_raw()`.
- Supporting: [rag/loader.py](rag/loader.py), [rag/processor.py](rag/processor.py), [rag/embedding_manager.py](rag/embedding_manager.py), [rag/vector_store.py](rag/vector_store.py), [rag/graph_store.py](rag/graph_store.py).

**Memory**: [memory/manager_memory.py](memory/manager_memory.py) — `ManagerMemory` facade composing `ShortTermMemory` (in-process) and `LongTermMemory` (Supabase-backed). `LongTermMemory.load()` catches all exceptions and falls back to empty state; `LongTermMemory.persist()` catches, logs, and **re-raises** — the asymmetry matters because `ManagerMemory.persist_long_term()` is called unguarded from `manager_agent.py`'s `_node_finalise()`/`_node_abort()` (Warning 5).

**Dependencies**: [requirements.txt](requirements.txt) already lists `langsmith` (line 10) but it is currently unused anywhere except an orphaned `tools/decorator.py`. `sentry-sdk` is **not present at all**.

---

## 2. Dependency Map

```
api/main.py (entry point)
  ├─ imports api/core/exceptions.py (AlphaAgentError)
  ├─ imports api/config.py (settings, validate_settings)
  ├─ instantiates ResearchAgent, FinancialAnalystAgent, SentimentAgent
  ├─ instantiates ManagerMemory (memory/manager_memory.py)
  ├─ instantiates ManagerAgent (agents/manager_agent.py)
  └─ mounts api/routes/analyze.py (router)
        └─ depends on api/dependencies.py (get_manager_memory)
        └─ calls ManagerAgent.run()
              └─ LangGraph: hydrate → brain_route → dispatch → evaluate → persist → finalise|abort
                    └─ _dispatch() → {ResearchAgent, FinancialAnalystAgent, SentimentAgent}.run()
                          ├─ ResearchAgent.run() → LangGraph brain→executor→checker
                          │     └─ MCP stdio → tools/research_tools/research_server.py
                          │           └─ rag/hybrid_rag.py (rag_vector_search/rag_graph_traverse/rag_hybrid_query)
                          ├─ FinancialAnalystAgent.run() → brain→extract→compute→check loop
                          │     └─ MCP stdio → tools/financial_tools/financial_server.py (FastMCP)
                          └─ SentimentAgent.run() → brain_plan→execute→brain_analyze loop
                                └─ MCP stdio → tools/sentiment_tools/sentiment_server.py
                                      └─ rag/retriever.py (AlphaRetriever.retrieve_raw)
                    └─ _node_finalise/_node_abort → ManagerMemory.persist_long_term()
                          └─ memory/manager_memory.py LongTermMemory.persist() → Supabase

rag/ingestion.py (separate CLI entry point, not wired into the API)
  └─ rag/loader.py → rag/processor.py → rag/embedding_manager.py → rag/vector_store.py → rag/graph_store.py
```

Instrumentation must be layered at every box above without changing any function's signature.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Create the centralized observability bootstrap module

**File to create/edit**: `core/observability.py` (new directory `core/` at the project root, alongside `agents/`, `api/`, `rag/`, `tools/`, `memory/`)

**Why**: Every other module needs one place to import `init_sentry()` and `init_langsmith()` so initialization logic, env-var checks, and graceful-degradation behavior live in a single auditable spot rather than being duplicated across `api/main.py`, the three agent files, and three MCP server scripts.

**What to add**: Two idempotent functions:
- `init_sentry() -> bool` — reads `SENTRY_DSN` from the environment. If unset or empty, log at INFO level that Sentry is disabled and return `False` without importing `sentry_sdk`. If set, call `sentry_sdk.init(dsn=..., traces_sample_rate=..., environment=settings.APP_ENV)` and return `True`. Wrap the whole body in `try/except` so a malformed DSN or missing `sentry-sdk` package degrades to disabled rather than crashing the process.
- `init_langsmith() -> bool` — reads `LANGSMITH_API_KEY`. If unset, log at INFO level that LangSmith tracing is disabled and return `False`. If set, set `os.environ["LANGCHAIN_TRACING_V2"] = "true"` and `os.environ["LANGCHAIN_PROJECT"]` from a `LANGSMITH_PROJECT` env var (default `"alpha-agent-node"`); LangSmith's Python SDK auto-instruments via these env vars plus the `@traceable` decorator described in later steps, so no client object needs to be returned. Also add a module-level helper, e.g. `def langsmith_enabled() -> bool`, that other modules can check before wrapping calls in `@traceable` (or simply always apply `@traceable` — the LangSmith SDK itself no-ops safely when tracing is disabled, but exposing the helper lets MCP server scripts skip the import entirely when no API key is present, avoiding an unnecessary dependency import in subprocess-launched servers).

**Depends on**: Nothing (this is the foundational module). Must be created before Steps 2–20.

---

### Step 2: Create the async error-handling wrapper

**File to create/edit**: `core/error_handler.py` (new file, same new `core/` directory as Step 1)

**Why**: Several places in this codebase already catch exceptions and log them (e.g. every MCP `call_tool()`, `ResearchAgent._executor_node`, `FinancialAnalystAgent._execute_data_extraction`) but none of them report to Sentry. Rather than editing every `try/except` block individually with raw `sentry_sdk.capture_exception()` calls, a single reusable decorator/context-manager keeps the instrumentation consistent and makes it trivial to add breadcrumbs.

**What to add**: An async-aware decorator, e.g. `with_error_reporting(component: str)`, that wraps an `async def` function: on entry it adds a Sentry breadcrumb (category=`component`, message=the function's `__name__`, data=the call's kwargs where safe to serialize); on exception it calls `sentry_sdk.capture_exception(exc)` tagged with `component`, then re-raises so existing control flow (including `manager_agent.py`'s deliberate exception-swallowing in `_dispatch()`) is unchanged. Guard every Sentry call with the `langsmith`/`sentry` enabled-flags from Step 1 (i.e., no-op cleanly if `init_sentry()` returned `False`). Because several functions being wrapped are synchronous (e.g. `FinancialAnalystAgent._brain()`, `SentimentAgent._brain_plan()`), also provide a sync variant or detect via `asyncio.iscoroutinefunction`.

**Depends on**: Step 1 (`core/observability.py` must already expose the Sentry init state this module checks).

---

### Step 3: Wire observability into the application entry point

**File to create/edit**: [api/main.py](api/main.py)

**Why**: `api/main.py`'s `lifespan()` function is the single startup hook already used to validate settings and build shared resources (`app.state.supabase`, `app.state.manager_agent`) — it is the natural place to initialize Sentry and LangSmith exactly once per process, before any agent or route handler runs.

**What to add**: At the top of `lifespan()`, immediately after `validate_settings()` (line ~79), call `init_sentry()` and `init_langsmith()` from `core/observability.py` and log their boolean results (e.g. `"Sentry enabled: %s, LangSmith enabled: %s"`). In the existing `alpha_agent_exception_handler()` (the `@app.exception_handler(AlphaAgentError)` function), add a Sentry breadcrumb/capture call using `exc.trace_id` as a Sentry tag so the `trace_id` already returned to API callers also appears in Sentry's UI — this lets you correlate a user-reported `trace_id` with a Sentry event directly. Do the same in `unhandled_exception_handler()` (the catch-all `@app.exception_handler(Exception)`), since this is the last line of defense for any exception that escapes `ManagerAgent.run()` uncaught.

**Depends on**: Steps 1 and 2.

---

### Step 4: Bridge `AlphaAgentError` raising in the analyze route

**File to create/edit**: [api/routes/analyze.py](api/routes/analyze.py)

**Why**: The `analyze()` handler currently raises `HTTPException` directly (line ~284) when `manager_agent.run()` fails, which bypasses the `AlphaAgentError` handler registered in `api/main.py` and therefore bypasses the Sentry capture added in Step 3 for that handler. **Warning**: this is a deviation from the rest of the codebase's intended error-handling design (the existence of `api/core/exceptions.py`'s rich hierarchy implies routes should raise `AgentError`, not `HTTPException`) — flagging this explicitly because fixing it changes existing behavior (response body shape changes from the ad-hoc `detail={...}` dict to `AgentError.to_dict()`'s `{error, message, detail, trace_id}` shape).

**What to add**: Replace the `raise HTTPException(status_code=500, detail={...})` block with `raise AgentError(message="Agent pipeline failed", detail=error_message)` (importing `AgentError` from `api.core.exceptions`), so it flows through the handler instrumented in Step 3 and gets a `trace_id` for free. Additionally, wrap the whole `try/except` around `manager_agent.run(...)` (lines ~248–257) with a Sentry breadcrumb before the call, recording `analysis_id`, `ticker`, and `search_depth` as breadcrumb data — this is the single highest-value breadcrumb in the system since it marks the start of every external-facing request.

**Depends on**: Steps 1–3.

---

### Step 5: Add a root LangSmith trace to `ManagerAgent.run()`

**File to create/edit**: [agents/manager_agent.py](agents/manager_agent.py)

**Why**: `ManagerAgent.run()` is the single entry point for every analysis request and the root of the entire agent execution tree (hydrate → brain_route → dispatch → evaluate → persist → finalise/abort, recursively dispatching into three sub-agents). A LangSmith root trace here gives a single trace ID that every downstream agent/tool/LLM-call span can attach to as a child, producing one coherent visual timeline per `/api/v1/analyze` request.

**What to add**: Apply the LangSmith `@traceable(name="ManagerAgent.run", run_type="chain")` decorator (from the `langsmith` package, already in `requirements.txt`) to `ManagerAgent.run()`. Because the decorator is a no-op-safe wrapper when `LANGCHAIN_TRACING_V2` is unset (per Step 1's `init_langsmith()`), no conditional logic is needed inside this file — the graceful degradation lives entirely in `core/observability.py`. Pass `metadata={"task_query": task_query}` (or similar) into the trace via `langsmith.run_helpers.get_current_run_tree()` or the decorator's metadata kwarg if the installed `langsmith` version supports it, so each trace is searchable by query text in the LangSmith UI.

**Depends on**: Step 1.

---

### Step 6: Instrument every LangGraph node in `ManagerAgent`

**File to create/edit**: [agents/manager_agent.py](agents/manager_agent.py)

**Why**: `_node_hydrate`, `_node_brain_route`, `_node_dispatch`, `_node_evaluate`, `_node_persist`, `_node_finalise`, `_node_abort` are the seven LangGraph nodes that make up the orchestration loop. Without per-node spans, a LangSmith trace would show only the single root span from Step 5 with no visibility into which node consumed the most time or where a routing decision was made.

**What to add**: Apply `@traceable(name="<node_name>", run_type="chain")` to each of the seven `_node_*` methods individually (they will automatically nest under the `ManagerAgent.run` root trace from Step 5 because LangSmith propagates trace context via `contextvars` across the call stack). For `_node_dispatch`, additionally apply `core/error_handler.py`'s `with_error_reporting(component="manager_agent.dispatch")` decorator from Step 2, since `_dispatch()` (the function `_node_dispatch` calls into) already absorbs specialist-agent exceptions without re-raising — a Sentry breadcrumb here is the only way to surface a swallowed agent failure to Sentry, since it will never propagate to `api/main.py`'s exception handlers.

**Depends on**: Steps 1, 2, 5.

---

### Step 7: Add a breadcrumb before `_dispatch()`'s agent invocation

**File to create/edit**: [agents/manager_agent.py](agents/manager_agent.py)

**Why**: `_dispatch()` (around line 600) is the single chokepoint where the Manager hands control to one of the three specialist agents via `await agent.run(state)`. It already records dispatch metadata into `ManagerMemory` (`self._memory.log_dispatch(...)`) and `state["orchestrator_logs"]`, but none of that is visible in Sentry if the dispatched agent throws — `_dispatch()`'s own `except Exception as exc:` block (around line 656) currently only logs locally and appends to `agent_execution_history`; it does not re-raise, so Sentry will never see it unless this step adds an explicit capture.

**What to add**: Immediately before `state = await agent.run(state)` (line 633), add a Sentry breadcrumb recording `agent_class` and `action`. Inside the existing `except Exception as exc:` block (line 656), add `sentry_sdk.capture_exception(exc)` tagged with `agent_name=agent_class` — guarded by checking `init_sentry()`'s returned state (or simply calling a Sentry SDK function, which itself no-ops when `sentry_sdk.init()` was never called). This is the only way to capture a specialist-agent failure given the deliberate "absorb and continue" design of `_dispatch()` — do not change that control flow; only add the capture call alongside it.

**Depends on**: Steps 1, 2.

---

### Step 8: Guard the unguarded `persist_long_term()` calls

**File to create/edit**: [agents/manager_agent.py](agents/manager_agent.py)

**Why**: **Warning** — `_node_finalise()` and `_node_abort()` both call `self._memory.persist_long_term()` with no `try/except` around it. `ManagerMemory.persist_long_term()` delegates to `LongTermMemory.persist()` in [memory/manager_memory.py](memory/manager_memory.py), which **re-raises** on Supabase failure (unlike `LongTermMemory.load()`, which swallows). This means a transient Supabase outage during the *finalise* step of an otherwise-successful analysis can crash the entire LangGraph run after all the expensive agent work has already completed — this is a real bug independent of observability, surfaced here because adding error reporting is the natural moment to also report it.

**What to add**: Wrap each `persist_long_term()` call site (inside `_node_finalise` and `_node_abort`) in `try/except Exception as exc:`, call `sentry_sdk.capture_exception(exc)` tagged `component="memory.persist_long_term"`, log a warning, and continue (do not re-raise) so a memory-persistence failure does not erase an otherwise-valid `final_report` from the API response. This guidance only adds the missing handling around an existing call — it does not change `LongTermMemory.persist()` itself (which legitimately should still raise for direct callers that want strict semantics).

**Depends on**: Steps 1, 2.

---

### Step 9: Root trace + node spans for `ResearchAgent`

**File to create/edit**: [agents/research_agent.py](agents/research_agent.py)

**Why**: `ResearchAgent.run()` is dispatched as a child of `ManagerAgent`'s `dispatch` node (Step 6/7) but has its own internal LangGraph loop (`brain → executor → checker`, with a conditional loop-back edge) that deserves its own nested trace so a slow research pass can be diagnosed down to which of the up-to-`max_loops` iterations was expensive.

**What to add**: Apply `@traceable(name="ResearchAgent.run", run_type="chain")` to `run()` (it will nest under the `ManagerAgent.run`/`dispatch` span from Step 5–6 via LangSmith's contextvar propagation across the `await agent.run(state)` call). Apply `@traceable(name="research.brain", run_type="llm")` to `_brain_node`, `@traceable(name="research.executor", run_type="tool")` to `_executor_node`, and `@traceable(name="research.checker", run_type="llm")` to `_checker_node`. Tag the `_brain_node` and `_checker_node` traces with `run_type="llm"` since they each make exactly one `self._llm.messages.create(...)` call — this lets LangSmith aggregate token/cost metrics correctly.

**Depends on**: Steps 1, 5.

---

### Step 10: Breadcrumbs and child spans around `ResearchAgent`'s MCP tool calls

**File to create/edit**: [agents/research_agent.py](agents/research_agent.py)

**Why**: `_executor_node` (around line 352) is where the actual MCP tool calls happen — it opens a `stdio_client`/`ClientSession`, then loops over the Brain's parsed actions calling `session.call_tool(tool_name, arguments)` (line 424). Each of those calls is a separate external subprocess round-trip and each already has a local `try/except` (line 423–442) that appends an error string into `context_chunks` rather than raising — exactly the kind of failure that needs a breadcrumb since it is otherwise invisible outside this function's own log output.

**What to add**: Before the `await session.call_tool(...)` call (line 424), add a Sentry breadcrumb with `category="mcp.research"`, recording `tool_name` and `arguments`. Inside the existing `except Exception as tool_exc:` block (line 437), add `sentry_sdk.capture_exception(tool_exc)` tagged `tool=tool_name` — do not change the existing behavior of appending `err_msg` to `new_chunks` and continuing the loop. Also wrap the outer `except Exception as mcp_exc:` (line 444, the MCP *connection* failure case, distinct from a tool-call failure) with the same capture pattern tagged `component="mcp_connection"`, since a connection failure to the `research_server.py` subprocess is a more severe condition worth distinguishing in Sentry from an individual tool error. Additionally, wrap each `session.call_tool(...)` invocation with a LangSmith child span (`run_type="tool"`, `name=tool_name`) so the per-tool latency shows up nested under the `research.executor` span from Step 9.

**Depends on**: Steps 1, 2, 9.

---

### Step 11: Root trace + node spans for `FinancialAnalystAgent`

**File to create/edit**: [agents/financial_agent.py](agents/financial_agent.py)

**Why**: Unlike `ResearchAgent`, `FinancialAnalystAgent` does not use an internal LangGraph — its loop is a plain `while state["loop_counter"] < max_loops:` inside `run()` (around line 866) calling `_brain()`, `_execute_data_extraction()`, `_execute_ratio_computation()`, `_check_data_quality()` in sequence each iteration. The same per-call tracing goal applies, just without LangGraph's node abstraction to hang the decorator on — apply directly to the methods.

**What to add**: Apply `@traceable(name="FinancialAnalystAgent.run", run_type="chain")` to `run()`. Apply `@traceable(name="financial.brain", run_type="llm")` to `_brain()` (makes one `self._llm.messages.create(...)` call). Apply `@traceable(name="financial.extract", run_type="tool")` to `_execute_data_extraction()` and `@traceable(name="financial.compute", run_type="tool")` to `_execute_ratio_computation()`. Apply `@traceable(name="financial.checker", run_type="llm")` to `_check_data_quality()` (it calls `self._llm.messages.create(...)` with the `_CHECKER_SYSTEM_PROMPT`).

**Depends on**: Steps 1, 5.

---

### Step 12: Breadcrumbs around `FinancialAnalystAgent`'s MCP tool calls

**File to create/edit**: [agents/financial_agent.py](agents/financial_agent.py)

**Why**: `_execute_data_extraction()` makes three sequential MCP calls — `tool_get_financial_ratios`, `tool_get_revenue_growth`, `tool_get_xbrl_financials` — each in its own `try/except` block (lines ~284–328) that appends to a local `errors: list[str]` instead of raising. `_execute_ratio_computation()` has an internal `_call()` helper (line 378) used for `tool_calc_pe`, `tool_calc_roe`, `tool_calc_net_margin`, `tool_calc_debt_to_equity`, `tool_calc_cagr`, `tool_calc_composite_score`, which already catches and returns `{"error": str(exc)}` rather than raising.

**What to add**: In `_execute_data_extraction()`, add a breadcrumb (`category="mcp.financial"`, data=`{"tool": <name>, "ticker": ticker}`) immediately before each of the three `await session.call_tool(...)` calls, and a `sentry_sdk.capture_exception(exc)` call inside each of the three corresponding `except Exception as exc:` blocks, tagged with the tool name — without changing the existing `errors.append(...)` behavior. In `_execute_ratio_computation()`, add the same breadcrumb-before / capture-inside pattern to the internal `_call()` helper's body (this single helper is reused by all six ratio-calculation tool calls, so instrumenting it once covers all six call sites). Wrap each MCP call with a LangSmith child span (`run_type="tool"`) nested under the `financial.extract`/`financial.compute` spans from Step 11.

**Depends on**: Steps 1, 2, 11.

---

### Step 13: Root trace + node spans for `SentimentAgent`

**File to create/edit**: [agents/sentiment_agent.py](agents/sentiment_agent.py)

**Why**: `SentimentAgent` uses the same plain-`while`-loop pattern as `FinancialAnalystAgent` (no internal LangGraph), with a two-pass Brain (`_brain_plan` before the pipeline, `_brain_analyze` after it) wrapping a fixed four-step Executor pipeline in `_execute_sentiment_pipeline()`.

**What to add**: Apply `@traceable(name="SentimentAgent.run", run_type="chain")` to `run()`. Apply `@traceable(name="sentiment.brain_plan", run_type="llm")` to `_brain_plan()` and `@traceable(name="sentiment.brain_analyze", run_type="llm")` to `_brain_analyze()` (both make exactly one `self._llm.messages.create(...)` call each). Apply `@traceable(name="sentiment.executor", run_type="tool")` to `_execute_sentiment_pipeline()`.

**Depends on**: Steps 1, 5.

---

### Step 14: Breadcrumbs around `SentimentAgent`'s four-step MCP pipeline

**File to create/edit**: [agents/sentiment_agent.py](agents/sentiment_agent.py)

**Why**: `_execute_sentiment_pipeline()` has its own internal `_call()` helper (line 542) used for all four MCP tool calls — `retrieve_social_data`, `analyze_finbert`, `score_vader`, `calculate_fear_greed`. It already appends to `state["extraction_errors"]` on failure (both the `payload.get("error")` case and the `except Exception` case) without raising, mirroring `FinancialAnalystAgent`'s pattern.

**What to add**: Add a breadcrumb (`category="mcp.sentiment"`, data=`{"tool": tool}`) before the `await session.call_tool(tool, arguments=args)` line inside the internal `_call()` helper, and `sentry_sdk.capture_exception(exc)` inside its `except Exception as exc:` block, tagged with the tool name — a single instrumentation point covers all four pipeline steps since they all funnel through this one helper. Wrap each call with a LangSmith child span (`run_type="tool"`) nested under the `sentiment.executor` span from Step 13.

**Depends on**: Steps 1, 2, 13.

---

### Step 15: Instrument the raw `Server`-pattern MCP servers (`research_server.py`, `sentiment_server.py`)

**File to create/edit**: [tools/research_tools/research_server.py](tools/research_tools/research_server.py) and [tools/sentiment_tools/sentiment_server.py](tools/sentiment_tools/sentiment_server.py)

**Why**: These two servers run as separate subprocesses (launched by `StdioServerParameters` in the agent files) — instrumentation added to the agent-side `call_tool()` invocation (Steps 10, 14) only sees the *client* side of the call. Instrumenting the server-side `@app.call_tool()` dispatcher itself catches failures inside the actual tool implementation (e.g. a Tavily API timeout inside `tavily_search()`) at the point closest to the root cause, with full argument context, and independent of whether the MCP transport layer successfully relays the error back to the client.

**What to add**: In both files, the single `@app.call_tool()` async function (`research_server.py` line 293, `sentiment_server.py` line 401) already wraps its entire body in `try/except Exception as exc:` and returns an `isError=True` `CallToolResult` rather than raising — this existing pattern must be preserved. Add `sentry_sdk.capture_exception(exc)` inside that `except` block (tagged `tool=name`, `server="research-agent-mcp"` or `server="sentiment-agent-mcp"` respectively) before constructing the existing error `CallToolResult`. Because these scripts run as standalone subprocesses (not imported as a module from `api/main.py`'s process), each must call `core/observability.py`'s `init_sentry()` itself near the top of `main()` (after `load_dotenv()`) — Sentry initialization is process-scoped, so the parent process's `init_sentry()` call in `api/main.py` does not cover these subprocesses.

**Depends on**: Step 1 (each subprocess independently imports and calls `core/observability.py`'s functions).

---

### Step 16: Instrument the `FastMCP`-pattern server (`financial_server.py`)

**File to create/edit**: [tools/financial_tools/financial_server.py](tools/financial_tools/financial_server.py)

**Why**: **Warning** — this file has no central `call_tool()` dispatcher to instrument once; FastMCP's `@mcp.tool()` decorator registers each of the 16 functions (`tool_get_price_history` through `tool_calc_composite_score`) individually, and FastMCP itself owns the request-routing internals. There is no single chokepoint inside this file analogous to `research_server.py`'s `call_tool()`.

**What to add**: Two viable approaches, in order of preference: (1) **Preferred** — add the Sentry capture inside the underlying implementation files instead of this server file: [tools/financial_tools/yahoo_finance.py](tools/financial_tools/yahoo_finance.py), [tools/financial_tools/sec_edgar.py](tools/financial_tools/sec_edgar.py), and [tools/financial_tools/financial_ratio_calculator.py](tools/financial_tools/financial_ratio_calculator.py) — wherever those modules' functions (`get_price_history`, `get_financial_ratios`, `get_revenue_growth`, `get_peer_comparison`, `get_cik`, `list_filings`, `get_filing_text`, `get_xbrl_financials`, and the dozen `calc_*` functions) already catch exceptions and return an `{"error": ...}` dict, add `sentry_sdk.capture_exception()` at that catch site. This keeps the instrumentation co-located with the actual error-prone I/O (Yahoo Finance HTTP calls, SEC EDGAR HTTP calls) rather than relying on FastMCP internals. (2) **Alternative if per-function annotation is wanted instead** — wrap each of the 16 `@mcp.tool()`-decorated functions individually with `core/error_handler.py`'s sync error-reporting decorator from Step 2 (applied between `@mcp.tool()` and `def tool_get_...`), which requires touching 16 call sites in this one file but needs no changes to the three implementation files. Either approach must call `init_sentry()` once near the top of this file (it already runs as its own subprocess via `mcp.run(transport="stdio")` at the bottom) — do not assume the parent API process's initialization applies here, same caveat as Step 15.

**Depends on**: Step 1.

---

### Step 17: Instrument `rag/hybrid_rag.py`'s three RAG functions

**File to create/edit**: [rag/hybrid_rag.py](rag/hybrid_rag.py)

**Why**: `rag_vector_search()`, `rag_graph_traverse()`, and `rag_hybrid_query()` are called both as MCP tools (via `research_server.py`'s dispatcher, already covered in Step 15) and could be called directly by future code — instrumenting at this layer ensures coverage regardless of caller. Each already has partial error handling: `rag_vector_search()` catches the Supabase RPC call (`except Exception as exc:` around line 102) and returns `{"results": [], "error": str(exc)}`; `rag_graph_traverse()` catches the Neo4j traversal (around line 184) similarly; `rag_hybrid_query()` has no try/except of its own since it just calls the other two via `asyncio.gather()`.

**What to add**: Inside `rag_vector_search()`'s existing `except Exception as exc:` block (around line 102, the Supabase RPC error), add `sentry_sdk.capture_exception(exc)` tagged `component="rag.vector_search"`. Inside `rag_graph_traverse()`'s existing `except Exception as exc:` block (around line 184, the Neo4j traversal error), add the same pattern tagged `component="rag.graph_traverse"`. Also add a breadcrumb before each external call: before the `await asyncio.to_thread(lambda: _sb.rpc(...))` call in `rag_vector_search()` (data: `query`, `ticker_filter`), and before `await session.run(cypher, ...)` in `rag_graph_traverse()` (data: `entity`, `max_hops`). Apply `@traceable(run_type="retriever")` to all three functions so they appear as spans nested under whichever parent agent span invoked them (Step 9's `research.executor`, since these are the implementations behind `research_server.py`'s `rag_*` tools).

**Depends on**: Steps 1, 9.

---

### Step 18: Instrument `rag/ingestion.py`'s ETL pipeline

**File to create/edit**: [rag/ingestion.py](rag/ingestion.py)

**Why**: `run_ingestion_pipeline()` already has partial error handling — each of its five stages (Load, Process, Embed+Vector, Graph) is wrapped in its own `try/except Exception as exc:` that logs and `return`s early (Stage 1: line ~71, Stage 2: line ~86, Stage 3+4: line ~111, Stage 5: line ~130). This is a *missing piece* situation: the logging exists, but nothing reports these failures to Sentry, and this is the one pipeline in the codebase that runs standalone (via `if __name__ == "__main__":`) rather than inside the FastAPI process, so its failures are otherwise only visible in whatever process/cron captures its stdout.

**What to add**: Inside each of the four existing `except Exception as exc:` blocks, add `sentry_sdk.capture_exception(exc)` tagged with the stage name (`component="ingestion.load"`, `"ingestion.process"`, `"ingestion.vector"`, `"ingestion.graph"` respectively) — do not change the existing early-`return` short-circuit behavior, since that is deliberate sequential-stage gating (a later stage should not run on a previous stage's empty/failed output). Add a breadcrumb at the start of each stage (`category="ingestion"`, data=`{"tickers": tickers, "stage": "<name>"}`). Since this script is its own process entry point, call `init_sentry()` near the top of `run_ingestion_pipeline()` itself (after the existing `load_dotenv()` call at module level), same reasoning as Steps 15–16.

**Depends on**: Step 1.

---

### Step 19: Instrument remaining RAG support functions called from ingestion and retrieval

**File to create/edit**: [rag/loader.py](rag/loader.py), [rag/processor.py](rag/processor.py), [rag/vector_store.py](rag/vector_store.py), [rag/graph_store.py](rag/graph_store.py), [rag/retriever.py](rag/retriever.py)

**Why**: These five files implement the methods `rag/ingestion.py` (Step 18) and `tools/sentiment_tools/sentiment_server.py` (Step 15, via `AlphaRetriever.retrieve_raw()`) call into — `AlphaLoader.load()`, `AlphaProcessor.process()`, `AlphaVectorStore.upsert()`/`.hybrid_search()`, `AlphaGraphStore.extract_batch()`/`.upsert_batch()`, `AlphaRetriever.retrieve_raw()`. Read each file before editing to determine which of these methods currently have no exception handling at all versus partial handling — this guide cannot enumerate exact line numbers here since these files were not part of the original research scope summarized in Section 1, so **read the file first to confirm whether a given method already has a try/except before deciding whether to add one or just add a capture call inside an existing one.**

**What to add**: For each public method on these classes that performs network I/O (Supabase calls, Neo4j calls, HTTP fetches in `AlphaLoader`, model inference in `AlphaProcessor`/embedding), add a breadcrumb before the I/O call and a Sentry capture in the nearest enclosing `except` block — following the same before/inside pattern established in Steps 10, 12, 14, 17. Do not introduce new `try/except` blocks around code that currently has none unless the method is called from a context where an uncaught exception would crash a request (e.g. `AlphaRetriever.retrieve_raw()` is called synchronously inside `sentiment_server.py`'s `_retrieve_social_data()` via `asyncio.to_thread()`, which is itself inside that server's already-present top-level `try/except` in `call_tool()` — so an uncaught exception here is already caught one level up at the MCP boundary, and only needs a capture added at that existing boundary, not a new one in this file).

**Depends on**: Steps 1, 15, 18.

---

### Step 20: Add `.env.example`

**File to create/edit**: `.env.example` (new file at the project root)

**Why**: **Warning** — no `.env.example` file currently exists anywhere in this repository, even though `rag/ingestion.py` and `memory/manager_memory.py` both call `load_dotenv(dotenv_path=...)` pointing at a `.env` file that is assumed to exist, and `api/config.py`'s `validate_settings()` checks for `ANTHROPIC_API_KEY`/`SUPABASE_URL`/`SUPABASE_KEY` at startup. Every other environment variable referenced via raw `os.environ[...]`/`os.environ.get(...)` across the codebase (`TAVILY_API_KEY`, `NEWSAPI_KEY`, `NEO4J_URI`, `NEO4J_PASSWORD`, `NEO4J_USER`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_SERVICE_KEY`, `FEAR_GREED_FINBERT_WEIGHT`, `FEAR_GREED_VADER_WEIGHT`, `DEFAULT_USER_ID`) has zero documentation anywhere, and several of those raise uncaught `KeyError` only at first tool invocation deep inside an MCP subprocess (e.g. `tools/research_tools/research_server.py` and `rag/hybrid_rag.py` use `os.environ["NEO4J_URI"]`/`os.environ["NEO4J_PASSWORD"]` with no fallback) rather than failing fast at startup — this is a pre-existing gap independent of observability, surfaced here since adding `SENTRY_DSN`/`LANGSMITH_API_KEY` is the proximate reason to finally create this file.

**What to add**: Create `.env.example` listing every env var referenced across the codebase with placeholder values and a one-line comment per group: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY` (note: some files fall back to `SUPABASE_SERVICE_KEY` instead — flag this inconsistency in a comment rather than silently picking one), `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `TAVILY_API_KEY`, `NEWSAPI_KEY`, `FEAR_GREED_FINBERT_WEIGHT`, `FEAR_GREED_VADER_WEIGHT`, `DEFAULT_USER_ID`, `MAX_ROUTING_LOOPS`, `APP_ENV`, `LOG_LEVEL`, `ALLOWED_ORIGINS`, plus the two new ones this guide introduces: `SENTRY_DSN` (comment: "optional — error tracking disabled if unset") and `LANGSMITH_API_KEY` + `LANGSMITH_PROJECT` (comment: "optional — LLM tracing disabled if unset; project defaults to alpha-agent-node"). Cross-reference [api/config.py](api/config.py) to confirm the exact set of variables `Settings`/`validate_settings()` reads, since that file is the closest thing to an authoritative settings schema currently in the repo.

**Depends on**: Step 1 (to know the exact two new var names to document) — otherwise independent of all other steps and can be done at any time.

---

### Step 21: Add `sentry-sdk` to requirements

**File to create/edit**: [requirements.txt](requirements.txt)

**Why**: `langsmith` is already present (line 10) but `sentry-sdk` is entirely absent — every step above that calls `sentry_sdk.capture_exception()`/`sentry_sdk.init()` will `ImportError` without this addition.

**What to add**: Add `sentry-sdk>=2.0.0` under the existing `# Monitoring, Tracing & Observability` section header (line 7–10), alongside `langsmith`, so both observability dependencies are grouped together as the section header already implies they should be.

**Depends on**: Nothing — can be done at any point, but should be done before Step 1's `core/observability.py` is exercised at runtime.

---

## 4. Verification Checklist

- [ ] `core/observability.py` exists; `init_sentry()` returns `False` and logs a clear message when `SENTRY_DSN` is unset, without raising or importing `sentry_sdk` in a way that fails if the package is missing.
- [ ] `init_langsmith()` returns `False` and logs a clear message when `LANGSMITH_API_KEY` is unset.
- [ ] Starting `uvicorn api.main:app` with no `SENTRY_DSN`/`LANGSMITH_API_KEY` set in the environment boots successfully and serves `/health` — confirms graceful degradation end-to-end.
- [ ] Starting the same app with both env vars set shows "Sentry enabled: True, LangSmith enabled: True" (or equivalent) in the startup logs.
- [ ] A request to `POST /api/v1/analyze` with a deliberately invalid ticker or a temporarily-disabled MCP server (e.g. rename `tools/financial_tools/financial_server.py` temporarily) produces a Sentry event with a breadcrumb trail showing the request path leading up to the failure, tagged with the correct `agent_name`/`tool` fields, and the response still returns a structured `AgentError` JSON body (not a raw 500) per Step 4.
- [ ] The same failing request produces a `trace_id` in the JSON response body that matches a tag visible in the corresponding Sentry event (per Step 3).
- [ ] A successful `/api/v1/analyze` request produces exactly one root trace in the LangSmith UI named `ManagerAgent.run`, containing nested child spans for `hydrate`, `brain_route`, `dispatch`, `evaluate`, `persist`, and either `finalise` or `abort`.
- [ ] Within the `dispatch` span (or as siblings reachable from it), the LangSmith trace shows nested spans for whichever of `ResearchAgent.run`, `FinancialAnalystAgent.run`, `SentimentAgent.run` were dispatched for that query, each further nested with their brain/executor/checker (or brain_plan/executor/brain_analyze) sub-spans.
- [ ] Within `ResearchAgent.run`'s trace, individual MCP tool calls (`tavily_search`, `rag_hybrid_query`, etc.) appear as distinct child spans with correct latency.
- [ ] Temporarily unsetting `NEO4J_PASSWORD` and triggering a `rag_graph_traverse` call produces a Sentry-captured exception tagged `component="rag.graph_traverse"` rather than an uncaught `KeyError` crashing the MCP subprocess silently.
- [ ] Running `python -m rag.ingestion` (or `rag/ingestion.py`'s `__main__` block) with a deliberately broken Supabase key produces a Sentry event tagged `component="ingestion.vector"` while still logging the existing "Embed/vector stage failed" message — confirms Step 18 added capture without altering existing control flow.
- [ ] `tools/financial_tools/financial_server.py` is independently confirmed to call `init_sentry()` in its own process (e.g. via a print/log statement at startup) since it runs as a separate subprocess from the main API process — per the Step 15/16 warning about per-process initialization.
- [ ] `.env.example` exists at the project root and lists every variable currently read via `os.environ`/`os.getenv` anywhere in `agents/`, `api/`, `rag/`, `tools/`, `memory/`, including the new `SENTRY_DSN` and `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT`.
- [ ] `requirements.txt` contains both `langsmith` and `sentry-sdk`.
- [ ] No existing function signature in `agents/`, `api/`, `rag/`, `tools/`, or `memory/` was changed — confirm via `git diff --stat` that only `core/observability.py`, `core/error_handler.py`, `.env.example`, and `requirements.txt` are new files, and that all other diffs are additive (decorators, breadcrumbs, capture calls) rather than signature or control-flow changes, with the sole intentional exceptions being Step 4 (`HTTPException` → `AgentError` in `api/routes/analyze.py`) and Step 8 (adding a `try/except` around the previously-unguarded `persist_long_term()` calls in `agents/manager_agent.py`), both of which are explicitly flagged above as deviations from "instrumentation only."
