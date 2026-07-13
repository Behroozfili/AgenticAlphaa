# TECHNICAL DOCUMENTATION TICKET — AgenticAlpha

> **Type:** Read-only codebase audit & function inventory
> **Repository:** `D:/AgenticAlpha`
> **Generated:** 2026-07-13
> **Scope:** All first-party source (`agents/`, `api/`, `core/`, `memory/`, `rag/`, `tools/`, `scheduler/`, `evaluation/`). Vendored/virtual-env code (`evaluation/.venv-eval/**`), caches (`.ruff_cache`, `.pytest_cache`), and generated artifacts are excluded.
> **Mode:** READ-ONLY — no source code was modified during this analysis.

> **Note on method:** This ticket was produced by static reading of the source plus the repo's own `SYSTEM_ARCHITECTURE.md`. Purpose lines for functions in files that were read in full are verified against code; purpose lines for functions whose bodies were only sampled (signature + docstring + architecture doc) are marked **[Inferred from code]**. Where a function's exact type is not annotated, the return is described rather than typed.

---

## 1. PROJECT OVERVIEW

**AgenticAlpha** ("Alpha-Agent Node") is a multi-agent **financial-intelligence platform**. Given a natural-language query (e.g. *"Is NVIDIA a good buy for Q1 2025?"*), it orchestrates a swarm of specialist AI agents to produce a structured investment-analysis report that fuses live market data, SEC regulatory filings, and real-time news/social sentiment.

### Primary technologies / frameworks

| Layer | Technology |
|-------|-----------|
| Language | Python ≥ 3.12 (async/await throughout) |
| Web API | FastAPI + Uvicorn, Pydantic v2 / pydantic-settings |
| LLM provider | Anthropic Claude (`claude-haiku-4-5` default) via `anthropic` SDK (sync + async) |
| Agent orchestration | LangGraph `StateGraph` (compiled graphs) |
| Tool transport | MCP (Model Context Protocol) over stdio JSON-RPC — `mcp` / `FastMCP` |
| Vector DB | Supabase Postgres + pgvector (`alpha_hybrid_search` RPC) |
| Knowledge graph | Neo4j (Bolt) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) via sentence-transformers |
| Sentiment (deep) | `ProsusAI/finbert` (HuggingFace Transformers + Torch) |
| Sentiment (lexical) | NLTK VADER |
| Market data | yfinance, SEC EDGAR (XBRL + full-text), Tavily, NewsAPI, Reddit RSS (feedparser) |
| Persistence | Supabase Postgres (`analyses`, `long_term_memory` tables) |
| Observability | Sentry SDK (errors), LangSmith (`@traceable` LLM traces) |
| Eval | RAGAS, custom period-consistency validator |
| Delivery | Docker, docker-compose, GitHub Actions (`deploy.yml`) |

### High-level architecture pattern

A **hierarchical multi-agent orchestrator** with **contract-based shared state**:

- **API layer** (FastAPI) — thin HTTP surface; owns singleton resources on `app.state`.
- **Orchestrator** — `ManagerAgent`, a 7-node LangGraph state machine that routes work, evaluates output quality, and synthesizes the final report (3 distinct "Brain" LLM passes).
- **Specialist agents** — `ResearchAgent`, `FinancialAnalystAgent`, `SentimentAgent`, each with its own internal Brain→Executor(→Checker) loop and a private MCP tool server.
- **Tool servers** — three MCP stdio subprocesses (research / financial / sentiment) wrapping the data/compute tools.
- **RAG subsystem** — ingestion ETL + hybrid (vector + graph) retrieval.
- **Memory** — two-tier (`ShortTermMemory` in-process + `LongTermMemory` Supabase-backed) behind a `ManagerMemory` facade.
- **Cross-cutting** — observability, an in-process progress pub/sub bus, and a custom exception hierarchy.

Design principles (per `SYSTEM_ARCHITECTURE.md`): **separation of concerns** (each agent owns exactly one shared-state field), **graceful degradation** (every external dependency has a fallback path), and **cognitive memory** (cross-session learning).

---

## 2. DIRECTORY STRUCTURE

```
AgenticAlpha/
├── main.py                      # Trivial "hello" stub (project scaffold placeholder)
├── api/                         # FastAPI application layer
│   ├── main.py                  # App factory, lifespan singletons, CORS, exception handlers, health/index routes
│   ├── config.py                # Pydantic BaseSettings singleton + startup validation
│   ├── dependencies.py          # FastAPI Depends factories (per-request ManagerMemory, user_id)
│   ├── routes/
│   │   ├── analyze.py           # POST /api/v1/analyze — request/response models, Supabase persistence
│   │   └── progress.py          # GET /api/v1/analyze/stream/{session_id} — SSE progress stream
│   └── core/
│       └── exceptions.py        # AlphaAgentError hierarchy → structured JSON error responses
├── agents/                      # Agent implementations
│   ├── state.py                 # ALL state TypedDicts (single source of truth)
│   ├── manager_agent.py         # ManagerAgent: 7-node LangGraph orchestrator, 3 Brain passes
│   ├── research_agent.py        # ResearchAgent: 3-node LangGraph (brain→executor→checker)
│   ├── financial_agent.py       # FinancialAnalystAgent: 3-layer imperative loop
│   └── sentiment_agent.py       # SentimentAgent: 2-tier Brain→Executor
├── memory/
│   └── manager_memory.py        # ShortTermMemory + LongTermMemory + ManagerMemory facade
├── core/                        # Cross-cutting infrastructure
│   ├── observability.py         # Sentry + LangSmith bootstrap
│   ├── progress_bus.py          # In-process pub/sub for SSE progress events
│   └── error_handler.py         # Reusable error-reporting decorator/context managers
├── rag/                         # RAG pipeline & knowledge infrastructure
│   ├── ingestion.py             # 5-stage ETL orchestrator
│   ├── loader.py                # AlphaLoader: yfinance news + Reddit RSS → RawDocument
│   ├── processor.py             # AlphaProcessor: chunking + SHA-256 dedup
│   ├── embedding_manager.py     # AlphaEmbedder singleton (bge-small, CUDA/MPS/CPU)
│   ├── vector_store.py          # AlphaVectorStore: Supabase pgvector upsert + hybrid search
│   ├── graph_store.py           # AlphaGraphStore: Claude entity extraction + Neo4j MERGE
│   ├── hybrid_rag.py            # rag_vector_search / rag_graph_traverse / rag_hybrid_query (+RRF)
│   ├── retriever.py             # AlphaRetriever: 5-stage retrieval pipeline
│   ├── evaluation.py            # AlphaEvaluator: LLM-judge RAG metrics
│   └── seed.py                  # CLI entry point for ingestion
├── tools/
│   ├── research_tools/
│   │   ├── research_server.py   # MCP server (Server API): 8 research tools
│   │   ├── tavily_search.py     # Tavily web/news search client
│   │   ├── news_search.py       # NewsAPI client with keyword restructuring
│   │   ├── sec_edgar.py         # EDGAR full-text search + filing section extraction
│   │   ├── comprehensive_analysis.py  # tavily + sec_edgar concurrent combo
│   │   └── context_synthesizer.py     # Claude compression of research chunks
│   ├── financial_tools/
│   │   ├── financial_server.py  # FastMCP server: ~30 financial tools
│   │   ├── yahoo_finance.py     # yfinance wrapper (prices, ratios, revenue, peers)
│   │   ├── sec_edgar.py         # EDGAR XBRL / CIK / filing text
│   │   └── financial_ratio_calculator.py  # Pure ratio math + DCF + composite score
│   └── sentiment_tools/
│       ├── sentiment_server.py  # MCP server: 4 sentiment tools (lazy singletons)
│       ├── finbert_analyzer.py  # FinBERT deep sentiment
│       ├── vader_scorer.py      # VADER lexical sentiment
│       └── fear_greed_calculator.py  # Weighted FinBERT+VADER fusion
├── scheduler/
│   └── daily_refresh.py         # Scheduled ingestion entry point
├── evaluation/
│   ├── run_ragas.py             # RAGAS evaluation harness
│   └── validate_period_consistency.py  # Period-provenance QA validator
├── frontend/
│   └── alpha-agent-app.html     # Single-page client (served at GET /)
├── tests/                       # pytest suite (unit + integration), asyncio_mode=auto
├── SYSTEM_ARCHITECTURE.md       # Detailed architecture reference
├── Dockerfile / docker-compose.yml / .github/workflows/deploy.yml
├── pyproject.toml / requirements.txt / uv.lock / pytest.ini
```

**Organization strategy:** classic **layered + domain-partitioned** structure. Horizontal layers (`api` → `agents` → `tools`/`rag` → `core`) are separated from vertical domains (research / financial / sentiment), with each domain replicated consistently across `agents/`, `tools/*_tools/`, and `tests/*_test/`. State contracts are centralized in `agents/state.py`; cross-cutting concerns live in `core/`.

---

## 3. FUNCTION INVENTORY

Organized by file path, then alphabetically by function name within each file. Nested/local helper functions are listed under their enclosing file. Pydantic/dataclass/TypedDict classes are listed with their notable methods.

### `agents/state.py`

Contains no executable functions — it defines the platform's state contracts as TypedDicts.

| Class | Kind | Purpose |
|-------|------|---------|
| `SharedManagerState` | `TypedDict(total=True)` | Public cross-agent contract. Fields: `task_query`, `manager_directives`, `aggregated_research_context`, `financial_metrics_summary`, `sentiment_analysis_summary`, `agent_execution_history`, `orchestrator_logs`, `final_report`. Each field owned by exactly one agent. |
| `ResearchAgentState` | `TypedDict` | Private Research loop state: `messages` / `context_chunks` (both `Annotated[list, operator.add]`), `loop_counter`, `validation_feedback`, `is_complete`, `shared_manager_ref`. |
| `FinancialAgentState` | `TypedDict` | Private Financial loop state: `messages`, `raw_numerical_data`, `calculated_ratios`, `loop_counter`, `validation_feedback`, `is_complete`, `shared_manager_ref`. |
| `SentimentAgentState` | `TypedDict` | Private Sentiment loop state: `messages`, `retrieved_chunks`, `sources_metadata`, `finbert_result`, `vader_result`, `fear_greed_result`, `brain_reasoning`, `loop_counter`, `extraction_errors`, `shared_manager_ref`. |
| `EvaluationSnapshot` | `TypedDict(total=False)` | Lightweight eval snapshot stored in graph state: `step`, `passed`, `score`, `next_action`, `issues`. |
| `ManagerGraphState` | `TypedDict` | LangGraph-internal orchestration state: `shared_state`, `loop_counter`, `last_action`, `last_agent_key`, `evaluation_passed`, `last_evaluation`, `ticker`, `session_id`. |

---

### `agents/manager_agent.py`

**Module-level helpers**

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `_extract_chunk_text` | `chunk: str \| dict` | `str` | Safely pull plain text from a research context chunk (dict `"text"` key or str). | — | `_brain_evaluate`, `_brain_finalise`, `_format_*_chunk_fairly` callers |
| `_feedback_to_snapshot` | `fb: EvaluationFeedback` | `EvaluationSnapshot` | Convert an `EvaluationFeedback` dataclass into a plain graph-state snapshot dict. | `EvaluationSnapshot` | `_node_evaluate` |
| `_format_filing_chunk_fairly` | `chunk_text: str`, `per_section_budget=3000` | `str` | Parse an embedded `sec_edgar_filing` JSON blob and give each requested filing section its own char budget so large sections don't starve smaller ones. Flat-truncation fallback. | `json` | `_brain_finalise` |
| `_format_news_chunk_fairly` | `chunk_text: str`, `max_articles=5`, `per_article_budget=220` | `str` | Parse an embedded `news_search` JSON blob and surface title+snippet for up to N articles instead of a flat truncation that would show only a fragment of the first. | `json` | `_brain_finalise` |
| `_infer_result_keys` | `agent_key: str`, `state: SharedManagerState` | `list[str]` | Determine which shared-state key an agent populated (research→`aggregated_research_context`, etc.). | — | `_dispatch` |

**Class `ManagerAgent`** — central orchestrator; owns the compiled LangGraph and 3 Brain LLM passes.

| Method | Params | Returns | Purpose | Dependencies | Called By |
|--------|--------|---------|---------|--------------|-----------|
| `__init__` | `research_agent`, `financial_agent`, `sentiment_agent`, `memory`, `model=_DEFAULT_MODEL`, `max_routing_loops=8`, `llm_client=None` | `None` | Store injected agents/memory/LLM (`AsyncAnthropic`), build the agent registry, compile the graph. | `anthropic.AsyncAnthropic`, `_build_graph` | `api/main.py` lifespan |
| `_hydrate_state` | `task_query`, `manager_directives` | `SharedManagerState` | Initialize a fresh shared state with all fields empty. | — | `run` |
| `_recall` | `ticker: str \| None` | `dict` | Pull short+long-term context from `ManagerMemory`. | `ManagerMemory.recall` | `_node_brain_route`, `_node_evaluate` |
| `_brain_route` (async) | `state`, `memory_recall`, `loop_counter` | `dict` (routing decision) | **Brain pass 1**: ask Claude for the next of 8 routing actions; deterministic fallback chain on API failure. | LLM, Sentry, `ManagerMemory` | `_node_brain_route` |
| `_brain_evaluate` (async) | `agent_name`, `state`, `memory_ctx` | `EvaluationFeedback` | **Brain pass 2**: grade the last agent's output (score 0-100, passed, next_action). Fallback verdict `passed=True, score=50` on failure. | LLM, Sentry | `_node_evaluate` |
| `_brain_finalise` (async) | `state` | `str` | **Brain pass 3**: synthesize the final investment report; prefers a synthesized research summary, applies fair per-section/article budgeting, carries period tags. | LLM, `_format_*_chunk_fairly`, `_extract_chunk_text` | `_node_finalise` |
| `_dispatch` (async) | `action`, `state` | `SharedManagerState` | Route to the correct specialist agent, await `agent.run()`, record timing/outcome, publish progress. | agents, `ManagerMemory`, `progress_bus`, Sentry | `_node_dispatch` |
| `_persist` | `agent_key`, `state`, `evaluation` | `None` | Write heuristics + ticker insights into memory after a step (no `add_evaluation`). | `ManagerMemory` | `_node_persist` |
| `_node_hydrate` (async) | `g: ManagerGraphState` | `dict` | NODE: apply saved `search_depth` preference, cache ticker. | `ManagerMemory`, `progress_bus` | graph |
| `_node_brain_route` (async) | `g` | `dict` | NODE: increment loop counter, recall memory, call `_brain_route`, merge directive updates. | `_recall`, `_brain_route` | graph |
| `_node_dispatch` (async) | `g` | `dict` | NODE: map action→agent, call `_dispatch`. | `_dispatch` | graph |
| `_node_evaluate` (async) | `g` | `dict` | NODE: call `_brain_evaluate`, write eval to memory exactly once, snapshot into graph state. | `_brain_evaluate`, `ManagerMemory` | graph |
| `_node_persist` (async) | `g` | `dict` | NODE: persist memory; compute `updated_action`, only forcing `rerun_*` when evaluator retargets the same failed agent (DC-5). | `_persist` | graph |
| `_node_finalise` (async) | `g` | `dict` | NODE: build report via `_brain_finalise`, run non-blocking period-consistency QA (`check_narration_vs_period`), persist long-term memory. | `_brain_finalise`, `check_narration_vs_period`, `ManagerMemory` | graph |
| `_node_abort` (async) | `g` | `dict` | NODE: guardrail/error exit; log abort and persist long-term memory. | `ManagerMemory` | graph |
| `_should_route` | `g` | `str` | Conditional edge after `brain_route`: `dispatch` / `finalise` / `abort`. | — | graph |
| `_should_continue_after_persist` | `g` | `str` | Conditional edge after `persist`: `brain_route` (passed) / `dispatch` (rerun) / `abort`. | — | graph |
| `_build_graph` | — | compiled `StateGraph` | Wire the 7 nodes + 2 conditional edges and compile. | LangGraph | `__init__` |
| `run` (async) | `task_query`, `manager_directives=None`, `user_preferences=None`, `client_session_id=None` | `SharedManagerState` | **Primary entry point**: seed session/memory, invoke compiled graph with recursion limit `(max_loops+2)*4`, return populated state. | graph, `ManagerMemory`, `progress_bus` | `api/routes/analyze.py` |

---

### `agents/research_agent.py`

**Module-level helper**

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `_ensure_ticker_argument` | `tool_name`, `arguments`, `directives` | `dict` | Defensive net: inject the ticker/entity arg (under the tool-specific key) if the Brain omitted it for a ticker-scoped tool. | `_TICKER_ARG_BY_TOOL` | `_executor_node` |

**Class `ResearchAgent`** — 3-node LangGraph (brain→executor→checker); sync `anthropic.Anthropic` wrapped in `asyncio.to_thread`.

| Method | Params | Returns | Purpose | Dependencies | Called By |
|--------|--------|---------|---------|--------------|-----------|
| `__init__` | `anthropic_api_key=None`, `model="claude-haiku-4-5"`, `max_loops=3`, `mcp_server_params=None`, `llm_client=None` | `None` | Init LLM client + MCP params, compile internal graph. | anthropic, `_build_graph` | `api/main.py` |
| `run` (async) | `shared_state` | `SharedManagerState` | Public gateway: hydrate private state, run graph, synthesize a summary via `synthesize_research_context`, append chunks to `aggregated_research_context`. | graph, `synthesize_research_context` | `ManagerAgent._dispatch` |
| `_brain_node` (async) | `state` | `dict` | Planner: Claude produces a JSON action plan (1-3 MCP tool calls). | LLM, `progress_bus` | graph |
| `_executor_node` (async) | `state` | `dict` | Open stdio MCP session to `research_server.py`, run each planned tool, append labelled chunks; per-tool try/except. | `mcp`, `_ensure_ticker_argument`, `_format_tool_result`, `_parse_plan`, Sentry | graph |
| `_checker_node` (async) | `state` | `dict` | Critic: Claude audits chunk completeness with per-chunk budgeting; sets `is_complete` + feedback. | LLM, `progress_bus` | graph |
| `_should_continue` | `state` | `Literal["brain","__end__"]` | Conditional edge: end on max_loops or completeness, else loop to brain. | — | graph |
| `_build_graph` | — | compiled `StateGraph` | Wire brain→executor→checker with conditional edge. | LangGraph | `__init__` |
| `_parse_plan` | `plan_text` | `list[dict]` | Strip markdown fences, extract `actions` array; returns `[]` on parse failure. | `json` | `_executor_node` |
| `_format_tool_result` (static) | `tool_name`, `arguments`, `raw_text` | `str` | Wrap a raw MCP result in a labelled, LLM-friendly chunk string. | — | `_executor_node` |
| `_budget_chunk` (local, in `_checker_node`) | `c: str` | `str` | Truncate a single chunk to its per-chunk budget with a marker. | — | `_checker_node` |

---

### `agents/financial_agent.py`

**Module-level helpers**

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `_infer_peers` | `sector`, `industry`, `ticker` | `list[str]` | Look up static peer tickers by industry (from `_INDUSTRY_PEERS`), excluding the ticker itself, max 5. | `_INDUSTRY_PEERS` | `_execute_data_extraction` **[Inferred from code]** |
| `_sanitize_nans` | `value: Any` | `Any` | Recursively replace `NaN`/`inf` floats with `None` so results are JSON-safe. **[Inferred from code]** | `math` | ratio/summary builders |

**Class `FinancialAnalystAgent`** — 3-layer imperative loop (Brain→Executors→Checker); long-lived MCP session to `financial_server.py`.

| Method | Params | Returns | Purpose | Dependencies | Called By |
|--------|--------|---------|---------|--------------|-----------|
| `__init__` | (agent config; `llm_client`, `model`, `max_loops`, MCP params) | `None` | Init LLM + MCP config. **[Inferred from code]** | anthropic, mcp | `api/main.py` |
| `_extract_ticker` | `task_query`, `directives` | `str \| None` | Resolve ticker: directives → regex scan of query → None. | `re` | `run` |
| `_execute_data_extraction` (async) | `state`, (session/priority) | `dict` | LAYER 1a-c: call Yahoo ratios / revenue growth / XBRL financials + peers via MCP; collect `raw_numerical_data`. **[Inferred from code]** | MCP tools, `_infer_peers`, Sentry | `run` |
| `_execute_ratio_computation` (async) | `state`, (session) | `dict` | LAYER 1d: call ratio calculator tools (PE/ROE/margins/CAGR/DCF/composite), attaching period provenance. **[Inferred from code]** | MCP calc tools, `_call`, `_is_valid_number` | `run` |
| `_call` (local, async, in `_execute_ratio_computation`) | `tool`, `args`, `period=None` | `dict` | Invoke one MCP calc tool and tag result with its reporting `_period`. | MCP session | `_execute_ratio_computation` |
| `_is_valid_number` (local, in `_execute_ratio_computation`) | `x: Any` | `bool` | Guard: true only for finite real numbers. | `math` | `_execute_ratio_computation` |
| `_check_data_quality` (async) | `state` | `dict` | LAYER 2 Checker: Claude audits 7 quality criteria → `{is_complete, feedback}`. **[Inferred from code]** | LLM | `run` |
| `_brain` (async) | `state` | `dict` | LAYER 3 Brain: Claude plans the next iteration / priority tools. **[Inferred from code]** | LLM | `run` |
| `run` (async) | `shared_state` | `SharedManagerState` | Entry point: drive Brain→Executors→Checker loop (≤ max_loops), commit `financial_metrics_summary`. | MCP, all layers, `progress_bus` | `ManagerAgent._dispatch` |

---

### `agents/sentiment_agent.py`

**Class `SentimentAgent`** — 2-tier Brain→Executor (no separate Checker); MCP session to `sentiment_server.py`.

| Method | Params | Returns | Purpose | Dependencies | Called By |
|--------|--------|---------|---------|--------------|-----------|
| `__init__` | (config incl. `server_script_path`, `llm_client`, `model`, `max_loops`) | `None` | Init LLM + MCP server path. **[Inferred from code]** | anthropic, mcp | `api/main.py` |
| `_extract_ticker` | `task_query`, `directives`, (financial summary) | `str \| None` | Resolve ticker: directives → `financial_metrics_summary["ticker"]` → regex scan. | `re` | `run` |
| `_brain_plan` (async) | `state` | `dict` | Brain pass 1: Claude produces `{retrieval_query, ticker, days_back, reasoning}`. **[Inferred from code]** | LLM | `run` |
| `_brain_analyze` (async) | `state` | `str` | Brain pass 2: Claude synthesizes narrative verdict from FinBERT/VADER/Fear-Greed signals. **[Inferred from code]** | LLM | `run` |
| `_execute_sentiment_pipeline` (async) | `state`, (session) | `dict` | Executor: sequential MCP calls retrieve→finbert→vader→fear_greed. **[Inferred from code]** | MCP tools, `_call` | `run` |
| `_call` (local, async) | `tool`, `args` | `dict` | Invoke one sentiment MCP tool, capturing errors. | MCP session | `_execute_sentiment_pipeline` |
| `run` (async) | `shared_state` | `SharedManagerState` | Entry point: plan→execute (retry once if 0 chunks, max_loops=2)→analyze, commit `sentiment_analysis_summary`. | MCP, brain passes, `progress_bus` | `ManagerAgent._dispatch` |

---

### `api/main.py`

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `lifespan` (async ctx mgr) | `app: FastAPI` | async generator | Startup: validate settings, init Sentry/LangSmith, connect Supabase, instantiate all agents + memory, compile `ManagerAgent` onto `app.state`. | `validate_settings`, `create_client`, agents, `ManagerMemory`, observability | FastAPI |
| `request_timing_middleware` (async) | `request`, `call_next` | `Response` | Log method/path/status/duration for every request. | — | FastAPI |
| `alpha_agent_exception_handler` (async) | `request`, `exc: AlphaAgentError` | `JSONResponse` | Convert any `AlphaAgentError` to structured JSON (`trace_id`, `code`), capture to Sentry. | `exc.to_dict`, Sentry | FastAPI |
| `unhandled_exception_handler` (async) | `request`, `exc: Exception` | `JSONResponse` | Catch-all 500; hides internal detail in production, emits `trace_id`. | Sentry | FastAPI |
| `health` (async) | — | `dict` | Liveness probe → `{"status":"ok"}`. | — | GET `/health` |
| `readiness` (async) | `request` | `JSONResponse` | Readiness probe: light Supabase query → 200/503. | Supabase | GET `/readiness` |
| `serve_index` (async) | — | `FileResponse` | Serve the SPA (`frontend/alpha-agent-app.html`) at `/`. | — | GET `/` |

---

### `api/config.py`

| Function / Class | Params | Returns | Purpose | Dependencies | Called By |
|------------------|--------|---------|---------|--------------|-----------|
| `Settings` (class) | — | pydantic settings | Env-driven config: Anthropic, Supabase, `APP_ENV`, `MAX_ROUTING_LOOPS`, `LOG_LEVEL`, `ALLOWED_ORIGINS`, etc. `SUPABASE_KEY` aliases `SUPABASE_SERVICE_ROLE_KEY`. | pydantic-settings | — |
| `get_settings` | — | `Settings` | Lazy, `lru_cache`-d singleton factory. | `Settings` | app-wide, `validate_settings` |
| `validate_settings` | — | `None` | Assert required env vars present; raise `ConfigurationError` if missing. | `get_settings`, `ConfigurationError` | `lifespan` |

---

### `api/dependencies.py`

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `get_user_id` | `request` | `str` | Resolve user id from `X-User-Id` header or `DEFAULT_USER_ID`. | `settings` | `get_manager_memory` |
| `get_manager_memory` | `request`, `user_id=Depends(get_user_id)` | `ManagerMemory` | Per-request, user-scoped `ManagerMemory` using the shared Supabase client. | `ManagerMemory` | `analyze` route |

---

### `api/routes/analyze.py`

| Function / Class | Params | Returns | Purpose | Dependencies | Called By |
|------------------|--------|---------|---------|--------------|-----------|
| `AnalyzeRequest` (class) | pydantic fields | model | Request body: `query`, `ticker`, `user_id`, `search_depth`, `days_back`, `include_sentiment`, `session_id`. | pydantic | route |
| `AnalyzeRequest.ticker_uppercase` (validator) | `cls`, `v` | `str \| None` | Force ticker uppercase; validate 1-5 alpha chars. | — | pydantic |
| `AnalyzeRequest.valid_search_depth` (validator) | `cls`, `v` | `str` | Restrict `search_depth` to `basic`/`advanced`. | — | pydantic |
| `AnalyzeResponse` (class) | pydantic fields | model | Structured response mapped from `SharedManagerState`. | pydantic | route |
| `_persist_analysis` (async) | `request`, `analysis_id`, `user_id`, `req`, `result`, `status`, `error_message`, `created_at`, `completed_at`, `duration_s` | `None` | Fire-and-forget insert into Supabase `analyses`; never raises. | Supabase | `analyze` |
| `analyze` (async) | `req`, `request`, `memory=Depends(get_manager_memory)` | `AnalyzeResponse` | POST `/api/v1/analyze`: build directives, run `ManagerAgent.run`, persist, return report; raises `AgentError` on failure. | `ManagerAgent`, `_persist_analysis`, Sentry | FastAPI |

---

### `api/routes/progress.py`

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `stream_progress` (async) | `session_id`, `request` | `StreamingResponse` | GET `/api/v1/analyze/stream/{session_id}`: SSE stream of progress events; heartbeats; closes on `pipeline_complete`/`pipeline_error`. | `progress_bus.get_queue/close_session` | FastAPI |
| `event_generator` (local async gen) | — | async generator | Yield SSE frames from the session queue with 15 s heartbeat timeout. | `get_queue` | `stream_progress` |

---

### `api/core/exceptions.py`

| Class | Base | HTTP | Purpose |
|-------|------|------|---------|
| `AlphaAgentError` | `Exception` | 500 | Base error with `message`, `code`, `detail`, auto `trace_id`; `to_dict()` serializer. Methods: `__init__`, `to_dict`. |
| `ValidationError` | `AlphaAgentError` | 400 | Request payload failed validation. |
| `AgentError` | `AlphaAgentError` | 500 | `ManagerAgent.run()` raised. |
| `AgentTimeoutError` | `AlphaAgentError` | 504 | Run exceeded timeout. |
| `MemoryError` | `AlphaAgentError` | 500 | Memory load/persist failed. |
| `ExternalServiceError` | `AlphaAgentError` | 503 | Downstream service unreachable. |
| `ConfigurationError` | `AlphaAgentError` | 500 | Missing/invalid startup config. |

`AlphaAgentError.__init__(message, detail=None)` → `None`; `AlphaAgentError.to_dict()` → `dict` (`error`, `message`, `detail`, `trace_id`). Called by the global exception handler in `api/main.py`.

---

### `core/observability.py`

| Function | Params | Returns | Purpose | Called By |
|----------|--------|---------|---------|-----------|
| `init_sentry` | `app_env=None` | `bool` | Idempotently init Sentry if `SENTRY_DSN` set (`traces_sample_rate=1.0`). | `lifespan`, MCP server `__main__` |
| `sentry_enabled` | — | `bool` | Whether Sentry initialised in this process. | everywhere (guard before capture) |
| `init_langsmith` | — | `bool` | Set LangChain tracing env vars if `LANGSMITH_API_KEY` set. | `lifespan`, MCP `__main__` |
| `langsmith_enabled` | — | `bool` | Whether LangSmith tracing enabled. | subprocess servers (guard before importing `@traceable`) |

---

### `core/progress_bus.py`

In-process pub/sub (one `asyncio.Queue` per session, max 300, dropped on overflow).

| Function | Params | Returns | Purpose | Called By |
|----------|--------|---------|---------|-----------|
| `_get_queue` | `session_id` | `asyncio.Queue` | Get-or-create the session queue. | `publish`, `subscribe`, `get_queue` |
| `publish` | `session_id`, `event_type`, `agent=None`, `message=""`, `detail=None` | `None` | Fire-and-forget event push; no-op if no session; drops on full queue. | all agents/nodes |
| `subscribe` (async gen) | `session_id` | `AsyncIterator[dict]` | Async-iterate events (do not wrap in `wait_for`). | (available; SSE route uses `get_queue`) |
| `get_queue` | `session_id` | `asyncio.Queue` | Public queue accessor for timeout-controlled consumers. | `stream_progress` |
| `session_from_shared` | `shared_manager_ref` | `str \| None` | Extract `_progress_session_id` from directives. | specialist agents |
| `close_session` | `session_id` | `None` | Drop a finished session's queue. | `stream_progress` |

---

### `core/error_handler.py`

Reusable error-reporting utilities (decorator + context managers) wrapping Sentry breadcrumbs/capture.

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_add_breadcrumb` | `component`, `function_name`, `extra=None` | `None` | Add a Sentry breadcrumb. **[Inferred from code]** |
| `_capture` | `component`, `exc` | `None` | Tag-and-capture an exception to Sentry. **[Inferred from code]** |
| `_safe_extra` | `kwargs` | `dict` | Produce a serialization-safe subset of kwargs for breadcrumb data. **[Inferred from code]** |
| `with_error_reporting` | `component` | `Callable` (decorator) | Decorator that wraps sync/async functions with breadcrumb + capture on error. **[Inferred from code]** |
| `decorator` / `async_wrapper` / `sync_wrapper` (nested) | `fn` / `*args,**kwargs` | wrapped callable | Inner decorator machinery selecting async vs sync wrapping. **[Inferred from code]** |
| `_sync_context` | `component` | `Iterator[None]` | `contextmanager` for a sync error-reporting block. **[Inferred from code]** |
| `_async_context` | `component` | `AsyncIterator[None]` | Async context-manager variant. **[Inferred from code]** |

> Note: a module-level `_executor_node(self, state)` also appears at line 15 — **[Inferred from code]** an illustrative/example snippet in this module rather than a live agent node.

---

### `memory/manager_memory.py`

**Dataclasses**

| Class | Purpose |
|-------|---------|
| `AgentExecutionRecord` | Dispatch record: agent name, timestamp, outcome, duration, result keys, error. |
| `EvaluationFeedback` | Eval verdict: `step`, `timestamp`, `passed`, `score`, `issues`, `next_action`, `raw_verdict`. |

**Class `ShortTermMemory`** (ephemeral, session-scoped)

| Method | Params | Returns | Purpose |
|--------|--------|---------|---------|
| `__init__` | `max_messages=50` | `None` | Init FIFO message buffer + logs. |
| `reset` | `session_id`, `task_query` | `None` | Clear all state for a new session. |
| `add_message` | `role`, `content` | `None` | Append an LLM message (FIFO trim). |
| `get_messages` | — | `list[dict]` | Return the message log. |
| `log_dispatch` | `agent_name`, `directives` | `AgentExecutionRecord` | Record an agent dispatch. |
| `get_agent_log` | — | `list[AgentExecutionRecord]` | Full dispatch history. |
| `get_last_dispatch` | — | `AgentExecutionRecord \| None` | Most recent dispatch. |
| `agents_run` | — | `list[str]` | Names of agents dispatched this session. |
| `add_evaluation` | `feedback` | `None` | Append an evaluation. |
| `get_last_evaluation` | — | `EvaluationFeedback \| None` | Latest evaluation. |
| `get_evaluations` | — | `list[EvaluationFeedback]` | All evaluations. |
| `to_context_dict` | — | `dict` | Serialize short-term context for LLM prompts. |

**Class `LongTermMemory`** (cross-session, Supabase-backed)

| Method | Params | Returns | Purpose |
|--------|--------|---------|---------|
| `__init__` | `user_id`, `supabase_client`, ... | `None` | Side-effect-free init (no load). |
| `create` (classmethod) | `user_id`, `supabase_client` | `LongTermMemory` | Factory: construct + `load()` in one call. |
| `store_heuristic` / `get_heuristic` / `get_all_heuristics` | `key`,`value` / `key`,`default` / — | `None` / `Any` / `dict` | Operational heuristics store (FIFO cap 100). |
| `store_ticker_insight` / `get_ticker_insight` | `ticker`,`insight` / `ticker` | `None` / `dict` | Per-ticker insight store (FIFO cap 200). |
| `store_preference` / `get_preference` / `get_all_preferences` | `key`,`value` / `key`,`default` / — | `None` / `Any` / `dict` | User preferences store. |
| `recall` | `ticker=None` | `dict` | Bundle heuristics + ticker insight for a ticker. |
| `load` | — | `None` | SELECT from `long_term_memory` by `user_id`; init empty on first use. |
| `persist` | — | `None` | UPSERT the three stores keyed on `user_id`. |

**Class `ManagerMemory`** (facade over both layers)

| Method | Params | Returns | Purpose |
|--------|--------|---------|---------|
| `__init__` | `user_id`, `supabase_client`, ... | `None` | Compose `ShortTermMemory` + `LongTermMemory` (loads long-term). |
| `new_session` | `session_id`, `task_query` | `None` | Reset short-term for a new run. |
| `add_message` / `get_messages` | ... | delegate | Proxy to short-term. |
| `log_dispatch` / `add_evaluation` / `get_last_evaluation` / `agents_run` | ... | delegate | Proxy to short-term. |
| `store_heuristic` / `get_heuristic` / `store_ticker_insight` / `get_ticker_insight` / `store_preference` / `get_preference` | ... | delegate | Proxy to long-term. |
| `persist_long_term` | — | `None` | Proxy to `LongTermMemory.persist`. |
| `recall` | `ticker=None` | `dict` | Merge short-term context dict + long-term recall. |

Callers: `ManagerAgent` (all memory ops), `api/dependencies.py` (`get_manager_memory`), `api/main.py` (system memory).

---

### `rag/ingestion.py`

| Function | Params | Returns | Purpose | Dependencies | Called By |
|----------|--------|---------|---------|--------------|-----------|
| `run_ingestion_pipeline` (async) | `tickers`, `skip_graph=False` | pipeline result | 5-stage ETL: Load → Process → Embed → Vector upsert → (optional) Graph upsert; stage 5 gated on vector success. | `AlphaLoader`, `AlphaProcessor`, `AlphaEmbedder`, `AlphaVectorStore`, `AlphaGraphStore` | `seed.py`, `scheduler/daily_refresh.py` |

### `rag/seed.py`

| Function | Params | Returns | Purpose | Called By |
|----------|--------|---------|---------|-----------|
| `main` (async) | `tickers` | `None` | CLI entry point that runs the ingestion pipeline. **[Inferred from code]** | `__main__` |

### `rag/loader.py`

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `RawDocument` (dataclass) | — | — | Loaded doc: title, content, url, source_type, ticker, `published_at` (UTC ISO-8601). |
| `_mentions_ticker` | `text`, `ticker`, `company_name=None` | `bool` | Whether text references the ticker/company. **[Inferred from code]** |
| `_get_company_name` | `ticker` | `str \| None` | Resolve company name for a ticker. **[Inferred from code]** |
| `_to_utc_iso8601` | `value` | `str` | Normalize a timestamp to UTC ISO-8601. **[Inferred from code]** |
| `_safe_timestamp` | `value`, `fallback_label="unknown"` | `str` | Timestamp normalization with fallback. **[Inferred from code]** |
| `AlphaLoader.__init__` | `max_news_per_ticker=20`, `max_rss_per_feed=30` | `None` | Configure source caps. |
| `AlphaLoader.load` | `tickers` | `list[RawDocument]` | Fetch yfinance news + Reddit RSS with per-source circuit breakers. |
| `AlphaLoader._fetch_yfinance` | `tickers` | `list[RawDocument]` | Fetch yfinance news across tickers. **[Inferred from code]** |
| `AlphaLoader._yfinance_news` | `ticker` | `list[RawDocument]` | Fetch/normalize one ticker's news. **[Inferred from code]** |
| `AlphaLoader._fetch_reddit_rss` | `tickers` | `list[RawDocument]` | Fetch Reddit RSS feeds. **[Inferred from code]** |
| `AlphaLoader._parse_rss` | (feed/args) | `list[RawDocument]` | Parse an RSS feed into documents. **[Inferred from code]** |

### `rag/processor.py`

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `ProcessedChunk` (dataclass) | — | — | A chunk ready for embedding (text + metadata). |
| `ProcessorMetrics` (dataclass) | — | — | Chunking metrics; `report()` → `dict[str,int]`. |
| `_sha256` | `text` | `str` | Content hash for dedup. |
| `_url_hash` | `url` | `str` | URL hash for per-URL dedup tracking. |
| `AlphaProcessor.__init__` | (chunk size/overlap) | `None` | Configure `RecursiveCharacterTextSplitter`. |
| `AlphaProcessor.process` | `docs` | `list[ProcessedChunk]` | Chunk + dedup documents. |
| `AlphaProcessor._process_doc` | `doc` | `list[ProcessedChunk]` | Chunk one document. |
| `AlphaProcessor._add_to_seen` | `u_hash`, `c_hash` | `None` | Track seen hashes for dedup. |

### `rag/embedding_manager.py`

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `_select_device` | — | `str` | Choose CUDA → MPS → CPU. |
| `reset_embedder` | — | `None` | Reset the singleton (frees model). |
| `clean_embedder` (nested) | — | `None` | Cleanup callback for reset. |
| `get_embedder` | (config) | `AlphaEmbedder` | Return the process-wide singleton. |
| `AlphaEmbedder.__init__` | (model/config) | `None` | Configure model/device/batch. |
| `AlphaEmbedder._load_model` | — | `None` | Lazy-load `BAAI/bge-small-en-v1.5`. |
| `AlphaEmbedder.embed_chunks` | `chunks` | `list[dict]` | Embed `ProcessedChunk`s → records (embedding + metadata + text). |
| `AlphaEmbedder.embed_query` | `query` | `list[float]` | Embed a single query string (384-dim). |
| `AlphaEmbedder._encode_batch` | `texts` | `np.ndarray` | Batched, L2-normalized encoding (GPU-OOM → CPU fallback). |

### `rag/vector_store.py`

| Method | Params | Returns | Purpose |
|--------|--------|---------|---------|
| `AlphaVectorStore.__init__` | (supabase config) | `None` | Init pgvector-backed store. |
| `AlphaVectorStore.upsert` | `records` | `int` | UPSERT embedding records keyed by chunk id; returns count. |
| `AlphaVectorStore.hybrid_search` | (query, filters, top_k...) | results | Call Supabase `alpha_hybrid_search` RPC (vector + FTS, RRF). |
| `AlphaVectorStore._to_row` (static) | `record` | `dict` | Map a record to a DB row. |

### `rag/graph_store.py`

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `Entity` / `Relation` / `GraphDocument` (dataclasses) | — | — | Extracted graph schema (entities, relations, per-document bundle). |
| `AlphaGraphStore.__init__` | (neo4j config) | `None` | Configure driver params (no connect yet). |
| `AlphaGraphStore.connect` | — | `None` | Open Bolt driver; warn + `_driver=None` if unavailable (graceful degradation). |
| `AlphaGraphStore.extract_batch` (async) | `docs` | `list[GraphDocument]` | Claude entity/relation extraction over raw docs. |
| `AlphaGraphStore.upsert_batch` | `graph_docs` | `dict[str,int]` | Idempotent Neo4j `MERGE` of nodes/edges (weight averaging); no-op counts if disconnected. |
| `AlphaGraphStore.close` | — | `None` | Close the driver. |
| `AlphaGraphStore._extract_one` (async) | `doc`/`text`,`ticker` | graph doc | Extract one doc via Claude with `"{"` prefill (forced JSON). |
| `AlphaGraphStore._extract_json_block` (static) | `raw` | `str \| None` | Balanced-brace JSON scanner (repairs fences/preamble/trailing text). |
| `AlphaGraphStore._parse_graph_doc` | (raw/json) | `GraphDocument` | Validate & coerce entities/relations (invalid types → `Company`/`RELATED_TO`). |
| `AlphaGraphStore._merge_entity` (static) | `tx`, `entity` | `None` | Cypher `MERGE` node with `ON CREATE`/`ON MATCH`. |
| `AlphaGraphStore._merge_relation` (static) | `tx`, `rel`, `source_url` | `None` | Cypher `MERGE` edge with running-average weight. |
| `AlphaGraphStore._ensure_constraints` | — | `None` | Create uniqueness constraints + ticker index for the 6 entity types. |

### `rag/hybrid_rag.py`

| Function | Params | Returns | Purpose | Called By |
|----------|--------|---------|---------|-----------|
| `_get_neo4j` | — | driver/None | Lazy Neo4j driver accessor (degrades if unset). | graph functions |
| `rag_vector_search` (async) | `query`, `top_k`, `ticker_filter`, `days_back` | dict | Supabase `alpha_hybrid_search` RPC (vector+FTS, RRF k=60). | `research_server`, `rag_hybrid_query` |
| `rag_graph_traverse` (async) | `entity`, `relation_types`, `max_hops`, `limit` | dict | Cypher variable-length traversal, hops hard-capped at 3. | `research_server`, `rag_hybrid_query` |
| `rag_hybrid_query` (async) | `query`, `entity`, `top_k`, `max_hops`, `fusion` | dict | Parallel vector + graph (`asyncio.gather`), fuse via RRF/weighted/union; degrades to vector-only. | `research_server` |
| `_embed` | `text` | `list[float]` | Embed a query (via `AlphaEmbedder`). | vector search |
| `_key` | `item` | `str` | Stable text-hash key for fusion dedup. | `_rrf`, `_weighted` |
| `_extract_ticker_from_query` | `query` | `str \| None` | Best-effort ALL-CAPS ticker extraction fallback. | hybrid query |
| `_rrf` | `a`, `b`, `k=60` | `list[dict]` | Reciprocal-rank-fusion of two result lists. | `rag_hybrid_query` |
| `_weighted` | `vec`, `graph`, `w=0.7` | `list[dict]` | Weighted blend fusion. | `rag_hybrid_query` |

### `rag/retriever.py`

**Class `AlphaRetriever`** — 5-stage retrieval used by `sentiment_server.py`.

| Method | Params | Returns | Purpose |
|--------|--------|---------|---------|
| `__init__` | (store/embedder config) | `None` | Init retriever backends. |
| `retrieve` | (query, filters) | `str` | Full pipeline → formatted citation string. |
| `retrieve_raw` | (query, filters) | `list[dict]` | Full pipeline → structured dicts (used by sentiment). |
| `_run_pipeline` | (query, filters) | `list[dict]` | Hybrid search → freshness → diversity → budget. |
| `_rerank_by_freshness` | `chunks` | `list[dict]` | `fresh_score = rrf × exp(-hours/72)` (72 h half-life). |
| `_hours_since` (static) | `pub_at`, `now` | `float` | Hours since publication. |
| `_diversity_filter` | `chunks` | `list[dict]` | Max 2/URL, 3/source_type. |
| `_apply_token_budget` | `chunks` | `list[dict]` | Greedy inclusion within ~8000-char budget. |
| `_format_context` (static) | `chunks` | `str` | Citation-formatted output. |

### `rag/evaluation.py`

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `MetricResult` / `EvaluationReport` (dataclasses) | — | — | RAG metric result + report; `overall_score()`, `to_dict()`, `summary()`. |
| `AlphaEvaluator.__init__` | (judge config) | `None` | Init LLM-judge evaluator. |
| `AlphaEvaluator.evaluate` | (query, context, answer, ...) | `EvaluationReport` | Score one sample across metrics. |
| `AlphaEvaluator.batch_evaluate` | (samples) | `list[EvaluationReport]` | Evaluate a batch. |
| `AlphaEvaluator.aggregate_scores` | `reports` | `dict[str,float]` | Aggregate metric means. |
| `AlphaEvaluator._faithfulness` / `_context_precision` / `_context_recall` / `_answer_relevance` | (query/context/answer) | `MetricResult` | Individual LLM-judged metrics. |
| `AlphaEvaluator._call_judge` | `prompt` | `str` | Invoke the judge LLM. |
| `AlphaEvaluator._parse_json` (static) | `text`, `default_score=0.0` | `dict` | Parse judge JSON with fallback. |

---

### `tools/research_tools/`

**`research_server.py`** (MCP `Server`; routes via match/case)

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_normalize_sections` | `raw` | `list[str]` | Normalize filing `sections` arg (string/list/"all"). |
| `list_tools` (async) | — | `ListToolsResult` | Advertise the 8 research tools + schemas. |
| `call_tool` (async) | `name`, `arguments` | `CallToolResult` | Dispatch a tool call (tavily/news/sec/rag/comprehensive). |
| `main` (async) | — | `None` | Run the stdio MCP server. |

**`tavily_search.py`** — `tavily_search(async)` → results. Purpose: Tavily web/news search client. **[Inferred from code]**

**`news_search.py`**

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_extract_terms` | `query` | `list[str]` | Extract keyword terms from a query. |
| `_build_boolean_query` | `terms` | `str` | Build a boolean query string. |
| `_restructure_query` | `query` | `str` | Convert sentence → keyword boolean query. |
| `_count_term_matches` | `article`, `terms` | `list[str]` | Which terms an article matches. |
| `_min_required_matches` | `n_terms` | `int` | Relevance threshold by term count. |
| `news_search` (async) | (query, filters) | dict | NewsAPI search with restructuring + relevance filtering. |

**`sec_edgar.py`** (research variant)

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_sanitize_query` | `query` | `str` | Clean an EDGAR full-text query. |
| `sec_edgar_search` (async) | (query, ticker, form_type) | dict | EDGAR full-text filing search. |
| `sec_edgar_filing` (async) | (ticker, form_type, sections) | dict | Fetch + section-extract a 10-K/10-Q. |
| `_resolve_cik` (async) | `ticker` | `str \| None` | Ticker → CIK. |
| `_get_submissions` (async) | `cik` | `dict \| None` | Fetch EDGAR submissions. |
| `_find_latest` | `submissions`, `form_type` | `dict \| None` | Latest filing of a type. |
| `_fetch_text` (async) | `cik`, `acc_dash`, `acc_nodash` | `str` | Fetch filing HTML/text. |
| `_html_to_text` | `raw` | `str` | Strip HTML → text. |
| `_extract_sections` | `text`, `sections`, `max_chars`, ... | dict | Extract named 10-K/Q sections. |
| `_end_boundary_after` (nested) | `start` | `int` | Section end-boundary finder. |

**`comprehensive_analysis.py`** — `comprehensive_analysis(async)` → dict. Purpose: run `tavily_search` + `sec_edgar_filing` concurrently via `asyncio.gather`. **[Inferred from code]**

**`context_synthesizer.py`** — `synthesize_research_context(chunks, task_query, llm_client, model)` (async) → summary chunk or `None`. Purpose: Claude-compress raw research chunks into one dense summary without discarding the raw chunks; returns `None` on failure (never raises). Called by `ResearchAgent.run`.

---

### `tools/financial_tools/`

**`financial_ratio_calculator.py`** — pure math (each returns a dict `{value, interpretation, ...}`).

| Function | Params | Returns |
|----------|--------|---------|
| `_safe_div` | `numerator`, `denominator`, ... | `float \| None` |
| `_label` | `value`, `thresholds`, ... | `str` |
| `price_to_earnings` | `price`, `eps` | `dict` |
| `price_to_book` | `price`, `book_value_per_share` | `dict` |
| `ev_to_ebitda` | `enterprise_value`, `ebitda` | `dict` |
| `peg_ratio` | `pe`, `earnings_growth_rate_pct` | `dict` |
| `gross_margin` | `revenue`, `cogs` | `dict` |
| `operating_margin` | `operating_income`, `revenue` | `dict` |
| `net_margin` | `net_income`, `revenue` | `dict` |
| `return_on_equity` | `net_income`, `shareholders_equity` | `dict` |
| `return_on_assets` | `net_income`, `total_assets` | `dict` |
| `current_ratio` | `current_assets`, `current_liabilities` | `dict` |
| `quick_ratio` | `cash`, `short_term_investments`, ... | `dict` |
| `debt_to_equity` | `total_debt`, `shareholders_equity` | `dict` |
| `interest_coverage` | `ebit`, `interest_expense` | `dict` |
| `asset_turnover` | `revenue`, `avg_total_assets` | `dict` |
| `cagr` | `start_value`, `end_value`, `years` | `dict` |
| `compute_revenue_cagr_from_growth` | `annual_revenue` | `dict` |
| `composite_financial_score` | (all ratios) | `dict` (weighted 0-100 + grade A-F) |
| `_add` (nested) | `name`, `raw`, `min_val`, `max_val`, `weight`, ... | — (accumulate sub-score) |
| `discounted_cash_flow` | (fcf, growth, discount, ...) | `dict` |
| `dcf_scenario_range` | (bear/base/bull inputs) | `dict` |
| `_run` (nested in `dcf_scenario_range`) | `growth_pct` | `dict` |
| `dcf_monte_carlo` | (simulation params) | `dict` (P10/P50/P90) |
| `_percentile` (nested) | `sorted_vals`, `p` | `float` |

Purpose: interpretation-labelled ratio math + DCF variants + weighted composite health score. **[Inferred from code / arch doc].** Called by `financial_server.py` tool wrappers.

**`yahoo_finance.py`**

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_safe_float` | `value` | `float \| None` | Coerce to float or None. |
| `_safe_get` | `info`, `key`, `default=None` | `Any` | Safe dict access. |
| `get_price_history` | `ticker`, `period="1y"` | `dict` | OHLCV price history. |
| `get_financial_ratios` | `ticker` | `dict` | yfinance `.info` ratios/quote payload. |
| `_fiscal_quarter_label` | `period_end_year`, `period_end_month`, `fye_month` | `str` | Fiscal-quarter label from period end. |
| `get_revenue_growth` | `ticker` | `dict` | Annual/quarterly revenue history + growth. |
| `get_peer_comparison` | `ticker`, `peers=None` | `dict` | Relative valuation vs peers. |

**`sec_edgar.py`** (financial variant)

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_get` | `url`, `*`, `host`, `as_json=True` | `Any` | HTTP GET with EDGAR host/UA + retry handling. |
| `_pad_cik` | `cik` | `str` | Zero-pad CIK to 10 digits. |
| `_load_ticker_map` | — | `dict` | Load `company_tickers.json` map. |
| `_strip_html` | `raw` | `str` | HTML → text. |
| `get_cik` | `ticker` | `dict` | Resolve ticker → CIK. |
| `list_filings` | `ticker`, `form_type="10-K"`, `limit=5` | `dict` | List recent filings. |
| `get_filing_text` | `accession_number`, `cik` | `dict` | Fetch a filing's text. |
| `_extract_annual` | `facts`, `tags` | `list[dict]` | Extract annual XBRL facts for tags. |
| `get_xbrl_financials` | `ticker` | `dict` | Balance-sheet XBRL (assets/liabilities). |

**`financial_server.py`** (FastMCP; each `tool_*` is an `@mcp.tool()` wrapper delegating to the modules above with Sentry wrapping)

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_sentry_capture` | `tool_name`, `exc` | `None` | Tag-capture a tool exception. |
| `_sentry_tool` | `tool_name`, `fn`, `*args`, `**kwargs` | result | Run a tool fn with Sentry breadcrumb/capture. |
| `tool_get_price_history` | `ticker`, `period="1y"` | `dict` | → `get_price_history`. |
| `tool_get_financial_ratios` | `ticker` | `dict` | → `get_financial_ratios`. |
| `tool_get_revenue_growth` | `ticker` | `dict` | → `get_revenue_growth`. |
| `tool_get_peer_comparison` | `ticker`, `peers=None` | `dict` | → `get_peer_comparison`. |
| `tool_get_cik` | `ticker` | `dict` | → `get_cik`. |
| `tool_list_filings` | `ticker`, `form_type`, `limit` | `dict` | → `list_filings`. |
| `tool_get_filing_text` | `accession_number`, `cik` | `dict` | → `get_filing_text`. |
| `tool_get_xbrl_financials` | `ticker` | `dict` | → `get_xbrl_financials`. |
| `tool_calc_pe` / `tool_calc_pb` / `tool_calc_ev_ebitda` / `tool_calc_peg` | ratio inputs | `dict` | → valuation ratios. |
| `tool_calc_gross_margin` / `tool_calc_operating_margin` / `tool_calc_net_margin` | ratio inputs | `dict` | → margin ratios. |
| `tool_calc_roe` / `tool_calc_roa` | ratio inputs | `dict` | → return ratios. |
| `tool_calc_current_ratio` / `tool_calc_quick_ratio` | ratio inputs | `dict` | → liquidity ratios. |
| `tool_calc_debt_to_equity` / `tool_calc_interest_coverage` | ratio inputs | `dict` | → leverage ratios. |
| `tool_calc_asset_turnover` | `revenue`, `avg_total_assets` | `dict` | → efficiency ratio. |
| `tool_calc_cagr` / `tool_calc_revenue_cagr_from_growth` | growth inputs | `dict` | → growth metrics. |
| `tool_calc_dcf` / `tool_calc_dcf_scenarios` / `tool_calc_dcf_monte_carlo` | DCF inputs | `dict` | → DCF valuations. |
| `tool_calc_composite_score` | (all ratios) | `dict` | → weighted composite health score. |

---

### `tools/sentiment_tools/`

**`sentiment_server.py`** (MCP `Server`; lazy singletons; `asyncio.to_thread` for sync models)

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_get_retriever` | — | `AlphaRetriever` | Lazy-init the retriever singleton. |
| `_get_finbert` | — | `FinBertSentimentAnalyzer` | Lazy-init FinBERT. |
| `_get_vader` | — | `VaderLexiconScorer` | Lazy-init VADER. |
| `_get_fear_greed` | — | `FearGreedIndexCalculator` | Lazy-init the fusion calculator. |
| `_retrieve_social_data` | (query, ticker, days_back) | dict | Run `AlphaRetriever.retrieve_raw`. |
| `_to_dict` | `obj` | `Any` | Convert dataclass results to plain dicts. |
| `list_tools` (async) | — | `ListToolsResult` | Advertise the 4 sentiment tools. |
| `call_tool` (async) | `name`, `arguments` | `CallToolResult` | Dispatch retrieve/finbert/vader/fear_greed (via `to_thread`). |
| `main` (async) | — | `None` | Run the stdio MCP server. |

**`finbert_analyzer.py`**

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `reset_finbert` / `clean_finbert` | — | `None` | Reset the model singleton. |
| `_select_device` | — | `str` | CUDA/MPS/CPU selection. |
| `ChunkSentiment` / `FinBertResult` (dataclasses) | — | — | Per-chunk + aggregate FinBERT output. |
| `FinBertSentimentAnalyzer.__init__` | (config) | `None` | Configure model. |
| `FinBertSentimentAnalyzer.analyze` | `texts` | `FinBertResult` | Batch inference → mean bullish/bearish/neutral probs. |
| `FinBertSentimentAnalyzer._infer_batch` | `texts` | `torch.Tensor` | Run tokenized batch inference. |
| `FinBertSentimentAnalyzer._load_model` | — | `None` | Lazy-load `ProsusAI/finbert`. |
| `FinBertSentimentAnalyzer._argmax_label` (static) | `bullish`, `bearish`, `neutral` | `str` | Label from max prob. |
| `FinBertSentimentAnalyzer._empty_result` (static) | `skipped=0` | `FinBertResult` | Neutral default for empty input. |

**`vader_scorer.py`**

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `reset_vader` / `clean_vader` | — | `None` | Reset the scorer singleton. |
| `_ensure_vader_lexicon` | — | `None` | Download/ensure the NLTK VADER lexicon. |
| `ChunkVaderScore` / `VaderResult` (dataclasses) | — | — | Per-chunk + aggregate VADER output. |
| `VaderLexiconScorer.__init__` | — | `None` | Init analyzer. |
| `VaderLexiconScorer.score` | `texts` | `VaderResult` | Mean compound + pos/neg/neu means. |
| `VaderLexiconScorer.score_single` | `text` | `ChunkVaderScore` | Score one text. |
| `VaderLexiconScorer._compound_label` (static) | `compound` | `str` | Label from compound score. |
| `VaderLexiconScorer._empty_result` (static) | `skipped=0` | `VaderResult` | Neutral default for empty input. |

**`fear_greed_calculator.py`**

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `FearGreedResult` (dataclass) | — | — | Score, label, per-model scores, confidence, diagnostics. |
| `FearGreedIndexCalculator.__init__` | (weights) | `None` | Configure FinBERT/VADER weights (0.65/0.35). |
| `FearGreedIndexCalculator.calculate` | (finbert, vader) | `FearGreedResult` | Weighted fusion → score `[-1,+1]` + 5-band label. |
| `FearGreedIndexCalculator.calculate_from_dict` | `finbert_dict`, `vader_dict` | `FearGreedResult` | Fusion from raw dict payloads. |
| `FearGreedIndexCalculator._score_to_label` (static) | `score` | `str` | Map score → Extreme Fear…Extreme Greed. |
| `FearGreedIndexCalculator._validate_weights` (static) | `w_finbert`, `w_vader` | `None` | Assert weights valid/sum. |

---

### `scheduler/daily_refresh.py`

Scheduled ingestion entry point that calls `run_ingestion_pipeline([...tickers])`. **[Inferred from code — no top-level `def`/`class` matched; likely a script `__main__` body.]**

### `evaluation/run_ragas.py`

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `_infer_ticker` | `text` | `str \| None` | Guess a ticker from sample text. |
| `load_samples` | `path` | samples | Load eval samples from disk. |
| `main` | — | `None` | Run the RAGAS evaluation harness. |

### `evaluation/validate_period_consistency.py`

| Function / Class | Params | Returns | Purpose |
|------------------|--------|---------|---------|
| `_close` | `a`, `b`, `tol=TOLERANCE` | `bool` | Float near-equality. |
| `PeriodRegistry` (class) | — | — | Registry mapping financial values → reporting periods. |
| `PeriodRegistry.from_raw_numerical_data` (classmethod) | `raw` | `PeriodRegistry` | Build from raw extraction data. |
| `PeriodRegistry.match_period` | `value`, `field_hint=None` | `list[str]` | Periods a value could belong to. |
| `RatioInput` (dataclass) | — | — | A ratio + its inputs for period inference. |
| `infer_periods` | `ratio`, `registry` | `set[str]` | Infer which periods a ratio draws from. |
| `check_section_consistency` | `ratios`, `registry` | `list[dict]` | Detect period mismatches within a section. |
| `_find_value_context` | `report_text`, `value`, `window=CONTEXT_WINDOW` | `list[str]` | Text windows around a cited value. |
| `check_narration_vs_period` | `final_report`, `financial_metrics_summary` | `list[dict]` | **QA:** flag report sentences narrating a metric with the wrong period. Called (non-blocking) by `ManagerAgent._node_finalise`. |
| `_load_state` | `path` | `dict` | Load a saved state file. |
| `build_ratio_inputs_from_state` | `state`, `section_map`, ... | ratio inputs | Build `RatioInput`s from state. |
| `_demo_state` | — | `dict` | Demo state fixture. |
| `main` | — | `None` | CLI demo/validation runner. |

### `main.py` (repo root)

| Function | Params | Returns | Purpose |
|----------|--------|---------|---------|
| `main` | — | `None` | Prints `"Hello from agenticalpha!"` — scaffold placeholder; not the app entry point (that is `uvicorn api.main:app`). |

---

## 4. STRATEGY & DESIGN PATTERNS

**Design patterns observed**

- **Orchestrator / Mediator** — `ManagerAgent` mediates all specialist agents; agents never call each other, only communicate through `SharedManagerState`.
- **State machine (LangGraph)** — Manager (7 nodes) and Research (3 nodes) are compiled `StateGraph`s with conditional edges and guardrails; Financial/Sentiment implement equivalent loops imperatively.
- **Facade** — `ManagerMemory` fronts `ShortTermMemory` + `LongTermMemory`.
- **Factory** — `LongTermMemory.create()` (construct + load), `get_settings()` (`lru_cache` singleton), `get_embedder()` / `_get_retriever()` etc. (lazy singletons), `get_manager_memory()` (DI factory).
- **Singleton** — process-wide model singletons (`AlphaEmbedder`, FinBERT, VADER) with explicit `reset_*` for tests.
- **Dependency Injection** — FastAPI `Depends`, plus every agent accepting an injected `llm_client` so tests pass mocks without network calls.
- **Strategy** — pluggable fusion in `rag_hybrid_query` (RRF / weighted / union); interpretation-thresholds in ratio `_label`.
- **Publish/Subscribe (Observer)** — `progress_bus` decouples pipeline events from the SSE transport.
- **Decorator** — `@traceable` (LangSmith) on all agent/node entry points; `with_error_reporting` in `core/error_handler.py`.
- **Adapter / Wrapper** — `financial_server.py` `tool_*` functions adapt pure module functions into MCP tools; `_format_tool_result` adapts raw MCP output to LLM-friendly chunks.
- **Circuit breaker (lightweight)** — per-ticker/per-feed try/except in `AlphaLoader`; per-tool try/except in executors.

**Strategic approach & key architectural decisions**

- **Contract-based state ownership** (`agents/state.py`): exactly one writer per shared field; three isolation levels (public shared / agent-private / graph-internal). `EvaluationSnapshot` is copied into graph state so routing never depends on memory-layer availability.
- **LLM as control plane, code as data plane**: three Manager "Brain" passes (route / evaluate / finalise) make decisions, but deterministic fallbacks exist for every LLM failure, and pure-Python tools do the numeric work.
- **JSON discipline with Claude**: `"{"` assistant prefill + a balanced-brace repair scanner (`_extract_json_block`) + markdown-fence stripping everywhere JSON is parsed.
- **Idempotent knowledge graph**: Neo4j `MERGE` with `ON CREATE`/`ON MATCH` and running-average edge weights makes re-ingestion safe.
- **Fair budgeting for LLM context**: per-section / per-article / per-chunk char budgets prevent one large document from starving others in prompts.
- **Period provenance**: each ratio carries a `_period` tag, and a non-blocking QA validator cross-checks the final report's narration against it.

**Notable algorithms / business logic**

- Reciprocal Rank Fusion (`k=60`) for hybrid retrieval.
- Exponential **freshness decay** (72-hour half-life) + source-diversity capping + token-budget greedy selection.
- **Weighted composite financial score** (ROE 25%, Net Margin 20%, Revenue CAGR 20%, P/E 15%, Current Ratio 10%, D/E 10%) → 0-100 + letter grade.
- **DCF** point estimate + bear/base/bull scenario range + Monte-Carlo P10/P50/P90 over the growth assumption.
- **Fear/Greed fusion** (FinBERT 0.65 + VADER 0.35) → `[-1,+1]` with 5 bands.

**Error-handling strategy**

- **Typed exception hierarchy** (`AlphaAgentError` → 6 subclasses) mapped to HTTP codes, each carrying a grep-able `trace_id`; two global FastAPI handlers (typed + catch-all), production hides internals.
- **Graceful degradation everywhere**: missing Neo4j/Supabase/Sentry/LangSmith → warn + disable, never crash; LLM failures → deterministic fallback verdicts/actions; tool failures → error-marker chunks; QA and long-term persistence are non-blocking.
- **Guardrails**: `max_routing_loops` (Manager) and `max_loops` (specialist loops) with recursion-limit sizing prevent infinite loops.

**Data flow**

`POST /analyze` → Pydantic validation → per-request `ManagerMemory` (DI) → `ManagerAgent.run` → LangGraph: hydrate → (brain_route → dispatch → evaluate → persist)* → finalise/abort → each dispatch runs a specialist agent that opens an MCP stdio subprocess and calls tools (Yahoo/SEC/Tavily/RAG/sentiment) → results accumulate in `SharedManagerState` → Brain synthesizes `final_report` → fire-and-forget insert into Supabase `analyses` → `AnalyzeResponse`. Progress events flow out-of-band via `progress_bus` → SSE.

---

## 5. DEPENDENCY MAP

**Internal dependency graph (high level)**

```
api/main.py ──► api/config, api/core/exceptions, api/routes/*, core/observability
            ──► agents/{research,financial,sentiment,manager}_agent, memory/manager_memory
api/routes/analyze.py ──► api/dependencies ──► memory/manager_memory
                      ──► app.state.manager_agent (ManagerAgent)
api/routes/progress.py ──► core/progress_bus
agents/manager_agent.py ──► agents/{research,financial,sentiment}_agent, agents/state,
                            memory/manager_memory, core/observability, core/progress_bus,
                            evaluation/validate_period_consistency
agents/research_agent.py ──► agents/state, tools/research_tools/context_synthesizer,
                             core/{observability,progress_bus}, MCP→research_server.py
agents/financial_agent.py ──► agents/state, core/*, MCP→financial_server.py
agents/sentiment_agent.py ──► agents/state, core/*, MCP→sentiment_server.py
tools/research_tools/research_server.py ──► tavily_search, news_search, sec_edgar,
                                            comprehensive_analysis, rag/hybrid_rag
tools/financial_tools/financial_server.py ──► yahoo_finance, sec_edgar, financial_ratio_calculator
tools/sentiment_tools/sentiment_server.py ──► finbert_analyzer, vader_scorer,
                                              fear_greed_calculator, rag/retriever
rag/ingestion.py ──► loader, processor, embedding_manager, vector_store, graph_store
rag/{hybrid_rag,retriever}.py ──► embedding_manager, vector_store, (Neo4j)
```

**External dependencies (from `pyproject.toml`)**

`anthropic`, `langgraph`, `langchain-*`, `langsmith`, `mcp[cli]`, `fastapi`, `supabase`, `neo4j`, `sentence-transformers`, `transformers`, `torch`, `nltk`, `yfinance`, `feedparser`, `httpx`, `requests`, `numpy`, `pandas`, `python-dotenv`, `sentry-sdk`, `ragas`, `datasets`, `pydantic`/`pydantic-settings`, `google-generativeai` / `langchain-google-vertexai` (declared; not referenced in read paths). Dev: `ruff`, `pytest`, `pytest-asyncio`, `pytest-cov`.

**Database / API connections**

| Connection | Purpose |
|-----------|---------|
| Supabase Postgres | `long_term_memory` (memory), `analyses` (results), `alpha_hybrid_search` RPC (pgvector + FTS). |
| Neo4j (Bolt) | Knowledge graph (entities/relations); optional, degrades to vector-only. |
| Anthropic API | All LLM Brain/Checker/extraction/judge calls. |
| Yahoo Finance (yfinance) | Prices, ratios, revenue, peers. |
| SEC EDGAR | CIK, filings, XBRL, full-text search. |
| Tavily / NewsAPI / Reddit RSS | Web + news + social ingestion. |
| Sentry / LangSmith | Optional observability. |

---

## 6. KEY WORKFLOWS

1. **Analysis request (primary):** `POST /api/v1/analyze` → validate → DI memory → `ManagerAgent.run` → LangGraph loop (route→dispatch→evaluate→persist) across Research→Financial→Sentiment → finalise report → persist to Supabase → `AnalyzeResponse`. Typical latency 15-60 s.
2. **Live progress:** client opens `GET /api/v1/analyze/stream/{session_id}` (SSE) *before* POSTing the same `session_id`; every node/agent calls `progress_bus.publish`; stream closes on `pipeline_complete`/`pipeline_error`.
3. **Research sub-loop:** brain (plan JSON) → executor (MCP tools on `research_server.py`) → checker (completeness audit) → loop or END; then a post-loop synthesis compresses chunks.
4. **Financial sub-loop:** brain → executor data extraction (Yahoo/SEC/peers) → executor ratio computation (calc tools + DCF + composite) → 7-criterion checker → loop or commit.
5. **Sentiment sub-loop:** brain_plan → executor (retrieve→FinBERT→VADER→Fear/Greed) → brain_analyze narrative; retries once if 0 chunks.
6. **Ingestion ETL (background/scheduled):** `run_ingestion_pipeline` Load→Process→Embed→Vector-upsert→Graph-upsert; triggered by `rag/seed.py` (CLI) and `scheduler/daily_refresh.py` (GitHub Actions cron).
7. **Health/readiness:** `/health` (liveness), `/readiness` (Supabase probe) for platform orchestration.

---

## 7. PERFORMANCE CONSIDERATIONS

- **Singleton warm models**: `AlphaEmbedder`, FinBERT, VADER loaded once per process; MCP sentiment singletons lazy-init on first call (2-5 s cold start acknowledged).
- **Startup singletons**: Supabase client, compiled graphs, and agents built once in FastAPI lifespan and reused across requests.
- **Async concurrency**: Manager Brain passes use `AsyncAnthropic`; `rag_hybrid_query` runs vector + graph via `asyncio.gather`; sync models offloaded via `asyncio.to_thread`.
- **Retrieval optimization**: hybrid search caps candidates (top 50→10→5), freshness decay, diversity filter, and token budgeting reduce prompt size and cost.
- **Prompt budgeting**: per-section/article/chunk char caps keep LLM context bounded and avoid starvation.
- **Batched embedding/inference**: batch size 64 (embeddings), batched FinBERT inference; CUDA/MPS/CPU device selection with GPU-OOM→CPU retry.
- **Graph guardrail**: traversal hops capped at 3.
- **DB access**: pgvector RPC pushes vector+FTS+RRF into Postgres; Neo4j `MERGE` upserts are idempotent; memory stores use FIFO caps (100 heuristics / 200 tickers). Readiness uses a `limit(1)` probe.
- **Backpressure**: progress queue capped at 300 with drop-on-full so a slow SSE client can't stall the pipeline.

*(No explicit response caching layer or DB index DDL is present in first-party code; the `alpha_hybrid_search` RPC and any indexes are provisioned externally in Supabase.)*

---

## 8. SECURITY OBSERVATIONS

- **Authentication/authorization:** **None implemented.** `user_id` is taken directly from the request body (or `X-User-Id` header) — any caller can read/write any user's long-term memory. Code comments explicitly flag JWT as future work (`get_user_id`, `analyze.py`). **This must be addressed before production exposure.**
- **Input validation:** Pydantic models enforce `query` length (10-500), `ticker` format (1-5 alpha, uppercased), `days_back` range (1-365), and `search_depth` enum. EDGAR/news queries are sanitized (`_sanitize_query`).
- **CORS:** wildcard + credentials-off in development; explicit `ALLOWED_ORIGINS` + credentials-on in production.
- **Secrets:** all keys via env/`.env` (`pydantic-settings`); `send_default_pii=False` in Sentry; startup fails fast on missing required secrets. **Note:** a real `.env` file exists on disk (untracked) — ensure it is never committed.
- **Error disclosure:** production error responses omit internal detail, exposing only a `trace_id`; docs/redoc disabled in production.
- **Service role key:** `SUPABASE_SERVICE_ROLE_KEY` (full DB privileges) is used server-side — appropriate, but combined with missing auth it means the API is the only trust boundary.
- **SSRF/rate-limit surface:** outbound calls to EDGAR/Yahoo/Tavily/NewsAPI/Reddit; no application-level rate limiting on `/analyze` observed.

---

## 9. TECHNICAL DEBT & OBSERVATIONS

**Explicit debt markers / self-documented issues**
- `get_peer_comparison` auto-peer inference is broken (fetches `recommendations` then discards it) — worked around by a static `_INDUSTRY_PEERS` map (comment in `financial_agent.py`).
- Numerous `DC-*` / `M-*` fix comments document past bugs (e.g. DC-5 rerun mislabeling, chunk-truncation regressions, current_ratio/de_ratio sub-score misattribution) — indicating an evolving codebase with regression history baked into comments.
- `core/error_handler.py` contains a stray `_executor_node` example that looks like leftover illustrative code.

**Code smells / refactor candidates**
- **No authentication** (highest priority) — see §8.
- **`main.py` (root)** is a dead scaffold stub.
- **README.md** is effectively empty/garbled (BOM + stray chars); real docs live in `SYSTEM_ARCHITECTURE.md`.
- **A committed virtual environment** (`evaluation/.venv-eval/**`, thousands of files) is tracked/present in the tree — should be gitignored/removed.
- **Filename typos** in tests (`test_finbert_anayzer.py`, `test_core_eceptions.py`, `test_hybried.py`, `test_vectore_store.py`, `__ini__.py`) — cosmetic but reduce discoverability.
- **Two separate `sec_edgar.py`** implementations (research vs financial tools) with overlapping responsibilities — candidate for consolidation.
- **`google-generativeai`/vertex** deps declared but unused in the read code paths — dependency bloat.
- **In-process progress bus & memory** won't survive multi-worker deployment (documented; needs Redis/broker before horizontal scaling).
- **Fire-and-forget Supabase persistence** means a failed insert silently loses the analysis record (logged only).
- **LLM-dependent evaluation** can advance the pipeline on a fallback `passed=True, score=50` when the evaluator API fails — low-quality outputs may not be retried.

**Potential improvements**
- Add JWT/auth + per-user authorization and request rate limiting.
- Provision the `alpha_hybrid_search` RPC and indexes via migrations checked into the repo.
- Consolidate duplicate SEC EDGAR clients; add retry/backoff for Reddit RSS.
- Remove the committed venv; fix README; rename mistyped test files.
- Add a response/idempotency cache keyed by (query, ticker, day) to cut cost on repeat requests.

---

*End of ticket — generated in read-only mode. No source files were modified.*
