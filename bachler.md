# AgenticAlpha — Full Technical Documentation

> **Purpose of this document**
> This is a single, self-contained technical reference for the **AgenticAlpha** project, written as source material for an academic thesis. It documents the software architecture, a file-by-file breakdown of the code, and — most importantly for the methodology chapter — the exact scientific and algorithmic content (RAG pipeline, hybrid search and fusion, embeddings, the three-brain agent loops, the financial models, and the sentiment-fusion index), always tied back to concrete files and line numbers.
>
> All statements below were extracted by reading the actual source code, not inferred from filenames. Where the code and the pre-existing `SYSTEM_ARCHITECTURE.md` disagree, this document follows the code (e.g. `docker-compose.yml` is effectively empty; the financial MCP server exposes 20 tools, not 17; `evaluation/` contains `run_ragas.py` and `validate_period_consistency.py`, not the `metrics.py`/`backtester.py` named in the older doc).

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [File-by-File Breakdown](#2-file-by-file-breakdown)
   - 2.1 [`agents/` — Agent Layer](#21-agents--agent-layer)
   - 2.2 [`rag/` — Retrieval-Augmented Generation Pipeline](#22-rag--retrieval-augmented-generation-pipeline)
   - 2.3 [`tools/` — MCP Tool Servers](#23-tools--mcp-tool-servers)
   - 2.4 [`api/`, `core/`, `memory/`, `scheduler/` — Application Layer](#24-api-core-memory-scheduler--application-layer)
3. [Scientific / Algorithmic Layer](#3-scientific--algorithmic-layer)
4. [Infrastructure & Tools](#4-infrastructure--tools)
5. [Summary Table](#5-summary-table)

---

# 1. Project Overview & Architecture

## 1.1 Purpose

**AgenticAlpha** (internally also called *"Alpha-Agent Node"*) is a **multi-agent financial-intelligence platform**. Given a natural-language investment question such as *"Is NVIDIA a good buy for Q1 2025?"*, it orchestrates a coordinated team of specialist AI agents that together produce a structured, human-readable investment-analysis report. The report fuses three independent evidence streams:

1. **Qualitative research** — live web/news search, SEC EDGAR filings, and a pre-ingested Retrieval-Augmented Generation (RAG) knowledge base (vector store + knowledge graph).
2. **Quantitative financial analysis** — market data and SEC XBRL financials, converted into verified financial ratios, a weighted composite health score, a scenario-based Discounted Cash Flow (DCF) valuation, and a Monte-Carlo DCF.
3. **Market sentiment** — deep-NLP (FinBERT) and lexical (VADER) sentiment scoring of social/news text, fused into a single Fear/Greed index.

A central **ManagerAgent** decides which specialist to run next, evaluates each result with a second LLM "critic" pass, persists learned facts into a two-level cognitive memory, and finally synthesises the report with a third LLM pass.

The system is designed around three architectural principles, all visible in the code:

| Principle | How it is implemented |
|-----------|-----------------------|
| **Separation of concerns / contract-based state** | Every specialist agent owns exactly one field of the shared state contract (`agents/state.py`); agents never overwrite fields they do not own. |
| **Graceful degradation** | Every external dependency (Neo4j, Supabase, Tavily, NewsAPI, SEC EDGAR, Sentry, LangSmith, even the LLM API) has an isolated fallback path; missing services degrade output quality but never crash the pipeline. |
| **Cognitive memory** | A two-level `ManagerMemory` (ephemeral short-term session memory + Supabase-backed long-term memory) lets the Manager learn heuristics and per-ticker facts across sessions. |

## 1.2 Technology Stack (from `pyproject.toml` / `requirements.txt`)

| Layer | Technology |
|-------|------------|
| Language | Python ≥ 3.12 (async/await throughout) |
| Web framework | FastAPI + Uvicorn, Pydantic v2 / `pydantic-settings` |
| LLM provider | Anthropic Claude — default model `claude-haiku-4-5` |
| Agent orchestration | LangGraph (`StateGraph`) — used inside ManagerAgent and ResearchAgent |
| Tool protocol | MCP (Model Context Protocol) over **stdio JSON-RPC**; both the classic `mcp.server.Server` API and the decorator-based `FastMCP` API are used |
| Vector database | Supabase Postgres + `pgvector`, with an `alpha_hybrid_search` SQL RPC (vector + full-text, RRF fusion) |
| Knowledge graph | Neo4j (Bolt protocol), populated by LLM entity/relation extraction |
| Embeddings | `BAAI/bge-small-en-v1.5` — 384-dim, L2-normalised, via `sentence-transformers` |
| Sentiment (deep) | `ProsusAI/finbert` via HuggingFace `transformers` |
| Sentiment (lexical) | NLTK VADER |
| Market data | Yahoo Finance via `yfinance` |
| Filings | SEC EDGAR (full-text search, submissions API, XBRL company-facts API) |
| News/web search | Tavily API + NewsAPI |
| Persistence | Supabase Postgres tables `alpha_documents`, `long_term_memory`, `analyses` |
| Observability | Sentry (errors) + LangSmith (`@traceable` LLM tracing) — both optional |
| Evaluation | RAGAS dataset + `AlphaEvaluator` (LLM-as-judge) + a bespoke period-consistency validator |
| CI/CD | GitHub Actions (`deploy.yml`, `daily_refresh.yml`) |

## 1.3 Folder Structure

```
AgenticAlpha/
├── api/                          # FastAPI application layer
│   ├── main.py                   # App factory, lifespan singletons, CORS, exception handlers, SSE + index routes
│   ├── config.py                 # Pydantic BaseSettings singleton + validate_settings()
│   ├── dependencies.py           # DI factories: get_user_id(), get_manager_memory()
│   ├── routes/
│   │   ├── analyze.py            # POST /api/v1/analyze — request/response models, calls ManagerAgent.run()
│   │   └── progress.py           # GET /api/v1/analyze/stream/{session_id} — SSE live progress
│   └── core/
│       └── exceptions.py         # AlphaAgentError hierarchy → structured JSON + HTTP status codes
│
├── agents/                       # All agent implementations
│   ├── state.py                  # Single source of truth for every state TypedDict
│   ├── manager_agent.py          # ManagerAgent: 7-node LangGraph, 3 Brain LLM passes
│   ├── research_agent.py         # ResearchAgent: 3-node LangGraph (brain→executor→checker)
│   ├── financial_agent.py        # FinancialAnalystAgent: 3-tier imperative loop
│   └── sentiment_agent.py        # SentimentAgent: 2-tier Brain→Executor loop
│
├── memory/
│   └── manager_memory.py         # ManagerMemory facade: ShortTermMemory + LongTermMemory (Supabase)
│
├── rag/                          # RAG pipeline + knowledge infrastructure
│   ├── ingestion.py              # 5-stage ETL orchestrator run_ingestion_pipeline()
│   ├── loader.py                 # AlphaLoader: yfinance news + Reddit RSS → RawDocument
│   ├── processor.py              # AlphaProcessor: recursive chunking + SHA-256 dedup
│   ├── embedding_manager.py      # AlphaEmbedder singleton (BAAI/bge-small-en-v1.5)
│   ├── vector_store.py           # AlphaVectorStore: Supabase pgvector + alpha_hybrid_search RPC (SQL inline)
│   ├── graph_store.py            # AlphaGraphStore: Claude entity/relation extraction → Neo4j MERGE
│   ├── hybrid_rag.py             # rag_vector_search / rag_graph_traverse / rag_hybrid_query + RRF
│   ├── retriever.py              # AlphaRetriever: 5-stage retrieval pipeline
│   ├── evaluation.py             # AlphaEvaluator: LLM-as-judge RAG metrics
│   └── seed.py                   # CLI seeder — runs run_ingestion_pipeline for a ticker list
│
├── tools/
│   ├── research_tools/
│   │   ├── research_server.py    # MCP server (Server API) — 8 research tools
│   │   ├── tavily_search.py      # Tavily web-search client
│   │   ├── news_search.py        # NewsAPI client with query restructuring + relevance filter
│   │   ├── sec_edgar.py          # EDGAR full-text search + filing fetch/parse (research variant)
│   │   ├── comprehensive_analysis.py  # Concurrent Tavily + SEC filing fetch
│   │   └── context_synthesizer.py     # Post-loop LLM compression of research chunks
│   ├── financial_tools/
│   │   ├── financial_server.py   # FastMCP server — 20 tools
│   │   ├── yahoo_finance.py      # yfinance wrapper (ratios, revenue growth, peers)
│   │   ├── sec_edgar.py          # EDGAR CIK/filings/XBRL company-facts (financial variant)
│   │   └── financial_ratio_calculator.py  # Pure-math ratios, composite score, DCF, Monte-Carlo DCF
│   └── sentiment_tools/
│       ├── sentiment_server.py   # MCP server (Server API) — 4 sentiment tools, lazy singletons
│       ├── finbert_analyzer.py   # FinBertSentimentAnalyzer (ProsusAI/finbert)
│       ├── vader_scorer.py       # VaderLexiconScorer (NLTK VADER)
│       └── fear_greed_calculator.py  # FearGreedIndexCalculator (weighted FinBERT+VADER fusion)
│
├── core/
│   ├── observability.py          # init_sentry() / init_langsmith() + enabled() flags
│   ├── progress_bus.py           # In-process pub/sub (asyncio.Queue per session) for SSE progress
│   └── error_handler.py          # with_error_reporting decorator/context managers (Sentry breadcrumbs)
│
├── scheduler/
│   └── daily_refresh.py          # (stub) daily ingestion entry point
│
├── evaluation/
│   ├── run_ragas.py              # RAGAS evaluation runner
│   ├── validate_period_consistency.py  # Period-consistency QA validator (used by ManagerAgent)
│   ├── ragas_dataset.json        # RAGAS eval dataset (~488 KB)
│   └── ragas_dataset2.json       # RAGAS eval dataset (~550 KB)
│
├── frontend/
│   └── alpha-agent-app.html      # Single-page frontend (served at GET /)
│
├── tests/                        # pytest suite: unit_tests/{api,core,financial,search,sentiment,agents,memory,rag} + integration_tests
│
├── .github/workflows/            # deploy.yml + daily_refresh.yml
├── pyproject.toml / requirements.txt / pytest.ini
├── docker-compose.yml            # (effectively empty)
├── SYSTEM_ARCHITECTURE.md        # older architecture doc (partly aspirational)
└── main.py                       # trivial "Hello" stub — NOT the entry point (uvicorn api.main:app is)
```

## 1.4 Component Relationships & Data Flow

**High-level flow (a single `/analyze` request):**

```
Client ──POST /api/v1/analyze──> api/routes/analyze.py
     │                                  │  (Depends: per-request ManagerMemory, user-scoped)
     ▼                                  ▼
 (optional) opens SSE stream    ManagerAgent.run(task_query, directives, prefs)
 GET /analyze/stream/{id}              │
     ▲                                 ▼  compiled LangGraph StateGraph
     │                    hydrate → brain_route ──> dispatch ──> evaluate ──> persist ──┐
 core/progress_bus       (Brain pass 1)   │        (specialist   (Brain      (memory)   │
 (asyncio.Queue/session) <── publish() ───┘         agent.run)    pass 2)               │
                                          │                                             │
                              finalise (Brain pass 3) <────── loop back ────────────────┘
                                          │
                                          ▼
                          Supabase `analyses` (fire-and-forget) + AnalyzeResponse
```

**Specialist agents → MCP tool servers → data/knowledge stores:**

```
ResearchAgent  ──stdio MCP──>  research_server.py   ──> Tavily, NewsAPI, SEC EDGAR,
                                                        rag_vector_search / rag_graph_traverse / rag_hybrid_query
FinancialAgent ──stdio MCP──>  financial_server.py  ──> yfinance, SEC EDGAR XBRL, ratio/DCF calculator
SentimentAgent ──stdio MCP──>  sentiment_server.py  ──> AlphaRetriever (RAG), FinBERT, VADER, Fear/Greed

RAG tools ──> Supabase pgvector (alpha_hybrid_search RPC)  +  Neo4j knowledge graph
Ingestion (rag/ingestion.py) pre-populates BOTH stores from yfinance news + Reddit RSS.
```

Each specialist agent launches its MCP server as a **subprocess** (`StdioServerParameters(command="python", args=[server.py])`) and talks to it over stdin/stdout JSON-RPC; there is no HTTP between an agent and its tools.

## 1.5 The Agents

| Agent | File | Internal architecture | Owns state field |
|-------|------|-----------------------|------------------|
| **ManagerAgent** | `agents/manager_agent.py` | 7-node LangGraph state machine; 3 Claude "Brain" passes (route, evaluate, finalise); loop guardrail | `agent_execution_history`, `orchestrator_logs`, `final_report` |
| **ResearchAgent** | `agents/research_agent.py` | 3-node LangGraph: `brain → executor → checker`, looping (max 3) | `aggregated_research_context` |
| **FinancialAnalystAgent** | `agents/financial_agent.py` | 3-tier imperative loop: Brain → Executors (extract + compute) → Checker, looping (max 3) | `financial_metrics_summary` |
| **SentimentAgent** | `agents/sentiment_agent.py` | 2-tier: Brain-plan → Executor pipeline → Brain-analyze, looping (max 2) | `sentiment_analysis_summary` |

## 1.6 MCP Servers & Tools Exposed

| Server | File | API style | Tools |
|--------|------|-----------|-------|
| **research-agent-mcp** | `tools/research_tools/research_server.py` | `mcp.server.Server` + `match/case` | `tavily_search`, `news_search`, `sec_edgar_search`, `sec_edgar_filing`, `rag_vector_search`, `rag_graph_traverse`, `rag_hybrid_query`, `comprehensive_analysis` (8) |
| **financial-analyst-agent** | `tools/financial_tools/financial_server.py` | `FastMCP` (`@mcp.tool()`) | **20 tools**: Yahoo (`tool_get_price_history`, `tool_get_financial_ratios`, `tool_get_revenue_growth`, `tool_get_peer_comparison`); SEC (`tool_get_cik`, `tool_list_filings`, `tool_get_filing_text`, `tool_get_xbrl_financials`); Calc (`tool_calc_pe`, `tool_calc_pb`, `tool_calc_ev_ebitda`, `tool_calc_peg`, `tool_calc_gross_margin`, `tool_calc_operating_margin`, `tool_calc_net_margin`, `tool_calc_roe`, `tool_calc_roa`, `tool_calc_current_ratio`, `tool_calc_quick_ratio`, `tool_calc_debt_to_equity`, `tool_calc_interest_coverage`, `tool_calc_asset_turnover`, `tool_calc_cagr`, `tool_calc_revenue_cagr_from_growth`, `tool_calc_composite_score`, `tool_calc_dcf`, `tool_calc_dcf_scenarios`, `tool_calc_dcf_monte_carlo`) |
| **sentiment-agent-mcp** | `tools/sentiment_tools/sentiment_server.py` | `mcp.server.Server` + `match/case` | `retrieve_social_data`, `analyze_finbert`, `score_vader`, `calculate_fear_greed` (4) |

> Note: the source docstrings in `financial_server.py` still say "17 tools", but the actual `@mcp.tool()` decorators register **20** callables (the extra three are the DCF family). The FinancialAnalystAgent itself only calls a deterministic subset per run.

---

# 2. File-by-File Breakdown

This section documents every non-trivial Python file. Empty/near-empty `__init__.py` files, `main.py` (a 6-line "Hello" stub), the `.venv`/test-fixture data, and the two large `evaluation/ragas_dataset*.json` files (~1 MB of pre-generated RAGAS eval samples) are noted here but not detailed.

## 2.1 `agents/` — Agent Layer

### `agents/state.py`

- **Role.** The single source of truth for **all** state `TypedDict`s in the platform. It encodes the *contract-based design*: a public shared state that crosses agent boundaries, three private per-agent states, and a private LangGraph orchestration state.
- **Main types.**
  - `SharedManagerState` (Level 1, `total=True`) — the public contract. Fields and owners: `task_query`, `manager_directives` (Manager writes, all read); `aggregated_research_context: list[str|dict]` (ResearchAgent); `financial_metrics_summary: dict` (FinancialAnalystAgent); `sentiment_analysis_summary: dict` (SentimentAgent); `agent_execution_history`, `orchestrator_logs`, `final_report` (ManagerAgent). The docstring enumerates every guaranteed key of the summary dicts (20+ financial keys, 15+ sentiment keys).
  - `ResearchAgentState` (Level 2) — `messages` and `context_chunks` both annotated `Annotated[list, operator.add]` so LangGraph **appends** across loop iterations rather than overwriting; plus `loop_counter`, `validation_feedback`, `is_complete`, `shared_manager_ref`.
  - `FinancialAgentState` (Level 2) — `raw_numerical_data`, `calculated_ratios`, `messages` (manually appended, *not* `operator.add`), `loop_counter`, `validation_feedback`, `is_complete`, `shared_manager_ref`.
  - `SentimentAgentState` (Level 2) — `retrieved_chunks`, `sources_metadata`, `finbert_result`, `vader_result`, `fear_greed_result`, `brain_reasoning`, `loop_counter`, `extraction_errors`, `shared_manager_ref`.
  - `EvaluationSnapshot` (`total=False`) + `ManagerGraphState` (Level 3) — the LangGraph-internal state: `shared_state`, `loop_counter`, `last_action`, `last_agent_key`, `evaluation_passed`, `last_evaluation`, `ticker`, `session_id`.
- **Design pattern.** *Contract-Based Design* with strict field ownership; agent-private states are instantiated inside `run()` and destroyed on return, never leaked back to the Manager. The `EvaluationSnapshot` is stored **inside** graph state (not read back from the memory layer) so routing stays deterministic even if memory is unavailable.
- **Dependencies.** Pure typing (`typing`, `typing_extensions`, `operator`); imported by every agent, the memory layer, and the API.

### `agents/manager_agent.py`

- **Role.** The central orchestrator. Owns the `SharedManagerState` lifecycle, drives the specialist pipeline via a compiled LangGraph `StateGraph`, runs three Claude "Brain" passes, and manages cognitive memory. This is the largest file (~1,770 lines), most of it prompt text and defensive comments.
- **Key constants / prompts.** `_DEFAULT_MODEL = "claude-haiku-4-5"`; `_DEFAULT_MAX_ROUTING_LOOPS = 8`; `_VALID_ACTIONS` (frozenset of 8 routing actions); three large system prompts — `_ROUTER_SYSTEM_PROMPT`, `_EVALUATOR_SYSTEM_PROMPT`, `_FINALISER_SYSTEM_PROMPT`. The finaliser prompt is unusually detailed: it encodes rules for DCF-scenario reporting, peer-PEG comparison, **numeric fidelity** (never cite `composite_score.sub_scores` as if they were raw ratios) and **period fidelity** (respect each ratio's `_period` provenance tag).
- **Main class `ManagerAgent`.** Constructed with the three specialist agents, a `ManagerMemory`, a model string, `max_routing_loops`, and an optional injected `AsyncAnthropic` client (for tests). Uses `anthropic.AsyncAnthropic` so Brain calls never block the event loop.
  - `_hydrate_state(task_query, directives)` → fresh `SharedManagerState`.
  - `_recall(ticker)` → pulls short + long-term memory.
  - `_brain_route(state, memory_recall, loop_counter)` — **Brain pass 1**: builds a compact state summary, calls Claude (`max_tokens=768, temperature=0`), parses JSON, validates the action against `_VALID_ACTIONS`. **Fallback on API failure:** deterministically advances `run_research → run_financial → run_sentiment → finalise` based on which agents already ran.
  - `_brain_evaluate(agent_name, state, memory_ctx)` → `EvaluationFeedback` — **Brain pass 2**: grades the just-run agent's output. **Fallback:** on API failure returns `passed=True, score=50` so the pipeline advances. Does *not* write to memory (single-write invariant enforced by `_node_evaluate`).
  - `_brain_finalise(state)` — **Brain pass 3**: synthesises the report (`max_tokens=3584, temperature=0`). Prefers a synthesized research-summary chunk if present; otherwise uses per-section "fair budgeting" helpers (`_format_filing_chunk_fairly`, `_format_news_chunk_fairly`) so one huge SEC section cannot starve MD&A / risk-factor text in the prompt.
  - `_dispatch(action, state)` — resolves the agent key (strips `run_`/`rerun_`), logs the dispatch, publishes progress, awaits `agent.run(state)`, records timing/outcome, and appends to `agent_execution_history`.
  - `_persist(agent_key, state, evaluation)` — writes heuristics + ticker insights into long-term memory per agent type.
  - **7 LangGraph nodes** (`_node_hydrate`, `_node_brain_route`, `_node_dispatch`, `_node_evaluate`, `_node_persist`, `_node_finalise`, `_node_abort`), all `@traceable`. `_node_finalise` additionally runs the **non-blocking period-consistency QA check** (`check_narration_vs_period`) and stashes findings in `shared_state["qa_period_findings"]`.
  - **2 conditional routers:** `_should_route` (after brain_route → `dispatch`/`finalise`/`abort`, enforcing `loop_counter >= max_routing_loops → abort`) and `_should_continue_after_persist` (after persist → `brain_route`/`dispatch`/`abort`).
  - `_build_graph()` — wires the topology and compiles it.
  - `run(task_query, manager_directives, user_preferences, client_session_id)` — public entry point: seeds a session id, stores preferences, hydrates state, and calls `self._graph.ainvoke(initial, config={"recursion_limit": (max_routing_loops+2)*4})`.
- **Notable technical details.** Graph-based orchestration; three-pass LLM (route/evaluate/finalise) with deterministic fallbacks at every pass; a strict "evaluation written exactly once" invariant; a careful distinction between a genuine `rerun_*` (same agent retried) and simply advancing to a not-yet-run agent (the "DC-5" comment); recursion-limit guardrail derived from `max_routing_loops`.

### `agents/research_agent.py`

- **Role.** LangGraph-powered research specialist. Plans MCP tool calls, executes them against `research_server.py`, audits completeness, and loops until complete or capped. Uses the **synchronous** `anthropic.Anthropic` client wrapped in `asyncio.to_thread` inside async nodes.
- **Key constants / prompts.** `_DEFAULT_MAX_LOOPS = 3`; `_MCP_SERVER_PARAMS` (launches `research_server.py`); `_BRAIN_SYSTEM_PROMPT` (exact per-tool argument schemas, ticker-injection rules, and a crucial *keyword-vs-semantic* query-format rule: NewsAPI/EDGAR are keyword engines needing 2–5 keywords, whereas Tavily/RAG handle natural language); `_CHECKER_SYSTEM_PROMPT` (completeness criteria + "negative findings are valid findings").
- **Class `ResearchAgent`.**
  - `run(shared_state)` — instantiates the private `ResearchAgentState`, runs the compiled graph, then calls `synthesize_research_context(...)` to **append** (never replace) a dense summary chunk to `aggregated_research_context`.
  - `_brain_node` — Claude produces a JSON `actions` plan (1–3 tools). `@traceable(run_type="llm")`.
  - `_executor_node` — opens an stdio MCP session via `AsyncExitStack`, injects a missing ticker argument (`_ensure_ticker_argument`, mapping each tool to its ticker key), executes each tool, and appends labelled chunks. Tool failures become error markers, not crashes. `@traceable(run_type="tool")`.
  - `_checker_node` — audits the chunks. Uses a **per-chunk budget** (`max(800, 12000 // n_chunks)`) so one giant SEC chunk cannot hide later tool calls from the auditor (a fix for confirmed redundant re-fetching); truncated chunks are explicitly marked `[truncated for audit]`.
  - `_should_continue` — routes `brain` vs `__end__`, enforcing `loop_counter >= max_loops` and `is_complete`.
- **Helpers.** `_ensure_ticker_argument` (defensive ticker injection independent of LLM behaviour), `_parse_plan`, `_format_tool_result`, `_TICKER_ARG_BY_TOOL` map.
- **Notable technical details.** LangGraph 3-node loop; robust JSON parsing (strips markdown fences); progress events published at every node; ticker-injection makes correctness independent of whether the LLM remembered to pass the ticker.

### `agents/financial_agent.py`

- **Role.** Quantitative specialist. A three-tier imperative loop (Brain → Executors → Checker) over a long-lived stdio MCP session to `financial_server.py`. Produces `financial_metrics_summary` with verified ratios, composite score, DCF, and capital-allocation data.
- **Key elements.**
  - `_INDUSTRY_PEERS` — a static industry→peers map (used because `get_peer_comparison`'s auto-inference is broken); `_infer_peers(sector, industry, ticker)`.
  - `_CHECKER_SYSTEM_PROMPT` — 7 audit criteria, with an extended note that **different ratios are expected to use different reporting periods** (each carries a `_period` tag) and that only a missing tag, a `MIXED(...)` tag, or a formula mismatch is a real inconsistency.
  - `_sanitize_nans(value)` — recursively replaces `NaN`/`Inf` with `None` (strict-JSON safety for Supabase).
  - `_extract_ticker(task_query, directives)` — directive → regex scan of the query (with a stop-word set).
  - `_execute_data_extraction(session, state)` — **Executor 1**: calls `tool_get_financial_ratios`, then a deterministic `tool_get_peer_comparison` (peers from the static map), `tool_get_revenue_growth`, `tool_get_xbrl_financials`; records per-tool errors.
  - `_execute_ratio_computation(session, state)` — **Executor 2**: an internal `_call(tool, args, period)` helper tags each result with a `_period` provenance label. Computes P/E (ttm), ROE (annual, with a year-alignment check → possible `MIXED(...)`), Net Margin (**prefers same-quarter** quarterly figures, else annual, tagged accordingly), D/E (yfinance MRQ, else an XBRL approximation clearly flagged as *not comparable to standard thresholds*), Revenue CAGR (filters `None`/`NaN`, sorts oldest→newest), **DCF scenarios** and **Monte-Carlo DCF** (grounded on the computed CAGR), and the **composite score** (tagged `"mixed"`).
  - `_check_data_quality(state)` — **Checker**: Claude critic over 7 criteria, with a hard pre-flight guard (skips the API call if state is empty). `@traceable(run_type="llm")`.
  - `_brain(state)` — advisory planner producing `{plan, priority_tools}`.
  - `run(shared_state)` — opens the MCP session, loops Brain→Extract→Compute→Check up to `max_loops`, then assembles `financial_metrics_summary` (identification, market data, all verified ratios, `dcf_valuation`, `dcf_monte_carlo`, `peer_comparison`, `capital_allocation`, revenue history, execution metadata), runs `_sanitize_nans`, and commits it.
- **Notable technical details.** Period-provenance tagging is the file's central idea — it exists to catch a real bug where an annual net-margin was narrated next to quarterly figures. NaN handling is defence-in-depth (source `_safe_float`, CAGR filter, final `_sanitize_nans`). DCF uses the scenario-range tool deliberately (not a single point) so a high-growth stock's value gap reads as a "growth premium", not a modelling error.

### `agents/sentiment_agent.py`

- **Role.** Sentiment specialist. A two-tier Brain→Executor design (no separate Checker — the Brain's second pass does inline semantic QA) over an stdio MCP session to `sentiment_server.py`.
- **Key elements.**
  - `_BRAIN_PLAN_SYSTEM_PROMPT` (produces `{retrieval_query, ticker, days_back, reasoning}`) and `_BRAIN_ANALYZE_SYSTEM_PROMPT` (produces `{overall_sentiment, conviction_level, key_signals, model_agreement, narrative, risk_flags, data_quality_note}` with explicit rules on conviction/agreement thresholds).
  - `_extract_ticker(task_query, directives, financial_summary)` — directive → financial-summary ticker → regex scan.
  - `_brain_plan(state)` — **Brain pass 1**: plans retrieval, grounded in the financial context from the upstream FinancialAnalystAgent.
  - `_execute_sentiment_pipeline(session, state, plan)` — **Executor**: fixed sequence `retrieve_social_data → analyze_finbert → score_vader → calculate_fear_greed`, with defensive skips (empty payloads → Neutral defaults) and optional custom weights from directives.
  - `_brain_analyze(state)` — **Brain pass 2**: interprets FinBERT+VADER+Fear/Greed into an actionable narrative.
  - `run(shared_state)` — loops (max 2; a second pass only if zero chunks were retrieved on pass 1), runs Brain pass 2 outside the MCP session, and assembles `sentiment_analysis_summary`.
- **Notable technical details.** Loop retries only on empty retrieval; the second Brain pass replaces a dedicated Checker; the summary carries both raw model outputs and the LLM narrative for downstream auditability.

### `agents/__init__.py`

- Re-exports the four state TypedDicts from `agents/state.py` for convenience.

## 2.2 `rag/` — Retrieval-Augmented Generation Pipeline

### `rag/ingestion.py`

- **Role.** The 5-stage ETL orchestrator that populates both knowledge stores. Entry point `run_ingestion_pipeline(tickers, skip_graph=False)`.
- **Stages.** (1) **Load** via `AlphaLoader`; (2) **Process** via `AlphaProcessor`; (3+4) **Embed + Vector upsert** via `AlphaEmbedder` + `AlphaVectorStore`; (5) **Graph extract + upsert** via `AlphaGraphStore` — Stage 5 only runs if the vector stage succeeded (`vector_stage_ok`) and `skip_graph` is False.
- **Dependencies / details.** Reads Supabase env vars directly (never imports from `api/`, keeping `rag/` self-contained). Each stage is wrapped in try/except with Sentry breadcrumbs. `__main__` runs `run_ingestion_pipeline(["MSFT","NVDA"])`.

### `rag/loader.py`

- **Role.** Multi-source document loader with UTC normalisation and per-source circuit breakers. Produces `RawDocument(title, content, url, source_type, ticker, published_at)`.
- **Main functions/classes.**
  - `_mentions_ticker(text, ticker, company_name)` — a heuristic guard: a document is only tagged with a ticker if the text contains the ticker symbol as a whole word **or** the first token of the company name. Prevents mislabeling (e.g. a Microsoft article surfaced in Apple's yfinance feed, or a generic Reddit post).
  - `_get_company_name(ticker)` — best-effort yfinance `shortName`/`longName` lookup.
  - `_to_utc_iso8601` / `_safe_timestamp` — normalise RFC-2822, epoch, and several ISO variants to UTC ISO-8601; fall back to `utcnow()` on failure.
  - `AlphaLoader` — `load(tickers)` calls `_fetch_yfinance` (per-ticker news, capped at 20) and `_fetch_reddit_rss` (r/investing + r/wallstreetbets, capped at 30/feed). Reddit posts are validated against **every** ticker in the batch; a post mentioning N tickers produces N tagged copies; a post matching none is dropped.
- **Notable technical details.** Circuit breakers isolate a broken source; the ticker-mention validator prevents corpus contamination that no query-time filter could later fix.

### `rag/processor.py`

- **Role.** Turns `RawDocument`s into deduplicated, metadata-enriched `ProcessedChunk`s.
- **Main elements.** `AlphaProcessor` uses a `RecursiveCharacterTextSplitter` (`chunk_size=512`, `chunk_overlap=64`, custom `SEPARATORS`). Double-key idempotency: `content_hash = SHA256(full text)`, `url_hash = SHA256(url)` — same URL + same content → skip; same URL + new content → update; else → ingest. The in-memory `_seen` dedup store is FIFO-capped at `_DEFAULT_MAX_SEEN = 100_000`. Emits `ProcessorMetrics` (docs, chunks, duplicates, updates). Each chunk carries `content_hash`, `url_hash`, `ticker`, `source_type`, `published_at_utc`, `ingested_at`, `chunk_index`, `url`, `title`.

### `rag/embedding_manager.py`

- **Role.** Thread-safe singleton embedding engine.
- **Main elements.** `get_embedder()` (double-checked locking); `AlphaEmbedder` loads `BAAI/bge-small-en-v1.5` (384-dim). `_select_device()` prioritises CUDA → MPS → CPU; `_load_model` and `_encode_batch` both fall back to CPU on `OutOfMemoryError`. Batch size 64; `normalize_embeddings=True` plus a belt-and-suspenders explicit L2 normalisation (guarding against zero-norm division). `embed_chunks(chunks)` → list of `{embedding, metadata, text}`; `embed_query(str)` → single vector. `reset_embedder()` for test teardown.

### `rag/vector_store.py`

- **Role.** Supabase pgvector wrapper. Also documents (as an inline SQL block) the full database schema the system depends on.
- **Documented schema.** `alpha_documents` table (with `embedding vector(384)`, a generated `fts tsvector`, unique key `(url_hash, chunk_index)`); an **HNSW** index (`m=16, ef_construction=64`, cosine ops); a **GIN** index on `fts`; and the `alpha_hybrid_search(...)` SQL function that performs the RRF hybrid search (see §3).
- **Class `AlphaVectorStore`.** `upsert(records)` (idempotent on `url_hash,chunk_index`); `hybrid_search(query_embedding, query_text, ticker, days_back, top_k, rrf_k=60, score_threshold, limit, offset)` calls the RPC and applies client-side threshold + pagination; `_to_row` maps a record to the DB schema.

### `rag/graph_store.py`

- **Role.** Builds the Neo4j knowledge graph by LLM extraction. The most technically nuanced RAG component.
- **Main elements.** Dataclasses `Entity`, `Relation`, `GraphDocument`. `_EXTRACTION_PROMPT` instructs Claude to emit entities (`Company`, `Person`, `GeopoliticalEvent`, `MacroEvent`, `Product`, `Sector`) and relations (`COMPETES_WITH`, `SUPPLIES_TO`, `AFFECTED_BY`, `LED_BY`, `PART_OF`, `RELATED_TO`, `ACQUIRED_BY`).
  - `AlphaGraphStore.connect()` — opens the Anthropic client + Neo4j driver (network I/O kept out of `__init__` for testability); Neo4j is optional (graph writes disabled if creds absent) and `_ensure_constraints()` creates uniqueness constraints + a `Company.ticker` index.
  - `extract_batch(raw_docs)` → `_extract_one` uses a **JSON-prefill mechanic** (assistant turn prefilled with `"{"`, `temperature=0.0`) to force raw JSON, then `_parse_graph_doc`.
  - `_extract_json_block(raw)` — a **balanced-brace parser** (tracks depth, string state, and escapes) robust to markdown fences, preambles, trailing commentary, and braces inside strings.
  - `upsert_batch(graph_docs)` → idempotent Neo4j `MERGE`: `_merge_entity` (MERGE on name, `ON CREATE`/`ON MATCH SET`) and `_merge_relation` (MERGE the typed edge; **weight averaging** `r.weight = ($weight + r.weight)/2.0` on match).
- **Notable technical details.** Invalid entity types default to `Company`, invalid relations to `RELATED_TO`. Full idempotency means re-running ingestion converges relationship weights rather than duplicating.

### `rag/retriever.py`

- **Role.** `AlphaRetriever` — the multi-stage retrieval pipeline used by both the sentiment server and (indirectly) research vector search.
- **Constants.** `DECAY_HALF_LIFE_HOURS=72`, `STAGE1_TOP_K=50`, `STAGE2_TOP_K=10`, `STAGE3_TOP_K=5`, `TOKEN_BUDGET=2000`, `CHARS_PER_TOKEN=4`.
- **Pipeline (`_run_pipeline`).** Stage 1 — `store.hybrid_search` (always runs). Stage 2 — freshness reranking (`fresh = rrf_score * exp(-hours_old/72)`), optional. Stage 3 — source-diversity filter (≤2 chunks/URL, ≤3 chunks/source_type), optional. Stage 4 — greedy token-budget inclusion, optional. Stage 5 (`_format_context`) — citation-formatted string for LLM prompts. Public methods `retrieve()` (formatted string) and `retrieve_raw()` (raw dicts). The three `apply_*` flags let a consumer bypass narrowing (the sentiment server disables freshness + diversity, keeps a larger token budget as a crash-guard).

### `rag/hybrid_rag.py`

- **Role.** Exposes the three RAG tools that `research_server.py` registers. Handles graceful degradation of Supabase/Neo4j/embedder at import time.
- **Main functions.**
  - `rag_vector_search(query, top_k, ticker_filter, days_back, threshold)` — routes through `AlphaRetriever.retrieve_raw` (full pipeline), returns scored chunks; degrades to `{"results": [], "warning": ...}` if the retriever is unavailable.
  - `rag_graph_traverse(entity, relation_types, max_hops, limit)` — Cypher variable-length traversal `[*1..max_hops]`, **`max_hops` hard-capped at 3**; degrades to `{"nodes": []}` if Neo4j is down.
  - `rag_hybrid_query(query, entity, top_k, max_hops, fusion)` — runs vector + graph in parallel (`asyncio.gather`), fuses via RRF / weighted / union; **degrades to vector-only** if no entity can be resolved (`_extract_ticker_from_query` best-effort fallback).
  - Helpers: `_rrf(a, b, k=60)`, `_weighted(vec, graph, w=0.7)`, `_key` (md5 of first 100 chars), `_extract_ticker_from_query`.
- **Notable technical details.** Graph node scores are `1/(hops+1)`; fusion merges by text-hash key; all functions `@traceable(run_type="retriever")`.

### `rag/evaluation.py`

- **Role.** `AlphaEvaluator` — an LLM-as-judge RAG evaluation framework.
- **Main elements.** Dataclasses `MetricResult`, `EvaluationReport` (with a weighted `overall_score`: faithfulness 0.35, precision 0.25, recall 0.25, relevance 0.15). Four judge prompts: **Faithfulness** (claims supported / total), **Context Precision** (relevant chunks / total), **Context Recall** (key facts covered / total, needs ground truth), **Answer Relevance**. `evaluate`, `batch_evaluate`, `aggregate_scores`, plus `_call_judge` and a fence-stripping `_parse_json`. (Note: `JUDGE_MODEL` is an empty string in the source — the model must be set before use.)

### `rag/seed.py`

- **Role.** CLI seeder. Runs the real `run_ingestion_pipeline` (vector + graph) for a default ticker list `["NVDA","MSFT","AAPL","TSLA","AMD"]` or CLI-supplied tickers, and prints verification SQL/Cypher.

## 2.3 `tools/` — MCP Tool Servers

### `tools/research_tools/research_server.py`

- **Role.** MCP server (`mcp.server.Server`) exposing 8 research tools over stdio. Advertises tool schemas via `@app.list_tools()` and routes calls via `@app.call_tool()` + `match/case`.
- **Tools.** `tavily_search`, `news_search`, `sec_edgar_search`, `sec_edgar_filing`, `rag_vector_search`, `rag_graph_traverse`, `rag_hybrid_query`, `comprehensive_analysis`.
- **Notable details.** `_normalize_sections` coerces loose LLM section names (e.g. `"Item 7"`, `"MD&A"`) to canonical keywords (`business`, `risk_factors`, `mda`, `financial_statements`, `all`). `sec_edgar_filing` floors `max_chars` at 25,000 so MD&A/R&D discussion is not cut off. On Windows it sets `WindowsSelectorEventLoopPolicy`. Every tool error is captured to Sentry and returned as `{"error": ...}` with `isError=True`, never raised across the MCP boundary.

### `tools/research_tools/tavily_search.py`

- **Role.** Async Tavily web-search client (`httpx`). `tavily_search(query, max_results, search_depth, include_domains, topic="finance")` → `{query, answer, results:[{title,url,snippet,score,published_date}]}`. `include_answer=True`.

### `tools/research_tools/news_search.py`

- **Role.** NewsAPI `/v2/everything` client with substantial query-engineering logic.
- **Main functions.** `_extract_terms` (strips relative-time phrases via `_RELATIVE_TIME_PATTERN`, drops `_GENERIC_FILLER_WORDS`, dedupes); `_build_boolean_query` (AND-all if ≤3 terms, else `head AND (rest OR'd)`); `_restructure_query`; `_count_term_matches` / `_min_required_matches` (post-fetch relevance filter requiring ≥ min terms in title/description); `DEFAULT_EXCLUDED_DOMAINS` (github, HN, stackoverflow, reddit, medium). `news_search(...)` defaults `sort_by="relevancy"`, applies the relevance filter, and attaches `low_confidence`/`warning` flags when results are thin.
- **Notable details.** Encodes hard-won lessons: long natural-language queries against a keyword engine return near-zero results; the boolean rewrite + relevance filter combat both over- and under-matching.

### `tools/research_tools/sec_edgar.py` (research variant)

- **Role.** Async EDGAR full-text search + filing fetch/parse (`httpx`). `sec_edgar_search(query, ticker, form_type, max_results)` and `sec_edgar_filing(...)`. `_sanitize_query` caps the term count at `_MAX_QUERY_TERMS=6` and drops bare 4-digit years, because EDGAR full-text (`efts.sec.gov`) returns HTTP 500 on long ANDed queries. Gracefully degrades on upstream failure.

### `tools/research_tools/comprehensive_analysis.py`

- **Role.** `comprehensive_analysis(ticker, company_name, topic_query, form_type, sections, max_results)` runs `tavily_search` (soft/analyst data) **and** `sec_edgar_filing` (hard/official data) concurrently via `asyncio.gather(..., return_exceptions=True)`. Tavily gets the full natural-language query; EDGAR gets only structured `ticker`/`form_type`, so the long query never truncates the filing fetch. Returns `{ticker, news, filing}`.

### `tools/research_tools/context_synthesizer.py`

- **Role.** `synthesize_research_context(chunks, task_query, llm_client, model, max_tokens)` — a plain importable async function called once at the tail of `ResearchAgent.run()`. Compresses raw chunks into a dense executive summary (marked `[SYNTHESIZED RESEARCH SUMMARY ...]`) at `temperature=0`.
- **Design invariants (from the docstring).** (1) **Additive** — appended, never replacing raw chunks, so exact numbers stay traceable for numeric-faithfulness validation; (2) `temperature=0` for reproducibility; (3) **never raises** — returns `None` on failure so a synthesis error can never lose data; (4) not an agent-callable tool — the ResearchAgent's own Brain never decides to call it.

### `tools/financial_tools/financial_server.py`

- **Role.** `FastMCP` server exposing **20** `@mcp.tool()` functions across three groups (Yahoo, SEC EDGAR, Ratio/DCF calculator). All logging goes to **stderr** so stdout stays a clean MCP channel. `_sentry_tool` wraps a call and returns `{"error": ...}` on failure.
- **Tools.** Yahoo: `tool_get_price_history`, `tool_get_financial_ratios`, `tool_get_revenue_growth`, `tool_get_peer_comparison`. SEC: `tool_get_cik`, `tool_list_filings`, `tool_get_filing_text`, `tool_get_xbrl_financials`. Calculator: `tool_calc_pe`, `tool_calc_pb`, `tool_calc_ev_ebitda`, `tool_calc_peg`, `tool_calc_gross_margin`, `tool_calc_operating_margin`, `tool_calc_net_margin`, `tool_calc_roe`, `tool_calc_roa`, `tool_calc_current_ratio`, `tool_calc_quick_ratio`, `tool_calc_debt_to_equity`, `tool_calc_interest_coverage`, `tool_calc_asset_turnover`, `tool_calc_cagr`, `tool_calc_revenue_cagr_from_growth`, `tool_calc_composite_score`, `tool_calc_dcf`, `tool_calc_dcf_scenarios`, `tool_calc_dcf_monte_carlo`. Each is a thin, well-documented wrapper over the corresponding pure function.

### `tools/financial_tools/yahoo_finance.py`

- **Role.** `yfinance` wrapper.
- **Main functions.** `_safe_float` (collapses pandas `NaN`/`Inf` to `None` — critical, since `float(numpy.nan)` passes `is not None` but breaks CAGR comparisons and JSON persistence); `_safe_get`; `get_price_history`; `get_financial_ratios` (P/E, forward P/E, PEG, P/B, P/S, EV/EBITDA, EPS, beta, `current_ratio`, `quick_ratio`, 52-week range, current price); `_fiscal_quarter_label` (builds company-accurate `FYyyyy-Qn` labels from the firm's own fiscal-year-end month — fixes mislabeling non-December fiscal years); `get_revenue_growth` (annual + **quarterly** revenue *and* net income with YoY growth, plus `revenue_growth_ttm` with an explicit period caveat and `year_labeling_warnings`); `get_peer_comparison` (peer averages + a **growth-adjusted PEG comparison** vs peer-average PEG).
- **Notable details.** Adding quarterly net income was the fix that lets the FinancialAnalystAgent compute a genuinely same-quarter net margin instead of falling back to annual figures.

### `tools/financial_tools/sec_edgar.py` (financial variant)

- **Role.** SEC EDGAR client (`requests`, no API key; mandatory descriptive `User-Agent`). Retry/backoff on 429, clear 403 message, connection-pooled session, cached ticker→CIK map.
- **Main functions.** `get_cik` (ticker→CIK via `company_tickers.json`), `list_filings` (submissions API), `get_filing_text` (locates the primary HTML doc via `index.json`, strips HTML/XBRL with `_strip_html`), `get_xbrl_financials` (company-facts XBRL). `_extract_annual(facts, tags)` tries multiple candidate XBRL concept tags per metric and returns up to 10 most-recent annual (`fp="FY"` or `form="10-K"`) `{period_end, value, unit}` entries. Tag families cover revenue, net income, assets, liabilities, **operating cash flow**, **capex**, **dividends paid**, **buybacks** — the last four feed DCF's Free Cash Flow and the capital-allocation summary.

### `tools/financial_tools/financial_ratio_calculator.py`

- **Role.** Pure, source-agnostic math. No external data dependency — every function takes plain numbers and returns a flat dict with value + interpretation label + formula. This is the quantitative core (detailed in §3).
- **Functions.** Valuation: `price_to_earnings`, `price_to_book`, `ev_to_ebitda`, `peg_ratio`. Profitability: `gross_margin`, `operating_margin`, `net_margin`, `return_on_equity`, `return_on_assets`. Liquidity: `current_ratio`, `quick_ratio`. Leverage: `debt_to_equity`, `interest_coverage`. Efficiency: `asset_turnover`. Growth: `cagr`, `compute_revenue_cagr_from_growth`. Scoring: `composite_financial_score` (weighted 0–100, grade A–F). Valuation models: `discounted_cash_flow`, `dcf_scenario_range` (bear/base/bull), `dcf_monte_carlo` (P10/P50/P90). Helpers `_safe_div`, `_label`.

### `tools/sentiment_tools/sentiment_server.py`

- **Role.** MCP server (`mcp.server.Server`) exposing 4 sentiment tools, with **lazy singletons** (`_get_retriever`, `_get_finbert`, `_get_vader`, `_get_fear_greed`) so heavy models load only on first call. All calls run via `asyncio.to_thread` (the models and Supabase I/O are synchronous).
- **Tools.** `retrieve_social_data` (its `AlphaRetriever` is configured with `apply_freshness_rerank=False`, `apply_diversity_filter=False`, `apply_token_budget=True`, `stage1_k` env-tunable — because its consumer is FinBERT/VADER batch scoring, not an LLM prompt, so it wants breadth not a small diverse set); `analyze_finbert`; `score_vader`; `calculate_fear_greed` (weights overridable via args or `FEAR_GREED_*_WEIGHT` env vars). `_to_dict` serialises dataclasses to JSON-safe dicts.

### `tools/sentiment_tools/finbert_analyzer.py`

- **Role.** `FinBertSentimentAnalyzer` — deep financial NLP via `ProsusAI/finbert` (a BERT fine-tuned on Financial PhraseBank). Thread-safe singleton model (440 MB) with CUDA→MPS→CPU fallback. `analyze(texts)` filters empties, runs batched softmax inference (`_infer_batch`, truncation at 512 tokens), and aggregates per-chunk probabilities into corpus-level means → `FinBertResult(bullish_prob, bearish_prob, neutral_prob, label, chunk_scores, ...)`. Label order per config: `{0: positive, 1: negative, 2: neutral}`; `_argmax_label` maps to Bullish/Bearish/Neutral. `reset_finbert()` for tests.

### `tools/sentiment_tools/vader_scorer.py`

- **Role.** `VaderLexiconScorer` — NLTK VADER lexical baseline, tuned for social/slang/emoji. `score(texts)` runs `polarity_scores` per chunk and averages into `VaderResult(compound, positive_mean, negative_mean, neutral_mean, label, ...)`. Thresholds: compound ≥ +0.05 → Bullish, ≤ −0.05 → Bearish, else Neutral. Lexicon auto-downloaded once (`_ensure_vader_lexicon`, double-checked locking). `reset_vader()` for tests.

### `tools/sentiment_tools/fear_greed_calculator.py`

- **Role.** `FearGreedIndexCalculator` — fuses FinBERT + VADER into one normalised Fear/Greed score (detailed in §3). Default weights FinBERT 0.65, VADER 0.35 (validated to sum to 1.0). `calculate(finbert_result, vader_result)` and `calculate_from_dict(...)` for JSON payloads; `_score_to_label` maps to the five bands (Extreme Fear → Extreme Greed); `confidence = |score|`.

## 2.4 `api/`, `core/`, `memory/`, `scheduler/` — Application Layer

### `api/main.py`

- **Role.** FastAPI entry point (`uvicorn api.main:app`). Uses a **lifespan-owned singleton** pattern: all expensive resources are built once at startup and stored on `app.state`.
- **Startup (lifespan).** `validate_settings()` (fail-fast) → `init_sentry(app_env)` + `init_langsmith()` → `create_client(SUPABASE_URL, SUPABASE_KEY)` → instantiate `ResearchAgent`, `FinancialAnalystAgent`, `SentimentAgent(server_script_path=...)` → a system-level `ManagerMemory` → compile the `ManagerAgent` graph.
- **Middleware & handlers.** CORS (wildcard + credentials off in development; explicit `ALLOWED_ORIGINS` + credentials on in production); a request-timing middleware; a global `AlphaAgentError` handler (structured JSON with `trace_id`, Sentry capture) and a catch-all `Exception` handler (500; hides `detail` in production).
- **Routes.** Includes `analyze_router` and `progress_router` under `/api/v1`; `GET /health`, `GET /readiness` (checks Supabase), and `GET /` (serves `frontend/alpha-agent-app.html`).

### `api/config.py`

- **Role.** `pydantic-settings` configuration. `Settings` fields: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL="claude-haiku-4-5"`, `MAX_ROUTING_LOOPS=8`, `SUPABASE_URL`, `SUPABASE_KEY` (validation alias `SUPABASE_SERVICE_ROLE_KEY`), `APP_ENV`, `DEFAULT_USER_ID`, `REQUEST_TIMEOUT_S=300`, `LOG_LEVEL`, `ALLOWED_ORIGINS`. `get_settings()` is `lru_cache`d; `settings` is a backward-compatible alias. `validate_settings()` raises `ConfigurationError` if any required var is missing.

### `api/routes/analyze.py`

- **Role.** The core endpoint `POST /api/v1/analyze`.
- **Models.** `AnalyzeRequest` (`query` 10–500 chars; `ticker` validated 1–5 uppercase letters; `user_id`; `search_depth` basic/advanced; `days_back` 1–365; `include_sentiment`; optional `session_id` used to attach a live SSE stream). `AnalyzeResponse` maps directly from `SharedManagerState`.
- **Flow.** Builds `manager_directives` from the request, pulls `user_preferences` from the injected memory, calls `manager_agent.run(...)`, times it, persists to the Supabase `analyses` table via a fire-and-forget `_persist_analysis` (whose docstring includes the required table DDL), raises `AgentError` on pipeline failure, and returns the structured response.
- **Notable details.** The docstring flags that **JWT auth is not yet implemented** — `user_id` comes straight from the body (a documented security limitation).

### `api/routes/progress.py`

- **Role.** `GET /api/v1/analyze/stream/{session_id}` — Server-Sent Events, implemented with a plain `StreamingResponse` (no extra dependency). Forwards events from `core/progress_bus` as `data: <json>\n\n`; emits a heartbeat every 15 s; closes on `pipeline_complete`/`pipeline_error` or client disconnect. The frontend must open this stream **before** POSTing `/analyze` with the same `session_id` so no early events are missed.

### `api/dependencies.py`

- **Role.** FastAPI DI factories. `get_user_id(request)` (X-User-Id header → `DEFAULT_USER_ID`); `get_manager_memory(request, user_id)` builds a per-request, user-scoped `ManagerMemory` reusing the shared Supabase client from `app.state`. Both are trivially overridable in tests.

### `api/core/exceptions.py`

- **Role.** Exception hierarchy. Base `AlphaAgentError` carries `message`, machine-readable `code`, `http_status`, optional `detail`, and an auto-generated 8-char `trace_id`; `to_dict()` produces the JSON body. Subclasses: `ValidationError` (400), `AgentError` (500), `AgentTimeoutError` (504), `MemoryError` (500), `ExternalServiceError` (503), `ConfigurationError` (500).

### `core/observability.py`

- **Role.** Optional Sentry + LangSmith bootstrap, with zero dependency on `api/`. `init_sentry(app_env)` (idempotent; `traces_sample_rate=1.0`; no-op if `SENTRY_DSN` unset); `init_langsmith()` (sets `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT`; no-op if `LANGSMITH_API_KEY` unset). Module-level flags queried via `sentry_enabled()` / `langsmith_enabled()`. Called at API startup and in every MCP server's `__main__`.

### `core/progress_bus.py`

- **Role.** In-process pub/sub for live progress. One `asyncio.Queue` per `session_id` (`_MAX_QUEUE_SIZE=300`, backpressure via `put_nowait` — full queue drops events). `publish(session_id, event_type, agent, message, detail)` (no-op if `session_id` falsy, so instrumenting the graph never needs guards); `subscribe`, `get_queue`, `close_session`, and `session_from_shared` (extracts the `_progress_session_id` the Manager stashed into `manager_directives`). Documents that a multi-worker deployment would need to swap this for Redis pub/sub.

### `core/error_handler.py`

- **Role.** `with_error_reporting(component)` — a decorator (works for sync **and** async callables) that adds a Sentry breadcrumb on entry and captures-then-reraises on exception (never suppresses control flow). Also exposes `.context` / `.async_context` inline context managers. `_safe_extra` keeps only JSON-scalar kwargs in breadcrumbs.

### `memory/manager_memory.py`

- **Role.** The two-level cognitive memory facade.
- **Dataclasses.** `AgentExecutionRecord` (dispatch log entry) and `EvaluationFeedback` (Brain verdict: `step, timestamp, passed, score, issues, next_action, raw_verdict`).
- **`ShortTermMemory`.** Ephemeral, session-scoped. `messages` (FIFO cap 50), `agent_log`, `eval_feedback`. `reset`, `add_message`/`get_messages`, `log_dispatch`, `add_evaluation`/`get_last_evaluation`, `agents_run`, `to_context_dict` (Brain-readable summary).
- **`LongTermMemory`.** Supabase-backed (`long_term_memory` table; DDL in the docstring). Three stores: `operational_heuristics`, `ticker_insights`, `user_preferences`, with FIFO caps (100 heuristics, 200 tickers). Factory `create(user_id, supabase_client)` constructs **and** loads (keeps `__init__` side-effect-free for tests). `load()` (`SELECT` by `user_id`, silently empty on first use) and `persist()` (`UPSERT`). `recall(ticker)` builds the injection payload.
- **`ManagerMemory`.** Composes both layers behind one API and is the only memory surface the ManagerAgent touches. `new_session`, short-term delegations, long-term delegations, `persist_long_term`, and a unified `recall(ticker)` returning `{short_term, long_term}`.

### `scheduler/daily_refresh.py`

- **Role.** Intended daily-ingestion entry point (referenced by `.github/workflows/daily_refresh.yml`). **In the current source it is a stub** — a single comment line with no implementation; the real ingestion logic lives in `rag/ingestion.py` / `rag/seed.py`.

### `evaluation/` (support / research artifacts)

- `evaluation/validate_period_consistency.py` — a standalone QA validator (also imported by `manager_agent.py` as `check_narration_vs_period`). It builds a `PeriodRegistry` from `raw_numerical_data`, recovers each ratio's source period (via explicit `_period` tags or value-matching), and flags (a) a metric narrated with period language that contradicts its tag, or (b) metrics from different periods narrated together in one section. Includes a worked `_demo_state()` reconstructed from a real NVDA net-margin bug.
- `evaluation/run_ragas.py` — a RAGAS evaluation runner over the two `ragas_dataset*.json` files.
- `evaluation/ragas_dataset.json`, `ragas_dataset2.json` — ~1 MB of pre-generated RAGAS evaluation samples (data, not code).

### Frontend & misc

- `frontend/alpha-agent-app.html` — a ~40 KB single-page app served at `GET /`; it opens the SSE progress stream, then POSTs `/analyze` with the same `session_id`.
- `main.py` — a 6-line "Hello from agenticalpha!" stub; **not** the runtime entry point.
- `docker-compose.yml` — effectively empty (1 line); the local Neo4j/services stack it is described as in the older architecture doc is not actually defined here.

---

# 3. Scientific / Algorithmic Layer

This section is the methodological core: for each scientific/algorithmic component it states the problem, the exact step-by-step method as implemented, the parameters and formulas present in the code, and the file/line references.

## 3.1 Embeddings — `BAAI/bge-small-en-v1.5`, 384-dim, L2-normalised

- **Problem.** Map variable-length financial text chunks into a fixed 384-dimensional vector space where cosine similarity is meaningful for semantic retrieval.
- **Method (`rag/embedding_manager.py`).** A singleton `AlphaEmbedder` loads the sentence-transformer model on the best device (CUDA→MPS→CPU, `_select_device` lines 25–48). `_encode_batch` (lines 191–226) encodes in batches of 64 with `normalize_embeddings=True`, then applies an explicit belt-and-suspenders L2 normalisation:

```python
# rag/embedding_manager.py:224-226
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
return embeddings / norms
```

- **Formula.** For an embedding vector **v**, the stored vector is **v / ‖v‖₂** (with the zero-norm guard). Because all vectors are unit-norm, cosine similarity reduces to a dot product, and the pgvector cosine operator `<=>` is a valid distance.
- **Parameters.** `model_name="BAAI/bge-small-en-v1.5"`, `batch_size=64`, dimension 384.

## 3.2 Vector store & hybrid SQL search (Reciprocal Rank Fusion in Postgres)

- **Problem.** Combine dense semantic similarity with sparse keyword (full-text) relevance so that both "means the same thing" and "contains the exact term" retrieval succeed.
- **Method (`rag/vector_store.py`, inline SQL `alpha_hybrid_search`, lines 44–113).** Two ranked candidate lists are produced independently and fused by **Reciprocal Rank Fusion (RRF)**:
  1. `vector_ranked` — rows ordered by `embedding <=> query_embedding` (cosine distance), `row_number()` as rank, filtered by optional `ticker` and `days_back` (lines 66–76).
  2. `fts_ranked` — rows ordered by `ts_rank_cd(fts, plainto_tsquery('english', query_text))`, `row_number()` as rank (lines 77–87).
  3. `rrf` — `FULL OUTER JOIN` on id, summing the reciprocal-rank contributions (lines 88–97).
- **Formula (RRF, in SQL, lines 91–92):**

```sql
coalesce(1.0 / (rrf_k + v.rank), 0) + coalesce(1.0 / (rrf_k + f.rank), 0) as rrf_score
```

  i.e. for a document *d*, `score(d) = Σ_list 1 / (k + rank_list(d))` with **k = rrf_k = 60** (default). Missing membership in a list contributes 0. Results are ordered by `rrf_score` descending, then `LIMIT top_k OFFSET page_offset`.
- **Indexing.** HNSW on the embedding (`m=16, ef_construction=64`, `vector_cosine_ops`) for approximate nearest-neighbour, and a GIN index on the generated `fts tsvector`.
- **Client side (`AlphaVectorStore.hybrid_search`, lines 190–238).** Applies a `score_threshold` filter and pagination in Python after the RPC.

## 3.3 `AlphaRetriever` — the five-stage retrieval pipeline

- **Problem.** Turn the raw hybrid-search candidates into a small, fresh, source-diverse, token-bounded, citation-formatted context for an LLM prompt — or, for sentiment, a broad raw sample.
- **Method (`rag/retriever.py`).** Constants (lines 23–27): `DECAY_HALF_LIFE_HOURS=72`, `STAGE1_TOP_K=50`, `STAGE2_TOP_K=10`, `STAGE3_TOP_K=5`, `TOKEN_BUDGET=2000`, `CHARS_PER_TOKEN=4`.
  - **Stage 1 — Hybrid search** (always runs; top 50 candidates).
  - **Stage 2 — Freshness reranking** (`_rerank_by_freshness`, lines 178–199). Exponential time decay:

```python
# rag/retriever.py:193-194
decay       = math.exp(-hours_old / self.decay_half_life_hours)
fresh_score = rrf_score * decay
```

  **Formula:** `freshness_score = rrf_score · exp(−hours_old / 72)`. With a 72-hour half-life, a 3-day-old document keeps ≈ 50 % of its weight; a document that fails to parse a timestamp defaults to `hours_old = 720` (30 days). Top `STAGE2_TOP_K=10` kept.
  - **Stage 3 — Source diversity** (`_diversity_filter`, lines 223–252): at most 2 chunks per URL and 3 per `source_type`, stopping at `STAGE3_TOP_K=5`.
  - **Stage 4 — Token budget** (`_apply_token_budget`, lines 258–274): greedy inclusion until `char_budget = token_budget × 4` (= 8000 chars by default) is exhausted.
  - **Stage 5 — Citation formatting** (`_format_context`, lines 280–317): `[N] SOURCE: … | TICKER: … | DATE: … | SCORE: …` blocks.
- **Configurability.** The three `apply_*` flags let a consumer bypass stages 2–4. The sentiment server disables freshness + diversity and raises the budget (it wants many chunks for batch scoring, not a small LLM-prompt set).

## 3.4 Graph traversal & hybrid (vector+graph) fusion

- **Problem.** Some questions need *relationships* (competitors, suppliers, geopolitical exposure), not just semantically similar text.
- **Graph traversal (`rag/hybrid_rag.py`, `rag_graph_traverse`, lines 166–240).** Cypher variable-length path `MATCH path = (s {name:$entity})-[*1..max_hops]-(e)`; **`max_hops` is hard-capped at 3** (line 183) to avoid unreliable deep reasoning chains. Returns nodes with `{name, type, relation, hops, path_nodes}` ordered by hop count.
- **Hybrid fusion (`rag_hybrid_query`, lines 247–342).** Runs `rag_vector_search` and `rag_graph_traverse` concurrently (`asyncio.gather`). Vector items keep their retriever score; graph items are scored by inverse hop distance:

```python
# rag/hybrid_rag.py:316-324  (graph item score)
"score": round(1.0 / (n["hops"] + 1), 4)
```

  Fusion modes: `rrf` (default), `weighted`, or `union`.
- **RRF helper (`_rrf`, lines 379–387).** `score(item) = Σ 1 / (k + rank)`, **k = 60**, keyed by an md5 hash of the item's first 100 chars so the same text from both lists merges. Weighted mode (`_weighted`, lines 390–393): vector items × 0.7, graph items × 0.3.
- **Graceful degradation.** If no `entity` can be resolved, `rag_hybrid_query` degrades to vector-only (lines 270–292) rather than failing.

## 3.5 Knowledge-graph construction — LLM extraction with prefill + balanced-brace repair

- **Problem.** Convert unstructured financial text into a structured, idempotent knowledge graph.
- **Method (`rag/graph_store.py`).**
  - **JSON-prefill mechanic (`_extract_one`, lines 269–298).** The assistant turn is prefilled with `"{"` and `temperature=0.0`, forcing raw JSON with no markdown fences or preamble; the `"{"` is prepended back:

```python
# rag/graph_store.py:279-287
messages=[
    {"role": "user",      "content": prompt},
    {"role": "assistant", "content": "{"},   # prefill forces JSON
],
raw = "{" + response.content[0].text
```

  - **Balanced-brace JSON repair (`_extract_json_block`, lines 304–342).** A stateful scanner tracks brace depth, in-string state, and escape sequences, returning the first balanced `{...}` — robust to markdown fences, trailing commentary, leading preamble, and braces inside string values.
  - **Idempotent Neo4j upsert.** `_merge_entity` (lines 405–419) `MERGE (e:Type {name})` with `ON CREATE`/`ON MATCH SET`. `_merge_relation` (lines 421–444) merges the typed edge and **averages the weight** on repeat observation:

```cypher
-- rag/graph_store.py:433-435
ON MATCH SET  r.weight = ($weight + r.weight) / 2.0, r.updated_at = timestamp()
```

  - **Validation fallbacks.** Invalid entity type → `Company`; invalid relation → `RELATED_TO`.
- **Formula.** Repeated ingestion of the same edge converges its weight to a running average: after observing weights w₁…wₙ in order, the stored weight is the iterated pairwise mean `((…((w₁+w₂)/2 + w₃)/2 …) + wₙ)/2`.

## 3.6 Financial ratios & the weighted composite health score

- **Problem.** Convert raw statement figures into standard, interpretable ratios and a single 0–100 health score.
- **Ratios (`tools/financial_tools/financial_ratio_calculator.py`).** Each function uses `_safe_div` (returns `None` on divide-by-zero/None) and `_label` (threshold→qualitative label). Examples: P/E `price/eps` (lines 92–129), Net Margin `net_income/revenue×100` (lines 296–320), ROE `net_income/equity×100` (lines 323–347), D/E `total_debt/equity` (lines 457–487), Current Ratio (lines 381–411).
- **CAGR (`cagr`, lines 564–594).**

```python
# tools/financial_tools/financial_ratio_calculator.py:585
rate = (end_value / start_value) ** (1 / years) - 1
```

  **Formula:** `CAGR = (End/Start)^(1/years) − 1`, guarded by `start_value > 0` and `years > 0`. `compute_revenue_cagr_from_growth` (lines 597–683) sorts the annual-revenue list oldest→newest, drops `None` revenues, and requires ≥ 2 valid years.
- **Composite score (`composite_financial_score`, lines 690–789).** Each metric is min–max normalised to 0–10, optionally inverted (lower-is-better), and weighted. Normalisation (lines 752–764):

```python
clamped = max(min_val, min(raw, max_val))
norm    = (clamped - min_val) / (max_val - min_val) * 10
if invert: norm = 10 - norm
```

  **Weights and ranges (lines 769–774):** ROE 25 % (0–40), Net Margin 20 % (0–30), Revenue CAGR 20 % (0–50), P/E 15 % (5–60, inverted), Current Ratio 10 % (0–3), D/E 10 % (0–3, inverted). Only weights for which data is present are used:

```python
# tools/financial_tools/financial_ratio_calculator.py:780
score = round((weighted_total / weight_used) * 10, 1)
```

  **Grade bands (line 782):** A ≥ 80, B ≥ 65, C ≥ 50, D ≥ 35, else F. Normalised sub-scores are stored under `<metric>_normalised` keys to prevent them being mistaken for the raw ratios.

## 3.7 Discounted Cash Flow — single-point, three-scenario, and Monte-Carlo

- **Problem.** Produce an intrinsic-value sanity check, honestly exposing the growth-rate sensitivity that a single-point DCF hides.
- **Single-point DCF (`discounted_cash_flow`, lines 795–980).**
  - **Discount rate = CAPM cost of equity** (used as a WACC stand-in), line 919:

```python
discount_rate = (risk_free_rate_pct + beta * market_risk_premium_pct) / 100.0
```

  **Formula:** `r = risk_free + β · market_risk_premium` (defaults: risk-free 4.3 %, premium 5.0 %).
  - **Projection & discounting** (lines 942–948): for years 1…N, `FCFₜ = FCFₜ₋₁·(1+g)`, present value `Σ FCFₜ / (1+r)ᵗ`.
  - **Terminal value (Gordon Growth)** (line 951): `TV = FCF_N·(1+g_term) / (r − g_term)`, discounted `TV / (1+r)ᴺ`; guarded by `r > g_term` (defaults g_term 2.5 %, N=5).
  - **Enterprise value** = PV(FCF) + PV(TV) (line 954); optional per-share = EV / shares. The docstring explicitly discloses simplifications: CAPM-only discount rate (no cost of debt), constant-then-terminal growth (no fade), enterprise value with no net-debt adjustment.
- **Three-scenario DCF (`dcf_scenario_range`, lines 983–1069).** Runs the single-point DCF at bear/base/bull growth (bear ≈ 0.5× base floored at g_term+0.5; bull ≈ 1.75× base). Returns all three plus a `note` stressing that **no probability is assigned**. This is the tool the FinancialAnalystAgent actually calls, grounding `base_growth` in the firm's own revenue CAGR.
- **Monte-Carlo DCF (`dcf_monte_carlo`, lines 1072–1251).** Samples the growth rate from a normal distribution `g ~ N(base, std)` (default `std = max(base·0.4, 2.0)`), clamps `g ≥ g_term + 0.5`, runs `n_simulations=2000` DCFs, sorts enterprise values, and reports P10/P50/P90 percentiles (`_percentile`, lines 1217–1219). Scope is disclosed: **only the growth rate is randomised** (margin, discount rate, terminal growth fixed), so it is a wider-but-not-fully-calibrated uncertainty picture, not a full multi-factor model.

## 3.8 Period-provenance tagging & consistency validation

- **Problem.** A ratio computed from annual data must not be narrated as if it described a quarter. Every individual number can be internally correct while the *combination* is misleading (the real NVDA net-margin bug).
- **Tagging (`agents/financial_agent.py`, `_call` helper, lines 617–655).** Every computed ratio is stamped with a `_period` provenance label (`"ttm"`, `"annual:2025"`, `"quarterly:2026-Q1"`, `"mrq"`, `"mixed"`, `"MIXED(...)"`, or `"derived:annual_projection"`). Net margin prefers *same-quarter* quarterly revenue+net income (lines 737–771), falling back to annual with an explicit `MIXED(...)` tag if the two annual arrays' index-0 years disagree.
- **Validation (`evaluation/validate_period_consistency.py`, `check_narration_vs_period`, lines 282–345).** For each tagged ratio it locates the value in the final report text (`_find_value_context`, strict digit-boundary regex) and checks whether nearby prose uses quarterly / interim-YTD / annual language (`QUARTER_MARKERS`, `INTERIM_MARKERS`, `ANNUAL_MARKERS`) that contradicts the tag. This runs **non-blockingly** inside `ManagerAgent._node_finalise`; findings are logged and attached to state, never blocking the report.

## 3.9 Sentiment models & the Fear/Greed fusion index

- **FinBERT (`tools/sentiment_tools/finbert_analyzer.py`).** Per chunk, softmax over the 3 logits (`_infer_batch`, line 341); labels ordered `{0:positive, 1:negative, 2:neutral}`. Corpus aggregation is the simple mean of per-chunk probabilities (lines 268–283); the corpus label is the argmax of the means. 512-token truncation; batch inference.
- **VADER (`tools/sentiment_tools/vader_scorer.py`).** Per chunk `polarity_scores` → compound ∈ [−1,+1]; corpus compound is the mean of per-chunk compounds (lines 247–252). Thresholds (lines 56–57, `_compound_label` 309–330): ≥ +0.05 Bullish, ≤ −0.05 Bearish, else Neutral.
- **Fear/Greed fusion (`tools/sentiment_tools/fear_greed_calculator.py`).** Steps (`calculate`, lines 166–252):
  1. FinBERT directional score (line 201): `finbert_score = bullish_prob − bearish_prob ∈ [−1,+1]`.
  2. VADER score = compound directly (line 204).
  3. Weighted average (lines 207–210): `raw = finbert_score·W_finbert + vader_score·W_vader`, defaults **W_finbert = 0.65, W_vader = 0.35** (validated to sum to 1.0, `_validate_weights`).
  4. Clamp to [−1,+1] (line 213).
  5. Five-band label (`_BANDS`, lines 72–78): `[0.60,1.0]` Extreme Greed, `[0.20,0.60)` Greed, `(−0.20,0.20)` Neutral, `(−0.60,−0.20]` Fear, `[−1.0,−0.60]` Extreme Fear.
  6. Confidence heuristic = `|score|` (line 219).
- **Rationale.** FinBERT is weighted higher for financial-domain precision; VADER contributes social slang/intensity and an independent cross-check. Both degrade to a Neutral zero-result on empty input rather than crashing.

## 3.10 The three-Brain orchestration loop (agent decision-making)

- **Problem.** Decide which specialist to run, judge each result, and synthesise a report — robustly, even when the LLM API fails.
- **Method (`agents/manager_agent.py`).** A compiled LangGraph state machine with 7 nodes and 2 conditional routers (`_build_graph`, lines 1574–1633). Three LLM passes: **route** (pass 1, `_brain_route`), **evaluate** (pass 2, `_brain_evaluate`), **finalise** (pass 3, `_brain_finalise`), all `temperature=0`.
- **Guardrails & determinism.** `max_routing_loops=8`; recursion limit `(max_routing_loops+2)×4`; on any Brain-API failure the router advances deterministically (`run_research → run_financial → run_sentiment → finalise`) and the evaluator assumes `passed=True, score=50`. The evaluation snapshot is stored in graph state so routing never depends on the memory layer being available.
- **RAG-agent completeness loop (`agents/research_agent.py`).** The checker's completeness criteria include recency (≥ 1 source in 30 days or the latest periodic filing), factual depth, multi-source coverage (≥ 2 tool types), and no-hallucination grounding — with the explicit rule that a genuinely absent disclosure is itself a *complete* answer.
- **RAG evaluation (`rag/evaluation.py`).** The `EvaluationReport.overall_score` weights Faithfulness 0.35, Context Precision 0.25, Context Recall 0.25, Answer Relevance 0.15 — an LLM-as-judge quality metric usable for offline benchmarking.

---

# 4. Infrastructure & Tools

## 4.1 Databases, Vector Store, Cache & External APIs

| Component | Technology | How it is connected |
|-----------|------------|---------------------|
| **Vector store** | Supabase Postgres + `pgvector` | `AlphaVectorStore` (`rag/vector_store.py`) via the Supabase Python client; semantic + FTS search through the `alpha_hybrid_search` RPC. Table `alpha_documents` with an HNSW index on `embedding vector(384)` and a GIN index on the generated `fts tsvector`. **The RPC + table must be created manually** (the DDL is documented inline in `vector_store.py`). |
| **Knowledge graph** | Neo4j (Bolt) | `AlphaGraphStore` (`rag/graph_store.py`) writes idempotent `MERGE` nodes/edges; `hybrid_rag.py` reads via async Cypher. Optional — absent creds disable graph writes and force vector-only retrieval. |
| **Relational persistence** | Supabase Postgres | Tables `long_term_memory` (cognitive memory, DDL in `manager_memory.py`) and `analyses` (per-request results, DDL in `analyze.py`). |
| **LLM** | Anthropic Claude (`claude-haiku-4-5`) | `anthropic.AsyncAnthropic` in the Manager, `anthropic.Anthropic` in the specialists (wrapped in `asyncio.to_thread`), and `AsyncAnthropic` in the graph extractor. |
| **Embeddings model** | `BAAI/bge-small-en-v1.5` | Loaded once via the `AlphaEmbedder` singleton; shared across ingestion, research vector search, and sentiment retrieval. |
| **FinBERT / VADER** | `ProsusAI/finbert`, NLTK VADER | Lazy singletons inside `sentiment_server.py`. |
| **Web / news search** | Tavily API, NewsAPI | `tavily_search.py`, `news_search.py` (async `httpx`). |
| **Filings** | SEC EDGAR | `sec_edgar.py` (both a research variant using `httpx` and a financial variant using `requests`); no API key, mandatory `User-Agent`, retry/backoff on 429. |
| **Market data** | Yahoo Finance | `yahoo_finance.py` via `yfinance`. |
| **Error tracking** | Sentry | `core/observability.init_sentry`; breadcrumbs at each MCP tool call; optional (needs `SENTRY_DSN`). |
| **LLM tracing** | LangSmith | `core/observability.init_langsmith`; `@traceable` decorators on agent nodes; optional (needs `LANGSMITH_API_KEY`). |
| **Live progress** | In-process pub/sub | `core/progress_bus.py` (`asyncio.Queue` per session) → SSE at `api/routes/progress.py`. Not a real broker — single-process only. |
| **Tool transport** | MCP over stdio JSON-RPC | Each specialist launches its server as a `python <server>.py` subprocess. |

## 4.2 Config Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata; `requires-python = ">=3.12"`; the authoritative dependency list (anthropic, fastapi, langgraph, langchain-*, `mcp[cli]`, neo4j, supabase, sentence-transformers, transformers, torch, yfinance, feedparser, nltk, numpy, pandas, ragas, sentry-sdk, pytest*, etc.). |
| `requirements.txt` | Pinned/loose deps grouped by concern (infra, observability, LLM/MCP, vector+graph DB, RAG/NLP, deep learning, data sources, HTTP clients). Adds `uvicorn`, `ragas`, `datasets`, `google-generativeai` not all in pyproject. |
| `pytest.ini` | `pythonpath=.`; `testpaths=tests/unit_tests tests/integration_tests`; `asyncio_mode=auto`; markers `asyncio`, `slow`. |
| `.mcp.json` | Registers only the external **codebase-memory-mcp** dev tool (a local indexing helper) — **not** the project's own three MCP servers, which are launched programmatically by the agents, not via config. |
| `.claude/mcp.json`, `.claude/settings.json` | Claude Code tooling config (codebase-memory server). |
| `docker-compose.yml` | Effectively empty (1 line) — no services are actually defined. |
| `.github/workflows/deploy.yml`, `daily_refresh.yml` | CI/CD deploy pipeline and a scheduled daily-ingestion cron. |
| `frontend/alpha-agent-app.html` | Single-page UI served at `GET /`. |

## 4.3 Environment Variables

| Variable | Required | Default | Used by |
|----------|----------|---------|---------|
| `ANTHROPIC_API_KEY` | ✅ | — | all agents, graph extractor, RAG evaluator |
| `ANTHROPIC_MODEL` | ❌ | `claude-haiku-4-5` | all agents |
| `SUPABASE_URL` | ✅ | — | vector store, memory, hybrid RAG, API |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | — | all Supabase writes (aliased to `SUPABASE_KEY` in Settings; `SUPABASE_KEY` also accepted as a fallback) |
| `NEO4J_URI` / `NEO4J_PASSWORD` | ❌ | — | graph store & graph traversal (graph disabled if absent) |
| `NEO4J_USER` | ❌ | `neo4j` | graph store |
| `TAVILY_API_KEY` | ❌ | — | `tavily_search.py` |
| `NEWSAPI_KEY` | ❌ | — | `news_search.py` |
| `SEC_EDGAR_USER_AGENT` | ❌ | `Financial Analyst Agent admin@example.com` | financial `sec_edgar.py` |
| `SENTRY_DSN` | ❌ | — | Sentry (disabled if absent) |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | ❌ | — / `alpha-agent-node` | LangSmith tracing |
| `APP_ENV` | ❌ | `development` | CORS, Sentry env, docs visibility |
| `MAX_ROUTING_LOOPS` | ❌ | `8` | ManagerAgent guardrail |
| `DEFAULT_USER_ID` | ❌ | `anonymous` | memory (CLI/test) |
| `FEAR_GREED_FINBERT_WEIGHT` / `FEAR_GREED_VADER_WEIGHT` | ❌ | `0.65` / `0.35` | Fear/Greed calculator |
| `SENTIMENT_RETRIEVAL_STAGE1_K` / `SENTIMENT_RETRIEVAL_TOKEN_BUDGET` | ❌ | `20` / `8000` | sentiment retriever |
| `LOG_LEVEL` | ❌ | `INFO` | API logging |

## 4.4 Key Dependencies (grouped)

- **Agents / LLM / MCP:** `anthropic`, `langgraph`, `langchain-*`, `langsmith`, `mcp[cli]`.
- **Vector + graph DB:** `supabase`, `neo4j`.
- **RAG / NLP:** `sentence-transformers`, `langchain-text-splitters`, `nltk`, `numpy`, `pandas`.
- **Deep learning:** `torch`, `transformers`.
- **Data sources:** `yfinance`, `feedparser`, `httpx`, `requests`.
- **Web:** `fastapi`, `uvicorn`, `pydantic`/`pydantic-settings`.
- **Observability:** `sentry-sdk`, `langsmith`.
- **Testing / eval:** `pytest`, `pytest-asyncio`, `pytest-cov`, `ragas`, `datasets`.

## 4.5 Resilience & Fallback Summary

| Failure | Behaviour |
|---------|-----------|
| Claude routing/eval/finalise call fails | Deterministic route advance / `passed=True score=50` / raw-fields fallback report |
| Neo4j down | Graph writes/reads disabled; RAG runs vector-only |
| Supabase pgvector down | `rag_vector_search` returns empty with a warning; hybrid degrades to graph-only |
| Zero social chunks | SentimentAgent retries once with a broader query; models return Neutral defaults |
| MCP tool failure | Error marker appended to context/extraction_errors; loop continues |
| GPU OOM | Embedder/FinBERT fall back to CPU and retry |
| NaN in financials | `_safe_float` → CAGR filter → `_sanitize_nans` (three layers) keep JSON valid |
| Routing loop runaway | `abort` node at `loop ≥ max_routing_loops`; long-term memory still persisted |

## 4.6 Known Limitations (from code comments)

- **No authentication** — `user_id` is taken from the request body (`analyze.py`); any caller can address any user's long-term memory until JWT is added.
- **Graph depth capped at 3 hops** — deeper supply-chain/geopolitical chains are truncated (`hybrid_rag.py:183`).
- **`alpha_hybrid_search` RPC is not auto-created** — must be deployed via the Supabase SQL editor.
- **yfinance freshness / rate limits** — unofficial endpoint; failures surface as `extraction_errors`.
- **`revenue_growth_ttm` is a naming quirk** — it is the most-recent-quarter YoY, not a true TTM figure (tagged as such).
- **DCF simplifications** — CAPM-only discount rate, no net-debt adjustment, single-stage-then-terminal growth; Monte-Carlo randomises growth only.
- **`scheduler/daily_refresh.py` is a stub** and `docker-compose.yml` is empty — the daily-refresh workflow relies on `rag/ingestion.py`/`seed.py` directly.

---

# 5. Summary Table

| File | Role | Key Class / Function | Main Dependencies |
|------|------|----------------------|-------------------|
| `agents/state.py` | All state contracts | `SharedManagerState`, `*AgentState`, `ManagerGraphState`, `EvaluationSnapshot` | typing, operator |
| `agents/manager_agent.py` | Central orchestrator (7-node LangGraph, 3 Brain passes) | `ManagerAgent`, `_brain_route/_brain_evaluate/_brain_finalise`, `_build_graph` | anthropic, langgraph, memory, progress_bus, validate_period_consistency |
| `agents/research_agent.py` | Research specialist (brain→executor→checker loop) | `ResearchAgent`, `_brain_node/_executor_node/_checker_node`, `_ensure_ticker_argument` | anthropic, langgraph, mcp, context_synthesizer |
| `agents/financial_agent.py` | Quant specialist (Brain→Executors→Checker) | `FinancialAnalystAgent`, `_execute_data_extraction/_execute_ratio_computation/_check_data_quality`, `_sanitize_nans` | anthropic, mcp |
| `agents/sentiment_agent.py` | Sentiment specialist (Brain→Executor→Brain) | `SentimentAgent`, `_brain_plan/_execute_sentiment_pipeline/_brain_analyze` | anthropic, mcp |
| `memory/manager_memory.py` | Two-level cognitive memory | `ManagerMemory`, `ShortTermMemory`, `LongTermMemory`, `EvaluationFeedback` | supabase |
| `rag/ingestion.py` | 5-stage ETL orchestrator | `run_ingestion_pipeline` | loader, processor, embedding_manager, vector_store, graph_store |
| `rag/loader.py` | Multi-source loader + ticker validation | `AlphaLoader`, `_mentions_ticker`, `_to_utc_iso8601` | yfinance, feedparser |
| `rag/processor.py` | Chunking + dedup | `AlphaProcessor`, `ProcessedChunk`, `_sha256` | langchain-text-splitters |
| `rag/embedding_manager.py` | Embedding singleton | `AlphaEmbedder`, `get_embedder`, `_encode_batch` | sentence-transformers, torch, numpy |
| `rag/vector_store.py` | pgvector + hybrid SQL RPC | `AlphaVectorStore`, `hybrid_search`, `alpha_hybrid_search` (SQL) | supabase |
| `rag/graph_store.py` | LLM graph extraction + Neo4j MERGE | `AlphaGraphStore`, `_extract_json_block`, `_merge_entity/_merge_relation` | anthropic, neo4j |
| `rag/retriever.py` | 5-stage retrieval pipeline | `AlphaRetriever`, `_rerank_by_freshness/_diversity_filter/_apply_token_budget` | embedding_manager, vector_store |
| `rag/hybrid_rag.py` | 3 RAG tools + RRF fusion | `rag_vector_search/rag_graph_traverse/rag_hybrid_query`, `_rrf`, `_weighted` | supabase, neo4j, retriever |
| `rag/evaluation.py` | LLM-as-judge RAG metrics | `AlphaEvaluator`, `EvaluationReport` | anthropic |
| `rag/seed.py` | Ingestion CLI | `main` | ingestion |
| `tools/research_tools/research_server.py` | Research MCP server (8 tools) | `list_tools`, `call_tool`, `_normalize_sections` | mcp, tavily/news/sec_edgar/rag |
| `tools/research_tools/tavily_search.py` | Tavily client | `tavily_search` | httpx |
| `tools/research_tools/news_search.py` | NewsAPI client + query engineering | `news_search`, `_extract_terms/_build_boolean_query` | httpx |
| `tools/research_tools/sec_edgar.py` | EDGAR search/filing (research) | `sec_edgar_search/sec_edgar_filing`, `_sanitize_query` | httpx |
| `tools/research_tools/comprehensive_analysis.py` | Concurrent news + filing | `comprehensive_analysis` | tavily_search, sec_edgar |
| `tools/research_tools/context_synthesizer.py` | Post-loop research compression | `synthesize_research_context` | anthropic, langsmith |
| `tools/financial_tools/financial_server.py` | Financial MCP server (20 tools) | `@mcp.tool()` wrappers | FastMCP, yahoo_finance, sec_edgar, ratio calculator |
| `tools/financial_tools/yahoo_finance.py` | yfinance wrapper | `get_financial_ratios/get_revenue_growth/get_peer_comparison`, `_safe_float`, `_fiscal_quarter_label` | yfinance |
| `tools/financial_tools/sec_edgar.py` | EDGAR CIK/filings/XBRL (financial) | `get_cik/list_filings/get_filing_text/get_xbrl_financials`, `_extract_annual` | requests |
| `tools/financial_tools/financial_ratio_calculator.py` | Pure-math ratios, score, DCF | `composite_financial_score`, `cagr`, `discounted_cash_flow`, `dcf_scenario_range`, `dcf_monte_carlo` | math, random |
| `tools/sentiment_tools/sentiment_server.py` | Sentiment MCP server (4 tools) | `list_tools`, `call_tool`, lazy `_get_*` singletons | mcp, retriever, finbert, vader, fear_greed |
| `tools/sentiment_tools/finbert_analyzer.py` | FinBERT deep NLP | `FinBertSentimentAnalyzer`, `analyze`, `_infer_batch` | transformers, torch |
| `tools/sentiment_tools/vader_scorer.py` | VADER lexical baseline | `VaderLexiconScorer`, `score`, `_compound_label` | nltk |
| `tools/sentiment_tools/fear_greed_calculator.py` | Weighted sentiment fusion | `FearGreedIndexCalculator`, `calculate`, `_score_to_label` | finbert_analyzer, vader_scorer |
| `api/main.py` | FastAPI app + lifespan singletons | `lifespan`, exception handlers, `/health`, `/readiness`, `/` | fastapi, supabase, agents, memory |
| `api/config.py` | Settings | `Settings`, `get_settings`, `validate_settings` | pydantic-settings |
| `api/routes/analyze.py` | `POST /analyze` | `analyze`, `AnalyzeRequest/Response`, `_persist_analysis` | fastapi, manager_agent, memory |
| `api/routes/progress.py` | SSE progress stream | `stream_progress` | fastapi/starlette, progress_bus |
| `api/dependencies.py` | DI factories | `get_user_id`, `get_manager_memory` | fastapi, memory |
| `api/core/exceptions.py` | Exception hierarchy | `AlphaAgentError` + subclasses | uuid |
| `core/observability.py` | Sentry + LangSmith bootstrap | `init_sentry/init_langsmith`, `sentry_enabled/langsmith_enabled` | sentry-sdk, langsmith |
| `core/progress_bus.py` | In-process pub/sub for SSE | `publish`, `subscribe`, `get_queue`, `session_from_shared` | asyncio |
| `core/error_handler.py` | Sentry breadcrumb decorator/CM | `with_error_reporting` | observability |
| `evaluation/validate_period_consistency.py` | Period-consistency QA | `check_narration_vs_period`, `PeriodRegistry`, `check_section_consistency` | re, json |
| `evaluation/run_ragas.py` | RAGAS eval runner | (script) | ragas, datasets |

---

## Closing Note

AgenticAlpha is a coherent, production-oriented realisation of a **multi-agent, tool-using, memory-augmented financial analyst**. Its thesis-relevant contributions cluster in four areas: (1) a **contract-based multi-agent architecture** with three-pass LLM orchestration and deterministic fallbacks; (2) a **hybrid RAG system** fusing dense (pgvector/HNSW) and sparse (Postgres FTS) retrieval via Reciprocal Rank Fusion, plus an LLM-built Neo4j knowledge graph with idempotent weight-averaging; (3) a **quantitative valuation layer** with a weighted composite health score and a three-tier DCF (point, bear/base/bull scenario, Monte-Carlo) that honestly exposes growth-rate sensitivity; and (4) a **sentiment-fusion index** combining deep-NLP (FinBERT) and lexical (VADER) signals. A recurring engineering theme — visible throughout the code comments — is **provenance and faithfulness**: period-tagging every ratio, keeping raw chunks alongside summaries for numeric traceability, and validating that narrated numbers match their true reporting period.

*Document generated by static, read-only analysis of the AgenticAlpha source tree.*
