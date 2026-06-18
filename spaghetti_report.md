# Spaghetti Code Analysis Report

> **Scope:** All 41 source files across `api/`, `agents/`, `core/`, `rag/`, `tools/`, `memory/`
> **Method:** Static analysis — no files were modified.

---

## Executive Summary

| Category | Issues Found | Severity |
|---|---|---|
| Dependency & Coupling | 8 | 🔴 Critical |
| State & Side Effects | 6 | 🔴 Critical |
| Testability Blockers | 6 | 🔴 Critical |
| Error Handling | 3 | 🟠 High |
| Dead Code | 4 | 🟡 Medium |
| Structural Problems | 5 | 🟠 High |
| Boundary Problems | 2 | 🟡 Medium |
| **TOTAL** | **34** | |

**Severity key:**
🔴 Critical — breaks in production or makes testing impossible
🟠 High — major maintenance risk
🟡 Medium — code smell, low immediate risk

---

## Critical Issues (Fix First)

---

### [C-1] Broken module-level imports in `fear_greed_calculator.py`

- **File:** `tools/sentiment_tools/fear_greed_calculator.py` lines 56–57 and 280–281
- **Category:** Dependency & Coupling
- **Problem:** The file has two top-level imports that reference package paths that do not exist:
  ```python
  from tools.finbert_analyzer import FinBertResult   # line 56
  from tools.vader_scorer     import VaderResult      # line 57
  ```
  These files live at `tools/sentiment_tools/finbert_analyzer.py` and `tools/sentiment_tools/vader_scorer.py`. There is no `tools/finbert_analyzer.py` or `tools/vader_scorer.py`. The same wrong paths are repeated as local imports inside `calculate_from_dict()` at lines 280–281.
- **Impact:** `ModuleNotFoundError` is raised the moment anything imports `fear_greed_calculator`. The entire sentiment pipeline (`SentimentAgent`, `sentiment_server.py`, `FearGreedIndexCalculator`) is non-functional in its current state. This is a silent production outage.
- **Evidence:**
  ```python
  from tools.finbert_analyzer import FinBertResult   # WRONG — file does not exist at this path
  from tools.vader_scorer     import VaderResult      # WRONG — file does not exist at this path
  ```

---

### [C-2] `core/` layer imports from `api/` — inverted dependency

- **File:** `core/observability.py` line 62
- **Category:** Dependency & Coupling
- **Problem:** `core/` is the foundational observability layer that every other module imports. Inside `init_sentry()`, it does a runtime import of `api.config.settings`:
  ```python
  from api.config import settings   # line 62, inside init_sentry()
  ```
  The `core/` layer must not depend on `api/`. This inverts the dependency hierarchy. The three MCP server subprocesses (`research_server.py`, `financial_server.py`, `sentiment_server.py`) all call `init_sentry()` at module level — but they do **not** run the FastAPI application and have no `api/` package in their import path. Every MCP server will fail to initialise Sentry because `api.config` does not resolve in their process context.
- **Impact:** Sentry is silently broken in all three MCP server subprocesses. Any future use of `core/` outside the `api/` context will also fail.
- **Evidence:**
  ```python
  def init_sentry() -> bool:
      ...
      try:
          import sentry_sdk
          from api.config import settings  # ← core imports api — inverted layer
          sentry_sdk.init(dsn=dsn, environment=settings.APP_ENV, ...)
  ```

---

### [C-3] `validate_settings()` calls `sys.exit(1)` inside library code

- **File:** `api/config.py` line 94
- **Category:** Testability Blockers / State & Side Effects
- **Problem:** `validate_settings()` is a library function called from the FastAPI lifespan. It terminates the entire Python process with `sys.exit(1)` on invalid configuration rather than raising an exception. Calling `sys.exit()` inside business logic is an anti-pattern: it cannot be caught, cannot be tested without patching the builtin, and provides no programmatic way for callers to react.
- **Impact:** Unit tests cannot assert on *which* key was missing — the process simply dies. Integration tests need to patch `sys.exit` globally. The error message is logged but never surfaced as a structured exception.
- **Evidence:**
  ```python
  def validate_settings() -> None:
      ...
      if missing:
          log.critical("Missing required environment variables: %s", ", ".join(missing))
          sys.exit(1)   # ← kills the process, untestable without patching builtins
  ```

---

### [C-4] `LongTermMemory.__init__` performs I/O unconditionally

- **File:** `memory/manager_memory.py` line 431
- **Category:** State & Side Effects
- **Problem:** The `LongTermMemory` constructor calls `self.load()` as its final line. `load()` executes a live Supabase `SELECT` query. This means: instantiating `LongTermMemory`, `ManagerMemory`, or any code path that creates these objects makes a real database call without any signal in the signature or type system. The caller has no way to opt out.
  Additionally, `ManagerMemory.__init__` (line 705) instantiates `LongTermMemory` — so creating a `ManagerMemory` always hits the database.
- **Impact:** Every unit test that creates a `ManagerMemory` makes a network call. Any environment without Supabase credentials raises immediately at object construction, making local testing without mocks impossible.
- **Evidence:**
  ```python
  class LongTermMemory:
      def __init__(self, user_id, supabase_client=None, ...):
          ...
          self.load()   # line 431 — live DB SELECT on every instantiation
  ```

---

### [C-5] `rag/ingestion.py` calls `init_sentry()` at module level

- **File:** `rag/ingestion.py` line 28
- **Category:** State & Side Effects
- **Problem:** `init_sentry()` is called at the top level of the module, outside any function:
  ```python
  load_dotenv(...)
  init_sentry()   # line 28 — runs on every import
  ```
  Importing `rag.ingestion` anywhere — including test files — unconditionally attempts to initialise Sentry. The same pattern exists in the three MCP server entry points (`research_server.py`, `financial_server.py`, `sentiment_server.py`). Module-level side effects are a well-known Python anti-pattern.
- **Impact:** Test files that import `rag.ingestion` trigger Sentry initialisation. In CI environments without `SENTRY_DSN`, this generates warning logs on every test run. In environments with `SENTRY_DSN` set, test exceptions are sent to the production error tracker.
- **Evidence:**
  ```python
  load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
  init_sentry()   # ← module-level side effect on import
  logging.basicConfig(...)
  ```

---

### [C-6] `AlphaGraphStore.__init__` opens network connections in constructor

- **File:** `rag/graph_store.py` lines 142–157
- **Category:** State & Side Effects / Testability Blockers
- **Problem:** The constructor immediately:
  1. Creates an `anthropic.Anthropic()` client (line 142)
  2. Opens a Neo4j `GraphDatabase.driver()` connection (line 156)
  3. Calls `self._driver.verify_connectivity()` — a live network call (line 157)

  All of this happens synchronously inside `__init__`. There is no way to construct the object without these side effects. The Anthropic API key is read directly from `os.environ["ANTHROPIC_API_KEY"]` without a fallback or injection.
- **Impact:** Cannot be unit tested without a live Neo4j instance and valid Anthropic credentials. Any environment without `ANTHROPIC_API_KEY` raises a `KeyError` (not a `ValueError`) with no descriptive message.
- **Evidence:**
  ```python
  def __init__(self, anthropic_api_key=None, neo4j_uri=None, ...):
      self._claude = anthropic.Anthropic(
          api_key=anthropic_api_key or os.environ["ANTHROPIC_API_KEY"]  # KeyError if missing
      )
      ...
      self._driver = GraphDatabase.driver(uri, auth=(user, password))
      self._driver.verify_connectivity()  # live network call in constructor
  ```

---

## High Priority Issues

---

### [H-1] `ManagerAgent.__init__` creates `AsyncAnthropic()` client inline — no injection seam

- **File:** `agents/manager_agent.py` line 285
- **Category:** Testability Blockers / Dependency & Coupling
- **Problem:** The constructor hardwires the Anthropic client:
  ```python
  self._llm = anthropic.AsyncAnthropic()   # line 285
  ```
  There is no `client` parameter, no factory method, and no way to inject a mock. All three brain methods (`_brain_route`, `_brain_evaluate`, `_brain_finalise`) use `self._llm` directly.
- **Impact:** Every test of ManagerAgent routing logic must make a real Anthropic API call (costs money, is non-deterministic) or patch `anthropic.AsyncAnthropic` at the module level. This is the core business logic class — it must be testable.

---

### [H-2] Three MCP tool agents have zero injection seam for the subprocess

- **File:** `agents/research_agent.py`, `agents/financial_agent.py`, `agents/sentiment_agent.py`
- **Category:** Testability Blockers
- **Problem:** Each agent's `run()` method launches a real `stdio` subprocess (`mcp.client.stdio.stdio_client`) to communicate with the tool server. There is no interface, no factory, and no `mcp_client` parameter that would allow injecting a mock transport. The subprocess path is hardcoded to the actual server script.
- **Impact:** The entire agent layer can only be tested as an integration test with all three tool-server processes running. There is no unit test path for any routing or decision logic in these agents.

---

### [H-3] `logging.basicConfig()` called inside library modules

- **File:** `agents/manager_agent.py` line 83, `rag/ingestion.py` line 30
- **Category:** Structural Problems
- **Problem:** Both files call `logging.basicConfig()` at module level. `basicConfig()` is an application-level call that should only be made once by the entry point (e.g., `api/main.py`). When called inside a library module, it hijacks the application's logging configuration. Python's `basicConfig()` is idempotent only on the first call — subsequent calls (including the one in `api/main.py`) are silently ignored if any handler has already been attached.
- **Impact:** Log format, level, and stream configuration in `api/main.py` may be silently overridden by whichever library module is imported first. Log output from the production API may use `manager_agent.py`'s format instead of the application format.
- **Evidence:**
  ```python
  # agents/manager_agent.py line 83 — library module, not entry point
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
      stream=sys.stderr,
  )
  ```

---

### [H-4] `AlphaEmbedder` singleton has no reset mechanism

- **File:** `rag/embedding_manager.py` lines 56–57, 59–72
- **Category:** Testability Blockers
- **Problem:** The singleton is stored in a module-level global `_INSTANCE`. Once loaded (including by a test), it persists for the entire process lifetime. There is no `reset()`, no `_INSTANCE = None` escape hatch, and no way to inject a different model or device configuration.
- **Impact:** Tests that import anything from `rag/` that transitively calls `get_embedder()` will load the 384-dim BAAI model onto the test machine's memory. Tests cannot configure a lightweight mock model. The 200MB model download happens silently on first test run in CI.
- **Evidence:**
  ```python
  _INSTANCE: Optional["AlphaEmbedder"] = None   # process-level, never reset

  def get_embedder(...) -> "AlphaEmbedder":
      global _INSTANCE
      if _INSTANCE is None:
          with _LOCK:
              if _INSTANCE is None:
                  _INSTANCE = AlphaEmbedder(...)
      return _INSTANCE
  ```

---

### [H-5] `_sentry_ok` is a module-level mutable boolean shared across all requests

- **File:** `api/main.py` lines 64, 87
- **Category:** State & Side Effects
- **Problem:** `_sentry_ok` is declared as a module-level global (`_sentry_ok: bool = False`) and then mutated inside the async `lifespan` context manager using a `global` statement. This boolean is later read by both exception handlers (`alpha_agent_exception_handler` and `unhandled_exception_handler`) on every single request. It creates implicit shared mutable state that is not thread-safe.
- **Impact:** In a multi-worker deployment (e.g., multiple uvicorn workers), each worker has its own module state, so this can produce inconsistent Sentry reporting. The pattern of using a `global` inside an `async` context manager is also a smell — it violates the "app.state" convention that the same file correctly uses for every other shared resource (Supabase client, ManagerAgent).
- **Evidence:**
  ```python
  _sentry_ok: bool = False   # module-level mutable global

  async def lifespan(app: FastAPI):
      global _sentry_ok          # mutated inside async lifecycle
      _sentry_ok = init_sentry()
      ...

  @app.exception_handler(AlphaAgentError)
  async def alpha_agent_exception_handler(request, exc):
      if _sentry_ok:             # read on every request
          ...
  ```

---

### [H-6] `AlphaGraphStore` uses synchronous Anthropic client in an async server

- **File:** `rag/graph_store.py` line 142
- **Category:** Structural Problems
- **Problem:** `AlphaGraphStore` uses `anthropic.Anthropic()` (the synchronous client), not `anthropic.AsyncAnthropic()`. When `graph_store.extract_batch()` is called from the ingestion pipeline (which runs in an async context via `asyncio.to_thread()` or directly), the synchronous HTTP call to the Anthropic API **blocks the event loop** for the entire duration of the Claude response (typically 1–5 seconds per document). For a batch of 50 documents, this is 50–250 seconds of event-loop blocking.
- **Impact:** Under concurrent requests, the API server becomes unresponsive during graph extraction runs. Response times for all other requests spike.

---

### [H-7] `AlphaProcessor._seen` grows without bound — production memory leak

- **File:** `rag/processor.py` line 99
- **Category:** Structural Problems
- **Problem:** The `_seen` dict is described in comments as an "in-memory dedup store" with the note "In production, replace with a Redis / Postgres lookup." This replacement never happened. The dict accumulates one entry per unique URL ever ingested in the process lifetime with no eviction policy, TTL, or maximum size.
- **Impact:** A production ingestion server that runs continuously will grow `_seen` to millions of entries. Each entry is two SHA256 hex strings (~130 bytes). At 1M URLs: ~130 MB of heap. No OOM protection exists.
- **Evidence:**
  ```python
  # In production, replace with a Redis / Postgres lookup.
  self._seen: dict[str, str] = {}   # url_hash -> content_hash — grows forever
  ```

---

## Medium Priority Issues

---

### [M-1] `settings` singleton evaluated at module import time

- **File:** `api/config.py` line 70
- **Category:** State & Side Effects
- **Problem:** `settings = Settings()` is executed at module level. Any module that does `from api.config import settings` triggers environment-variable loading at import time. Tests that need to vary configuration must patch `api.config.settings` in-place after import — a fragile approach that can leak between tests if not properly restored.
- **Evidence:**
  ```python
  settings = Settings()   # line 70 — evaluated on import
  ```

---

### [M-2] `is_prod` flag evaluated at import time before lifespan runs

- **File:** `api/main.py` line 141
- **Category:** State & Side Effects
- **Problem:** `is_prod = settings.APP_ENV == "production"` is evaluated at module-level, before the FastAPI `lifespan` context manager runs `validate_settings()`. If `settings.APP_ENV` is patched for testing after import, `is_prod` will not reflect the patch. More importantly, the FastAPI `app` object is constructed with `docs_url=None if is_prod else "/docs"` — the docs URL is hardwired at app creation time, not at request time.
- **Evidence:**
  ```python
  is_prod = settings.APP_ENV == "production"   # line 141 — evaluated on import

  app = FastAPI(
      docs_url=None if is_prod else "/docs",   # frozen at import time
  )
  ```

---

### [M-3] `unhandled_exception_handler` leaks `str(exc)` to clients

- **File:** `api/main.py` line 233
- **Category:** Boundary Problems
- **Problem:** The catch-all exception handler returns `str(exc)` directly in the response body as `"detail"`. In Python, `str(exc)` can include internal paths, class names, database connection strings, and stack frames depending on the exception type.
- **Impact:** In production, this can expose: SQL error messages containing schema names, file-system paths, environment variable names, and internal class hierarchies. This is a CWE-209 information exposure vulnerability.
- **Evidence:**
  ```python
  return JSONResponse(
      status_code=500,
      content={
          "error":   "INTERNAL_ERROR",
          "message": "An unexpected error occurred.",
          "detail":  str(exc),   # ← may expose internal details in production
      },
  )
  ```

---

### [M-4] CORS configured with `allow_origins=["*"]` AND `allow_credentials=True`

- **File:** `api/main.py` lines 162–168
- **Category:** Boundary Problems
- **Problem:** The CORS middleware uses `allow_origins=["*"]` in development mode together with `allow_credentials=True`. The W3C CORS specification explicitly forbids this combination — browsers will reject credentialed requests to wildcard-origin servers. Moreover, the `allow_credentials=True` flag carries over to production mode even when `ALLOWED_ORIGINS` is properly set, meaning cookies and auth headers are unconditionally allowed from any listed origin without per-origin validation.
- **Evidence:**
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"] if settings.APP_ENV == "development" else settings.ALLOWED_ORIGINS,
      allow_credentials=True,   # ← always True regardless of environment
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```

---

### [M-5] `rag/ingestion.py` reads env vars directly, bypassing `Settings`

- **File:** `rag/ingestion.py` lines 51–55
- **Category:** Dependency & Coupling
- **Problem:** The ingestion pipeline reads `os.environ.get("SUPABASE_URL")` and `os.environ.get("SUPABASE_SERVICE_ROLE_KEY")` directly, bypassing the central `api.config.Settings` object. The rest of the codebase uses `settings.SUPABASE_URL` and `settings.SUPABASE_KEY`. Worse, `ingestion.py` checks for a second alias `SUPABASE_SERVICE_KEY` that does not exist in `Settings`, creating a config inconsistency.
- **Impact:** Configuration management is split across two systems. A developer who changes the env var name in `Settings` will break `ingestion.py` silently. The `Settings` validation check in `validate_settings()` does not protect the ingestion pipeline.
- **Evidence:**
  ```python
  supabase_url = os.environ.get("SUPABASE_URL")
  supabase_key = (
      os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
      or os.environ.get("SUPABASE_SERVICE_KEY")   # ← alias not in Settings
  )
  ```

---

### [M-6] `_VALID_ACTIONS` duplicated between constant and prompt string

- **File:** `agents/manager_agent.py` lines 98–102 and 108–145
- **Category:** Structural Problems
- **Problem:** The 8 valid routing actions are defined twice: once as a `frozenset` for runtime validation (`_VALID_ACTIONS`), and again hard-coded as a list inside the `_ROUTER_SYSTEM_PROMPT` string. These two definitions can diverge silently — adding a new action to `_VALID_ACTIONS` but forgetting to update the prompt means the Brain never learns about it, and vice versa.
- **Evidence:**
  ```python
  _VALID_ACTIONS: frozenset[str] = frozenset({
      "run_research", "run_financial", "run_sentiment",     # defined here
      ...
  })

  _ROUTER_SYSTEM_PROMPT = """
  Available actions:
    "run_research"   — ...                                  # also defined here
    ...
  """
  ```

---

### [M-7] `_VADER_LOADED` and FinBERT model globals have no reset mechanism

- **File:** `tools/sentiment_tools/vader_scorer.py` lines 64–65; `tools/sentiment_tools/finbert_analyzer.py` lines 50–53
- **Category:** Testability Blockers
- **Problem:** Both files store model state in module-level globals (`_VADER_LOADED`, `_TOKENIZER`, `_MODEL`, `_DEVICE`). Once set, these globals persist for the process lifetime with no `reset()` mechanism. Tests cannot simulate first-load behavior, cannot test the download path for VADER, and cannot inject a mock model for FinBERT.
- **Evidence:**
  ```python
  # vader_scorer.py
  _VADER_LOCK:   threading.Lock = threading.Lock()
  _VADER_LOADED: bool           = False   # never reset between tests

  # finbert_analyzer.py
  _TOKENIZER: Optional[BertTokenizer]                = None
  _MODEL:     Optional[BertForSequenceClassification] = None   # never reset
  ```

---

### [M-8] `AlphaRetriever.__init__` calls `get_embedder()` as hidden default

- **File:** `rag/retriever.py` line 57
- **Category:** Dependency & Coupling
- **Problem:** `self.embedder = embedder or get_embedder()` — when `embedder` is `None`, the constructor silently loads the 384-dim sentence-transformer model (200MB). This is a hidden side effect with no indication in the class signature or docstring that construction may trigger model loading.
- **Evidence:**
  ```python
  def __init__(self, vector_store, embedder=None, ...):
      self.embedder = embedder or get_embedder()   # silent model load if no embedder
  ```

---

## Dead Code Registry

| File | Symbol | Type | Evidence |
|---|---|---|---|
| `tools/sentiment_tools/local_social_retriever.py` | `LocalSocialDataRetriever` | Legacy class, entire file | Docstring states "sentiment_server.py now calls AlphaRetriever.retrieve_raw() directly; this class is legacy". Never imported by any active module. |
| `tools/financial_tools/financial_ratio_calculator.py:110` | `_label()` call in `price_to_earnings` | Dead call, result discarded | `_label()` is called on line 110 and its return value is stored in `interp`, but `interp` is immediately overwritten by the `if pe < 0 / elif pe < 15 / elif pe <= 30 / else` chain on lines 114–121. The `_label()` call on line 110 is never used. |
| `agents/manager_agent.py:212` | `_extract_chunk_text` legacy branch | Unreachable legacy branch | Comment says "handles both legacy plain-string chunks and dict chunks". The `isinstance(chunk, str)` branch handles a format that no current code path produces. |
| `tools/sentiment_tools/finbert_analyzer.py` header comment | File path in docstring | Wrong metadata | The module docstring says `tools/finbert_analyzer.py` but the actual path is `tools/sentiment_tools/finbert_analyzer.py`. Will mislead `grep` and IDE navigation. |

---

## Testability Blockers Summary

| Module | Symbol | Blocker Reason |
|---|---|---|
| `memory/manager_memory.py` | `LongTermMemory` | Constructor calls `self.load()` — live Supabase SELECT on every instantiation |
| `memory/manager_memory.py` | `ManagerMemory` | Composes `LongTermMemory` — inherits the DB-on-construction blocker |
| `agents/manager_agent.py` | `ManagerAgent` | `AsyncAnthropic()` created inline in `__init__` — no injection seam for LLM client |
| `agents/research_agent.py` | `ResearchAgent` | Launches real stdio subprocess in `run()` — no mock transport injection |
| `agents/financial_agent.py` | `FinancialAnalystAgent` | Same as ResearchAgent — real subprocess, no injection |
| `agents/sentiment_agent.py` | `SentimentAgent` | Same as above |
| `rag/graph_store.py` | `AlphaGraphStore` | Constructor opens Neo4j connection and requires `ANTHROPIC_API_KEY` — live I/O |
| `rag/embedding_manager.py` | `get_embedder` / `AlphaEmbedder` | Module-level singleton with no reset — loads 200MB model on first call |
| `tools/sentiment_tools/finbert_analyzer.py` | `FinBertSentimentAnalyzer` | Module-level model globals — loads 400MB FinBERT model, no reset |
| `api/config.py` | `validate_settings` | Calls `sys.exit(1)` — cannot assert on missing keys without patching builtins |
| `tools/sentiment_tools/fear_greed_calculator.py` | `FearGreedIndexCalculator` | Module-level imports fail with `ModuleNotFoundError` before any test can run |

---

## Recommended Fix Order

1. **Fix broken imports in `fear_greed_calculator.py`** (C-1)
   Change `from tools.finbert_analyzer import ...` to `from tools.sentiment_tools.finbert_analyzer import ...` in both the module-level imports and the `calculate_from_dict()` local imports. This is a one-line fix that restores the entire sentiment pipeline.

2. **Remove `from api.config import settings` from `core/observability.py`** (C-2)
   Pass `APP_ENV` as a parameter to `init_sentry()`, or read it directly from `os.environ.get("APP_ENV", "development")`. The `core/` layer must have zero dependencies on `api/`.

3. **Replace `sys.exit(1)` in `validate_settings()` with a raised exception** (C-3)
   Define a `ConfigurationError(RuntimeError)` in `api/core/exceptions.py`. Raise it from `validate_settings()`. Let `lifespan` in `api/main.py` catch it and call `sys.exit(1)` as the application entry point's responsibility.

4. **Decouple `LongTermMemory.__init__` from `self.load()`** (C-4)
   Remove `self.load()` from `__init__`. Require callers to call `.load()` explicitly, or provide a `LongTermMemory.create(user_id, client)` classmethod factory. This makes every unit test that creates `ManagerMemory` work without a database.

5. **Move `init_sentry()` calls from module-level to `if __name__ == "__main__"` blocks** (C-5)
   In `rag/ingestion.py` and all three MCP server files, move `init_sentry()` into the `if __name__ == "__main__"` block or into an explicit `setup()` function. Module imports must not have network side effects.

6. **Add an `llm_client` parameter to `ManagerAgent.__init__`** (H-1)
   Accept an optional `anthropic.AsyncAnthropic` instance and fall back to creating one internally only when `None` is passed. This single change enables the entire test suite for the ManagerAgent without live API calls.

7. **Add MCP transport injection to all three specialist agents** (H-2)
   Define an `McpTransport` protocol and accept it as an `__init__` parameter. The default implementation launches the real subprocess; tests inject a `MockMcpTransport`.

8. **Remove `logging.basicConfig()` calls from library modules** (H-3)
   Delete the `logging.basicConfig()` call from `agents/manager_agent.py:83` and `rag/ingestion.py:30`. Library modules should only call `logging.getLogger(__name__)`.

9. **Delete `tools/sentiment_tools/local_social_retriever.py`** (Dead Code)
   The file is self-described as legacy and unused. Remove it to reduce the maintenance surface.

10. **Add a size cap to `AlphaProcessor._seen`** (H-7)
    Cap `_seen` at a configurable maximum size (e.g., 100,000 entries) with LRU eviction, or document that the class is intended for single-run usage only and must be re-created each pipeline run.

11. **Replace `"detail": str(exc)` in the catch-all handler with a sanitised message** (M-3)
    In production mode (`settings.APP_ENV == "production"`), return a generic string like `"An internal error occurred. Refer to trace_id in logs."`. In development, `str(exc)` is acceptable.

12. **Centralise all env-var reading in `rag/ingestion.py` through `Settings`** (M-5)
    Import `settings` from `api.config` and use `settings.SUPABASE_URL` and `settings.SUPABASE_KEY` instead of the direct `os.environ` reads. Remove the `SUPABASE_SERVICE_KEY` alias that does not exist in `Settings`.

---

## Assumptions & Limitations

- **Static analysis only.** No runtime execution was performed. Circular import issues (e.g., `core → api → core` potential cycles) were not verified by actually running the import chain.
- **`AlphaRetriever.retrieve_raw()`** is referenced in `local_social_retriever.py`'s docstring but not confirmed to exist in the version of `rag/retriever.py` analysed. If this method is absent, `LocalSocialDataRetriever` would fail even if it were called.
- **MCP server stdio transport** details were not fully traced — the blocking vs. non-blocking behaviour of `asyncio.to_thread()` wrappers inside the sentiment server was inferred from the pattern described in docstrings.
- **`agents/state.py` TypedDicts** were not flagged for structural issues as TypedDicts are intentionally thin data contracts with no behaviour.
- **The `rag/test_rag_pipeline.py`** test file stubs out `torch`, `sentence_transformers`, and `supabase` before importing — this is the correct mitigation pattern for the singleton testability issues identified in H-4, confirming the team is aware of the problem but has not solved it at the source.
