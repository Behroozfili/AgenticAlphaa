# AgenticAlpha — Professional Pytest Test Suite

---

## Phase 1: Project Analysis

### Codebase Summary

41 source files across 8 directories. Three distinct runtime processes: FastAPI server (`api/`), three MCP stdio subprocesses (`tools/*/`), and a data-ingestion pipeline (`rag/ingestion.py`).

### External Dependencies Detected

| Category | Dependency | Modules Affected |
|---|---|---|
| LLM | Anthropic SDK | agents/* |
| LLM tracing | LangSmith | agents/*, rag/hybrid_rag.py |
| Error tracking | Sentry SDK | core/observability.py + all servers |
| Vector DB | Supabase (pgvector) | rag/vector_store.py, memory/ |
| Graph DB | Neo4j | rag/graph_store.py |
| ML model | ProsusAI/finbert + PyTorch | tools/sentiment_tools/finbert_analyzer.py |
| NLP lexicon | NLTK VADER | tools/sentiment_tools/vader_scorer.py |
| Embeddings | BAAI/bge-small-en-v1.5 | rag/embedding_manager.py |
| Finance data | yfinance | tools/financial_tools/yahoo_finance.py |
| News data | Tavily, NewsAPI | tools/research_tools/* |
| Graph framework | LangGraph | agents/manager_agent.py, agents/research_agent.py |

### Dead Code / Issues Identified

- `tools/sentiment_tools/local_social_retriever.py` — `LocalSocialDataRetriever` is declared legacy in docstring; `sentiment_server.py` no longer uses it.
- `tools/sentiment_tools/fear_greed_calculator.py:56–57` — **Bug**: top-level imports `from tools.finbert_analyzer import FinBertResult` and `from tools.vader_scorer import VaderResult` use wrong package paths (files live at `tools.sentiment_tools.*`; this will raise `ModuleNotFoundError` on import).
- `core/observability.py` — module-level `_sentry_enabled`/`_langsmith_enabled` flags are process-global; test isolation requires explicit patching.
- `api/config.py:94` — `validate_settings()` calls `sys.exit(1)` directly; hard to assert in tests without patching `sys.exit`.
- `memory/manager_memory.py:431` — `LongTermMemory.__init__` calls `self.load()` unconditionally; creating an instance without a Supabase connection raises immediately in unit tests.

---

## Phase 2: Test Coverage Plan

| Module | Function / Class | Risk | Priority | Test Type | Reason |
|---|---|---|---|---|---|
| `tools/financial_tools/financial_ratio_calculator.py` | All ratio functions | High | P1 | Unit | Pure functions, zero external deps, core business logic |
| `tools/sentiment_tools/fear_greed_calculator.py` | `FearGreedIndexCalculator` | High | P1 | Unit | Core aggregation math, weight validation, label bands |
| `tools/sentiment_tools/vader_scorer.py` | `VaderLexiconScorer` | High | P1 | Unit/Integration | Real NLTK; compound thresholds define Bullish/Bearish signal |
| `core/error_handler.py` | `with_error_reporting` | High | P1 | Unit | Cross-cutting concern; must **never** suppress exceptions |
| `core/observability.py` | `init_sentry`, `init_langsmith` | High | P1 | Unit | Idempotency, env-var gating, module-flag state |
| `api/core/exceptions.py` | Exception hierarchy | High | P1 | Unit | HTTP status codes, `trace_id` uniqueness, `to_dict()` contract |
| `api/config.py` | `Settings`, `validate_settings` | High | P1 | Unit | Startup gate; missing vars must abort process |
| `rag/processor.py` | `AlphaProcessor` | High | P1 | Unit | Double-key idempotency; chunking logic; metadata correctness |
| `memory/manager_memory.py` | `ShortTermMemory`, `LongTermMemory`, `ManagerMemory` | High | P1 | Unit (mocked Supabase) | Cap eviction, FIFO trim, recall payload contract |
| `api/routes/analyze.py` | `POST /api/v1/analyze` | Medium | P2 | Integration | HTTP layer, Pydantic validation, 400/500 responses |
| `api/main.py` | FastAPI lifespan | Medium | P2 | Integration | Sentry/LangSmith called at startup, /health endpoint |
| `api/dependencies.py` | `get_user_id` | Medium | P2 | Unit | Header fallback to `DEFAULT_USER_ID` |

---

## Phase 3: Coverage Requirements

For every critical function/class/module, the tests cover:

- **Happy Path** — correct behavior under normal conditions
- **Edge Cases** — valid but unusual inputs
- **Boundary Values** — exact threshold boundaries (e.g. P/E = 15, P/E = 30)
- **Invalid Inputs** — `None`, zero denominators, empty strings, wrong types
- **Exception Handling** — exceptions always re-raised, never swallowed
- **Branch Coverage** — both positive and negative branches tested

---

## File Path

`tests/__init__.py`

### Objective

Empty package markers required for pytest collection.

### Python Code

```python
# tests/__init__.py
```

---

## File Path

`tests/core/__init__.py`

```python
# tests/core/__init__.py
```

---

## File Path

`tests/api/__init__.py`

```python
# tests/api/__init__.py
```

---

## File Path

`tests/api/core/__init__.py`

```python
# tests/api/core/__init__.py
```

---

## File Path

`tests/tools/__init__.py`

```python
# tests/tools/__init__.py
```

---

## File Path

`tests/tools/financial_tools/__init__.py`

```python
# tests/tools/financial_tools/__init__.py
```

---

## File Path

`tests/tools/sentiment_tools/__init__.py`

```python
# tests/tools/sentiment_tools/__init__.py
```

---

## File Path

`tests/rag/__init__.py`

```python
# tests/rag/__init__.py
```

---

## File Path

`tests/memory/__init__.py`

```python
# tests/memory/__init__.py
```

---

## File Path

`tests/conftest.py`

### Objective

Shared fixtures used across the entire test suite:
- Observability flag isolation (reset `_sentry_enabled` / `_langsmith_enabled` to `False` before each test)
- Mock Supabase client factory
- Canonical `FinBertResult` and `VaderResult` builder fixtures
- Sample `RawDocument` fixture

### Expected Result

Fixtures collected and injected correctly; observability flags always `False` at the start of each test that requests `reset_observability`.

### Python Code

```python
"""tests/conftest.py — shared fixtures."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Observability flag isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def reset_observability(monkeypatch):
    """
    Reset core.observability module-level flags to False before (and after)
    each test that requests this fixture.

    Without this, a test that calls init_sentry() or init_langsmith()
    would pollute all subsequent tests in the same process.
    """
    import core.observability as obs
    monkeypatch.setattr(obs, "_sentry_enabled",   False)
    monkeypatch.setattr(obs, "_langsmith_enabled", False)
    yield
    monkeypatch.setattr(obs, "_sentry_enabled",   False)
    monkeypatch.setattr(obs, "_langsmith_enabled", False)


# ---------------------------------------------------------------------------
# Mock Supabase client
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_supabase():
    """
    Return a MagicMock that mimics the Supabase client interface used by
    LongTermMemory: client.table(...).select(...).eq(...).execute() and
    client.table(...).upsert(...).execute().

    The chain returns an object with .data = [] by default (no existing row).
    """
    client = MagicMock()
    execute_result = MagicMock()
    execute_result.data = []
    chain = MagicMock()
    chain.execute.return_value = execute_result
    client.table.return_value.select.return_value.eq.return_value = chain
    client.table.return_value.upsert.return_value = chain
    return client


@pytest.fixture()
def mock_supabase_with_row(mock_supabase):
    """
    Variant of mock_supabase where a row already exists for user_id='test_user'.
    """
    row = {
        "operational_heuristics": {"key1": "val1"},
        "ticker_insights":        {"NVDA": {"sector": "Tech"}},
        "user_preferences":       {"format": "concise"},
    }
    execute_result = MagicMock()
    execute_result.data = [row]
    mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        execute_result
    )
    return mock_supabase


# ---------------------------------------------------------------------------
# Sentiment dataclass stubs
# (avoids importing from broken tools.finbert_analyzer / tools.vader_scorer paths)
# ---------------------------------------------------------------------------

@dataclass
class _FinBertResult:
    bullish_prob:   float
    bearish_prob:   float
    neutral_prob:   float
    label:          str  = "Neutral"
    total_chunks:   int  = 1
    skipped_chunks: int  = 0


@dataclass
class _VaderResult:
    compound:       float
    positive_mean:  float  = 0.0
    negative_mean:  float  = 0.0
    neutral_mean:   float  = 1.0
    label:          str    = "Neutral"
    chunk_scores:   list   = field(default_factory=list)
    total_chunks:   int    = 1
    skipped_chunks: int    = 0


@pytest.fixture()
def finbert_bullish():
    return _FinBertResult(bullish_prob=0.80, bearish_prob=0.10, neutral_prob=0.10, label="Bullish")


@pytest.fixture()
def finbert_bearish():
    return _FinBertResult(bullish_prob=0.10, bearish_prob=0.80, neutral_prob=0.10, label="Bearish")


@pytest.fixture()
def finbert_neutral():
    return _FinBertResult(bullish_prob=0.33, bearish_prob=0.33, neutral_prob=0.34, label="Neutral")


@pytest.fixture()
def vader_bullish():
    return _VaderResult(compound=0.60, positive_mean=0.55, negative_mean=0.05, label="Bullish")


@pytest.fixture()
def vader_bearish():
    return _VaderResult(compound=-0.60, positive_mean=0.05, negative_mean=0.55, label="Bearish")


@pytest.fixture()
def vader_neutral():
    return _VaderResult(compound=0.0, positive_mean=0.1, negative_mean=0.1, label="Neutral")


# ---------------------------------------------------------------------------
# Raw document fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_doc():
    """A minimal RawDocument with deterministic content."""
    from rag.loader import RawDocument
    return RawDocument(
        title="NVDA crushes Q4 estimates",
        content="Revenue beat by 20%. EPS guidance raised for FY2026.",
        url="https://example.com/nvda-q4",
        source_type="news",
        ticker="NVDA",
        published_at="2025-01-15T10:00:00Z",
    )


@pytest.fixture()
def raw_doc_duplicate(raw_doc):
    """Same URL and content as raw_doc — should be detected as exact duplicate."""
    from rag.loader import RawDocument
    return RawDocument(
        title=raw_doc.title,
        content=raw_doc.content,
        url=raw_doc.url,
        source_type=raw_doc.source_type,
        ticker=raw_doc.ticker,
        published_at=raw_doc.published_at,
    )


@pytest.fixture()
def raw_doc_updated_content(raw_doc):
    """Same URL as raw_doc but different content — should trigger UPDATE path."""
    from rag.loader import RawDocument
    return RawDocument(
        title=raw_doc.title,
        content="Revenue beat by 25%. EPS guidance significantly raised.",
        url=raw_doc.url,
        source_type=raw_doc.source_type,
        ticker=raw_doc.ticker,
        published_at=raw_doc.published_at,
    )
```

---

## File Path

`tests/core/test_observability.py`

### Objective

Test `core/observability.py` bootstrap functions and query predicates.

Critical properties:
- **Idempotency**: calling `init_*` twice returns the same result without re-running setup
- **Graceful degradation**: missing env vars log and return `False` without raising
- **Module flag state**: `sentry_enabled()` / `langsmith_enabled()` reflect `init_*` outcomes
- **LangSmith env-var injection**: `init_langsmith()` sets `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`

### Expected Result

All paths return `bool`; no exceptions raised for missing/invalid config; module flags reflect call outcomes.

### Python Code

```python
"""tests/core/test_observability.py"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

import core.observability as obs


def _reset(monkeypatch):
    monkeypatch.setattr(obs, "_sentry_enabled",   False)
    monkeypatch.setattr(obs, "_langsmith_enabled", False)


# ---------------------------------------------------------------------------
# sentry_enabled() / init_sentry()
# ---------------------------------------------------------------------------

class TestSentryEnabled:
    def test_default_false(self, monkeypatch):
        _reset(monkeypatch)
        assert obs.sentry_enabled() is False

    def test_reflects_module_flag(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_sentry_enabled", True)
        assert obs.sentry_enabled() is True


class TestInitSentry:
    def test_no_dsn_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.delenv("SENTRY_DSN", raising=False)
        assert obs.init_sentry() is False
        assert obs.sentry_enabled() is False

    def test_empty_dsn_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "   ")
        assert obs.init_sentry() is False

    def test_idempotent_when_already_enabled(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_sentry_enabled", True)
        assert obs.init_sentry() is True

    def test_import_error_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.io/123")
        with patch.dict("sys.modules", {"sentry_sdk": None}):
            result = obs.init_sentry()
        assert result is False
        assert obs.sentry_enabled() is False

    def test_successful_init_sets_flag(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.io/123")
        mock_sentry = MagicMock()
        fake_settings = MagicMock()
        fake_settings.APP_ENV = "test"
        with patch.dict("sys.modules", {
            "sentry_sdk": mock_sentry,
            "api.config":  MagicMock(settings=fake_settings),
        }):
            result = obs.init_sentry()
        assert result is True
        assert obs.sentry_enabled() is True

    def test_exception_in_sdk_init_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("SENTRY_DSN", "https://fake@sentry.io/123")
        mock_sentry = MagicMock()
        mock_sentry.init.side_effect = RuntimeError("Network error")
        fake_settings = MagicMock()
        fake_settings.APP_ENV = "test"
        with patch.dict("sys.modules", {
            "sentry_sdk": mock_sentry,
            "api.config":  MagicMock(settings=fake_settings),
        }):
            result = obs.init_sentry()
        assert result is False
        assert obs.sentry_enabled() is False


# ---------------------------------------------------------------------------
# langsmith_enabled() / init_langsmith()
# ---------------------------------------------------------------------------

class TestLangsmithEnabled:
    def test_default_false(self, monkeypatch):
        _reset(monkeypatch)
        assert obs.langsmith_enabled() is False

    def test_reflects_module_flag(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_langsmith_enabled", True)
        assert obs.langsmith_enabled() is True


class TestInitLangsmith:
    def test_no_key_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        assert obs.init_langsmith() is False
        assert obs.langsmith_enabled() is False

    def test_empty_key_returns_false(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("LANGSMITH_API_KEY", "  ")
        assert obs.init_langsmith() is False

    def test_valid_key_sets_flag(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls_test_key_abc123")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        assert obs.init_langsmith() is True
        assert obs.langsmith_enabled() is True

    def test_sets_langchain_env_vars(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls_test_key_abc123")
        monkeypatch.setenv("LANGSMITH_PROJECT", "my-project")
        obs.init_langsmith()
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY")    == "ls_test_key_abc123"
        assert os.environ.get("LANGCHAIN_PROJECT")    == "my-project"

    def test_default_project_name(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls_test_key")
        monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
        obs.init_langsmith()
        assert os.environ.get("LANGCHAIN_PROJECT") == "alpha-agent-node"

    def test_idempotent_when_already_enabled(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setattr(obs, "_langsmith_enabled", True)
        assert obs.init_langsmith() is True

    def test_idempotent_called_twice(self, monkeypatch):
        _reset(monkeypatch)
        monkeypatch.setenv("LANGSMITH_API_KEY", "key123")
        assert obs.init_langsmith() is True
        assert obs.init_langsmith() is True
```

---

## File Path

`tests/core/test_error_handler.py`

### Objective

Test `core/error_handler.py`:
- `_safe_extra()` — JSON-scalar filtering
- `with_error_reporting()` — sync and async decorator variants
- Context manager variants `.context` and `.async_context`
- Guarantee exceptions are **always re-raised** (never swallowed)
- Guarantee `functools.wraps` preserves `__name__`

### Expected Result

Exceptions always propagate. Breadcrumbs emitted only when Sentry is enabled.

### Python Code

```python
"""tests/core/test_error_handler.py"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import core.observability as obs
from core.error_handler import _safe_extra, with_error_reporting


# ---------------------------------------------------------------------------
# _safe_extra
# ---------------------------------------------------------------------------

class TestSafeExtra:
    def test_keeps_str(self):
        assert _safe_extra({"k": "v"}) == {"k": "v"}

    def test_keeps_int(self):
        assert _safe_extra({"n": 42}) == {"n": 42}

    def test_keeps_float(self):
        assert _safe_extra({"f": 3.14}) == {"f": 3.14}

    def test_keeps_bool(self):
        assert _safe_extra({"b": True}) == {"b": True}

    def test_keeps_none(self):
        assert _safe_extra({"x": None}) == {"x": None}

    def test_filters_list(self):
        assert _safe_extra({"lst": [1, 2, 3]}) == {}

    def test_filters_dict(self):
        assert _safe_extra({"d": {"nested": "val"}}) == {}

    def test_filters_object(self):
        assert _safe_extra({"obj": object()}) == {}

    def test_mixed_keeps_only_scalars(self):
        result = _safe_extra({"name": "alpha", "count": 5, "data": [1, 2]})
        assert result == {"name": "alpha", "count": 5}

    def test_empty_input(self):
        assert _safe_extra({}) == {}


# ---------------------------------------------------------------------------
# Decorator: sync function
# ---------------------------------------------------------------------------

class TestWithErrorReportingSync:
    def test_happy_path(self):
        @with_error_reporting("test.component")
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_exception_is_reraised(self):
        @with_error_reporting("test.component")
        def boom():
            raise ValueError("sync error")

        with pytest.raises(ValueError, match="sync error"):
            boom()

    def test_preserves_function_name(self):
        @with_error_reporting("test.component")
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_no_capture_when_sentry_disabled(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", False)

        @with_error_reporting("test.component")
        def fail():
            raise RuntimeError("test")

        with patch("core.error_handler._capture") as mock_capture:
            with pytest.raises(RuntimeError):
                fail()
            mock_capture.assert_not_called()

    def test_capture_called_when_sentry_enabled(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", True)

        @with_error_reporting("my.component")
        def fail():
            raise ValueError("sentry test")

        with patch("core.error_handler._capture") as mock_capture:
            with pytest.raises(ValueError):
                fail()
            mock_capture.assert_called_once()
            assert mock_capture.call_args[0][0] == "my.component"
            assert isinstance(mock_capture.call_args[0][1], ValueError)

    def test_breadcrumb_added_when_sentry_enabled(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", True)

        @with_error_reporting("bread.test")
        def noop():
            return "ok"

        with patch("core.error_handler._add_breadcrumb") as mock_bc:
            noop()
            mock_bc.assert_called_once()
            assert mock_bc.call_args[0][0] == "bread.test"

    def test_return_value_preserved(self):
        @with_error_reporting("test")
        def identity(x):
            return x * 2

        assert identity(21) == 42


# ---------------------------------------------------------------------------
# Decorator: async function
# ---------------------------------------------------------------------------

class TestWithErrorReportingAsync:
    def test_happy_path(self):
        @with_error_reporting("async.component")
        async def async_add(a, b):
            return a + b

        result = asyncio.get_event_loop().run_until_complete(async_add(3, 4))
        assert result == 7

    def test_exception_is_reraised(self):
        @with_error_reporting("async.component")
        async def async_boom():
            raise TypeError("async error")

        with pytest.raises(TypeError, match="async error"):
            asyncio.get_event_loop().run_until_complete(async_boom())

    def test_preserves_function_name(self):
        @with_error_reporting("async.component")
        async def my_async_fn():
            pass

        assert my_async_fn.__name__ == "my_async_fn"

    def test_capture_called_on_exception(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", True)

        @with_error_reporting("async.comp")
        async def fail():
            raise RuntimeError("async sentry")

        with patch("core.error_handler._capture") as mock_cap:
            with pytest.raises(RuntimeError):
                asyncio.get_event_loop().run_until_complete(fail())
            mock_cap.assert_called_once()

    def test_no_capture_when_disabled(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", False)

        @with_error_reporting("async.comp")
        async def fail():
            raise RuntimeError("no sentry")

        with patch("core.error_handler._capture") as mock_cap:
            with pytest.raises(RuntimeError):
                asyncio.get_event_loop().run_until_complete(fail())
            mock_cap.assert_not_called()


# ---------------------------------------------------------------------------
# Context manager: sync
# ---------------------------------------------------------------------------

class TestSyncContextManager:
    def test_happy_path(self):
        results = []
        with with_error_reporting.context("ctx.test"):
            results.append(1)
        assert results == [1]

    def test_exception_is_reraised(self):
        with pytest.raises(KeyError, match="missing"):
            with with_error_reporting.context("ctx.test"):
                raise KeyError("missing")

    def test_capture_called_on_exception(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", True)
        with patch("core.error_handler._capture") as mock_cap:
            with pytest.raises(ValueError):
                with with_error_reporting.context("ctx.sentry"):
                    raise ValueError("ctx error")
            mock_cap.assert_called_once()

    def test_no_capture_when_disabled(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", False)
        with patch("core.error_handler._capture") as mock_cap:
            with pytest.raises(RuntimeError):
                with with_error_reporting.context("ctx.nodisable"):
                    raise RuntimeError("no capture")
            mock_cap.assert_not_called()


# ---------------------------------------------------------------------------
# Context manager: async
# ---------------------------------------------------------------------------

class TestAsyncContextManager:
    def test_happy_path(self):
        async def run():
            results = []
            async with with_error_reporting.async_context("async.ctx"):
                results.append(42)
            return results

        assert asyncio.get_event_loop().run_until_complete(run()) == [42]

    def test_exception_is_reraised(self):
        async def run():
            async with with_error_reporting.async_context("async.ctx"):
                raise IndexError("out of bounds")

        with pytest.raises(IndexError, match="out of bounds"):
            asyncio.get_event_loop().run_until_complete(run())

    def test_capture_called_on_exception(self, monkeypatch):
        monkeypatch.setattr(obs, "_sentry_enabled", True)

        async def run():
            async with with_error_reporting.async_context("async.sentry"):
                raise OverflowError("big")

        with patch("core.error_handler._capture") as mock_cap:
            with pytest.raises(OverflowError):
                asyncio.get_event_loop().run_until_complete(run())
            mock_cap.assert_called_once()
```

---

## File Path

`tests/api/core/test_exceptions.py`

### Objective

Verify the exception hierarchy in `api/core/exceptions.py`: HTTP status codes, `code` fields, `to_dict()` contract, `trace_id` format.

### Expected Result

Each exception maps to exactly the right HTTP status. `to_dict()` always contains `error`, `message`, `detail`, `trace_id`. `trace_id` is always 8 characters. All exceptions are subclasses of `AlphaAgentError`.

### Python Code

```python
"""tests/api/core/test_exceptions.py"""
from __future__ import annotations

import pytest

from api.core.exceptions import (
    AgentError,
    AgentTimeoutError,
    AlphaAgentError,
    ExternalServiceError,
    MemoryError,
    ValidationError,
)


class TestAlphaAgentError:
    def test_default_message(self):
        exc = AlphaAgentError()
        assert exc.message == "An unexpected error occurred."

    def test_custom_message(self):
        exc = AlphaAgentError(message="Something broke")
        assert exc.message == "Something broke"
        assert str(exc) == "Something broke"

    def test_detail_defaults_to_none(self):
        assert AlphaAgentError().detail is None

    def test_custom_detail(self):
        exc = AlphaAgentError(message="err", detail="extra context")
        assert exc.detail == "extra context"

    def test_trace_id_is_8_chars(self):
        exc = AlphaAgentError()
        assert isinstance(exc.trace_id, str)
        assert len(exc.trace_id) == 8

    def test_trace_id_unique_per_instance(self):
        assert AlphaAgentError().trace_id != AlphaAgentError().trace_id

    def test_http_status_500(self):
        assert AlphaAgentError.http_status == 500

    def test_code_internal_error(self):
        assert AlphaAgentError.code == "INTERNAL_ERROR"

    def test_to_dict_keys(self):
        assert set(AlphaAgentError("t").to_dict().keys()) == {
            "error", "message", "detail", "trace_id"
        }

    def test_to_dict_values(self):
        exc = AlphaAgentError(message="test msg", detail="test detail")
        d = exc.to_dict()
        assert d["error"]   == "INTERNAL_ERROR"
        assert d["message"] == "test msg"
        assert d["detail"]  == "test detail"
        assert len(d["trace_id"]) == 8

    def test_to_dict_none_detail(self):
        assert AlphaAgentError("no detail").to_dict()["detail"] is None

    def test_is_exception(self):
        assert isinstance(AlphaAgentError(), Exception)


class TestValidationError:
    def test_http_status_400(self):
        assert ValidationError.http_status == 400

    def test_code(self):
        assert ValidationError.code == "VALIDATION_ERROR"

    def test_inherits_base(self):
        assert isinstance(ValidationError("bad"), AlphaAgentError)

    def test_raise_and_catch_as_base(self):
        with pytest.raises(AlphaAgentError):
            raise ValidationError("bad input")

    def test_to_dict_code(self):
        assert ValidationError("bad").to_dict()["error"] == "VALIDATION_ERROR"


class TestAgentError:
    def test_http_status_500(self):
        assert AgentError.http_status == 500

    def test_code(self):
        assert AgentError.code == "AGENT_ERROR"

    def test_detail_propagated(self):
        exc = AgentError(message="Graph failed", detail="node X crashed")
        assert exc.detail == "node X crashed"


class TestAgentTimeoutError:
    def test_http_status_504(self):
        assert AgentTimeoutError.http_status == 504

    def test_code(self):
        assert AgentTimeoutError.code == "AGENT_TIMEOUT"


class TestMemoryError:
    def test_http_status_500(self):
        assert MemoryError.http_status == 500

    def test_code(self):
        assert MemoryError.code == "MEMORY_ERROR"


class TestExternalServiceError:
    def test_http_status_503(self):
        assert ExternalServiceError.http_status == 503

    def test_code(self):
        assert ExternalServiceError.code == "EXTERNAL_SERVICE_ERROR"


@pytest.mark.parametrize("exc_cls, expected_status, expected_code", [
    (AlphaAgentError,      500, "INTERNAL_ERROR"),
    (ValidationError,      400, "VALIDATION_ERROR"),
    (AgentError,           500, "AGENT_ERROR"),
    (AgentTimeoutError,    504, "AGENT_TIMEOUT"),
    (MemoryError,          500, "MEMORY_ERROR"),
    (ExternalServiceError, 503, "EXTERNAL_SERVICE_ERROR"),
])
def test_exception_status_and_code(exc_cls, expected_status, expected_code):
    assert exc_cls.http_status == expected_status
    assert exc_cls.code        == expected_code


@pytest.mark.parametrize("exc_cls", [
    ValidationError, AgentError, AgentTimeoutError, MemoryError, ExternalServiceError
])
def test_all_subclass_alphaageneterror(exc_cls):
    assert issubclass(exc_cls, AlphaAgentError)


@pytest.mark.parametrize("exc_cls", [
    AlphaAgentError, ValidationError, AgentError,
    AgentTimeoutError, MemoryError, ExternalServiceError,
])
def test_to_dict_always_has_required_keys(exc_cls):
    keys = exc_cls(message="x").to_dict().keys()
    for required in ("error", "message", "detail", "trace_id"):
        assert required in keys
```

---

## File Path

`tests/api/test_config.py`

### Objective

Verify `api/config.py` defaults and `validate_settings()` startup gate.

### Expected Result

`validate_settings()` calls `sys.exit(1)` when any required key is empty; returns normally when all are set.

### Python Code

```python
"""tests/api/test_config.py"""
from __future__ import annotations

import pytest


class TestSettingsDefaults:
    def test_anthropic_model_non_empty(self):
        from api.config import Settings
        s = Settings()
        assert isinstance(s.ANTHROPIC_MODEL, str) and len(s.ANTHROPIC_MODEL) > 0

    def test_max_routing_loops_default(self):
        from api.config import Settings
        assert Settings().MAX_ROUTING_LOOPS == 8

    def test_app_env_valid_literal(self):
        from api.config import Settings
        assert Settings().APP_ENV in ("development", "production")

    def test_request_timeout_default(self):
        from api.config import Settings
        assert Settings().REQUEST_TIMEOUT_S == 300

    def test_allowed_origins_is_list(self):
        from api.config import Settings
        assert isinstance(Settings().ALLOWED_ORIGINS, list)


class TestValidateSettings:
    def _patch(self, monkeypatch, *, api_key="", url="", key=""):
        from api import config
        monkeypatch.setattr(config.settings, "ANTHROPIC_API_KEY", api_key)
        monkeypatch.setattr(config.settings, "SUPABASE_URL",      url)
        monkeypatch.setattr(config.settings, "SUPABASE_KEY",      key)

    def test_exits_when_all_missing(self, monkeypatch):
        self._patch(monkeypatch)
        from api.config import validate_settings
        with pytest.raises(SystemExit) as exc_info:
            validate_settings()
        assert exc_info.value.code == 1

    def test_exits_when_api_key_missing(self, monkeypatch):
        self._patch(monkeypatch, api_key="", url="https://x.supabase.co", key="srk")
        from api.config import validate_settings
        with pytest.raises(SystemExit):
            validate_settings()

    def test_exits_when_supabase_url_missing(self, monkeypatch):
        self._patch(monkeypatch, api_key="sk-ant", url="", key="srk")
        from api.config import validate_settings
        with pytest.raises(SystemExit):
            validate_settings()

    def test_exits_when_supabase_key_missing(self, monkeypatch):
        self._patch(monkeypatch, api_key="sk-ant", url="https://x.supabase.co", key="")
        from api.config import validate_settings
        with pytest.raises(SystemExit):
            validate_settings()

    def test_passes_when_all_set(self, monkeypatch):
        self._patch(monkeypatch, api_key="sk-ant", url="https://x.supabase.co", key="srk")
        from api.config import validate_settings
        validate_settings()  # must not raise
```

---

## File Path

`tests/tools/financial_tools/test_financial_ratio_calculator.py`

### Objective

Exhaustively test every function in `tools/financial_tools/financial_ratio_calculator.py`. This module is pure Python with zero external dependencies — maximum testability, highest business risk.

### Expected Result

All functions return the correct dict structure with accurate numeric values; interpretation labels match documented threshold boundaries exactly; `None` and zero-denominator inputs are handled safely.

### Python Code

```python
"""tests/tools/financial_tools/test_financial_ratio_calculator.py"""
from __future__ import annotations

import pytest

from tools.financial_tools.financial_ratio_calculator import (
    _label,
    _safe_div,
    asset_turnover,
    cagr,
    composite_financial_score,
    current_ratio,
    debt_to_equity,
    ev_to_ebitda,
    gross_margin,
    interest_coverage,
    net_margin,
    operating_margin,
    peg_ratio,
    price_to_book,
    price_to_earnings,
    quick_ratio,
    return_on_assets,
    return_on_equity,
)


# ---------------------------------------------------------------------------
# _safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10.0, 2.0) == pytest.approx(5.0)

    def test_returns_float(self):
        assert isinstance(_safe_div(7, 2), float)

    def test_zero_denominator_returns_default(self):
        assert _safe_div(10.0, 0) is None

    def test_zero_denominator_custom_default(self):
        assert _safe_div(10.0, 0, default=0.0) == 0.0

    def test_none_numerator(self):
        assert _safe_div(None, 5.0) is None

    def test_none_denominator(self):
        assert _safe_div(5.0, None) is None

    def test_both_none(self):
        assert _safe_div(None, None) is None

    def test_negative_numerator(self):
        assert _safe_div(-10.0, 2.0) == pytest.approx(-5.0)

    def test_negative_denominator(self):
        assert _safe_div(10.0, -2.0) == pytest.approx(-5.0)

    def test_fractional_result(self):
        assert _safe_div(1.0, 3.0) == pytest.approx(0.3333, rel=1e-3)


# ---------------------------------------------------------------------------
# _label
# ---------------------------------------------------------------------------

class TestLabel:
    THRESHOLDS = [(60, "excellent"), (40, "good"), (20, "moderate"), (0, "low")]

    def test_none_returns_unavailable(self):
        assert _label(None, self.THRESHOLDS) == "unavailable"

    def test_above_highest(self):
        assert _label(70, self.THRESHOLDS) == "excellent"

    def test_at_highest(self):
        assert _label(60, self.THRESHOLDS) == "excellent"

    def test_between(self):
        assert _label(50, self.THRESHOLDS) == "good"

    def test_at_zero(self):
        assert _label(0, self.THRESHOLDS) == "low"

    def test_below_all(self):
        assert _label(-10, self.THRESHOLDS) == "low"

    def test_lower_is_better(self):
        t = [(0, "undervalued"), (15, "fairly_valued"), (30, "overvalued")]
        assert _label(5,  t, higher_is_better=False) == "undervalued"
        assert _label(20, t, higher_is_better=False) == "fairly_valued"
        assert _label(35, t, higher_is_better=False) == "overvalued"


# ---------------------------------------------------------------------------
# price_to_earnings
# ---------------------------------------------------------------------------

class TestPriceToEarnings:
    def test_undervalued(self):
        r = price_to_earnings(100.0, 10.0)  # PE = 10
        assert r["pe_ratio"] == pytest.approx(10.0)
        assert r["interpretation"] == "undervalued"

    def test_fairly_valued(self):
        assert price_to_earnings(300.0, 15.0)["interpretation"] == "fairly_valued"  # PE = 20

    def test_overvalued(self):
        assert price_to_earnings(3000.0, 60.0)["interpretation"] == "overvalued"  # PE = 50

    def test_negative_eps(self):
        assert price_to_earnings(100.0, -5.0)["interpretation"] == "negative_earnings"

    def test_zero_eps_returns_none(self):
        assert price_to_earnings(100.0, 0.0)["pe_ratio"] is None

    def test_formula_present(self):
        assert "formula" in price_to_earnings(100.0, 10.0)

    def test_boundary_pe_exactly_30(self):
        assert price_to_earnings(300.0, 10.0)["interpretation"] == "fairly_valued"

    def test_rounding_to_2_decimals(self):
        r = price_to_earnings(100.0, 3.0)
        assert r["pe_ratio"] == round(100.0 / 3.0, 2)


# ---------------------------------------------------------------------------
# price_to_book
# ---------------------------------------------------------------------------

class TestPriceToBook:
    def test_below_book(self):
        r = price_to_book(8.0, 10.0)  # PB = 0.8
        assert r["pb_ratio"] == pytest.approx(0.8)
        assert r["interpretation"] == "trading_below_book"

    def test_fairly_valued(self):
        assert price_to_book(20.0, 10.0)["interpretation"] == "fairly_valued"

    def test_premium(self):
        assert price_to_book(50.0, 10.0)["interpretation"] == "premium_to_book"

    def test_zero_book_value(self):
        r = price_to_book(50.0, 0.0)
        assert r["pb_ratio"] is None
        assert r["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# ev_to_ebitda
# ---------------------------------------------------------------------------

class TestEvToEbitda:
    def test_undervalued(self):
        assert ev_to_ebitda(1_000_000, 200_000)["interpretation"] == "undervalued"

    def test_fairly_valued(self):
        assert ev_to_ebitda(1_500_000, 100_000)["interpretation"] == "fairly_valued"

    def test_expensive(self):
        assert ev_to_ebitda(3_000_000, 100_000)["interpretation"] == "expensive"

    def test_zero_ebitda(self):
        r = ev_to_ebitda(1_000_000, 0)
        assert r["ev_ebitda"] is None
        assert r["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# peg_ratio
# ---------------------------------------------------------------------------

class TestPegRatio:
    def test_undervalued(self):
        r = peg_ratio(20.0, 25.0)  # PEG = 0.8
        assert r["peg"] == pytest.approx(0.8)
        assert r["interpretation"] == "undervalued"

    def test_fairly_valued(self):
        assert peg_ratio(30.0, 20.0)["interpretation"] == "fairly_valued"

    def test_overvalued(self):
        assert peg_ratio(60.0, 20.0)["interpretation"] == "overvalued"

    def test_zero_growth(self):
        r = peg_ratio(20.0, 0.0)
        assert r["peg"] is None
        assert r["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# gross_margin
# ---------------------------------------------------------------------------

class TestGrossMargin:
    @pytest.mark.parametrize("cogs, expected", [
        (30,  "excellent"),
        (55,  "good"),
        (75,  "moderate"),
        (95,  "low"),
    ])
    def test_interpretations(self, cogs, expected):
        assert gross_margin(100, cogs)["interpretation"] == expected

    def test_zero_revenue(self):
        assert gross_margin(0, 50)["gross_margin_pct"] is None

    def test_negative_margin(self):
        assert gross_margin(100, 110)["gross_margin_pct"] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# operating_margin
# ---------------------------------------------------------------------------

class TestOperatingMargin:
    @pytest.mark.parametrize("op_income, expected", [
        (30, "excellent"),
        (18, "good"),
        (8,  "moderate"),
        (2,  "low"),
    ])
    def test_interpretations(self, op_income, expected):
        assert operating_margin(op_income, 100)["interpretation"] == expected

    def test_zero_revenue(self):
        assert operating_margin(10, 0)["operating_margin_pct"] is None


# ---------------------------------------------------------------------------
# net_margin
# ---------------------------------------------------------------------------

class TestNetMargin:
    @pytest.mark.parametrize("net, expected", [
        (25, "excellent"),
        (12, "good"),
        (6,  "moderate"),
        (3,  "low"),
    ])
    def test_interpretations(self, net, expected):
        assert net_margin(net, 100)["interpretation"] == expected

    def test_zero_revenue(self):
        assert net_margin(10, 0)["net_margin_pct"] is None


# ---------------------------------------------------------------------------
# return_on_equity
# ---------------------------------------------------------------------------

class TestReturnOnEquity:
    @pytest.mark.parametrize("net, expected", [
        (25, "excellent"),
        (16, "good"),
        (11, "moderate"),
        (5,  "low"),
    ])
    def test_interpretations(self, net, expected):
        assert return_on_equity(net, 100)["interpretation"] == expected

    def test_zero_equity(self):
        assert return_on_equity(10, 0)["roe_pct"] is None


# ---------------------------------------------------------------------------
# return_on_assets
# ---------------------------------------------------------------------------

class TestReturnOnAssets:
    @pytest.mark.parametrize("net, expected", [
        (12, "excellent"),
        (6,  "good"),
        (3,  "moderate"),
        (1,  "low"),
    ])
    def test_interpretations(self, net, expected):
        assert return_on_assets(net, 100)["interpretation"] == expected

    def test_zero_assets(self):
        assert return_on_assets(10, 0)["roa_pct"] is None


# ---------------------------------------------------------------------------
# current_ratio
# ---------------------------------------------------------------------------

class TestCurrentRatio:
    def test_strong(self):
        r = current_ratio(250, 100)  # CR = 2.5
        assert r["current_ratio"] == pytest.approx(2.5)
        assert r["interpretation"] == "strong"

    def test_adequate(self):
        assert current_ratio(150, 100)["interpretation"] == "adequate"

    def test_weak(self):
        assert current_ratio(80, 100)["interpretation"] == "weak"

    def test_boundary_cr_2(self):
        assert current_ratio(200, 100)["interpretation"] == "strong"

    def test_boundary_cr_1(self):
        assert current_ratio(100, 100)["interpretation"] == "adequate"

    def test_zero_liabilities(self):
        r = current_ratio(100, 0)
        assert r["current_ratio"] is None
        assert r["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# quick_ratio
# ---------------------------------------------------------------------------

class TestQuickRatio:
    def test_strong(self):
        assert quick_ratio(80, 20, 50, 100)["interpretation"] == "strong"

    def test_moderate(self):
        assert quick_ratio(40, 10, 20, 100)["interpretation"] == "moderate"

    def test_weak(self):
        assert quick_ratio(10, 5, 10, 100)["interpretation"] == "weak"

    def test_zero_liabilities(self):
        r = quick_ratio(10, 10, 10, 0)
        assert r["quick_ratio"] is None
        assert r["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# debt_to_equity
# ---------------------------------------------------------------------------

class TestDebtToEquity:
    def test_low_leverage(self):
        assert debt_to_equity(30, 100)["interpretation"] == "low_leverage"

    def test_moderate_leverage(self):
        assert debt_to_equity(100, 100)["interpretation"] == "moderate_leverage"

    def test_high_leverage(self):
        assert debt_to_equity(200, 100)["interpretation"] == "high_leverage"

    def test_zero_equity(self):
        r = debt_to_equity(100, 0)
        assert r["de_ratio"] is None
        assert r["interpretation"] == "unavailable"


# ---------------------------------------------------------------------------
# interest_coverage
# ---------------------------------------------------------------------------

class TestInterestCoverage:
    def test_strong(self):
        assert interest_coverage(600, 100)["interpretation"] == "strong"

    def test_adequate(self):
        assert interest_coverage(300, 100)["interpretation"] == "adequate"

    def test_at_risk(self):
        assert interest_coverage(150, 100)["interpretation"] == "at_risk"

    def test_zero_interest(self):
        assert interest_coverage(500, 0)["interest_coverage"] is None


# ---------------------------------------------------------------------------
# asset_turnover
# ---------------------------------------------------------------------------

class TestAssetTurnover:
    def test_efficient(self):
        assert asset_turnover(120, 100)["interpretation"] == "efficient"

    def test_moderate(self):
        assert asset_turnover(70, 100)["interpretation"] == "moderate"

    def test_low_efficiency(self):
        assert asset_turnover(30, 100)["interpretation"] == "low_efficiency"

    def test_zero_assets(self):
        assert asset_turnover(100, 0)["asset_turnover"] is None


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------

class TestCagr:
    def test_hypergrowth(self):
        r = cagr(100, 200, 2)  # ≈ 41.4%
        assert r["cagr_pct"] == pytest.approx(41.42, rel=1e-2)
        assert r["interpretation"] == "hypergrowth"

    def test_slow_growth(self):
        assert cagr(100, 105, 5)["interpretation"] == "slow"

    def test_zero_start_value(self):
        r = cagr(0, 100, 3)
        assert r["cagr_pct"] is None
        assert r["interpretation"] == "unavailable"

    def test_negative_start_value(self):
        assert cagr(-50, 100, 3)["cagr_pct"] is None

    def test_zero_years(self):
        assert cagr(100, 200, 0)["cagr_pct"] is None

    def test_formula_present(self):
        assert "formula" in cagr(100, 200, 5)

    def test_decimal_years(self):
        assert cagr(100, 150, 2.5)["cagr_pct"] is not None


# ---------------------------------------------------------------------------
# composite_financial_score
# ---------------------------------------------------------------------------

class TestCompositeFinancialScore:
    def test_all_inputs_returns_score(self):
        r = composite_financial_score(
            pe=15, pb=2, roe_pct=20, net_margin_pct=15,
            current_ratio=2.0, de_ratio=0.5, revenue_cagr_pct=20
        )
        assert r["score"] is not None
        assert 0 <= r["score"] <= 100
        assert r["grade"] in ("A", "B", "C", "D", "F")

    def test_no_inputs_returns_none_score(self):
        r = composite_financial_score()
        assert r["score"] is None
        assert r["grade"] == "N/A"

    def test_partial_inputs_still_computed(self):
        r = composite_financial_score(roe_pct=25)
        assert r["score"] is not None
        assert "roe" in r["sub_scores"]

    def test_missing_inputs_tracked(self):
        r = composite_financial_score(roe_pct=20)
        assert "net_margin" in r["missing_inputs"]
        assert "pe"         in r["missing_inputs"]

    def test_grade_a_for_excellent_inputs(self):
        r = composite_financial_score(
            pe=10, roe_pct=40, net_margin_pct=30,
            current_ratio=3.0, de_ratio=0.1, revenue_cagr_pct=50
        )
        assert r["grade"] in ("A", "B")

    def test_grade_f_for_poor_inputs(self):
        r = composite_financial_score(
            pe=60, roe_pct=0, net_margin_pct=0,
            current_ratio=0.1, de_ratio=3.0, revenue_cagr_pct=0
        )
        assert r["grade"] in ("D", "F")

    def test_sub_scores_in_0_to_10_range(self):
        r = composite_financial_score(
            pe=15, roe_pct=20, net_margin_pct=15,
            current_ratio=2.0, de_ratio=0.5, revenue_cagr_pct=20
        )
        for name, val in r["sub_scores"].items():
            assert 0 <= val <= 10, f"{name} = {val} out of [0, 10]"

    def test_pb_not_in_weighting(self):
        r1 = composite_financial_score(pe=15, roe_pct=20, pb=1)
        r2 = composite_financial_score(pe=15, roe_pct=20, pb=100)
        assert r1["score"] == r2["score"]
```

---

## File Path

`tests/tools/sentiment_tools/test_fear_greed_calculator.py`

### Objective

Test `FearGreedIndexCalculator` in isolation from FinBERT/VADER model loading. Uses stub dataclasses to avoid the broken module-level imports.

### Expected Result

Scores computed as `(finbert_score × w_finbert) + (vader_score × w_vader)`, clamped to `[-1.0, +1.0]`. Labels match five-band thresholds. `ValueError` raised for invalid weights.

### Python Code

```python
"""tests/tools/sentiment_tools/test_fear_greed_calculator.py"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest


# Patch broken top-level imports BEFORE importing the module under test
@dataclass
class _FinBertResult:
    bullish_prob:   float
    bearish_prob:   float
    neutral_prob:   float
    label:          str = "Neutral"
    total_chunks:   int = 1
    skipped_chunks: int = 0


@dataclass
class _VaderResult:
    compound:       float
    positive_mean:  float = 0.0
    negative_mean:  float = 0.0
    neutral_mean:   float = 1.0
    label:          str   = "Neutral"
    chunk_scores:   list  = field(default_factory=list)
    total_chunks:   int   = 1
    skipped_chunks: int   = 0


@pytest.fixture(autouse=True, scope="module")
def patch_broken_imports():
    fake_finbert = MagicMock()
    fake_finbert.FinBertResult = _FinBertResult
    fake_vader = MagicMock()
    fake_vader.VaderResult = _VaderResult
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, {
        "tools.finbert_analyzer": fake_finbert,
        "tools.vader_scorer":     fake_vader,
    }):
        yield


from tools.sentiment_tools.fear_greed_calculator import (  # noqa: E402
    FearGreedIndexCalculator,
    FearGreedResult,
)


class TestValidateWeights:
    def test_default_weights_valid(self):
        c = FearGreedIndexCalculator()
        assert c.finbert_weight == pytest.approx(0.65)
        assert c.vader_weight   == pytest.approx(0.35)

    def test_custom_valid_weights(self):
        FearGreedIndexCalculator(finbert_weight=0.5, vader_weight=0.5)

    def test_negative_finbert_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FearGreedIndexCalculator(finbert_weight=-0.1, vader_weight=1.1)

    def test_negative_vader_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FearGreedIndexCalculator(finbert_weight=1.1, vader_weight=-0.1)

    def test_weights_not_summing_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            FearGreedIndexCalculator(finbert_weight=0.5, vader_weight=0.4)

    def test_within_tolerance_accepted(self):
        FearGreedIndexCalculator(finbert_weight=0.6501, vader_weight=0.3499)


class TestScoreToLabel:
    @pytest.mark.parametrize("score, expected", [
        ( 1.00, "Extreme Greed"),
        ( 0.60, "Extreme Greed"),
        ( 0.59, "Greed"),
        ( 0.20, "Greed"),
        ( 0.19, "Neutral"),
        ( 0.00, "Neutral"),
        (-0.19, "Neutral"),
        (-0.20, "Fear"),
        (-0.59, "Fear"),
        (-0.60, "Fear"),
        (-0.61, "Extreme Fear"),
        (-1.00, "Extreme Fear"),
    ])
    def test_label_bands(self, score, expected):
        assert FearGreedIndexCalculator._score_to_label(score) == expected


class TestCalculate:
    @pytest.fixture()
    def calc(self):
        return FearGreedIndexCalculator()

    def _fb(self, bullish, bearish):
        return _FinBertResult(bullish_prob=bullish, bearish_prob=bearish,
                              neutral_prob=max(0.0, 1 - bullish - bearish))

    def _vd(self, compound):
        return _VaderResult(compound=compound)

    def test_extreme_greed(self, calc):
        r = calc.calculate(self._fb(0.9, 0.0), self._vd(0.9))
        assert r.label == "Extreme Greed"
        assert r.score >= 0.60

    def test_extreme_fear(self, calc):
        r = calc.calculate(self._fb(0.0, 0.9), self._vd(-0.9))
        assert r.label == "Extreme Fear"
        assert r.score <= -0.60

    def test_neutral_zero_inputs(self, calc):
        r = calc.calculate(self._fb(0.33, 0.33), self._vd(0.0))
        assert r.label == "Neutral"

    def test_score_clamped_at_plus_one(self, calc):
        assert calc.calculate(self._fb(1.0, 0.0), self._vd(1.0)).score <= 1.0

    def test_score_clamped_at_minus_one(self, calc):
        assert calc.calculate(self._fb(0.0, 1.0), self._vd(-1.0)).score >= -1.0

    def test_returns_feargreed_result_type(self, calc):
        assert isinstance(calc.calculate(self._fb(0.5, 0.3), self._vd(0.2)), FearGreedResult)

    def test_confidence_equals_abs_score(self, calc):
        r = calc.calculate(self._fb(0.8, 0.1), self._vd(0.5))
        assert r.confidence == pytest.approx(abs(r.score), rel=1e-3)

    def test_weights_in_result(self, calc):
        r = calc.calculate(self._fb(0.5, 0.3), self._vd(0.2))
        assert r.weights["finbert"] == pytest.approx(0.65)
        assert r.weights["vader"]   == pytest.approx(0.35)

    def test_diagnostics_keys(self, calc):
        r = calc.calculate(self._fb(0.6, 0.2), self._vd(0.3))
        assert "raw_score"            in r.diagnostics
        assert "finbert_bullish_prob" in r.diagnostics
        assert "vader_compound"       in r.diagnostics

    def test_math_correctness(self, calc):
        # finbert_score = 0.7 - 0.2 = 0.5; vader = 0.4
        # raw = 0.5 * 0.65 + 0.4 * 0.35 = 0.325 + 0.14 = 0.465
        r = calc.calculate(self._fb(0.7, 0.2), self._vd(0.4))
        assert r.score == pytest.approx(0.465, rel=1e-3)

    def test_custom_weights_affect_score(self):
        calc = FearGreedIndexCalculator(finbert_weight=0.2, vader_weight=0.8)
        # finbert_score = 0.9 - 0.1 = 0.8; vader = -0.8
        # raw = 0.8*0.2 + (-0.8)*0.8 = 0.16 - 0.64 = -0.48
        r = calc.calculate(
            _FinBertResult(bullish_prob=0.9, bearish_prob=0.1, neutral_prob=0.0),
            _VaderResult(compound=-0.8),
        )
        assert r.score == pytest.approx(-0.48, rel=1e-3)
        assert r.label == "Fear"


class TestCalculateFromDict:
    def test_valid_dicts(self):
        calc = FearGreedIndexCalculator()
        r = calc.calculate_from_dict(
            {"bullish_prob": 0.7, "bearish_prob": 0.2, "neutral_prob": 0.1,
             "label": "Bullish", "total_chunks": 5, "skipped_chunks": 0},
            {"compound": 0.4, "positive_mean": 0.3, "negative_mean": 0.1,
             "neutral_mean": 0.6, "label": "Bullish",
             "total_chunks": 5, "skipped_chunks": 0},
        )
        assert isinstance(r, FearGreedResult)
        assert -1.0 <= r.score <= 1.0

    def test_missing_key_raises(self):
        calc = FearGreedIndexCalculator()
        with pytest.raises(KeyError):
            calc.calculate_from_dict(
                {"bearish_prob": 0.2, "neutral_prob": 0.1},  # missing bullish_prob
                {"compound": 0.3},
            )
```

---

## File Path

`tests/tools/sentiment_tools/test_vader_scorer.py`

### Objective

Test `VaderLexiconScorer` with real NLTK VADER. Covers corpus scoring, empty-text handling, single-text scoring, label thresholds, and the `_empty_result` fallback.

**Prerequisite**: NLTK `vader_lexicon` must be downloadable or already present.

### Expected Result

`score([])` returns neutral zero result. `score_single("")` raises `ValueError`. Mean compound is arithmetic mean of per-chunk compounds.

### Python Code

```python
"""tests/tools/sentiment_tools/test_vader_scorer.py"""
from __future__ import annotations

import pytest

from tools.sentiment_tools.vader_scorer import (
    ChunkVaderScore,
    VaderLexiconScorer,
    VaderResult,
    _compound_label,
    _empty_result,
)

pytestmark = pytest.mark.nltk


class TestCompoundLabel:
    @pytest.mark.parametrize("compound, expected", [
        ( 0.05,  "Bullish"),
        ( 1.00,  "Bullish"),
        ( 0.50,  "Bullish"),
        (-0.05,  "Bearish"),
        (-1.00,  "Bearish"),
        (-0.50,  "Bearish"),
        ( 0.04,  "Neutral"),
        (-0.04,  "Neutral"),
        ( 0.00,  "Neutral"),
    ])
    def test_label_mapping(self, compound, expected):
        assert _compound_label(compound) == expected

    def test_positive_threshold_exact(self):
        assert _compound_label(0.05) == "Bullish"

    def test_negative_threshold_exact(self):
        assert _compound_label(-0.05) == "Bearish"


class TestEmptyResult:
    def test_all_zeros(self):
        r = _empty_result(skipped=3)
        assert r.compound      == 0.0
        assert r.positive_mean == 0.0
        assert r.negative_mean == 0.0
        assert r.neutral_mean  == 0.0

    def test_label_neutral(self):
        assert _empty_result().label == "Neutral"

    def test_skipped_count(self):
        r = _empty_result(skipped=5)
        assert r.skipped_chunks == 5
        assert r.total_chunks   == 0

    def test_empty_chunk_scores(self):
        assert _empty_result().chunk_scores == []


@pytest.fixture(scope="module")
def scorer():
    return VaderLexiconScorer()


class TestScore:
    def test_empty_list_returns_neutral(self, scorer):
        r = scorer.score([])
        assert r.label        == "Neutral"
        assert r.compound     == 0.0
        assert r.total_chunks == 0

    def test_all_empty_strings(self, scorer):
        r = scorer.score(["", "   ", "\t"])
        assert r.total_chunks    == 0
        assert r.skipped_chunks  == 3
        assert r.label           == "Neutral"

    def test_positive_text(self, scorer):
        r = scorer.score(["Amazing outstanding excellent quarter! 🚀"])
        assert r.label in ("Bullish", "Neutral")
        assert r.compound > -0.1

    def test_negative_text(self, scorer):
        r = scorer.score(["Terrible crash. Horrible losses. Collapsing market."])
        assert r.label in ("Bearish", "Neutral")
        assert r.compound < 0.1

    def test_result_type(self, scorer):
        assert isinstance(scorer.score(["hello"]), VaderResult)

    def test_total_chunks_equals_valid_texts(self, scorer):
        r = scorer.score(["good", "bad", ""])
        assert r.total_chunks   == 2
        assert r.skipped_chunks == 1

    def test_chunk_scores_populated(self, scorer):
        r = scorer.score(["Market up!", "Market down."])
        assert len(r.chunk_scores) == 2
        assert all(isinstance(c, ChunkVaderScore) for c in r.chunk_scores)

    def test_mean_compound_is_arithmetic_mean(self, scorer):
        r = scorer.score(["Great day!", "Terrible loss."])
        expected = sum(c.compound for c in r.chunk_scores) / len(r.chunk_scores)
        assert r.compound == pytest.approx(expected, abs=1e-4)

    def test_chunk_scores_sum_to_one(self, scorer):
        r = scorer.score(["The stock rose sharply today"])
        cs = r.chunk_scores[0]
        assert (cs.positive + cs.negative + cs.neutral) == pytest.approx(1.0, abs=1e-3)

    def test_text_truncated_at_120_chars(self, scorer):
        r = scorer.score(["a" * 200])
        assert len(r.chunk_scores[0].text) <= 123


class TestScoreSingle:
    def test_empty_raises(self, scorer):
        with pytest.raises(ValueError, match="non-empty"):
            scorer.score_single("")

    def test_whitespace_raises(self, scorer):
        with pytest.raises(ValueError, match="non-empty"):
            scorer.score_single("   ")

    def test_valid_text_returns_chunk_score(self, scorer):
        assert isinstance(scorer.score_single("Stocks rising fast!"), ChunkVaderScore)

    def test_compound_in_range(self, scorer):
        r = scorer.score_single("neutral market today")
        assert -1.0 <= r.compound <= 1.0

    def test_label_valid(self, scorer):
        assert scorer.score_single("some text").label in ("Bullish", "Bearish", "Neutral")
```

---

## File Path

`tests/rag/test_processor.py`

### Objective

Test `rag/processor.py` double-key idempotency, chunk metadata correctness, `ProcessorMetrics` reporting, and hashing utilities.

### Expected Result

Exact duplicates skipped; updated content generates new chunks; fresh URLs ingested. All metadata keys always present.

### Python Code

```python
"""tests/rag/test_processor.py"""
from __future__ import annotations

import hashlib

import pytest

from rag.processor import (
    AlphaProcessor,
    ProcessedChunk,
    ProcessorMetrics,
    _sha256,
    _url_hash,
)


class TestHashing:
    def test_sha256_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_sha256_different_inputs(self):
        assert _sha256("abc") != _sha256("xyz")

    def test_sha256_is_64_chars(self):
        assert len(_sha256("test")) == 64

    def test_sha256_matches_stdlib(self):
        text = "alpha agent"
        assert _sha256(text) == hashlib.sha256(text.encode()).hexdigest()

    def test_url_hash_deterministic(self):
        assert _url_hash("https://example.com") == _url_hash("https://example.com")

    def test_url_hash_different_urls(self):
        assert _url_hash("https://a.com") != _url_hash("https://b.com")


class TestProcessorMetrics:
    def test_defaults_all_zero(self):
        m = ProcessorMetrics()
        assert m.total_docs == m.chunks_created == m.duplicates_skipped == m.content_updates == 0

    def test_report_keys(self):
        r = ProcessorMetrics(total_docs=3).report()
        assert set(r.keys()) == {
            "total_docs", "chunks_created", "duplicates_skipped", "content_updates"
        }

    def test_report_values(self):
        m = ProcessorMetrics(total_docs=2, chunks_created=5, duplicates_skipped=1)
        r = m.report()
        assert r["total_docs"] == 2 and r["chunks_created"] == 5 and r["duplicates_skipped"] == 1


class TestAlphaProcessorInit:
    def test_default_chunk_size(self):
        assert AlphaProcessor().splitter._chunk_size == 512

    def test_default_overlap(self):
        assert AlphaProcessor().splitter._chunk_overlap == 64

    def test_custom_chunk_size(self):
        assert AlphaProcessor(chunk_size=256, chunk_overlap=32).splitter._chunk_size == 256

    def test_seen_starts_empty(self):
        assert AlphaProcessor()._seen == {}


class TestProcess:
    def test_empty_input(self):
        assert AlphaProcessor().process([]) == []

    def test_metrics_reset_each_call(self, raw_doc, raw_doc_duplicate):
        p = AlphaProcessor()
        p.process([raw_doc])
        p.process([raw_doc_duplicate])
        assert p.metrics.total_docs == 1  # reset, not accumulated

    def test_single_doc_creates_chunks(self, raw_doc):
        chunks = AlphaProcessor().process([raw_doc])
        assert len(chunks) >= 1
        assert all(isinstance(c, ProcessedChunk) for c in chunks)

    def test_chunk_text_non_empty(self, raw_doc):
        for c in AlphaProcessor().process([raw_doc]):
            assert c.text.strip() != ""

    def test_chunk_metadata_required_keys(self, raw_doc):
        required = {
            "content_hash", "url_hash", "ticker", "source_type",
            "published_at_utc", "ingested_at", "chunk_index", "url", "title",
        }
        for c in AlphaProcessor().process([raw_doc]):
            assert required <= set(c.metadata.keys())

    def test_ticker_in_metadata(self, raw_doc):
        for c in AlphaProcessor().process([raw_doc]):
            assert c.metadata["ticker"] == "NVDA"

    def test_url_in_metadata(self, raw_doc):
        for c in AlphaProcessor().process([raw_doc]):
            assert c.metadata["url"] == raw_doc.url

    def test_chunk_index_increments(self, raw_doc):
        chunks = AlphaProcessor().process([raw_doc])
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_chunks_created_count(self, raw_doc):
        p = AlphaProcessor()
        chunks = p.process([raw_doc])
        assert p.metrics.chunks_created == len(chunks)


class TestExactDuplicate:
    def test_duplicate_returns_empty(self, raw_doc, raw_doc_duplicate):
        p = AlphaProcessor()
        p.process([raw_doc])
        assert p.process([raw_doc_duplicate]) == []

    def test_duplicate_increments_skipped(self, raw_doc, raw_doc_duplicate):
        p = AlphaProcessor()
        p.process([raw_doc])
        p.process([raw_doc_duplicate])
        assert p.metrics.duplicates_skipped == 1

    def test_same_doc_twice_in_batch(self, raw_doc):
        from rag.loader import RawDocument
        dup = RawDocument(
            title=raw_doc.title, content=raw_doc.content, url=raw_doc.url,
            source_type=raw_doc.source_type, ticker=raw_doc.ticker,
            published_at=raw_doc.published_at,
        )
        p = AlphaProcessor()
        p.process([raw_doc, dup])
        assert p.metrics.duplicates_skipped == 1


class TestContentUpdate:
    def test_updated_content_generates_chunks(self, raw_doc, raw_doc_updated_content):
        p = AlphaProcessor()
        p.process([raw_doc])
        assert len(p.process([raw_doc_updated_content])) >= 1

    def test_content_update_metric(self, raw_doc, raw_doc_updated_content):
        p = AlphaProcessor()
        p.process([raw_doc])
        p.process([raw_doc_updated_content])
        assert p.metrics.content_updates == 1

    def test_seen_hash_updated(self, raw_doc, raw_doc_updated_content):
        p = AlphaProcessor()
        p.process([raw_doc])
        old_hash = p._seen[_url_hash(raw_doc.url)]
        p.process([raw_doc_updated_content])
        assert p._seen[_url_hash(raw_doc.url)] != old_hash
```

---

## File Path

`tests/memory/test_manager_memory.py`

### Objective

Test all three memory classes. Critical paths: FIFO trim, cap eviction, ticker uppercase normalisation, Supabase error recovery, `recall()` payload structure.

### Expected Result

Cap limits enforced; ticker always uppercase; `recall()` has `heuristics`, `user_preferences`, `total_tickers_cached`; Supabase errors in `load()` result in empty state, not raised exceptions.

### Python Code

```python
"""tests/memory/test_manager_memory.py"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from memory.manager_memory import (
    AgentExecutionRecord,
    EvaluationFeedback,
    LongTermMemory,
    ManagerMemory,
    ShortTermMemory,
)


# ============================================================================
# ShortTermMemory
# ============================================================================

class TestShortTermMemoryReset:
    def test_reset_clears_messages(self):
        stm = ShortTermMemory()
        stm.add_message("user", "hello")
        stm.reset("s1", "new query")
        assert stm.get_messages() == []

    def test_reset_sets_session_id(self):
        stm = ShortTermMemory()
        stm.reset("sess-abc", "query")
        assert stm.session_id == "sess-abc"

    def test_reset_sets_task_query(self):
        stm = ShortTermMemory()
        stm.reset("s1", "Analyse NVDA")
        assert stm.task_query == "Analyse NVDA"

    def test_reset_clears_agent_log(self):
        stm = ShortTermMemory()
        stm.reset("s1", "q")
        stm.log_dispatch("ResearchAgent", {})
        stm.reset("s2", "q2")
        assert stm.get_agent_log() == []

    def test_reset_clears_eval_feedback(self):
        stm = ShortTermMemory()
        stm.reset("s1", "q")
        stm.add_evaluation(EvaluationFeedback(
            step="r", timestamp=1.0, passed=True,
            score=80, issues=[], next_action="proceed", raw_verdict="{}",
        ))
        stm.reset("s2", "q2")
        assert stm.get_evaluations() == []


class TestShortTermMemoryMessages:
    def test_add_and_retrieve(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        stm.add_message("user", "hello")
        assert stm.get_messages() == [{"role": "user", "content": "hello"}]

    def test_get_messages_returns_copy(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        stm.add_message("user", "x")
        copy = stm.get_messages()
        copy.append({"role": "system", "content": "injected"})
        assert len(stm.get_messages()) == 1

    def test_fifo_trim_at_max(self):
        stm = ShortTermMemory(max_messages=3)
        stm.reset("s", "q")
        for i in range(5):
            stm.add_message("user", f"msg{i}")
        msgs = stm.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg2"
        assert msgs[-1]["content"] == "msg4"

    def test_no_trim_below_max(self):
        stm = ShortTermMemory(max_messages=10)
        stm.reset("s", "q")
        for i in range(5):
            stm.add_message("user", f"msg{i}")
        assert len(stm.get_messages()) == 5


class TestShortTermMemoryDispatch:
    def test_returns_record(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        r = stm.log_dispatch("ResearchAgent", {"ticker": "NVDA"})
        assert isinstance(r, AgentExecutionRecord)
        assert r.agent_name == "ResearchAgent"
        assert r.outcome    == "pending"

    def test_directives_deep_copied(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        dirs = {"ticker": "AAPL"}
        r = stm.log_dispatch("FinancialAgent", dirs)
        dirs["ticker"] = "CHANGED"
        assert r.directives["ticker"] == "AAPL"

    def test_get_last_dispatch_empty(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        assert stm.get_last_dispatch() is None

    def test_get_last_dispatch_returns_latest(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        stm.log_dispatch("AgentA", {})
        r = stm.log_dispatch("AgentB", {})
        assert stm.get_last_dispatch() is r

    def test_agents_run(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        stm.log_dispatch("ResearchAgent", {})
        stm.log_dispatch("FinancialAgent", {})
        assert stm.agents_run() == ["ResearchAgent", "FinancialAgent"]


class TestShortTermMemoryEvaluation:
    def _fb(self, step="research", passed=True, score=80):
        return EvaluationFeedback(
            step=step, timestamp=time.time(), passed=passed,
            score=score, issues=[], next_action="proceed", raw_verdict="{}",
        )

    def test_add_and_get_last(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        fb = self._fb()
        stm.add_evaluation(fb)
        assert stm.get_last_evaluation() is fb

    def test_empty_returns_none(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        assert stm.get_last_evaluation() is None

    def test_get_all_evaluations(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        stm.add_evaluation(self._fb("step1"))
        stm.add_evaluation(self._fb("step2"))
        assert [e.step for e in stm.get_evaluations()] == ["step1", "step2"]


class TestShortTermMemoryContextDict:
    def test_required_keys(self):
        stm = ShortTermMemory()
        stm.reset("s1", "Analyse AAPL")
        ctx = stm.to_context_dict()
        for k in ("session_id", "task_query", "session_elapsed_s",
                   "agents_dispatched", "last_evaluation"):
            assert k in ctx

    def test_no_evaluations_is_none(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        assert stm.to_context_dict()["last_evaluation"] is None

    def test_agents_dispatched_reflects_log(self):
        stm = ShortTermMemory()
        stm.reset("s", "q")
        r = stm.log_dispatch("ResearchAgent", {"ticker": "NVDA"})
        r.outcome = "success"
        agents = stm.to_context_dict()["agents_dispatched"]
        assert len(agents) == 1
        assert agents[0]["agent"]   == "ResearchAgent"
        assert agents[0]["outcome"] == "success"


# ============================================================================
# LongTermMemory  (Supabase mocked)
# ============================================================================

@pytest.fixture()
def ltm(mock_supabase):
    return LongTermMemory(user_id="test_user", supabase_client=mock_supabase)


@pytest.fixture()
def ltm_with_row(mock_supabase_with_row):
    return LongTermMemory(user_id="test_user", supabase_client=mock_supabase_with_row)


class TestLongTermMemoryHeuristics:
    def test_store_and_retrieve(self, ltm):
        ltm.store_heuristic("depth", "advanced")
        assert ltm.get_heuristic("depth") == "advanced"

    def test_missing_key_returns_default(self, ltm):
        assert ltm.get_heuristic("nope", default=42) == 42

    def test_missing_key_default_none(self, ltm):
        assert ltm.get_heuristic("missing") is None

    def test_overwrite_existing(self, ltm):
        ltm.store_heuristic("key", "v1")
        ltm.store_heuristic("key", "v2")
        assert ltm.get_heuristic("key") == "v2"

    def test_cap_eviction_fifo(self, mock_supabase):
        ltm = LongTermMemory(user_id="u", supabase_client=mock_supabase, max_heuristics=3)
        ltm.store_heuristic("k1", "v1")
        ltm.store_heuristic("k2", "v2")
        ltm.store_heuristic("k3", "v3")
        ltm.store_heuristic("k4", "v4")  # evicts k1
        assert ltm.get_heuristic("k1") is None
        assert ltm.get_heuristic("k4") == "v4"

    def test_get_all_returns_copy(self, ltm):
        ltm.store_heuristic("x", 1)
        copy = ltm.get_all_heuristics()
        copy["y"] = 2
        assert "y" not in ltm.operational_heuristics


class TestLongTermMemoryTickerInsights:
    def test_store_and_retrieve(self, ltm):
        ltm.store_ticker_insight("NVDA", {"sector": "Tech"})
        assert ltm.get_ticker_insight("NVDA")["sector"] == "Tech"

    def test_uppercase_normalisation(self, ltm):
        ltm.store_ticker_insight("nvda", {"sector": "Tech"})
        assert ltm.get_ticker_insight("NVDA")["sector"] == "Tech"

    def test_insight_merge(self, ltm):
        ltm.store_ticker_insight("AAPL", {"sector": "Tech"})
        ltm.store_ticker_insight("AAPL", {"grade": "A"})
        r = ltm.get_ticker_insight("AAPL")
        assert r["sector"] == "Tech" and r["grade"] == "A"

    def test_last_updated_set(self, ltm):
        ltm.store_ticker_insight("MSFT", {"foo": "bar"})
        assert "last_updated" in ltm.get_ticker_insight("MSFT")

    def test_missing_ticker_empty_dict(self, ltm):
        assert ltm.get_ticker_insight("XYZ") == {}

    def test_get_returns_copy(self, ltm):
        ltm.store_ticker_insight("AMZN", {"k": "v"})
        copy = ltm.get_ticker_insight("AMZN")
        copy["injected"] = True
        assert "injected" not in ltm.get_ticker_insight("AMZN")

    def test_cap_eviction(self, mock_supabase):
        ltm = LongTermMemory(user_id="u", supabase_client=mock_supabase,
                              max_ticker_insights=2)
        ltm.store_ticker_insight("T1", {"v": 1})
        ltm.store_ticker_insight("T2", {"v": 2})
        ltm.store_ticker_insight("T3", {"v": 3})  # evicts T1
        assert ltm.get_ticker_insight("T1") == {}
        assert ltm.get_ticker_insight("T3")["v"] == 3


class TestLongTermMemoryPreferences:
    def test_store_and_get(self, ltm):
        ltm.store_preference("format", "concise")
        assert ltm.get_preference("format") == "concise"

    def test_missing_returns_default(self, ltm):
        assert ltm.get_preference("absent", default="x") == "x"

    def test_overwrite(self, ltm):
        ltm.store_preference("k", "old")
        ltm.store_preference("k", "new")
        assert ltm.get_preference("k") == "new"

    def test_get_all_returns_copy(self, ltm):
        ltm.store_preference("x", 1)
        copy = ltm.get_all_preferences()
        copy["y"] = 2
        assert "y" not in ltm.user_preferences


class TestLongTermMemoryRecall:
    def test_recall_keys(self, ltm):
        r = ltm.recall()
        assert "heuristics"           in r
        assert "user_preferences"     in r
        assert "total_tickers_cached" in r

    def test_recall_with_ticker(self, ltm):
        ltm.store_ticker_insight("GOOG", {"pe": 25})
        r = ltm.recall(ticker="GOOG")
        assert "ticker_insight" in r
        assert r["ticker_insight"]["pe"] == 25

    def test_recall_without_ticker_no_insight_key(self, ltm):
        assert "ticker_insight" not in ltm.recall()

    def test_total_tickers_cached(self, ltm):
        ltm.store_ticker_insight("A1", {})
        ltm.store_ticker_insight("A2", {})
        assert ltm.recall()["total_tickers_cached"] == 2


class TestLongTermMemoryPersistence:
    def test_load_existing_row(self, ltm_with_row):
        assert ltm_with_row.operational_heuristics == {"key1": "val1"}
        assert ltm_with_row.ticker_insights        == {"NVDA": {"sector": "Tech"}}
        assert ltm_with_row.user_preferences       == {"format": "concise"}

    def test_load_empty_when_no_row(self, ltm):
        assert ltm.operational_heuristics == {}
        assert ltm.ticker_insights        == {}
        assert ltm.user_preferences       == {}

    def test_load_graceful_on_error(self, mock_supabase):
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.side_effect = (
            RuntimeError("connection refused")
        )
        ltm = LongTermMemory(user_id="u", supabase_client=mock_supabase)
        assert ltm.operational_heuristics == {}

    def test_persist_calls_upsert(self, ltm, mock_supabase):
        ltm.store_heuristic("key", "val")
        ltm.persist()
        mock_supabase.table.return_value.upsert.assert_called_once()

    def test_persist_raises_on_error(self, mock_supabase):
        mock_supabase.table.return_value.upsert.return_value.execute.side_effect = (
            RuntimeError("write failed")
        )
        ltm = LongTermMemory(user_id="u", supabase_client=mock_supabase)
        with pytest.raises(RuntimeError, match="write failed"):
            ltm.persist()


# ============================================================================
# ManagerMemory (facade)
# ============================================================================

@pytest.fixture()
def mem(mock_supabase):
    return ManagerMemory(user_id="test_user", supabase_client=mock_supabase)


class TestManagerMemorySession:
    def test_new_session_resets_short_term(self, mem):
        mem.add_message("user", "hello")
        mem.new_session("s2", "new query")
        assert mem.get_messages() == []

    def test_new_session_preserves_long_term(self, mem):
        mem.store_heuristic("depth", "advanced")
        mem.new_session("s2", "new query")
        assert mem.get_heuristic("depth") == "advanced"

    def test_session_id_propagated(self, mem):
        mem.new_session("sess-xyz", "Analyse TSLA")
        assert mem.short.session_id == "sess-xyz"


class TestManagerMemoryDelegation:
    def test_add_message(self, mem):
        mem.new_session("s", "q")
        mem.add_message("assistant", "done")
        assert mem.get_messages()[0]["role"] == "assistant"

    def test_log_dispatch(self, mem):
        mem.new_session("s", "q")
        r = mem.log_dispatch("SentimentAgent", {"ticker": "TSLA"})
        assert isinstance(r, AgentExecutionRecord)
        assert mem.agents_run() == ["SentimentAgent"]

    def test_add_evaluation(self, mem):
        mem.new_session("s", "q")
        fb = EvaluationFeedback(
            step="sentiment", timestamp=time.time(), passed=True,
            score=90, issues=[], next_action="finalise", raw_verdict="{}"
        )
        mem.add_evaluation(fb)
        assert mem.get_last_evaluation() is fb

    def test_store_get_heuristic(self, mem):
        mem.store_heuristic("h", "v")
        assert mem.get_heuristic("h") == "v"

    def test_store_get_ticker_insight(self, mem):
        mem.store_ticker_insight("nvda", {"grade": "A"})
        assert mem.get_ticker_insight("NVDA")["grade"] == "A"

    def test_store_get_preference(self, mem):
        mem.store_preference("fmt", "verbose")
        assert mem.get_preference("fmt") == "verbose"


class TestManagerMemoryRecall:
    def test_has_short_and_long_keys(self, mem):
        mem.new_session("s", "q")
        r = mem.recall()
        assert "short_term" in r and "long_term" in r

    def test_recall_with_ticker(self, mem):
        mem.store_ticker_insight("AAPL", {"pe": 28})
        assert mem.recall(ticker="AAPL")["long_term"]["ticker_insight"]["pe"] == 28
```

---

## File Path

`tests/api/test_dependencies.py`

### Objective

Test `api/dependencies.py`: `get_user_id()` header-vs-fallback resolution.

### Expected Result

Header `X-User-Id` takes precedence over `settings.DEFAULT_USER_ID`.

### Python Code

```python
"""tests/api/test_dependencies.py"""
from __future__ import annotations

import pytest
from fastapi import Request

from api.dependencies import get_user_id
from api.config import settings


class TestGetUserId:
    def _make_request(self, headers: dict) -> Request:
        scope = {
            "type":         "http",
            "method":       "GET",
            "path":         "/",
            "headers":      [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
        }
        return Request(scope)

    def test_returns_x_user_id_header(self):
        req = self._make_request({"X-User-Id": "user_abc123"})
        assert get_user_id(req) == "user_abc123"

    def test_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr(settings, "DEFAULT_USER_ID", "fallback_user")
        req = self._make_request({})
        assert get_user_id(req) == "fallback_user"

    def test_header_overrides_default(self, monkeypatch):
        monkeypatch.setattr(settings, "DEFAULT_USER_ID", "default")
        req = self._make_request({"X-User-Id": "explicit_user"})
        assert get_user_id(req) == "explicit_user"
```

---

## Code Review Section

### Potential Bugs

**B1 — Broken module-level imports in `fear_greed_calculator.py` (lines 56–57)**

```python
# WRONG — tools/finbert_analyzer.py does not exist
from tools.finbert_analyzer import FinBertResult
from tools.vader_scorer     import VaderResult
```

Files live at `tools/sentiment_tools/finbert_analyzer.py` and `tools/sentiment_tools/vader_scorer.py`. Importing `tools.sentiment_tools.fear_greed_calculator` will raise `ModuleNotFoundError` at process startup. The same wrong paths are repeated inside `calculate_from_dict()` as local imports. Fix:

```python
from tools.sentiment_tools.finbert_analyzer import FinBertResult
from tools.sentiment_tools.vader_scorer      import VaderResult
```

---

**B2 — `validate_settings()` calls `sys.exit(1)` directly**

`api/config.py:94` calls `sys.exit(1)` on missing env vars. This is impossible to test without patching `sys.exit` and loses specific information about which key was missing. Recommended pattern: raise a custom `ConfigurationError("Missing: ANTHROPIC_API_KEY")` and let the `lifespan` handler translate it to `sys.exit(1)`.

---

**B3 — `LongTermMemory.__init__` calls `self.load()` unconditionally**

Constructing `LongTermMemory` without a valid Supabase connection immediately executes a SELECT. Any unit test that instantiates the class directly — without injecting a mock — will make a real network call or raise. The `load()` call should be explicit and separated from construction.

---

**B4 — Dead `_label()` call in `price_to_earnings()`**

`price_to_earnings()` calls `_label()` at line 110 to produce `interp`, then immediately overwrites it with an explicit `if/elif/else` chain. The `_label()` return value is discarded. The dead call is misleading because it uses the wrong threshold direction (`higher_is_better=False` but ascending-ordered thresholds). Remove the dead `_label()` call.

---

**B5 — `AlphaProcessor._seen` grows without bound**

`process()` resets `self.metrics` but not `self._seen`. The in-memory dedup store accumulates every URL hash ever ingested in a single process lifetime. For a long-running server this is an unbounded memory leak. A TTL-based eviction or maximum size with LRU should be documented and enforced.

---

### Design Issues

**D1 — `with_error_reporting` breaks on `classmethod` / `staticmethod`**

`inspect.iscoroutinefunction(fn)` is called on the raw unbound callable. Applying the decorator to a `classmethod` before it is bound to a class will misclassify it as sync. The decorator works correctly only on free functions and regular instance methods.

**D2 — `ManagerMemory` manually delegates every method**

`ManagerMemory` re-implements 14 delegation methods one-by-one. Adding any new method to `ShortTermMemory` or `LongTermMemory` requires three changes: the sub-layer, the facade, and the facade docstring. A `Protocol` interface for each sub-layer would enforce the contract and reduce boilerplate.

**D3 — `settings` singleton created at import time**

`api/config.py:70` creates `settings = Settings()` at module import. Tests that vary env vars must patch `api.config.settings` in-place via `monkeypatch.setattr`. If a test fails to restore the patch, all subsequent tests in the same process see the mutated singleton.

**D4 — `LocalSocialDataRetriever` is undocumented legacy**

`tools/sentiment_tools/local_social_retriever.py` is explicitly marked "legacy" in its docstring but still shipped. No deprecation warning, no `@deprecated` decorator, no tracking issue. It expands the maintenance surface and will confuse future contributors.

---

### Testability Issues

**T1 — `AlphaEmbedder` singleton cannot be reset between tests**

`rag/embedding_manager.py` stores the singleton in a class-level `_instance`. Once initialised in one test, it persists for the entire process lifetime. Tests requiring different model configurations must use `monkeypatch.setattr(AlphaEmbedder, "_instance", None)` — fragile and undocumented.

**T2 — `VaderLexiconScorer._ensure_vader_lexicon()` uses a module-level flag**

`_VADER_LOADED = False` is set once per process. Tests that want to verify the download branch must manually reset this flag via `monkeypatch.setattr`. The double-checked locking pattern makes this non-obvious.

**T3 — All agents require live MCP subprocess connections**

`ResearchAgent`, `FinancialAnalystAgent`, and `SentimentAgent` spawn real stdio subprocesses in `run()`. There is no seam — no interface, factory, or injectable MCP client — that allows mock injection. Agent-layer tests are therefore always integration tests requiring all tool-server executables to be running.

**T4 — `ManagerAgent` brain passes call the live Anthropic API inline**

`_brain_route()`, `_brain_evaluate()`, and `_brain_finalise()` create an `AsyncAnthropic()` client inline. There is no constructor injection or factory method. Unit-testing routing logic requires patching `anthropic.AsyncAnthropic` at the module level, coupling tests tightly to the SDK's internal class name.
