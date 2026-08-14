"""
Tests for: core/error_handler.py
Phase: 5 — Core Infrastructure

Mocking strategy: sentry_enabled() is mocked at the core.observability
import point used by this module. sentry_sdk.add_breadcrumb/push_scope/
capture_exception are mocked individually per test. We verify both the
async and sync code paths since the decorator branches on
inspect.iscoroutinefunction(fn).
"""
from unittest.mock import patch, MagicMock
import pytest

from core.error_handler import (
    with_error_reporting,
    _add_breadcrumb,
    _capture,
    _safe_extra,
)


# ---------------------------------------------------------------------------
# _safe_extra
# ---------------------------------------------------------------------------

class TestSafeExtra:
    def test_keeps_json_safe_scalars(self):
        result = _safe_extra({"a": "str", "b": 1, "c": 1.5, "d": True, "e": None})
        assert result == {"a": "str", "b": 1, "c": 1.5, "d": True, "e": None}

    def test_drops_non_scalar_values(self):
        result = _safe_extra({"a": [1, 2], "b": {"x": 1}, "c": object()})
        assert result == {}

    def test_empty_dict_returns_empty(self):
        assert _safe_extra({}) == {}


# ---------------------------------------------------------------------------
# _add_breadcrumb
# ---------------------------------------------------------------------------

class TestAddBreadcrumb:
    def test_noop_when_sentry_disabled(self):
        with patch("core.error_handler.sentry_enabled", return_value=False), \
             patch("sentry_sdk.add_breadcrumb") as mock_bc:
            _add_breadcrumb("comp", "fn_name")
            mock_bc.assert_not_called()

    def test_adds_breadcrumb_when_enabled(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.add_breadcrumb") as mock_bc:
            _add_breadcrumb("comp", "fn_name", {"k": "v"})
            mock_bc.assert_called_once_with(
                category="comp", message="fn_name", data={"k": "v"}, level="info"
            )

    def test_extra_defaults_to_empty_dict_when_none(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.add_breadcrumb") as mock_bc:
            _add_breadcrumb("comp", "fn_name", None)
            _, kwargs = mock_bc.call_args
            assert kwargs["data"] == {}

    def test_exception_in_sentry_call_is_swallowed(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.add_breadcrumb", side_effect=RuntimeError("sdk error")):
            _add_breadcrumb("comp", "fn_name")  # must not raise


# ---------------------------------------------------------------------------
# _capture
# ---------------------------------------------------------------------------

class TestCapture:
    def test_noop_when_sentry_disabled(self):
        with patch("core.error_handler.sentry_enabled", return_value=False), \
             patch("sentry_sdk.capture_exception") as mock_capture:
            _capture("comp", RuntimeError("boom"))
            mock_capture.assert_not_called()

    def test_captures_with_component_tag_when_enabled(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc:
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope
            exc = RuntimeError("boom")

            _capture("comp", exc)

            scope.set_tag.assert_called_once_with("component", "comp")
            mock_capture_exc.assert_called_once_with(exc)

    def test_exception_in_sentry_call_is_swallowed(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.push_scope", side_effect=RuntimeError("sdk down")):
            _capture("comp", RuntimeError("boom"))  # must not raise


# ---------------------------------------------------------------------------
# with_error_reporting — async function path
# ---------------------------------------------------------------------------

class TestDecoratorAsync:
    @pytest.mark.asyncio
    async def test_successful_call_returns_value_unchanged(self):
        @with_error_reporting(component="test.component")
        async def fn(x):
            return x * 2

        with patch("core.error_handler.sentry_enabled", return_value=False):
            result = await fn(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_exception_is_captured_and_reraised(self):
        @with_error_reporting(component="test.component")
        async def fn():
            raise ValueError("boom")

        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc, \
             patch("sentry_sdk.add_breadcrumb"):
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope

            with pytest.raises(ValueError, match="boom"):
                await fn()

            mock_capture_exc.assert_called_once()
            scope.set_tag.assert_called_once_with("component", "test.component")

    @pytest.mark.asyncio
    async def test_breadcrumb_added_on_entry_with_safe_kwargs(self):
        @with_error_reporting(component="test.component")
        async def fn(x, unsafe=None):
            return x

        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.add_breadcrumb") as mock_bc:
            await fn(1, unsafe=[1, 2, 3])

        _, kwargs = mock_bc.call_args
        assert kwargs["message"] == "fn"
        assert kwargs["category"] == "test.component"
        assert "unsafe" not in kwargs["data"]  # non-scalar dropped

    @pytest.mark.asyncio
    async def test_functools_wraps_preserves_function_name(self):
        @with_error_reporting(component="c")
        async def my_named_function():
            return 1

        assert my_named_function.__name__ == "my_named_function"

    @pytest.mark.asyncio
    async def test_no_exception_means_capture_never_called(self):
        @with_error_reporting(component="c")
        async def fn():
            return "ok"

        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.capture_exception") as mock_capture, \
             patch("sentry_sdk.add_breadcrumb"):
            await fn()
        mock_capture.assert_not_called()


# ---------------------------------------------------------------------------
# with_error_reporting — sync function path
# ---------------------------------------------------------------------------

class TestDecoratorSync:
    def test_successful_call_returns_value_unchanged(self):
        @with_error_reporting(component="test.component")
        def fn(x):
            return x * 2

        with patch("core.error_handler.sentry_enabled", return_value=False):
            result = fn(5)
        assert result == 10

    def test_exception_is_captured_and_reraised(self):
        @with_error_reporting(component="test.component")
        def fn():
            raise ValueError("boom")

        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc, \
             patch("sentry_sdk.add_breadcrumb"):
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope

            with pytest.raises(ValueError, match="boom"):
                fn()

            mock_capture_exc.assert_called_once()

    def test_sync_function_not_wrapped_as_coroutine(self):
        @with_error_reporting(component="c")
        def fn():
            return 1

        import inspect
        assert not inspect.iscoroutinefunction(fn)

    def test_functools_wraps_preserves_function_name(self):
        @with_error_reporting(component="c")
        def my_sync_function():
            return 1

        assert my_sync_function.__name__ == "my_sync_function"


# ---------------------------------------------------------------------------
# with_error_reporting.context — sync context manager
# ---------------------------------------------------------------------------

class TestSyncContext:
    def test_successful_block_runs_without_capture(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.capture_exception") as mock_capture, \
             patch("sentry_sdk.add_breadcrumb"):
            with with_error_reporting.context(component="test.block"):
                x = 1 + 1
        assert x == 2
        mock_capture.assert_not_called()

    def test_exception_in_block_is_captured_and_reraised(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc, \
             patch("sentry_sdk.add_breadcrumb"):
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope

            with pytest.raises(RuntimeError, match="boom"):
                with with_error_reporting.context(component="test.block"):
                    raise RuntimeError("boom")

            scope.set_tag.assert_called_once_with("component", "test.block")
            mock_capture_exc.assert_called_once()

    def test_breadcrumb_uses_block_placeholder_as_message(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.add_breadcrumb") as mock_bc:
            with with_error_reporting.context(component="test.block"):
                pass
        _, kwargs = mock_bc.call_args
        assert kwargs["message"] == "<block>"


# ---------------------------------------------------------------------------
# with_error_reporting.async_context — async context manager
# ---------------------------------------------------------------------------

class TestAsyncContext:
    @pytest.mark.asyncio
    async def test_successful_block_runs_without_capture(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.capture_exception") as mock_capture, \
             patch("sentry_sdk.add_breadcrumb"):
            async with with_error_reporting.async_context(component="test.block"):
                x = 1 + 1
        assert x == 2
        mock_capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_in_block_is_captured_and_reraised(self):
        with patch("core.error_handler.sentry_enabled", return_value=True), \
             patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc, \
             patch("sentry_sdk.add_breadcrumb"):
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope

            with pytest.raises(RuntimeError, match="boom"):
                async with with_error_reporting.async_context(component="test.block"):
                    raise RuntimeError("boom")

            mock_capture_exc.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_breadcrumb_when_sentry_disabled(self):
        with patch("core.error_handler.sentry_enabled", return_value=False), \
             patch("sentry_sdk.add_breadcrumb") as mock_bc:
            async with with_error_reporting.async_context(component="test.block"):
                pass
        mock_bc.assert_not_called()


# ---------------------------------------------------------------------------
# Static attribute wiring
# ---------------------------------------------------------------------------

class TestStaticAttributeWiring:
    def test_context_attribute_is_attached(self):
        assert hasattr(with_error_reporting, "context")

    def test_async_context_attribute_is_attached(self):
        assert hasattr(with_error_reporting, "async_context")