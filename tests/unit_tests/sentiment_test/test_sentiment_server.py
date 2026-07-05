"""
Tests for: tools/sentiment_tools/sentiment_server.py
Phase: 2c — Sentiment Tools (MCP server dispatch layer)

Scope decision: FinBertSentimentAnalyzer/VaderLexiconScorer/FearGreedIndexCalculator
are fully tested in their own files. This file only tests:
  1. call_tool() routing for all 4 tool names + the "unknown tool" path.
  2. _retrieve_social_data() chunk/metadata extraction logic (pure, testable).
  3. _to_dict() dataclass-to-JSON-safe-dict recursive conversion (pure).
  4. The analyze_finbert batch_size save/restore behavior (a subtle stateful
     side effect worth locking in with a test).
  5. Sentry capture on exception, same pattern as the other two servers.
"""
import json
import dataclasses
from unittest.mock import patch, AsyncMock, MagicMock
import pytest

from tools.sentiment_tools.sentiment_server import (
    call_tool,
    _retrieve_social_data,
    _to_dict,
)


# ---------------------------------------------------------------------------
# _retrieve_social_data (pure transformation logic)
# ---------------------------------------------------------------------------

class TestRetrieveSocialData:
    @patch("tools.sentiment_tools.sentiment_server._get_retriever")
    def test_extracts_text_and_metadata_in_parallel(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = [
            {"text": "NVDA is mooning", "ticker": "NVDA", "source_type": "reddit",
             "published_at": "2024-11-01", "url": "https://x.com", "title": "t",
             "rrf_score": 0.9},
        ]
        mock_get_retriever.return_value = mock_retriever

        result = _retrieve_social_data("NVDA sentiment", ticker="NVDA", days_back=7)

        assert result["chunks"] == ["NVDA is mooning"]
        assert result["sources_metadata"][0]["ticker"] == "NVDA"
        assert result["total_retrieved"] == 1

    @patch("tools.sentiment_tools.sentiment_server._get_retriever")
    def test_chunks_with_empty_text_are_skipped(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = [
            {"text": "   ", "ticker": "NVDA"},
            {"text": "real content", "ticker": "NVDA"},
        ]
        mock_get_retriever.return_value = mock_retriever

        result = _retrieve_social_data("query", ticker="NVDA", days_back=7)

        assert result["total_retrieved"] == 1
        assert result["chunks"] == ["real content"]

    @patch("tools.sentiment_tools.sentiment_server._get_retriever")
    def test_rrf_score_falls_back_to_freshness_score(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = [
            {"text": "x", "freshness_score": 0.42},  # no "rrf_score" key
        ]
        mock_get_retriever.return_value = mock_retriever

        result = _retrieve_social_data("q", ticker=None, days_back=7)
        assert result["sources_metadata"][0]["rrf_score"] == 0.42

    @patch("tools.sentiment_tools.sentiment_server._get_retriever")
    def test_no_chunks_returned_gives_empty_result(self, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = []
        mock_get_retriever.return_value = mock_retriever

        result = _retrieve_social_data("q", ticker=None, days_back=7)
        assert result == {"chunks": [], "sources_metadata": [], "total_retrieved": 0}


# ---------------------------------------------------------------------------
# _to_dict (pure recursive conversion)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Inner:
    x: int

@dataclasses.dataclass
class _Outer:
    inner: _Inner
    items: list

class TestToDict:
    def test_converts_nested_dataclass_recursively(self):
        obj = _Outer(inner=_Inner(x=1), items=[_Inner(x=2), _Inner(x=3)])
        result = _to_dict(obj)
        assert result == {"inner": {"x": 1}, "items": [{"x": 2}, {"x": 3}]}

    def test_primitives_passed_through_unchanged(self):
        assert _to_dict(42) == 42
        assert _to_dict("hello") == "hello"
        assert _to_dict(None) is None

    def test_plain_dict_recurses_into_values(self):
        obj = {"a": _Inner(x=5), "b": [1, 2]}
        result = _to_dict(obj)
        assert result == {"a": {"x": 5}, "b": [1, 2]}

    def test_class_itself_not_instance_is_not_treated_as_dataclass(self):
        # dataclasses.is_dataclass(_Inner) is True for the class too, but the
        # `not isinstance(obj, type)` guard should prevent calling asdict()
        # on the class object itself.
        assert _to_dict(_Inner) is _Inner


# ---------------------------------------------------------------------------
# call_tool — routing
# ---------------------------------------------------------------------------

class TestCallToolRouting:
    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server._retrieve_social_data")
    async def test_routes_to_retrieve_social_data_with_defaults(self, mock_retrieve):
        mock_retrieve.return_value = {"chunks": [], "sources_metadata": [], "total_retrieved": 0}

        result = await call_tool("retrieve_social_data", {"query": "NVDA"})

        mock_retrieve.assert_called_once_with(query="NVDA", ticker=None, days_back=7)
        assert result.isError is False

    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server._get_finbert")
    async def test_routes_to_analyze_finbert_and_restores_batch_size(self, mock_get_finbert):
        analyzer = MagicMock()
        analyzer.batch_size = 16
        analyzer.analyze.return_value = MagicMock()
        mock_get_finbert.return_value = analyzer

        await call_tool("analyze_finbert", {"texts": ["a", "b"], "batch_size": 4})

        # batch_size temporarily set to 4 during the call...
        analyzer.analyze.assert_called_once_with(["a", "b"])
        # ...then restored to its original value (16) afterward.
        assert analyzer.batch_size == 16

    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server._get_vader")
    async def test_routes_to_score_vader(self, mock_get_vader):
        scorer = MagicMock()
        scorer.score.return_value = MagicMock()
        mock_get_vader.return_value = scorer

        await call_tool("score_vader", {"texts": ["a", "b"]})

        scorer.score.assert_called_once_with(texts=["a", "b"])

    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server._get_fear_greed")
    async def test_routes_to_calculate_fear_greed_uses_singleton_when_no_weights(
        self, mock_get_fear_greed
    ):
        calculator = MagicMock()
        calculator.calculate_from_dict.return_value = MagicMock()
        mock_get_fear_greed.return_value = calculator

        await call_tool("calculate_fear_greed", {
            "finbert_result": {"bullish_prob": 0.5}, "vader_result": {"compound": 0.1},
        })

        mock_get_fear_greed.assert_called_once()
        calculator.calculate_from_dict.assert_called_once_with(
            finbert_dict={"bullish_prob": 0.5}, vader_dict={"compound": 0.1},
        )

    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server.FearGreedIndexCalculator")
    @patch("tools.sentiment_tools.sentiment_server._get_fear_greed")
    async def test_calculate_fear_greed_builds_custom_calculator_when_weights_given(
        self, mock_get_fear_greed, mock_calc_cls
    ):
        custom_calc = MagicMock()
        custom_calc.calculate_from_dict.return_value = MagicMock()
        mock_calc_cls.return_value = custom_calc

        await call_tool("calculate_fear_greed", {
            "finbert_result": {}, "vader_result": {},
            "finbert_weight": 0.5, "vader_weight": 0.5,
        })

        mock_calc_cls.assert_called_once_with(finbert_weight=0.5, vader_weight=0.5)
        mock_get_fear_greed.assert_not_called()  # singleton bypassed
        custom_calc.calculate_from_dict.assert_called_once()


# ---------------------------------------------------------------------------
# Unknown tool / missing args
# ---------------------------------------------------------------------------

class TestUnknownToolAndErrors:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result(self):
        result = await call_tool("not_a_real_tool", {})
        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert "Unknown tool" in payload["error"]

    @pytest.mark.asyncio
    async def test_missing_required_arg_is_caught_not_raised(self):
        # "texts" is required for analyze_finbert; omitting it raises KeyError
        # internally, which must be caught and wrapped, not propagated.
        result = await call_tool("analyze_finbert", {})
        assert result.isError is True
        payload = json.loads(result.content[0].text)
        assert payload["tool"] == "analyze_finbert"


# ---------------------------------------------------------------------------
# Sentry capture on exception
# ---------------------------------------------------------------------------

class TestSentryCaptureOnException:
    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server._get_vader")
    @patch("tools.sentiment_tools.sentiment_server.sentry_enabled", return_value=True)
    async def test_sentry_captures_with_correct_tags(self, mock_enabled, mock_get_vader):
        mock_get_vader.side_effect = RuntimeError("model crashed")

        with patch("sentry_sdk.push_scope") as mock_push_scope, \
             patch("sentry_sdk.capture_exception") as mock_capture_exc:
            scope = MagicMock()
            mock_push_scope.return_value.__enter__.return_value = scope

            result = await call_tool("score_vader", {"texts": ["x"]})

            scope.set_tag.assert_any_call("tool", "score_vader")
            scope.set_tag.assert_any_call("server", "sentiment-agent-mcp")
            mock_capture_exc.assert_called_once()
            assert result.isError is True

    @pytest.mark.asyncio
    @patch("tools.sentiment_tools.sentiment_server._get_vader")
    @patch("tools.sentiment_tools.sentiment_server.sentry_enabled", return_value=False)
    async def test_sentry_not_invoked_when_disabled(self, mock_enabled, mock_get_vader):
        mock_get_vader.side_effect = RuntimeError("model crashed")

        with patch("sentry_sdk.capture_exception") as mock_capture_exc:
            result = await call_tool("score_vader", {"texts": ["x"]})
            mock_capture_exc.assert_not_called()
            assert result.isError is True

    @pytest.mark.asyncio
    async def test_error_payload_includes_original_arguments(self):
        result = await call_tool("analyze_finbert", {"unexpected": "value"})
        payload = json.loads(result.content[0].text)
        assert payload["arguments"] == {"unexpected": "value"}