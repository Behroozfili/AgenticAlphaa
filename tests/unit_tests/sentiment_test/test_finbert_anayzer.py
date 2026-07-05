"""
Tests for: tools/sentiment_tools/finbert_analyzer.py
Phase: 2c — Sentiment Tools

Mocking strategy: torch and transformers (BertTokenizer/BertForSequenceClassification)
are mocked entirely — we never load the real 440 MB FinBERT model in tests.
We patch the module's _TOKENIZER/_MODEL/_DEVICE globals directly and patch
FinBertSentimentAnalyzer._infer_batch to return controlled probability
tensors, since mocking the internals of torch's softmax/no_grad chain is
fragile and provides little test value.

IMPORTANT: reset_finbert() must be called between tests (see autouse fixture)
since _TOKENIZER/_MODEL are module-level singletons — without resetting,
tests would leak mocked state into each other.
"""
from unittest.mock import patch, MagicMock
import pytest

import tools.sentiment_tools.finbert_analyzer as fb_module
from tools.sentiment_tools.finbert_analyzer import (
    FinBertSentimentAnalyzer,
    FinBertResult,
    ChunkSentiment,
    reset_finbert,
    _select_device,
)


@pytest.fixture(autouse=True)
def clean_finbert():
    """Ensure no singleton state leaks between tests."""
    reset_finbert()
    yield
    reset_finbert()


@pytest.fixture
def mock_model_load():
    """Bypass _load_model entirely so __init__ doesn't try to download anything."""
    with patch.object(FinBertSentimentAnalyzer, "_load_model", return_value=None):
        yield


# ---------------------------------------------------------------------------
# _select_device
# ---------------------------------------------------------------------------

class TestSelectDevice:
    @patch("tools.sentiment_tools.finbert_analyzer.torch")
    def test_returns_cpu_when_no_accelerator_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert _select_device() == "cpu"

    @patch("tools.sentiment_tools.finbert_analyzer.torch")
    def test_returns_cuda_when_available_and_allocatable(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.return_value = MagicMock()
        assert _select_device() == "cuda"

    @patch("tools.sentiment_tools.finbert_analyzer.torch")
    def test_falls_back_to_cpu_on_cuda_allocation_failure(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.side_effect = RuntimeError("OOM")
        mock_torch.backends.mps.is_available.return_value = False
        assert _select_device() == "cpu"


# ---------------------------------------------------------------------------
# analyze() — empty/skip handling
# ---------------------------------------------------------------------------

class TestAnalyzeEmptyHandling:
    def test_all_empty_texts_returns_neutral_zero_result(self, mock_model_load):
        analyzer = FinBertSentimentAnalyzer()
        result = analyzer.analyze(["", "   ", ""])

        assert isinstance(result, FinBertResult)
        assert result.label == "Neutral"
        assert result.bullish_prob == 0.0
        assert result.total_chunks == 0
        assert result.skipped_chunks == 3

    def test_mixed_empty_and_valid_texts_skips_only_empty(self, mock_model_load):
        analyzer = FinBertSentimentAnalyzer()
        with patch.object(
            analyzer, "_infer_batch",
            return_value=[[0.8, 0.1, 0.1]],
        ):
            result = analyzer.analyze(["", "valid text", "   "])
        assert result.total_chunks == 1
        assert result.skipped_chunks == 2


# ---------------------------------------------------------------------------
# analyze() — aggregation math
# ---------------------------------------------------------------------------

class TestAnalyzeAggregation:
    def test_mean_probabilities_computed_correctly(self, mock_model_load):
        analyzer = FinBertSentimentAnalyzer()
        # _IDX_POSITIVE=0, _IDX_NEGATIVE=1, _IDX_NEUTRAL=2
        with patch.object(
            analyzer, "_infer_batch",
            return_value=[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]],
        ):
            result = analyzer.analyze(["text a", "text b"])

        assert result.bullish_prob == pytest.approx(0.5, abs=1e-4)
        assert result.bearish_prob == pytest.approx(0.4, abs=1e-4)
        assert result.neutral_prob == pytest.approx(0.1, abs=1e-4)
        assert result.total_chunks == 2

    def test_corpus_label_is_argmax_of_means(self, mock_model_load):
        analyzer = FinBertSentimentAnalyzer()
        with patch.object(
            analyzer, "_infer_batch",
            return_value=[[0.9, 0.05, 0.05]],
        ):
            result = analyzer.analyze(["bullish text"])
        assert result.label == "Bullish"

    def test_batching_splits_texts_by_batch_size(self, mock_model_load):
        analyzer = FinBertSentimentAnalyzer(batch_size=2)
        texts = [f"text {i}" for i in range(5)]  # 5 texts, batch_size=2 -> 3 calls
        with patch.object(
            analyzer, "_infer_batch",
            return_value=[[0.5, 0.3, 0.2]] * 2,  # called per-batch; last call gets 1 item used
        ) as mock_infer:
            # Allow variable-length return per call
            mock_infer.side_effect = lambda batch: [[0.5, 0.3, 0.2]] * len(batch)
            result = analyzer.analyze(texts)
        assert mock_infer.call_count == 3  # ceil(5/2)
        assert result.total_chunks == 5

    def test_chunk_text_truncated_to_120_chars_with_ellipsis(self, mock_model_load):
        analyzer = FinBertSentimentAnalyzer()
        long_text = "x" * 200
        with patch.object(analyzer, "_infer_batch", return_value=[[0.4, 0.3, 0.3]]):
            result = analyzer.analyze([long_text])
        chunk = result.chunk_scores[0]
        assert chunk.text.endswith("...")
        assert len(chunk.text) == 123  # 120 chars + "..."


# ---------------------------------------------------------------------------
# _argmax_label / _empty_result (pure helpers)
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_argmax_label_bullish(self):
        assert FinBertSentimentAnalyzer._argmax_label(0.7, 0.2, 0.1) == "Bullish"

    def test_argmax_label_bearish(self):
        assert FinBertSentimentAnalyzer._argmax_label(0.1, 0.8, 0.1) == "Bearish"

    def test_argmax_label_neutral(self):
        assert FinBertSentimentAnalyzer._argmax_label(0.2, 0.2, 0.6) == "Neutral"

    def test_argmax_label_tie_breaks_by_dict_order(self):
        """Documents current behavior: a 3-way tie picks 'Bullish' first since
        Python's max() on equal values returns the first key in dict order
        ({"Bullish": ..., "Bearish": ..., "Neutral": ...})."""
        assert FinBertSentimentAnalyzer._argmax_label(0.33, 0.33, 0.33) == "Bullish"

    def test_empty_result_shape(self):
        result = FinBertSentimentAnalyzer._empty_result(skipped=3)
        assert result.label == "Neutral"
        assert result.skipped_chunks == 3
        assert result.chunk_scores == []


# ---------------------------------------------------------------------------
# _load_model — singleton behavior (no real model download)
# ---------------------------------------------------------------------------

class TestLoadModelSingleton:
    @patch("tools.sentiment_tools.finbert_analyzer.BertForSequenceClassification")
    @patch("tools.sentiment_tools.finbert_analyzer.BertTokenizer")
    @patch("tools.sentiment_tools.finbert_analyzer._select_device", return_value="cpu")
    def test_model_loaded_only_once_across_instances(
        self, mock_select_device, mock_tokenizer_cls, mock_model_cls
    ):
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        mock_model_cls.from_pretrained.return_value = MagicMock()

        FinBertSentimentAnalyzer()
        FinBertSentimentAnalyzer()

        # from_pretrained should only be called once even though we
        # instantiated the analyzer twice (singleton pattern).
        assert mock_tokenizer_cls.from_pretrained.call_count == 1
        assert mock_model_cls.from_pretrained.call_count == 1

    @patch("tools.sentiment_tools.finbert_analyzer.BertForSequenceClassification")
    @patch("tools.sentiment_tools.finbert_analyzer.BertTokenizer")
    @patch("tools.sentiment_tools.finbert_analyzer._select_device", return_value="cuda")
    def test_falls_back_to_cpu_if_model_load_fails_on_gpu(
        self, mock_select_device, mock_tokenizer_cls, mock_model_cls
    ):
        mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
        good_model = MagicMock()
        # First from_pretrained call's .to(device) raises (simulating GPU OOM at load time)
        first_model = MagicMock()
        first_model.to.side_effect = RuntimeError("CUDA OOM")
        mock_model_cls.from_pretrained.side_effect = [first_model, good_model]

        FinBertSentimentAnalyzer()

        assert mock_model_cls.from_pretrained.call_count == 2
        good_model.to.assert_called_with("cpu")