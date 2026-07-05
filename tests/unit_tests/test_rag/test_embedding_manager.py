"""
Tests for: rag/embedding_manager.py
Phase: 3 — RAG Pipeline (3rd: depends on rag/processor.py's ProcessedChunk)

Mocking strategy: torch and sentence_transformers.SentenceTransformer are
mocked entirely — we never load the real 200 MB embedding model. We patch
SentenceTransformer.encode to return controlled numpy arrays so we can
verify L2-normalisation, batching params, and the CUDA/MPS->CPU fallback
chain deterministically.

IMPORTANT: reset_embedder() must run between tests since _INSTANCE is a
module-level singleton.
"""
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

import rag.embedding_manager as em_module
from rag.embedding_manager import (
    AlphaEmbedder,
    get_embedder,
    reset_embedder,
    _select_device,
)
from rag.processor import ProcessedChunk


@pytest.fixture(autouse=True)
def clean_embedder():
    reset_embedder()
    yield
    reset_embedder()


@pytest.fixture
def mock_sentence_transformer():
    with patch("rag.embedding_manager.SentenceTransformer") as mock_cls:
        instance = MagicMock()
        # Default: encode returns a 2-row, 3-dim unnormalised array
        instance.encode.return_value = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        mock_cls.return_value = instance
        yield mock_cls, instance


# ---------------------------------------------------------------------------
# _select_device
# ---------------------------------------------------------------------------

class TestSelectDevice:
    @patch("rag.embedding_manager.torch")
    def test_returns_cpu_when_nothing_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert _select_device() == "cpu"

    @patch("rag.embedding_manager.torch")
    def test_returns_cuda_when_allocatable(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.return_value = MagicMock()
        assert _select_device() == "cuda"

    @patch("rag.embedding_manager.torch")
    def test_falls_back_to_mps_when_cuda_unavailable(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.zeros.return_value = MagicMock()
        assert _select_device() == "mps"

    @patch("rag.embedding_manager.torch")
    def test_cuda_allocation_failure_falls_back(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.zeros.side_effect = RuntimeError("OOM")
        mock_torch.backends.mps.is_available.return_value = False
        assert _select_device() == "cpu"


# ---------------------------------------------------------------------------
# get_embedder() — singleton behavior
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_returns_same_instance_across_calls(self, mock_sentence_transformer):
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            e1 = get_embedder()
            e2 = get_embedder()
        assert e1 is e2

    def test_model_loaded_only_once(self, mock_sentence_transformer):
        mock_cls, _ = mock_sentence_transformer
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            get_embedder()
            get_embedder()
        assert mock_cls.call_count == 1

    def test_reset_embedder_allows_new_instance(self, mock_sentence_transformer):
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            e1 = get_embedder()
            reset_embedder()
            e2 = get_embedder()
        assert e1 is not e2


# ---------------------------------------------------------------------------
# AlphaEmbedder._load_model — CPU fallback on OOM at load time
# ---------------------------------------------------------------------------

class TestLoadModelFallback:
    @patch("rag.embedding_manager.torch")
    def test_falls_back_to_cpu_if_gpu_load_fails(self, mock_torch):
        mock_torch.cuda.OutOfMemoryError = RuntimeError  # alias for except clause
        with patch("rag.embedding_manager._select_device", return_value="cuda"), \
             patch("rag.embedding_manager.SentenceTransformer") as mock_cls:
            good_instance = MagicMock()
            mock_cls.side_effect = [RuntimeError("CUDA OOM"), good_instance]

            embedder = AlphaEmbedder()

            assert embedder.device == "cpu"
            assert mock_cls.call_count == 2

    @patch("rag.embedding_manager.torch")
    def test_reraises_if_cpu_load_also_fails(self, mock_torch):
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        with patch("rag.embedding_manager._select_device", return_value="cpu"), \
             patch("rag.embedding_manager.SentenceTransformer") as mock_cls:
            mock_cls.side_effect = RuntimeError("no memory anywhere")
            with pytest.raises(RuntimeError):
                AlphaEmbedder()


# ---------------------------------------------------------------------------
# embed_chunks / embed_query — public API + L2 normalisation
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def test_embed_chunks_returns_normalised_vectors_with_metadata(
        self, mock_sentence_transformer
    ):
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            embedder = AlphaEmbedder()
        chunk = ProcessedChunk(text="hello world", metadata={"ticker": "NVDA"})

        results = embedder.embed_chunks([chunk])

        assert len(results) == 1
        vec = results[0]["embedding"]
        # [3,4,0] has norm 5 -> normalised to [0.6, 0.8, 0.0]
        assert vec == pytest.approx([0.6, 0.8, 0.0], abs=1e-6)
        assert results[0]["metadata"] == {"ticker": "NVDA"}
        assert results[0]["text"] == "hello world"

    def test_embed_chunks_empty_list_returns_empty(self, mock_sentence_transformer):
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            embedder = AlphaEmbedder()
        assert embedder.embed_chunks([]) == []

    def test_embed_query_returns_single_normalised_vector(self, mock_sentence_transformer):
        _, instance = mock_sentence_transformer
        instance.encode.return_value = np.array([[3.0, 4.0, 0.0]])
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            embedder = AlphaEmbedder()
        vec = embedder.embed_query("hello")
        assert vec == pytest.approx([0.6, 0.8, 0.0], abs=1e-6)

    def test_zero_vector_division_by_zero_avoided(self, mock_sentence_transformer):
        _, instance = mock_sentence_transformer
        instance.encode.return_value = np.array([[0.0, 0.0, 0.0]])
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            embedder = AlphaEmbedder()
        vec = embedder.embed_query("empty")
        assert vec == [0.0, 0.0, 0.0]  # no NaN/division error


# ---------------------------------------------------------------------------
# _encode_batch — OOM fallback mid-encoding
# ---------------------------------------------------------------------------

class TestEncodeBatchOomFallback:
    @patch("rag.embedding_manager.torch")
    def test_oom_during_encode_falls_back_to_cpu_and_retries(self, mock_torch):
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        with patch("rag.embedding_manager._select_device", return_value="cuda"), \
             patch("rag.embedding_manager.SentenceTransformer") as mock_cls:
            instance = MagicMock()
            instance.encode.side_effect = [
                RuntimeError("OOM mid-batch"),
                np.array([[1.0, 0.0, 0.0]]),
            ]
            mock_cls.return_value = instance

            embedder = AlphaEmbedder()
            vec = embedder.embed_query("text")

            assert embedder.device == "cpu"
            assert instance.encode.call_count == 2
            instance.to.assert_called_with("cpu")
            assert vec == [1.0, 0.0, 0.0]

    @patch("rag.embedding_manager.torch")
    def test_oom_on_cpu_reraises(self, mock_torch):
        mock_torch.cuda.OutOfMemoryError = RuntimeError
        with patch("rag.embedding_manager._select_device", return_value="cpu"), \
             patch("rag.embedding_manager.SentenceTransformer") as mock_cls:
            instance = MagicMock()
            instance.encode.side_effect = RuntimeError("OOM even on CPU")
            mock_cls.return_value = instance

            embedder = AlphaEmbedder()
            with pytest.raises(RuntimeError):
                embedder.embed_query("text")


# ---------------------------------------------------------------------------
# Unsupported model name — warning only, not an error
# ---------------------------------------------------------------------------

class TestUnsupportedModelName:
    def test_unsupported_model_logs_warning_but_proceeds(self, mock_sentence_transformer):
        with patch("rag.embedding_manager._select_device", return_value="cpu"):
            embedder = AlphaEmbedder(model_name="some-random-model")
        assert embedder.model_name == "some-random-model"