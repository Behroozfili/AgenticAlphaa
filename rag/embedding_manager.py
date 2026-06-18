"""
rag/embedding_manager.py — AlphaEmbedder
Singleton embedding model with CUDA/MPS/CPU graceful fallback,
optimized batch processing, and L2 normalization.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag.processor import ProcessedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device Detection
# ---------------------------------------------------------------------------

def _select_device() -> str:
    """
    Priority: CUDA → MPS (Apple Silicon) → CPU.
    Falls back automatically if GPU memory allocation fails at runtime.
    """
    if torch.cuda.is_available():
        # Quick sanity: try to allocate a tiny tensor to detect OOM upfront
        try:
            _ = torch.zeros(1, device="cuda")
            logger.info("AlphaEmbedder device: CUDA")
            return "cuda"
        except RuntimeError as exc:
            logger.warning("CUDA available but allocation failed (%s); falling back.", exc)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps")
            logger.info("AlphaEmbedder device: MPS (Apple Silicon)")
            return "mps"
        except RuntimeError as exc:
            logger.warning("MPS available but allocation failed (%s); falling back.", exc)

    logger.info("AlphaEmbedder device: CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_INSTANCE: Optional["AlphaEmbedder"] = None



def reset_embedder() -> None:
    """
    Reset the singleton AlphaEmbedder instance.

    Intended for use in test teardown fixtures so each test can start
    with a clean state without loading the real 200 MB model.

    Example (pytest)::

        @pytest.fixture(autouse=True)
        def clean_embedder():
            yield
            reset_embedder()
    """
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None


def get_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> "AlphaEmbedder":
    """
    Module-level factory that returns the single shared AlphaEmbedder instance.
    Thread-safe via double-checked locking.
    """
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = AlphaEmbedder(model_name=model_name, batch_size=batch_size)
    return _INSTANCE


# ---------------------------------------------------------------------------
# AlphaEmbedder
# ---------------------------------------------------------------------------

class AlphaEmbedder:
    """
    Production-ready embedding engine.

    - Loaded once (singleton via get_embedder()).
    - Gracefully falls back from CUDA/MPS to CPU on insufficient GPU RAM.
    - Encodes in optimised batches.
    - Returns L2-normalised vectors for consistent cosine similarity.
    """

    # Supported drop-in models (same 384-dim output, interchangeable)
    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
    ) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(
                "Model '%s' not in known-good list %s; proceeding anyway.",
                model_name, self.SUPPORTED_MODELS,
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.device     = _select_device()
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading with CPU fallback
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("AlphaEmbedder loaded '%s' on %s.", self.model_name, self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if self.device != "cpu":
                logger.warning(
                    "Failed to load on %s (%s); retrying on CPU.", self.device, exc
                )
                self.device = "cpu"
                self._model = SentenceTransformer(self.model_name, device="cpu")
                logger.info(
                    "AlphaEmbedder loaded '%s' on CPU (fallback).", self.model_name
                )
            else:
                raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[ProcessedChunk]) -> list[dict]:
        """
        Embed a list of ProcessedChunks.

        Returns a list of dicts, each containing:
            {
                "embedding": List[float],   # L2-normalised, 384-dim
                "metadata":  dict,
                "text":      str,
            }
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self._encode_batch(texts)  # (N, D) float32, already L2-normalised

        results = []
        for chunk, vec in zip(chunks, vectors):
            results.append({
                "embedding": vec.tolist(),
                "metadata":  chunk.metadata,
                "text":      chunk.text,
            })
        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string and return an L2-normalised vector."""
        vec = self._encode_batch([query])[0]
        return vec.tolist()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts in mini-batches, concatenate, and L2-normalise.
        Falls back to CPU if a GPU OOM occurs mid-batch.
        """
        assert self._model is not None, "Model not initialised."

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # built-in L2 norm
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if self.device != "cpu":
                logger.warning(
                    "OOM on %s during encoding (%s); switching to CPU.", self.device, exc
                )
                self.device = "cpu"
                self._model.to("cpu")   # type: ignore[arg-type]
                embeddings = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            else:
                raise

        # Belt-and-suspenders explicit L2 norm (in case the model flag was ignored)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
        return embeddings / norms