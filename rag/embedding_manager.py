"""Versioned embedding generation with an explicit deterministic cache key."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import asdict, dataclass
from typing import MutableMapping, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag.processor import ProcessedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingManifest:
    model_name: str
    model_version: str
    dimension: int | None
    cache_schema_version: str = "embedding-cache-v1"


def _select_device() -> str:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError as exc:
            logger.warning("CUDA allocation failed (%s); falling back.", exc)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except RuntimeError as exc:
            logger.warning("MPS allocation failed (%s); falling back.", exc)
    return "cpu"


_LOCK = threading.Lock()
_INSTANCE: Optional["AlphaEmbedder"] = None


def reset_embedder() -> None:
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None


def get_embedder(model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64, *, model_version: str | None = None) -> "AlphaEmbedder":
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                _INSTANCE = AlphaEmbedder(model_name=model_name, batch_size=batch_size, model_version=model_version)
    return _INSTANCE


class AlphaEmbedder:
    SUPPORTED_MODELS = {"all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"}

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64, *, model_version: str | None = None, cache: MutableMapping[str, list[float]] | None = None) -> None:
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning("Model '%s' not in known-good list %s; proceeding anyway.", model_name, self.SUPPORTED_MODELS)
        self.model_name = model_name
        self.model_version = model_version or model_name
        self.batch_size = batch_size
        self.device = _select_device()
        self._model: Optional[SentenceTransformer] = None
        self._cache: MutableMapping[str, list[float]] = cache if cache is not None else {}
        self._dimension: int | None = None
        self._load_model()

    @property
    def manifest(self) -> EmbeddingManifest:
        return EmbeddingManifest(self.model_name, self.model_version, self._dimension)

    def manifest_dict(self) -> dict[str, object]:
        return asdict(self.manifest)

    def _load_model(self) -> None:
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self._model = SentenceTransformer(self.model_name, device="cpu")

    def cache_key(self, chunk: ProcessedChunk) -> str:
        content_hash = chunk.metadata.get("chunk_hash") or hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{content_hash}|{self.model_version}".encode("utf-8")).hexdigest()

    def embed_chunks(self, chunks: list[ProcessedChunk]) -> list[dict]:
        if not chunks:
            return []
        keys = [self.cache_key(chunk) for chunk in chunks]
        missing_indexes = [index for index, key in enumerate(keys) if key not in self._cache]
        if missing_indexes:
            vectors = self._encode_batch([chunks[index].text for index in missing_indexes])
            for index, vector in zip(missing_indexes, vectors):
                self._cache[keys[index]] = vector.tolist()
        results: list[dict] = []
        for chunk, key in zip(chunks, keys):
            vector = list(self._cache[key])
            self._dimension = len(vector)
            metadata = dict(chunk.metadata)
            metadata.update({
                "embedding_model": self.model_name,
                "embedding_version": self.model_version,
                "embedding_dimension": self._dimension,
                "embedding_cache_key": key,
            })
            results.append({"embedding": vector, "metadata": metadata, "text": chunk.text})
        return results

    def embed_query(self, query: str) -> list[float]:
        vector = self._encode_batch([query])[0].tolist()
        self._dimension = len(vector)
        return vector

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        assert self._model is not None, "Model not initialised."
        try:
            embeddings = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self._model.to("cpu")  # type: ignore[arg-type]
            embeddings = self._model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.where(norms == 0, 1.0, norms)


__all__ = ["AlphaEmbedder", "EmbeddingManifest", "get_embedder", "reset_embedder"]
