"""
tools/sentiment_tools/finbert_analyzer.py — FinBertSentimentAnalyzer
======================================================
Financial sentiment analysis using the open-source ProsusAI/finbert model.

FinBERT is a BERT variant fine-tuned on the Financial PhraseBank dataset.
It produces three probability scores per text:
    - positive  (Bullish signal)
    - negative  (Bearish signal)
    - neutral

Model card: https://huggingface.co/ProsusAI/finbert

Architecture decisions
───────────────────────
  • Singleton model loading — loaded once at first instantiation, then shared.
  • Batched inference — texts are grouped into mini-batches to prevent OOM.
  • Truncation — FinBERT has a 512-token limit; inputs are auto-truncated.
  • Device fallback — CUDA → MPS → CPU (matches AlphaEmbedder pattern).
  • Aggregation — per-chunk scores are averaged into a corpus-level result.

Public interface
─────────────────
  analyzer  = FinBertSentimentAnalyzer()
  result    = analyzer.analyze(texts=["NVIDIA smashed estimates ...", ...])
  # result.bullish_prob  → 0.72
  # result.bearish_prob  → 0.15
  # result.neutral_prob  → 0.13
  # result.label         → "Bullish"
  # result.chunk_scores  → per-text breakdown
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton cache
# ---------------------------------------------------------------------------

_LOCK:      threading.Lock                         = threading.Lock()
_TOKENIZER: Optional[BertTokenizer]                = None
_MODEL:     Optional[BertForSequenceClassification] = None
_DEVICE:    Optional[str]                          = None

MODEL_NAME = "ProsusAI/finbert"

def reset_finbert() -> None:
    """
    Reset the FinBERT singleton model, tokenizer, and device globals.

    Intended for test teardown so each test starts with a clean state
    without the 440 MB model pre-loaded.

    Example (pytest)::

        @pytest.fixture(autouse=True)
        def clean_finbert():
            yield
            reset_finbert()
    """
    global _TOKENIZER, _MODEL, _DEVICE
    with _LOCK:
        _TOKENIZER = None
        _MODEL     = None
        _DEVICE    = None



# Label order as defined in ProsusAI/finbert config.json
# id2label: {0: "positive", 1: "negative", 2: "neutral"}
_IDX_POSITIVE = 0
_IDX_NEGATIVE = 1
_IDX_NEUTRAL  = 2


# ---------------------------------------------------------------------------
# Return Schema
# ---------------------------------------------------------------------------

@dataclass
class ChunkSentiment:
    """
    Sentiment scores for a single text chunk.

    Attributes:
        text         : Truncated source text (first 120 chars shown).
        bullish_prob : Probability of positive / bullish sentiment  [0, 1].
        bearish_prob : Probability of negative / bearish sentiment  [0, 1].
        neutral_prob : Probability of neutral sentiment             [0, 1].
        label        : Argmax label: "Bullish", "Bearish", or "Neutral".
    """
    text:         str
    bullish_prob: float
    bearish_prob: float
    neutral_prob: float
    label:        str


@dataclass
class FinBertResult:
    """
    Aggregated FinBERT output across all analyzed text chunks.

    Attributes:
        bullish_prob  : Mean positive probability across all chunks.
        bearish_prob  : Mean negative probability across all chunks.
        neutral_prob  : Mean neutral probability across all chunks.
        label         : Corpus-level label derived from argmax of mean probs.
        chunk_scores  : Per-chunk breakdown for auditability.
        total_chunks  : Number of chunks successfully analyzed.
        skipped_chunks: Number of empty / invalid chunks that were skipped.
    """
    bullish_prob:   float
    bearish_prob:   float
    neutral_prob:   float
    label:          str
    chunk_scores:   list[ChunkSentiment] = field(default_factory=list)
    total_chunks:   int                  = 0
    skipped_chunks: int                  = 0


# ---------------------------------------------------------------------------
# Device helper (mirrors AlphaEmbedder._select_device)
# ---------------------------------------------------------------------------

def _select_device() -> str:
    """
    Select the best available compute device.

    Priority order: CUDA → MPS (Apple Silicon) → CPU.
    Falls back to CPU on allocation failure.

    Returns
    -------
    str
        One of "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            logger.info("FinBertSentimentAnalyzer device: CUDA")
            return "cuda"
        except RuntimeError as exc:
            logger.warning("CUDA allocation failed (%s); falling back.", exc)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            logger.info("FinBertSentimentAnalyzer device: MPS")
            return "mps"
        except RuntimeError as exc:
            logger.warning("MPS allocation failed (%s); falling back.", exc)

    logger.info("FinBertSentimentAnalyzer device: CPU")
    return "cpu"


# ---------------------------------------------------------------------------
# FinBertSentimentAnalyzer
# ---------------------------------------------------------------------------

class FinBertSentimentAnalyzer:
    """
    Financial sentiment analysis using ProsusAI/finbert.

    Thread-safe singleton model loading ensures the 440 MB model
    is downloaded and initialised exactly once per process.

    Parameters
    ----------
    batch_size : int
        Number of texts processed per forward pass.  Lower values
        reduce GPU memory pressure.  Default: 16.
    max_length : int
        Maximum token length per chunk (FinBERT hard limit: 512).
        Default: 512.

    Example
    -------
    >>> analyzer = FinBertSentimentAnalyzer()
    >>> texts = ["Apple beat Q3 earnings estimates by 12%.",
    ...          "Recession fears drag tech sector lower."]
    >>> result = analyzer.analyze(texts)
    >>> print(result.label, result.bullish_prob)
    Neutral 0.48
    """

    def __init__(
        self,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, texts: list[str]) -> FinBertResult:
        """
        Run FinBERT sentiment analysis over a list of text chunks.

        Filters out empty strings, processes valid texts in batches,
        and returns aggregated probabilities + per-chunk breakdown.

        Parameters
        ----------
        texts : list[str]
            Raw text chunks (e.g. from LocalSocialDataRetriever).
            Empty strings are skipped gracefully.

        Returns
        -------
        FinBertResult
            Aggregated + per-chunk sentiment probabilities.
            Returns a neutral zero-score result if all texts are empty.

        Raises
        ------
        RuntimeError
            If model inference fails on all available devices.
        """
        # Filter out empty / whitespace-only strings
        valid:   list[str] = []
        skipped: int       = 0
        for t in texts:
            clean = t.strip()
            if clean:
                valid.append(clean)
            else:
                skipped += 1

        if not valid:
            logger.warning("FinBertSentimentAnalyzer: all %d texts were empty.", len(texts))
            return self._empty_result(skipped=len(texts))

        chunk_scores: list[ChunkSentiment] = []

        # Process in mini-batches
        for batch_start in range(0, len(valid), self.batch_size):
            batch_texts = valid[batch_start: batch_start + self.batch_size]
            batch_probs = self._infer_batch(batch_texts)      # shape (B, 3)

            for text, probs in zip(batch_texts, batch_probs):
                bullish = float(probs[_IDX_POSITIVE])
                bearish = float(probs[_IDX_NEGATIVE])
                neutral = float(probs[_IDX_NEUTRAL])
                chunk_scores.append(ChunkSentiment(
                    text         = text[:120] + ("..." if len(text) > 120 else ""),
                    bullish_prob = round(bullish, 4),
                    bearish_prob = round(bearish, 4),
                    neutral_prob = round(neutral, 4),
                    label        = self._argmax_label(bullish, bearish, neutral),
                ))

        # Corpus-level aggregation: simple mean across chunks
        n              = len(chunk_scores)
        mean_bullish   = sum(c.bullish_prob for c in chunk_scores) / n
        mean_bearish   = sum(c.bearish_prob for c in chunk_scores) / n
        mean_neutral   = sum(c.neutral_prob for c in chunk_scores) / n
        corpus_label   = self._argmax_label(mean_bullish, mean_bearish, mean_neutral)

        result = FinBertResult(
            bullish_prob   = round(mean_bullish, 4),
            bearish_prob   = round(mean_bearish, 4),
            neutral_prob   = round(mean_neutral, 4),
            label          = corpus_label,
            chunk_scores   = chunk_scores,
            total_chunks   = n,
            skipped_chunks = skipped,
        )
        logger.info(
            "FinBERT analysis complete: %d chunks | %s (bull=%.3f bear=%.3f neu=%.3f)",
            n, corpus_label, mean_bullish, mean_bearish, mean_neutral,
        )
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_batch(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenize and run a single forward pass for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            A batch of clean, non-empty strings.

        Returns
        -------
        torch.Tensor
            Shape (B, 3) — softmax probabilities [positive, negative, neutral].

        Raises
        ------
        RuntimeError
            Propagated from torch if inference fails on the selected device.
        """
        global _DEVICE

        encoding = _TOKENIZER(   # type: ignore[call-arg]
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Move input tensors to the active device
        encoding = {k: v.to(_DEVICE) for k, v in encoding.items()}

        with torch.no_grad():
            try:
                logits = _MODEL(**encoding).logits   # type: ignore[operator]
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if _DEVICE != "cpu":
                    logger.warning(
                        "OOM on %s during FinBERT inference (%s); switching to CPU.",
                        _DEVICE, exc,
                    )
                    _DEVICE = "cpu"
                    _MODEL.to("cpu")   # type: ignore[union-attr]
                    encoding = {k: v.to("cpu") for k, v in encoding.items()}
                    logits   = _MODEL(**encoding).logits   # type: ignore[operator]
                else:
                    raise

        return F.softmax(logits, dim=-1).cpu()

    # ------------------------------------------------------------------
    # Model loading (singleton, thread-safe)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load ProsusAI/finbert tokenizer and model exactly once.

        Uses double-checked locking for thread safety.
        Falls back to CPU if the GPU device fails to allocate the model.
        """
        global _TOKENIZER, _MODEL, _DEVICE

        if _TOKENIZER is not None and _MODEL is not None:
            return

        with _LOCK:
            if _TOKENIZER is not None and _MODEL is not None:
                return  # Another thread beat us to it

            logger.info("Loading ProsusAI/finbert (first time; this may take a moment)...")
            _DEVICE    = _select_device()
            _TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)

            try:
                _MODEL = BertForSequenceClassification.from_pretrained(MODEL_NAME)
                _MODEL.to(_DEVICE)
                _MODEL.eval()
                logger.info("FinBERT loaded on %s.", _DEVICE)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if _DEVICE != "cpu":
                    logger.warning(
                        "Failed to load FinBERT on %s (%s); falling back to CPU.",
                        _DEVICE, exc,
                    )
                    _DEVICE = "cpu"
                    _MODEL  = BertForSequenceClassification.from_pretrained(MODEL_NAME)
                    _MODEL.to("cpu")
                    _MODEL.eval()
                    logger.info("FinBERT loaded on CPU (fallback).")
                else:
                    raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _argmax_label(bullish: float, bearish: float, neutral: float) -> str:
        """
        Return the human-readable label for the dominant sentiment.

        Parameters
        ----------
        bullish : float
            Probability for positive / bullish.
        bearish : float
            Probability for negative / bearish.
        neutral : float
            Probability for neutral.

        Returns
        -------
        str
            One of "Bullish", "Bearish", "Neutral".
        """
        scores = {"Bullish": bullish, "Bearish": bearish, "Neutral": neutral}
        return max(scores, key=scores.__getitem__)

    @staticmethod
    def _empty_result(skipped: int = 0) -> FinBertResult:
        """
        Return a neutral zero-score FinBertResult when no valid text is available.

        Parameters
        ----------
        skipped : int
            Number of invalid / empty chunks that were filtered out.

        Returns
        -------
        FinBertResult
            All probabilities set to 0.0, label "Neutral".
        """
        return FinBertResult(
            bullish_prob   = 0.0,
            bearish_prob   = 0.0,
            neutral_prob   = 0.0,
            label          = "Neutral",
            chunk_scores   = [],
            total_chunks   = 0,
            skipped_chunks = skipped,
        )