"""
tools/vader_scorer.py — VaderLexiconScorer
==========================================
Rule-based, lightweight sentiment baseline using NLTK VADER.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
tuned for social-media text, slang, abbreviations, punctuation intensity,
and ALL-CAPS emphasis — making it ideal for Reddit posts, tweets, and
short financial news headlines.

Why VADER alongside FinBERT?
──────────────────────────────
  • Speed   : ~1 ms per text vs ~50 ms for FinBERT; processes thousands of
              texts in under a second without GPU.
  • Coverage: Handles domain-specific slang ("moon", "to the moon", "rekt",
              "yolo") and emoji sentiment cues that FinBERT misses.
  • Baseline: Provides a stable independent signal for FearGreedIndexCalculator
              cross-validation.

VADER compound score range
───────────────────────────
  +1.0  → Maximally positive
   0.0  → Perfectly neutral
  -1.0  → Maximally negative

  Thresholds (standard VADER convention):
    compound >= +0.05  → Positive / Bullish
    compound <= -0.05  → Negative / Bearish
    otherwise          → Neutral

Public interface
─────────────────
  scorer = VaderLexiconScorer()
  result = scorer.score(texts=["NVDA to the moon! 🚀🚀", "Market crash incoming 😱"])
  # result.compound      → float in [-1, +1]
  # result.positive_mean → mean positive score across chunks
  # result.negative_mean → mean negative score across chunks
  # result.neutral_mean  → mean neutral score across chunks
  # result.label         → "Bullish" | "Bearish" | "Neutral"
  # result.chunk_scores  → per-text breakdown
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# VADER thresholds (NLTK recommended values)
_POSITIVE_THRESHOLD: float =  0.05
_NEGATIVE_THRESHOLD: float = -0.05

# ---------------------------------------------------------------------------
# VADER resource bootstrap (download once per environment)
# ---------------------------------------------------------------------------

_VADER_LOCK:   threading.Lock = threading.Lock()
_VADER_LOADED: bool           = False


def reset_vader() -> None:
    """
    Reset the VADER lexicon loaded flag.

    Intended for test teardown so each test starts from a clean state
    without the VADER lexicon pre-loaded.

    Example (pytest)::

        @pytest.fixture(autouse=True)
        def clean_vader():
            yield
            reset_vader()
    """
    global _VADER_LOADED
    with _VADER_LOCK:
        _VADER_LOADED = False


def _ensure_vader_lexicon() -> None:
    """
    Download the VADER lexicon if it is not already present.

    Thread-safe; subsequent calls are no-ops after the first download.
    Uses double-checked locking to avoid redundant I/O.
    """
    global _VADER_LOADED
    if _VADER_LOADED:
        return
    with _VADER_LOCK:
        if _VADER_LOADED:
            return
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
            logger.info("VADER lexicon already present.")
        except LookupError:
            logger.info("Downloading VADER lexicon (one-time)...")
            nltk.download("vader_lexicon", quiet=True)
        _VADER_LOADED = True


# ---------------------------------------------------------------------------
# Return Schema
# ---------------------------------------------------------------------------

@dataclass
class ChunkVaderScore:
    """
    VADER scores for a single text chunk.

    Attributes:
        text      : Truncated source text (first 120 chars shown).
        positive  : Proportion of text with positive valence  [0, 1].
        negative  : Proportion of text with negative valence  [0, 1].
        neutral   : Proportion of text with neutral valence   [0, 1].
        compound  : Normalised, weighted composite score      [-1, +1].
        label     : Threshold-based label: "Bullish", "Bearish", "Neutral".
    """
    text:     str
    positive: float
    negative: float
    neutral:  float
    compound: float
    label:    str


@dataclass
class VaderResult:
    """
    Aggregated VADER output across all scored text chunks.

    Attributes:
        compound      : Mean compound score across all chunks  [-1, +1].
        positive_mean : Mean positive proportion across chunks [0, 1].
        negative_mean : Mean negative proportion across chunks [0, 1].
        neutral_mean  : Mean neutral proportion across chunks  [0, 1].
        label         : Corpus-level threshold-based label.
        chunk_scores  : Per-chunk VaderScore objects for auditability.
        total_chunks  : Number of chunks successfully scored.
        skipped_chunks: Number of empty / invalid texts skipped.
    """
    compound:       float
    positive_mean:  float
    negative_mean:  float
    neutral_mean:   float
    label:          str
    chunk_scores:   list[ChunkVaderScore] = field(default_factory=list)
    total_chunks:   int                   = 0
    skipped_chunks: int                   = 0


# ---------------------------------------------------------------------------
# VaderLexiconScorer
# ---------------------------------------------------------------------------

class VaderLexiconScorer:
    """
    Fast rule-based sentiment baseline using NLTK VADER.

    Designed for high-throughput scoring of social media text, Reddit posts,
    financial news snippets, and any short-form content where FinBERT's
    deep contextual understanding is overkill or too slow.

    The VADER lexicon is auto-downloaded on first use and cached globally.

    Example
    -------
    >>> scorer = VaderLexiconScorer()
    >>> result = scorer.score([
    ...     "NVDA absolutely crushed earnings! 🚀🚀🚀",
    ...     "This market crash will wipe out everyone.",
    ...     "Tech stocks traded sideways today.",
    ... ])
    >>> print(result.label, result.compound)
    Neutral -0.031
    """

    def __init__(self) -> None:
        _ensure_vader_lexicon()
        # SentimentIntensityAnalyzer is not thread-safe; create one per instance.
        self._sia = SentimentIntensityAnalyzer()
        logger.info("VaderLexiconScorer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, texts: list[str]) -> VaderResult:
        """
        Score a list of text chunks using VADER.

        Processes each chunk independently with polarity_scores(),
        then computes mean aggregates for corpus-level output.

        Parameters
        ----------
        texts : list[str]
            Raw text chunks from LocalSocialDataRetriever or any source.
            HTML tags and URLs do not require pre-cleaning; VADER handles them.

        Returns
        -------
        VaderResult
            Aggregated compound score, directional means, corpus label,
            and per-chunk breakdown.
            Returns a neutral zero-score result if all texts are empty.

        Examples
        --------
        >>> scorer = VaderLexiconScorer()
        >>> r = scorer.score(["Market is BOOMING!!!", "Stocks plummeted."])
        >>> assert r.label in ("Bullish", "Bearish", "Neutral")
        """
        valid:   list[str] = []
        skipped: int       = 0

        for t in texts:
            clean = t.strip()
            if clean:
                valid.append(clean)
            else:
                skipped += 1

        if not valid:
            logger.warning("VaderLexiconScorer: all %d texts were empty.", len(texts))
            return self._empty_result(skipped=len(texts))

        chunk_scores: list[ChunkVaderScore] = []

        for text in valid:
            scores = self._sia.polarity_scores(text)
            chunk_scores.append(ChunkVaderScore(
                text     = text[:120] + ("..." if len(text) > 120 else ""),
                positive = round(scores["pos"],  4),
                negative = round(scores["neg"],  4),
                neutral  = round(scores["neu"],  4),
                compound = round(scores["compound"], 4),
                label    = self._compound_label(scores["compound"]),
            ))

        n              = len(chunk_scores)
        mean_compound  = sum(c.compound  for c in chunk_scores) / n
        mean_positive  = sum(c.positive  for c in chunk_scores) / n
        mean_negative  = sum(c.negative  for c in chunk_scores) / n
        mean_neutral   = sum(c.neutral   for c in chunk_scores) / n
        corpus_label   = self._compound_label(mean_compound)

        result = VaderResult(
            compound       = round(mean_compound, 4),
            positive_mean  = round(mean_positive, 4),
            negative_mean  = round(mean_negative, 4),
            neutral_mean   = round(mean_neutral,  4),
            label          = corpus_label,
            chunk_scores   = chunk_scores,
            total_chunks   = n,
            skipped_chunks = skipped,
        )
        logger.info(
            "VADER scoring complete: %d chunks | %s (compound=%.4f)",
            n, corpus_label, mean_compound,
        )
        return result

    def score_single(self, text: str) -> ChunkVaderScore:
        """
        Score a single text string.

        Convenience method for ad-hoc scoring without batching overhead.

        Parameters
        ----------
        text : str
            A single text string to score.

        Returns
        -------
        ChunkVaderScore
            VADER scores for the individual text.

        Raises
        ------
        ValueError
            If text is empty or whitespace-only.
        """
        text = text.strip()
        if not text:
            raise ValueError("text must be a non-empty string.")

        scores = self._sia.polarity_scores(text)
        return ChunkVaderScore(
            text     = text[:120] + ("..." if len(text) > 120 else ""),
            positive = round(scores["pos"],     4),
            negative = round(scores["neg"],     4),
            neutral  = round(scores["neu"],     4),
            compound = round(scores["compound"], 4),
            label    = self._compound_label(scores["compound"]),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compound_label(compound: float) -> str:
        """
        Map a VADER compound score to a directional market label.

        Parameters
        ----------
        compound : float
            VADER compound score in [-1.0, +1.0].

        Returns
        -------
        str
            "Bullish"  if compound >= +0.05
            "Bearish"  if compound <= -0.05
            "Neutral"  otherwise
        """
        if compound >= _POSITIVE_THRESHOLD:
            return "Bullish"
        if compound <= _NEGATIVE_THRESHOLD:
            return "Bearish"
        return "Neutral"

    @staticmethod
    def _empty_result(skipped: int = 0) -> VaderResult:
        """
        Return a zero-score neutral VaderResult when no valid text exists.

        Parameters
        ----------
        skipped : int
            Number of empty chunks filtered out.

        Returns
        -------
        VaderResult
            All scores at 0.0, label "Neutral".
        """
        return VaderResult(
            compound       = 0.0,
            positive_mean  = 0.0,
            negative_mean  = 0.0,
            neutral_mean   = 0.0,
            label          = "Neutral",
            chunk_scores   = [],
            total_chunks   = 0,
            skipped_chunks = skipped,
        )