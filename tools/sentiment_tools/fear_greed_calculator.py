"""
tools/fear_greed_calculator.py — FearGreedIndexCalculator
==========================================================
Aggregates FinBERT (deep NLP) and VADER (rule-based) outputs into
a single normalised Fear/Greed index score.

Output range
─────────────
  +1.0  →  Extreme Greed  (Highly Bullish)
   0.0  →  Neutral
  -1.0  →  Extreme Fear   (Highly Bearish)

Aggregation logic
──────────────────
  1. Convert FinBERT probabilities to a directional score:
         finbert_score = bullish_prob − bearish_prob   ∈ [−1, +1]

  2. Use VADER compound directly:
         vader_score   = compound                       ∈ [−1, +1]

  3. Weighted average:
         raw_score = (finbert_score × W_FINBERT) + (vader_score × W_VADER)

     Default weights:
         W_FINBERT = 0.65   (higher weight for deep financial NLP)
         W_VADER   = 0.35   (lightweight social-text baseline)

  4. Clamp output to [−1, +1] to absorb floating-point edge cases.

  5. Map to a human-readable market label using five-band thresholds.

Label bands
────────────
  score ∈ [ 0.60,  1.00]  → "Extreme Greed"
  score ∈ [ 0.20,  0.60)  → "Greed"
  score ∈ (-0.20,  0.20)  → "Neutral"
  score ∈ (-0.60, -0.20]  → "Fear"
  score ∈ [-1.00, -0.60]  → "Extreme Fear"

Public interface
─────────────────
  calc   = FearGreedIndexCalculator()
  result = calc.calculate(finbert_result, vader_result)
  # result.score        → −1.0 … +1.0
  # result.label        → "Extreme Greed" | "Greed" | … | "Extreme Fear"
  # result.finbert_score→ directional FinBERT component
  # result.vader_score  → VADER compound component
  # result.weights      → {"finbert": 0.65, "vader": 0.35}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tools.finbert_analyzer import FinBertResult
from tools.vader_scorer import VaderResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

_W_FINBERT: float = 0.65
_W_VADER:   float = 0.35

# ---------------------------------------------------------------------------
# Label band thresholds  (lower bound of each band, inclusive)
# ---------------------------------------------------------------------------

_BANDS: list[tuple[float, str]] = [
    ( 0.60, "Extreme Greed"),
    ( 0.20, "Greed"),
    (-0.20, "Neutral"),
    (-0.60, "Fear"),
    (-1.01, "Extreme Fear"),   # -1.01 so -1.0 is captured
]


# ---------------------------------------------------------------------------
# Return Schema
# ---------------------------------------------------------------------------

@dataclass
class FearGreedResult:
    """
    Final Fear/Greed index output.

    Attributes:
        score         : Normalised composite score in [-1.0, +1.0].
                        Positive = Greed/Bullish; Negative = Fear/Bearish.
        label         : Human-readable five-band market label.
        finbert_score : Directional FinBERT component (bullish − bearish).
        vader_score   : VADER compound score used as input.
        weights       : Dict of {"finbert": float, "vader": float} weights applied.
        confidence    : Simple heuristic: |score| mapped to [0, 1].
                        Higher absolute score = higher model agreement.
        diagnostics   : Intermediate values for transparency / debugging.
    """
    score:         float
    label:         str
    finbert_score: float
    vader_score:   float
    weights:       dict[str, float] = field(default_factory=dict)
    confidence:    float            = 0.0
    diagnostics:   dict             = field(default_factory=dict)


# ---------------------------------------------------------------------------
# FearGreedIndexCalculator
# ---------------------------------------------------------------------------

class FearGreedIndexCalculator:
    """
    Aggregates FinBERT + VADER signals into a unified Fear/Greed index.

    Weights are configurable at construction time.  Defaults favour FinBERT
    (0.65) because it understands financial domain language more precisely,
    while VADER (0.35) captures social slang and intensity signals.

    Parameters
    ----------
    finbert_weight : float
        Weight applied to the FinBERT directional score.  Default: 0.65.
        Must be in [0, 1]; finbert_weight + vader_weight must equal 1.0.
    vader_weight : float
        Weight applied to the VADER compound score.  Default: 0.35.

    Raises
    ------
    ValueError
        If weights are negative or do not sum to 1.0 (±0.001 tolerance).

    Example
    -------
    >>> from tools.finbert_analyzer import FinBertSentimentAnalyzer
    >>> from tools.vader_scorer import VaderLexiconScorer
    >>> from tools.fear_greed_calculator import FearGreedIndexCalculator
    >>>
    >>> texts = ["NVDA crushes estimates!", "AI chip shortage looms."]
    >>> finbert_r = FinBertSentimentAnalyzer().analyze(texts)
    >>> vader_r   = VaderLexiconScorer().score(texts)
    >>> result    = FearGreedIndexCalculator().calculate(finbert_r, vader_r)
    >>> print(result.score, result.label)
    0.27 Greed
    """

    def __init__(
        self,
        finbert_weight: float = _W_FINBERT,
        vader_weight:   float = _W_VADER,
    ) -> None:
        self._validate_weights(finbert_weight, vader_weight)
        self.finbert_weight = finbert_weight
        self.vader_weight   = vader_weight
        logger.info(
            "FearGreedIndexCalculator initialised (w_finbert=%.2f, w_vader=%.2f).",
            finbert_weight, vader_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        finbert_result: FinBertResult,
        vader_result:   VaderResult,
    ) -> FearGreedResult:
        """
        Compute the Fear/Greed index from FinBERT and VADER outputs.

        Steps:
          1. Derive directional FinBERT score:  bullish_prob − bearish_prob
          2. Take VADER compound directly.
          3. Weighted average → raw score.
          4. Clamp to [−1, +1].
          5. Map to label band.
          6. Compute confidence heuristic.

        Parameters
        ----------
        finbert_result : FinBertResult
            Output from FinBertSentimentAnalyzer.analyze().
        vader_result : VaderResult
            Output from VaderLexiconScorer.score().

        Returns
        -------
        FearGreedResult
            Normalised composite score with label and diagnostics.

        Notes
        -----
        If both models analyzed zero valid chunks (both returned empty results),
        the calculator still runs and returns a score of 0.0 / "Neutral".
        This is intentional: a silent neutral is preferable to a crash.
        """
        # Step 1 — FinBERT directional score  ∈ [−1, +1]
        finbert_score: float = finbert_result.bullish_prob - finbert_result.bearish_prob

        # Step 2 — VADER compound  ∈ [−1, +1]
        vader_score: float = vader_result.compound

        # Step 3 — Weighted average
        raw_score: float = (
            (finbert_score * self.finbert_weight)
            + (vader_score  * self.vader_weight)
        )

        # Step 4 — Clamp to [−1, +1]
        clamped: float = max(-1.0, min(1.0, raw_score))

        # Step 5 — Label band
        label: str = self._score_to_label(clamped)

        # Step 6 — Confidence heuristic: |score| (0 = fully uncertain, 1 = max signal)
        confidence: float = round(abs(clamped), 4)

        result = FearGreedResult(
            score         = round(clamped, 4),
            label         = label,
            finbert_score = round(finbert_score, 4),
            vader_score   = round(vader_score, 4),
            weights       = {
                "finbert": self.finbert_weight,
                "vader":   self.vader_weight,
            },
            confidence    = confidence,
            diagnostics   = {
                "raw_score":              round(raw_score,     4),
                "finbert_bullish_prob":   finbert_result.bullish_prob,
                "finbert_bearish_prob":   finbert_result.bearish_prob,
                "finbert_neutral_prob":   finbert_result.neutral_prob,
                "finbert_label":          finbert_result.label,
                "finbert_total_chunks":   finbert_result.total_chunks,
                "vader_compound":         vader_result.compound,
                "vader_positive_mean":    vader_result.positive_mean,
                "vader_negative_mean":    vader_result.negative_mean,
                "vader_label":            vader_result.label,
                "vader_total_chunks":     vader_result.total_chunks,
            },
        )
        logger.info(
            "Fear/Greed index: score=%.4f label='%s' confidence=%.4f "
            "(finbert=%.4f × %.2f + vader=%.4f × %.2f)",
            clamped, label, confidence,
            finbert_score, self.finbert_weight,
            vader_score,   self.vader_weight,
        )
        return result

    def calculate_from_dict(self, finbert_dict: dict, vader_dict: dict) -> FearGreedResult:
        """
        Convenience overload for callers that receive JSON payloads.

        Reconstructs FinBertResult and VaderResult from plain dicts
        (e.g. deserialized from MCP tool responses) and delegates to calculate().

        Parameters
        ----------
        finbert_dict : dict
            Dict with keys: bullish_prob, bearish_prob, neutral_prob, label,
            total_chunks, skipped_chunks.
        vader_dict : dict
            Dict with keys: compound, positive_mean, negative_mean, neutral_mean,
            label, total_chunks, skipped_chunks.

        Returns
        -------
        FearGreedResult
            Same as calculate().

        Raises
        ------
        KeyError
            If required keys are missing from either dict.
        """
        from tools.finbert_analyzer import FinBertResult
        from tools.vader_scorer import VaderResult

        finbert_result = FinBertResult(
            bullish_prob   = float(finbert_dict["bullish_prob"]),
            bearish_prob   = float(finbert_dict["bearish_prob"]),
            neutral_prob   = float(finbert_dict["neutral_prob"]),
            label          = str(finbert_dict.get("label", "Neutral")),
            total_chunks   = int(finbert_dict.get("total_chunks", 0)),
            skipped_chunks = int(finbert_dict.get("skipped_chunks", 0)),
        )
        vader_result = VaderResult(
            compound       = float(vader_dict["compound"]),
            positive_mean  = float(vader_dict.get("positive_mean", 0.0)),
            negative_mean  = float(vader_dict.get("negative_mean", 0.0)),
            neutral_mean   = float(vader_dict.get("neutral_mean",  0.0)),
            label          = str(vader_dict.get("label", "Neutral")),
            total_chunks   = int(vader_dict.get("total_chunks", 0)),
            skipped_chunks = int(vader_dict.get("skipped_chunks", 0)),
        )
        return self.calculate(finbert_result, vader_result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_label(score: float) -> str:
        """
        Map a normalised score to a Fear/Greed label band.

        Parameters
        ----------
        score : float
            Clamped score in [−1.0, +1.0].

        Returns
        -------
        str
            One of: "Extreme Greed", "Greed", "Neutral", "Fear", "Extreme Fear".
        """
        for threshold, label in _BANDS:
            if score >= threshold:
                return label
        return "Extreme Fear"   # Fallback (score < −1.0 after float edge cases)

    @staticmethod
    def _validate_weights(w_finbert: float, w_vader: float) -> None:
        """
        Validate that weights are non-negative and sum to 1.0.

        Parameters
        ----------
        w_finbert : float
            FinBERT weight to validate.
        w_vader : float
            VADER weight to validate.

        Raises
        ------
        ValueError
            If any weight is negative or the sum deviates from 1.0 by > 0.001.
        """
        if w_finbert < 0 or w_vader < 0:
            raise ValueError(
                f"Weights must be non-negative. Got finbert={w_finbert}, vader={w_vader}."
            )
        total = w_finbert + w_vader
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"Weights must sum to 1.0; got {total:.4f} "
                f"(finbert={w_finbert}, vader={w_vader})."
            )
