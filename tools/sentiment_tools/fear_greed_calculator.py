

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tools.sentiment_tools.finbert_analyzer import FinBertResult
from tools.sentiment_tools.vader_scorer import VaderResult

logger = logging.getLogger(__name__)
FEAR_GREED_MODEL_VERSION = "fear-greed-fusion-v1"

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
        direction_strength: Absolute directional score; zero can still be a
                            high-confidence neutral result.
        model_agreement_score: FinBERT/VADER agreement in [0, 1], or None when
                               only one model is available.
        evidence_independence: Optional independent-domain ratio in [0, 1].
        confidence    : Agreement/model-coverage confidence, optionally scaled
                        by evidence independence. It is not direction strength.
        diagnostics   : Intermediate values for transparency / debugging.
    """
    score:         float | None
    label:         str | None
    finbert_score: float | None
    vader_score:   float | None
    direction_strength: float = 0.0
    model_agreement_score: float | None = None
    evidence_independence: float | None = None
    weights:       dict[str, float] = field(default_factory=dict)
    confidence:    float            = 0.0
    diagnostics:   dict             = field(default_factory=dict)
    status:        str              = "VALID"
    polarity:      str | None       = None
    model_version: str              = FEAR_GREED_MODEL_VERSION


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
        *,
        evidence_independence: float | None = None,
    ) -> FearGreedResult:
        """
        Compute the Fear/Greed index from FinBERT and VADER outputs.

        Steps:
          1. Derive directional FinBERT score:  bullish_prob − bearish_prob
          2. Take VADER compound directly.
          3. Weighted average → raw score.
          4. Clamp to [−1, +1].
          5. Map to label band.
          6. Compute direction strength, model agreement, and confidence.

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
        If both models analyzed zero valid chunks, the calculator returns an
        explicit NO_DATA result. Missing evidence must not become silent Neutral.
        """
        fb_valid = (
            finbert_result.total_chunks > 0
            and finbert_result.bullish_prob is not None
            and finbert_result.bearish_prob is not None
        )
        vd_valid = vader_result.total_chunks > 0 and vader_result.compound is not None
        if not fb_valid and not vd_valid:
            return FearGreedResult(
                score=None, label=None, finbert_score=None, vader_score=None,
                direction_strength=0.0, model_agreement_score=None,
                evidence_independence=self._normalise_independence(evidence_independence),
                weights={"finbert": self.finbert_weight, "vader": self.vader_weight},
                confidence=0.0, diagnostics={"reason": "no valid model input"},
                status="NO_DATA", polarity=None,
                model_version=FEAR_GREED_MODEL_VERSION,
            )

        # Step 1 — FinBERT directional score  ∈ [−1, +1]
        finbert_score: float = (
            finbert_result.bullish_prob - finbert_result.bearish_prob if fb_valid else 0.0
        )

        # Step 2 — VADER compound  ∈ [−1, +1]
        vader_score: float = vader_result.compound if vd_valid else 0.0

        # Step 3 — Weighted average
        effective_fb_weight = self.finbert_weight if fb_valid else 0.0
        effective_vd_weight = self.vader_weight if vd_valid else 0.0
        weight_total = effective_fb_weight + effective_vd_weight
        effective_fb_weight /= weight_total
        effective_vd_weight /= weight_total
        raw_score: float = (
            (finbert_score * effective_fb_weight)
            + (vader_score * effective_vd_weight)
        )

        # Step 4 — Clamp to [−1, +1]
        clamped: float = max(-1.0, min(1.0, raw_score))

        # Step 5 — Label band
        label: str = self._score_to_label(clamped)

        # Step 6 — Keep direction strength separate from evidence confidence.
        direction_strength = round(abs(clamped), 4)
        independence = self._normalise_independence(evidence_independence)
        if fb_valid and vd_valid:
            model_agreement_score: float | None = round(
                1.0 - (abs(finbert_score - vader_score) / 2.0), 4
            )
            model_coverage_score = 1.0
        else:
            model_agreement_score = None
            model_coverage_score = 0.5
        confidence_base = (
            model_agreement_score
            if model_agreement_score is not None
            else model_coverage_score
        )
        confidence = round(
            confidence_base * (independence if independence is not None else 1.0),
            4,
        )

        result = FearGreedResult(
            score         = round(clamped, 4),
            label         = label,
            finbert_score = round(finbert_score, 4),
            vader_score   = round(vader_score, 4),
            direction_strength = direction_strength,
            model_agreement_score = model_agreement_score,
            evidence_independence = independence,
            weights       = {
                "finbert": effective_fb_weight,
                "vader":   effective_vd_weight,
            },
            confidence    = confidence,
            diagnostics   = {
                "raw_score":              round(raw_score,     4),
                "direction_strength":      direction_strength,
                "model_agreement_score":  model_agreement_score,
                "model_coverage_score":   model_coverage_score,
                "evidence_independence":  independence,
                "legacy_absolute_score_confidence": direction_strength,
                "partial_model_coverage": not (fb_valid and vd_valid),
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
            status        = "VALID",
            polarity      = (
                "BULLISH" if clamped > 0.2 else "BEARISH" if clamped < -0.2
                else "NEUTRAL"
            ),
            model_version = FEAR_GREED_MODEL_VERSION,
        )
        logger.info(
            "Fear/Greed index: score=%.4f label='%s' confidence=%.4f "
            "(finbert=%.4f × %.2f + vader=%.4f × %.2f)",
            clamped, label, confidence,
            finbert_score, effective_fb_weight,
            vader_score,   effective_vd_weight,
        )
        return result

    def calculate_from_dict(
        self,
        finbert_dict: dict,
        vader_dict: dict,
        *,
        evidence_independence: float | None = None,
    ) -> FearGreedResult:
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

        Missing model fields are treated as unavailable model coverage so legacy
        and partial payloads can still be scored safely.
        """
        from tools.sentiment_tools.finbert_analyzer import FinBertResult
        from tools.sentiment_tools.vader_scorer import VaderResult

        fb_values_available = (
            finbert_dict.get("status") not in {"NO_DATA", "ERROR"}
            and all(
                finbert_dict.get(key) is not None
                for key in ("bullish_prob", "bearish_prob", "neutral_prob")
            )
        )
        vd_values_available = (
            vader_dict.get("status") not in {"NO_DATA", "ERROR"}
            and vader_dict.get("compound") is not None
        )
        fb_total = int(finbert_dict.get("total_chunks", 1 if fb_values_available else 0))
        vd_total = int(vader_dict.get("total_chunks", 1 if vd_values_available else 0))

        finbert_result = FinBertResult(
            bullish_prob   = float(finbert_dict["bullish_prob"]) if fb_values_available else None,
            bearish_prob   = float(finbert_dict["bearish_prob"]) if fb_values_available else None,
            neutral_prob   = float(finbert_dict["neutral_prob"]) if fb_values_available else None,
            label          = str(finbert_dict.get("label", "Neutral")) if fb_values_available else None,
            total_chunks   = fb_total,
            skipped_chunks = int(finbert_dict.get("skipped_chunks", 0)),
        )
        vader_result = VaderResult(
            compound       = float(vader_dict["compound"]) if vd_values_available else None,
            positive_mean  = float(vader_dict.get("positive_mean", 0.0)),
            negative_mean  = float(vader_dict.get("negative_mean", 0.0)),
            neutral_mean   = float(vader_dict.get("neutral_mean",  0.0)),
            label          = str(vader_dict.get("label", "Neutral")) if vd_values_available else None,
            total_chunks   = vd_total,
            skipped_chunks = int(vader_dict.get("skipped_chunks", 0)),
        )
        return self.calculate(
            finbert_result,
            vader_result,
            evidence_independence=evidence_independence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_independence(value: float | None) -> float | None:
        if value is None:
            return None
        return round(max(0.0, min(1.0, float(value))), 4)

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
