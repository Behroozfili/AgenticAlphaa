"""
Tests for: tools/sentiment_tools/fear_greed_calculator.py
Phase: 1 — Pure-Logic / Zero-Mock Foundations

No mocking required — pure computation over already-extracted FinBERT/VADER
scores. We use lightweight stand-ins (SimpleNamespace) for FinBertResult and
VaderResult instead of mocks: calculate() only reads attributes (duck typing),
so no behavior needs to be mocked, only data needs to be supplied.

NOTE ON IMPORTS: fear_greed_calculator.py imports FinBertResult and VaderResult
from tools.sentiment_tools.finbert_analyzer / vader_scorer at module load time.
If tools/sentiment_tools/__init__.py still has the broken imports documented
in Phase 0, this entire test file will fail to collect. Fix Phase 0 first.
"""
from types import SimpleNamespace

import pytest

from tools.sentiment_tools.fear_greed_calculator import (
    FearGreedIndexCalculator,
    FearGreedResult,
)


def make_finbert(bullish=0.5, bearish=0.3, neutral=0.2, label="Neutral",
                  total_chunks=10, skipped_chunks=0):
    return SimpleNamespace(
        bullish_prob=bullish,
        bearish_prob=bearish,
        neutral_prob=neutral,
        label=label,
        total_chunks=total_chunks,
        skipped_chunks=skipped_chunks,
    )


def make_vader(compound=0.0, positive_mean=0.0, negative_mean=0.0,
               neutral_mean=0.0, label="Neutral", total_chunks=10,
               skipped_chunks=0):
    return SimpleNamespace(
        compound=compound,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        neutral_mean=neutral_mean,
        label=label,
        total_chunks=total_chunks,
        skipped_chunks=skipped_chunks,
    )


# ---------------------------------------------------------------------------
# Construction / weight validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_weights(self):
        calc = FearGreedIndexCalculator()
        assert calc.finbert_weight == 0.65
        assert calc.vader_weight == 0.35

    def test_custom_weights_accepted(self):
        calc = FearGreedIndexCalculator(finbert_weight=0.5, vader_weight=0.5)
        assert calc.finbert_weight == 0.5
        assert calc.vader_weight == 0.5

    def test_negative_weight_raises_value_error(self):
        with pytest.raises(ValueError):
            FearGreedIndexCalculator(finbert_weight=-0.1, vader_weight=1.1)

    def test_weights_not_summing_to_one_raises_value_error(self):
        with pytest.raises(ValueError):
            FearGreedIndexCalculator(finbert_weight=0.5, vader_weight=0.6)

    def test_weights_within_tolerance_are_accepted(self):
        # 0.001 tolerance: 0.6505 + 0.3495 = 1.0000 (within 1e-3 of 1.0)
        calc = FearGreedIndexCalculator(finbert_weight=0.6505, vader_weight=0.3495)
        assert calc.finbert_weight == 0.6505


# ---------------------------------------------------------------------------
# calculate() — directional score + weighted average
# ---------------------------------------------------------------------------

class TestCalculateScoreMath:
    def test_finbert_score_is_bullish_minus_bearish(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(
            make_finbert(bullish=0.8, bearish=0.1), make_vader(compound=0.0)
        )
        assert result.finbert_score == 0.7

    def test_vader_score_is_compound_directly(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(make_finbert(bullish=0.0, bearish=0.0), make_vader(compound=0.4))
        assert result.vader_score == 0.4

    def test_weighted_average_formula(self):
        calc = FearGreedIndexCalculator(finbert_weight=0.6, vader_weight=0.4)
        result = calc.calculate(
            make_finbert(bullish=0.5, bearish=0.0),  # finbert_score = 0.5
            make_vader(compound=0.5),                # vader_score = 0.5
        )
        # raw = 0.5*0.6 + 0.5*0.4 = 0.5
        assert result.score == 0.5

    def test_weights_dict_reflects_instance_weights(self):
        calc = FearGreedIndexCalculator(finbert_weight=0.7, vader_weight=0.3)
        result = calc.calculate(make_finbert(), make_vader())
        assert result.weights == {"finbert": 0.7, "vader": 0.3}


# ---------------------------------------------------------------------------
# calculate() — clamping to [-1, +1]
# ---------------------------------------------------------------------------

class TestClamping:
    def test_score_clamped_at_positive_one(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(
            make_finbert(bullish=1.0, bearish=0.0),  # finbert_score=1.0
            make_vader(compound=1.0),                 # vader_score=1.0
        )
        assert result.score <= 1.0
        assert result.score == 1.0

    def test_score_clamped_at_negative_one(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(
            make_finbert(bullish=0.0, bearish=1.0),
            make_vader(compound=-1.0),
        )
        assert result.score >= -1.0
        assert result.score == -1.0


# ---------------------------------------------------------------------------
# calculate() — label bands
# ---------------------------------------------------------------------------

class TestLabelBands:
    calc = FearGreedIndexCalculator()

    def test_extreme_greed_at_0_60(self):
        result = self.calc.calculate(make_finbert(bullish=0.6, bearish=0.0), make_vader(compound=0.6))
        assert result.score >= 0.60
        assert result.label == "Extreme Greed"

    def test_greed_band(self):
        result = self.calc.calculate(make_finbert(bullish=0.3, bearish=0.0), make_vader(compound=0.3))
        assert result.label == "Greed"

    def test_neutral_band(self):
        result = self.calc.calculate(make_finbert(bullish=0.0, bearish=0.0), make_vader(compound=0.0))
        assert result.label == "Neutral"

    def test_fear_band(self):
        result = self.calc.calculate(make_finbert(bullish=0.0, bearish=0.3), make_vader(compound=-0.3))
        assert result.label == "Fear"

    def test_extreme_fear_band(self):
        result = self.calc.calculate(make_finbert(bullish=0.0, bearish=0.7), make_vader(compound=-0.7))
        assert result.label == "Extreme Fear"

    def test_boundary_exactly_0_20_is_greed_not_neutral(self):
        # band thresholds use >= comparisons in order, 0.20 is lower bound of "Greed"
        calc = FearGreedIndexCalculator(finbert_weight=1.0, vader_weight=0.0)
        result = calc.calculate(make_finbert(bullish=0.20, bearish=0.0), make_vader(compound=0.0))
        assert result.score == 0.20
        assert result.label == "Greed"


# ---------------------------------------------------------------------------
# calculate() — confidence + diagnostics
# ---------------------------------------------------------------------------

class TestConfidenceAndDiagnostics:
    def test_confidence_equals_absolute_score(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(
            make_finbert(bullish=0.0, bearish=0.5), make_vader(compound=-0.5)
        )
        assert result.confidence == round(abs(result.score), 4)

    def test_confidence_zero_for_neutral_score(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(make_finbert(bullish=0.0, bearish=0.0), make_vader(compound=0.0))
        assert result.confidence == 0.0

    def test_diagnostics_contains_all_raw_inputs(self):
        calc = FearGreedIndexCalculator()
        fb = make_finbert(bullish=0.4, bearish=0.2, neutral=0.4, label="Bullish", total_chunks=5)
        vd = make_vader(compound=0.1, positive_mean=0.3, negative_mean=0.1, label="Positive", total_chunks=5)
        result = calc.calculate(fb, vd)
        assert result.diagnostics["finbert_bullish_prob"] == 0.4
        assert result.diagnostics["finbert_label"] == "Bullish"
        assert result.diagnostics["vader_compound"] == 0.1
        assert result.diagnostics["vader_label"] == "Positive"

    def test_zero_chunks_both_models_returns_neutral_not_crash(self):
        """If both models analyzed zero chunks, calculator must still run and
        default to a neutral 0.0 score rather than raising — per docstring."""
        calc = FearGreedIndexCalculator()
        fb = make_finbert(bullish=0.0, bearish=0.0, total_chunks=0)
        vd = make_vader(compound=0.0, total_chunks=0)
        result = calc.calculate(fb, vd)
        assert result.score == 0.0
        assert result.label == "Neutral"

    def test_result_is_feargreedresult_instance(self):
        calc = FearGreedIndexCalculator()
        result = calc.calculate(make_finbert(), make_vader())
        assert isinstance(result, FearGreedResult)


# ---------------------------------------------------------------------------
# calculate_from_dict()
# ---------------------------------------------------------------------------

class TestCalculateFromDict:
    def test_reconstructs_and_matches_calculate(self):
        calc = FearGreedIndexCalculator()
        finbert_dict = {
            "bullish_prob": 0.7, "bearish_prob": 0.2, "neutral_prob": 0.1,
            "label": "Bullish", "total_chunks": 3, "skipped_chunks": 0,
        }
        vader_dict = {
            "compound": 0.3, "positive_mean": 0.2, "negative_mean": 0.0,
            "neutral_mean": 0.8, "label": "Positive", "total_chunks": 3,
            "skipped_chunks": 0,
        }
        result_from_dict = calc.calculate_from_dict(finbert_dict, vader_dict)
        result_direct = calc.calculate(
            make_finbert(bullish=0.7, bearish=0.2, neutral=0.1, label="Bullish", total_chunks=3),
            make_vader(compound=0.3, positive_mean=0.2, negative_mean=0.0,
                       neutral_mean=0.8, label="Positive", total_chunks=3),
        )
        assert result_from_dict.score == result_direct.score
        assert result_from_dict.label == result_direct.label

    def test_missing_required_key_raises_keyerror(self):
        calc = FearGreedIndexCalculator()
        with pytest.raises(KeyError):
            calc.calculate_from_dict({}, {"compound": 0.0})

    def test_missing_optional_keys_use_defaults(self):
        calc = FearGreedIndexCalculator()
        finbert_dict = {"bullish_prob": 0.5, "bearish_prob": 0.1, "neutral_prob": 0.4}
        vader_dict = {"compound": 0.2}
        result = calc.calculate_from_dict(finbert_dict, vader_dict)
        # label defaults to "Neutral", total_chunks defaults to 0 — should not raise
        assert result.diagnostics["finbert_label"] == "Neutral"
        assert result.diagnostics["finbert_total_chunks"] == 0


# ---------------------------------------------------------------------------
# Internal helpers: _score_to_label / _validate_weights (white-box)
# ---------------------------------------------------------------------------

class TestInternalHelpers:
    def test_score_to_label_directly(self):
        assert FearGreedIndexCalculator._score_to_label(0.9) == "Extreme Greed"
        assert FearGreedIndexCalculator._score_to_label(-0.9) == "Extreme Fear"
        assert FearGreedIndexCalculator._score_to_label(0.0) == "Neutral"

    def test_validate_weights_negative_raises(self):
        with pytest.raises(ValueError):
            FearGreedIndexCalculator._validate_weights(-0.1, 1.1)

    def test_validate_weights_sum_mismatch_raises(self):
        with pytest.raises(ValueError):
            FearGreedIndexCalculator._validate_weights(0.5, 0.6)

    def test_validate_weights_valid_does_not_raise(self):
        FearGreedIndexCalculator._validate_weights(0.65, 0.35)  # should not raise