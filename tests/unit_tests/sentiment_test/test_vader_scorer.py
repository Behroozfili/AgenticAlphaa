"""
Tests for: tools/sentiment_tools/vader_scorer.py
Phase: 2c — Sentiment Tools

Mocking strategy: nltk.download and the lexicon lookup are mocked so tests
never hit the network or touch the filesystem cache. SentimentIntensityAnalyzer
itself is lightweight and deterministic (pure lexicon lookup, no ML model),
so we let the REAL analyzer run rather than mocking it — this gives us
actual confidence the wiring (mean aggregation, thresholds, truncation)
works correctly end to end. We only mock the lexicon bootstrap (_ensure_vader_lexicon)
to avoid the one-time network download in CI.
"""
from unittest.mock import patch, MagicMock
import pytest

import tools.sentiment_tools.vader_scorer as vader_module
from tools.sentiment_tools.vader_scorer import (
    VaderLexiconScorer,
    VaderResult,
    ChunkVaderScore,
    reset_vader,
)


@pytest.fixture(autouse=True)
def clean_vader():
    reset_vader()
    yield
    reset_vader()


@pytest.fixture(autouse=True)
def mock_lexicon_bootstrap():
    """
    Prevent our own one-time download check from hitting the network, WITHOUT
    touching nltk.data.find/nltk.download globally — SentimentIntensityAnalyzer
    itself also calls nltk.data.find internally to locate the lexicon file, so
    mocking it globally breaks real scoring (all scores silently become 0.0).
    Instead we mock our own `_ensure_vader_lexicon` wrapper directly, assuming
    the lexicon is already present in the test environment (downloaded once
    in CI setup, see test file docstring).
    """
    with patch("tools.sentiment_tools.vader_scorer._ensure_vader_lexicon"):
        yield


# ---------------------------------------------------------------------------
# score() — empty/skip handling
# ---------------------------------------------------------------------------

class TestScoreEmptyHandling:
    def test_all_empty_texts_returns_neutral_zero_result(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["", "   ", ""])

        assert isinstance(result, VaderResult)
        assert result.label == "Neutral"
        assert result.compound == 0.0
        assert result.total_chunks == 0
        assert result.skipped_chunks == 3

    def test_mixed_empty_and_valid_skips_only_empty(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["", "Great news for investors!", "   "])
        assert result.total_chunks == 1
        assert result.skipped_chunks == 2


# ---------------------------------------------------------------------------
# score() — directional labeling (using the real VADER lexicon)
# ---------------------------------------------------------------------------

class TestScoreDirectionalLabeling:
    def test_clearly_positive_text_is_bullish(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["This is wonderful, amazing, fantastic news!"])
        assert result.compound >= 0.05
        assert result.label == "Bullish"

    def test_clearly_negative_text_is_bearish(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["This is terrible, awful, horrible news."])
        assert result.compound <= -0.05
        assert result.label == "Bearish"

    def test_neutral_text_is_neutral(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["The meeting is scheduled for Tuesday."])
        assert -0.05 < result.compound < 0.05
        assert result.label == "Neutral"

    def test_corpus_label_uses_mean_compound_not_majority_vote(self):
        scorer = VaderLexiconScorer()
        # one strongly positive + one mildly negative -> mean should stay positive
        result = scorer.score([
            "Absolutely fantastic wonderful amazing results!",
            "Slightly disappointing quarter.",
        ])
        assert result.label in ("Bullish", "Neutral")  # mean dominated by the strong positive


# ---------------------------------------------------------------------------
# score() — aggregation shape
# ---------------------------------------------------------------------------

class TestScoreAggregation:
    def test_chunk_scores_length_matches_valid_texts(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["good", "bad", "ok"])
        assert len(result.chunk_scores) == 3
        assert all(isinstance(c, ChunkVaderScore) for c in result.chunk_scores)

    def test_means_are_arithmetic_mean_of_chunks(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["text one", "text two"])
        manual_mean = round(
            sum(c.compound for c in result.chunk_scores) / 2, 4
        )
        assert result.compound == manual_mean

    def test_chunk_text_truncated_to_120_chars(self):
        scorer = VaderLexiconScorer()
        long_text = "good " * 50  # > 120 chars
        result = scorer.score([long_text])
        assert result.chunk_scores[0].text.endswith("...")


# ---------------------------------------------------------------------------
# score_single()
# ---------------------------------------------------------------------------

class TestScoreSingle:
    def test_returns_chunk_vader_score(self):
        scorer = VaderLexiconScorer()
        result = scorer.score_single("Great results!")
        assert isinstance(result, ChunkVaderScore)
        assert result.label == "Bullish"

    def test_empty_string_raises_valueerror(self):
        scorer = VaderLexiconScorer()
        with pytest.raises(ValueError):
            scorer.score_single("   ")

    def test_whitespace_only_raises_valueerror(self):
        scorer = VaderLexiconScorer()
        with pytest.raises(ValueError):
            scorer.score_single("")


# ---------------------------------------------------------------------------
# _compound_label threshold boundaries (pure helper, white-box)
# ---------------------------------------------------------------------------

class TestCompoundLabelThresholds:
    def test_exactly_positive_threshold_is_bullish(self):
        assert VaderLexiconScorer._compound_label(0.05) == "Bullish"

    def test_exactly_negative_threshold_is_bearish(self):
        assert VaderLexiconScorer._compound_label(-0.05) == "Bearish"

    def test_just_inside_band_is_neutral(self):
        assert VaderLexiconScorer._compound_label(0.049) == "Neutral"
        assert VaderLexiconScorer._compound_label(-0.049) == "Neutral"

    def test_zero_is_neutral(self):
        assert VaderLexiconScorer._compound_label(0.0) == "Neutral"


# ---------------------------------------------------------------------------
# Lexicon bootstrap (_ensure_vader_lexicon) — double-checked locking
# ---------------------------------------------------------------------------

class TestLexiconBootstrap:
    @pytest.fixture(autouse=True)
    def mock_lexicon_bootstrap(self):
        """
        Override the module-level autouse fixture (which mocks
        _ensure_vader_lexicon itself) — this class needs the REAL
        _ensure_vader_lexicon body to run so it can verify the
        find/download double-checked-locking logic.
        """
        yield  # no-op: intentionally does NOT mock _ensure_vader_lexicon

    def test_downloads_only_when_not_found_locally(self):
        """
        Calls _ensure_vader_lexicon() DIRECTLY rather than via
        VaderLexiconScorer() — constructing the full scorer here would make
        the REAL SentimentIntensityAnalyzer.__init__() hit the SAME globally
        -mocked nltk.data.find() and also raise LookupError, which is not
        what this test is about (it only checks our own bootstrap wrapper).
        """
        from tools.sentiment_tools.vader_scorer import _ensure_vader_lexicon
        with patch("tools.sentiment_tools.vader_scorer.nltk.data.find",
                   side_effect=LookupError), \
             patch("tools.sentiment_tools.vader_scorer.nltk.download") as mock_download:
            reset_vader()
            _ensure_vader_lexicon()
            mock_download.assert_called_once_with("vader_lexicon", quiet=True)

    def test_skips_download_when_already_present(self):
        with patch("tools.sentiment_tools.vader_scorer.nltk.data.find") as mock_find, \
             patch("tools.sentiment_tools.vader_scorer.nltk.download") as mock_download:
            reset_vader()
            VaderLexiconScorer()
            # NOTE: real nltk.data.find() may legitimately call itself more
            # than once internally (e.g. once to detect a .zip component,
            # once for the actual file lookup) — so we assert it WAS called
            # (lexicon found locally), not that it was called exactly once.
            mock_find.assert_called()
            mock_download.assert_not_called()

    def test_second_instantiation_does_not_re_check_lexicon(self):
        """After the first load sets _VADER_LOADED=True, our OWN bootstrap
        wrapper should skip its find/download check on a second
        instantiation (NLTK's own internal SIA construction may still do
        its own lookup each time — that's outside our control)."""
        from tools.sentiment_tools.vader_scorer import _ensure_vader_lexicon
        with patch("tools.sentiment_tools.vader_scorer.nltk.data.find") as mock_find, \
             patch("tools.sentiment_tools.vader_scorer.nltk.download") as mock_download:
            reset_vader()
            _ensure_vader_lexicon()
            calls_after_first = mock_find.call_count
            _ensure_vader_lexicon()
            assert mock_find.call_count == calls_after_first  # no new calls
            mock_download.assert_not_called()