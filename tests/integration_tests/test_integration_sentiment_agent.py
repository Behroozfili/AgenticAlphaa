"""
test_sentiment_agent.py — full integration test suite for SentimentAgent,
in one file.

Organized in three tiers, run selectively via pytest markers:

    pytest test_sentiment_agent.py -v -m "not slow and not db"
        # Tier 1 only — pure computation, no LLM, no DB, no API. Fastest.
    pytest test_sentiment_agent.py -v -m "not slow"
        # Tier 1 + Tier 1.5 (needs a configured retriever/DB, no LLM)
    pytest test_sentiment_agent.py -v
        # everything, including the full agent run

Tier 1 (no marker): FinBertSentimentAnalyzer.analyze(), VaderLexiconScorer.score(),
    FearGreedIndexCalculator.calculate() — all synchronous, pure computation
    on text you supply directly. No network, no DB, no LLM. Should run in
    well under a second total.
Tier 1.5 (@pytest.mark.db): sentiment_server's retriever configuration and
    retrieve_social_data — needs a live vector store (Supabase) with real
    embedded data, but no LLM.
Tier 2 (@pytest.mark.slow): SentimentAgent.run() end-to-end — real LLM
    calls (Brain + Checker) plus the full retrieval pipeline.

Every test here is tied to a specific bug found and fixed during this
project's development — see each test's docstring for which one.
"""
import importlib
import json
from types import SimpleNamespace
import pytest

from tools.sentiment_tools.finbert_analyzer import FinBertSentimentAnalyzer
from tools.sentiment_tools.vader_scorer import VaderLexiconScorer, reset_vader
from tools.sentiment_tools.fear_greed_calculator import FearGreedIndexCalculator


# =============================================================================
# TIER 1 — FinBertSentimentAnalyzer
# =============================================================================

class TestFinBertAnalyzer:
    """Baseline schema/sanity tests — FinBERT itself has no known bug in
    this project, but a silent model/library upgrade changing the output
    schema would otherwise only surface two layers up, in a confusing
    Fear/Greed calculation error."""

    def test_clearly_bullish_text_scores_bullish(self):
        analyzer = FinBertSentimentAnalyzer()
        result = analyzer.analyze(["Company beats earnings estimates by a wide margin, raises guidance."])
        assert result.bullish_prob > result.bearish_prob
        assert result.label == "Bullish"

    def test_clearly_bearish_text_scores_bearish(self):
        analyzer = FinBertSentimentAnalyzer()
        result = analyzer.analyze(["Company misses earnings badly, slashes guidance, shares plunge."])
        assert result.bearish_prob > result.bullish_prob
        assert result.label == "Bearish"

    def test_probabilities_sum_to_approximately_one(self):
        analyzer = FinBertSentimentAnalyzer()
        result = analyzer.analyze(["The company reported quarterly results in line with expectations."])
        total = result.bullish_prob + result.bearish_prob + result.neutral_prob
        assert total == pytest.approx(1.0, abs=0.01)

    def test_empty_input_does_not_crash(self):
        analyzer = FinBertSentimentAnalyzer()
        result = analyzer.analyze([])
        assert result.total_chunks == 0
        assert result.status == "NO_DATA"
        assert result.label is None
        assert result.bullish_prob is None


# =============================================================================
# TIER 1 — VaderLexiconScorer
# =============================================================================

class TestVaderScorer:

    @pytest.fixture(autouse=True)
    def _clean_vader(self):
        yield
        reset_vader()

    def test_clearly_positive_text_scores_bullish(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["This is absolutely amazing news, stock is going to the moon! 🚀"])
        assert result.compound >= 0.05
        assert result.label == "Bullish"

    def test_clearly_negative_text_scores_bearish(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["This is a disaster, everyone is losing money, total crash."])
        assert result.compound <= -0.05
        assert result.label == "Bearish"

    def test_empty_list_returns_no_data_not_neutral(self):
        scorer = VaderLexiconScorer()
        result = scorer.score(["", "   "])
        assert result.total_chunks == 0
        assert result.skipped_chunks == 2
        assert result.status == "NO_DATA"
        assert result.label is None
        assert result.compound is None

    def test_score_single_rejects_empty_string(self):
        scorer = VaderLexiconScorer()
        with pytest.raises(ValueError):
            scorer.score_single("   ")


# =============================================================================
# TIER 1 — FearGreedIndexCalculator
# =============================================================================

class TestFearGreedCalculator:
    """The weighted-average formula and label bands are the core logic of
    this whole module — worth locking down precisely, not just
    approximately, since a silent change here (e.g. weights drifting from
    0.65/0.35) would shift every downstream sentiment label without
    anyone noticing."""

    def test_weighted_average_is_computed_exactly(self):
        from tools.sentiment_tools.finbert_analyzer import FinBertResult
        from tools.sentiment_tools.vader_scorer import VaderResult

        finbert = FinBertResult(bullish_prob=0.8, bearish_prob=0.2, neutral_prob=0.0,
                                 label="Bullish", total_chunks=1, skipped_chunks=0)
        vader = VaderResult(compound=0.4, positive_mean=0.5, negative_mean=0.1,
                             neutral_mean=0.4, label="Bullish", total_chunks=1, skipped_chunks=0)

        calc = FearGreedIndexCalculator()
        result = calc.calculate(finbert, vader)

        # finbert_score = 0.8 - 0.2 = 0.6; expected = 0.6*0.65 + 0.4*0.35 = 0.53
        assert result.score == pytest.approx(0.53, abs=1e-4)
        assert result.label == "Greed"  # falls in [0.20, 0.60)

    def test_label_band_boundaries(self):
        calc = FearGreedIndexCalculator()
        assert calc._score_to_label(0.60) == "Extreme Greed"
        assert calc._score_to_label(0.599) == "Greed"
        assert calc._score_to_label(0.20) == "Greed"
        assert calc._score_to_label(0.199) == "Neutral"
        assert calc._score_to_label(-0.20) == "Neutral"
        assert calc._score_to_label(-0.201) == "Fear"
        assert calc._score_to_label(-0.60) == "Fear"
        assert calc._score_to_label(-0.601) == "Extreme Fear"

    def test_invalid_weights_raise_valueerror(self):
        with pytest.raises(ValueError):
            FearGreedIndexCalculator(finbert_weight=0.5, vader_weight=0.6)  # sums to 1.1

    def test_negative_weight_raises_valueerror(self):
        with pytest.raises(ValueError):
            FearGreedIndexCalculator(finbert_weight=-0.1, vader_weight=1.1)

    def test_calculate_from_dict_matches_calculate(self):
        """Regression-style equivalence check: the dict convenience wrapper
        (used when deserializing MCP tool responses) must produce identical
        output to calling calculate() with the reconstructed dataclasses —
        a field-name mismatch here would silently produce a wrong score
        rather than an error, since missing dict keys have defaults."""
        finbert_dict = {"bullish_prob": 0.7, "bearish_prob": 0.1, "neutral_prob": 0.2,
                         "label": "Bullish", "total_chunks": 5, "skipped_chunks": 0}
        vader_dict = {"compound": 0.3, "positive_mean": 0.4, "negative_mean": 0.05,
                      "neutral_mean": 0.55, "label": "Bullish", "total_chunks": 5, "skipped_chunks": 0}

        calc = FearGreedIndexCalculator()
        result = calc.calculate_from_dict(finbert_dict, vader_dict)
        expected_finbert_score = 0.7 - 0.1
        expected_raw = expected_finbert_score * 0.65 + 0.3 * 0.35
        assert result.score == pytest.approx(round(expected_raw, 4))

    def test_all_zero_input_is_no_data_not_neutral(self):
        from tools.sentiment_tools.finbert_analyzer import FinBertResult
        from tools.sentiment_tools.vader_scorer import VaderResult

        finbert = FinBertResult(bullish_prob=0.0, bearish_prob=0.0, neutral_prob=0.0,
                                 label="Neutral", total_chunks=0, skipped_chunks=0)
        vader = VaderResult(compound=0.0, positive_mean=0.0, negative_mean=0.0,
                            neutral_mean=0.0, label="Neutral", total_chunks=0, skipped_chunks=0)
        result = FearGreedIndexCalculator().calculate(finbert, vader)
        assert result.status == "NO_DATA"
        assert result.score is None
        assert result.label is None


# =============================================================================
# TIER 1.5 — sentiment_server retriever configuration (needs live DB)
# =============================================================================

@pytest.mark.db
class TestSentimentRetrieverConfig:
    """Regression for the 'always exactly 3 chunks' bug: SentimentAgent's
    retriever used to run through the SAME diversity filter as
    ResearchAgent's (capped at 3 per source_type), starving it of context
    regardless of how much data was actually available. The fix disables
    freshness_rerank and diversity_filter specifically for the sentiment
    retriever. This test checks the CONFIGURATION directly — cheap and
    precise — rather than running a real query and counting chunks, which
    would also work but costs a live DB round-trip to prove the same
    thing."""

    def test_diversity_filter_and_freshness_rerank_are_disabled(self):
        from tools.sentiment_tools.sentiment_server import _get_retriever
        retriever = _get_retriever()
        assert retriever.apply_diversity_filter is False, (
            "diversity_filter re-enabled — this will silently re-cap "
            "SentimentAgent at ~3 chunks regardless of data available"
        )
        assert retriever.apply_freshness_rerank is False

    def test_token_budget_is_still_enabled_as_safety_net(self):
        """Disabling the other two filters doesn't mean disabling ALL
        limits — token_budget must stay on so a pathologically large
        stage1_k doesn't blow the LLM's context window."""
        from tools.sentiment_tools.sentiment_server import _get_retriever
        retriever = _get_retriever()
        assert retriever.apply_token_budget is True


# =============================================================================
# TIER 2 — SentimentAgent end-to-end (real LLM calls, mark as slow)
# =============================================================================

@pytest.mark.slow
class TestSentimentAgentEndToEnd:

    @pytest.mark.asyncio
    async def test_live_corpus_count_matches_returned_evidence_contract(self, known_ticker, sentiment_agent_instance):
        """Verify the live result describes this run's returned evidence.

        The live 14-day corpus may legitimately contain any number of rows.
        The deterministic diversity-filter regression is covered directly by
        ``TestSentimentRetrieverConfig`` above, so this integration test must
        not assume a time-varying minimum corpus size.
        """
        agent = sentiment_agent_instance
        result = await agent.run({
            "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker} CORPORATION ({known_ticker})",
            "manager_directives": {"ticker": known_ticker, "days_back": 14},
        })
        summary = result["sentiment_analysis_summary"]
        attempts = summary["attempts"]
        selected_attempt = next(
            (
                attempt for attempt in attempts
                if attempt.get("attempt_number") == summary.get("selected_attempt_number")
            ),
            attempts[-1] if attempts else {"retrieval": {"chunks": []}},
        )
        returned_chunks = selected_attempt.get("retrieval", {}).get("chunks", [])

        assert summary["total_chunks_analyzed"] == len(returned_chunks)
        if returned_chunks:
            assert summary["status"] in {"VALID", "LOW_COVERAGE"}
            assert summary["overall_sentiment"] is not None
        else:
            assert summary["status"] == "NO_DATA"
            assert summary["polarity"] is None
            assert summary["overall_sentiment"] is None

    @pytest.mark.asyncio
    async def test_populates_all_expected_fields(self, known_ticker, sentiment_agent_instance):
        agent = sentiment_agent_instance
        result = await agent.run({
            "task_query": f"INVESTMENT ANALYSIS REPORT: {known_ticker} CORPORATION ({known_ticker})",
            "manager_directives": {"ticker": known_ticker, "days_back": 14},
        })
        summary = result["sentiment_analysis_summary"]
        required_keys = [
            "overall_sentiment", "conviction_level", "fear_greed_score",
            "fear_greed_label", "finbert_label", "vader_label",
            "model_agreement", "total_chunks_analyzed", "narrative",
            "execution_success", "quality_passed", "confidence_components",
            "source_document_counts", "aspect_observations",
            "causal_aspects_supported", "attempts", "extraction_errors",
        ]
        missing = [k for k in required_keys if k not in summary]
        assert not missing, f"sentiment_analysis_summary missing keys: {missing}"

    @pytest.mark.asyncio
    async def test_brain_analyze_is_deterministic_given_identical_input(self, sentiment_agent_instance):
        """
        Regression for the pre-temperature=0 instability bug (the original
        motivation for this whole project) — but isolated from retrieval.

        A previous version of this test ran the full SentimentAgent.run()
        (live retrieval + LLM) 2-3 times and compared categorical labels.
        That approach was scrapped after real runs showed the underlying
        RETRIEVED CHUNKS themselves differ between successive live calls
        (confirmed directly from logs: one run's 7 chunks included two
        articles — "Jim Cramer...", "worst month since 2000" — that
        appeared in NO other run, while another run's article set was
        missing an article every other run had). With genuinely different
        input evidence each time, different categorical output is CORRECT
        behavior, not a temperature bug — and that test could never
        reliably distinguish "LLM is unstable" from "the news changed".

        This test removes that confound entirely: it calls _brain_analyze
        directly with a FROZEN, hardcoded finbert/vader/fear_greed input
        (captured from a real run) three times, and requires the
        categorical fields in the output to be identical every time. Since
        the input never changes, any disagreement here can ONLY be the
        LLM, not retrieval drift — a much cleaner regression test for the
        actual thing (temperature=0) this project fixed.
        """
        agent = sentiment_agent_instance

        # Frozen input, captured from a real MSFT run — deliberately a
        # borderline/mixed case (FinBERT bearish-leaning, VADER mildly
        # bullish) since that's exactly the kind of input where an
        # under-determined (non-zero-temperature) LLM would be most
        # likely to flip between calls.
        frozen_finbert = {
            "bullish_prob": 0.2235, "bearish_prob": 0.4626, "neutral_prob": 0.3139,
            "label": "Bearish", "total_chunks": 6, "skipped_chunks": 0,
        }
        frozen_vader = {
            "compound": 0.1287, "positive_mean": 0.0553, "negative_mean": 0.0443,
            "neutral_mean": 0.9002, "label": "Bullish", "total_chunks": 6, "skipped_chunks": 0,
        }
        frozen_fear_greed = {
            "score": -0.1104, "label": "Neutral",
            "finbert_score": -0.2391, "vader_score": 0.1287,
            "weights": {"finbert": 0.65, "vader": 0.35}, "confidence": 0.1104,
            "diagnostics": {"ticker": "MSFT"},
        }

        def make_fresh_state():
            return {
                "shared_manager_ref": {
                    "task_query": "INVESTMENT ANALYSIS REPORT: MSFT CORPORATION (MSFT)",
                    "financial_metrics_summary": {},
                },
                "fear_greed_result": frozen_fear_greed,
                "finbert_result": frozen_finbert,
                "vader_result": frozen_vader,
                "retrieved_chunks": ["chunk"] * 6,
                "sources_metadata": [],
                "aspect_observations": [],
                "sentiment_status": "VALID",
                "manager_directives": {},
                "messages": [],
            }

        outcomes = []
        for _ in range(3):
            raw_json = await agent._brain_analyze(make_fresh_state())
            parsed = json.loads(raw_json)
            outcomes.append((parsed.get("overall_sentiment"), parsed.get("conviction_level")))

        assert len(set(outcomes)) == 1, (
            f"_brain_analyze produced different categorical output across 3 calls "
            f"with IDENTICAL frozen input: {outcomes} — since the input never "
            f"changed here, this points at temperature not actually being 0 on "
            f"the LLM call, not at retrieval/live-data variance."
        )


class TestSentimentValidationExport:
    @staticmethod
    def _trace_module(monkeypatch):
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        return importlib.import_module("langsmith_trace")

    @pytest.mark.parametrize(
        ("overrides", "expected_status"),
        [
            ({"status": "LOW_COVERAGE"}, "LOW_COVERAGE"),
            ({"causal_mismatch_detected": True}, "VALID"),
            ({"fear_greed_score": None}, "VALID"),
        ],
        ids=["low-coverage", "causal-mismatch", "missing-score"],
    )
    def test_failed_sentiment_quality_is_never_exported_as_success(
        self, monkeypatch, overrides, expected_status,
    ):
        trace_module = self._trace_module(monkeypatch)
        summary = {
            "status": "VALID",
            "execution_success": True,
            "pipeline_execution_success": True,
            "quality_passed": True,
            "fear_greed_score": 0.2,
            "unique_document_count": 3,
            "unique_domain_count": 2,
            "extraction_errors": [],
            "causal_mismatch_detected": False,
        }
        summary.update(overrides)

        exported = trace_module.extract_sentiment_validation(
            {"sentiment_analysis_summary": summary}
        )

        assert exported["status"] == expected_status
        assert exported["execution_success"] is True
        assert exported["quality_passed"] is False

    def test_parse_trace_exports_complete_sentiment_audit(self, monkeypatch):
        trace_module = self._trace_module(monkeypatch)
        summary = {
            "status": "VALID",
            "execution_success": True,
            "pipeline_execution_success": True,
            "quality_passed": True,
            "fear_greed_score": 0.2,
            "fear_greed_label": "Greed",
            "fear_greed_confidence": 0.8,
            "unique_document_count": 3,
            "unique_domain_count": 2,
            "extraction_errors": [],
            "causal_mismatch_detected": False,
            "confidence_components": {
                "confidence": 0.8,
                "direction_strength": 0.2,
                "model_agreement_score": 0.9,
                "evidence_independence": 0.6667,
            },
            "source_document_counts": {
                "raw_chunk_count": 5,
                "unique_documents": 3,
                "unique_domains": 2,
            },
            "aspect_observations": [{"aspect": "valuation"}],
            "causal_aspects_supported": [],
            "attempts": [{"attempt_number": 1, "status": "VALID"}],
            "attempt_extraction_errors": ["first attempt timeout"],
        }
        root = SimpleNamespace(
            inputs={
                "task_query": "Analyze NVDA",
                "manager_directives": {"ticker": "NVDA"},
            },
            outputs={"sentiment_analysis_summary": summary},
            id="run-1",
            trace_id="trace-1",
            start_time="2026-08-24T00:00:00Z",
            end_time=None,
            latency=1.0,
            status="success",
            error=None,
        )
        monkeypatch.setattr(
            trace_module,
            "client",
            SimpleNamespace(list_runs=lambda **kwargs: []),
        )

        artifact = trace_module.parse_trace(root, "test-project")

        assert artifact["sentiment_analysis_summary"] == summary
        audit = artifact["sentiment_validation"]
        assert audit["quality_passed"] is True
        assert audit["confidence_components"]["model_agreement_score"] == 0.9
        assert audit["source_document_counts"]["unique_documents"] == 3
        assert audit["aspect_observations"] == [{"aspect": "valuation"}]
        assert audit["causal_aspects_supported"] == []
        assert audit["attempts"][0]["attempt_number"] == 1
        assert audit["attempt_extraction_errors"] == ["first attempt timeout"]
