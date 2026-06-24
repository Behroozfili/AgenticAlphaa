"""
Tests for: rag/evaluation.py
Phase: 3 — RAG Pipeline (8th: final stage, depends on nothing else in rag/)

Mocking strategy: anthropic.Anthropic is mocked entirely — no real LLM calls.
Each judge-call test controls the mocked response text to exercise JSON
parsing, markdown-fence stripping, and the catch-and-default-to-0.0 path
on malformed judge output.
"""
import os
from unittest.mock import patch, MagicMock
import pytest

from rag.evaluation import AlphaEvaluator, MetricResult, EvaluationReport


def make_judge_response(json_text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=json_text)]
    return resp


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with patch("rag.evaluation.anthropic.Anthropic") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        ev = AlphaEvaluator()
        ev._client = client
        yield ev, client


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_missing_api_key_raises_keyerror(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("rag.evaluation.anthropic.Anthropic"):
            with pytest.raises(KeyError):
                AlphaEvaluator()

    def test_explicit_api_key_used_over_env(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with patch("rag.evaluation.anthropic.Anthropic") as mock_cls:
            AlphaEvaluator(api_key="explicit-key")
            mock_cls.assert_called_once_with(api_key="explicit-key")


# ---------------------------------------------------------------------------
# _parse_json — JSON extraction from judge responses
# ---------------------------------------------------------------------------

class TestParseJson:
    def test_valid_json_parsed_directly(self):
        result = AlphaEvaluator._parse_json('{"score": 0.8, "explanation": "good"}')
        assert result == {"score": 0.8, "explanation": "good"}

    def test_markdown_fenced_json_is_stripped(self):
        text = '```json\n{"score": 0.5, "explanation": "ok"}\n```'
        result = AlphaEvaluator._parse_json(text)
        assert result["score"] == 0.5

    def test_invalid_json_returns_default_score_and_truncated_text(self):
        result = AlphaEvaluator._parse_json("not json at all", default_score=0.0)
        assert result["score"] == 0.0
        assert "not json" in result["explanation"]

    def test_custom_default_score_used_on_parse_failure(self):
        result = AlphaEvaluator._parse_json("garbage", default_score=0.25)
        assert result["score"] == 0.25


# ---------------------------------------------------------------------------
# _call_judge — LLM invocation + failure fallback
# ---------------------------------------------------------------------------

class TestCallJudge:
    def test_returns_model_text_on_success(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 1.0}')
        result = ev._call_judge("some prompt")
        assert result == '{"score": 1.0}'

    def test_exception_returns_safe_fallback_json_string(self, evaluator):
        ev, client = evaluator
        client.messages.create.side_effect = RuntimeError("API down")
        result = ev._call_judge("prompt")
        assert '"score": 0.0' in result
        assert "LLM call failed" in result

    def test_uses_configured_model_max_tokens_temperature(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 1.0}')
        ev._call_judge("prompt")
        _, kwargs = client.messages.create.call_args
        assert kwargs["model"] == AlphaEvaluator.JUDGE_MODEL
        assert kwargs["max_tokens"] == ev.max_tokens
        assert kwargs["temperature"] == ev.temperature


# ---------------------------------------------------------------------------
# Individual metric methods — correct prompt formatting + result mapping
# ---------------------------------------------------------------------------

class TestIndividualMetrics:
    def test_faithfulness_maps_score_and_explanation(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response(
            '{"score": 0.9, "explanation": "well supported"}'
        )
        result = ev._faithfulness(context="ctx", answer="ans")
        assert isinstance(result, MetricResult)
        assert result.score == 0.9
        assert result.metric == "faithfulness"

    def test_context_precision_maps_correctly(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.6}')
        result = ev._context_precision(query="q", context="ctx")
        assert result.metric == "context_precision"
        assert result.score == 0.6

    def test_context_recall_uses_ground_truth_in_prompt(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.7}')
        ev._context_recall(query="q", context="ctx", ground_truth="gt answer")
        sent_prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
        assert "gt answer" in sent_prompt

    def test_answer_relevance_maps_correctly(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 1.0}')
        result = ev._answer_relevance(query="q", answer="a")
        assert result.metric == "answer_relevance"
        assert result.score == 1.0

    def test_missing_explanation_falls_back_to_raw_text_truncated(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.5}')
        result = ev._faithfulness(context="ctx", answer="ans")
        assert result.explanation == '{"score": 0.5}'  # raw[:200] fallback


# ---------------------------------------------------------------------------
# evaluate() — full orchestration + overall_score weighting
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_calls_all_four_metrics(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.8}')
        report = ev.evaluate(query="q", answer="a", context="c")
        assert client.messages.create.call_count == 4
        assert isinstance(report, EvaluationReport)

    def test_overall_score_weighted_correctly(self, evaluator):
        ev, client = evaluator
        # Return different scores per call to verify weights are applied distinctly
        responses = ['{"score": 1.0}', '{"score": 0.0}', '{"score": 0.0}', '{"score": 0.0}']
        client.messages.create.side_effect = [make_judge_response(r) for r in responses]

        report = ev.evaluate(query="q", answer="a", context="c")
        # faithfulness=1.0 weight 0.35, others 0.0 -> overall = 0.35
        assert report.overall_score == 0.35

    def test_ground_truth_defaults_to_answer_when_not_provided(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.5}')
        ev.evaluate(query="q", answer="my answer", context="c")
        # context_recall is the 3rd call; its prompt should use "my answer" as ground truth
        third_call_prompt = client.messages.create.call_args_list[2].kwargs["messages"][0]["content"]
        assert "my answer" in third_call_prompt

    def test_latency_seconds_is_non_negative_float(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.5}')
        report = ev.evaluate(query="q", answer="a", context="c")
        assert report.latency_seconds >= 0.0


# ---------------------------------------------------------------------------
# EvaluationReport — to_dict / summary
# ---------------------------------------------------------------------------

class TestEvaluationReportFormatting:
    def _make_report(self):
        m = lambda s, name: MetricResult(score=s, explanation="x", metric=name)
        return EvaluationReport(
            query="q", answer="a",
            faithfulness=m(1.0, "faithfulness"),
            context_precision=m(0.5, "context_precision"),
            context_recall=m(0.5, "context_recall"),
            answer_relevance=m(1.0, "answer_relevance"),
            latency_seconds=1.23,
        )

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["query"] == "q"
        assert "overall_score" in d
        assert d["faithfulness"]["score"] == 1.0

    def test_summary_contains_all_metric_labels(self):
        report = self._make_report()
        summary = report.summary()
        assert "Overall" in summary
        assert "Faith" in summary
        assert "Prec" in summary
        assert "Recall" in summary
        assert "Rel" in summary


# ---------------------------------------------------------------------------
# batch_evaluate / aggregate_scores
# ---------------------------------------------------------------------------

class TestBatchEvaluate:
    def test_batch_evaluate_processes_all_samples(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.5}')
        samples = [
            {"query": "q1", "answer": "a1", "context": "c1"},
            {"query": "q2", "answer": "a2", "context": "c2"},
        ]
        reports = ev.batch_evaluate(samples)
        assert len(reports) == 2

    def test_batch_evaluate_uses_explicit_ground_truth_when_provided(self, evaluator):
        ev, client = evaluator
        client.messages.create.return_value = make_judge_response('{"score": 0.5}')
        samples = [{"query": "q", "answer": "a", "context": "c", "ground_truth": "gt"}]
        ev.batch_evaluate(samples)
        recall_call_prompt = client.messages.create.call_args_list[2].kwargs["messages"][0]["content"]
        assert "gt" in recall_call_prompt

    def test_empty_samples_returns_empty_list(self, evaluator):
        ev, client = evaluator
        assert ev.batch_evaluate([]) == []

    def test_aggregate_scores_computes_means(self, evaluator):
        ev, client = evaluator
        m = lambda s, name: MetricResult(score=s, explanation="x", metric=name)
        reports = [
            EvaluationReport(query="q1", answer="a1", faithfulness=m(1.0, "faithfulness"),
                             context_precision=m(1.0, "context_precision"),
                             context_recall=m(1.0, "context_recall"),
                             answer_relevance=m(1.0, "answer_relevance")),
            EvaluationReport(query="q2", answer="a2", faithfulness=m(0.0, "faithfulness"),
                             context_precision=m(0.0, "context_precision"),
                             context_recall=m(0.0, "context_recall"),
                             answer_relevance=m(0.0, "answer_relevance")),
        ]
        agg = ev.aggregate_scores(reports)
        assert agg["avg_faithfulness"] == 0.5
        assert agg["avg_overall"] == 0.5

    def test_aggregate_scores_empty_list_returns_empty_dict(self, evaluator):
        ev, client = evaluator
        assert ev.aggregate_scores([]) == {}