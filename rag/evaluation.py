"""
rag/evaluation.py — AlphaEvaluator
LLM-as-Judge evaluation framework for the RAG pipeline.

Metrics implemented:
─────────────────────────────────────────────────────────────────────────────
  1. Faithfulness        — Does every claim in the answer exist in the context?
                           (Reduces hallucination risk)
  2. Context Precision   — Are the retrieved chunks actually relevant to the query?
                           (Measures retrieval signal-to-noise ratio)
  3. Context Recall      — Does the context contain enough info to answer the query?
                           (Measures retrieval completeness)
  4. Answer Relevance    — How well does the answer address the original question?
─────────────────────────────────────────────────────────────────────────────

Each metric returns a normalised score in [0, 1] and an explanation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Schemas
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    score: float           # [0, 1]
    explanation: str
    metric: str


@dataclass
class EvaluationReport:
    query:             str
    answer:            str
    faithfulness:      MetricResult
    context_precision: MetricResult
    context_recall:    MetricResult
    answer_relevance:  MetricResult
    latency_seconds:   float = 0.0

    @property
    def overall_score(self) -> float:
        """Weighted mean: faithfulness has highest weight to penalise hallucination."""
        weights = {
            "faithfulness":      0.35,
            "context_precision": 0.25,
            "context_recall":    0.25,
            "answer_relevance":  0.15,
        }
        total = (
            self.faithfulness.score      * weights["faithfulness"]
            + self.context_precision.score * weights["context_precision"]
            + self.context_recall.score    * weights["context_recall"]
            + self.answer_relevance.score  * weights["answer_relevance"]
        )
        return round(total, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":             self.query,
            "overall_score":     self.overall_score,
            "faithfulness":      asdict(self.faithfulness),
            "context_precision": asdict(self.context_precision),
            "context_recall":    asdict(self.context_recall),
            "answer_relevance":  asdict(self.answer_relevance),
            "latency_seconds":   self.latency_seconds,
        }

    def summary(self) -> str:
        return (
            f"Overall: {self.overall_score:.2f} | "
            f"Faith: {self.faithfulness.score:.2f} | "
            f"Prec: {self.context_precision.score:.2f} | "
            f"Recall: {self.context_recall.score:.2f} | "
            f"Rel: {self.answer_relevance.score:.2f}"
        )


# ---------------------------------------------------------------------------
# LLM Judge Prompts
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """\
You are a strict fact-checker evaluating whether a generated answer is faithful to a given context.

CONTEXT:
{context}

ANSWER:
{answer}

TASK:
1. Extract every factual claim from the ANSWER.
2. For each claim, check if it can be directly inferred from the CONTEXT.
3. Score = (claims supported by context) / (total claims).

Return ONLY valid JSON with this exact structure:
{{
  "score": <float 0.0-1.0>,
  "total_claims": <int>,
  "supported_claims": <int>,
  "explanation": "<one-sentence rationale>"
}}"""

_CONTEXT_PRECISION_PROMPT = """\
You are evaluating the precision of retrieved context for answering a query.

QUERY: {query}

RETRIEVED CONTEXT CHUNKS:
{context}

TASK:
For each chunk, decide if it is relevant to answering the QUERY (relevant=1, not relevant=0).
Score = (relevant chunks) / (total chunks).

Return ONLY valid JSON:
{{
  "score": <float 0.0-1.0>,
  "total_chunks": <int>,
  "relevant_chunks": <int>,
  "explanation": "<one-sentence rationale>"
}}"""

_CONTEXT_RECALL_PROMPT = """\
You are evaluating whether the retrieved context contains sufficient information to answer a query.

QUERY: {query}
GROUND_TRUTH ANSWER: {ground_truth}

RETRIEVED CONTEXT:
{context}

TASK:
1. Extract the key facts needed to answer the query (from the ground truth).
2. Check which of those facts appear in the context.
3. Score = (key facts covered) / (total key facts).

Return ONLY valid JSON:
{{
  "score": <float 0.0-1.0>,
  "total_key_facts": <int>,
  "covered_facts": <int>,
  "explanation": "<one-sentence rationale>"
}}"""

_ANSWER_RELEVANCE_PROMPT = """\
You are evaluating how well a generated answer addresses the original query.

QUERY: {query}
ANSWER: {answer}

TASK:
Score how directly and completely the answer addresses the query.
- 1.0 = perfectly on-topic and complete
- 0.5 = partially relevant or incomplete
- 0.0 = off-topic or completely irrelevant

Return ONLY valid JSON:
{{
  "score": <float 0.0-1.0>,
  "explanation": "<one-sentence rationale>"
}}"""


# ---------------------------------------------------------------------------
# AlphaEvaluator
# ---------------------------------------------------------------------------

class AlphaEvaluator:
    """
    LLM-as-Judge RAG evaluator using Claude Sonnet 4 as the grader.

    Designed to run either as a one-shot evaluation or as a batch
    benchmark across multiple query-answer pairs.
    """

    JUDGE_MODEL = ""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )
        self.max_tokens  = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> EvaluationReport:
        """
        Run all four metrics and return an EvaluationReport.

        Args:
            query:        The original user query.
            answer:       The LLM-generated answer to evaluate.
            context:      The retrieved context string (from AlphaRetriever).
            ground_truth: Optional reference answer (required for Context Recall).
        """
        t0 = time.perf_counter()

        faith  = self._faithfulness(context, answer)
        prec   = self._context_precision(query, context)
        recall = self._context_recall(query, context, ground_truth or answer)
        rel    = self._answer_relevance(query, answer)

        report = EvaluationReport(
            query=query,
            answer=answer,
            faithfulness=faith,
            context_precision=prec,
            context_recall=recall,
            answer_relevance=rel,
            latency_seconds=round(time.perf_counter() - t0, 2),
        )
        logger.info("EvaluationReport: %s", report.summary())
        return report

    def batch_evaluate(
        self,
        samples: list[dict[str, str]],
    ) -> list[EvaluationReport]:
        """
        Evaluate multiple samples.

        Each dict must have keys: query, answer, context.
        Optional key: ground_truth.
        """
        reports = []
        for i, s in enumerate(samples):
            logger.info("Evaluating sample %d/%d ...", i + 1, len(samples))
            report = self.evaluate(
                query=s["query"],
                answer=s["answer"],
                context=s["context"],
                ground_truth=s.get("ground_truth"),
            )
            reports.append(report)

        # Aggregate summary
        if reports:
            avg_overall = sum(r.overall_score for r in reports) / len(reports)
            logger.info(
                "Batch evaluation complete: %d samples | Avg overall: %.3f",
                len(reports), avg_overall,
            )
        return reports

    def aggregate_scores(self, reports: list[EvaluationReport]) -> dict[str, float]:
        """Return mean scores across a batch of reports."""
        if not reports:
            return {}
        n = len(reports)
        return {
            "avg_overall":           round(sum(r.overall_score                  for r in reports) / n, 4),
            "avg_faithfulness":      round(sum(r.faithfulness.score              for r in reports) / n, 4),
            "avg_context_precision": round(sum(r.context_precision.score         for r in reports) / n, 4),
            "avg_context_recall":    round(sum(r.context_recall.score            for r in reports) / n, 4),
            "avg_answer_relevance":  round(sum(r.answer_relevance.score          for r in reports) / n, 4),
        }

    # ------------------------------------------------------------------
    # Individual Metric Evaluators
    # ------------------------------------------------------------------

    def _faithfulness(self, context: str, answer: str) -> MetricResult:
        prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        raw = self._call_judge(prompt)
        data = self._parse_json(raw, default_score=0.0)
        return MetricResult(
            score=float(data.get("score", 0.0)),
            explanation=data.get("explanation", raw[:200]),
            metric="faithfulness",
        )

    def _context_precision(self, query: str, context: str) -> MetricResult:
        prompt = _CONTEXT_PRECISION_PROMPT.format(query=query, context=context)
        raw = self._call_judge(prompt)
        data = self._parse_json(raw, default_score=0.0)
        return MetricResult(
            score=float(data.get("score", 0.0)),
            explanation=data.get("explanation", raw[:200]),
            metric="context_precision",
        )

    def _context_recall(
        self, query: str, context: str, ground_truth: str
    ) -> MetricResult:
        prompt = _CONTEXT_RECALL_PROMPT.format(
            query=query, ground_truth=ground_truth, context=context
        )
        raw = self._call_judge(prompt)
        data = self._parse_json(raw, default_score=0.0)
        return MetricResult(
            score=float(data.get("score", 0.0)),
            explanation=data.get("explanation", raw[:200]),
            metric="context_recall",
        )

    def _answer_relevance(self, query: str, answer: str) -> MetricResult:
        prompt = _ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
        raw = self._call_judge(prompt)
        data = self._parse_json(raw, default_score=0.0)
        return MetricResult(
            score=float(data.get("score", 0.0)),
            explanation=data.get("explanation", raw[:200]),
            metric="answer_relevance",
        )

    # ------------------------------------------------------------------
    # LLM Call
    # ------------------------------------------------------------------

    def _call_judge(self, prompt: str) -> str:
        try:
            message = self._client.messages.create(
                model=self.JUDGE_MODEL,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            logger.error("Judge LLM call failed: %s", exc)
            return '{"score": 0.0, "explanation": "LLM call failed."}'

    # ------------------------------------------------------------------
    # JSON Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str, default_score: float = 0.0) -> dict[str, Any]:
        """Extract JSON from a possibly markdown-wrapped judge response."""
        # Strip ```json ... ``` fences
        clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("Could not parse judge response as JSON: %s", text[:300])
            return {"score": default_score, "explanation": text[:200]}
