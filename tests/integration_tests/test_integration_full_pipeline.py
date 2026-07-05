"""
Tier 3 — full pipeline integration test.

Runs ManagerAgent.run() end-to-end: Research -> Financial -> Sentiment ->
Finalize. This is the slowest and most expensive tier (real LLM calls
across 4 agents, real API calls to Yahoo/SEC/news). Run deliberately, not
on every commit.

NOTE on the sentiment-stability test that used to live here: an earlier
version ran the same ticker through the full pipeline 3 times and required
the categorical sentiment fields to agree, as a regression test for the
temperature=0 determinism fix. That test was removed after real runs
proved it was measuring the wrong thing: SentimentAgent's live retrieval
returns a genuinely different set of articles on each call (confirmed
directly from logs — one run's chunk set included two articles no other
run saw, e.g. "Jim Cramer...", "...worst month since 2000"). With
genuinely different input evidence each time, different categorical
output is correct behavior, not a bug — this test could never reliably
tell "the LLM is unstable" apart from "the news changed between calls".

The properly-isolated version of this regression test now lives in
test_integration_sentiment_agent.py
(test_brain_analyze_is_deterministic_given_identical_input) — it calls
_brain_analyze directly with FROZEN finbert/vader/fear_greed input,
removing the retrieval confound entirely.
"""
import pytest


pytestmark = pytest.mark.slow


@pytest.mark.asyncio
async def test_full_pipeline_produces_complete_report(known_ticker, manager_agent_instance):
    manager = manager_agent_instance
    result = await manager.run(
        task_query=f"INVESTMENT ANALYSIS REPORT: {known_ticker} CORPORATION ({known_ticker})",
        manager_directives={"ticker": known_ticker, "search_depth": "advanced", "days_back": 14},
    )

    report = result.get("final_report", "")
    assert len(report) > 1000, "final_report suspiciously short — likely a partial failure upstream"

    # Regression for the filing-chunk truncation bug: if this section
    # header never appears, MD&A/risk_factors text likely never reached
    # the finalizer (the 200-char truncation bug silently ate it).
    expected_sections = [
        "Financial Health", "Market Sentiment", "Research Highlights",
        "Risk Factors", "Scenario Analysis", "Management Commentary",
        "Conclusion",
    ]
    missing_sections = [s for s in expected_sections if s not in report]
    assert not missing_sections, f"final_report missing sections: {missing_sections}"

    # The three data sources that must all feed the report (see the
    # RAGAS faithfulness investigation — this was the root cause of the
    # near-zero faithfulness scores when it silently failed).
    assert result.get("financial_metrics_summary")
    assert result.get("sentiment_analysis_summary")
    assert result.get("aggregated_research_context")