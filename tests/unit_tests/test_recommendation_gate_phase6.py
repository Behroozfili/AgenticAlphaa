from evaluation.recommendation_gate import build_recommendation_gate, contains_actionable_recommendation


def state(*, sentiment="VALID", sanity_passed=True, allowed=True, cap="HIGH"):
    return {"financial_metrics_summary": {"financial_sanity": {"passed": sanity_passed, "actionable_valuation_allowed": allowed, "confidence_cap": cap}}, "sentiment_analysis_summary": {"status": sentiment}}


def test_high_confidence_allows_actionable_recommendation():
    gate = build_recommendation_gate(state())
    assert gate["confidence_level"] == "HIGH"
    assert gate["actionable_recommendation_allowed"] is True


def test_medium_confidence_is_conditional_only():
    gate = build_recommendation_gate(state(cap="MEDIUM"))
    assert gate["confidence_level"] == "MEDIUM"
    assert gate["recommendation_mode"] == "CONDITIONAL_ONLY"
    assert gate["actionable_recommendation_allowed"] is False


def test_low_coverage_sentiment_prevents_actionable_output():
    gate = build_recommendation_gate(state(sentiment="LOW_COVERAGE"))
    assert gate["confidence_level"] == "LOW"
    assert gate["actionable_recommendation_allowed"] is False


def test_actionable_language_detection():
    assert contains_actionable_recommendation("Investors should accumulate shares")
    assert not contains_actionable_recommendation("The bull and bear scenarios remain highly uncertain")
