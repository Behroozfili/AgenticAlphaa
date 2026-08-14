import pytest

from evaluation.publication_policy import assemble_publication


def state(*, actionable=True, data=True):
    fm={"current_price":100,"roe":{"roe_pct":12}} if data else {}
    return {"financial_metrics_summary":fm,"sentiment_analysis_summary":{"status":"VALID"} if data else {"status":"NO_DATA"},"aggregated_research_context":[],"recommendation_gate":{"confidence_level":"HIGH" if actionable else "LOW","recommendation_mode":"ACTIONABLE_ALLOWED" if actionable else "SCENARIO_SUMMARY_ONLY","actionable_recommendation_allowed":actionable}}


def validation(*findings):
    rows=[{"check":check,"message":message} for check,message in findings]
    return {"passed":not rows,"critical_findings":rows,"warnings":[],"required_regeneration_context":rows,"checks":{}}


def test_valid_report_is_published():
    out=assemble_publication("# Analysis\nValidated analysis.",validation(),state())
    assert out["publication_status"]=="PUBLISHED"
    assert "Validated analysis" in out["final_report"]
    assert "Valuation Reliability: RELIABLE" in out["final_report"]
    assert all(x["status"]=="RELIABLE" for x in out["section_reliability"])


def test_minor_inconsistency_publishes_with_warnings():
    out=assemble_publication("# Analysis\nUseful analysis.",validation(("arithmetic_consistency","Minor narrative arithmetic mismatch.")),state())
    assert out["publication_status"]=="PUBLISHED_WITH_WARNINGS"
    assert "Useful analysis" in out["final_report"]
    assert "Financial Performance Reliability: LIMITED" in out["final_report"]
    assert "Minor narrative arithmetic mismatch" not in out["final_report"]


def test_critical_valuation_issue_withholds_only_valuation_section():
    report="# Executive Summary\nBusiness remains profitable.\n## DCF Valuation\nUnreliable valuation.\n## Risks\nCompetition remains material."
    out=assemble_publication(report,validation(("valuation_bridge_consistency","Equity bridge incomplete.")),state())
    assert out["publication_status"]=="LIMITED_REPORT"
    assert "Business remains profitable" in out["final_report"]
    assert "Competition remains material" in out["final_report"]
    assert "Unreliable valuation" not in out["final_report"]
    assert "DCF Valuation" in out["withheld_sections"]
    assert "Valuation Reliability: WITHHELD" in out["final_report"]
    assert "No directional valuation conclusion" in out["final_report"]
    assert "Recommendation confidence: **AFFECTED**" in out["final_report"]


def test_low_recommendation_confidence_keeps_analysis_visible():
    report="# Analysis\nRevenue improved. Investors should buy shares.\n## Risks\nExecution risk remains."
    out=assemble_publication(report,validation(),state(actionable=False))
    assert out["publication_status"]=="BLOCKED_RECOMMENDATION"
    assert out["final_report"]
    assert "Execution risk remains" in out["final_report"]
    assert "should buy" not in out["final_report"].lower()


def test_insufficient_data_returns_explanatory_report():
    out=assemble_publication("",validation(),state(data=False))
    assert out["publication_status"]=="DATA_INSUFFICIENT"
    assert "Data available" in out["final_report"]
    assert "Data missing or unusable" in out["final_report"]
    assert "Conclusions still supportable" in out["final_report"]
    assert "Valuation Reliability: DATA_INSUFFICIENT" in out["final_report"]


@pytest.mark.parametrize("finding",[
    ("scenario_consistency","Scenario conflict."),
    ("management_quote_fidelity","Unsupported quote."),
    ("recommendation_gate_consistency","Recommendation blocked."),
])
def test_no_validation_condition_silently_empties_report(finding):
    out=assemble_publication("# DCF Valuation\nCandidate valuation text.",validation(finding),state())
    assert out["final_report"].strip()


def test_user_disclosure_never_exposes_raw_validator_details():
    raw="regex matched internal_field_name at stack frame 42"
    out=assemble_publication("# Analysis\nUsable content.",validation(("management_quote_fidelity",raw)),state())
    assert raw not in out["final_report"]
    assert "management_quote_fidelity" not in out["final_report"]
    assert all("check" not in x and "message" not in x for x in out["publication_warnings"])
    assert out["publication_warnings"][0]["section"]=="Management Commentary"


def test_low_confidence_sentiment_has_explicit_decision_grade_disclosure():
    out=assemble_publication("# Sentiment\nSample observations.",validation(("low_confidence_sentiment_scope","raw detail")),state(actionable=False))
    assert "Sentiment Reliability: LIMITED" in out["final_report"]
    assert "sample-limited" in out["final_report"]
    assert "Recommendation confidence: **AFFECTED**" in out["final_report"]
