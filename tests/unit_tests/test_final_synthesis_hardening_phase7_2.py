from evaluation.canonical_synthesis import build_canonical_synthesis_payload
from evaluation.final_publication import _extract_metric_mentions, validate_final_publication


def shared(*, de_ratio=0.66, sentiment=None, claims=None):
    return {
        "financial_metrics_summary": {
            "de_ratio": {"de_ratio": de_ratio},
            "typed_claims": claims or [],
            "financial_sanity": {"passed": True, "actionable_valuation_allowed": True, "confidence_cap": "HIGH"},
        },
        "sentiment_analysis_summary": sentiment or {"status": "VALID", "confidence": .8, "causal_aspects_supported": []},
        "recommendation_gate": {"actionable_recommendation_allowed": True},
    }


def findings(result, check):
    return [x["message"] for x in result["critical_findings"] if x["check"] == check]


def test_conflicting_repeated_de_ratio_fails():
    report="## Financial Health\nD/E = 0.66.\n## Risk\nDebt-to-equity = 10.8%."
    result=validate_final_publication(report,shared(de_ratio=.66))
    assert findings(result,"repeated_metric_consistency")


def test_equivalent_decimal_and_percentage_de_ratio_pass():
    report="D/E = 0.108.\n## Risk\nDebt-to-equity = 10.8%."
    result=validate_final_publication(report,shared(de_ratio=.108))
    assert not findings(result,"repeated_metric_consistency")


def test_identical_repeated_material_metric_values_pass():
    report="D/E = 0.66.\n## Risk\nDebt-to-equity ratio is 0.66."
    extracted=_extract_metric_mentions(report)
    assert [x["value"] for x in extracted["de_ratio"]] == [.66,.66]
    assert not findings(validate_final_publication(report,shared()),"repeated_metric_consistency")


def management_claim(*, evidence_span, text="Management discussed profitability"):
    return {"claim_type":"MANAGEMENT_STATEMENT","text":text,"source_url":"https://example.com/filing","speaker":"Tesla management","evidence_span":evidence_span}


def test_management_quote_without_matching_evidence_fails():
    report='Management stated it was "intentionally suppressing profitability during this phase."'
    result=validate_final_publication(report,shared(claims=[management_claim(evidence_span="We are investing for long-term growth.")]))
    assert findings(result,"management_quote_fidelity")


def test_supported_management_paraphrase_passes():
    report="Management stated that near-term investment is weighing on profitability."
    claim=management_claim(evidence_span="Near-term investment is weighing on profitability.")
    assert not findings(validate_final_publication(report,shared(claims=[claim])),"management_quote_fidelity")


def test_exact_sourced_management_quote_with_evidence_passes():
    quote="intentionally suppressing profitability during this phase"
    report=f'Management stated it was "{quote}."'
    claim=management_claim(evidence_span=f"We are {quote}.",text=f'Management stated "{quote}."')
    assert not findings(validate_final_publication(report,shared(claims=[claim])),"management_quote_fidelity")


def test_low_confidence_unqualified_pricing_in_claim_fails():
    sentiment={"status":"VALID","confidence":.0573,"causal_aspects_supported":[]}
    result=validate_final_publication("Investors are pricing in a prove-it scenario.",shared(sentiment=sentiment))
    assert findings(result,"low_confidence_sentiment_scope")


def test_low_confidence_sample_limited_wording_passes():
    sentiment={"status":"VALID","confidence":.0573,"causal_aspects_supported":[]}
    report="Within the analyzed sample, the weak signal may be consistent with caution; it does not support a high-confidence market-wide conclusion."
    assert not findings(validate_final_publication(report,shared(sentiment=sentiment)),"low_confidence_sentiment_scope")


def test_high_confidence_diverse_supported_aspect_claim_passes():
    sentiment={"status":"VALID","confidence":.9,"evidence_diversity":{"source_count":5,"document_count":8},"causal_aspects_supported":["valuation"]}
    report="The analyzed evidence suggests investors are pricing in valuation strength."
    result=validate_final_publication(report,shared(sentiment=sentiment))
    assert not findings(result,"low_confidence_sentiment_scope")
    assert not findings(result,"sentiment_evidence_fidelity")


def test_canonical_context_restricts_weak_sentiment_and_unsupported_quotes():
    s=shared(sentiment={"status":"LOW_COVERAGE","confidence":.0573,"aspect_observations":[]},claims=[{"claim_type":"MANAGEMENT_STATEMENT","text":"Unsupported quote","source_url":"https://example.com","speaker":"Management"}])
    payload=build_canonical_synthesis_payload(s)
    assert payload["sentiment"]["narrative_policy"]["market_wide_claims_allowed"] is False
    assert payload["sentiment"]["narrative_policy"]["pricing_in_claims_allowed"] is False
    assert payload["claims"] == []


def test_evidence_backed_but_unattributed_management_claim_downgrades_to_inference():
    claim={"claim_type":"MANAGEMENT_STATEMENT","text":"Profitability is temporarily constrained","evidence_span":"Profitability is temporarily constrained by investment."}
    payload=build_canonical_synthesis_payload(shared(claims=[claim]))
    assert payload["claims"][0]["claim_type"] == "INFERENCE"
    assert payload["claims"][0]["direct_quote_allowed"] is False
