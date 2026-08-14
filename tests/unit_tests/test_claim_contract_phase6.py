from evaluation.claim_contract import ClaimType, typed_financial_claim, validate_claim


def test_financial_fact_is_typed_fact_with_provenance():
    claim = typed_financial_claim({"metric": "revenue", "value": 10, "unit": "USD", "accession_number": "a1"})
    assert claim["claim_type"] == "FACT"
    assert validate_claim(claim) == []


def test_management_intent_requires_source_and_speaker():
    claim = {"claim_type": ClaimType.MANAGEMENT_STATEMENT.value, "text": "Management sees shares as undervalued"}
    assert set(validate_claim(claim)) == {"MANAGEMENT_STATEMENT requires a source", "MANAGEMENT_STATEMENT requires speaker"}


def test_inference_is_allowed_without_false_source_attribution():
    assert validate_claim({"claim_type": "INFERENCE", "text": "Buybacks may offset dilution"}) == []


def test_assumption_requires_derivation():
    assert validate_claim({"claim_type": "ASSUMPTION", "text": "Tax rate is 21%"}) == ["ASSUMPTION requires derivation"]
