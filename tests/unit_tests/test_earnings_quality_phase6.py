from evaluation.earnings_quality import normalize_earnings_quality
from evaluation.financial_sanity import evaluate_financial_sanity


def row(value, period="2025-12-31"):
    return {"value": value, "period_end": period, "fiscal_year": 2025,
            "filing_type": "10-K", "accession_number": "0001"}


def payload(**overrides):
    data = {
        "revenue_annual": [row(1_000)], "net_income_annual": [row(180)],
        "operating_income_annual": [row(150)], "pretax_income_annual": [row(200)],
        "income_tax_annual": [row(20)], "other_income_annual": [row(50)],
        "unusual_items_annual": [],
    }
    data.update(overrides)
    return data


def test_material_non_operating_income_is_flagged_and_both_margins_reported():
    result = normalize_earnings_quality(payload())
    assert result["status"] == "NORMALIZED"
    assert result["material_one_off_income_pct"] == 25.0
    assert result["one_off_flagged"] is True
    assert result["gaap_net_margin_pct"] == 18.0
    assert result["normalized_net_margin_pct"] == 13.5
    assert result["other_income_reported"] == 50.0


def test_immaterial_non_operating_contribution_is_not_flagged():
    result = normalize_earnings_quality(payload(
        operating_income_annual=[row(195)], pretax_income_annual=[row(200)]
    ))
    assert result["material_one_off_income_pct"] == 2.5
    assert result["one_off_flagged"] is False


def test_missing_tax_is_not_silently_converted_to_zero():
    result = normalize_earnings_quality(payload(income_tax_annual=[]))
    assert result["status"] == "PARTIAL"
    assert result["income_tax"] is None
    assert result["normalized_net_income"] is None
    assert result["normalized_net_margin_pct"] is None


def test_mismatched_periods_do_not_get_merged():
    result = normalize_earnings_quality(payload(
        pretax_income_annual=[row(200, "2024-12-31")]
    ))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["normalized_net_margin_pct"] is None


def test_flagged_material_item_passes_existing_hard_rule():
    quality = normalize_earnings_quality(payload())
    result = evaluate_financial_sanity({"earnings_quality": quality})
    assert all(x["rule"] != "UNFLAGGED_ONE_OFF_INCOME" for x in result["critical_failures"])
