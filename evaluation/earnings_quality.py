"""Deterministic earnings-quality normalization from same-period SEC facts."""

from __future__ import annotations

from typing import Any

MATERIALITY_THRESHOLD_PCT = 10.0


def _by_period(rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    return {str(row["period_end"]): row for row in (rows or []) if isinstance(row, dict) and row.get("period_end") and row.get("value") is not None}


def normalize_earnings_quality(xbrl: dict[str, Any], *, materiality_threshold_pct: float = MATERIALITY_THRESHOLD_PCT) -> dict[str, Any]:
    """Return GAAP and operating-normalized margins for the newest common FY."""
    names = ("revenue_annual", "net_income_annual", "operating_income_annual", "pretax_income_annual", "income_tax_annual", "other_income_annual", "unusual_items_annual")
    series = {name: _by_period(xbrl.get(name)) for name in names}
    required = names[:4]
    common = set.intersection(*(set(series[name]) for name in required))
    if not common:
        return {"status": "INSUFFICIENT_DATA", "one_off_flagged": False, "material_one_off_income_pct": None, "gaap_net_margin_pct": None, "normalized_net_margin_pct": None, "missing_required_series": [name for name in required if not series[name]], "methodology": "No normalization performed; required same-period SEC facts were unavailable."}

    period_end = max(common)
    revenue = float(series["revenue_annual"][period_end]["value"])
    net_income = float(series["net_income_annual"][period_end]["value"])
    operating_income = float(series["operating_income_annual"][period_end]["value"])
    pretax_income = float(series["pretax_income_annual"][period_end]["value"])
    tax_row = series["income_tax_annual"].get(period_end)
    tax = float(tax_row["value"]) if tax_row else None
    effective_tax_rate = tax / pretax_income if tax is not None and pretax_income != 0 else None
    bounded_tax_rate = min(max(effective_tax_rate, 0.0), 0.50) if effective_tax_rate is not None else None
    non_operating_contribution = pretax_income - operating_income
    material_pct = abs(non_operating_contribution) / abs(pretax_income) * 100.0 if pretax_income != 0 else None
    flagged = material_pct is not None and material_pct >= materiality_threshold_pct
    normalized_net_income = operating_income * (1.0 - bounded_tax_rate) if bounded_tax_rate is not None else None
    source = series["pretax_income_annual"][period_end]
    other_row = series["other_income_annual"].get(period_end)
    unusual_row = series["unusual_items_annual"].get(period_end)
    return {
        "status": "NORMALIZED" if normalized_net_income is not None else "PARTIAL",
        "period_end": period_end, "fiscal_year": source.get("fiscal_year"),
        "filing_type": source.get("filing_type"), "accession_number": source.get("accession_number"),
        "gaap_status": "GAAP_AND_NORMALIZED", "revenue": revenue,
        "gaap_net_income": net_income, "gaap_net_margin_pct": round(net_income / revenue * 100.0, 2) if revenue else None,
        "operating_income": operating_income, "pretax_income": pretax_income, "income_tax": tax,
        "effective_tax_rate_pct": round(effective_tax_rate * 100.0, 2) if effective_tax_rate is not None else None,
        "normalized_net_income": round(normalized_net_income, 2) if normalized_net_income is not None else None,
        "normalized_net_margin_pct": round(normalized_net_income / revenue * 100.0, 2) if normalized_net_income is not None and revenue else None,
        "non_operating_contribution": non_operating_contribution,
        "other_income_reported": float(other_row["value"]) if other_row else None,
        "unusual_items_reported": float(unusual_row["value"]) if unusual_row else None,
        "material_one_off_income_pct": round(material_pct, 2) if material_pct is not None else None,
        "materiality_threshold_pct": materiality_threshold_pct, "one_off_flagged": flagged,
        "quality_warning": "Material non-operating contribution; use both GAAP and normalized margins." if flagged else None,
        "methodology": "Normalized net income equals operating income after the observed effective tax rate; non-operating contribution is pretax income minus operating income. Explicit other/unusual tags are disclosed but not subtracted again.",
    }
