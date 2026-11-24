"""Deterministic fixed financial feature adapter for monthly forecasts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from core.forecast_contract import ForecastContext
from tools.financial_tools import (
    FINANCIAL_CONTRACT_VERSION, deterministic_financial_reference,
    financial_lifecycle_event, require_financial_context,
)

_MAX_SHARED_EVIDENCE_REFERENCES = 64


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _metric(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(payload.get(key))
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class MonthlyFinancialFeatures:
    valuation_gap: float | None
    dcf_long_term_value_per_share: float | None
    revenue_growth: float | None
    earnings_growth: float | None
    roe_pct: float | None
    net_margin_pct: float | None
    roic_pct: float | None
    net_debt_to_ebitda: float | None
    debt_ratio: float | None
    current_ratio: float | None
    quick_ratio: float | None
    fcf_margin_pct: float | None
    fcf_yield_pct: float | None
    ocf_ratio: float | None
    filing_age_days: int | None
    availability: Mapping[str, bool]
    missing_flags: tuple[str, ...]
    degraded_flags: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_reference_count: int
    accession_references: tuple[str, ...]
    dataset_version: str
    feature_version: str
    run_id: str

    def to_state_slot(self) -> dict[str, Any]:
        """Compact, typed, reference-only Shared State representation."""
        values = {
            name: getattr(self, name) for name in (
                "valuation_gap", "dcf_long_term_value_per_share", "revenue_growth",
                "earnings_growth", "roe_pct", "net_margin_pct", "roic_pct",
                "net_debt_to_ebitda", "debt_ratio", "current_ratio", "quick_ratio",
                "fcf_margin_pct", "fcf_yield_pct", "ocf_ratio", "filing_age_days",
            )
        }
        values.update({
            "availability": dict(self.availability),
            "accession_references": list(self.accession_references),
            "dataset_version": self.dataset_version, "run_id": self.run_id,
            "evidence_reference_count": self.evidence_reference_count,
            "evidence_references_truncated": self.evidence_reference_count > len(self.evidence_ids),
            "dcf_semantics": "LONG_TERM_VALUATION_FEATURE_NOT_21_DAY_FORECAST",
        })
        return {
            "version": self.feature_version, "values": values,
            "evidence_ids": list(self.evidence_ids),
            "missing": bool(self.missing_flags),
            "degraded": bool(self.degraded_flags),
            "errors": [], "missing_flags": list(self.missing_flags),
            "degraded_flags": list(self.degraded_flags),
        }


def build_monthly_financial_features(
    *, raw: Mapping[str, Any], calculated: Mapping[str, Any],
    forecast_context: ForecastContext | Mapping[str, Any] | None,
    run_id: str | None,
) -> MonthlyFinancialFeatures:
    """Build one fixed feature vector; analysis intents cannot change its shape."""
    started_at = time.perf_counter()
    context, identity = require_financial_context(forecast_context, run_id)
    yahoo = raw.get("yahoo_ratios") if isinstance(raw.get("yahoo_ratios"), Mapping) else {}
    growth = raw.get("revenue_growth") if isinstance(raw.get("revenue_growth"), Mapping) else {}
    xbrl = raw.get("xbrl_financials") if isinstance(raw.get("xbrl_financials"), Mapping) else {}

    dcf = calculated.get("dcf") if isinstance(calculated.get("dcf"), Mapping) else {}
    dcf_base = dcf.get("base") if isinstance(dcf.get("base"), Mapping) else dcf
    long_term_value = _metric(dcf_base, "intrinsic_value_per_share", "value_per_share")
    current_price = _number(yahoo.get("current_price"))
    valuation_gap = (
        long_term_value / current_price - 1.0
        if long_term_value is not None and current_price is not None and current_price > 0
        else None
    )

    annual_revenue = growth.get("annual_revenue") if isinstance(growth.get("annual_revenue"), list) else []
    annual_income = growth.get("annual_net_income") if isinstance(growth.get("annual_net_income"), list) else []
    revenue_growth = _number(growth.get("revenue_growth_ttm"))
    if revenue_growth is None and annual_revenue:
        revenue_growth = _number(annual_revenue[0].get("yoy_growth")) if isinstance(annual_revenue[0], Mapping) else None
    earnings_growth = _number(annual_income[0].get("yoy_growth")) if annual_income and isinstance(annual_income[0], Mapping) else None

    fields = {
        "valuation_gap": valuation_gap,
        "growth": revenue_growth,
        "quality_profitability": _metric(calculated.get("roe", {}), "roe_pct") if isinstance(calculated.get("roe"), Mapping) else None,
        "leverage": _metric(calculated.get("net_debt_to_ebitda", {}), "net_debt_to_ebitda") if isinstance(calculated.get("net_debt_to_ebitda"), Mapping) else None,
        "liquidity": _metric(calculated.get("current_ratio_calc", {}), "current_ratio") if isinstance(calculated.get("current_ratio_calc"), Mapping) else _number(yahoo.get("current_ratio")),
        "cash_flow_quality": _metric(calculated.get("fcf_margin", {}), "fcf_margin_pct") if isinstance(calculated.get("fcf_margin"), Mapping) else None,
    }

    evidence_ids: set[str] = set()
    accessions: set[str] = set()
    filed_dates: list[date] = []
    for series in xbrl.values():
        if not isinstance(series, list):
            continue
        for fact in series:
            if not isinstance(fact, Mapping):
                continue
            accession = str(fact.get("accession_number") or "").strip()
            filed_at = str(fact.get("filed_at") or "").strip()
            if accession:
                accessions.add(accession)
            if filed_at:
                try:
                    filed_dates.append(date.fromisoformat(filed_at[:10]))
                except ValueError:
                    pass
            evidence_id = str(fact.get("evidence_id") or "").strip()
            if not evidence_id and (accession or filed_at):
                evidence_id = deterministic_financial_reference(
                    provider="sec_companyfacts", ticker=context.ticker,
                    accession_number=accession or None,
                    metric=str(fact.get("xbrl_concept") or "") or None,
                    period_end=str(fact.get("period_end") or "") or None,
                    filed_at=filed_at or None, dataset_version=context.dataset_version,
                )
            if evidence_id:
                evidence_ids.add(evidence_id)

    filing_age = (
        (context.as_of.date() - max(filed_dates)).days  # type: ignore[union-attr]
        if filed_dates else None
    )
    availability = {key: value is not None for key, value in fields.items()}
    missing_flags = tuple(sorted(key for key, available in availability.items() if not available))
    degraded: set[str] = set(missing_flags)
    if not evidence_ids:
        degraded.add("missing_provenance")
    if any(isinstance(raw.get(key), Mapping) and raw[key].get("status") == "DATA_UNAVAILABLE" for key in ("yahoo_ratios", "revenue_growth", "balance_sheet_snapshot")):
        degraded.add("historical_current_snapshot_unavailable")

    def calc(name: str, *keys: str) -> float | None:
        payload = calculated.get(name)
        return _metric(payload, *keys) if isinstance(payload, Mapping) else None

    features = MonthlyFinancialFeatures(
        valuation_gap=valuation_gap, dcf_long_term_value_per_share=long_term_value,
        revenue_growth=revenue_growth, earnings_growth=earnings_growth,
        roe_pct=calc("roe", "roe_pct"), net_margin_pct=calc("net_margin", "net_margin_pct"),
        roic_pct=calc("roic", "roic_pct"),
        net_debt_to_ebitda=calc("net_debt_to_ebitda", "net_debt_to_ebitda"),
        debt_ratio=calc("debt_ratio", "debt_ratio"),
        current_ratio=fields["liquidity"], quick_ratio=calc("quick_ratio", "quick_ratio"),
        fcf_margin_pct=calc("fcf_margin", "fcf_margin_pct"),
        fcf_yield_pct=calc("fcf_yield", "fcf_yield_pct"),
        ocf_ratio=calc("ocf_ratio", "ocf_ratio"), filing_age_days=filing_age,
        availability=availability, missing_flags=missing_flags,
        degraded_flags=tuple(sorted(degraded)),
        evidence_ids=tuple(sorted(evidence_ids))[:_MAX_SHARED_EVIDENCE_REFERENCES],
        evidence_reference_count=len(evidence_ids),
        accession_references=tuple(sorted(accessions)),
        dataset_version=context.dataset_version,
        feature_version=FINANCIAL_CONTRACT_VERSION, run_id=identity,
    )
    financial_lifecycle_event(
        context=context, run_id=identity, boundary="monthly_financial_adapter",
        started_at=started_at, outcome="degraded" if features.degraded_flags else "success",
        counts={
            "result_count": len(fields), "evidence_count": len(features.evidence_ids),
            "accession_count": len(features.accession_references),
            "missing_count": len(features.missing_flags),
        },
        missing=bool(features.missing_flags), degraded=bool(features.degraded_flags),
    )
    return features


__all__ = ["MonthlyFinancialFeatures", "build_monthly_financial_features"]
