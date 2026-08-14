"""Build the single authoritative input contract for Manager final synthesis.

This module performs no analysis or calculation.  It selects, labels, and
normalizes already-produced Phase 3-6 outputs so legacy prose cannot override
validated structured data in the finaliser context.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


SOURCE_PRECEDENCE = (
    "validated_structured_output",
    "typed_structured_claim",
    "validated_research_evidence",
    "legacy_summary_fallback",
    "llm_inference",
)

_METRICS = {
    "pe_ratio": "pe_ratio",
    "roe": "roe_pct",
    "net_margin": "net_margin_pct",
    "de_ratio": "de_ratio",
    "revenue_cagr": "cagr_pct",
    "net_debt_to_ebitda": "net_debt_to_ebitda",
    "fcf_margin": "fcf_margin_pct",
    "fcf_yield": "fcf_yield_pct",
    "ocf_ratio": "ocf_ratio",
    "roic": "roic_pct",
    "roce": "roce_pct",
    "debt_ratio": "debt_ratio",
    "equity_ratio": "equity_ratio",
}


def _period_label(item: dict[str, Any], fact: dict[str, Any] | None = None) -> str | None:
    fact = fact or {}
    explicit = item.get("_period") or item.get("period_label")
    if explicit:
        return str(explicit)
    year = item.get("fiscal_year", fact.get("fiscal_year"))
    quarter = item.get("fiscal_quarter", fact.get("fiscal_quarter"))
    scope = item.get("metric_scope", fact.get("metric_scope"))
    if quarter and year:
        return f"quarterly:{year}-{quarter}"
    if year:
        return f"annual:{year}"
    return str(scope) if scope else None


def _metric_fact(facts: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((x for x in facts if str(x.get("metric", "")).lower() == name.lower()), {})


def _financial_metrics(fm: dict[str, Any]) -> dict[str, Any]:
    facts = [x for x in fm.get("financial_facts", []) if isinstance(x, dict)]
    result: dict[str, Any] = {}
    for name, value_key in _METRICS.items():
        raw = fm.get(name)
        item = raw if isinstance(raw, dict) else {value_key: raw}
        fact = _metric_fact(facts, name)
        value = item.get(value_key)
        if value is None:
            value = fact.get("value")
        if value is None:
            continue
        result[name] = {
            "value": value,
            "period_label": _period_label(item, fact),
            "period_start": item.get("period_start", fact.get("period_start")),
            "period_end": item.get("period_end", fact.get("period_end")),
            "scope": item.get("metric_scope", fact.get("metric_scope")),
            "source": item.get("source_url", fact.get("source_url")),
            "authority": "validated_structured_output",
        }
    return result


def _canonical_scenarios(fm: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    forecast = fm.get("forecast", {}) if isinstance(fm.get("forecast"), dict) else {}
    paths = forecast.get("scenarios", {}) if isinstance(forecast.get("scenarios"), dict) else {}
    dcf = fm.get("dcf_valuation", {}) if isinstance(fm.get("dcf_valuation"), dict) else {}
    result: dict[str, Any] = {}
    for name in ("bear", "base", "bull"):
        path = deepcopy(paths.get(name, {})) if isinstance(paths.get(name), dict) else {}
        valuation = deepcopy(dcf.get(name, {})) if isinstance(dcf.get(name), dict) else {}
        equity = valuation.get("equity_value_per_share")
        intrinsic = valuation.get("intrinsic_value_per_share")
        if equity is not None and intrinsic is not None and float(equity) != float(intrinsic):
            conflicts.append({
                "domain": "valuation.scenarios",
                "scenario": name,
                "fields": {"equity_value_per_share": equity, "intrinsic_value_per_share": intrinsic},
                "action": "PRESERVE_FOR_PHASE_7_BLOCKING",
            })
        price = equity if equity is not None else intrinsic
        result[name] = {
            "yearly_forecast": deepcopy(path.get("yearly_forecast", [])),
            "revenue_path": deepcopy(path.get("revenue_by_year", [])),
            "fcf_path": deepcopy(path.get("fcf_by_year", [])),
            "wacc_pct": valuation.get("wacc_pct"),
            "terminal_growth_rate_pct": (valuation.get("assumptions") or {}).get("terminal_growth_rate_pct"),
            "enterprise_value": valuation.get("enterprise_value"),
            "equity_value": valuation.get("equity_value"),
            "equity_value_per_share": price,
            "equity_bridge": deepcopy(valuation.get("equity_bridge", {})),
            "equity_bridge_complete": valuation.get("equity_bridge_complete"),
            "wacc_components": deepcopy(valuation.get("wacc_components", {})),
            "authority": "validated_structured_output",
        }
    return result


def _canonical_claims(fm: dict[str, Any]) -> list[dict[str, Any]]:
    claims = fm.get("typed_claims") or fm.get("financial_facts") or []
    rules = {
        "FACT": "state as sourced fact",
        "MANAGEMENT_STATEMENT": "attribute explicitly to management/speaker",
        "ANALYST_ESTIMATE": "attribute explicitly to analyst/source",
        "INFERENCE": "use uncertainty language such as may/suggests; never state as fact",
        "ASSUMPTION": "state explicitly as a model assumption",
        "MODEL_OUTPUT": "attribute explicitly to the model",
    }
    result = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        row = deepcopy(claim)
        kind = str(row.get("claim_type") or "FACT").upper()
        if kind == "MANAGEMENT_STATEMENT":
            evidence = str(row.get("evidence_span") or "").strip()
            source = row.get("source_url") or row.get("accession_number")
            speaker = row.get("speaker") or row.get("attributable_entity")
            if not evidence:
                continue
            if not (source and speaker):
                kind = "INFERENCE"
                row["claim_type"] = kind
                row["direct_quote_allowed"] = False
                row["downgrade_reason"] = "management attribution lacks source or attributable speaker/entity"
            else:
                quoted = [x or y for x,y in re.findall(r'["“]([^"”]+)["”]|\'([^\']+)\'',str(row.get("text") or ""))]
                normalized_evidence = " ".join(re.findall(r"[a-z0-9]+",evidence.lower()))
                row["direct_quote_allowed"] = bool(quoted) and all(
                    " ".join(re.findall(r"[a-z0-9]+",q.lower())) in normalized_evidence for q in quoted
                )
        row["claim_type"] = kind
        row["synthesis_wording_rule"] = rules.get(kind, "do not present as sourced fact")
        row["authority"] = "typed_structured_claim"
        result.append(row)
    return result


def _canonical_sentiment(sm: dict[str, Any]) -> dict[str, Any]:
    observations = sm.get("aspect_observations", [])
    supported = [
        deepcopy(row) for row in observations
        if isinstance(row, dict) and row.get("aspect") and row.get("polarity")
        and row.get("evidence_span") and (row.get("document_url") or row.get("document_title"))
    ]
    structured_available = bool(sm.get("status") or observations or sm.get("causal_aspects_supported"))
    confidence=sm.get("confidence", sm.get("fear_greed_confidence"))
    try: low_confidence=confidence is not None and float(confidence) < .20
    except (TypeError,ValueError): low_confidence=True
    weak=sm.get("status") in {"LOW_COVERAGE","NO_DATA"} or low_confidence
    result = {
        "status": sm.get("status", "NO_DATA"),
        "confidence": confidence,
        "coverage": sm.get("coverage", sm.get("total_chunks_analyzed")),
        "evidence_diversity": deepcopy(sm.get("evidence_diversity", {})),
        "aspects": supported,
        "causal_aspects_supported": sorted({row["aspect"] for row in supported}),
        "aggregate_signals": {
            "fear_greed_score": sm.get("fear_greed_score"),
            "fear_greed_label": sm.get("fear_greed_label"),
            "overall_sentiment": sm.get("overall_sentiment"),
            "model_agreement": sm.get("model_agreement"),
        },
        "authority": "validated_structured_output" if structured_available else "legacy_summary_fallback",
        "narrative_policy": {
            "market_wide_claims_allowed": not weak,
            "pricing_in_claims_allowed": not weak and bool(supported),
            "required_scope": "sample_limited_and_uncertainty_qualified" if weak else "evidence_bounded",
            "allowed_examples": [
                "The limited available evidence is consistent with...",
                "Within the analyzed sample...",
                "This weak signal does not support a high-confidence market-wide conclusion.",
            ] if weak else [],
            "forbidden_when_weak": ["unqualified market-state assertions", "investors are pricing in statements presented as fact"],
        },
    }
    if not structured_available and sm.get("narrative"):
        result["legacy_narrative_fallback"] = sm["narrative"]
    return result


def _recommendation_policy(gate: dict[str, Any], sentiment_status: str) -> dict[str, Any]:
    level = str(gate.get("confidence_level", "LOW")).upper()
    allowed = bool(gate.get("actionable_recommendation_allowed"))
    if sentiment_status != "VALID":
        language_class = "BLOCKED" if gate.get("recommendation_mode") == "SCENARIO_SUMMARY_ONLY" else "LOW"
        allowed = False
    elif level == "MEDIUM":
        language_class = "MEDIUM"
        allowed = False
    elif level == "HIGH" and allowed:
        language_class = "HIGH"
    else:
        language_class = "BLOCKED" if gate.get("recommendation_mode") == "SCENARIO_SUMMARY_ONLY" else "LOW"
        allowed = False
    return {
        **deepcopy(gate),
        "allowed_language_class": language_class,
        "actionable_recommendation_allowed": allowed,
        "wording_constraint": {
            "HIGH": "directional/actionable language permitted",
            "MEDIUM": "qualified conditional directional language only",
            "LOW": "observational and uncertainty-aware language only",
            "BLOCKED": "no actionable recommendation; scenario summary only",
        }[language_class],
        "authority": "validated_structured_output",
    }


def build_canonical_synthesis_payload(
    state: dict[str, Any], *, research_evidence: str | list[Any] | None = None,
) -> dict[str, Any]:
    """Return the sole factual context supplied to the Manager finaliser."""
    fm = state.get("financial_metrics_summary", {}) or {}
    sm = state.get("sentiment_analysis_summary", {}) or {}
    gate = state.get("recommendation_gate", {}) or {}
    conflicts: list[dict[str, Any]] = []
    scenarios = _canonical_scenarios(fm, conflicts)
    structured_scenarios_exist = any(x.get("yearly_forecast") for x in scenarios.values())
    structured_dcf_exists = any(x.get("equity_value_per_share") is not None for x in scenarios.values())
    sentiment = _canonical_sentiment(sm)
    return {
        "schema_version": "phase-7.1",
        "source_precedence": list(SOURCE_PRECEDENCE),
        "task": {"query": state.get("task_query"), "ticker": fm.get("ticker") or state.get("manager_directives", {}).get("ticker")},
        "financial": {
            "company": {k: fm.get(k) for k in ("ticker", "company_name", "sector", "industry")},
            "market": {k: fm.get(k) for k in ("current_price", "market_cap", "beta", "52w_high", "52w_low")},
            "performance_metrics": _financial_metrics(fm),
            "earnings_quality": deepcopy(fm.get("earnings_quality", {})),
            "period_validation": deepcopy(fm.get("period_validation", state.get("qa_period_findings", []))),
            "financial_sanity": deepcopy(fm.get("financial_sanity", {})),
        },
        "valuation": {
            "current_price": fm.get("current_price"),
            "scenarios": scenarios,
            "forecast_assumptions": deepcopy((fm.get("forecast") or {}).get("assumptions", {})),
            "reverse_dcf": deepcopy(fm.get("reverse_dcf", {})),
            "multi_factor_monte_carlo": deepcopy(fm.get("dcf_monte_carlo", {})),
        },
        "claims": _canonical_claims(fm),
        "sentiment": sentiment,
        "recommendation_policy": _recommendation_policy(gate, str(sentiment["status"])),
        "research": {"validated_evidence": research_evidence or [], "authority": "validated_research_evidence"},
        "data_quality": {
            "financial": deepcopy(fm.get("data_quality", {})),
            "canonical_conflicts": conflicts,
            "conflict_policy": "Do not guess or reconcile; preserve for Phase 7 validation.",
        },
        "legacy_sources": {
            "excluded_when_structured_available": [
                *( ["legacy DCF summaries", "scalar scenario-growth summaries"] if structured_dcf_exists or structured_scenarios_exist else []),
                *( ["pre-Phase-4/6 sentiment narrative"] if sentiment["authority"] == "validated_structured_output" else []),
                "duplicated free-text financial summaries",
            ],
            "policy": "fallback only when the corresponding structured domain is genuinely unavailable",
        },
    }
