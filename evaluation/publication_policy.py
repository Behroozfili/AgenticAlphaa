"""Graded publication assembly for deterministic Phase 7 findings.

Validators remain authoritative and unchanged.  This module determines what
validated content remains user-visible and never converts a failed check into a
pass.  It only scopes visibility, confidence, warnings, and recommendations.
"""
from __future__ import annotations

import re
from typing import Any

from evaluation.recommendation_gate import contains_actionable_recommendation


_SECTION_CHECKS = {
    "earnings_quality_consistency": {"financial health", "financial performance", "earnings quality"},
    "period_consistency": {"financial health", "financial performance", "research highlights"},
    "valuation_bridge_consistency": {"valuation", "dcf valuation", "scenario analysis"},
    "scenario_consistency": {"valuation", "dcf valuation", "scenario analysis"},
    "reverse_dcf_semantics": {"valuation", "dcf valuation", "sentiment", "market sentiment"},
    "repeated_metric_consistency": {"financial health", "financial performance", "research highlights"},
}

_WARNING_ONLY_CHECKS = {
    "arithmetic_consistency",
    "typed_claim_fidelity",
    "management_quote_fidelity",
    "sentiment_evidence_fidelity",
    "low_confidence_sentiment_scope",
}

_RELIABILITY_SECTIONS = {
    "Financial Performance": {
        "checks":{"earnings_quality_consistency":"WITHHELD","period_consistency":"WITHHELD","repeated_metric_consistency":"WITHHELD","arithmetic_consistency":"LIMITED"},
        "reason":"A material financial consistency, period, or arithmetic issue affects this section.",
        "supportable":"Financial observations outside the affected metrics remain usable when supported by validated structured data.",
        "recommendation":True,
    },
    "Valuation": {
        "checks":{"valuation_bridge_consistency":"WITHHELD","scenario_consistency":"WITHHELD","reverse_dcf_semantics":"WITHHELD","arithmetic_consistency":"LIMITED"},
        "reason":"A material inconsistency was detected in valuation inputs, scenarios, or valuation arithmetic.",
        "supportable":"No directional valuation conclusion is treated as decision-grade until the discrepancy is resolved.",
        "recommendation":True,
    },
    "Capital Allocation": {
        "checks":{"period_consistency":"WITHHELD","repeated_metric_consistency":"LIMITED"},
        "reason":"Period alignment or cash-flow consistency is insufficient for a fully reliable capital-allocation assessment.",
        "supportable":"Only individually validated cash-flow facts remain supportable; reinvestment-intensity inference may be withheld.",
        "recommendation":True,
    },
    "Sentiment": {
        "checks":{"sentiment_evidence_fidelity":"LIMITED","low_confidence_sentiment_scope":"LIMITED","reverse_dcf_semantics":"WITHHELD"},
        "reason":"Sentiment evidence coverage, confidence, or claim scope is below decision-grade reliability.",
        "supportable":"Only evidence-bounded, sample-limited sentiment observations remain supportable.",
        "recommendation":True,
    },
    "Management Commentary": {
        "checks":{"management_quote_fidelity":"LIMITED","typed_claim_fidelity":"LIMITED"},
        "reason":"Attribution or supporting evidence is insufficient for part of the management commentary.",
        "supportable":"Sourced paraphrases remain supportable; unsupported direct quotations are not treated as reliable.",
        "recommendation":False,
    },
    "Research Evidence": {
        "checks":{"typed_claim_fidelity":"LIMITED","period_consistency":"LIMITED","repeated_metric_consistency":"WITHHELD"},
        "reason":"Some research claims lack sufficient provenance, typing, or period alignment.",
        "supportable":"Validated evidence with explicit provenance remains supportable.",
        "recommendation":True,
    },
    "Recommendation": {
        "checks":{"recommendation_gate_consistency":"WITHHELD"},
        "reason":"The available confidence level does not permit actionable investment language.",
        "supportable":"The analysis may still describe evidence, scenarios, and uncertainty without an action recommendation.",
        "recommendation":True,
    },
}

_RELIABILITY_RANK={"RELIABLE":0,"LIMITED":1,"WITHHELD":2,"DATA_INSUFFICIENT":3}

_SUCCESSFUL_PUBLICATION_STATUSES = {
    "PUBLISHED", "PUBLISHED_WITH_WARNINGS", "BLOCKED_RECOMMENDATION",
}


def execution_status_from_publication(
    publication_status: str | None,
    *,
    validation_passed: bool | None = None,
    aborted: bool = False,
) -> str:
    """Map publication truth to the API's completed/failed execution status."""
    if aborted or validation_passed is False:
        return "failed"
    return (
        "completed"
        if str(publication_status or "") in _SUCCESSFUL_PUBLICATION_STATUSES
        else "failed"
    )


def _critical_findings(validation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        x for x in validation.get("critical_findings", [])
        if isinstance(x, dict)
        and not (
            x.get("check") in _WARNING_ONLY_CHECKS
            and str(x.get("severity") or "WARNING").upper() != "CRITICAL"
        )
    ]

def _findings(validation: dict[str, Any]) -> list[dict[str, Any]]:
    # Warning-only checks still affect section reliability/disclosure, but must
    # not be promoted back into publication blockers by the assembly layer.
    return [
        x for x in validation.get("critical_findings", []) if isinstance(x, dict)
    ] + [
        x for x in validation.get("warnings", []) if isinstance(x, dict)
    ]


def _data_insufficient(shared: dict[str, Any]) -> bool:
    fm=shared.get("financial_metrics_summary",{}) or {}; sm=shared.get("sentiment_analysis_summary",{}) or {}
    sanity=fm.get("financial_sanity",{}) or {}
    classifications={str(x.get("classification")) for x in sanity.get("findings",[]) if isinstance(x,dict)}
    critical="CRITICAL_MISSING" in classifications
    no_financial=not any(fm.get(k) for k in ("current_price","financial_facts","dcf_valuation","roe","net_margin","revenue_cagr"))
    no_research=not shared.get("aggregated_research_context")
    no_sentiment=sm.get("status") in {None,"NO_DATA"}
    return critical or (no_financial and no_research and no_sentiment)


def _availability(shared: dict[str, Any]) -> tuple[list[str],list[str]]:
    fm=shared.get("financial_metrics_summary",{}) or {}; sm=shared.get("sentiment_analysis_summary",{}) or {}
    available=[]; missing=[]
    domains={
        "structured financial metrics":bool(fm.get("financial_facts") or any(fm.get(k) for k in ("roe","net_margin","revenue_cagr"))),
        "current market price":fm.get("current_price") is not None,
        "structured valuation":bool(fm.get("dcf_valuation")),
        "research evidence":bool(shared.get("aggregated_research_context")),
        "sentiment evidence":sm.get("status") not in {None,"NO_DATA"},
    }
    for label,present in domains.items(): (available if present else missing).append(label)
    return available,missing


def _diagnostic_report(shared: dict[str, Any], title: str, reason: str) -> str:
    available,missing=_availability(shared)
    supportable=[]
    if "current market price" in available: supportable.append("The observed current market price may be reported without an investment recommendation.")
    if "structured financial metrics" in available: supportable.append("Validated structured financial facts may be reviewed individually with their provenance.")
    if not supportable: supportable.append("No directional investment conclusion is supportable from the available evidence.")
    bullets=lambda xs:"\n".join(f"- {x}" for x in xs) if xs else "- None identified"
    return f"# {title}\n\n{reason}\n\n## Data available\n{bullets(available)}\n\n## Data missing or unusable\n{bullets(missing)}\n\n## Conclusions still supportable\n{bullets(supportable)}\n\n## Recommendation\nNo actionable recommendation is provided."


def _remove_sections(report: str, withheld: set[str]) -> tuple[str,list[str]]:
    if not report.strip(): return "",[]
    heading=re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
    matches=list(heading.finditer(report)); removed=[]; spans=[]
    for i,m in enumerate(matches):
        name=re.sub(r"[*_`]","",m.group(2)).strip().lower()
        if any(token in name for token in withheld):
            spans.append((m.start(),matches[i+1].start() if i+1<len(matches) else len(report)))
            removed.append(m.group(2).strip())
    for start,end in reversed(spans): report=report[:start]+report[end:]
    return report.strip(),removed


def _suppress_actionable_language(report: str) -> str:
    pieces=re.split(r"(?<=[.!?])(?=\s|$)",report)
    kept=[x for x in pieces if not contains_actionable_recommendation(x)]
    return "".join(kept).strip()


def _section_reliability(findings: list[dict[str,Any]], *, recommendation_blocked: bool, data_insufficient: bool=False) -> list[dict[str,Any]]:
    checks={str(x.get("check")) for x in findings}
    rows=[]
    for section,policy in _RELIABILITY_SECTIONS.items():
        statuses=[status for check,status in policy["checks"].items() if check in checks]
        if section=="Recommendation" and recommendation_blocked: statuses.append("WITHHELD")
        status="DATA_INSUFFICIENT" if data_insufficient else max(statuses or ["RELIABLE"],key=_RELIABILITY_RANK.get)
        if status=="RELIABLE":
            reason="No material deterministic reliability issue was identified for this section."
            supportable="The section is suitable for use at the report's stated overall confidence level."
        elif status=="DATA_INSUFFICIENT":
            reason="Available data is insufficient to establish section-level reliability."
            supportable="No decision-grade conclusion is supportable for this section."
        else:
            reason=policy["reason"]; supportable=policy["supportable"]
        rows.append({"section":section,"status":status,"reason":reason,"supportable_conclusion":supportable,"affects_recommendation_confidence":bool(policy["recommendation"] and status!="RELIABLE")})
    return rows


def _append_disclosure(report: str, *, status: str, reliability: list[dict[str,Any]]) -> str:
    lines=["## Section Reliability",f"Overall publication status: **{status}**."]
    for row in reliability:
        lines.extend([
            f"### {row['section']} Reliability: {row['status']}",
            row["reason"],
            f"Still supportable: {row['supportable_conclusion']}",
            "Recommendation confidence: **AFFECTED**." if row["affects_recommendation_confidence"] else "Recommendation confidence: **NOT DIRECTLY AFFECTED**.",
        ])
    return (report.strip()+"\n\n"+"\n".join(lines)).strip()


def _public_warnings(reliability: list[dict[str,Any]]) -> list[dict[str,Any]]:
    return [{"section":x["section"],"status":x["status"],"reason":x["reason"]} for x in reliability if x["status"]!="RELIABLE"]


def assemble_publication(candidate: str, validation: dict[str,Any], shared: dict[str,Any]) -> dict[str,Any]:
    """Return a non-empty user report and graded publication metadata."""
    findings=_findings(validation); critical_findings=_critical_findings(validation); gate=shared.get("recommendation_gate",{}) or {}
    if _data_insufficient(shared):
        report=_diagnostic_report(shared,"Data insufficient for full analysis","Critical or broadly insufficient inputs prevent a reliable full report.")
        reliability=_section_reliability(findings,recommendation_blocked=True,data_insufficient=True)
        report=_append_disclosure(report,status="DATA_INSUFFICIENT",reliability=reliability)
        return {
            "final_report": report,
            "publication_status": "DATA_INSUFFICIENT",
            "publication_succeeded": False,
            "publication_warnings": _public_warnings(reliability),
            "withheld_sections": ["full analysis", "actionable recommendation"],
            "section_reliability": reliability,
            "confidence": "INSUFFICIENT",
        }

    section_checks={x.get("check") for x in critical_findings if x.get("check") in _SECTION_CHECKS}
    withheld_tokens=set().union(*(_SECTION_CHECKS[x] for x in section_checks)) if section_checks else set()
    report,removed=_remove_sections(candidate or "",withheld_tokens)
    recommendation_blocked=not gate.get("actionable_recommendation_allowed",False)
    if recommendation_blocked: report=_suppress_actionable_language(report)

    if critical_findings:
        status="LIMITED_REPORT"; confidence="LOW"
    elif recommendation_blocked:
        status="BLOCKED_RECOMMENDATION"; confidence=str(gate.get("confidence_level","LOW"))
    elif findings:
        status="PUBLISHED_WITH_WARNINGS"; confidence="REDUCED"
    else:
        status="PUBLISHED"; confidence=str(gate.get("confidence_level","HIGH"))

    if not report.strip():
        report=_diagnostic_report(shared,"Limited analysis report","All candidate sections affected by deterministic validation were withheld.")
        status="LIMITED_REPORT"; confidence="LOW"
    reliability=_section_reliability(findings,recommendation_blocked=recommendation_blocked)
    report=_append_disclosure(report,status=status,reliability=reliability)
    return {
        "final_report": report,
        "publication_status": status,
        "publication_succeeded": status in _SUCCESSFUL_PUBLICATION_STATUSES,
        "publication_warnings": _public_warnings(reliability),
        "withheld_sections": removed or sorted(withheld_tokens),
        "section_reliability": reliability,
        "confidence": confidence,
    }
