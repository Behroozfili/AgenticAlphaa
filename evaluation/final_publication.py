"""Fail-closed deterministic validation of the Manager's final narrative."""
from __future__ import annotations
import re
from typing import Any
from evaluation.claim_contract import validate_claim
from evaluation.recommendation_gate import contains_actionable_recommendation
from evaluation.validate_period_consistency import check_narration_vs_period

def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\$?(-?\d[\d,]*(?:\.\d+)?)", text)]

def _as_float(value: Any) -> float | None:
    try: return float(value)
    except (TypeError, ValueError): return None

def _structured_value(fm: dict, field: str, value_key: str) -> float | None:
    raw=fm.get(field)
    if isinstance(raw,dict): return _as_float(raw.get(value_key))
    return _as_float(raw)

def _money_value(raw: str, scale: str | None) -> float:
    value=float(raw.replace(",","")); unit=(scale or "").lower()
    return value * {"thousand":1e3,"million":1e6,"billion":1e9,"trillion":1e12}.get(unit,1.0)

def _materially_equal(actual: float, expected: float) -> bool:
    return abs(actual-expected) <= max(.02, abs(expected)*.005)

def _extract_metric_mentions(report: str) -> dict[str,list[dict[str,Any]]]:
    """Extract explicitly labelled material metrics without cross-paragraph guessing."""
    number=r"(-?\d[\d,]*(?:\.\d+)?)"; money=rf"\$\s*{number}\s*(thousand|million|billion|trillion)?"
    specs={
        "de_ratio":(r"(?:debt[-\s]*(?:to[-\s]*)?equity|D/E)(?:\s+ratio)?",number),
        "roe":(r"(?:return on equity|ROE)",number), "roic":(r"(?:return on invested capital|ROIC)",number),
        "roce":(r"(?:return on capital employed|ROCE)",number), "net_margin":(r"(?:net (?:profit )?margin)",number),
        "current_ratio":(r"current ratio",number), "cash":(r"(?:cash(?: and equivalents)?|cash and short[-\s]*term investments|cash position|cash balance)",money),
        "debt":(r"(?:total debt|debt balance)",money), "capex":(r"(?:capital expenditures?|capex)",money),
        "operating_cash_flow":(r"(?:operating cash flow|cash flow from operations|OCF)",money),
        "current_price":(r"current (?:market|share)?\s*price",money),
    }
    found={k:[] for k in specs}
    for name,(label,value) in specs.items():
        pattern=re.compile(rf"\b{label}\b(?:\s*\([^)]*\))?\s*(?:=|:|of|is|at|was|reached|stood at)\s*~?\s*{value}\s*(%)?",re.I)
        for m in pattern.finditer(report):
            numeric=m.group(1); scale=m.group(2) if name in {"cash","debt","capex","operating_cash_flow","current_price"} else None
            pct=m.group(3) if name in {"cash","debt","capex","operating_cash_flow","current_price"} else m.group(2)
            parsed=_money_value(numeric,scale) if name in {"cash","debt","capex","operating_cash_flow","current_price"} else float(numeric.replace(",",""))
            if name=="de_ratio" and pct: parsed/=100.0
            found[name].append({"value":parsed,"percentage":bool(pct),"matched_text":m.group(0),"source_section":_scenario_section(report,m.start())})
    return found

def _capital_value(fm: dict, name: str) -> float | None:
    raw=(fm.get("capital_allocation") or {}).get(name)
    if isinstance(raw,list): raw=raw[0] if raw else None
    if isinstance(raw,dict):
        for key in ("value","amount",name,"cash_flow","capex"):
            if raw.get(key) is not None: return _as_float(raw[key])
    value=_as_float(raw)
    if value is not None: return value
    if name in {"cash","debt","total_debt"}:
        bridge=((fm.get("dcf_valuation") or {}).get("base") or {}).get("equity_bridge",{})
        keys=("cash","cash_and_equivalents","marketable_securities") if name=="cash" else ("total_debt","debt")
        for key in keys:
            if bridge.get(key) is not None: return _as_float(bridge[key])
    return None

def _check_repeated_metrics(report: str, fm: dict) -> list[str]:
    mentions=_extract_metric_mentions(report); errors=[]
    expected={
        "de_ratio":_structured_value(fm,"de_ratio","de_ratio"), "roe":_structured_value(fm,"roe","roe_pct"),
        "roic":_structured_value(fm,"roic","roic_pct"), "roce":_structured_value(fm,"roce","roce_pct"),
        "net_margin":_structured_value(fm,"net_margin","net_margin_pct"), "current_ratio":_structured_value(fm,"current_ratio","current_ratio"),
        "cash":_capital_value(fm,"cash"), "debt":_capital_value(fm,"debt") or _capital_value(fm,"total_debt"),
        "capex":_capital_value(fm,"capex"), "operating_cash_flow":_capital_value(fm,"operating_cash_flow"),
        "current_price":_as_float(fm.get("current_price")),
    }
    labels={"de_ratio":"debt/equity","roe":"ROE","roic":"ROIC","roce":"ROCE","net_margin":"net margin","current_ratio":"current ratio","cash":"cash","debt":"debt","capex":"capex","operating_cash_flow":"operating cash flow","current_price":"current market price"}
    for name,items in mentions.items():
        target=expected.get(name)
        if target is None: continue
        bad=[x for x in items if not _materially_equal(x["value"],target)]
        if bad: errors.append(f"Repeated {labels[name]} value contradicts canonical structured value {target}: {bad[0]['matched_text']}.")
    return errors

def _normalize_quote(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+",text.lower()))

def _check_management_quotes(report: str, fm: dict) -> list[str]:
    claims=fm.get("typed_claims",fm.get("financial_facts",[])); errors=[]
    patterns=(r"(?:management|the company|executives?|[A-Z][a-z]+\s+(?:said|stated|noted|described))[^\n]{0,100}?[\"“]([^\"”]+)[\"”]",r"[\"“]([^\"”]+)[\"”][^\n]{0,100}?(?:management|the company|executives?\s+(?:said|stated|noted))")
    quotes=[m.group(1) for p in patterns for m in re.finditer(p,report,re.I)]
    for quote in quotes:
        normalized=_normalize_quote(quote)
        supported=False
        for claim in claims:
            if claim.get("claim_type")!="MANAGEMENT_STATEMENT": continue
            source=claim.get("source_url") or claim.get("accession_number")
            speaker=claim.get("speaker") or claim.get("attributable_entity")
            evidence=_normalize_quote(str(claim.get("evidence_span") or ""))
            if source and speaker and normalized and normalized in evidence: supported=True; break
        if not supported: errors.append(f'Quoted management statement lacks exact sourced speaker-backed evidence: "{quote}".')
    return errors

def _check_earnings(report: str, fm: dict) -> list[str]:
    eq = fm.get("earnings_quality", {})
    if not eq or eq.get("status") == "INSUFFICIENT_DATA": return []
    material = eq.get("one_off_flagged") is True
    if material and not all(term in report.lower() for term in ("gaap", "normalized")):
        return ["Material earnings-quality adjustment omitted: both GAAP and normalized margins are required."]
    if material:
        gaap, normalized = eq.get("gaap_net_margin_pct"), eq.get("normalized_net_margin_pct")
        def represented(value: Any) -> bool:
            return value is not None and bool(re.search(rf"(?<!\d){re.escape(str(value))}\s*%", report))
        if not represented(gaap) or (normalized is not None and not represented(normalized)):
            return ["Material earnings-quality margins are not numerically traceable to both structured GAAP and normalized outputs."]
    if eq.get("status") == "PARTIAL" and not any(x in report.lower() for x in ("partial", "uncertain", "unavailable")):
        return ["PARTIAL earnings-quality status omitted from narrative."]
    return []

def _period_of(item: Any) -> str | None:
    if isinstance(item, list): item = item[0] if item else {}
    if not isinstance(item, dict): return None
    return str(item.get("period_end") or item.get("_period") or "") or None

def _check_capital_allocation(report: str, fm: dict) -> list[str]:
    ca = fm.get("capital_allocation", {}); periods = {k: _period_of(v) for k, v in ca.items() if _period_of(v)}
    if len(set(periods.values())) > 1 and re.search(r"minimal reinvestment|limited .*reinvestment|buybacks?.*capex|operating cash flow.*capex", report, re.I|re.S):
        return [f"Capital-allocation comparison combines incompatible periods: {periods}."]
    return []

def _check_bridge(report: str, fm: dict) -> list[str]:
    dcf = fm.get("dcf_valuation", {}); scenarios = [v for k,v in dcf.items() if k in {"bear","base","bull"} and isinstance(v,dict)]
    has_per_share = any(v.get("equity_value_per_share") is not None or v.get("intrinsic_value_per_share") is not None for v in scenarios)
    if has_per_share and any(v.get("equity_bridge_complete") is not True for v in scenarios): return ["Per-share DCF published with incomplete equity bridge."]
    if has_per_share and re.search(r"does not (?:adjust|account) for (?:net debt|cash|debt)|no (?:net debt|equity bridge) adjustment", report, re.I): return ["Narrative contradicts the completed EV-to-equity bridge."]
    if re.search(r"net cash .*lower(?:s|ing)? (?:the )?wacc", report, re.I): return ["Narrative incorrectly assigns net cash to WACC rather than the equity bridge."]
    return []

def _scenario_section(report: str, offset: int) -> str:
    """Return the nearest Markdown heading for diagnostic extraction metadata."""
    headings = list(re.finditer(r"(?m)^#{1,6}\s+(.+?)\s*$", report[:offset]))
    return headings[-1].group(1).strip() if headings else "document"

def _extract_scenario_prices(report: str) -> dict[str, list[dict[str, Any]]]:
    """Extract only prices explicitly associated with a scenario.

    Range endpoints are associated by their named order.  Other matches require
    either a parenthesized scenario price or explicit price/value/valuation
    grammar; a nearby dollar amount alone is deliberately insufficient.
    """
    extracted: dict[str, list[dict[str, Any]]] = {name: [] for name in ("bear", "base", "bull")}
    occupied: set[tuple[str, int, int]] = set()

    def add(name: str, value: str, match: re.Match[str], source: str) -> None:
        key = (name, match.start(), match.end())
        if key in occupied:
            return
        occupied.add(key)
        extracted[name].append({
            "value": float(value.replace(",", "")),
            "source_section": _scenario_section(report, match.start()),
            "matched_text": match.group(0),
            "source": source,
        })

    number = r"(-?\d[\d,]*(?:\.\d+)?)"
    dash = r"(?:-|\u2013|\u2014|\u2212)"
    range_pattern = re.compile(
        rf"\b(bear|base|bull)\s*{dash}\s*to\s*{dash}?\s*(bear|base|bull)\b"
        rf"[^\n$]{{0,50}}\$\s*{number}\s*{dash}\s*\$?\s*{number}",
        re.I,
    )
    for match in range_pattern.finditer(report):
        first, second = match.group(1).lower(), match.group(2).lower()
        add(first, match.group(3), match, "named_range_start")
        add(second, match.group(4), match, "named_range_end")

    for name in ("bear", "base", "bull"):
        label = rf"\b{name}\s*-?\s*(?:case|scenario)\b"
        patterns = (
            ("parenthesized", re.compile(rf"{label}\s*\(\s*\$\s*{number}\s*(?:/\s*share|per\s+share)?\s*\)", re.I)),
            ("assigned", re.compile(rf"{label}\s*(?:price|value|valuation)?\s*(?:=|:|is|at)\s*\$\s*{number}(?:\s*(?:/\s*share|per\s+share))?", re.I)),
            ("valued", re.compile(rf"{label}[^\n.$]{{0,80}}\b(?:price|value|valuation)\s+(?:of|is|at)\s*\$\s*{number}(?:\s*(?:/\s*share|per\s+share))?", re.I)),
            ("bare_assigned", re.compile(rf"\b{name}\b\s*(?:=|:)\s*\$\s*{number}(?:\s*(?:/\s*share|per\s+share))?", re.I)),
            ("bare_metric", re.compile(rf"\b{name}\b\s+(?:price|value|valuation)\s*(?:=|:|is|of|at)\s*\$\s*{number}(?:\s*(?:/\s*share|per\s+share))?", re.I)),
        )
        for source, pattern in patterns:
            for match in pattern.finditer(report):
                add(name, match.group(1), match, source)
    return extracted

def _check_scenarios(report: str, fm: dict) -> list[str]:
    forecast = fm.get("forecast", {}).get("scenarios", {}); paths=[]
    for name in ("bear","base","bull"):
        rows=forecast.get(name,{}).get("yearly_forecast",[]); paths.append(tuple(r.get("revenue_growth_pct") for r in rows))
    differ = len(paths)==3 and all(paths) and len(set(paths))>1
    errors=[]
    if differ and re.search(r"(?:bear|base|bull).{0,120}(?:all|each|same).{0,60}8(?:\.0)?%|all (?:three )?scenarios.{0,80}8(?:\.0)?%", report, re.I|re.S):
        errors.append("Narrative replaces distinct explicit scenario paths with the same 8% growth assumption.")
    dcf=fm.get("dcf_valuation",{}); extracted=_extract_scenario_prices(report)
    for name in ("bear","base","bull"):
        structured=dcf.get(name,{})
        expected=structured.get("equity_value_per_share",structured.get("intrinsic_value_per_share"))
        if expected is None: continue
        cited=[item["value"] for item in extracted[name]]
        if any(abs(value-float(expected))>.02 for value in cited): errors.append(f"{name} scenario price contradicts structured valuation {expected}.")
        if len({round(value,2) for value in cited})>1: errors.append(f"{name} scenario price is inconsistent across report sections.")
    return errors

def _check_arithmetic(report: str, fm: dict) -> list[str]:
    errors=[]
    for m in re.finditer(r"\$?([\d,.]+)\s+is\s+([\d,.]+)%\s+(above|below)\s+\$?([\d,.]+)", report, re.I):
        a,pct,word,b= float(m[1].replace(',','')),float(m[2].replace(',','')),m[3].lower(),float(m[4].replace(',',''))
        expected=abs(a-b)/abs(b)*100 if b else 0; actual_word="above" if a>b else "below"
        if word!=actual_word or abs(pct-expected)>1.0: errors.append(f"Arithmetic contradiction: '{m.group(0)}'; expected about {expected:.1f}% {actual_word}.")
    for m in re.finditer(r"([\d,.]+)x?\s+(exceeds|is (?:greater|higher) than)\s+(?:a )?(?:peer average of )?([\d,.]+)x?", report, re.I):
        if float(m[1].replace(',','')) <= float(m[3].replace(',','')): errors.append(f"Numeric ordering contradiction: '{m.group(0)}'.")
    current=fm.get("current_price")
    if current and re.search(r"current (?:share )?price",report,re.I):
        cited=[float(x.replace(',','')) for x in re.findall(r"current (?:share )?price(?: of| is|:)?\s*\$([\d,.]+)",report,re.I)]
        if any(abs(x-current)>.02 for x in cited): errors.append("Current price is inconsistent with structured output.")
    return errors

def _check_claims(report: str, fm: dict) -> list[str]:
    claims=fm.get("typed_claims",fm.get("financial_facts",[])); errors=[]
    for claim in claims:
        errors.extend(validate_claim(claim))
    if re.search(r"management (?:believes?|views?|sees?) .{0,80}(?:undervalued|limited reinvestment)",report,re.I):
        supported=any(c.get("claim_type")=="MANAGEMENT_STATEMENT" and not validate_claim(c) for c in claims)
        if not supported: errors.append("Material management-intent claim lacks sourced speaker-backed MANAGEMENT_STATEMENT.")
    return errors

def _check_sentiment(report: str, sm: dict) -> list[str]:
    supported=set(sm.get("causal_aspects_supported",[])); errors=[]
    causal={"valuation":r"(?:market|investors?).{0,50}(?:priced in|valuation)","ai_demand":r"(?:market|investors?).{0,50}ai demand","competition":r"(?:market|investors?).{0,50}competition","geopolitical":r"(?:market|investors?).{0,50}(?:geopolitical|regulatory)"}
    for aspect,pattern in causal.items():
        if re.search(pattern,report,re.I) and aspect not in supported: errors.append(f"Causal sentiment claim for {aspect} lacks document-aspect-polarity evidence.")
    if sm.get("status")=="LOW_COVERAGE" and "low coverage" not in report.lower(): errors.append("LOW_COVERAGE sentiment is not disclosed.")
    return errors

def _check_low_confidence_market_claims(report: str, sm: dict) -> list[str]:
    confidence=_as_float(sm.get("confidence",sm.get("fear_greed_confidence")))
    weak=sm.get("status") in {"LOW_COVERAGE","NO_DATA"} or (confidence is not None and confidence < .20)
    if not weak: return []
    broad=re.compile(r"(?:\b(?:the )?market\b[^\n.!?]{0,100}\b(?:panic|euphoria|believes?|expects?|pricing|priced|sentiment)\b|\binvestors?\b[^\n.!?]{0,100}\b(?:pricing|priced|believe|expect|demand)\b)",re.I)
    qualifier=re.compile(r"(?:limited available evidence|within (?:the )?analy[sz]ed sample|weak signal|low confidence|low coverage|cannot support|does not support|may|might|suggests?|appears?|could)",re.I)
    errors=[]
    for match in broad.finditer(report):
        start=max(report.rfind(".",0,match.start())+1,report.rfind("\n",0,match.start())+1)
        ends=[x for x in (report.find(".",match.end()),report.find("\n",match.end())) if x>=0]
        end=min(ends) if ends else min(len(report),match.end()+180)
        sentence=report[start:end]
        if sentence.strip().startswith("#"):
            continue
        if not qualifier.search(sentence):
            errors.append(f"Low-confidence sentiment cannot support unqualified market-wide claim: '{sentence.strip()}'.")
    return errors

def validate_final_publication(report: str, shared: dict[str,Any]) -> dict[str,Any]:
    names=("earnings_quality_consistency","period_consistency","valuation_bridge_consistency","scenario_consistency","arithmetic_consistency","typed_claim_fidelity","sentiment_evidence_fidelity","recommendation_gate_consistency","repeated_metric_consistency","management_quote_fidelity","low_confidence_sentiment_scope")
    try:
        fm=shared.get("financial_metrics_summary",{}); sm=shared.get("sentiment_analysis_summary",{}); gate=shared.get("recommendation_gate",{})
        checks={
            names[0]:_check_earnings(report,fm), names[1]:_check_capital_allocation(report,fm)+[x.get("message",str(x)) for x in check_narration_vs_period(report,fm) if x.get("severity")=="HIGH"],
            names[2]:_check_bridge(report,fm), names[3]:_check_scenarios(report,fm), names[4]:_check_arithmetic(report,fm), names[5]:_check_claims(report,fm), names[6]:_check_sentiment(report,sm),
            names[7]:["Actionable recommendation violates recommendation-confidence gate."] if not gate.get("actionable_recommendation_allowed",False) and contains_actionable_recommendation(report) else [],
            names[8]:_check_repeated_metrics(report,fm), names[9]:_check_management_quotes(report,fm), names[10]:_check_low_confidence_market_claims(report,sm),
        }
        critical=[{"check":k,"message":msg} for k,items in checks.items() for msg in items]
        return {"passed":not critical,"critical_findings":critical,"warnings":[],"required_regeneration_context":critical,"publication_status":"PUBLISHED" if not critical else "REGENERATION_REQUIRED","checks":{k:{"passed":not v,"findings":v} for k,v in checks.items()}}
    except Exception as exc:
        finding={"check":"validator_exception","message":f"Final publication validation failed closed: {exc}"}
        return {"passed":False,"critical_findings":[finding],"warnings":[],"required_regeneration_context":[finding],"publication_status":"BLOCKED_PUBLICATION_VALIDATION_ERROR","checks":{}}
