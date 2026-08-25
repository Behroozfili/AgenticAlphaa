from unittest.mock import AsyncMock, MagicMock
import pytest
from agents.manager_agent import ManagerAgent
from evaluation.final_publication import _extract_scenario_prices, validate_final_publication

def base_state():
    return {"financial_metrics_summary": {"current_price": 225.75, "financial_sanity": {"passed": True, "actionable_valuation_allowed": True, "confidence_cap": "HIGH"}, "financial_facts": []}, "sentiment_analysis_summary": {"status": "VALID", "causal_aspects_supported": []}, "recommendation_gate": {"actionable_recommendation_allowed": True}}

def messages(result): return {x["message"] for x in result["critical_findings"]}

def test_material_earnings_quality_omission_fails():
    s=base_state(); s["financial_metrics_summary"]["earnings_quality"]={"status":"NORMALIZED","one_off_flagged":True,"gaap_net_margin_pct":20,"normalized_net_margin_pct":14}
    assert not validate_final_publication("GAAP margin was 20%.",s)["passed"]

def test_stale_capex_comparison_fails():
    s=base_state(); s["financial_metrics_summary"]["capital_allocation"]={"operating_cash_flow":[{"period_end":"2026-01-31"}],"capex":[{"period_end":"2011-01-31"}]}
    assert not validate_final_publication("Operating cash flow versus capex implies minimal reinvestment.",s)["passed"]

def test_equity_bridge_narrative_contradiction_fails():
    s=base_state(); s["financial_metrics_summary"]["dcf_valuation"]={"base":{"equity_value_per_share":89,"equity_bridge_complete":True}}
    assert not validate_final_publication("DCF is $89 per share but does not adjust for net debt.",s)["passed"]

def test_distinct_scenarios_cannot_be_called_same_eight_percent():
    s=base_state(); s["financial_metrics_summary"]["forecast"]={"scenarios":{k:{"yearly_forecast":[{"revenue_growth_pct":v}]} for k,v in {"bear":4,"base":8,"bull":12}.items()}}
    assert not validate_final_publication("All three scenarios use the same 8% growth.",s)["passed"]

def test_bad_below_percentage_fails():
    assert not validate_final_publication("$89 is 154% below $225.75.",base_state())["passed"]

def test_bad_exceeds_ordering_fails():
    assert not validate_final_publication("34.6x exceeds a peer average of 44.5x.",base_state())["passed"]

def test_unsourced_management_intent_fails():
    assert not validate_final_publication("Management believes the shares are undervalued.",base_state())["passed"]

def test_causal_sentiment_without_aspect_evidence_fails():
    assert not validate_final_publication("The market has already priced in valuation strength.",base_state())["passed"]

def test_low_coverage_actionable_recommendation_fails():
    s=base_state(); s["sentiment_analysis_summary"]["status"]="LOW_COVERAGE"; s["recommendation_gate"]={"actionable_recommendation_allowed":False}
    assert not validate_final_publication("Low coverage applies, but investors should buy shares.",s)["passed"]

def test_consistent_structured_output_is_publishable():
    s=base_state(); s["financial_metrics_summary"]["earnings_quality"]={"status":"NORMALIZED","one_off_flagged":True,"gaap_net_margin_pct":20,"normalized_net_margin_pct":14}
    assert validate_final_publication("GAAP margin is 20%; normalized margin is 14% after the material adjustment.",s)["passed"]

def scenario_state():
    s=base_state()
    s["financial_metrics_summary"]["dcf_valuation"]={
        "bear":{"intrinsic_value_per_share":132,"equity_bridge_complete":True},
        "base":{"intrinsic_value_per_share":182,"equity_bridge_complete":True},
        "bull":{"intrinsic_value_per_share":232,"equity_bridge_complete":True},
    }
    return s

def test_msft_range_and_bull_downside_do_not_create_scenario_conflict():
    report="""## DCF Valuation
The bear-to-bull range spans $132–$232 per share.
- **Bear case ($132/share)**
- **Base case ($182/share)**
- **Bull case ($232/share):** the bull case suggests 53% downside.
## Scenario Analysis
**Bull Case ($232/share)** remains the upside scenario.
"""
    extracted=_extract_scenario_prices(report)
    assert {k:[x["value"] for x in v] for k,v in extracted.items()} == {
        "bear":[132.0,132.0], "base":[182.0], "bull":[232.0,232.0,232.0],
    }
    assert all(x["source_section"] for values in extracted.values() for x in values)
    assert all(x["matched_text"] for values in extracted.values() for x in values)
    assert validate_final_publication(report,scenario_state())["passed"]

def test_genuine_bull_price_inconsistency_still_fails():
    report="Bull case ($232/share).\n## Conclusion\nBull = $240 per share."
    result=validate_final_publication(report,scenario_state())
    assert not result["passed"]
    assert "bull scenario price contradicts structured valuation 232." in messages(result)
    assert "bull scenario price is inconsistent across report sections." in messages(result)

def test_current_market_price_near_bull_is_not_a_scenario_price():
    report="Bull sentiment remains possible, but the current market price is $496.95. Bull case ($232/share)."
    extracted=_extract_scenario_prices(report)
    assert [x["value"] for x in extracted["bull"]] == [232.0]
    s=scenario_state(); s["financial_metrics_summary"]["current_price"]=496.95
    assert validate_final_publication(report,s)["passed"]

def test_named_bear_to_bull_range_assigns_endpoints_in_order():
    extracted=_extract_scenario_prices("DCF bear-to-bull range: $132–$232 per share.")
    assert [x["value"] for x in extracted["bear"]] == [132.0]
    assert [x["value"] for x in extracted["base"]] == []
    assert [x["value"] for x in extracted["bull"]] == [232.0]

def test_second_candidate_validation_cannot_see_first_candidate_prices():
    first=validate_final_publication("Bull case ($240/share).",scenario_state())
    second=validate_final_publication("Bull case ($232/share).",scenario_state())
    assert not first["passed"]
    assert second["passed"]

def manager():
    memory=MagicMock(); memory.store_ticker_insight=MagicMock(); memory.store_heuristic=MagicMock(); memory.persist_long_term=MagicMock()
    return ManagerAgent(research_agent=MagicMock(),financial_agent=MagicMock(),sentiment_agent=MagicMock(),memory=memory,llm_client=AsyncMock())

@pytest.mark.asyncio
async def test_invalid_first_synthesis_regenerates_once_and_publishes_corrected():
    corrected=("$89 is about 60.6% below $225.75.\n"
        "## Executive Summary\nSummary.\n## Financial Health\nHealth.\n"
        "## Market Sentiment\nSentiment.\n## Research Highlights\nHighlights.\n"
        "## Risk Factors\nRisks.\n## Scenario Analysis\nScenarios.\n"
        "## Management Commentary Assessment\nNot available.\n"
        "## Conclusion & Outlook\nConclusion.")
    mgr=manager(); mgr._brain_finalise=AsyncMock(side_effect=["$89 is 154% below $225.75.",corrected])
    shared=base_state(); shared.update({"task_query":"NVDA","agent_execution_history":[]})
    result=await mgr._node_finalise({"shared_state":shared,"ticker":"NVDA","session_id":"s","loop_counter":1})
    out=result["shared_state"]
    assert out["publication_status"]=="PUBLISHED"
    assert out["final_report"].startswith("$89 is about 60.6% below $225.75.")
    assert "Section Reliability" in out["final_report"]
    assert out.get("blocked_final_report", "")==""
    assert len(out["publication_validation_attempts"])==2

@pytest.mark.asyncio
async def test_two_invalid_syntheses_block_and_preserve_candidate():
    mgr=manager(); mgr._brain_finalise=AsyncMock(side_effect=["$89 is 154% below $225.75.","34.6x exceeds a peer average of 44.5x."])
    shared=base_state(); shared.update({"task_query":"NVDA","agent_execution_history":[]})
    result=await mgr._node_finalise({"shared_state":shared,"ticker":"NVDA","session_id":"s","loop_counter":1})
    out=result["shared_state"]
    assert out["publication_status"]=="LIMITED_REPORT"
    assert out["execution_status"]=="failed"
    assert out["final_report"]
    assert "Financial Performance Reliability: LIMITED" in out["final_report"]
    assert "Numeric ordering contradiction" not in out["final_report"]
    assert out["final_publication_validation"]["critical_findings"]
    assert out["blocked_final_report"]=="34.6x exceeds a peer average of 44.5x."
    assert out["first_blocked_final_report"]=="$89 is 154% below $225.75."
