import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
from evaluation.canonical_synthesis import build_canonical_synthesis_payload


def state():
    rows = {
        "bear": [{"year": 1, "revenue_growth_pct": 4, "revenue": 104, "operating_margin_pct": 17, "tax_rate_pct": 21, "da": 9, "capex": 14, "working_capital_change": 2, "fcf": 70}],
        "base": [{"year": 1, "revenue_growth_pct": 8, "revenue": 108, "operating_margin_pct": 20, "tax_rate_pct": 21, "da": 10, "capex": 13, "working_capital_change": 2, "fcf": 90}],
        "bull": [{"year": 1, "revenue_growth_pct": 12, "revenue": 112, "operating_margin_pct": 22, "tax_rate_pct": 21, "da": 11, "capex": 12, "working_capital_change": 2, "fcf": 110}],
    }
    dcf = {}
    for name, price in {"bear": 132, "base": 182, "bull": 232}.items():
        dcf[name] = {
            "intrinsic_value_per_share": price,
            "enterprise_value": price * 10,
            "equity_value": price * 9,
            "wacc_pct": 9,
            "assumptions": {"terminal_growth_rate_pct": 2.5},
            "equity_bridge_complete": True,
            "equity_bridge": {"enterprise_value": price * 10, "cash": 50, "total_debt": 20, "diluted_shares": 10, "complete": True},
        }
    return {
        "task_query": "Analyze MSFT",
        "manager_directives": {"ticker": "MSFT"},
        "financial_metrics_summary": {
            "ticker": "MSFT", "current_price": 496.95,
            "roe": {"roe_pct": 30.23, "_period": "annual:2026"},
            "de_ratio": {"de_ratio": .71, "_period": "annual:2026-06-30"},
            "forecast": {"scenarios": {k: {"yearly_forecast": v} for k, v in rows.items()}},
            "dcf_valuation": dcf,
            "earnings_quality": {"status": "COMPLETE", "one_off_flagged": True, "material_non_operating_income_pct": 8, "gaap_net_margin_pct": 39.74, "normalized_net_margin_pct": 31.74},
            "typed_claims": [{"claim_type": "INFERENCE", "text": "Demand may remain durable", "confidence": "MEDIUM"}],
            "legacy_dcf_summary": "Bull value is $999 and every case grows 8%.",
        },
        "sentiment_analysis_summary": {
            "status": "LOW_COVERAGE", "narrative": "Strong bullish market consensus",
            "aspect_observations": [
                {"aspect": "valuation", "polarity": "Bearish", "evidence_span": "Valuation concerns weighed on sentiment.", "document_url": "https://example/a", "causal_language_supported": True},
                {"aspect": "ai_demand", "polarity": "Bullish", "evidence_span": "", "document_url": "https://example/b", "causal_language_supported": True},
            ],
        },
        "recommendation_gate": {"confidence_level": "HIGH", "recommendation_mode": "ACTIONABLE_ALLOWED", "actionable_recommendation_allowed": True},
    }


def test_distinct_scenario_paths_are_not_collapsed_to_scalar_growth():
    p = build_canonical_synthesis_payload(state())
    paths = p["valuation"]["scenarios"]
    assert [paths[x]["yearly_forecast"][0]["revenue_growth_pct"] for x in ("bear", "base", "bull")] == [4, 8, 12]
    assert "legacy_dcf_summary" not in json.dumps(p)


def test_repeated_bull_price_has_one_canonical_origin():
    p = build_canonical_synthesis_payload(state())
    assert p["valuation"]["scenarios"]["bull"]["equity_value_per_share"] == 232
    assert json.dumps(p).count('"equity_value_per_share": 232') == 1


def test_material_earnings_quality_preserves_gaap_and_normalized_margins():
    eq = build_canonical_synthesis_payload(state())["financial"]["earnings_quality"]
    assert (eq["gaap_net_margin_pct"], eq["normalized_net_margin_pct"]) == (39.74, 31.74)
    assert eq["material_non_operating_income_pct"] == 8


@pytest.mark.parametrize(("metric", "expected"), [("roe", "annual:2026"), ("de_ratio", "annual:2026-06-30")])
def test_annual_metric_period_is_explicit(metric, expected):
    item = build_canonical_synthesis_payload(state())["financial"]["performance_metrics"][metric]
    assert item["period_label"] == expected


def test_completed_equity_bridge_is_authoritative_over_legacy_claim():
    s = state()
    s["financial_metrics_summary"]["legacy_dcf_summary"] = "Net debt was not adjusted."
    p = build_canonical_synthesis_payload(s)
    assert p["valuation"]["scenarios"]["base"]["equity_bridge_complete"] is True
    assert "Net debt was not adjusted" not in json.dumps(p)


def test_inference_has_non_fact_wording_rule():
    claim = build_canonical_synthesis_payload(state())["claims"][0]
    assert claim["claim_type"] == "INFERENCE"
    assert "never state as fact" in claim["synthesis_wording_rule"]


def test_low_coverage_cannot_produce_high_actionable_language():
    policy = build_canonical_synthesis_payload(state())["recommendation_policy"]
    assert policy["allowed_language_class"] in {"LOW", "BLOCKED"}
    assert policy["actionable_recommendation_allowed"] is False


def test_medium_gate_preserves_qualified_conditional_language():
    s = state()
    s["sentiment_analysis_summary"]["status"] = "VALID"
    s["recommendation_gate"] = {"confidence_level": "MEDIUM", "recommendation_mode": "CONDITIONAL_ONLY", "actionable_recommendation_allowed": False}
    policy = build_canonical_synthesis_payload(s)["recommendation_policy"]
    assert policy["allowed_language_class"] == "MEDIUM"
    assert policy["wording_constraint"] == "qualified conditional directional language only"
    assert policy["actionable_recommendation_allowed"] is False


def test_aspect_without_evidence_is_excluded_from_causal_context():
    sentiment = build_canonical_synthesis_payload(state())["sentiment"]
    assert [x["aspect"] for x in sentiment["aspects"]] == ["valuation"]
    assert sentiment["causal_aspects_supported"] == ["valuation"]
    assert sentiment["aspects"][0]["causal_language_supported"] is True


def test_msft_mentions_without_causal_language_do_not_gain_causal_support():
    s = state()
    s["sentiment_analysis_summary"].update({
        # Simulate a stale/upstream list that previously promoted mentions.
        "causal_aspects_supported": ["competition", "valuation"],
        "aspect_observations": [
            {
                "aspect": "valuation",
                "polarity": "Bearish",
                "evidence_span": "Microsoft's valuation remains elevated.",
                "document_url": "https://example.com/msft-valuation",
                "causal_language_supported": False,
            },
            {
                "aspect": "competition",
                "polarity": "Bearish",
                "evidence_span": "Competition from Amazon and Google is intensifying.",
                "document_url": "https://example.com/msft-competition",
                "causal_language_supported": False,
            },
        ],
    })

    sentiment = build_canonical_synthesis_payload(s)["sentiment"]

    assert sentiment["aspects"] == []
    assert sentiment["causal_aspects_supported"] == []
    assert sentiment["narrative_policy"]["pricing_in_claims_allowed"] is False


def test_canonical_sentiment_fails_closed_for_legacy_row_without_causal_flag():
    s = state()
    s["sentiment_analysis_summary"]["aspect_observations"] = [{
        "aspect": "valuation",
        "polarity": "Bearish",
        "evidence_span": "Valuation concerns weighed on sentiment.",
        "document_title": "Legacy MSFT observation",
    }]

    sentiment = build_canonical_synthesis_payload(s)["sentiment"]

    assert sentiment["aspects"] == []
    assert sentiment["causal_aspects_supported"] == []


def test_structured_dcf_wins_over_conflicting_legacy_summary():
    p = build_canonical_synthesis_payload(state())
    assert p["valuation"]["scenarios"]["bull"]["equity_value_per_share"] == 232
    assert "$999" not in json.dumps(p)


def test_structured_sentiment_wins_over_legacy_narrative():
    p = build_canonical_synthesis_payload(state())
    assert p["sentiment"]["status"] == "LOW_COVERAGE"
    assert "Strong bullish market consensus" not in json.dumps(p)


def test_canonical_structured_conflict_is_preserved_not_guessed():
    s = state()
    s["financial_metrics_summary"]["dcf_valuation"]["bull"]["equity_value_per_share"] = 240
    p = build_canonical_synthesis_payload(s)
    assert p["valuation"]["scenarios"]["bull"]["equity_value_per_share"] == 240
    conflict = p["data_quality"]["canonical_conflicts"][0]
    assert conflict["fields"] == {"equity_value_per_share": 240, "intrinsic_value_per_share": 232}
    assert conflict["action"] == "PRESERVE_FOR_PHASE_7_BLOCKING"


def test_msft_golden_context_has_one_authoritative_scenario_set_and_constraints():
    p = build_canonical_synthesis_payload(state())
    assert {k: v["equity_value_per_share"] for k, v in p["valuation"]["scenarios"].items()} == {"bear": 132, "base": 182, "bull": 232}
    assert p["financial"]["performance_metrics"]["roe"]["period_label"].startswith("annual:")
    assert p["financial"]["performance_metrics"]["de_ratio"]["period_label"].startswith("annual:")
    assert p["sentiment"]["status"] == "LOW_COVERAGE"
    assert p["recommendation_policy"]["actionable_recommendation_allowed"] is False


@pytest.mark.asyncio
async def test_manager_finaliser_receives_only_canonical_factual_context():
    llm = AsyncMock()
    response = MagicMock(stop_reason="end_turn")
    response.content = [MagicMock(text="Report")]
    llm.messages.create.return_value = response
    memory = MagicMock()
    manager = ManagerAgent(MagicMock(), MagicMock(), MagicMock(), memory, llm_client=llm)
    s = state()
    s.update({"aggregated_research_context": [], "agent_execution_history": []})

    assert await manager._brain_finalise(s) == "Report"
    content = llm.messages.create.await_args.kwargs["messages"][0]["content"]
    assert "CANONICAL SYNTHESIS PAYLOAD (SOLE FACTUAL AUTHORITY)" in content
    assert "FINANCIAL METRICS SUMMARY:" not in content
    assert "SENTIMENT ANALYSIS SUMMARY:" not in content
    assert "$999" not in content
    assert s["canonical_synthesis_payload"]["valuation"]["scenarios"]["bull"]["equity_value_per_share"] == 232
