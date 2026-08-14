"""
Tests for: agents/financial_agent.py — INTENT-DRIVEN tool selection.

Mocking strategy (mirrors test_financial_agent.py):
  - anthropic.Anthropic injected via `llm_client` (DI point, no real API calls).
  - The MCP session is a plain AsyncMock passed directly as the `session`
    argument to _execute_data_extraction / _execute_ratio_computation, both of
    which take it explicitly — no need to patch stdio_client.

What these tests prove:
  1. The Brain classifies INTENTS (never numbers, never raw tool names beyond
     an allow-listed escape hatch), and those intents reach the Executors.
  2. Different queries resolve to DIFFERENT, correctly-sized tool bundles —
     i.e. selection is query-dependent, not a fixed list.
  3. Tools whose inputs cannot be sourced from raw_numerical_data are skipped
     without a call and without a crash.
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.financial_agent import (
    ANALYSIS_PROFILES,
    COMPUTE_REGISTRY,
    EXTRACTION_REGISTRY,
    FinancialAnalystAgent,
    _FULL_REPORT_PROFILES,
    _MONTE_CARLO_TOOLS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_llm_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


def make_agent(llm=None, max_loops=3):
    return FinancialAnalystAgent(llm_client=llm or MagicMock(), max_loops=max_loops)


def make_agent_state(task_query="Analyze AMD", directives=None, **overrides):
    state = {
        "messages": [], "raw_numerical_data": {}, "calculated_ratios": {},
        "loop_counter": 0, "validation_feedback": "", "is_complete": False,
        "task_query": task_query, "manager_directives": directives or {},
        "intents": [], "extra_tools": [], "rationale": "",
    }
    state.update(overrides)
    return state


# A complete, realistic raw_numerical_data payload — every field a mapper can
# consume. Individual tests strip fields to prove the skip behaviour.
def make_raw(ticker="AMD"):
    return {
        "ticker": ticker,
        "yahoo_ratios": {
            "ticker": ticker, "industry": "Semiconductors", "sector": "Technology",
            "current_price": 150.0, "eps_trailing": 3.0, "pe_ratio": 50.0,
            "market_cap": 240_000_000_000.0, "enterprise_value": 245_000_000_000.0,
            "ebitda": 6_000_000_000.0, "beta": 1.7, "current_ratio": 2.4,
            "total_debt": 2_500_000_000.0, "total_cash": 5_800_000_000.0,
            "diluted_shares": 1_620_000_000.0, "interest_expense": 106_000_000.0,
            "effective_tax_rate": 0.13, "lease_liabilities": 500_000_000.0,
            "minority_interest": 0.0,
        },
        "revenue_growth": {
            "annual_revenue": [
                {"year": 2024, "revenue": 25_785_000_000.0},
                {"year": 2023, "revenue": 22_680_000_000.0},
                {"year": 2022, "revenue": 23_601_000_000.0},
                {"year": 2021, "revenue": 16_434_000_000.0},
            ],
            "annual_net_income": [
                {"year": 2024, "net_income": 1_641_000_000.0},
                {"year": 2023, "net_income": 854_000_000.0},
            ],
            "quarterly_revenue": [{"quarter": "2025-03-31", "revenue": 7_438_000_000.0}],
            "quarterly_net_income": [{"quarter": "2025-03-31", "net_income": 709_000_000.0}],
            "revenue_growth_ttm": 0.36,
        },
        "xbrl_financials": {
            "total_assets": [{"value": 69_226_000_000.0, "period_end": "2024-12-28"}],
            "total_liabilities": [{"value": 12_800_000_000.0, "period_end": "2024-12-28"}],
            "revenue_annual": [{"value": 25_785_000_000.0}],
            "operating_income_annual": [{"value": 1_900_000_000.0}],
            "operating_cash_flow_annual": [{"value": 3_041_000_000.0}],
            "capex_annual": [{"value": 636_000_000.0}],
        },
        "balance_sheet_snapshot": {
            "period_end": "2024-12-28", "ebit": 1_900_000_000.0,
            "current_assets": 19_000_000_000.0, "current_liabilities": 7_900_000_000.0,
            "working_capital": 11_100_000_000.0, "retained_earnings": 3_000_000_000.0,
            "total_equity": 57_000_000_000.0,
        },
    }


class RecordingSession:
    """AsyncMock-style MCP session that records every tool name called and
    returns a plausible, tool-appropriate payload."""

    def __init__(self, industry="Semiconductors", sector="Technology"):
        self.calls: list[tuple[str, dict]] = []
        self._industry, self._sector = industry, sector
        self.call_tool = AsyncMock(side_effect=self._call_tool)

    async def _call_tool(self, tool, arguments=None):
        self.calls.append((tool, arguments or {}))
        payload = {"ok": True}
        if tool == "tool_get_financial_ratios":
            payload = {"industry": self._industry, "sector": self._sector}
        elif tool == "tool_build_forecast":
            payload = {"scenarios": {"base": {"fcf_by_year": [1.0, 2.0, 3.0, 4.0, 5.0]}}}
        elif tool == "tool_calc_dcf_scenarios":
            payload = {"base": {"error": None, "wacc_pct": 9.5,
                                "assumptions": {"terminal_growth_rate_pct": 2.5}}}
        elif tool == "tool_calc_cagr":
            payload = {"cagr_pct": 16.2}
        elif tool == "tool_calc_pe":
            payload = {"pe_ratio": 50.0}
        result = MagicMock()
        result.content = [MagicMock(text=json.dumps(payload))]
        return result

    @property
    def tools_called(self) -> set[str]:
        return {t for t, _ in self.calls}


async def run_compute(agent, state, raw=None):
    """Run the compute executor against a recording session and return it."""
    state["raw_numerical_data"] = raw if raw is not None else make_raw()
    session = RecordingSession()
    await agent._execute_ratio_computation(session, state)
    return session


# ---------------------------------------------------------------------------
# _resolve_tools — the core of intent-driven selection
# ---------------------------------------------------------------------------

class TestResolveTools:
    def test_valuation_selects_dcf_bundle_and_not_health_only_tools(self):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(intents=["valuation"]))

        for tool in ("tool_calc_dcf_scenarios", "tool_calc_reverse_dcf",
                     "tool_calc_pb", "tool_calc_ev_ebitda", "tool_calc_peg"):
            assert tool in compute, f"{tool} missing from valuation bundle"
        # Pure balance-sheet-health tools must NOT come along for the ride.
        for tool in ("tool_calc_altman_z_score", "tool_calc_interest_coverage",
                     "tool_calc_net_debt_to_ebitda", "tool_calc_debt_ratio"):
            assert tool not in compute, f"{tool} leaked into the valuation bundle"

    def test_financial_health_selects_solvency_bundle_and_no_dcf(self):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(intents=["financial_health"]))

        for tool in ("tool_calc_altman_z_score", "tool_calc_quick_ratio",
                     "tool_calc_interest_coverage", "tool_calc_net_debt_to_ebitda",
                     "tool_calc_debt_ratio"):
            assert tool in compute, f"{tool} missing from financial_health bundle"
        for tool in ("tool_calc_dcf_scenarios", "tool_calc_reverse_dcf",
                     "tool_calc_dcf_monte_carlo",
                     "tool_calc_dcf_multi_factor_monte_carlo"):
            assert tool not in compute, f"{tool} leaked into the financial_health bundle"

    def test_multi_intent_is_the_union_of_both_bundles(self):
        agent = make_agent()
        _, val = agent._resolve_tools(make_agent_state(intents=["valuation"]))
        _, health = agent._resolve_tools(make_agent_state(intents=["financial_health"]))
        _, both = agent._resolve_tools(
            make_agent_state(intents=["valuation", "financial_health"])
        )
        assert set(both) == set(val) | set(health)
        assert "tool_calc_dcf_scenarios" in both and "tool_calc_altman_z_score" in both

    def test_ambiguous_query_gets_overview_bundle_only(self):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(intents=["overview"]))

        expected = set(ANALYSIS_PROFILES["overview"]["compute"]) | {"tool_calc_composite_score"}
        assert set(compute) == expected
        assert "tool_calc_dcf_scenarios" not in compute
        assert "tool_calc_altman_z_score" not in compute

    def test_no_intents_defaults_to_overview(self):
        agent = make_agent()
        _, from_empty = agent._resolve_tools(make_agent_state(intents=[]))
        _, from_overview = agent._resolve_tools(make_agent_state(intents=["overview"]))
        assert from_empty == from_overview

    def test_baseline_extraction_and_compute_always_present(self):
        agent = make_agent()
        for intent in ("liquidity", "growth", "peer_relative", "cash_flow"):
            extraction, compute = agent._resolve_tools(make_agent_state(intents=[intent]))
            assert "tool_get_financial_ratios" in extraction
            assert "tool_get_xbrl_financials" in extraction
            assert "tool_calc_composite_score" in compute

    def test_unknown_intent_is_ignored(self):
        agent = make_agent()
        _, weird = agent._resolve_tools(
            make_agent_state(intents=["overview", "astrology", "vibes"])
        )
        _, plain = agent._resolve_tools(make_agent_state(intents=["overview"]))
        assert weird == plain

    def test_unknown_extra_tools_are_ignored_known_ones_are_added(self):
        agent = make_agent()
        _, plain = agent._resolve_tools(make_agent_state(intents=["overview"]))

        _, junk = agent._resolve_tools(
            make_agent_state(intents=["overview"], extra_tools=["tool_calc_nonsense", "rm -rf"])
        )
        assert junk == plain

        extraction, compute = agent._resolve_tools(make_agent_state(
            intents=["overview"],
            extra_tools=["tool_calc_altman_z_score", "tool_get_peer_comparison"],
        ))
        assert "tool_calc_altman_z_score" in compute
        assert "tool_get_peer_comparison" in extraction

    def test_resolved_tools_are_deduped_and_registry_backed(self):
        agent = make_agent()
        extraction, compute = agent._resolve_tools(
            make_agent_state(intents=["full_report"])
        )
        assert len(extraction) == len(set(extraction))
        assert len(compute) == len(set(compute))
        assert all(t in EXTRACTION_REGISTRY for t in extraction)
        assert all(t in COMPUTE_REGISTRY for t in compute)


# ---------------------------------------------------------------------------
# full_report macro-intent + depth gate
# ---------------------------------------------------------------------------

class TestFullReportAndDepth:
    def test_full_report_is_the_union_of_the_narrow_profiles(self):
        agent = make_agent()
        _, full = agent._resolve_tools(make_agent_state(intents=["full_report"]))

        union: set[str] = {"tool_calc_composite_score"}
        for name in _FULL_REPORT_PROFILES:
            union |= set(ANALYSIS_PROFILES[name]["compute"])
        assert set(full) == union

    def test_full_report_is_much_broader_than_overview(self):
        agent = make_agent()
        _, full = agent._resolve_tools(make_agent_state(intents=["full_report"]))
        _, overview = agent._resolve_tools(make_agent_state(intents=["overview"]))
        assert set(overview) < set(full)
        assert len(full) > len(overview) * 2

    def test_full_report_defaults_to_deep_so_monte_carlo_is_included(self):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(intents=["full_report"]))
        assert _MONTE_CARLO_TOOLS <= set(compute)

    def test_full_report_still_honours_an_explicit_shallow_directive(self):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(
            intents=["full_report"], directives={"depth": "shallow"},
        ))
        assert not (_MONTE_CARLO_TOOLS & set(compute))
        # …but the cheap valuation tools survive.
        assert "tool_calc_dcf_scenarios" in compute
        assert "tool_calc_reverse_dcf" in compute

    def test_depth_shallow_drops_monte_carlo_deep_keeps_it(self):
        agent = make_agent()
        _, shallow = agent._resolve_tools(make_agent_state(
            intents=["valuation"], directives={"depth": "shallow"}))
        _, deep = agent._resolve_tools(make_agent_state(
            intents=["valuation"], directives={"depth": "deep"}))

        assert not (_MONTE_CARLO_TOOLS & set(shallow))
        assert _MONTE_CARLO_TOOLS <= set(deep)
        assert "tool_calc_dcf_scenarios" in shallow

    @pytest.mark.parametrize("directives", [
        {}, {"depth": "standard"}, {"depth": "wat"}, {"analysis_depth": "unknown"},
    ])
    def test_unrecognised_depth_defaults_to_standard(self, directives):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(
            intents=["valuation"], directives=directives))
        # standard keeps the Monte-Carlo variants (only shallow drops them)
        assert _MONTE_CARLO_TOOLS <= set(compute)

    @pytest.mark.parametrize("key", ["depth", "analysis_depth", "detail_level", "level"])
    def test_depth_key_is_tolerant_of_whatever_the_request_sends(self, key):
        agent = make_agent()
        _, compute = agent._resolve_tools(make_agent_state(
            intents=["valuation"], directives={key: "quick"}))
        assert not (_MONTE_CARLO_TOOLS & set(compute))


# ---------------------------------------------------------------------------
# Brain classification + state plumbing (bug (a))
# ---------------------------------------------------------------------------

class TestBrainClassification:
    @pytest.mark.asyncio
    async def test_valid_json_intents_parsed(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "intents": ["valuation", "growth"],
            "rationale": "Query asks about fair value and growth.",
            "extra_tools": [],
        }))
        out = await make_agent(llm)._brain(make_agent_state())
        assert out["intents"] == ["valuation", "growth"]
        assert "fair value" in out["rationale"]

    @pytest.mark.asyncio
    async def test_unknown_intents_are_dropped_and_empty_falls_back(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "intents": ["tarot", "moon_phase"], "rationale": "n/a", "extra_tools": [],
        }))
        out = await make_agent(llm)._brain(make_agent_state())
        assert out["intents"] == ["overview"]

    @pytest.mark.asyncio
    async def test_unknown_extra_tools_are_dropped(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "intents": ["overview"], "rationale": "generic",
            "extra_tools": ["tool_calc_altman_z_score", "tool_do_crimes"],
        }))
        out = await make_agent(llm)._brain(make_agent_state())
        assert out["extra_tools"] == ["tool_calc_altman_z_score"]

    @pytest.mark.asyncio
    async def test_non_json_response_defaults_to_overview(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response("I think we should look at AMD.")
        out = await make_agent(llm)._brain(make_agent_state())
        assert out["intents"] == ["overview"]

    @pytest.mark.asyncio
    async def test_api_exception_defaults_to_overview_without_raising(self):
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("api down")
        out = await make_agent(llm)._brain(make_agent_state())
        assert out["intents"] == ["overview"]
        assert out["extra_tools"] == []

    @pytest.mark.asyncio
    async def test_intents_flow_from_node_brain_into_state(self):
        """REGRESSION (bug a): _node_brain used to return ONLY loop_counter,
        discarding the Brain's output entirely — so no plan could ever reach
        the Executors and tool selection was effectively hardcoded."""
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "intents": ["liquidity"], "rationale": "short-term obligations",
            "extra_tools": ["tool_calc_altman_z_score"],
        }))
        out = await make_agent(llm)._node_brain(make_agent_state())

        assert out["loop_counter"] == 1
        assert out["intents"] == ["liquidity"]
        assert out["extra_tools"] == ["tool_calc_altman_z_score"]
        assert out["rationale"] == "short-term obligations"


# ---------------------------------------------------------------------------
# Executors actually honour the resolved bundle (bug (b))
# ---------------------------------------------------------------------------

class TestExecutorsHonourIntent:
    @pytest.mark.asyncio
    async def test_valuation_query_calls_dcf_bundle_not_health_tools(self):
        agent = make_agent()
        state = make_agent_state("What's a fair value for AMD — is it overvalued?",
                                 intents=["valuation"])
        session = await run_compute(agent, state)

        for tool in ("tool_calc_dcf_scenarios", "tool_calc_reverse_dcf",
                     "tool_calc_pb", "tool_calc_ev_ebitda", "tool_calc_peg"):
            assert tool in session.tools_called, f"{tool} was never called"
        assert "tool_calc_altman_z_score" not in session.tools_called

    @pytest.mark.asyncio
    async def test_financial_health_query_calls_solvency_tools_not_dcf(self):
        agent = make_agent()
        state = make_agent_state("Is AMD's balance sheet healthy? Bankruptcy risk?",
                                 intents=["financial_health"])
        session = await run_compute(agent, state)

        for tool in ("tool_calc_altman_z_score", "tool_calc_interest_coverage",
                     "tool_calc_net_debt_to_ebitda", "tool_calc_debt_ratio"):
            assert tool in session.tools_called, f"{tool} was never called"
        assert not (_MONTE_CARLO_TOOLS & session.tools_called)
        assert "tool_calc_dcf_scenarios" not in session.tools_called

    @pytest.mark.asyncio
    async def test_extraction_only_calls_resolved_tools(self):
        agent = make_agent()
        state = make_agent_state("Is AMD liquid enough?", intents=["liquidity"])
        session = RecordingSession()
        await agent._execute_data_extraction(session, state)

        expected, _ = agent._resolve_tools(state)
        assert session.tools_called == set(expected)
        assert "tool_get_peer_comparison" not in session.tools_called

    @pytest.mark.asyncio
    async def test_narrow_query_resolves_to_a_smaller_set_than_full_report(self):
        agent = make_agent()
        narrow = make_agent_state("is AMD financially healthy?", intents=["financial_health"])
        broad = make_agent_state("INVESTMENT ANALYSIS REPORT: AMD", intents=["full_report"])

        narrow_session = await run_compute(agent, narrow)
        broad_session = await run_compute(agent, broad)

        assert len(narrow_session.tools_called) < len(broad_session.tools_called)
        assert narrow_session.tools_called != broad_session.tools_called


# ---------------------------------------------------------------------------
# Determinism / graceful skipping
# ---------------------------------------------------------------------------

class TestDeterministicSkipping:
    @pytest.mark.asyncio
    async def test_tool_with_missing_inputs_is_skipped_without_call_or_crash(self):
        """EV/EBITDA needs enterprise_value + ebitda from
        tool_get_financial_ratios. Strip them and the tool must be skipped —
        no call, no exception, an explicit 'unavailable' placeholder."""
        agent = make_agent()
        raw = make_raw()
        del raw["yahoo_ratios"]["enterprise_value"]
        del raw["yahoo_ratios"]["ebitda"]

        state = make_agent_state(intents=["valuation"])
        session = await run_compute(agent, state, raw=raw)

        assert "tool_calc_ev_ebitda" not in session.tools_called
        assert state["calculated_ratios"]["ev_ebitda"]["interpretation"] == "unavailable"

    @pytest.mark.asyncio
    async def test_structurally_unsupplied_tools_never_call(self):
        """gross_margin needs COGS and quick_ratio needs receivables +
        short-term investments; no extraction tool in this pipeline supplies
        either, so both must skip even on a fully-populated payload."""
        agent = make_agent()
        state = make_agent_state(intents=["profitability", "liquidity"])
        session = await run_compute(agent, state)

        assert "tool_calc_gross_margin" not in session.tools_called
        assert "tool_calc_quick_ratio" not in session.tools_called
        assert state["calculated_ratios"]["gross_margin"]["interpretation"] == "unavailable"

    @pytest.mark.asyncio
    async def test_empty_raw_data_skips_everything_without_crashing(self):
        agent = make_agent()
        state = make_agent_state(intents=["full_report"])
        session = await run_compute(agent, state, raw={"ticker": "AMD"})

        # composite_score tolerates all-None inputs by design; nothing else runs.
        assert session.tools_called <= {"tool_calc_composite_score"}

    @pytest.mark.asyncio
    async def test_no_llm_supplied_numeric_reaches_any_tool(self):
        """Every argument of every compute call must be traceable to
        raw_numerical_data — the LLM is never consulted here at all."""
        agent = make_agent()
        state = make_agent_state(intents=["full_report"])
        raw = make_raw()
        session = await run_compute(agent, state, raw=raw)

        agent._llm.messages.create.assert_not_called()
        pe_args = dict(session.calls)["tool_calc_pe"]
        assert pe_args == {"price": raw["yahoo_ratios"]["current_price"],
                           "eps": raw["yahoo_ratios"]["eps_trailing"]}
        ev_args = dict(session.calls)["tool_calc_ev_ebitda"]
        assert ev_args == {"enterprise_value": raw["yahoo_ratios"]["enterprise_value"],
                           "ebitda": raw["yahoo_ratios"]["ebitda"]}

    @pytest.mark.asyncio
    async def test_period_provenance_is_attached_to_every_computed_ratio(self):
        agent = make_agent()
        state = make_agent_state(intents=["full_report"])
        await run_compute(agent, state)

        computed = [
            v for v in state["calculated_ratios"].values()
            if isinstance(v, dict) and v.get("_inputs") is not None
        ]
        assert computed, "no ratio was actually computed"
        assert all("_period" in v for v in computed)


# ---------------------------------------------------------------------------
# The AMD case — behaviour proof end to end through classification
# ---------------------------------------------------------------------------

class TestAmdFullReportCase:
    AMD_QUERY = "INVESTMENT ANALYSIS REPORT: ADVANCED MICRO DEVICES, INC. (AMD)"

    @pytest.mark.asyncio
    async def test_amd_report_query_runs_the_union_bundle_with_peers_and_mc(self):
        llm = MagicMock()
        llm.messages.create.return_value = make_llm_response(json.dumps({
            "intents": ["full_report"],
            "rationale": "Query requests a comprehensive investment analysis report.",
            "extra_tools": [],
        }))
        agent = make_agent(llm)
        state = make_agent_state(self.AMD_QUERY)

        state.update(await agent._node_brain(state))
        assert state["intents"] == ["full_report"]

        extract_session = RecordingSession()
        await agent._execute_data_extraction(extract_session, state)
        # AMD's industry ("Semiconductors") IS in the static INDUSTRY_PEERS
        # map, so the peer gate opens and the comparison actually fires.
        assert "tool_get_peer_comparison" in extract_session.tools_called
        peers = dict(extract_session.calls)["tool_get_peer_comparison"]["peers"]
        assert peers and "AMD" not in peers

        state["raw_numerical_data"] = make_raw("AMD")
        compute_session = RecordingSession()
        await agent._execute_ratio_computation(compute_session, state)

        assert "tool_calc_dcf_multi_factor_monte_carlo" in compute_session.tools_called
        assert "tool_calc_altman_z_score" in compute_session.tools_called
        assert "tool_calc_reverse_dcf" in compute_session.tools_called
        # NOT the thin overview fallback
        _, overview = agent._resolve_tools(make_agent_state(intents=["overview"]))
        assert not compute_session.tools_called <= set(overview)

    @pytest.mark.asyncio
    async def test_peer_comparison_skipped_when_industry_is_unmapped(self):
        agent = make_agent()
        state = make_agent_state("Compare ZZZZ to peers", intents=["peer_relative"])
        # tool_get_financial_ratios reports an industry absent from the
        # static INDUSTRY_PEERS map, so the peer gate must stay shut.
        session = RecordingSession(industry="Interpretive Dance", sector="Arts")
        await agent._execute_data_extraction(session, state)

        assert "tool_get_peer_comparison" not in session.tools_called
        note = state["raw_numerical_data"]["peer_comparison"]["note"]
        assert "No peer mapping available" in note
