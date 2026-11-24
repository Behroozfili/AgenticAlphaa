import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.manager_agent import ManagerAgent
from core.forecast_contract import ForecastContext
from evaluation.validate_period_consistency import (
    detect_material_period_conflicts, validate_financial_fact_contract,
)
from rag.loader import RawDocument
from rag.processor import AlphaProcessor
from rag.vector_store import AlphaVectorStore, VectorStoreConfig
from tests.unit_tests.financial_test import stage5_context
from tools.financial_tools.sec_edgar import _extract_annual


def fact(**overrides):
    value = {
        "metric": "revenue", "value": 100, "fiscal_year": 2026,
        "fiscal_quarter": "Q1", "period_start": "2026-01-01",
        "period_end": "2026-03-31", "duration_months": 3,
        "filing_type": "10-Q", "accession_number": "a1",
        "metric_scope": "duration", "gaap_status": "GAAP",
    }
    value.update(overrides)
    return value


def test_xbrl_extraction_preserves_full_period_and_filing_provenance():
    payload = {"facts": {"us-gaap": {"Revenue": {"units": {"USD": [{
        "start": "2025-01-01", "end": "2025-12-31", "val": 100,
        "form": "10-K", "fp": "FY", "fy": 2025,
        "accn": "0001-25-000001", "filed": "2026-02-01",
    }]}}}}}
    row = _extract_annual(
        payload, ["Revenue"], context=ForecastContext.from_dict(stage5_context("NVDA")),
    )[0]
    assert row["period_start"] == "2025-01-01"
    assert row["period_end"] == "2025-12-31"
    assert row["duration_months"] == 12
    assert row["filing_type"] == "10-K"
    assert row["accession_number"] == "0001-25-000001"
    assert row["metric_scope"] == "duration"
    assert row["gaap_status"] == "GAAP"


def test_processor_preserves_period_metadata_additively():
    processor = AlphaProcessor()
    doc = RawDocument(
        title="Q1 filing", content="Revenue increased.", url="https://example.com/q1",
        source_type="news", ticker="AMD", published_at="2026-04-01T00:00:00+00:00",
        metadata=fact(),
    )
    metadata = processor.process([doc])[0].metadata
    for field in ("fiscal_year", "fiscal_quarter", "period_start", "period_end",
                  "duration_months", "filing_type", "accession_number", "metric_scope", "gaap_status"):
        assert field in metadata


def test_exact_period_filters_are_sent_to_rpc_before_ranking():
    store = AlphaVectorStore.__new__(AlphaVectorStore)
    store.config = VectorStoreConfig(schema="legacy")
    store.rpc_name = store.config.legacy_rpc
    store.client = MagicMock()
    store.client.rpc.return_value.execute.return_value.data = []
    store.hybrid_search([0.1], "revenue", fiscal_year=2026, fiscal_quarter="Q2",
                        period_end="2026-06-30", metric_scope="duration")
    params = store.client.rpc.call_args.args[1]
    assert params["filter_fiscal_year"] == 2026
    assert params["filter_fiscal_quarter"] == "Q2"
    assert params["filter_period_end"] == "2026-06-30"
    assert params["filter_metric_scope"] == "duration"


def test_q1_cannot_silently_enter_q2_claim():
    findings = detect_material_period_conflicts([
        fact(), fact(fiscal_quarter="Q2", period_start="2026-04-01",
                     period_end="2026-06-30", accession_number="a2"),
    ])
    assert findings[0]["severity"] == "CRITICAL"
    assert findings[0]["blocking"] is True


def test_explicit_comparison_allows_distinct_periods():
    findings = detect_material_period_conflicts([
        fact(comparison=True),
        fact(fiscal_quarter="Q2", period_start="2026-04-01",
             period_end="2026-06-30", accession_number="a2", comparison=True),
    ])
    assert findings == []


def test_material_fact_missing_provenance_is_blocking():
    incomplete = {"metric": "net_income", "value": 10, "period_end": "2026-03-31"}
    assert "accession_number" in validate_financial_fact_contract(incomplete)
    assert detect_material_period_conflicts([incomplete])[0]["blocking"] is True


def manager():
    memory = MagicMock()
    memory.store_ticker_insight = MagicMock()
    memory.store_heuristic = MagicMock()
    memory.persist_long_term = MagicMock()
    return ManagerAgent(
        research_agent=MagicMock(), financial_agent=MagicMock(), sentiment_agent=MagicMock(),
        memory=memory, llm_client=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_manager_blocks_publication_before_synthesis_on_structured_conflict():
    mgr = manager()
    shared = {"financial_metrics_summary": {"financial_facts": [
        fact(), fact(fiscal_quarter="Q2", period_start="2026-04-01",
                     period_end="2026-06-30", accession_number="a2"),
    ]}}
    result = await mgr._node_finalise({
        "shared_state": shared, "ticker": "AMD", "session_id": "s", "loop_counter": 1,
    })
    assert result["shared_state"]["final_report"]
    assert "Financial Performance Reliability: WITHHELD" in result["shared_state"]["final_report"]
    assert "period_consistency" not in result["shared_state"]["final_report"]
    assert result["shared_state"]["publication_status"] == "LIMITED_REPORT"
    mgr._llm.messages.create.assert_not_awaited()
