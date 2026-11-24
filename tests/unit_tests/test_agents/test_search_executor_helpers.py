from agents.executor_helpers.search_executor_helpers import (
    build_sec_filing_coverage,
    sec_filing_doc_key,
    tool_call_signature,
)
from agents.prompts.prompt_research_agent import BRAIN_SYSTEM_PROMPT
from agents.executor_helpers.search_executor_helpers import inject_execution_identity
from core.forecast_contract import ForecastContext
import pytest
from unittest.mock import AsyncMock, patch
from tools.research_tools.research_server import list_tools, call_tool
import json
import sys
from unittest.mock import MagicMock


def test_different_accessions_have_different_document_keys():
    first = {
        "ticker": "NVDA",
        "form_type": "10-Q",
        "accession_number": "0001045810-25-000001",
        "sections": ["risk_factors"],
    }
    second = {
        **first,
        "accession_number": "0001045810-25-000002",
    }

    assert sec_filing_doc_key(first) != sec_filing_doc_key(second)


def test_dashed_and_compact_accession_share_document_key():
    dashed = {
        "ticker": "NVDA",
        "form_type": "10-Q",
        "accession_number": "0001045810-25-000001",
    }
    compact = {**dashed, "accession_number": "000104581025000001"}

    assert sec_filing_doc_key(dashed) == sec_filing_doc_key(compact)


def test_coverage_does_not_merge_sections_across_accessions():
    first = {
        "ticker": "NVDA",
        "form_type": "10-Q",
        "accession_number": "0001045810-25-000001",
        "sections": ["risk_factors"],
    }
    second = {
        "ticker": "NVDA",
        "form_type": "10-Q",
        "accession_number": "0001045810-25-000002",
        "sections": ["mda"],
    }
    signatures = {
        tool_call_signature("sec_edgar_filing", first),
        tool_call_signature("sec_edgar_filing", second),
    }

    coverage = build_sec_filing_coverage(signatures)

    assert coverage[sec_filing_doc_key(first)] == {"risk_factors"}
    assert coverage[sec_filing_doc_key(second)] == {"mda"}


def test_calls_without_accession_keep_existing_document_key_shape():
    arguments = {"ticker": "NVDA", "form_type": "10-Q", "mode": "preview"}

    assert sec_filing_doc_key(arguments) == "NVDA|10-Q|preview"


def test_research_brain_prompt_describes_both_sec_workflows():
    assert "latest/current filing" in BRAIN_SYSTEM_PROMPT
    assert "sec_edgar_search is used to discover" in BRAIN_SYSTEM_PROMPT


def test_execution_identity_overrides_model_supplied_values():
    context = ForecastContext(
        ticker="MSFT", as_of="2024-01-31T21:00:00Z",
        schema_version="v1", dataset_version="dataset-v1",
        feature_version="feature-v1", model_version="model-v1",
    )
    arguments = inject_execution_identity(
        "rag_hybrid_query",
        {"query": "risk", "forecast_context": {"ticker": "BAD"}, "run_id": "bad"},
        forecast_context=context, run_id="run-04b",
    )
    assert arguments["forecast_context"] == context.to_dict()
    assert arguments["run_id"] == "run-04b"


@pytest.mark.asyncio
async def test_every_research_server_schema_requires_context_and_run_id():
    tools = (await list_tools()).tools
    assert tools
    for tool in tools:
        required = set(tool.inputSchema["required"])
        assert {"forecast_context", "run_id"} <= required


@pytest.mark.asyncio
async def test_server_dispatches_validated_context_and_run_id():
    context = ForecastContext(
        ticker="MSFT", as_of="2024-01-31T21:00:00Z",
        schema_version="v1", dataset_version="dataset-v1",
        feature_version="feature-v1", model_version="model-v1",
    )
    with patch(
        "tools.research_tools.research_server.tavily_search",
        new=AsyncMock(return_value={"evidence": []}),
    ) as provider:
        result = await call_tool("tavily_search", {
            "query": "MSFT", "forecast_context": context.to_dict(), "run_id": "run-04b",
        })
    assert result.isError is False
    assert provider.await_args.kwargs["forecast_context"] == context
    assert provider.await_args.kwargs["run_id"] == "run-04b"
    assert '"accession_number"' in BRAIN_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_server_tool_error_drops_private_sentinel_from_logs_sentry_and_payload(caplog):
    fake_sentry = MagicMock()
    with patch.dict(sys.modules, {"sentry_sdk": fake_sentry}), patch(
        "tools.research_tools.research_server.sentry_enabled", return_value=True,
    ):
        result = await call_tool("unknown_tool", {
            "query": "PRIVATE_SENTINEL_QUERY", "run_id": "run-privacy-test",
            "forecast_context": {},
        })
    rendered = json.dumps([item.text for item in result.content]) + caplog.text
    assert "PRIVATE_SENTINEL_QUERY" not in rendered
    assert "RESEARCH_TOOL_FAIL_CLOSED" in rendered
    fake_sentry.capture_exception.assert_not_called()
    fake_sentry.capture_message.assert_called_once_with("RESEARCH_TOOL_FAIL_CLOSED", level="error")
