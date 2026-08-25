from agents.executor_helpers.search_executor_helpers import (
    build_sec_filing_coverage,
    sec_filing_doc_key,
    tool_call_signature,
)
from agents.prompts.propt_search_agent import BRAIN_SYSTEM_PROMPT


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
    assert '"accession_number"' in BRAIN_SYSTEM_PROMPT
