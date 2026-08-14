"""
conftest.py — shared fixtures for AgenticAlpha integration tests.

Run with:
    pytest tests/integration -v -m "not slow"      # tiers 1-2 only (fast, cheap)
    pytest tests/integration -v                     # everything including full pipeline

Cost note: tier 3 tests (test_full_pipeline.py) make real LLM + API calls.
Mark them with @pytest.mark.slow so they're excluded from routine runs and
only run deliberately (e.g. before a release, or in a scheduled nightly CI job).
"""
import os
import pytest

from agents.research_agent import ResearchAgent
from agents.financial_agent import FinancialAnalystAgent
from agents.sentiment_agent import SentimentAgent
from agents.manager_agent import ManagerAgent
from memory.manager_memory import ManagerMemory

# Confirmed against tools/research_tools/research_server.py and
# tools/financial_tools/financial_server.py's own layout, and further
# corroborated by sentiment_server.py's own imports
# (tools.sentiment_tools.fear_greed_calculator etc. live right next to it).
SENTIMENT_SERVER_PATH = "tools/sentiment_tools/sentiment_server.py"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full end-to-end pipeline test, costs real API calls")
    config.addinivalue_line("markers", "mcp: requires the MCP server subprocess to start")
    config.addinivalue_line("markers", "db: requires a configured, populated vector store (Supabase)")


@pytest.fixture(scope="session")
def require_api_keys():
    """Skip tests that need real API keys if they're not configured."""
    missing = [k for k in ("ANTHROPIC_API_KEY",) if not os.environ.get(k)]
    if missing:
        pytest.skip(f"Missing required env vars: {missing}")


@pytest.fixture
def known_ticker():
    """
    A ticker known to have clean data across Yahoo Finance, SEC XBRL, and
    news coverage — use this (not a random/obscure ticker) for integration
    tests so failures mean the PIPELINE is broken, not that the test ticker
    itself has missing upstream data.
    """
    return "MSFT"


@pytest.fixture
def sentiment_agent_instance():
    """Real SentimentAgent, constructed the same way production code would.
    ASSUMPTION: SENTIMENT_SERVER_PATH above — adjust if your real path differs."""
    return SentimentAgent(server_script_path=SENTIMENT_SERVER_PATH)


@pytest.fixture
def manager_agent_instance(sentiment_agent_instance):
    """
    Real ManagerAgent, constructed with real sub-agent instances — matching
    its actual __init__ signature (research_agent, financial_agent,
    sentiment_agent, memory are all required, no defaults).

    Confirmed against memory/manager_memory.py: ManagerMemory()'s
    LongTermMemory.create() falls back to SUPABASE_URL/
    SUPABASE_SERVICE_ROLE_KEY env vars when no client is injected — this
    works in this project's environment since those are already configured
    (see the earlier `python -m rag.seed` run). If those env vars aren't
    set wherever this test suite runs, ManagerMemory() raises ValueError;
    inject a real or mock supabase_client here in that case.
    """
    return ManagerAgent(
        research_agent=ResearchAgent(),
        financial_agent=FinancialAnalystAgent(),
        sentiment_agent=sentiment_agent_instance,
        memory=ManagerMemory(),
    )