"""
Tests for: rag/graph_store.py
Phase: 3 — RAG Pipeline (4th: parallel with vector_store.py)

Mocking strategy:
  - anthropic.AsyncAnthropic is mocked for entity/relation extraction.
  - neo4j.GraphDatabase is mocked for connect()/upsert_batch()/close().
  - connect() is tested separately from extract_batch()/upsert_batch() since
    the class explicitly keeps network I/O out of __init__ (testability by
    design — we honor that and never call real connect() in unit tests).
"""
import json
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from rag.graph_store import AlphaGraphStore, Entity, Relation, GraphDocument
from rag.loader import RawDocument


def make_raw_doc(title="NVIDIA beats earnings", content="NVIDIA reported strong results " * 5,
                  url="https://x.com/1", ticker="NVDA"):
    return RawDocument(title=title, content=content, url=url,
                       source_type="news", ticker=ticker,
                       published_at="2024-03-15T14:32:00+00:00")


# ---------------------------------------------------------------------------
# __init__ — credential validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_missing_anthropic_key_raises_valueerror(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AlphaGraphStore()

    def test_explicit_api_key_accepted(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        store = AlphaGraphStore(anthropic_api_key="explicit-key")
        assert store._api_key == "explicit-key"

    def test_no_network_io_in_init(self, monkeypatch):
        """connect() must be called explicitly — __init__ should not touch
        anthropic.AsyncAnthropic or neo4j.GraphDatabase at all."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        assert store._claude is None
        assert store._driver is None


# ---------------------------------------------------------------------------
# connect() — Neo4j optionality
# ---------------------------------------------------------------------------

class TestConnect:
    @patch("anthropic.AsyncAnthropic")
    def test_connect_without_neo4j_credentials_disables_graph_writes(
        self, mock_anthropic_cls, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        store = AlphaGraphStore()
        store.connect()

        assert store._claude is not None
        assert store._driver is None

    @patch("rag.graph_store.AlphaGraphStore._ensure_constraints")
    @patch("neo4j.GraphDatabase")
    @patch("anthropic.AsyncAnthropic")
    def test_connect_with_credentials_initialises_driver(
        self, mock_anthropic_cls, mock_graphdb, mock_ensure, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        mock_driver = MagicMock()
        mock_graphdb.driver.return_value = mock_driver

        store = AlphaGraphStore(
            neo4j_uri="bolt://localhost", neo4j_user="neo4j", neo4j_password="pw",
        )
        store.connect()

        mock_graphdb.driver.assert_called_once_with(
            "bolt://localhost", auth=("neo4j", "pw")
        )
        mock_driver.verify_connectivity.assert_called_once()
        assert store._driver is mock_driver

    @patch("neo4j.GraphDatabase")
    @patch("anthropic.AsyncAnthropic")
    def test_connect_neo4j_failure_disables_graph_writes_gracefully(
        self, mock_anthropic_cls, mock_graphdb, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        mock_graphdb.driver.side_effect = ConnectionError("neo4j down")

        store = AlphaGraphStore(
            neo4j_uri="bolt://localhost", neo4j_password="pw",
        )
        store.connect()  # should NOT raise

        assert store._driver is None


# ---------------------------------------------------------------------------
# extract_batch / _extract_one — Claude entity extraction
# ---------------------------------------------------------------------------

class TestExtractBatch:
    @pytest.mark.asyncio
    async def test_short_documents_are_skipped(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._claude = AsyncMock()

        short_doc = make_raw_doc(title="x", content="y")  # < 80 chars combined
        results = await store.extract_batch([short_doc])

        assert results == []
        store._claude.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_document_triggers_extraction(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._claude = AsyncMock()
        response = MagicMock()
        # _extract_one prepends "{" to this text (since the API call uses a
        # "{"-prefill trick to force raw JSON without markdown fences) — so
        # the mock must supply the JSON body WITHOUT its own leading brace,
        # matching what the real (prefilled) API response would actually
        # contain. Supplying the full json.dumps(...) string here (with its
        # own leading "{") would double the opening brace once _extract_one
        # re-adds its own, leaving the brace-counter unable to find a
        # balanced object.
        full_json = json.dumps({
            "entities": [{"name": "NVIDIA", "type": "Company", "ticker": "NVDA", "description": "GPU maker"}],
            "relations": [],
        })
        response.content = [MagicMock(text=full_json[1:])]  # strip the leading "{"
        store._claude.messages.create.return_value = response

        doc = make_raw_doc()
        results = await store.extract_batch([doc])

        assert len(results) == 1
        assert results[0].entities[0].name == "NVIDIA"
        store._claude.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_claude_failure_returns_empty_graph_doc_not_raises(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._claude = AsyncMock()
        store._claude.messages.create.side_effect = RuntimeError("API down")

        doc = make_raw_doc()
        results = await store.extract_batch([doc])

        assert len(results) == 1
        assert results[0].entities == []
        assert results[0].source_url == doc.url

    @pytest.mark.asyncio
    async def test_max_chars_per_doc_truncates_input_text(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._claude = AsyncMock()
        response = MagicMock()
        response.content = [MagicMock(text='"entities": [], "relations": []}')]
        store._claude.messages.create.return_value = response

        doc = make_raw_doc(content="x" * 5000)
        await store.extract_batch([doc], max_chars_per_doc=100)

        sent_prompt = store._claude.messages.create.call_args.kwargs["messages"][0]["content"]
        # text block in the prompt should be capped near 100 chars (plus prompt template text)
        assert "x" * 101 not in sent_prompt


# ---------------------------------------------------------------------------
# _parse_graph_doc — JSON parsing + validation
# ---------------------------------------------------------------------------

class TestParseGraphDoc:
    def setup_method(self):
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", "key")
        self.store = AlphaGraphStore()

    def test_valid_json_parsed_correctly(self):
        raw = json.dumps({
            "entities": [{"name": "NVIDIA", "type": "Company", "ticker": "nvda", "description": "d"}],
            "relations": [{"source": "NVIDIA", "target": "AMD", "rel_type": "competes_with",
                           "weight": 0.9, "evidence": "they compete"}],
        })
        doc = self.store._parse_graph_doc(raw, source_url="https://x.com", ticker="NVDA")
        assert doc.entities[0].name == "NVIDIA"
        assert doc.entities[0].ticker == "NVDA"  # uppercased
        assert doc.relations[0].rel_type == "COMPETES_WITH"  # uppercased

    def test_invalid_json_returns_empty_graph_doc(self):
        doc = self.store._parse_graph_doc("not json at all", source_url="u", ticker="NVDA")
        assert doc.entities == []
        assert doc.relations == []

    def test_markdown_fenced_json_is_cleaned(self):
        raw = "```json\n" + json.dumps({"entities": [], "relations": []}) + "\n```"
        doc = self.store._parse_graph_doc(raw, source_url="u", ticker="NVDA")
        assert doc.entities == []

    def test_invalid_entity_type_defaults_to_company(self):
        raw = json.dumps({
            "entities": [{"name": "Mystery Thing", "type": "NotARealType"}],
            "relations": [],
        })
        doc = self.store._parse_graph_doc(raw, source_url="u", ticker="NVDA")
        assert doc.entities[0].type == "Company"

    def test_invalid_rel_type_defaults_to_related_to(self):
        raw = json.dumps({
            "entities": [],
            "relations": [{"source": "A", "target": "B", "rel_type": "MADE_UP_TYPE"}],
        })
        doc = self.store._parse_graph_doc(raw, source_url="u", ticker="NVDA")
        assert doc.relations[0].rel_type == "RELATED_TO"

    def test_entity_with_empty_name_is_skipped(self):
        raw = json.dumps({"entities": [{"name": "", "type": "Company"}], "relations": []})
        doc = self.store._parse_graph_doc(raw, source_url="u", ticker="NVDA")
        assert doc.entities == []

    def test_relation_missing_source_or_target_is_skipped(self):
        raw = json.dumps({
            "entities": [],
            "relations": [{"source": "", "target": "B", "rel_type": "RELATED_TO"}],
        })
        doc = self.store._parse_graph_doc(raw, source_url="u", ticker="NVDA")
        assert doc.relations == []

    def test_description_and_evidence_truncated_to_200_chars(self):
        raw = json.dumps({
            "entities": [{"name": "X", "type": "Company", "description": "y" * 500}],
            "relations": [],
        })
        doc = self.store._parse_graph_doc(raw, source_url="u", ticker="NVDA")
        assert len(doc.entities[0].description) == 200


# ---------------------------------------------------------------------------
# upsert_batch — Neo4j write transactions
# ---------------------------------------------------------------------------

class TestUpsertBatch:
    def test_no_driver_returns_skipped_summary(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()  # _driver is None (connect() never called)

        graph_docs = [GraphDocument(entities=[Entity(name="X", type="Company")])]
        summary = store.upsert_batch(graph_docs)

        assert summary == {"nodes_merged": 0, "rels_merged": 0, "skipped": 1}

    def test_upserts_entities_and_relations_with_driver(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        store._driver = mock_driver

        graph_doc = GraphDocument(
            entities=[Entity(name="NVIDIA", type="Company")],
            relations=[Relation(source="NVIDIA", target="AMD", rel_type="COMPETES_WITH")],
            source_url="https://x.com",
        )
        summary = store.upsert_batch([graph_doc])

        assert summary["nodes_merged"] == 1
        assert summary["rels_merged"] == 1
        assert mock_session.execute_write.call_count == 2

    def test_empty_graph_docs_list_returns_zero_counts(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = MagicMock()
        store._driver = mock_driver

        summary = store.upsert_batch([])
        assert summary == {"nodes_merged": 0, "rels_merged": 0, "skipped": 0}


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_calls_driver_close_when_present(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._driver = MagicMock()
        store.close()
        store._driver.close.assert_called_once()

    def test_close_is_noop_when_no_driver(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store.close()  # should not raise