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
from dataclasses import replace
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from rag.graph_store import (
    AlphaGraphStore,
    Entity,
    Relation,
    GraphDocument,
    GraphUpsertError,
    _prepare_temporal_document,
)
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
        assert results[0].source_url == ""
        assert results[0].is_quarantined is True
        assert store.last_lifecycle["status"] == "degraded"
        assert store.last_lifecycle["error_code"] == "GRAPH_EXTRACTION_FAILED"
        assert store.last_lifecycle["failure_count"] == 1
        assert store.last_lifecycle["latency_ms"] >= 0

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
            write_mode="legacy",
            provenance="legacy_explicit",
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


class TestTemporalEvidenceV2:
    def test_entity_identity_is_stable_and_type_sensitive(self):
        assert Entity("NVIDIA", "Company").entity_id == Entity("nvidia", "Company").entity_id
        assert Entity("NVIDIA", "Company").entity_id != Entity("NVIDIA", "Product").entity_id

    @pytest.mark.asyncio
    async def test_extract_batch_reuses_stage03_vector_identities(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._extract_one = AsyncMock(return_value=GraphDocument(
            entities=[Entity("NVIDIA", "Company")],
            source_url="https://x.com/1",
            ticker="NVDA",
        ))
        raw = RawDocument(
            title="NVIDIA temporal evidence",
            content="A sufficiently long deterministic source body. " * 5,
            url="https://x.com/1",
            source_type="archive",
            ticker="NVDA",
            published_at="2024-01-30T15:00:00+00:00",
            retrieved_at="2024-01-30T16:00:00+00:00",
            effective_from="2024-01-30T15:00:00+00:00",
            dataset_version="pit-2024-01",
            source_id="source-1",
            revision="2",
        )

        graph_doc = (await store.extract_batch([raw]))[0]

        from rag.processor import AlphaProcessor
        vector = AlphaProcessor().process([raw])[0].metadata
        assert graph_doc.document_id == vector["document_id"]
        assert graph_doc.chunk_id == vector["chunk_id"]
        assert graph_doc.evidence_id == vector["evidence_id"]
        assert graph_doc.materialized_at == vector["ingested_at"]
        assert graph_doc.write_mode == "temporal_v2"
        assert graph_doc.provenance == "stage03_temporal"
        assert graph_doc.temporal_time_complete
        assert not graph_doc.is_quarantined

    def test_temporal_upsert_retains_observation_without_aggregate_edge(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        session = MagicMock()
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver
        doc = GraphDocument(
            entities=[Entity("NVIDIA", "Company"), Entity("TSMC", "Company")],
            relations=[Relation("NVIDIA", "TSMC", "SUPPLIES_TO", evidence="TSMC supplies NVIDIA")],
            source_url="https://example.test/evidence",
            ticker="NVDA",
            document_id="doc-1",
            chunk_id="chunk-1",
            evidence_id="evidence-1",
            canonical_source_id="source-1",
            content_hash="content-1",
            document_version="1",
            chunk_hash="chunk-hash-1",
            dataset_version="pit-2024-01",
            published_at="2024-01-30T15:00:00Z",
            retrieved_at="2024-01-30T16:00:00Z",
            effective_from="2024-01-30T15:00:00Z",
            materialized_at="2024-01-30T16:00:00Z",
            write_mode="temporal_v2",
            provenance="stage03_temporal",
        )

        summary = store.upsert_batch([doc])

        assert summary == {"nodes_merged": 2, "rels_merged": 1, "skipped": 0}
        temporal_call = session.execute_write.call_args_list[-1]
        assert temporal_call.args[0] == store._merge_temporal_observation
        tx = MagicMock()
        store._merge_temporal_observation(tx, doc.relations[0], doc)
        cypher = tx.run.call_args.args[0]
        assert "TemporalRelationObservationV2" in cypher
        assert "SUPPORTS_OBSERVATION" in cypher
        assert "ON MATCH SET r.weight" not in cypher

    def test_unknown_time_is_persisted_quarantined(self):
        tx = MagicMock()
        doc = GraphDocument(
            ticker="NVDA", document_id="d", chunk_id="c", evidence_id="e",
            canonical_source_id="s", content_hash="h", document_version="1",
            chunk_hash="ch", dataset_version="v", published_at=None,
        )
        AlphaGraphStore._merge_temporal_observation(
            tx, Relation("NVIDIA", "TSMC", "SUPPLIES_TO"), doc
        )
        assert tx.run.call_args.kwargs["is_quarantined"] is True


class TestGraphLifecyclePrivacy:
    def test_upsert_success_has_measured_compact_lifecycle(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        driver.session.return_value.__enter__.return_value = MagicMock()
        store._driver = driver

        assert store.upsert_batch([]) == {
            "nodes_merged": 0, "rels_merged": 0, "skipped": 0,
        }
        event = store.last_lifecycle
        assert event["status"] == "complete"
        assert event["latency_ms"] >= 0
        assert event["document_count"] == 0
        assert event["resume"] == "not_applicable"
        assert event["recovery"] == "caller_retry_deterministic_merge"

    def test_partial_upsert_failure_is_sanitized_and_retry_safe(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        session = MagicMock()
        session.execute_write.side_effect = RuntimeError(
            "SECRET_TOKEN raw evidence https://private.test/path"
        )
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver
        doc = GraphDocument(
            entities=[Entity("NVIDIA", "Company")],
            write_mode="legacy",
            provenance="legacy_explicit",
        )

        with pytest.raises(GraphUpsertError) as raised:
            store.upsert_batch([doc])

        assert str(raised.value) == "GRAPH_UPSERT_FAILED"
        assert "SECRET_TOKEN" not in caplog.text
        assert "private.test" not in caplog.text
        event = store.last_lifecycle
        assert event["status"] == "failed"
        assert event["error_code"] == "GRAPH_UPSERT_FAILED"
        assert event["degraded"] is True
        assert event["latency_ms"] >= 0
        assert event["recovery"] == "caller_retry_deterministic_merge"

    @pytest.mark.asyncio
    async def test_extraction_exception_never_exposes_secret_url_or_raw_exception(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._claude = MagicMock()
        store._claude.messages.create = AsyncMock(side_effect=RuntimeError(
            "SECRET_TOKEN raw evidence https://private.test/provider"
        ))
        with patch("rag.graph_store.sentry_enabled", return_value=True), \
             patch("rag.graph_store.sentry_sdk.capture_message") as capture:
            doc = await store._extract_one(
                "RAW_EVIDENCE_SENTINEL", "NVDA", "https://private.test/source"
            )

        assert doc.source_url == ""
        assert doc.is_quarantined is True
        assert "SECRET_TOKEN" not in caplog.text
        assert "RAW_EVIDENCE_SENTINEL" not in caplog.text
        assert "private.test" not in caplog.text
        capture.assert_called_once_with("GRAPH_EXTRACTION_FAILED", level="error")
        assert "SECRET_TOKEN" not in str(capture.call_args)


def make_temporal_graph_doc(**overrides):
    base = GraphDocument(
        entities=[Entity("NVIDIA", "Company"), Entity("TSMC", "Company")],
        relations=[Relation("NVIDIA", "TSMC", "SUPPLIES_TO", evidence="supply evidence")],
        source_url="https://example.test/evidence",
        ticker="NVDA",
        document_id="doc-1",
        chunk_id="chunk-1",
        evidence_id="evidence-1",
        canonical_source_id="source-1",
        content_hash="content-1",
        document_version="1",
        chunk_hash="chunk-hash-1",
        chunk_index=0,
        dataset_version="pit-2024-01",
        published_at="2024-01-30T15:00:00Z",
        retrieved_at="2024-01-30T16:00:00+00:00",
        filed_at=None,
        effective_from="2024-01-30T15:00:00-00:00",
        effective_to=None,
        materialized_at="2024-01-30T16:00:00Z",
        write_mode="temporal_v2",
        provenance="stage03_temporal",
    )
    return replace(base, **overrides)


class TestAttempt2ReviewBlockers:
    @pytest.mark.parametrize("missing_field", [
        "document_id", "chunk_id", "evidence_id", "canonical_source_id",
        "content_hash", "document_version", "chunk_hash", "ticker", "dataset_version",
    ])
    def test_each_missing_v2_identity_skips_before_any_write(
        self, monkeypatch, missing_field
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver

        summary = store.upsert_batch([
            make_temporal_graph_doc(**{missing_field: ""})
        ])

        assert summary == {"nodes_merged": 0, "rels_merged": 0, "skipped": 1}
        session.execute_write.assert_not_called()
        assert store.last_lifecycle["status"] == "degraded"
        assert store.last_lifecycle["rejected_count"] == 1
        assert store.last_lifecycle["rejection_codes"] == [
            "GRAPH_V2_IDENTITY_INCOMPLETE"
        ]
        assert store.last_lifecycle["error_code"] == "GRAPH_V2_WRITE_REJECTED"

    @pytest.mark.parametrize(
        ("overrides", "expected_code"),
        [
            ({"materialized_at": None}, "GRAPH_V2_MATERIALIZED_AT_MISSING"),
            ({"published_at": None}, "GRAPH_V2_TEMPORAL_REQUIRED"),
            ({"retrieved_at": None}, "GRAPH_V2_TEMPORAL_REQUIRED"),
            ({"effective_from": None}, "GRAPH_V2_TEMPORAL_REQUIRED"),
            ({"published_at": "not-a-time"}, "GRAPH_V2_TEMPORAL_INVALID"),
            ({"retrieved_at": "not-a-time"}, "GRAPH_V2_TEMPORAL_INVALID"),
            ({"filed_at": "not-a-time"}, "GRAPH_V2_TEMPORAL_INVALID"),
            ({"effective_from": "not-a-time"}, "GRAPH_V2_TEMPORAL_INVALID"),
            ({"effective_to": "not-a-time"}, "GRAPH_V2_TEMPORAL_INVALID"),
            ({"materialized_at": "not-a-time"}, "GRAPH_V2_TEMPORAL_INVALID"),
            (
                {
                    "effective_from": "2024-01-30T15:00:00Z",
                    "effective_to": "2024-01-30T14:59:59Z",
                },
                "GRAPH_V2_EFFECTIVE_INTERVAL_INVALID",
            ),
        ],
    )
    def test_missing_invalid_or_reversed_temporal_values_fail_closed(
        self, monkeypatch, overrides, expected_code
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver

        summary = store.upsert_batch([make_temporal_graph_doc(**overrides)])

        assert summary["skipped"] == 1
        session.execute_write.assert_not_called()
        assert store.last_lifecycle["rejection_codes"] == [expected_code]

    def test_quarantined_temporal_document_never_writes_any_namespace(
        self, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver

        summary = store.upsert_batch([
            make_temporal_graph_doc(is_quarantined=True)
        ])

        assert summary == {"nodes_merged": 0, "rels_merged": 0, "skipped": 1}
        session.execute_write.assert_not_called()
        assert store.last_lifecycle["rejection_codes"] == ["GRAPH_V2_QUARANTINED"]

    @pytest.mark.asyncio
    async def test_quarantined_extraction_failure_stays_quarantined_and_never_writes(
        self, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        store._claude = AsyncMock()
        store._claude.messages.create.side_effect = RuntimeError("provider failure")
        raw = RawDocument(
            title="NVIDIA temporal evidence",
            content="A sufficiently long deterministic source body. " * 5,
            url="https://example.test/private",
            source_type="archive",
            ticker="NVDA",
            published_at="2024-01-30T15:00:00Z",
            retrieved_at="2024-01-30T16:00:00Z",
            effective_from="2024-01-30T15:00:00Z",
            dataset_version="pit-2024-01",
            source_id="source-1",
        )
        extracted = (await store.extract_batch([raw]))[0]
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver

        summary = store.upsert_batch([extracted])

        assert extracted.is_quarantined is True
        assert extracted.provenance == "stage03_temporal"
        assert summary["skipped"] == 1
        session.execute_write.assert_not_called()
        assert store.last_lifecycle["rejection_codes"] == ["GRAPH_V2_QUARANTINED"]

    def test_v2_provenance_cannot_force_legacy_mode(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver

        store.upsert_batch([make_temporal_graph_doc(write_mode="legacy")])

        session.execute_write.assert_not_called()
        assert store.last_lifecycle["rejection_codes"] == ["GRAPH_WRITE_MODE_INVALID"]

    def test_explicit_legacy_mode_preserves_legacy_compatibility(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        store = AlphaGraphStore()
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__.return_value = session
        store._driver = driver
        legacy = GraphDocument(
            entities=[Entity("NVIDIA", "Company")],
            relations=[Relation("NVIDIA", "AMD", "COMPETES_WITH")],
            write_mode="legacy",
            provenance="legacy_explicit",
        )

        summary = store.upsert_batch([legacy])

        assert summary == {"nodes_merged": 1, "rels_merged": 1, "skipped": 0}
        assert session.execute_write.call_args_list[0].args[0] == store._merge_entity
        assert session.execute_write.call_args_list[1].args[0] == store._merge_relation

    def test_temporal_values_normalize_once_without_now_substitution(self):
        prepared, code = _prepare_temporal_document(make_temporal_graph_doc())

        assert code is None
        assert prepared is not None
        assert prepared.published_at == "2024-01-30T15:00:00+00:00"
        assert prepared.retrieved_at == "2024-01-30T16:00:00+00:00"
        assert prepared.effective_from == "2024-01-30T15:00:00+00:00"
        assert prepared.filed_at is None
        assert prepared.effective_to is None
        assert prepared.materialized_at == "2024-01-30T16:00:00+00:00"

    def test_write_shape_uses_null_safe_neo4j_datetime_for_every_time(self):
        prepared, code = _prepare_temporal_document(make_temporal_graph_doc())
        assert code is None and prepared is not None
        tx = MagicMock()

        AlphaGraphStore._merge_temporal_observation(
            tx, prepared.relations[0], prepared
        )

        cypher = tx.run.call_args.args[0]
        for field_name in (
            "published_at", "retrieved_at", "filed_at", "effective_from",
            "effective_to", "materialized_at",
        ):
            exact = (
                "e." + field_name + "=CASE WHEN $" + field_name
                + " IS NULL THEN null ELSE datetime($" + field_name + ") END"
            )
            assert exact in cypher
        kwargs = tx.run.call_args.kwargs
        assert kwargs["filed_at"] is None
        assert kwargs["effective_to"] is None
        assert kwargs["materialized_at"] == "2024-01-30T16:00:00+00:00"
        assert "timestamp()" not in cypher

    def test_materialized_at_is_create_only_and_stable_on_idempotent_retry(self):
        prepared, code = _prepare_temporal_document(make_temporal_graph_doc())
        assert code is None and prepared is not None
        tx = MagicMock()

        AlphaGraphStore._merge_temporal_observation(tx, prepared.relations[0], prepared)
        AlphaGraphStore._merge_temporal_observation(tx, prepared.relations[0], prepared)

        first, second = tx.run.call_args_list
        assert first.kwargs["materialized_at"] == second.kwargs["materialized_at"]
        assert first.kwargs["evidence_id"] == second.kwargs["evidence_id"]
        assert "ON CREATE SET" in first.args[0]
        assert "ON MATCH SET e.materialized_at" not in first.args[0]
