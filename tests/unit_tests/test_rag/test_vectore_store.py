"""
Tests for: rag/vector_store.py
Phase: 3 — RAG Pipeline (4th: parallel with graph_store.py)

Mocking strategy: supabase.create_client is mocked. The returned Client mock
has .table().upsert().execute() and .rpc().execute() chains mocked to return
controlled response objects (MagicMock with a `.data` attribute, matching
the real supabase-py response shape).
"""
from unittest.mock import patch, MagicMock
import pytest

from rag.vector_store import AlphaVectorStore, TemporalBoundaryUnsupported, VectorStoreConfig
from rag.loader import RawDocument
from rag.processor import AlphaProcessor


@pytest.fixture(autouse=True)
def explicit_legacy_schema(monkeypatch):
    monkeypatch.setenv("RAG_VECTOR_SCHEMA", "legacy")


@pytest.fixture
def mock_supabase_client():
    with patch("rag.vector_store.create_client") as mock_create:
        client = MagicMock()
        mock_create.return_value = client
        yield client


# ---------------------------------------------------------------------------
# __init__ — credential resolution
# ---------------------------------------------------------------------------

class TestInit:
    def test_uses_explicit_args_over_env(self, mock_supabase_client, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "env-url")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "env-key")
        with patch("rag.vector_store.create_client") as mock_create:
            AlphaVectorStore(supabase_url="explicit-url", supabase_key="explicit-key")
            mock_create.assert_called_once_with("explicit-url", "explicit-key")

    def test_falls_back_to_env_vars(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "env-url")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "env-key")
        with patch("rag.vector_store.create_client") as mock_create:
            AlphaVectorStore()
            mock_create.assert_called_once_with("env-url", "env-key")

    def test_missing_env_vars_raises_keyerror(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        with pytest.raises(KeyError):
            AlphaVectorStore()


# ---------------------------------------------------------------------------
# _to_row — mapping record -> DB column schema
# ---------------------------------------------------------------------------

class TestToRow:
    def test_maps_all_metadata_fields(self):
        record = {
            "embedding": [0.1, 0.2],
            "text": "chunk text",
            "metadata": {
                "content_hash": "ch", "url_hash": "uh", "ticker": "NVDA",
                "source_type": "news", "published_at_utc": "2024-01-01T00:00:00+00:00",
                "ingested_at": "2024-01-02T00:00:00+00:00", "chunk_index": 3,
                "url": "https://x.com", "title": "Title",
            },
        }
        row = AlphaVectorStore._to_row(record)
        expected = {
            "content_hash": "ch", "url_hash": "uh", "ticker": "NVDA",
            "source_type": "news", "published_at": "2024-01-01T00:00:00+00:00",
            "ingested_at": "2024-01-02T00:00:00+00:00", "chunk_index": 3,
            "url": "https://x.com", "title": "Title", "text": "chunk text",
            "embedding": [0.1, 0.2],
        }
        assert {key: row[key] for key in expected} == expected
        for field in (
            "fiscal_year", "fiscal_quarter", "period_start", "period_end",
            "duration_months", "filing_type", "accession_number",
            "metric_scope", "gaap_status",
        ):
            assert field in row
            assert row[field] is None

    def test_missing_metadata_defaults_gracefully(self):
        record = {"text": "x", "embedding": [0.0]}
        row = AlphaVectorStore._to_row(record)
        assert row["chunk_index"] == 0
        assert row["ticker"] is None


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------

class TestUpsert:
    def test_empty_records_returns_zero_without_calling_supabase(self, mock_supabase_client):
        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        result = store.upsert([])
        assert result == 0
        mock_supabase_client.table.assert_not_called()

    def test_upsert_calls_table_with_on_conflict_url_hash(self, mock_supabase_client):
        """on_conflict is "url_hash,chunk_index", not just "url_hash" — a
        single article produces many chunks sharing one url_hash, so the
        conflict key must include chunk_index or upserts fail with
        "ON CONFLICT DO UPDATE command cannot affect row a second time"
        (see the SQL schema comment at the top of vector_store.py)."""
        response = MagicMock()
        response.data = [{"id": 1}, {"id": 2}]
        mock_supabase_client.table.return_value.upsert.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        records = [
            {"text": "a", "embedding": [0.1], "metadata": {"url_hash": "h1"}},
            {"text": "b", "embedding": [0.2], "metadata": {"url_hash": "h2"}},
        ]
        count = store.upsert(records)

        assert count == 2
        mock_supabase_client.table.assert_called_with("alpha_documents")
        _, kwargs = mock_supabase_client.table.return_value.upsert.call_args
        assert kwargs["on_conflict"] == "url_hash,chunk_index"

    def test_legacy_batch_scopes_shared_article_hash_by_ticker(
        self, mock_supabase_client
    ):
        response = MagicMock()
        response.data = [{"id": 1}, {"id": 2}]
        mock_supabase_client.table.return_value.upsert.return_value.execute.return_value = response
        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        records = [
            {
                "text": "shared",
                "embedding": [0.1],
                "metadata": {
                    "url_hash": "shared-hash", "chunk_index": 0, "ticker": "NVDA",
                },
            },
            {
                "text": "shared",
                "embedding": [0.1],
                "metadata": {
                    "url_hash": "shared-hash", "chunk_index": 0, "ticker": "AMD",
                },
            },
        ]

        assert store.upsert(records) == 2

        rows = mock_supabase_client.table.return_value.upsert.call_args.args[0]
        assert len(rows) == 2
        assert {row["ticker"] for row in rows} == {"NVDA", "AMD"}
        assert len({row["url_hash"] for row in rows}) == 2
        assert all(len(row["url_hash"]) == 64 for row in rows)

    def test_upsert_returns_zero_when_response_data_is_none(self, mock_supabase_client):
        response = MagicMock()
        response.data = None
        mock_supabase_client.table.return_value.upsert.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        count = store.upsert([{"text": "a", "embedding": [0.1], "metadata": {}}])
        assert count == 0

    def test_v2_upsert_targets_evidence_scoped_revision2_constraint(self, mock_supabase_client):
        response = MagicMock()
        response.data = [{"chunk_id": "c"}]
        mock_supabase_client.table.return_value.upsert.return_value.execute.return_value = response
        store = AlphaVectorStore(
            supabase_url="u", supabase_key="k", config=VectorStoreConfig(schema="v2")
        )

        store.upsert([{"text": "x", "embedding": [0.1], "metadata": {"chunk_id": "c"}}])

        _, kwargs = mock_supabase_client.table.return_value.upsert.call_args
        assert kwargs["on_conflict"] == (
            "canonical_source_id,content_hash,document_version,document_id,"
            "evidence_id,dataset_version,chunk_index,chunk_hash,embedding_version"
        )

    def test_same_content_observations_satisfy_all_revision2_unique_identities(self):
        common = dict(
            title="immutable", content="same content", url="https://archive.invalid/shared",
            source_type="archive", ticker="NVDA", published_at="2024-01-10T00:00:00+00:00",
            source_id="a" * 64, revision="1",
        )
        documents = [
            RawDocument(**common, dataset_version="snapshot-v1", retrieved_at="2024-01-11T00:00:00+00:00", effective_from="2024-01-10T00:00:00+00:00"),
            RawDocument(**common, dataset_version="snapshot-v2", retrieved_at="2024-01-11T00:00:00+00:00", effective_from="2024-01-10T00:00:00+00:00"),
            RawDocument(**common, dataset_version="snapshot-v1", retrieved_at="2024-01-12T00:00:00+00:00", effective_from="2024-01-10T00:00:00+00:00"),
            RawDocument(**common, dataset_version="snapshot-v1", retrieved_at="2024-01-11T00:00:00+00:00", effective_from="2024-01-11T00:00:00+00:00"),
        ]
        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process(documents)
        rows = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata.update({"embedding_model": "fixture", "embedding_version": "weights-v1"})
            rows.append(AlphaVectorStore._to_v2_row({
                "text": chunk.text, "embedding": [0.1], "metadata": metadata,
            }))

        document_identity = {
            (
                row["canonical_source_id"], row["content_hash"], row["document_version"],
                row["document_id"], row["evidence_id"], row["dataset_version"],
                row["chunk_index"], row["chunk_hash"], row["embedding_version"],
            )
            for row in rows
        }
        evidence_identity = {
            (
                row["evidence_id"], row["dataset_version"], row["chunk_index"],
                row["chunk_hash"], row["embedding_version"],
            )
            for row in rows
        }
        assert len(rows) == 4
        assert len({row["chunk_id"] for row in rows}) == len(rows)
        assert len(document_identity) == len(rows)
        assert len(evidence_identity) == len(rows)


# ---------------------------------------------------------------------------
# hybrid_search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_calls_rpc_with_correct_params(self, mock_supabase_client):
        response = MagicMock()
        response.data = [{"id": 1, "rrf_score": 0.5}]
        mock_supabase_client.rpc.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        store.hybrid_search([0.1, 0.2], "query text", ticker="NVDA", days_back=7)

        call_name, call_params = mock_supabase_client.rpc.call_args[0]
        assert call_name == "alpha_hybrid_search"
        assert call_params["filter_ticker"] == "NVDA"
        assert call_params["days_back"] == 7
        assert call_params["query_text"] == "query text"

    def test_ticker_and_days_back_omitted_when_not_provided(self, mock_supabase_client):
        response = MagicMock()
        response.data = []
        mock_supabase_client.rpc.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        store.hybrid_search([0.1], "q")

        _, call_params = mock_supabase_client.rpc.call_args[0]
        assert "filter_ticker" not in call_params
        assert "days_back" not in call_params

    def test_score_threshold_filters_low_scoring_rows(self, mock_supabase_client):
        response = MagicMock()
        response.data = [
            {"id": 1, "rrf_score": 0.05},
            {"id": 2, "rrf_score": 0.5},
        ]
        mock_supabase_client.rpc.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        results = store.hybrid_search([0.1], "q", score_threshold=0.1)

        assert len(results) == 1
        assert results[0]["id"] == 2

    def test_limit_truncates_results_after_filtering(self, mock_supabase_client):
        response = MagicMock()
        response.data = [{"id": i, "rrf_score": 0.5} for i in range(10)]
        mock_supabase_client.rpc.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        results = store.hybrid_search([0.1], "q", limit=3)
        assert len(results) == 3

    def test_response_data_none_returns_empty_list(self, mock_supabase_client):
        response = MagicMock()
        response.data = None
        mock_supabase_client.rpc.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        results = store.hybrid_search([0.1], "q")
        assert results == []

    def test_row_missing_rrf_score_defaults_to_zero_and_is_filtered(self, mock_supabase_client):
        response = MagicMock()
        response.data = [{"id": 1}]  # no rrf_score key at all
        mock_supabase_client.rpc.return_value.execute.return_value = response

        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        results = store.hybrid_search([0.1], "q", score_threshold=0.01)
        assert results == []  # 0 < 0.01 threshold


class TestTemporalV2Boundary:
    def test_legacy_live_as_of_filters_before_local_vector_ranking(self):
        client = MagicMock()
        builder = MagicMock()
        client.table.return_value = builder
        for method in ("select", "eq", "lte", "gte", "order", "limit"):
            getattr(builder, method).return_value = builder
        builder.not_.is_.return_value = builder
        response = MagicMock()
        response.data = [{
            "id": 7,
            "url_hash": "a" * 64,
            "ticker": "MSFT",
            "source_type": "news",
            "published_at": "2026-08-24T13:02:00+00:00",
            "ingested_at": "2026-08-24T15:01:02+00:00",
            "chunk_index": 0,
            "url": "https://example.test/msft",
            "title": "Microsoft update",
            "text": "Microsoft demand improved.",
            "embedding": [1.0, 0.0],
        }]
        builder.execute.return_value = response
        store = AlphaVectorStore(
            client=client, config=VectorStoreConfig(schema="legacy")
        )

        rows = store.hybrid_search(
            [1.0, 0.0], "Microsoft demand", ticker="MSFT", days_back=31,
            as_of="2026-08-31T20:00:00Z",
            dataset_version="live-dataset-v1", historical=False,
        )

        assert len(rows) == 1
        assert rows[0]["dataset_version"] == "live-dataset-v1"
        assert rows[0]["retrieved_at"] == "2026-08-24T15:01:02+00:00"
        assert rows[0]["evidence_id"]
        builder.lte.assert_any_call("published_at", "2026-08-31T20:00:00+00:00")
        builder.lte.assert_any_call("ingested_at", "2026-08-31T20:00:00+00:00")
        builder.gte.assert_called_once_with(
            "published_at", "2026-07-31T20:00:00+00:00"
        )
        client.rpc.assert_not_called()

    def test_legacy_historical_query_fails_closed(self, mock_supabase_client):
        store = AlphaVectorStore(supabase_url="u", supabase_key="k")
        with pytest.raises(TemporalBoundaryUnsupported):
            store.hybrid_search([0.1], "q", ticker="NVDA", as_of="2024-02-01", dataset_version="v1", historical=True)
        mock_supabase_client.rpc.assert_not_called()

    def test_v2_filters_ticker_version_and_future_before_return(self, mock_supabase_client):
        response = MagicMock()
        response.data = [
            {"chunk_id": "ok", "ticker": "NVDA", "dataset_version": "v1", "published_at": "2024-01-20", "effective_from": "2024-01-20", "effective_to": None, "similarity": 0.5, "metadata": {}},
            {"chunk_id": "future", "ticker": "NVDA", "dataset_version": "v1", "published_at": "2024-02-02", "effective_from": "2024-02-02", "effective_to": None, "similarity": 0.99, "metadata": {}},
            {"chunk_id": "ticker", "ticker": "AMD", "dataset_version": "v1", "published_at": "2024-01-20", "effective_from": "2024-01-20", "effective_to": None, "similarity": 0.98, "metadata": {}},
            {"chunk_id": "version", "ticker": "NVDA", "dataset_version": "v2", "published_at": "2024-01-20", "effective_from": "2024-01-20", "effective_to": None, "similarity": 0.97, "metadata": {}},
        ]
        mock_supabase_client.rpc.return_value.execute.return_value = response
        store = AlphaVectorStore(supabase_url="u", supabase_key="k", config=VectorStoreConfig(schema="v2"))
        rows = store.hybrid_search([0.1], "q", ticker="NVDA", as_of="2024-02-01", dataset_version="v1", historical=True)
        assert [row["chunk_id"] for row in rows] == ["ok"]
        rpc, params = mock_supabase_client.rpc.call_args[0]
        assert rpc == "alpha_hybrid_search_v2"
        assert params["match_ticker"] == "NVDA"
        assert params["query_dataset_version"] == "v1"
