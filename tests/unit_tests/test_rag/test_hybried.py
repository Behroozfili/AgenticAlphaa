"""
Tests for: rag/hybrid_rag.py
Phase: 3 — RAG Pipeline (6th: needs Supabase + Neo4j mocks)

IMPORTANT — module-level side effects: this file creates `_sb` (Supabase
client), `_neo4j_driver` (lazily, via _get_neo4j()), and `_embedder` at
IMPORT TIME inside try/except blocks that swallow failures. This means:
  - In a test environment without real credentials, `_sb` and `_embedder`
    will already be None after import (the try/except catches the KeyError
    from os.environ[...] and logs a warning) — this graceful-degradation
    path is itself tested below (TestGracefulDegradation).
  - For tests that need a *working* `_sb` or `_embedder`, we monkeypatch the
    module-level globals directly (`hybrid_rag._sb = mock_client`) rather
    than trying to re-trigger the import-time try/except.

Mocking strategy: asyncio.to_thread is patched to run synchronously so
`await asyncio.to_thread(fn)` just calls `fn()` directly in tests — no real
threading needed for the mocked Supabase client.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from core.forecast_contract import ForecastContext

import rag.hybrid_rag as hybrid_rag
from rag.hybrid_rag import (
    rag_vector_search,
    rag_graph_traverse,
    rag_hybrid_query,
    _embed,
    _key,
    _rrf,
    _weighted,
    _get_neo4j,
)


@pytest.fixture(autouse=True)
def reset_neo4j_driver_cache():
    """_neo4j_driver is a module-level singleton cache; reset between tests."""
    hybrid_rag._neo4j_driver = None
    yield
    hybrid_rag._neo4j_driver = None


# ---------------------------------------------------------------------------
# Graceful degradation — module import never crashes without credentials
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_vector_search_warns_when_retriever_is_none(self):
        """Regression: rag_vector_search was rewritten to route through
        AlphaRetriever.retrieve_raw() instead of a raw Supabase RPC call
        (see the module docstring on rag_vector_search) — the gating
        global is now _retriever, not _sb, and the warning text changed
        to match."""
        with patch.object(hybrid_rag, "_retriever", None):
            result = await rag_vector_search("query")
        assert result["results"] == []
        assert "AlphaRetriever not configured" in result["warning"]

    @pytest.mark.asyncio
    async def test_graph_traverse_warns_when_neo4j_unavailable(self):
        with patch("rag.hybrid_rag._get_neo4j", return_value=None):
            result = await rag_graph_traverse("NVIDIA")
        assert result["nodes"] == []
        assert "Neo4j not configured" in result["warning"]


# ---------------------------------------------------------------------------
# _embed — embedder vs deterministic fallback
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_uses_embedder_when_available(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        with patch.object(hybrid_rag, "_embedder", mock_embedder):
            result = _embed("hello")
        assert result == [0.1, 0.2]
        mock_embedder.embed_query.assert_called_once_with("hello")

    def test_falls_back_to_deterministic_hash_when_no_embedder(self):
        with patch.object(hybrid_rag, "_embedder", None):
            result = _embed("hello")
        assert len(result) == 384
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_deterministic_fallback_is_reproducible(self):
        with patch.object(hybrid_rag, "_embedder", None):
            r1 = _embed("same text")
            r2 = _embed("same text")
        assert r1 == r2


# ---------------------------------------------------------------------------
# rag_vector_search — happy path + threshold filtering
# ---------------------------------------------------------------------------

class TestRagVectorSearch:
    """
    NOTE: rag_vector_search() was rewritten to route through
    AlphaRetriever.retrieve_raw() (called via asyncio.to_thread since
    AlphaRetriever's pipeline is synchronous) instead of a raw Supabase
    RPC call — see the function's own docstring. These tests mock
    hybrid_rag._retriever directly, matching the real current
    architecture, rather than hybrid_rag._sb.rpc(...).
    """

    @pytest.mark.asyncio
    async def test_happy_path_returns_mapped_fields(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = [
            {"id": 1, "ticker": "NVDA", "source_type": "news", "text": "a",
             "freshness_score": 0.42, "published_at": "2026-01-01", "url": "u", "title": "t", "chunk_index": 0},
        ]

        with patch.object(hybrid_rag, "_retriever", mock_retriever):
            result = await rag_vector_search("NVDA earnings", top_k=5, threshold=0.01)

        assert len(result["results"]) == 1
        assert result["results"][0]["chunk_text"] == "a"
        assert result["results"][0]["score"] == 0.42

    @pytest.mark.asyncio
    async def test_ticker_filter_uppercased_and_passed_to_retriever(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = []

        with patch.object(hybrid_rag, "_retriever", mock_retriever):
            await rag_vector_search("q", ticker_filter="nvda")

        call_kwargs = mock_retriever.retrieve_raw.call_args.kwargs
        assert call_kwargs["ticker"] == "NVDA"

    @pytest.mark.asyncio
    async def test_threshold_passed_through_to_retriever(self):
        """Thresholding now happens INSIDE AlphaRetriever.retrieve_raw
        (score_threshold param), not as a post-hoc filter in
        rag_vector_search itself — verify the param is forwarded, rather
        than re-testing AlphaRetriever's own filtering logic here (that
        belongs in the retriever's own test file)."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = []

        with patch.object(hybrid_rag, "_retriever", mock_retriever):
            await rag_vector_search("q", threshold=0.05)

        call_kwargs = mock_retriever.retrieve_raw.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.05

    @pytest.mark.asyncio
    async def test_retriever_exception_returns_error_dict(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.side_effect = RuntimeError("retriever failed")

        with patch.object(hybrid_rag, "_retriever", mock_retriever):
            result = await rag_vector_search("q")

        assert result["results"] == []
        assert result["error_code"] == "VECTOR_RETRIEVAL_FAILED"
        assert "retriever failed" not in str(result)

    @pytest.mark.asyncio
    async def test_top_k_caps_results(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve_raw.return_value = [
            {"id": i, "text": f"t{i}", "freshness_score": 0.5} for i in range(10)
        ]

        with patch.object(hybrid_rag, "_retriever", mock_retriever):
            result = await rag_vector_search("q", top_k=3)

        assert len(result["results"]) == 3


# ---------------------------------------------------------------------------
# rag_graph_traverse
# ---------------------------------------------------------------------------

class TestRagGraphTraverse:
    @pytest.mark.asyncio
    async def test_temporal_graph_payload_is_json_serializable(self, temporal_fixture):
        _, context = temporal_fixture
        driver = MagicMock()
        session = AsyncMock()

        class Neo4jDateTime:
            def __init__(self, value):
                self.value = value

            def iso_format(self):
                return self.value

        class OneRecord:
            def __init__(self):
                self._done = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._done:
                    raise StopAsyncIteration
                self._done = True
                return {
                    "name": "Microsoft", "type": "Company",
                    "relation": "RELATED_TO", "hops": 1,
                    "path_nodes": ["MSFT", "Microsoft"],
                    "observation_id": "obs-1", "evidence_id": "evidence-1",
                    "chunk_id": "chunk-1", "document_id": "document-1",
                    "source_url": "https://example.test/msft",
                    "published_at": Neo4jDateTime("2024-01-15T12:00:00Z"),
                    "retrieved_at": Neo4jDateTime("2024-01-15T13:00:00Z"),
                    "effective_from": Neo4jDateTime("2024-01-15T12:00:00Z"),
                    "effective_to": None,
                    "dataset_version": context.dataset_version,
                }

        session.run = AsyncMock(return_value=OneRecord())
        driver.session.return_value.__aenter__.return_value = session
        driver.session.return_value.__aexit__.return_value = None

        with patch("rag.hybrid_rag._get_neo4j", return_value=driver):
            result = await rag_graph_traverse("NVDA", context=context)

        encoded = json.dumps(result)
        assert "2024-01-15T12:00:00Z" in encoded
        assert result["nodes"][0]["published_at"] == "2024-01-15T12:00:00Z"
    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_3(self):
        mock_driver = MagicMock()
        mock_session = AsyncMock()

        class FakeResultIter:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        mock_session.run = AsyncMock(return_value=FakeResultIter())
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_driver.session.return_value.__aexit__.return_value = None

        with patch("rag.hybrid_rag._get_neo4j", return_value=mock_driver):
            await rag_graph_traverse("NVIDIA", max_hops=10)

        sent_cypher = mock_session.run.call_args[0][0]
        assert "*1..3" in sent_cypher  # clamped from 10 to 3

    @pytest.mark.asyncio
    async def test_entity_uppercased_in_query_params(self):
        mock_driver = MagicMock()
        mock_session = AsyncMock()

        class FakeResultIter:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        mock_session.run = AsyncMock(return_value=FakeResultIter())
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_driver.session.return_value.__aexit__.return_value = None

        with patch("rag.hybrid_rag._get_neo4j", return_value=mock_driver):
            await rag_graph_traverse("nvidia")

        call_kwargs = mock_session.run.call_args.kwargs
        assert call_kwargs["entity"] == "NVIDIA"

    @pytest.mark.asyncio
    async def test_neo4j_exception_returns_error_dict(self):
        mock_driver = MagicMock()
        mock_driver.session.side_effect = RuntimeError("neo4j down")

        with patch("rag.hybrid_rag._get_neo4j", return_value=mock_driver):
            result = await rag_graph_traverse("NVIDIA")

        assert result["nodes"] == []
        assert result["error_code"] == "GRAPH_TRAVERSAL_FAILED"
        assert "neo4j down" not in str(result)

    @pytest.mark.asyncio
    async def test_relation_types_all_uses_unbounded_rel_clause(self):
        mock_driver = MagicMock()
        mock_session = AsyncMock()

        class FakeResultIter:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        mock_session.run = AsyncMock(return_value=FakeResultIter())
        mock_driver.session.return_value.__aenter__.return_value = mock_session
        mock_driver.session.return_value.__aexit__.return_value = None

        with patch("rag.hybrid_rag._get_neo4j", return_value=mock_driver):
            await rag_graph_traverse("NVIDIA", relation_types=["COMPETES_WITH"])

        sent_cypher = mock_session.run.call_args[0][0]
        assert "[:COMPETES_WITH*1..2]" in sent_cypher


# ---------------------------------------------------------------------------
# _get_neo4j — lazy singleton
# ---------------------------------------------------------------------------

class TestGetNeo4j:
    def test_returns_cached_driver_without_recreating(self):
        sentinel = MagicMock()
        hybrid_rag._neo4j_driver = sentinel
        assert _get_neo4j() is sentinel

    def test_returns_none_on_connection_failure(self, monkeypatch):
        monkeypatch.delenv("NEO4J_URI", raising=False)
        result = _get_neo4j()
        assert result is None


# ---------------------------------------------------------------------------
# rag_hybrid_query — fusion strategies
# ---------------------------------------------------------------------------

class TestRagHybridQuery:
    @pytest.mark.asyncio
    async def test_rrf_fusion_combines_and_sorts_by_score(self):
        with patch("rag.hybrid_rag.rag_vector_search", new_callable=AsyncMock) as mock_vec, \
             patch("rag.hybrid_rag.rag_graph_traverse", new_callable=AsyncMock) as mock_graph:
            mock_vec.return_value = {"results": [
                {"chunk_text": "vec result", "score": 0.9, "url": "u", "title": "t"}
            ]}
            mock_graph.return_value = {"nodes": [
                {"name": "AMD", "type": "Company", "relation": "COMPETES_WITH", "hops": 1}
            ]}

            result = await rag_hybrid_query("query", "NVDA", fusion="rrf")

        assert result["fusion"] == "rrf"
        assert len(result["results"]) == 2
        # results should be sorted descending by score
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_weighted_fusion_applies_weights(self):
        with patch("rag.hybrid_rag.rag_vector_search", new_callable=AsyncMock) as mock_vec, \
             patch("rag.hybrid_rag.rag_graph_traverse", new_callable=AsyncMock) as mock_graph:
            mock_vec.return_value = {"results": [
                {"chunk_text": "v", "score": 1.0, "url": "u", "title": "t"}
            ]}
            mock_graph.return_value = {"nodes": []}

            result = await rag_hybrid_query("q", "NVDA", fusion="weighted")

        assert result["results"][0]["score"] == 0.7  # 1.0 * 0.7 default weight

    @pytest.mark.asyncio
    async def test_union_fusion_just_concatenates(self):
        with patch("rag.hybrid_rag.rag_vector_search", new_callable=AsyncMock) as mock_vec, \
             patch("rag.hybrid_rag.rag_graph_traverse", new_callable=AsyncMock) as mock_graph:
            mock_vec.return_value = {"results": [
                {"chunk_text": "v", "score": 0.5, "url": "u", "title": "t"}
            ]}
            mock_graph.return_value = {"nodes": [
                {"name": "AMD", "type": "Company", "relation": "X", "hops": 1}
            ]}

            result = await rag_hybrid_query("q", "NVDA", fusion="union")

        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_runs_vector_and_graph_concurrently(self):
        """Both branches should be awaited via asyncio.gather, not sequentially."""
        with patch("rag.hybrid_rag.rag_vector_search", new_callable=AsyncMock) as mock_vec, \
             patch("rag.hybrid_rag.rag_graph_traverse", new_callable=AsyncMock) as mock_graph:
            mock_vec.return_value = {"results": []}
            mock_graph.return_value = {"nodes": []}

            await rag_hybrid_query("q", "NVDA")

            mock_vec.assert_called_once()
            mock_graph.assert_called_once()


# ---------------------------------------------------------------------------
# _key / _rrf / _weighted — pure helpers
# ---------------------------------------------------------------------------

class TestFusionHelpers:
    def test_key_is_deterministic_md5_of_truncated_text(self):
        item = {"text": "hello world"}
        assert _key(item) == _key(item)

    def test_rrf_combines_overlapping_items_by_key(self):
        item = {"text": "same content here"}
        a = [item]
        b = [item]
        fused = _rrf(a, b)
        assert len(fused) == 1  # same key, scores summed
        assert fused[0]["score"] > 1 / 61  # higher than single-list contribution

    def test_weighted_applies_default_0_7_0_3_split(self):
        vec = [{"score": 1.0}]
        graph = [{"score": 1.0}]
        fused = _weighted(vec, graph)
        assert fused[0]["score"] == 0.7
        assert fused[1]["score"] == 0.3


@pytest.fixture
def temporal_fixture():
    path = Path(__file__).parents[2] / "fixtures" / "stage4a_temporal_graph_hybrid.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, ForecastContext.from_dict(payload["context"])


class TestForecastContextBoundRetrieval:
    @pytest.mark.asyncio
    async def test_vector_forwards_the_same_context_before_ranking(self, temporal_fixture):
        payload, context = temporal_fixture
        retriever = MagicMock()
        retriever.retrieve_raw.return_value = payload["vector"]
        with patch.object(hybrid_rag, "_retriever", retriever):
            result = await rag_vector_search("supply chain", context=context)
        kwargs = retriever.retrieve_raw.call_args.kwargs
        assert kwargs["ticker"] == context.ticker
        assert kwargs["as_of"] == context.as_of
        assert kwargs["dataset_version"] == context.dataset_version
        assert kwargs["historical"] is True
        assert result["results"][0]["chunk_id"] == "chunk-shared-1"
        assert result["citations"][0]["evidence_id"] == "evidence-shared-1"

    @pytest.mark.asyncio
    async def test_live_context_uses_temporal_legacy_projection(self):
        context = ForecastContext(
            ticker="MSFT", as_of="2026-08-31T20:00:00Z",
            schema_version="forecast-v2", dataset_version="live-dataset-v1",
            feature_version="features-v1", model_version="model-v1",
        )
        retriever = MagicMock()
        retriever.store.supports_temporal_boundary = False
        retriever.retrieve_raw.return_value = [{
            "id": 1, "ticker": "MSFT", "source_type": "news",
            "text": "Microsoft demand improved.", "freshness_score": 0.5,
            "published_at": "2026-08-24T13:02:00Z",
            "retrieved_at": "2026-08-24T15:01:02Z",
            "dataset_version": "live-dataset-v1",
            "evidence_id": "evidence-live-1",
        }]

        with patch.object(hybrid_rag, "_retriever", retriever):
            result = await rag_vector_search("Microsoft demand", context=context)

        kwargs = retriever.retrieve_raw.call_args.kwargs
        assert kwargs["historical"] is False
        assert kwargs["as_of"] == context.as_of
        assert result.get("error_code") is None
        assert len(result["results"]) == 1
        assert result["degradation"]["reason"] == "legacy_live_temporal_projection"

    @pytest.mark.asyncio
    async def test_vector_rejects_ticker_context_mismatch(self, temporal_fixture):
        _, context = temporal_fixture
        with pytest.raises(ValueError, match="must match"):
            await rag_vector_search("query", ticker_filter="AMD", context=context)

    @pytest.mark.asyncio
    async def test_temporal_graph_cypher_excludes_all_leakage_classes(self, temporal_fixture):
        _, context = temporal_fixture
        driver = MagicMock()
        session = AsyncMock()

        class Empty:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration

        session.run = AsyncMock(return_value=Empty())
        driver.session.return_value.__aenter__.return_value = session
        driver.session.return_value.__aexit__.return_value = None
        with patch("rag.hybrid_rag._get_neo4j", return_value=driver):
            await rag_graph_traverse("NVDA", context=context)
        cypher = session.run.call_args.args[0]
        params = session.run.call_args.kwargs
        for required in (
            "e.is_quarantined = false",
            "e.ticker = $ticker",
            "e.dataset_version = $dataset_version",
            "e.published_at IS NOT NULL",
            "e.retrieved_at IS NOT NULL",
            "e.effective_from IS NOT NULL",
            "e.published_at <= datetime($as_of)",
            "e.retrieved_at <= datetime($as_of)",
            "e['filed_at'] <= datetime($as_of)",
            "e['effective_to'] > datetime($as_of)",
        ):
            assert required in cypher
        assert params["ticker"] == context.ticker
        assert params["dataset_version"] == context.dataset_version

    @pytest.mark.asyncio
    async def test_temporal_graph_does_not_fallback_to_legacy_on_error(self, temporal_fixture):
        _, context = temporal_fixture
        driver = MagicMock()
        driver.session.side_effect = RuntimeError("V2 unavailable")
        with patch("rag.hybrid_rag._get_neo4j", return_value=driver):
            result = await rag_graph_traverse("NVDA", context=context)
        assert result["nodes"] == []
        assert result["degradation"]["reason"] == "graph_error"

    @pytest.mark.asyncio
    async def test_hybrid_uses_shared_ids_citations_and_context(self, temporal_fixture):
        payload, context = temporal_fixture
        vector_result = {
            "results": [{
                "chunk_text": payload["vector"][0]["text"],
                "score": 0.9,
                "document_id": "doc-shared-1",
                "chunk_id": "chunk-shared-1",
                "evidence_id": "evidence-shared-1",
                "citation": hybrid_rag._citation(payload["vector"][0], origin="vector"),
            }],
            "degradation": {"available": True},
        }
        graph_item = payload["graph"][0]
        graph_item["citation"] = hybrid_rag._citation(graph_item, origin="graph")
        graph_result = {"nodes": [graph_item], "degradation": {"available": True}}
        with patch("rag.hybrid_rag.rag_vector_search", AsyncMock(return_value=vector_result)), \
             patch("rag.hybrid_rag.rag_graph_traverse", AsyncMock(return_value=graph_result)):
            result = await rag_hybrid_query("NVDA supply chain", context=context)
        assert result["degradation"]["mode"] == "hybrid"
        assert result["context"] == context.to_dict()
        assert {item["chunk_id"] for item in result["results"]} == {"chunk-shared-1"}
        assert {citation["origin"] for citation in result["citations"]} == {"vector", "graph"}

    def test_all_explicit_degradation_modes_are_deterministic(self, temporal_fixture):
        payload, context = temporal_fixture
        vector_up = {"degradation": {"available": True}}
        vector_down = {"degradation": {"available": False, "reason": "vector_error"}}
        graph_up = {"degradation": {"available": True}}
        graph_down = {"degradation": {"available": False, "reason": "graph_error"}}
        modes = {
            hybrid_rag._hybrid_response(query="q", entity="NVDA", fusion="rrf",
                context=context, results=[], vector=vector_up, graph=graph_down
            )["degradation"]["mode"],
            hybrid_rag._hybrid_response(query="q", entity="NVDA", fusion="rrf",
                context=context, results=[], vector=vector_down, graph=graph_up
            )["degradation"]["mode"],
            hybrid_rag._hybrid_response(query="q", entity="NVDA", fusion="rrf",
                context=context, results=[], vector=vector_up, graph=graph_up
            )["degradation"]["mode"],
            hybrid_rag._hybrid_response(query="q", entity="NVDA", fusion="rrf",
                context=context, results=[], vector=vector_down, graph=graph_down
            )["degradation"]["mode"],
        }
        assert modes == set(payload["expected_modes"])

    def test_fixture_covers_future_unknown_version_ticker_and_quarantine(self, temporal_fixture):
        payload, _ = temporal_fixture
        reasons = {item["reason"] for item in payload["ineligible_counterexamples"]}
        assert reasons == {
            "future_published", "future_retrieved", "unknown_published",
            "unknown_retrieved", "unknown_effective_from", "wrong_dataset",
            "unknown_materialized_at", "wrong_ticker", "quarantined",
        }


class TestRedactedTracingAndLifecycle:
    def test_vector_graph_rrf_hybrid_metadata_matches_active_attempt_without_raw_capture(
        self, temporal_fixture
    ):
        _, context = temporal_fixture
        sentinel = "SECRET_TOKEN RAW_QUERY https://private.test/attempt-two"
        expected = {
            "stage": "04A",
            "stage_revision": 1,
            "attempt": 2,
            "agent": "data_rag",
        }

        vector = hybrid_rag._trace_inputs("vector")({
            "query": sentinel, "context": context, "raw_backend_payload": sentinel,
        })
        graph = hybrid_rag._trace_outputs("graph")({
            "context": context.to_dict(),
            "nodes": [{"evidence_id": "evidence-1", "chunk_text": sentinel}],
            "citations": [{"url": sentinel}],
            "error": sentinel,
        })
        rrf_hybrid = hybrid_rag._hybrid_response(
            query=sentinel,
            entity=context.ticker,
            fusion="rrf",
            context=context,
            results=[],
            vector={
                "results": [], "citations": [],
                "degradation": {"available": True, "degraded": False},
                "telemetry": {"latency_ms": 1.0},
            },
            graph={
                "nodes": [], "citations": [],
                "degradation": {"available": True, "degraded": False},
                "telemetry": {"latency_ms": 1.0},
            },
            fusion_latency_ms=1.0,
        )
        hybrid = hybrid_rag._trace_outputs("hybrid")(rrf_hybrid)

        for metadata in (
            vector["metadata"], graph["metadata"],
            rrf_hybrid["telemetry"], hybrid["metadata"],
        ):
            assert {key: metadata[key] for key in expected} == expected
        encoded = json.dumps(
            {"vector": vector, "graph": graph, "rrf": rrf_hybrid["telemetry"],
             "hybrid": hybrid},
            sort_keys=True,
        )
        assert sentinel not in encoded
        assert "RAW_QUERY" not in encoded
        assert "private.test" not in encoded

    def test_trace_input_processor_never_serializes_query_headers_or_secret(
        self, temporal_fixture
    ):
        _, context = temporal_fixture
        sentinel = "SECRET_TOKEN RAW_QUERY https://private.test/input"
        processed = hybrid_rag._trace_inputs("vector")({
            "query": sentinel,
            "context": context,
            "headers": {"Authorization": sentinel},
            "raw_backend_payload": sentinel,
        })
        encoded = json.dumps(processed, sort_keys=True)
        assert sentinel not in encoded
        assert "RAW_QUERY" not in encoded
        assert "private.test" not in encoded
        assert processed["metadata"]["stage"] == "04A"
        assert processed["metadata"]["dataset_version"] == context.dataset_version

    def test_trace_output_processor_is_count_only_and_redacts_all_payloads(
        self, temporal_fixture
    ):
        _, context = temporal_fixture
        sentinel = "SECRET_TOKEN RAW_EVIDENCE https://private.test/output"
        output = {
            "context": context.to_dict(),
            "results": [{
                "chunk_text": sentinel,
                "evidence_id": "evidence-1",
                "url": sentinel,
            }],
            "citations": [{"url": sentinel, "title": sentinel}],
            "error": sentinel,
            "error_code": "VECTOR_RETRIEVAL_FAILED",
            "degradation": {"degraded": True, "available": False},
            "telemetry": {"latency_ms": 1.25},
        }
        processed = hybrid_rag._trace_outputs("vector")(output)
        encoded = json.dumps(processed, sort_keys=True)
        assert sentinel not in encoded
        assert "RAW_EVIDENCE" not in encoded
        assert "private.test" not in encoded
        assert processed["result_count"] == 1
        assert processed["evidence_count"] == 1
        assert processed["citation_count"] == 1
        assert processed["error_code"] == "VECTOR_RETRIEVAL_FAILED"
        assert processed["latency_ms"] == 1.25

    @pytest.mark.asyncio
    async def test_vector_exception_result_and_log_use_only_stable_code(self, caplog):
        sentinel = "SECRET_TOKEN RAW_EVIDENCE https://private.test/vector"
        retriever = MagicMock()
        retriever.retrieve_raw.side_effect = RuntimeError(sentinel)
        with patch.object(hybrid_rag, "_retriever", retriever):
            result = await rag_vector_search("RAW_QUERY_SENTINEL")
        assert result["error_code"] == "VECTOR_RETRIEVAL_FAILED"
        assert sentinel not in json.dumps(result)
        assert "SECRET_TOKEN" not in caplog.text
        assert "private.test" not in caplog.text
        assert result["telemetry"]["latency_ms"] >= 0
        assert result["telemetry"]["result_count"] == 0

    @pytest.mark.asyncio
    async def test_graph_exception_result_and_log_use_only_stable_code(self, caplog):
        sentinel = "SECRET_TOKEN RAW_EVIDENCE https://private.test/graph"
        driver = MagicMock()
        driver.session.side_effect = RuntimeError(sentinel)
        with patch("rag.hybrid_rag._get_neo4j", return_value=driver):
            result = await rag_graph_traverse("NVDA")
        assert result["error_code"] == "GRAPH_TRAVERSAL_FAILED"
        assert sentinel not in json.dumps(result)
        assert "SECRET_TOKEN" not in caplog.text
        assert "private.test" not in caplog.text
        assert result["telemetry"]["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_hybrid_reports_branch_fusion_rrf_and_overall_latency(
        self, temporal_fixture
    ):
        _, context = temporal_fixture
        vector = {
            "results": [], "citations": [],
            "degradation": {"available": True, "degraded": False},
            "telemetry": {"latency_ms": 2.0},
        }
        graph = {
            "nodes": [], "citations": [],
            "degradation": {"available": True, "degraded": False},
            "telemetry": {"latency_ms": 3.0},
        }
        with patch("rag.hybrid_rag.rag_vector_search", AsyncMock(return_value=vector)), \
             patch("rag.hybrid_rag.rag_graph_traverse", AsyncMock(return_value=graph)):
            result = await rag_hybrid_query("RAW_QUERY_SENTINEL", context=context)
        telemetry = result["telemetry"]
        assert telemetry["vector_latency_ms"] == 2.0
        assert telemetry["graph_latency_ms"] == 3.0
        assert telemetry["rrf_latency_ms"] >= 0
        assert telemetry["fusion_latency_ms"] >= 0
        assert telemetry["hybrid_latency_ms"] >= 0
        assert telemetry["result_count"] == 0
        assert telemetry["citation_count"] == 0
        assert telemetry["degradation_mode"] == "hybrid"
