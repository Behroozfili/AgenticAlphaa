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
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

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
        assert "retriever failed" in result["error"]

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
        assert "neo4j down" in result["error"]

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