"""
Tests for: rag/ingestion.py
Phase: 3 — RAG Pipeline (7th: orchestrates loader+processor+embedder+vector_store+graph_store)

Mocking strategy: every collaborator (AlphaLoader, AlphaProcessor, get_embedder,
AlphaVectorStore, AlphaGraphStore) is mocked at the module level where
run_ingestion_pipeline() imports them. We verify orchestration order, the
fail-fast-but-don't-crash behavior at each stage, and the "skip graph if
vector stage failed" guard — the most important behavioral contract in
this file.
"""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from rag.ingestion import run_ingestion_pipeline


def _patch_all(monkeypatch, **overrides):
    """Helper: patch every collaborator with sane defaults, override as needed."""
    patches = {}
    loader = MagicMock()
    loader.load.return_value = overrides.get("raw_docs", [MagicMock()])
    patches["loader_cls"] = patch("rag.ingestion.AlphaLoader", return_value=loader)

    processor = MagicMock()
    processor.process.return_value = overrides.get("chunks", [MagicMock()])
    processor.metrics.report.return_value = {}
    patches["processor_cls"] = patch("rag.ingestion.AlphaProcessor", return_value=processor)

    embedder = MagicMock()
    embedder.embed_chunks.return_value = overrides.get("records", [{"embedding": [0.1]}])
    patches["get_embedder"] = patch("rag.ingestion.get_embedder", return_value=embedder)

    vector_store = MagicMock()
    vector_store.upsert.return_value = overrides.get("upserted", 1)
    patches["vector_store_cls"] = patch("rag.ingestion.AlphaVectorStore", return_value=vector_store)

    graph_store = MagicMock()
    graph_store.connect = MagicMock()
    graph_store.extract_batch = AsyncMock(return_value=overrides.get("graph_docs", []))
    graph_store.upsert_batch.return_value = {"nodes_merged": 0, "rels_merged": 0}
    patches["graph_store_cls"] = patch("rag.ingestion.AlphaGraphStore", return_value=graph_store)

    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")

    started = {name: p.start() for name, p in patches.items()}
    return loader, processor, embedder, vector_store, graph_store, patches


def _stop_all(patches):
    for p in patches.values():
        p.stop()


# ---------------------------------------------------------------------------
# Env validation
# ---------------------------------------------------------------------------

class TestEnvValidation:
    @pytest.mark.asyncio
    async def test_missing_supabase_url_aborts_before_any_stage(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)

        with patch("rag.ingestion.AlphaLoader") as mock_loader_cls:
            await run_ingestion_pipeline(["NVDA"])
            mock_loader_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_supabase_key_falls_back_to_alt_env_var(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "u")
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.setenv("SUPABASE_KEY", "alt-key")

        loader = MagicMock()
        loader.load.return_value = []  # empty docs -> terminates early, but loader IS called
        with patch("rag.ingestion.AlphaLoader", return_value=loader), \
             patch("rag.ingestion.AlphaProcessor"), \
             patch("rag.ingestion.get_embedder"), \
             patch("rag.ingestion.AlphaVectorStore"), \
             patch("rag.ingestion.AlphaGraphStore"):
            await run_ingestion_pipeline(["NVDA"], skip_graph=True)
        loader.load.assert_called_once()


# ---------------------------------------------------------------------------
# Stage progression / early termination
# ---------------------------------------------------------------------------

class TestStageProgression:
    @pytest.mark.asyncio
    async def test_no_raw_docs_terminates_before_processing(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(
            monkeypatch, raw_docs=[]
        )
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=True)
            processor.process.assert_not_called()
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_no_chunks_after_dedup_terminates_before_embedding(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(
            monkeypatch, chunks=[]
        )
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=True)
            embedder.embed_chunks.assert_not_called()
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_load_exception_aborts_pipeline_without_raising(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        loader.load.side_effect = RuntimeError("yfinance down")
        try:
            await run_ingestion_pipeline(["NVDA"])  # must not raise
            processor.process.assert_not_called()
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_process_exception_aborts_before_embedding(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        processor.process.side_effect = RuntimeError("chunking failed")
        try:
            await run_ingestion_pipeline(["NVDA"])
            embedder.embed_chunks.assert_not_called()
        finally:
            _stop_all(patches)


# ---------------------------------------------------------------------------
# Vector stage failure -> graph stage MUST be skipped (critical contract)
# ---------------------------------------------------------------------------

class TestVectorFailureSkipsGraph:
    @pytest.mark.asyncio
    async def test_embed_failure_skips_graph_stage_entirely(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        embedder.embed_chunks.side_effect = RuntimeError("model crashed")
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=False)
            gs.extract_batch.assert_not_called()
            gs.upsert_batch.assert_not_called()
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_vector_upsert_failure_skips_graph_stage(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        vs.upsert.side_effect = RuntimeError("supabase down")
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=False)
            gs.extract_batch.assert_not_called()
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_vector_success_runs_graph_stage(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=False)
            gs.connect.assert_called_once()
            gs.extract_batch.assert_called_once()
            gs.upsert_batch.assert_called_once()
            gs.close.assert_called_once()
        finally:
            _stop_all(patches)


# ---------------------------------------------------------------------------
# skip_graph=True respected even on vector success
# ---------------------------------------------------------------------------

class TestSkipGraphFlag:
    @pytest.mark.asyncio
    async def test_skip_graph_true_never_constructs_graph_store(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        try:
            with patch("rag.ingestion.AlphaGraphStore") as mock_gs_cls:
                await run_ingestion_pipeline(["NVDA"], skip_graph=True)
                mock_gs_cls.assert_not_called()
        finally:
            _stop_all(patches)


# ---------------------------------------------------------------------------
# Graph stage failure -> close() still called (finally block)
# ---------------------------------------------------------------------------

class TestGraphStageCleanup:
    @pytest.mark.asyncio
    async def test_graph_extract_failure_still_closes_driver(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        gs.extract_batch.side_effect = RuntimeError("claude api down")
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=False)
            gs.close.assert_called_once()
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_graph_upsert_failure_still_closes_driver(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        gs.upsert_batch.side_effect = RuntimeError("neo4j down")
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=False)
            gs.close.assert_called_once()
        finally:
            _stop_all(patches)


# ---------------------------------------------------------------------------
# Sentry breadcrumbs / capture (smoke test — exact call shape, no assertion overload)
# ---------------------------------------------------------------------------

class TestSentryIntegration:
    @pytest.mark.asyncio
    async def test_sentry_capture_called_on_load_failure_when_enabled(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        loader.load.side_effect = RuntimeError("boom")
        try:
            with patch("rag.ingestion.sentry_enabled", return_value=True), \
                 patch("sentry_sdk.push_scope") as mock_push_scope, \
                 patch("sentry_sdk.capture_exception") as mock_capture, \
                 patch("sentry_sdk.add_breadcrumb"):
                scope = MagicMock()
                mock_push_scope.return_value.__enter__.return_value = scope
                await run_ingestion_pipeline(["NVDA"])
                mock_capture.assert_called_once()
                scope.set_tag.assert_any_call("component", "ingestion.load")
        finally:
            _stop_all(patches)