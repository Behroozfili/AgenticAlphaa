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
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from rag.ingestion import StageStatus, _TELEMETRY_ALLOWED_FIELDS, run_ingestion_pipeline
from rag.loader import RawDocument
from rag.processor import ProcessedChunk


def _patch_all(monkeypatch, **overrides):
    """Helper: patch every collaborator with sane defaults, override as needed."""
    patches = {}
    loader = MagicMock()
    loader.load.return_value = overrides.get("raw_docs", [RawDocument(
        title="fixture title",
        content="fixture content",
        url="https://fixture.invalid/document",
        source_type="archive",
        ticker="NVDA",
        published_at="2024-01-01T00:00:00+00:00",
        retrieved_at="2024-01-02T00:00:00+00:00",
        effective_from="2024-01-01T00:00:00+00:00",
        dataset_version="legacy-v1",
        source_id="a" * 64,
    )])
    patches["loader_cls"] = patch("rag.ingestion.AlphaLoader", return_value=loader)

    processor = MagicMock()
    processor.process.return_value = overrides.get("chunks", [ProcessedChunk(
        text="fixture chunk",
        metadata={
            "document_id": "1" * 64,
            "evidence_id": "2" * 64,
            "chunk_id": "3" * 64,
        },
    )])
    processor.metrics.report.return_value = {}
    patches["processor_cls"] = patch("rag.ingestion.AlphaProcessor", return_value=processor)

    embedder = MagicMock()
    embedder.embed_chunks.return_value = overrides.get("records", [{"embedding": [0.1]}])
    embedder.manifest = SimpleNamespace(
        model_name="BAAI/bge-small-en-v1.5",
        model_version="weights-v1",
        cache_schema_version="embedding-cache-v1",
    )
    patches["get_embedder"] = patch("rag.ingestion.get_embedder", return_value=embedder)

    vector_store = MagicMock()
    vector_store.upsert.return_value = overrides.get("upserted", 1)
    vector_store.supports_temporal_boundary = overrides.get("supports_temporal_boundary", False)
    patches["vector_store_cls"] = patch("rag.ingestion.AlphaVectorStore", return_value=vector_store)

    graph_store = MagicMock()
    graph_store.connect = MagicMock()
    graph_store.extract_batch = AsyncMock(return_value=overrides.get("graph_docs", []))
    graph_store.upsert_batch.return_value = {"nodes_merged": 0, "rels_merged": 0}
    patches["graph_store_cls"] = patch("rag.ingestion.AlphaGraphStore", return_value=graph_store)

    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")
    monkeypatch.delenv("RAG_VECTOR_SCHEMA", raising=False)

    for p in patches.values():
        p.start()
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
            with patch("core.error_handler.sentry_enabled", return_value=True), \
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


class TestCheckpointResume:
    @pytest.mark.asyncio
    async def test_graph_failure_resumes_without_repeating_completed_vector(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "ingestion.json"
        gs.upsert_batch.side_effect = RuntimeError("graph unavailable")
        try:
            first = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            assert first.vector.status == "complete"
            assert first.graph.status == "failed"
            gs.upsert_batch.side_effect = None
            gs.upsert_batch.return_value = {"nodes_merged": 1, "rels_merged": 2}
            second = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            assert second.vector.status == "complete"
            assert second.graph.status == "complete"
            assert vs.upsert.call_count == 1
            assert gs.upsert_batch.call_count == 2
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_missing_credentials_checkpoint_retries_before_materialization(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "missing-credentials.json"
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_KEY", raising=False)
        try:
            first = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
            monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")
            second = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            assert first.vector.error == "INGESTION_CONFIG_MISSING_CREDENTIALS"
            assert first.materialization_fingerprint is None
            assert second.vector.status == "complete"
            assert second.materialization_fingerprint
            assert vs.upsert.call_count == 1
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_transient_load_failure_checkpoint_retries_before_materialization(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "load-retry.json"
        successful_docs = list(loader.load.return_value)
        loader.load.side_effect = [RuntimeError("transient"), successful_docs]
        try:
            first = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            second = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            assert first.vector.error == "INGESTION_LOAD_UNEXPECTED"
            assert first.materialization_fingerprint is None
            assert second.vector.status == "complete"
            assert vs.upsert.call_count == 1
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_transient_process_failure_checkpoint_retries_before_materialization(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "process-retry.json"
        successful_chunks = list(processor.process.return_value)
        processor.process.side_effect = [RuntimeError("transient"), successful_chunks]
        try:
            first = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            second = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            assert first.vector.error == "INGESTION_PROCESS_UNEXPECTED"
            assert first.materialization_fingerprint is None
            assert second.vector.status == "complete"
            assert vs.upsert.call_count == 1
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_completed_vector_without_fingerprint_still_fails_closed(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "completed-missing-fingerprint.json"
        try:
            first = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            for key in (
                "materialization_fingerprint", "document_manifest_hash",
                "evidence_manifest_hash", "chunk_manifest_hash",
                "raw_document_manifest_hash", "record_manifest_hash",
            ):
                payload[key] = None
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            second = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            assert first.vector.status == "complete"
            assert second.graph.error == "INGESTION_RESUME_FINGERPRINT_MISSING"
            assert vs.upsert.call_count == 1
        finally:
            _stop_all(patches)


class TestHistoricalArchiveBoundary:
    @pytest.mark.asyncio
    async def test_public_pipeline_ingests_archive_offline_through_v2(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "key")
        monkeypatch.setenv("RAG_VECTOR_SCHEMA", "v2")
        provider_calls = []

        def archive_provider(**boundaries):
            provider_calls.append(boundaries)
            return [{
                "title": "NVDA archived filing",
                "content": "NVDA reported deterministic archived evidence.",
                "url": "https://archive.invalid/nvda/filing-1",
                "source_type": "archive",
                "ticker": "NVDA",
                "published_at": "2024-01-10T10:00:00+00:00",
                "retrieved_at": "2024-01-11T09:00:00+00:00",
                "effective_from": "2024-01-10T10:00:00+00:00",
                "dataset_version": "snapshot-v1",
                "revision": "1",
            }]

        embedder = MagicMock()
        embedder.manifest = SimpleNamespace(
            model_name="fixture-embedder",
            model_version="weights-v1",
            cache_schema_version="embedding-cache-v1",
        )
        embedder.embed_chunks.side_effect = lambda chunks: [
            {"embedding": [0.1], "text": chunk.text, "metadata": dict(chunk.metadata)}
            for chunk in chunks
        ]
        vector_store = MagicMock()
        vector_store.supports_temporal_boundary = True
        vector_store.upsert.return_value = 1

        with patch("rag.loader.AlphaLoader._fetch_yfinance") as live_yfinance, \
             patch("rag.loader.AlphaLoader._fetch_reddit_rss") as live_reddit, \
             patch("rag.ingestion.get_embedder", return_value=embedder), \
             patch("rag.ingestion.AlphaVectorStore", return_value=vector_store) as store_cls:
            manifest = await run_ingestion_pipeline(
                ["nvda"],
                skip_graph=True,
                historical=True,
                start="2024-01-01T00:00:00Z",
                end="2024-01-31T23:59:59Z",
                as_of="2024-02-01T00:00:00Z",
                dataset_version="snapshot-v1",
                archive_provider=archive_provider,
                archive_provider_id="fixture-archive-v1",
            )

        assert manifest.vector.status == "complete"
        assert manifest.graph.status == "skipped"
        assert manifest.materialization_fingerprint
        assert provider_calls == [{
            "tickers": ["NVDA"],
            "start": "2024-01-01T00:00:00+00:00",
            "end": "2024-01-31T23:59:59+00:00",
            "as_of": "2024-02-01T00:00:00+00:00",
            "dataset_version": "snapshot-v1",
        }]
        live_yfinance.assert_not_called()
        live_reddit.assert_not_called()
        assert store_cls.call_args.kwargs["config"].schema == "v2"
        record = vector_store.upsert.call_args.kwargs["records"][0]
        assert record["metadata"]["ticker"] == "NVDA"
        assert record["metadata"]["dataset_version"] == "snapshot-v1"
        assert all(len(record["metadata"][key]) == 64 for key in ("document_id", "evidence_id", "chunk_id"))


class TestSnapshotIdentityFailClosed:
    @pytest.mark.asyncio
    async def test_provider_change_rejects_checkpoint_before_reuse(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(
            monkeypatch, supports_temporal_boundary=True
        )
        monkeypatch.setenv("RAG_VECTOR_SCHEMA", "v2")
        checkpoint = tmp_path / "provider.json"
        provider = lambda **_kwargs: []
        try:
            first = await run_ingestion_pipeline(
                ["NVDA"], skip_graph=True, historical=True,
                start="2024-01-01", end="2024-01-31", as_of="2024-02-01",
                dataset_version="snapshot-v1", checkpoint_path=checkpoint,
                archive_provider=provider, archive_provider_id="provider-v1",
            )
            second = await run_ingestion_pipeline(
                ["NVDA"], skip_graph=True, historical=True,
                start="2024-01-01", end="2024-01-31", as_of="2024-02-01",
                dataset_version="snapshot-v1", checkpoint_path=checkpoint,
                archive_provider=provider, archive_provider_id="provider-v2",
            )
            assert first.vector.status == "complete"
            assert second.vector.error == "INGESTION_RESUME_SNAPSHOT_MISMATCH"
            assert vs.upsert.call_count == 1
            persisted = checkpoint.read_text(encoding="utf-8")
            assert "fixture chunk" not in persisted
            assert "provider-v1" not in persisted
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_content_identity_change_blocks_graph_resume(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "content.json"
        gs.upsert_batch.side_effect = RuntimeError("graph unavailable")
        try:
            first = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            processor.process.return_value = [ProcessedChunk(
                text="changed fixture chunk",
                metadata={"document_id": "4" * 64, "evidence_id": "5" * 64, "chunk_id": "6" * 64},
            )]
            second = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            assert first.graph.status == "failed"
            assert second.graph.error == "INGESTION_RESUME_SNAPSHOT_MISMATCH"
            assert vs.upsert.call_count == 1
            assert gs.upsert_batch.call_count == 1
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mutation", ["ticker", "metadata"])
    async def test_raw_ticker_or_metadata_change_blocks_graph_resume(self, monkeypatch, tmp_path, mutation):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / f"raw-{mutation}.json"
        gs.upsert_batch.side_effect = RuntimeError("graph unavailable")
        try:
            first = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            original = loader.load.return_value[0]
            changed = RawDocument(**{
                **original.__dict__,
                "ticker": "AMD" if mutation == "ticker" else original.ticker,
                "metadata": {"association": "changed"} if mutation == "metadata" else dict(original.metadata),
            })
            loader.load.return_value = [changed]
            second = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            assert first.vector.status == "complete"
            assert second.graph.error == "INGESTION_RESUME_SNAPSHOT_MISMATCH"
            assert vs.upsert.call_count == 1
            assert gs.upsert_batch.call_count == 1
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_embedding_model_change_blocks_graph_resume(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "model.json"
        gs.upsert_batch.side_effect = RuntimeError("graph unavailable")
        try:
            first = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            embedder.manifest = SimpleNamespace(
                model_name="BAAI/bge-small-en-v1.5",
                model_version="weights-v2",
                cache_schema_version="embedding-cache-v1",
            )
            second = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
            assert first.graph.status == "failed"
            assert second.graph.error == "INGESTION_RESUME_SNAPSHOT_MISMATCH"
            assert vs.upsert.call_count == 1
            assert gs.upsert_batch.call_count == 1
        finally:
            _stop_all(patches)

    @pytest.mark.asyncio
    async def test_vector_adapter_change_rejects_checkpoint_before_reuse(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "adapter.json"
        try:
            first = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            monkeypatch.setenv("RAG_VECTOR_SCHEMA", "v2")
            second = await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            assert first.vector.status == "complete"
            assert second.vector.error == "INGESTION_RESUME_SNAPSHOT_MISMATCH"
            assert vs.upsert.call_count == 1
        finally:
            _stop_all(patches)


class TestLifecycleTelemetry:
    def test_legacy_raw_checkpoint_error_is_sanitized_on_materialization(self):
        status = StageStatus("failed", error="SENTINEL_SECRET raw provider payload")
        assert status.error == "INGESTION_CHECKPOINT_LEGACY_ERROR"

    @pytest.mark.asyncio
    async def test_success_telemetry_is_allowlisted_versioned_and_measured(self, monkeypatch):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        embedder.manifest = SimpleNamespace(
            model_name="BAAI.bge-small-en-v1.5",
            model_version="weights-1",
            cache_schema_version="embedding-cache-v1",
        )
        emitted = []
        try:
            with patch("rag.ingestion.add_breadcrumb", side_effect=lambda component, message, data: emitted.append((component, message, data))):
                manifest = await run_ingestion_pipeline(["nvda"], skip_graph=True, dataset_version="fixture-v1")
        finally:
            _stop_all(patches)
        assert manifest.vector.status == "complete"
        messages = {message for _, message, _ in emitted}
        assert {"stage03.load.complete", "stage03.process.complete", "stage03.vector.complete", "stage03.graph.skip", "stage03.checkpoint.complete"} <= messages
        for component, message, payload in emitted:
            assert component == "ingestion"
            assert message.startswith("stage03.")
            assert set(payload) <= _TELEMETRY_ALLOWED_FIELDS
            assert payload["stage"] == "03"
            assert payload["stage_revision"] == 2
            assert payload["attempt"] == 1
            assert payload["agent"] == "data_rag"
            assert payload["dataset_version"] == "fixture-v1"
            assert payload["tickers"] == ["NVDA"]
            assert isinstance(payload["latency_ms"], float)
            assert payload["latency_ms"] >= 0
            assert "url" not in payload and "text" not in payload and "credentials" not in payload
        vector_payload = next(payload for _, message, payload in emitted if message == "stage03.vector.complete")
        assert vector_payload["vector_schema"] == "legacy"
        assert vector_payload["vector_adapter"].startswith("legacy:")
        assert vector_payload["embedding_version"] == "weights-1"
        assert vector_payload["embedding_cache_schema"] == "embedding-cache-v1"
        graph_payload = next(payload for _, message, payload in emitted if message == "stage03.graph.skip")
        assert graph_payload["degraded"] is True
        assert graph_payload["status"] == "skipped_requested"

    @pytest.mark.asyncio
    async def test_resume_and_completed_skip_are_visible(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "resume.json"
        emitted = []
        try:
            with patch("rag.ingestion.add_breadcrumb", side_effect=lambda component, message, data: emitted.append((message, data))):
                await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
                await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
        finally:
            _stop_all(patches)
        resumed = [payload for message, payload in emitted if message == "stage03.checkpoint.load" and payload["status"] == "resumed"]
        skipped = [payload for message, payload in emitted if message == "stage03.resume.skip"]
        assert len(resumed) == 1
        assert len(skipped) == 1
        assert skipped[0]["vector_status"] == "complete"
        assert skipped[0]["degraded"] is False

    @pytest.mark.asyncio
    async def test_untrusted_exception_text_is_never_persisted_or_emitted(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        secret = "SENTINEL_SECRET_TOKEN"
        raw_payload = "SENTINEL_RAW_PAYLOAD"
        loader.load.side_effect = RuntimeError(f"provider failed {secret} https://private.invalid {raw_payload}")
        checkpoint = tmp_path / "sanitized.json"
        emitted = []
        captured = []
        try:
            with patch("rag.ingestion.add_breadcrumb", side_effect=lambda component, message, data: emitted.append((message, data))), patch("rag.ingestion.capture_exception", side_effect=lambda *args, **kwargs: captured.append((args, kwargs))):
                manifest = await run_ingestion_pipeline(["NVDA"], checkpoint_path=checkpoint)
        finally:
            _stop_all(patches)
        serialized = checkpoint.read_text(encoding="utf-8")
        observable = json.dumps({"checkpoint": serialized, "emitted": emitted, "captured": [(str(args), kwargs) for args, kwargs in captured]})
        assert secret not in observable
        assert raw_payload not in observable
        assert "private.invalid" not in observable
        assert manifest.vector.error == "INGESTION_LOAD_UNEXPECTED"
        assert json.loads(serialized)["vector"]["error"] == "INGESTION_LOAD_UNEXPECTED"
        failure = next(payload for message, payload in emitted if message == "stage03.load.failure")
        assert failure["error_code"] == "INGESTION_LOAD_UNEXPECTED"
        assert failure["degraded"] is True
        assert isinstance(failure["latency_ms"], float)

    @pytest.mark.asyncio
    async def test_completed_checkpoint_is_idempotent(self, monkeypatch, tmp_path):
        loader, processor, embedder, vs, gs, patches = _patch_all(monkeypatch)
        checkpoint = tmp_path / "ingestion.json"
        try:
            await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            await run_ingestion_pipeline(["NVDA"], skip_graph=True, checkpoint_path=checkpoint)
            # Resume re-materializes the source to validate the immutable
            # fingerprint before it may reuse completed Vector work.
            assert loader.load.call_count == 2
            assert vs.upsert.call_count == 1
        finally:
            _stop_all(patches)
