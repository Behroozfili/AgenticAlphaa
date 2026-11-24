"""Checkpointed ingestion bound to one reproducible materialization snapshot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from core.error_handler import add_breadcrumb, capture_exception
from core.observability import init_sentry
from rag.embedding_manager import get_embedder
from rag.graph_store import AlphaGraphStore
from rag.loader import AlphaLoader, ArchiveProvider, RawDocument
from rag.processor import AlphaProcessor, ProcessedChunk
from rag.vector_store import AlphaVectorStore, VectorStoreConfig

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
COMPONENT = "ingestion"

_CONTROL_IDENTITY = {
    "stage": "03",
    "stage_revision": 2,
    "attempt": 1,
    "agent": "data_rag",
    "contract_version": "stage03-r2-a1",
}
_STAGE_METADATA = {key: _CONTROL_IDENTITY[key] for key in ("stage", "stage_revision", "attempt", "agent")}
_TELEMETRY_ALLOWED_FIELDS = frozenset({
    "stage", "stage_revision", "attempt", "agent", "event", "run_id", "tickers",
    "as_of", "dataset_version", "historical", "vector_schema", "vector_adapter",
    "embedding_model", "embedding_version", "embedding_cache_schema", "latency_ms",
    "document_count", "chunk_count", "quarantine_count", "upsert_count", "relation_count",
    "vector_status", "graph_status", "status", "degraded", "error_code",
    "provider_identity_hash", "snapshot_fingerprint",
})
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.^-]{1,16}$")
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")
_ERROR_CODE_PATTERN = re.compile(r"^INGESTION_[A-Z0-9_]+$")
_HEX64_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class CheckpointSnapshotMismatch(RuntimeError):
    """A checkpoint belongs to a different request or materialization."""


class CheckpointFingerprintMissing(RuntimeError):
    """A completed checkpoint predates snapshot identity and cannot be reused."""


@dataclass
class StageStatus:
    status: str = "pending"
    count: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error is not None and not _ERROR_CODE_PATTERN.fullmatch(self.error):
            self.error = "INGESTION_CHECKPOINT_LEGACY_ERROR"


@dataclass
class IngestionManifest:
    run_id: str
    tickers: list[str]
    dataset_version: str
    historical: bool
    start: str | None
    end: str | None
    as_of: str | None
    vector: StageStatus = field(default_factory=StageStatus)
    graph: StageStatus = field(default_factory=StageStatus)
    quarantine_count: int = 0
    chunk_count: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_fingerprint: str | None = None
    materialization_fingerprint: str | None = None
    provider_identity_hash: str | None = None
    adapter_identity_hash: str | None = None
    embedding_identity_hash: str | None = None
    document_manifest_hash: str | None = None
    evidence_manifest_hash: str | None = None
    chunk_manifest_hash: str | None = None
    raw_document_manifest_hash: str | None = None
    record_manifest_hash: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IngestionManifest":
        payload = dict(value)
        payload["vector"] = StageStatus(**payload.get("vector", {}))
        payload["graph"] = StageStatus(**payload.get("graph", {}))
        return cls(**payload)


class CheckpointStore:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None

    def load(self, run_id: str, request_fingerprint: str | None = None) -> IngestionManifest | None:
        if self.path is None or not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("run_id") != run_id:
            raise CheckpointSnapshotMismatch("checkpoint request identity mismatch")
        manifest = IngestionManifest.from_dict(payload)
        if request_fingerprint is not None and manifest.request_fingerprint != request_fingerprint:
            raise CheckpointSnapshotMismatch("checkpoint request fingerprint mismatch")
        return manifest

    def save(self, manifest: IngestionManifest) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(manifest), sort_keys=True, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_boundary(value: object | None, *, name: str, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"historical ingestion requires {name}")
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid timestamp") from exc
    else:
        raise ValueError(f"{name} must be a valid timestamp")
    parsed = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _adapter_identity(config: VectorStoreConfig) -> dict[str, str]:
    if config.schema == "v2":
        return {"schema": "v2", "table": config.v2_table, "rpc": config.v2_rpc}
    return {"schema": "legacy", "table": config.legacy_table, "rpc": config.legacy_rpc}


def _provider_identity(*, historical: bool, archive_provider: ArchiveProvider | None, archive_provider_id: str | None) -> tuple[str, str]:
    if not historical:
        identity = "live-yfinance-reddit-v1"
        return identity, hashlib.sha256(identity.encode("utf-8")).hexdigest()
    if archive_provider is None:
        raise ValueError("historical ingestion requires an archive_provider; live fallback is disabled")
    identity = archive_provider_id or getattr(archive_provider, "provider_id", None)
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("historical ingestion requires a stable archive_provider_id")
    return identity, hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _request_fingerprint(*, tickers: list[str], dataset_version: str, historical: bool, start: str | None, end: str | None, as_of: str | None, provider_identity_hash: str, adapter: dict[str, str]) -> str:
    return _sha256_json({
        "control": _CONTROL_IDENTITY,
        "tickers": tickers,
        "dataset_version": dataset_version,
        "historical": historical,
        "start": start,
        "end": end,
        "as_of": as_of,
        "provider_identity_hash": provider_identity_hash,
        "adapter": adapter,
    })


def _embedding_identity(embedder: object) -> dict[str, str]:
    manifest = getattr(embedder, "manifest", None)
    values = {
        "model_name": getattr(manifest, "model_name", None),
        "model_version": getattr(manifest, "model_version", None),
        "cache_schema_version": getattr(manifest, "cache_schema_version", None),
    }
    if not all(isinstance(value, str) and _MODEL_PATTERN.fullmatch(value) for value in values.values()):
        raise ValueError("embedding manifest identity is incomplete")
    return values  # type: ignore[return-value]


def _ordered_identity_hash(chunks: list[ProcessedChunk], key: str) -> str:
    values: list[str] = []
    for chunk in chunks:
        metadata = chunk.metadata if isinstance(chunk.metadata, dict) else {}
        value = metadata.get(key)
        if not isinstance(value, str) or not _HEX64_PATTERN.fullmatch(value):
            raise ValueError(f"processed chunk is missing canonical {key}")
        values.append(value)
    return _sha256_json(values)


def _raw_document_manifest_hash(documents: list[RawDocument]) -> str:
    records = []
    for document in documents:
        records.append(_sha256_json({
            "title": document.title,
            "content": document.content,
            "url": document.url,
            "source_type": document.source_type,
            "ticker": document.ticker,
            "published_at": document.published_at,
            "retrieved_at": document.retrieved_at,
            "effective_from": document.effective_from,
            "effective_to": document.effective_to,
            "dataset_version": document.dataset_version,
            "source_id": document.source_id,
            "revision": document.revision,
            "metadata": document.metadata,
        }))
    return _sha256_json(sorted(records))


def _record_manifest_hash(chunks: list[ProcessedChunk]) -> str:
    records: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk.text, str) or not isinstance(chunk.metadata, dict):
            raise ValueError("processed chunk record is incomplete")
        records.append(_sha256_json({"text": chunk.text, "metadata": chunk.metadata}))
    return _sha256_json(sorted(records))


def _materialization_identity(*, request_fingerprint: str, provider_identity_hash: str, adapter: dict[str, str], embedding: dict[str, str], raw_documents: list[RawDocument], chunks: list[ProcessedChunk]) -> dict[str, str]:
    components = {
        "request_fingerprint": request_fingerprint,
        "provider_identity_hash": provider_identity_hash,
        "adapter_identity_hash": _sha256_json(adapter),
        "embedding_identity_hash": _sha256_json(embedding),
        "document_manifest_hash": _ordered_identity_hash(chunks, "document_id"),
        "evidence_manifest_hash": _ordered_identity_hash(chunks, "evidence_id"),
        "chunk_manifest_hash": _ordered_identity_hash(chunks, "chunk_id"),
        "raw_document_manifest_hash": _raw_document_manifest_hash(raw_documents),
        "record_manifest_hash": _record_manifest_hash(chunks),
    }
    return {**components, "materialization_fingerprint": _sha256_json(components)}


def _sanitize_error(stage: str, exc: BaseException) -> str:
    if isinstance(exc, CheckpointFingerprintMissing):
        return "INGESTION_RESUME_FINGERPRINT_MISSING"
    if isinstance(exc, CheckpointSnapshotMismatch):
        return "INGESTION_RESUME_SNAPSHOT_MISMATCH"
    prefix = f"INGESTION_{stage.upper()}"
    if isinstance(exc, TimeoutError):
        category = "TIMEOUT"
    elif isinstance(exc, (ConnectionError, OSError)):
        category = "IO_ERROR"
    elif isinstance(exc, (ValueError, TypeError, KeyError)):
        category = "INVALID_INPUT"
    else:
        category = "UNEXPECTED"
    return f"{prefix}_{category}"


def _capture_sanitized(stage: str, exc: BaseException) -> str:
    code = _sanitize_error(stage, exc)
    capture_exception(f"{COMPONENT}.{stage}", RuntimeError(code), tags={"error_code": code})
    return code


def _telemetry_base(*, run_id: str, tickers: list[str], as_of: str | None, dataset_version: str, historical: bool, vector_config: VectorStoreConfig, provider_identity_hash: str) -> dict[str, Any]:
    adapter = _adapter_identity(vector_config)
    return {
        **_STAGE_METADATA,
        "run_id": run_id,
        "tickers": list(tickers),
        "as_of": as_of,
        "dataset_version": dataset_version,
        "historical": historical,
        "vector_schema": adapter["schema"],
        "vector_adapter": f"{adapter['schema']}:{adapter['table']}:{adapter['rpc']}",
        "provider_identity_hash": provider_identity_hash,
    }


def _emit(event: str, base: dict[str, Any], *, started_at: float, status: str, degraded: bool, **fields: Any) -> None:
    payload = {
        **base,
        "event": event,
        "latency_ms": round(max(0.0, (time.perf_counter() - started_at) * 1000.0), 3),
        "status": status,
        "degraded": bool(degraded),
        **fields,
    }
    payload = {key: value for key, value in payload.items() if key in _TELEMETRY_ALLOWED_FIELDS and value is not None}
    add_breadcrumb(COMPONENT, f"stage03.{event}", payload)


def _embedding_fields(embedding: dict[str, str]) -> dict[str, str]:
    return {
        "embedding_model": embedding["model_name"],
        "embedding_version": embedding["model_version"],
        "embedding_cache_schema": embedding["cache_schema_version"],
    }


def _save_checkpoint(checkpoint: CheckpointStore, manifest: IngestionManifest, base: dict[str, Any]) -> None:
    started = time.perf_counter()
    try:
        checkpoint.save(manifest)
    except Exception as exc:
        code = _capture_sanitized("checkpoint", exc)
        _emit("checkpoint.failure", base, started_at=started, status="failed", degraded=True, error_code=code, vector_status=manifest.vector.status, graph_status=manifest.graph.status)
        raise RuntimeError(code) from None
    _emit("checkpoint.complete", base, started_at=started, status="complete", degraded=manifest.vector.status == "failed" or manifest.graph.status == "failed", vector_status=manifest.vector.status, graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint)


def _new_manifest(*, run_id: str, request_fingerprint: str, tickers: list[str], dataset_version: str, historical: bool, start: str | None, end: str | None, as_of: str | None, provider_identity_hash: str, adapter_identity_hash: str) -> IngestionManifest:
    return IngestionManifest(
        run_id=run_id,
        tickers=tickers,
        dataset_version=dataset_version,
        historical=historical,
        start=start,
        end=end,
        as_of=as_of,
        request_fingerprint=request_fingerprint,
        provider_identity_hash=provider_identity_hash,
        adapter_identity_hash=adapter_identity_hash,
    )


async def run_ingestion_pipeline(
    tickers: list[str],
    skip_graph: bool = False,
    *,
    start: object | None = None,
    end: object | None = None,
    as_of: object | None = None,
    dataset_version: str = "legacy-v1",
    historical: bool | None = None,
    checkpoint_path: str | Path | None = None,
    archive_provider: ArchiveProvider | None = None,
    archive_provider_id: str | None = None,
) -> IngestionManifest:
    pipeline_started = time.perf_counter()
    tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    if any(not _TICKER_PATTERN.fullmatch(ticker) for ticker in tickers):
        raise ValueError("tickers must be compact normalized symbols")
    if not _IDENTIFIER_PATTERN.fullmatch(dataset_version):
        raise ValueError("dataset_version must be a compact identifier")
    historical = historical if historical is not None else any(value is not None for value in (start, end, as_of))
    normalized_start = _normalize_boundary(start, name="start", required=historical)
    normalized_end = _normalize_boundary(end, name="end", required=historical)
    normalized_as_of = _normalize_boundary(as_of, name="as_of", required=historical)

    try:
        _, provider_identity_hash = _provider_identity(
            historical=historical,
            archive_provider=archive_provider,
            archive_provider_id=archive_provider_id,
        )
        vector_config = VectorStoreConfig.from_environment()
    except Exception as exc:
        # No untrusted provider/config value is persisted or emitted here.
        fallback_config = VectorStoreConfig()
        fallback_provider_hash = hashlib.sha256(b"invalid-provider").hexdigest()
        request_fingerprint = _sha256_json({"invalid_request": True, "control": _CONTROL_IDENTITY})
        run_id = request_fingerprint
        base = _telemetry_base(run_id=run_id, tickers=tickers, as_of=normalized_as_of, dataset_version=dataset_version, historical=historical, vector_config=fallback_config, provider_identity_hash=fallback_provider_hash)
        code = _capture_sanitized("config", exc)
        manifest = _new_manifest(run_id=run_id, request_fingerprint=request_fingerprint, tickers=tickers, dataset_version=dataset_version, historical=historical, start=normalized_start, end=normalized_end, as_of=normalized_as_of, provider_identity_hash=fallback_provider_hash, adapter_identity_hash=_sha256_json(_adapter_identity(fallback_config)))
        manifest.vector = StageStatus("failed", error=code)
        _emit("config.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status="pending")
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status="pending")
        return manifest

    adapter = _adapter_identity(vector_config)
    request_fingerprint = _request_fingerprint(
        tickers=tickers,
        dataset_version=dataset_version,
        historical=historical,
        start=normalized_start,
        end=normalized_end,
        as_of=normalized_as_of,
        provider_identity_hash=provider_identity_hash,
        adapter=adapter,
    )
    run_id = request_fingerprint
    base = _telemetry_base(run_id=run_id, tickers=tickers, as_of=normalized_as_of, dataset_version=dataset_version, historical=historical, vector_config=vector_config, provider_identity_hash=provider_identity_hash)
    _emit("pipeline.start", base, started_at=pipeline_started, status="started", degraded=False, vector_status="pending", graph_status="skipped" if skip_graph else "pending")

    checkpoint = CheckpointStore(checkpoint_path)
    checkpoint_started = time.perf_counter()
    try:
        loaded_manifest = checkpoint.load(run_id, request_fingerprint)
    except Exception as exc:
        code = _capture_sanitized("resume", exc)
        manifest = _new_manifest(run_id=run_id, request_fingerprint=request_fingerprint, tickers=tickers, dataset_version=dataset_version, historical=historical, start=normalized_start, end=normalized_end, as_of=normalized_as_of, provider_identity_hash=provider_identity_hash, adapter_identity_hash=_sha256_json(adapter))
        manifest.vector = StageStatus("failed", error=code)
        _emit("checkpoint.failure", base, started_at=checkpoint_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status="pending")
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status="pending")
        return manifest
    manifest = loaded_manifest or _new_manifest(run_id=run_id, request_fingerprint=request_fingerprint, tickers=tickers, dataset_version=dataset_version, historical=historical, start=normalized_start, end=normalized_end, as_of=normalized_as_of, provider_identity_hash=provider_identity_hash, adapter_identity_hash=_sha256_json(adapter))
    _emit("checkpoint.load", base, started_at=checkpoint_started, status="resumed" if loaded_manifest else "new", degraded=False, vector_status=manifest.vector.status, graph_status=manifest.graph.status)

    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") or os.environ.get("SUPABASE_KEY", "")
    if not supabase_url or not supabase_key:
        code = "INGESTION_CONFIG_MISSING_CREDENTIALS"
        manifest.vector = StageStatus("failed", error=code)
        _emit("config.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status=manifest.graph.status)
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status=manifest.graph.status)
        return manifest

    try:
        loader = AlphaLoader(max_news_per_ticker=20, max_rss_per_feed=30, archive_provider=archive_provider)
        processor = AlphaProcessor()
        vector_store = AlphaVectorStore(supabase_url=supabase_url, supabase_key=supabase_key, config=vector_config)
        if historical and not vector_store.supports_temporal_boundary:
            raise ValueError("historical ingestion requires the V2 adapter")
    except Exception as exc:
        code = _capture_sanitized("config", exc)
        manifest.vector = StageStatus("failed", error=code)
        _emit("config.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status=manifest.graph.status)
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, vector_status="failed", graph_status=manifest.graph.status)
        return manifest

    load_started = time.perf_counter()
    _emit("load.start", base, started_at=load_started, status="started", degraded=False, vector_status=manifest.vector.status, graph_status=manifest.graph.status)
    try:
        raw_docs: list[RawDocument] = loader.load(
            tickers=tickers,
            start=normalized_start,
            end=normalized_end,
            as_of=normalized_as_of,
            dataset_version=dataset_version,
            historical=historical,
        )
        manifest.quarantine_count = len(loader.quarantine)
    except Exception as exc:
        code = _capture_sanitized("load", exc)
        manifest.vector = StageStatus("failed", error=code)
        _emit("load.failure", base, started_at=load_started, status="failed", degraded=True, error_code=code, document_count=0, quarantine_count=manifest.quarantine_count, vector_status="failed", graph_status=manifest.graph.status)
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, document_count=0, quarantine_count=manifest.quarantine_count, vector_status="failed", graph_status=manifest.graph.status)
        return manifest
    _emit("load.complete", base, started_at=load_started, status="complete" if raw_docs else "empty", degraded=not raw_docs, document_count=len(raw_docs), quarantine_count=manifest.quarantine_count, vector_status=manifest.vector.status, graph_status=manifest.graph.status)
    if not raw_docs:
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.complete", base, started_at=pipeline_started, status="empty", degraded=True, document_count=0, quarantine_count=manifest.quarantine_count, vector_status=manifest.vector.status, graph_status=manifest.graph.status)
        return manifest

    process_started = time.perf_counter()
    _emit("process.start", base, started_at=process_started, status="started", degraded=False, document_count=len(raw_docs), vector_status=manifest.vector.status, graph_status=manifest.graph.status)
    try:
        chunks: list[ProcessedChunk] = processor.process(raw_docs)
        manifest.chunk_count = len(chunks)
    except Exception as exc:
        code = _capture_sanitized("process", exc)
        manifest.vector = StageStatus("failed", error=code)
        _emit("process.failure", base, started_at=process_started, status="failed", degraded=True, error_code=code, document_count=len(raw_docs), chunk_count=0, vector_status="failed", graph_status=manifest.graph.status)
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, document_count=len(raw_docs), chunk_count=0, vector_status="failed", graph_status=manifest.graph.status)
        return manifest
    _emit("process.complete", base, started_at=process_started, status="complete" if chunks else "empty", degraded=not chunks, document_count=len(raw_docs), chunk_count=len(chunks), vector_status=manifest.vector.status, graph_status=manifest.graph.status)
    if not chunks:
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.complete", base, started_at=pipeline_started, status="empty", degraded=True, document_count=len(raw_docs), chunk_count=0, vector_status=manifest.vector.status, graph_status=manifest.graph.status)
        return manifest

    vector_started = time.perf_counter()
    _emit("vector.start", base, started_at=vector_started, status="started", degraded=False, chunk_count=len(chunks), vector_status=manifest.vector.status, graph_status=manifest.graph.status)
    try:
        embedder = get_embedder()
        embedding = _embedding_identity(embedder)
        materialization = _materialization_identity(
            request_fingerprint=request_fingerprint,
            provider_identity_hash=provider_identity_hash,
            adapter=adapter,
            embedding=embedding,
            raw_documents=raw_docs,
            chunks=chunks,
        )
    except Exception as exc:
        code = _capture_sanitized("vector", exc)
        manifest.vector = StageStatus("failed", error=code)
        _emit("vector.failure", base, started_at=vector_started, status="failed", degraded=True, error_code=code, chunk_count=len(chunks), vector_status="failed", graph_status=manifest.graph.status)
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, chunk_count=len(chunks), vector_status="failed", graph_status=manifest.graph.status)
        return manifest

    if loaded_manifest is not None:
        try:
            if not manifest.materialization_fingerprint:
                completed_work_exists = (
                    manifest.vector.status == "complete"
                    or manifest.graph.status == "complete"
                )
                if completed_work_exists:
                    raise CheckpointFingerprintMissing("completed checkpoint has no materialization fingerprint")
                for key, value in materialization.items():
                    setattr(manifest, key, value)
            else:
                for key, value in materialization.items():
                    if getattr(manifest, key) != value:
                        raise CheckpointSnapshotMismatch(f"checkpoint {key} mismatch")
        except Exception as exc:
            code = _capture_sanitized("resume", exc)
            manifest.graph = StageStatus("failed", error=code)
            _emit("resume.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, chunk_count=len(chunks), vector_status=manifest.vector.status, graph_status="failed", snapshot_fingerprint=materialization["materialization_fingerprint"])
            _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, chunk_count=len(chunks), vector_status=manifest.vector.status, graph_status="failed", snapshot_fingerprint=materialization["materialization_fingerprint"])
            return manifest
        if manifest.vector.status == "complete" and (manifest.graph.status == "complete" or skip_graph):
            _emit("resume.skip", base, started_at=pipeline_started, status="complete", degraded=False, chunk_count=len(chunks), upsert_count=manifest.vector.count, relation_count=manifest.graph.count, vector_status=manifest.vector.status, graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint)
            _emit("pipeline.complete", base, started_at=pipeline_started, status="complete", degraded=False, chunk_count=len(chunks), upsert_count=manifest.vector.count, relation_count=manifest.graph.count, vector_status=manifest.vector.status, graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint)
            return manifest
    else:
        for key, value in materialization.items():
            setattr(manifest, key, value)

    if manifest.vector.status != "complete":
        try:
            records = embedder.embed_chunks(chunks)
            count = vector_store.upsert(records=records)
            manifest.vector = StageStatus("complete", count=count)
        except Exception as exc:
            code = _capture_sanitized("vector", exc)
            manifest.vector = StageStatus("failed", error=code)
            _emit("vector.failure", base, started_at=vector_started, status="failed", degraded=True, error_code=code, chunk_count=len(chunks), vector_status="failed", graph_status=manifest.graph.status)
            _save_checkpoint(checkpoint, manifest, base)
            _emit("pipeline.failure", base, started_at=pipeline_started, status="failed", degraded=True, error_code=code, chunk_count=len(chunks), vector_status="failed", graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint)
            return manifest
        _emit("vector.complete", base, started_at=vector_started, status="complete", degraded=False, chunk_count=len(chunks), upsert_count=count, vector_status="complete", graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint, **_embedding_fields(embedding))
        _save_checkpoint(checkpoint, manifest, base)
    else:
        _emit("vector.resume", base, started_at=pipeline_started, status="skipped_complete", degraded=False, chunk_count=len(chunks), upsert_count=manifest.vector.count, vector_status="complete", graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint)

    if skip_graph:
        manifest.graph = StageStatus("skipped")
        _emit("graph.skip", base, started_at=pipeline_started, status="skipped_requested", degraded=True, vector_status=manifest.vector.status, graph_status="skipped", snapshot_fingerprint=manifest.materialization_fingerprint)
        _save_checkpoint(checkpoint, manifest, base)
        _emit("pipeline.complete", base, started_at=pipeline_started, status="complete", degraded=True, chunk_count=len(chunks), upsert_count=manifest.vector.count, vector_status=manifest.vector.status, graph_status="skipped", snapshot_fingerprint=manifest.materialization_fingerprint)
        return manifest

    graph_started = time.perf_counter()
    _emit("graph.start", base, started_at=graph_started, status="started", degraded=False, document_count=len(raw_docs), vector_status=manifest.vector.status, graph_status=manifest.graph.status, snapshot_fingerprint=manifest.materialization_fingerprint)
    graph_store = None
    try:
        graph_store = AlphaGraphStore()
        graph_store.connect()
        graph_docs = await graph_store.extract_batch(raw_docs)
        summary = graph_store.upsert_batch(graph_docs)
        relation_count = int(summary.get("rels_merged", 0))
        manifest.graph = StageStatus("complete", count=relation_count)
        _emit("graph.complete", base, started_at=graph_started, status="complete", degraded=False, document_count=len(raw_docs), relation_count=relation_count, vector_status=manifest.vector.status, graph_status="complete", snapshot_fingerprint=manifest.materialization_fingerprint)
    except Exception as exc:
        code = _capture_sanitized("graph", exc)
        manifest.graph = StageStatus("failed", error=code)
        _emit("graph.failure", base, started_at=graph_started, status="failed", degraded=True, error_code=code, document_count=len(raw_docs), vector_status=manifest.vector.status, graph_status="failed", snapshot_fingerprint=manifest.materialization_fingerprint)
    finally:
        if graph_store is not None:
            try:
                graph_store.close()
            except Exception as exc:
                code = _capture_sanitized("graph", exc)
                manifest.graph = StageStatus("failed", error=code)
        _save_checkpoint(checkpoint, manifest, base)
    final_event = "pipeline.complete" if manifest.graph.status == "complete" else "pipeline.failure"
    _emit(final_event, base, started_at=pipeline_started, status="complete" if manifest.graph.status == "complete" else "failed", degraded=manifest.graph.status != "complete", chunk_count=len(chunks), upsert_count=manifest.vector.count, relation_count=manifest.graph.count, vector_status=manifest.vector.status, graph_status=manifest.graph.status, error_code=manifest.graph.error, snapshot_fingerprint=manifest.materialization_fingerprint)
    return manifest


if __name__ == "__main__":
    init_sentry()
    raise SystemExit("Use `python -m rag.seed ...` for explicit ingestion; module import/startup never auto-seeds")


__all__ = [
    "CheckpointFingerprintMissing",
    "CheckpointSnapshotMismatch",
    "CheckpointStore",
    "IngestionManifest",
    "StageStatus",
    "run_ingestion_pipeline",
]
