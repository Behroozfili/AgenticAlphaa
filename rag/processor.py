"""Deterministic, revision-preserving chunk processing."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import RawDocument

logger = logging.getLogger(__name__)
_DEFAULT_MAX_SEEN = 100_000


@dataclass
class ProcessedChunk:
    text: str
    metadata: dict[str, Any]


@dataclass
class ProcessorMetrics:
    total_docs: int = 0
    chunks_created: int = 0
    duplicates_skipped: int = 0
    content_updates: int = 0

    def report(self) -> dict[str, int]:
        return {
            "total_docs": self.total_docs,
            "chunks_created": self.chunks_created,
            "duplicates_skipped": self.duplicates_skipped,
            "content_updates": self.content_updates,
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _url_hash(url: str) -> str:
    return _sha256(url)


def _metadata_hash(metadata: dict[str, Any]) -> str:
    """Hash the complete JSON metadata association without persisting a copy."""
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256(encoded)


class AlphaProcessor:
    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, max_seen: int = _DEFAULT_MAX_SEEN) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self._seen: dict[str, str] = {}
        # Exact evidence observations only. URL/content similarity must never
        # suppress a distinct revision, dataset, or retrieval observation.
        self._seen_revisions: set[str] = set()
        self._max_seen = max_seen
        self.metrics = ProcessorMetrics()

    def process(self, docs: list[RawDocument]) -> list[ProcessedChunk]:
        self.metrics = ProcessorMetrics()
        chunks: list[ProcessedChunk] = []
        for doc in docs:
            self.metrics.total_docs += 1
            chunks.extend(self._process_doc(doc))
        logger.info("AlphaProcessor metrics: %s", self.metrics.report())
        return chunks

    def _process_doc(self, doc: RawDocument) -> list[ProcessedChunk]:
        full_text = f"{doc.title}\n\n{doc.content}".strip()
        content_hash = _sha256(full_text)
        url_hash = _url_hash(doc.url)
        source_id = doc.source_id or _sha256(f"{doc.source_type}|{doc.url}")
        document_version = str(doc.revision or "1")
        document_id = _sha256("|".join((source_id, content_hash, document_version)))
        ticker = doc.ticker.strip().upper()
        association_hash = _sha256("|".join((ticker, _metadata_hash(doc.metadata))))
        evidence_id = _sha256("|".join((
            document_id,
            association_hash,
            doc.dataset_version,
            doc.published_at or "unknown",
            doc.retrieved_at or "unknown",
            doc.effective_from or "unknown",
            doc.effective_to or "open",
            str(doc.metadata.get("filed_at") or "unknown"),
        )))
        evidence_key = evidence_id

        if evidence_key in self._seen_revisions:
            self.metrics.duplicates_skipped += 1
            return []
        if url_hash in self._seen:
            if self._seen[url_hash] != content_hash:
                self.metrics.content_updates += 1
        self._add_to_seen(url_hash, content_hash)
        self._seen_revisions.add(evidence_key)

        result: list[ProcessedChunk] = []
        for index, text in enumerate(self.splitter.split_text(full_text)):
            text = text.strip()
            if not text:
                continue
            chunk_hash = _sha256(text)
            # Include the immutable evidence observation so a later retrieval
            # or dataset snapshot cannot collide with an earlier Vector row.
            chunk_id = _sha256("|".join((evidence_id, str(index), chunk_hash)))
            metadata: dict[str, Any] = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "evidence_id": evidence_id,
                "canonical_source_id": source_id,
                "source_id": source_id,
                "source_version": document_version,
                "document_version": document_version,
                "content_hash": content_hash,
                "chunk_hash": chunk_hash,
                "url_hash": url_hash,
                "ticker": ticker,
                "association_hash": association_hash,
                "source_type": doc.source_type,
                "published_at": doc.published_at,
                "published_at_utc": doc.published_at,
                "retrieved_at": doc.retrieved_at,
                "effective_from": doc.effective_from,
                "effective_to": doc.effective_to,
                "dataset_version": doc.dataset_version,
                "chunk_index": index,
                "url": doc.url,
                "title": doc.title,
                "ingested_at": doc.retrieved_at,
            }
            for key in (
                "filed_at", "fiscal_year", "fiscal_quarter", "period_start", "period_end",
                "duration_months", "filing_type", "accession_number", "metric_scope", "gaap_status",
            ):
                metadata[key] = doc.metadata.get(key)
            result.append(ProcessedChunk(text=text, metadata=metadata))
            self.metrics.chunks_created += 1
        return result

    def _add_to_seen(self, url_hash: str, content_hash: str) -> None:
        if len(self._seen) >= self._max_seen:
            evict_count = max(1, self._max_seen // 2)
            for key in list(self._seen)[:evict_count]:
                del self._seen[key]
            logger.warning("AlphaProcessor._seen cap reached (%d); evicted %d oldest entries.", self._max_seen, evict_count)
        self._seen[url_hash] = content_hash


__all__ = ["AlphaProcessor", "ProcessedChunk", "ProcessorMetrics", "_sha256", "_url_hash"]
