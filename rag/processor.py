"""
rag/processor.py — AlphaProcessor
Double-key idempotency (SHA256 content + URL hash), semantic boundary chunking,
metadata enrichment, and observability metrics.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.loader import RawDocument

logger = logging.getLogger(__name__)

# Maximum number of url_hash entries kept in the in-memory dedup store.
# When the cap is reached, the oldest half is evicted (FIFO-LRU).
# Override at construction time via max_seen= if needed.
_DEFAULT_MAX_SEEN: int = 100_000

# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------

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
            "total_docs":        self.total_docs,
            "chunks_created":    self.chunks_created,
            "duplicates_skipped": self.duplicates_skipped,
            "content_updates":   self.content_updates,
        }


# ---------------------------------------------------------------------------
# Hashing Utilities
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# AlphaProcessor
# ---------------------------------------------------------------------------

class AlphaProcessor:
    """
    Processes RawDocuments into enriched, deduplicated chunks.

    Idempotency logic (double-key):
    - content_hash  = SHA256(full document text)
    - url_hash      = SHA256(url)

    Decision table:
    ┌──────────────────────┬──────────────────────────────────┐
    │ Condition            │ Action                           │
    ├──────────────────────┼──────────────────────────────────┤
    │ Both match           │ Skip (exact duplicate)           │
    │ URL matches, content │ Update (new version of same doc) │
    │   differs            │                                  │
    │ Neither matches      │ Ingest as new                    │
    └──────────────────────┴──────────────────────────────────┘
    """

    # Semantic boundary separators — ordered from coarsest to finest
    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        max_seen: int = _DEFAULT_MAX_SEEN,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=self.SEPARATORS,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        # In-memory dedup store: url_hash -> content_hash
        # Capped at max_seen entries to prevent unbounded memory growth
        # in long-running ingestion processes (H-7 fix).
        # In production, replace with a Redis / Postgres lookup.
        self._seen: dict[str, str] = {}
        self._max_seen: int = max_seen
        self.metrics = ProcessorMetrics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, docs: list[RawDocument]) -> list[ProcessedChunk]:
        """
        Convert a batch of RawDocuments into deduplicated, enriched chunks.
        Returns the metrics alongside chunks (metrics accessible via .metrics).
        """
        self.metrics = ProcessorMetrics()
        all_chunks: list[ProcessedChunk] = []

        for doc in docs:
            self.metrics.total_docs += 1
            chunks = self._process_doc(doc)
            all_chunks.extend(chunks)

        logger.info("AlphaProcessor metrics: %s", self.metrics.report())
        return all_chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_doc(self, doc: RawDocument) -> list[ProcessedChunk]:
        full_text    = f"{doc.title}\n\n{doc.content}".strip()
        c_hash       = _sha256(full_text)
        u_hash       = _url_hash(doc.url)
        ingested_at  = datetime.now(timezone.utc).isoformat()

        # --- Double-key idempotency check ---
        if u_hash in self._seen:
            existing_c_hash = self._seen[u_hash]
            if existing_c_hash == c_hash:
                logger.debug("SKIP duplicate url=%s", doc.url)
                self.metrics.duplicates_skipped += 1
                return []
            else:
                logger.info("UPDATE new content for url=%s", doc.url)
                self.metrics.content_updates += 1
        # Record / update the seen state (via capped helper)
        self._add_to_seen(u_hash, c_hash)

        # --- Semantic boundary chunking ---
        raw_chunks = self.splitter.split_text(full_text)
        chunks: list[ProcessedChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            metadata: dict[str, Any] = {
                "content_hash":    c_hash,
                "url_hash":        u_hash,
                "ticker":          doc.ticker,
                "source_type":     doc.source_type,
                "published_at_utc": doc.published_at,
                "ingested_at":     ingested_at,
                "chunk_index":     idx,
                "url":             doc.url,
                "title":           doc.title,
            }
            chunks.append(ProcessedChunk(text=chunk_text, metadata=metadata))
            self.metrics.chunks_created += 1

        return chunks

    def _add_to_seen(self, u_hash: str, c_hash: str) -> None:
        """
        Store a url_hash → content_hash mapping with a size cap.

        When the store reaches _max_seen entries, the oldest half is
        evicted (FIFO). This keeps memory bounded for long-running
        ingestion processes without requiring an external cache.
        """
        if len(self._seen) >= self._max_seen:
            # Evict the oldest half — dict preserves insertion order (Python 3.7+)
            evict_count = self._max_seen // 2
            keys_to_evict = list(self._seen.keys())[:evict_count]
            for k in keys_to_evict:
                del self._seen[k]
            logger.warning(
                "AlphaProcessor._seen cap reached (%d) — evicted %d oldest entries.",
                self._max_seen, evict_count,
            )
        self._seen[u_hash] = c_hash