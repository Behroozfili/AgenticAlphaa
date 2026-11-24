"""
Tests for: rag/processor.py
Phase: 3 — RAG Pipeline (2nd: depends on rag/loader.py's RawDocument)

Mocking strategy: RecursiveCharacterTextSplitter is the real langchain
implementation (lightweight, pure-Python, deterministic) — we don't mock it,
since mocking it would defeat the purpose of testing chunking behavior.
We only construct RawDocument inputs directly (no I/O involved in this file
at all — it's pure transformation logic).
"""

from rag.loader import RawDocument
from rag.processor import AlphaProcessor, ProcessorMetrics, _sha256, _url_hash


def make_doc(title="Title", content="Content here.", url="https://x.com/1",
             source_type="news", ticker="NVDA",
             published_at="2024-03-15T14:32:00+00:00"):
    return RawDocument(title=title, content=content, url=url,
                       source_type=source_type, ticker=ticker, published_at=published_at)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

class TestHashHelpers:
    def test_sha256_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_sha256_differs_for_different_input(self):
        assert _sha256("hello") != _sha256("world")

    def test_url_hash_deterministic(self):
        assert _url_hash("https://x.com") == _url_hash("https://x.com")


# ---------------------------------------------------------------------------
# ProcessorMetrics
# ---------------------------------------------------------------------------

class TestProcessorMetrics:
    def test_report_shape(self):
        m = ProcessorMetrics(total_docs=5, chunks_created=10,
                              duplicates_skipped=2, content_updates=1)
        assert m.report() == {
            "total_docs": 5, "chunks_created": 10,
            "duplicates_skipped": 2, "content_updates": 1,
        }


# ---------------------------------------------------------------------------
# process() — idempotency decision table
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_new_document_is_ingested(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        chunks = proc.process([make_doc()])
        assert len(chunks) >= 1
        assert proc.metrics.total_docs == 1
        assert proc.metrics.duplicates_skipped == 0

    def test_exact_duplicate_is_skipped(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc = make_doc()
        proc.process([doc])
        chunks = proc.process([doc])  # identical doc again
        assert chunks == []
        assert proc.metrics.duplicates_skipped == 1

    def test_same_url_different_content_is_update(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc1 = make_doc(content="Original content.")
        doc2 = make_doc(content="Updated content, totally different.")
        proc.process([doc1])
        chunks = proc.process([doc2])
        assert len(chunks) >= 1
        assert proc.metrics.content_updates == 1

    def test_different_url_same_content_is_ingested_separately(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc1 = make_doc(url="https://x.com/1")
        doc2 = make_doc(url="https://x.com/2")  # same title/content, diff URL
        chunks1 = proc.process([doc1])
        chunks2 = proc.process([doc2])
        assert len(chunks1) >= 1
        assert len(chunks2) >= 1
        assert proc.metrics.duplicates_skipped == 0

    def test_metrics_reset_on_each_process_call(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        proc.process([make_doc(url="https://x.com/1")])
        assert proc.metrics.total_docs == 1
        proc.process([make_doc(url="https://x.com/2")])
        # metrics object is replaced each call, not cumulative
        assert proc.metrics.total_docs == 1


# ---------------------------------------------------------------------------
# process() — chunking + metadata enrichment
# ---------------------------------------------------------------------------

class TestChunkingAndMetadata:
    def test_chunk_metadata_contains_all_expected_keys(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc = make_doc()
        chunks = proc.process([doc])
        meta = chunks[0].metadata
        assert {
            "content_hash", "url_hash", "ticker", "source_type",
            "published_at_utc", "ingested_at", "chunk_index", "url", "title",
        } <= set(meta.keys())
        assert {
            "fiscal_year", "fiscal_quarter", "period_start", "period_end",
            "duration_months", "filing_type", "accession_number",
            "metric_scope", "gaap_status",
        } <= set(meta.keys())
        assert meta["ticker"] == "NVDA"
        assert meta["url"] == doc.url

    def test_chunk_index_increments_sequentially(self):
        proc = AlphaProcessor(chunk_size=20, chunk_overlap=0)  # tiny chunks -> multiple
        long_content = "This is sentence one. " * 20
        doc = make_doc(content=long_content)
        chunks = proc.process([doc])
        assert len(chunks) > 1
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_chunks_after_strip_are_excluded(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc = make_doc(title="", content="   ")  # effectively empty after join+strip
        chunks = proc.process([doc])
        assert chunks == []
        # total_docs is still counted even if it produces 0 chunks
        assert proc.metrics.total_docs == 1
        assert proc.metrics.chunks_created == 0

    def test_title_and_content_joined_with_blank_line(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc = make_doc(title="MyTitle", content="MyBody")
        chunks = proc.process([doc])
        assert "MyTitle" in chunks[0].text
        assert "MyBody" in chunks[0].text

    def test_multiple_docs_processed_independently(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        doc1 = make_doc(url="https://x.com/1", ticker="NVDA")
        doc2 = make_doc(url="https://x.com/2", ticker="AAPL")
        chunks = proc.process([doc1, doc2])
        tickers = {c.metadata["ticker"] for c in chunks}
        assert tickers == {"NVDA", "AAPL"}
        assert proc.metrics.total_docs == 2


# ---------------------------------------------------------------------------
# _add_to_seen — capped FIFO eviction (memory bound)
# ---------------------------------------------------------------------------

class TestSeenCapEviction:
    def test_eviction_triggers_at_cap_and_halves_store(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0, max_seen=4)
        # Fill up to the cap with 4 distinct url_hashes
        for i in range(4):
            proc._add_to_seen(f"url_hash_{i}", f"content_hash_{i}")
        assert len(proc._seen) == 4

        # 5th insertion should trigger eviction of the oldest half (2 entries)
        proc._add_to_seen("url_hash_4", "content_hash_4")
        assert len(proc._seen) == 3  # 4 - 2 evicted + 1 new = 3

    def test_oldest_entries_are_evicted_first(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0, max_seen=4)
        for i in range(4):
            proc._add_to_seen(f"url_{i}", f"content_{i}")
        proc._add_to_seen("url_4", "content_4")
        # url_0 and url_1 (the oldest) should have been evicted
        assert "url_0" not in proc._seen
        assert "url_1" not in proc._seen
        assert "url_4" in proc._seen

    def test_below_cap_no_eviction(self):
        proc = AlphaProcessor(chunk_size=512, chunk_overlap=0, max_seen=100)
        for i in range(5):
            proc._add_to_seen(f"url_{i}", f"content_{i}")
        assert len(proc._seen) == 5


class TestRevisionPreservation:
    def test_changed_content_creates_distinct_immutable_revision_ids(self):
        first = make_doc(content="revision one")
        first.retrieved_at = "2024-03-16T00:00:00+00:00"
        first.effective_from = "2024-03-15T14:32:00+00:00"
        first.source_id = "a" * 64
        first.dataset_version = "snapshot-v1"
        first.revision = "1"
        second = make_doc(content="revision two")
        second.retrieved_at = "2024-03-17T00:00:00+00:00"
        second.effective_from = "2024-03-16T00:00:00+00:00"
        second.source_id = "a" * 64
        second.dataset_version = "snapshot-v1"
        second.revision = "2"
        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])
        assert len(chunks) == 2
        assert len({chunk.metadata["document_id"] for chunk in chunks}) == 2
        assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == 2
        assert [chunk.metadata["effective_from"] for chunk in chunks] == [first.effective_from, second.effective_from]

    def test_shared_ids_are_deterministic_across_processor_instances(self):
        doc = make_doc()
        doc.source_id = "b" * 64
        doc.retrieved_at = doc.published_at
        doc.effective_from = doc.published_at
        left = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([doc])[0]
        right = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([doc])[0]
        assert left.metadata["document_id"] == right.metadata["document_id"]
        assert left.metadata["chunk_id"] == right.metadata["chunk_id"]

    def test_same_content_distinct_document_versions_are_preserved(self):
        first = make_doc(content="same immutable content")
        second = make_doc(content="same immutable content")
        for doc in (first, second):
            doc.source_id = "c" * 64
            doc.dataset_version = "snapshot-v1"
            doc.retrieved_at = "2024-03-16T00:00:00+00:00"
            doc.effective_from = doc.published_at
        first.revision = "1"
        second.revision = "2"
        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])
        assert len(chunks) == 2
        assert len({chunk.metadata["document_id"] for chunk in chunks}) == 2
        assert len({chunk.metadata["evidence_id"] for chunk in chunks}) == 2
        assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == 2

    def test_same_content_distinct_dataset_versions_are_preserved(self):
        first = make_doc(content="same immutable content")
        second = make_doc(content="same immutable content")
        for doc in (first, second):
            doc.source_id = "d" * 64
            doc.revision = "1"
            doc.retrieved_at = "2024-03-16T00:00:00+00:00"
            doc.effective_from = doc.published_at
        first.dataset_version = "snapshot-v1"
        second.dataset_version = "snapshot-v2"
        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])
        assert len(chunks) == 2
        assert chunks[0].metadata["document_id"] == chunks[1].metadata["document_id"]
        assert chunks[0].metadata["evidence_id"] != chunks[1].metadata["evidence_id"]
        assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]

    def test_same_content_later_retrieval_observation_is_preserved(self):
        first = make_doc(content="same immutable content")
        second = make_doc(content="same immutable content")
        for doc in (first, second):
            doc.source_id = "e" * 64
            doc.revision = "1"
            doc.dataset_version = "snapshot-v1"
            doc.effective_from = doc.published_at
        first.retrieved_at = "2024-03-16T00:00:00+00:00"
        second.retrieved_at = "2024-03-17T00:00:00+00:00"
        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])
        assert len(chunks) == 2
        assert chunks[0].metadata["document_id"] == chunks[1].metadata["document_id"]
        assert chunks[0].metadata["evidence_id"] != chunks[1].metadata["evidence_id"]
        assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]

    def test_same_content_distinct_effective_observation_is_preserved(self):
        first = make_doc(content="same immutable content")
        second = make_doc(content="same immutable content")
        for doc in (first, second):
            doc.source_id = "9" * 64
            doc.revision = "1"
            doc.dataset_version = "snapshot-v1"
            doc.retrieved_at = "2024-03-16T00:00:00+00:00"
        first.effective_from = "2024-03-15T00:00:00+00:00"
        second.effective_from = "2024-03-16T00:00:00+00:00"
        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])
        assert len(chunks) == 2
        assert chunks[0].metadata["document_id"] == chunks[1].metadata["document_id"]
        assert chunks[0].metadata["evidence_id"] != chunks[1].metadata["evidence_id"]
        assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]

    def test_only_exact_evidence_identity_is_deduplicated(self):
        doc = make_doc(content="same immutable content")
        doc.source_id = "f" * 64
        doc.revision = "1"
        doc.dataset_version = "snapshot-v1"
        doc.retrieved_at = "2024-03-16T00:00:00+00:00"
        doc.effective_from = doc.published_at
        processor = AlphaProcessor(chunk_size=512, chunk_overlap=0)
        chunks = processor.process([doc, doc])
        assert len(chunks) == 1
        assert processor.metrics.duplicates_skipped == 1

    def test_same_evidence_associated_with_multiple_tickers_is_preserved(self):
        first = make_doc(content="shared evidence", ticker="NVDA")
        second = make_doc(content="shared evidence", ticker="AMD")
        for doc in (first, second):
            doc.source_id = "8" * 64
            doc.dataset_version = "snapshot-v1"
            doc.retrieved_at = "2024-03-16T00:00:00+00:00"
            doc.effective_from = doc.published_at

        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])

        assert len(chunks) == 2
        assert chunks[0].metadata["document_id"] == chunks[1].metadata["document_id"]
        assert {chunk.metadata["ticker"] for chunk in chunks} == {"NVDA", "AMD"}
        assert len({chunk.metadata["association_hash"] for chunk in chunks}) == 2
        assert len({chunk.metadata["evidence_id"] for chunk in chunks}) == 2
        assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == 2

    def test_metadata_change_creates_distinct_evidence_observation(self):
        first = make_doc(content="shared evidence")
        second = make_doc(content="shared evidence")
        for doc in (first, second):
            doc.source_id = "7" * 64
            doc.dataset_version = "snapshot-v1"
            doc.retrieved_at = "2024-03-16T00:00:00+00:00"
            doc.effective_from = doc.published_at
        first.metadata = {"association": "primary"}
        second.metadata = {"association": "secondary"}

        chunks = AlphaProcessor(chunk_size=512, chunk_overlap=0).process([first, second])

        assert len(chunks) == 2
        assert chunks[0].metadata["document_id"] == chunks[1].metadata["document_id"]
        assert chunks[0].metadata["evidence_id"] != chunks[1].metadata["evidence_id"]
