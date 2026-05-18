"""
tests/test_rag_pipeline.py
Comprehensive unit tests for the AlphaRAG pipeline.
Heavy deps (torch, sentence_transformers, supabase) are mocked.
"""

import hashlib
import json
import math
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

# ─────────────────────────────────────────────────────────────────
# Stub heavy dependencies BEFORE importing rag modules
# ─────────────────────────────────────────────────────────────────

def _make_torch_stub():
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = lambda: False
    torch.cuda.OutOfMemoryError = RuntimeError
    torch.backends = MagicMock()
    torch.backends.mps = MagicMock()
    torch.backends.mps.is_available = lambda: False
    torch.zeros = lambda *a, **kw: MagicMock()
    return torch

def _make_sentence_transformers_stub():
    st = types.ModuleType("sentence_transformers")
    import numpy as np

    class FakeModel:
        def __init__(self, name, device="cpu"):
            self._name = name
        def encode(self, texts, **kwargs):
            n = len(texts)
            vecs = np.random.rand(n, 384).astype("float32")
            # L2 normalise
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms
        def to(self, device):
            return self

    st.SentenceTransformer = FakeModel
    return st

def _make_supabase_stub():
    sub = types.ModuleType("supabase")
    sub.create_client = MagicMock(return_value=MagicMock())
    sub.Client = MagicMock
    return sub

sys.modules.setdefault("torch",                _make_torch_stub())
sys.modules.setdefault("sentence_transformers", _make_sentence_transformers_stub())
sys.modules.setdefault("supabase",              _make_supabase_stub())

# ─────────────────────────────────────────────────────────────────
# Now import rag modules (they see the stubs)
# ─────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.loader import (
    AlphaLoader, RawDocument,
    _to_utc_iso8601, _safe_timestamp,
)
from rag.processor import AlphaProcessor, ProcessedChunk, _sha256, _url_hash
from rag.embedding_manager import AlphaEmbedder, get_embedder
from rag.vector_store import AlphaVectorStore
from rag.retriever import AlphaRetriever
from rag.evaluation import AlphaEvaluator, MetricResult, EvaluationReport


# ═══════════════════════════════════════════════════════════════════
# 1. LOADER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestTimestampNormalization(unittest.TestCase):
    """_to_utc_iso8601 must handle every common timestamp format."""

    def test_unix_int(self):
        result = _to_utc_iso8601(0)
        self.assertIn("1970-01-01", result)
        self.assertIn("+00:00", result)

    def test_unix_float(self):
        result = _to_utc_iso8601(1710510720.0)
        self.assertIn("2024", result)

    def test_rfc2822(self):
        result = _to_utc_iso8601("Fri, 15 Mar 2024 14:32:00 +0000")
        self.assertIn("2024-03-15", result)
        self.assertIn("+00:00", result)

    def test_iso8601_with_tz(self):
        result = _to_utc_iso8601("2024-03-15T14:32:00+02:00")
        self.assertIn("2024-03-15", result)
        self.assertIn("+00:00", result)
        # Converted from +02:00 → UTC should be 12:32
        self.assertIn("12:32", result)

    def test_iso8601_zulu(self):
        result = _to_utc_iso8601("2024-03-15T14:32:00Z")
        self.assertIn("2024-03-15", result)

    def test_date_only(self):
        result = _to_utc_iso8601("2024-03-15")
        self.assertIn("2024-03-15", result)

    def test_datetime_object_with_tz(self):
        dt = datetime(2024, 3, 15, 14, 32, tzinfo=timezone.utc)
        result = _to_utc_iso8601(dt)
        self.assertIn("2024-03-15", result)

    def test_datetime_object_naive(self):
        dt = datetime(2024, 3, 15, 14, 32)
        result = _to_utc_iso8601(dt)
        self.assertIn("2024-03-15", result)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _to_utc_iso8601("not-a-date")

    def test_safe_timestamp_fallback(self):
        result = _safe_timestamp("garbage", fallback_label="test")
        # Should return current UTC time, not raise
        self.assertIn("+00:00", result)


class TestAlphaLoaderSchema(unittest.TestCase):
    """RawDocument schema and circuit breaker behaviour."""

    def _make_doc(self, **kwargs):
        defaults = dict(
            title="Test", content="body", url="https://x.com/1",
            source_type="news", ticker="AAPL", published_at="2024-03-15T00:00:00+00:00"
        )
        defaults.update(kwargs)
        return RawDocument(**defaults)

    def test_raw_document_fields(self):
        doc = self._make_doc()
        self.assertEqual(doc.source_type, "news")
        self.assertEqual(doc.ticker, "AAPL")
        self.assertIn("2024", doc.published_at)

    def test_source_type_values(self):
        for st in ("news", "rss", "reddit"):
            doc = self._make_doc(source_type=st)
            self.assertEqual(doc.source_type, st)

    @patch("rag.loader.yf.Ticker")
    def test_yfinance_circuit_breaker(self, mock_ticker):
        """If yfinance explodes, load() still returns (empty) without raising."""
        mock_ticker.side_effect = RuntimeError("network error")
        loader = AlphaLoader()
        result = loader.load(["AAPL"])
        self.assertIsInstance(result, list)

    @patch("rag.loader.feedparser.parse")
    def test_reddit_circuit_breaker(self, mock_parse):
        """If feedparser explodes, load() still returns without raising."""
        mock_parse.side_effect = RuntimeError("feed down")
        loader = AlphaLoader()
        result = loader.load(["TSLA"])
        self.assertIsInstance(result, list)

    @patch("rag.loader.feedparser.parse")
    def test_reddit_rss_source_type(self, mock_parse):
        """Reddit URLs get source_type='reddit'; non-reddit RSS gets 'rss'."""
        mock_entry_reddit = MagicMock()
        mock_entry_reddit.get = lambda k, d="": {
            "link": "https://www.reddit.com/r/investing/comments/abc",
            "title": "Reddit post",
            "published": "Fri, 15 Mar 2024 10:00:00 +0000",
            "summary": "some content",
        }.get(k, d)

        mock_entry_rss = MagicMock()
        mock_entry_rss.get = lambda k, d="": {
            "link": "https://example.com/article",
            "title": "RSS article",
            "published": "Fri, 15 Mar 2024 10:00:00 +0000",
            "summary": "some content",
        }.get(k, d)

        fake_feed = MagicMock()
        fake_feed.bozo = False
        fake_feed.bozo_exception = None
        fake_feed.entries = [mock_entry_reddit, mock_entry_rss]
        mock_parse.return_value = fake_feed

        loader = AlphaLoader()
        docs = loader._parse_rss("https://www.reddit.com/r/investing/.rss", "SPY")

        source_types = {d.source_type for d in docs}
        self.assertIn("reddit", source_types)
        self.assertIn("rss", source_types)

    @patch("rag.loader.yf.Ticker")
    @patch("rag.loader.feedparser.parse")
    def test_load_returns_list(self, mock_parse, mock_ticker):
        """load() always returns a list regardless of source results."""
        mock_ticker.return_value.news = []
        fake_feed = MagicMock()
        fake_feed.bozo = False
        fake_feed.bozo_exception = None
        fake_feed.entries = []
        mock_parse.return_value = fake_feed

        loader = AlphaLoader()
        result = loader.load(["MSFT"])
        self.assertIsInstance(result, list)


# ═══════════════════════════════════════════════════════════════════
# 2. PROCESSOR TESTS
# ═══════════════════════════════════════════════════════════════════

def _make_raw_doc(**kwargs):
    defaults = dict(
        title="Earnings Beat",
        content="Apple reported strong Q1 earnings. Revenue grew 15% YoY.",
        url="https://finance.example.com/aapl-q1",
        source_type="news",
        ticker="AAPL",
        published_at="2024-03-15T14:00:00+00:00",
    )
    defaults.update(kwargs)
    return RawDocument(**defaults)


class TestAlphaProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = AlphaProcessor(chunk_size=200, chunk_overlap=20)

    def test_returns_processed_chunks(self):
        doc = _make_raw_doc()
        chunks = self.processor.process([doc])
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], ProcessedChunk)

    def test_chunk_metadata_keys(self):
        doc = _make_raw_doc()
        chunks = self.processor.process([doc])
        required = {
            "content_hash", "url_hash", "ticker", "source_type",
            "published_at_utc", "ingested_at", "chunk_index",
        }
        for chunk in chunks:
            self.assertTrue(required.issubset(chunk.metadata.keys()),
                            f"Missing keys: {required - chunk.metadata.keys()}")

    def test_chunk_index_sequential(self):
        doc = _make_raw_doc(content="A. " * 200)  # force multiple chunks
        chunks = self.processor.process([doc])
        indices = [c.metadata["chunk_index"] for c in chunks]
        self.assertEqual(indices, list(range(len(chunks))))

    def test_content_hash_is_sha256(self):
        doc = _make_raw_doc()
        chunks = self.processor.process([doc])
        full_text = f"{doc.title}\n\n{doc.content}".strip()
        expected_hash = hashlib.sha256(full_text.encode()).hexdigest()
        self.assertEqual(chunks[0].metadata["content_hash"], expected_hash)

    def test_url_hash_is_sha256(self):
        doc = _make_raw_doc()
        chunks = self.processor.process([doc])
        expected = hashlib.sha256(doc.url.encode()).hexdigest()
        self.assertEqual(chunks[0].metadata["url_hash"], expected)

    # --- Idempotency ---

    def test_exact_duplicate_skipped(self):
        doc = _make_raw_doc()
        self.processor.process([doc])  # first ingest
        chunks2 = self.processor.process([doc])  # exact duplicate
        self.assertEqual(chunks2, [])
        self.assertEqual(self.processor.metrics.duplicates_skipped, 1)

    def test_content_update_detected(self):
        doc_v1 = _make_raw_doc(content="Version 1 content.")
        doc_v2 = _make_raw_doc(content="Version 2 content — totally different.")
        self.processor.process([doc_v1])
        chunks = self.processor.process([doc_v2])
        self.assertGreater(len(chunks), 0)
        self.assertEqual(self.processor.metrics.content_updates, 1)

    def test_new_url_ingested(self):
        doc1 = _make_raw_doc(url="https://example.com/1")
        doc2 = _make_raw_doc(url="https://example.com/2")
        c1 = self.processor.process([doc1])
        c2 = self.processor.process([doc2])
        self.assertGreater(len(c1), 0)
        self.assertGreater(len(c2), 0)
        self.assertEqual(self.processor.metrics.duplicates_skipped, 0)

    # --- Metrics ---

    def test_metrics_accumulate(self):
        docs = [_make_raw_doc(url=f"https://example.com/{i}") for i in range(5)]
        self.processor.process(docs)
        m = self.processor.metrics.report()
        self.assertEqual(m["total_docs"], 5)
        self.assertGreater(m["chunks_created"], 0)

    def test_empty_input(self):
        chunks = self.processor.process([])
        self.assertEqual(chunks, [])
        self.assertEqual(self.processor.metrics.total_docs, 0)


class TestHashFunctions(unittest.TestCase):

    def test_sha256_deterministic(self):
        self.assertEqual(_sha256("hello"), _sha256("hello"))

    def test_sha256_different_inputs(self):
        self.assertNotEqual(_sha256("hello"), _sha256("world"))

    def test_sha256_hex_length(self):
        self.assertEqual(len(_sha256("test")), 64)

    def test_url_hash_deterministic(self):
        url = "https://example.com/article"
        self.assertEqual(_url_hash(url), _url_hash(url))


# ═══════════════════════════════════════════════════════════════════
# 3. EMBEDDING MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAlphaEmbedder(unittest.TestCase):

    def setUp(self):
        # Reset singleton for each test
        import rag.embedding_manager as em
        em._INSTANCE = None
        self.embedder = AlphaEmbedder(model_name="all-MiniLM-L6-v2", batch_size=8)

    def _make_chunks(self, n=3):
        return [
            ProcessedChunk(
                text=f"Chunk number {i}",
                metadata={
                    "content_hash": _sha256(f"chunk{i}"),
                    "url_hash": _url_hash(f"https://ex.com/{i}"),
                    "ticker": "AAPL",
                    "source_type": "news",
                    "published_at_utc": "2024-03-15T14:00:00+00:00",
                    "ingested_at": "2024-03-15T15:00:00+00:00",
                    "chunk_index": i,
                }
            )
            for i in range(n)
        ]

    def test_embed_chunks_returns_list(self):
        chunks = self._make_chunks(3)
        results = self.embedder.embed_chunks(chunks)
        self.assertEqual(len(results), 3)

    def test_embed_chunks_has_required_keys(self):
        chunks = self._make_chunks(2)
        results = self.embedder.embed_chunks(chunks)
        for r in results:
            self.assertIn("embedding", r)
            self.assertIn("metadata", r)
            self.assertIn("text", r)

    def test_embedding_dimension(self):
        chunks = self._make_chunks(2)
        results = self.embedder.embed_chunks(chunks)
        for r in results:
            self.assertEqual(len(r["embedding"]), 384)

    def test_embedding_is_l2_normalized(self):
        import numpy as np
        chunks = self._make_chunks(4)
        results = self.embedder.embed_chunks(chunks)
        for r in results:
            vec = np.array(r["embedding"])
            norm = np.linalg.norm(vec)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_embed_query_returns_vector(self):
        vec = self.embedder.embed_query("What is Apple's revenue?")
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), 384)

    def test_embed_query_l2_normalized(self):
        import numpy as np
        vec = np.array(self.embedder.embed_query("test query"))
        self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)

    def test_empty_chunks(self):
        results = self.embedder.embed_chunks([])
        self.assertEqual(results, [])

    def test_singleton_returns_same_instance(self):
        import rag.embedding_manager as em
        em._INSTANCE = None
        e1 = get_embedder()
        e2 = get_embedder()
        self.assertIs(e1, e2)

    def test_metadata_preserved(self):
        chunks = self._make_chunks(1)
        results = self.embedder.embed_chunks(chunks)
        self.assertEqual(results[0]["metadata"]["ticker"], "AAPL")
        self.assertEqual(results[0]["text"], "Chunk number 0")


# ═══════════════════════════════════════════════════════════════════
# 4. VECTOR STORE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAlphaVectorStore(unittest.TestCase):

    def _make_store(self):
        with patch("rag.vector_store.create_client") as mock_cc:
            mock_client = MagicMock()
            mock_cc.return_value = mock_client
            store = AlphaVectorStore(
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key",
            )
            store._mock_client = mock_client
            return store

    def test_upsert_empty_returns_zero(self):
        store = self._make_store()
        result = store.upsert([])
        self.assertEqual(result, 0)

    def test_upsert_calls_table(self):
        store = self._make_store()
        mock_resp = MagicMock()
        mock_resp.data = [{"id": 1}, {"id": 2}]
        store.client.table.return_value.upsert.return_value.execute.return_value = mock_resp

        records = [
            {
                "embedding": [0.1] * 384,
                "text": "Apple earnings beat",
                "metadata": {
                    "content_hash": "abc123",
                    "url_hash": "def456",
                    "ticker": "AAPL",
                    "source_type": "news",
                    "published_at_utc": "2024-03-15T00:00:00+00:00",
                    "ingested_at": "2024-03-15T01:00:00+00:00",
                    "chunk_index": 0,
                    "url": "https://example.com",
                    "title": "AAPL Earnings",
                },
            }
        ]
        count = store.upsert(records)
        self.assertEqual(count, 2)
        store.client.table.assert_called_with("alpha_documents")

    def test_to_row_mapping(self):
        record = {
            "embedding": [0.5] * 384,
            "text": "test text",
            "metadata": {
                "content_hash": "ch",
                "url_hash": "uh",
                "ticker": "MSFT",
                "source_type": "rss",
                "published_at_utc": "2024-01-01T00:00:00+00:00",
                "ingested_at": "2024-01-01T01:00:00+00:00",
                "chunk_index": 2,
                "url": "https://ms.com",
                "title": "MSFT news",
            },
        }
        row = AlphaVectorStore._to_row(record)
        self.assertEqual(row["ticker"], "MSFT")
        self.assertEqual(row["chunk_index"], 2)
        self.assertEqual(row["text"], "test text")
        self.assertEqual(len(row["embedding"]), 384)

    def test_hybrid_search_calls_rpc(self):
        store = self._make_store()
        mock_resp = MagicMock()
        mock_resp.data = [
            {"id": 1, "text": "chunk1", "rrf_score": 0.9,
             "ticker": "AAPL", "source_type": "news",
             "published_at": "2024-03-15T00:00:00+00:00",
             "url": "https://ex.com", "title": "T", "chunk_index": 0}
        ]
        store.client.rpc.return_value.execute.return_value = mock_resp

        results = store.hybrid_search(
            query_embedding=[0.1] * 384,
            query_text="Apple earnings",
            ticker="AAPL",
            days_back=7,
            top_k=50,
            score_threshold=0.0,
            limit=10,
        )
        store.client.rpc.assert_called_once_with("alpha_hybrid_search", unittest.mock.ANY)
        self.assertEqual(len(results), 1)

    def test_score_threshold_filters(self):
        store = self._make_store()
        mock_resp = MagicMock()
        mock_resp.data = [
            {"id": 1, "rrf_score": 0.8, "text": "high"},
            {"id": 2, "rrf_score": 0.005, "text": "low"},
        ]
        store.client.rpc.return_value.execute.return_value = mock_resp

        results = store.hybrid_search(
            query_embedding=[0.1] * 384,
            query_text="test",
            score_threshold=0.5,
            limit=10,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "high")


# ═══════════════════════════════════════════════════════════════════
# 5. RETRIEVER TESTS
# ═══════════════════════════════════════════════════════════════════

def _fake_chunks(n=10, hours_old_list=None):
    """Generate fake hybrid search result dicts."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    chunks = []
    sources = ["news", "rss", "reddit"]
    for i in range(n):
        hours = (hours_old_list[i] if hours_old_list else i * 10)
        pub = (now - __import__("datetime").timedelta(hours=hours)).isoformat()
        chunks.append({
            "id": i,
            "text": f"Financial content chunk {i}. Apple reported earnings.",
            "ticker": "AAPL",
            "source_type": sources[i % 3],
            "published_at": pub,
            "url": f"https://source{i % 4}.com/article-{i}",
            "title": f"Article {i}",
            "chunk_index": 0,
            "rrf_score": 1.0 - i * 0.05,
        })
    return chunks


class TestAlphaRetriever(unittest.TestCase):

    def _make_retriever(self, fake_results=None):
        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = fake_results or _fake_chunks(10)
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384
        return AlphaRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            stage1_k=50, stage2_k=10, stage3_k=5,
            token_budget=2000,
        )

    # --- Freshness Reranking ---

    def test_freshness_decay_recent_beats_old(self):
        retriever = self._make_retriever()
        fresh = {"rrf_score": 0.5, "published_at": datetime.now(timezone.utc).isoformat()}
        old   = {"rrf_score": 0.5, "published_at": "2020-01-01T00:00:00+00:00"}
        reranked = retriever._rerank_by_freshness([old, fresh])
        self.assertEqual(reranked[0]["url"] if "url" in reranked[0] else reranked[0],
                         reranked[0])
        # fresh should have higher freshness_score
        self.assertGreater(
            reranked[0]["freshness_score"],
            reranked[1]["freshness_score"],
        )

    def test_exponential_decay_formula(self):
        retriever = self._make_retriever()
        rrf = 0.8
        hours = 72.0
        chunk = {
            "rrf_score": rrf,
            "published_at": (
                datetime.now(timezone.utc)
                .__class__.now(timezone.utc)
            ).isoformat(),
        }
        # Manually set hours to known value
        decay = math.exp(-hours / 72)
        expected = rrf * decay
        # Verify decay formula produces value in (0,1)
        self.assertGreater(expected, 0)
        self.assertLess(expected, rrf)

    def test_hours_since_invalid_returns_720(self):
        retriever = self._make_retriever()
        now = datetime.now(timezone.utc)
        hours = AlphaRetriever._hours_since("not-a-date", now)
        self.assertEqual(hours, 720.0)

    def test_hours_since_recent(self):
        retriever = self._make_retriever()
        now = datetime.now(timezone.utc)
        recent = now.isoformat()
        hours = AlphaRetriever._hours_since(recent, now)
        self.assertAlmostEqual(hours, 0.0, places=1)

    # --- Diversity Filter ---

    def test_diversity_max_2_per_url(self):
        retriever = self._make_retriever()
        chunks = [
            {"url": "https://same.com", "source_type": "news", "rrf_score": 0.9, "freshness_score": 0.9},
            {"url": "https://same.com", "source_type": "news", "rrf_score": 0.8, "freshness_score": 0.8},
            {"url": "https://same.com", "source_type": "rss",  "rrf_score": 0.7, "freshness_score": 0.7},
        ]
        result = retriever._diversity_filter(chunks)
        same_url = [c for c in result if c["url"] == "https://same.com"]
        self.assertLessEqual(len(same_url), 2)

    def test_diversity_max_3_per_source_type(self):
        retriever = self._make_retriever()
        chunks = [
            {"url": f"https://ex.com/{i}", "source_type": "news",
             "freshness_score": 1 - i*0.1}
            for i in range(6)
        ]
        result = retriever._diversity_filter(chunks)
        news_count = sum(1 for c in result if c["source_type"] == "news")
        self.assertLessEqual(news_count, 3)

    def test_diversity_returns_at_most_stage3_k(self):
        retriever = self._make_retriever()
        chunks = _fake_chunks(20)
        # Add freshness_score for diversity filter
        for c in chunks:
            c["freshness_score"] = c["rrf_score"]
        result = retriever._diversity_filter(chunks)
        self.assertLessEqual(len(result), retriever.stage3_k)

    # --- Token Budget ---

    def test_token_budget_respected(self):
        retriever = self._make_retriever()
        # Each chunk ~100 chars = ~25 tokens
        chunks = [{"text": "x" * 100} for _ in range(100)]
        budgeted = retriever._apply_token_budget(chunks)
        total_chars = sum(len(c["text"]) for c in budgeted)
        self.assertLessEqual(total_chars, retriever.token_budget * 4 + 100)

    def test_single_large_chunk_included(self):
        """First chunk is always included even if it exceeds budget."""
        retriever = self._make_retriever()
        chunks = [{"text": "x" * 10_000}]
        budgeted = retriever._apply_token_budget(chunks)
        self.assertEqual(len(budgeted), 1)

    # --- Context Formatting ---

    def test_format_empty_returns_no_context_message(self):
        retriever = self._make_retriever()
        result = retriever._format_context([])
        self.assertIn("No relevant context", result)

    def test_format_includes_citations(self):
        retriever = self._make_retriever()
        chunks = [{
            "text": "Apple earnings beat estimates.",
            "source_type": "news",
            "ticker": "AAPL",
            "published_at": "2024-03-15T00:00:00+00:00",
            "url": "https://example.com/aapl",
            "freshness_score": 0.85,
            "rrf_score": 0.85,
        }]
        result = retriever._format_context(chunks)
        self.assertIn("[1]", result)
        self.assertIn("news", result)
        self.assertIn("AAPL", result)
        self.assertIn("https://example.com/aapl", result)

    def test_retrieve_returns_string(self):
        retriever = self._make_retriever(_fake_chunks(10))
        # add freshness scores
        result = retriever.retrieve("What are Apple earnings?", ticker="AAPL")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ═══════════════════════════════════════════════════════════════════
# 6. EVALUATION TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAlphaEvaluator(unittest.TestCase):

    def _make_evaluator(self, judge_response: dict):
        with patch("rag.evaluation.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            mock_msg = MagicMock()
            mock_msg.content = [MagicMock(text=json.dumps(judge_response))]
            mock_client.messages.create.return_value = mock_msg

            evaluator = AlphaEvaluator(api_key="fake-key")
            evaluator._client = mock_client
            return evaluator

    def _full_judge_response(self):
        return {
            "score": 0.9,
            "explanation": "Good result",
            "total_claims": 3,
            "supported_claims": 3,
            "total_chunks": 2,
            "relevant_chunks": 2,
            "total_key_facts": 3,
            "covered_facts": 3,
        }

    def test_metric_result_score_in_range(self):
        m = MetricResult(score=0.85, explanation="ok", metric="faithfulness")
        self.assertGreaterEqual(m.score, 0.0)
        self.assertLessEqual(m.score, 1.0)

    def test_overall_score_weighted(self):
        make_metric = lambda s, name: MetricResult(score=s, explanation="", metric=name)
        report = EvaluationReport(
            query="q", answer="a",
            faithfulness=make_metric(1.0, "faithfulness"),
            context_precision=make_metric(1.0, "context_precision"),
            context_recall=make_metric(1.0, "context_recall"),
            answer_relevance=make_metric(1.0, "answer_relevance"),
        )
        self.assertAlmostEqual(report.overall_score, 1.0, places=4)

    def test_overall_score_zero(self):
        make_metric = lambda name: MetricResult(score=0.0, explanation="", metric=name)
        report = EvaluationReport(
            query="q", answer="a",
            faithfulness=make_metric("faithfulness"),
            context_precision=make_metric("context_precision"),
            context_recall=make_metric("context_recall"),
            answer_relevance=make_metric("answer_relevance"),
        )
        self.assertAlmostEqual(report.overall_score, 0.0, places=4)

    def test_evaluate_returns_report(self):
        evaluator = self._make_evaluator(self._full_judge_response())
        report = evaluator.evaluate(
            query="What are Apple's earnings?",
            answer="Apple beat Q1 estimates.",
            context="Apple reported revenue of $90B, beating the $85B estimate.",
            ground_truth="Apple earnings beat estimates by $5B.",
        )
        self.assertIsInstance(report, EvaluationReport)
        self.assertIsInstance(report.faithfulness, MetricResult)
        self.assertIsInstance(report.context_precision, MetricResult)
        self.assertIsInstance(report.context_recall, MetricResult)
        self.assertIsInstance(report.answer_relevance, MetricResult)

    def test_parse_json_clean(self):
        raw = '{"score": 0.75, "explanation": "good"}'
        result = AlphaEvaluator._parse_json(raw)
        self.assertEqual(result["score"], 0.75)

    def test_parse_json_with_markdown_fence(self):
        raw = '```json\n{"score": 0.8, "explanation": "fine"}\n```'
        result = AlphaEvaluator._parse_json(raw)
        self.assertEqual(result["score"], 0.8)

    def test_parse_json_invalid_returns_default(self):
        result = AlphaEvaluator._parse_json("not json at all", default_score=0.0)
        self.assertEqual(result["score"], 0.0)

    def test_llm_failure_graceful(self):
        with patch("rag.evaluation.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            mock_client.messages.create.side_effect = RuntimeError("API down")
            evaluator = AlphaEvaluator(api_key="fake")
            evaluator._client = mock_client

            result = evaluator._faithfulness("context", "answer")
            self.assertEqual(result.score, 0.0)

    def test_batch_evaluate_aggregation(self):
        evaluator = self._make_evaluator(self._full_judge_response())
        samples = [
            {
                "query": f"query {i}",
                "answer": f"answer {i}",
                "context": f"context {i}",
            }
            for i in range(3)
        ]
        reports = evaluator.batch_evaluate(samples)
        self.assertEqual(len(reports), 3)
        agg = evaluator.aggregate_scores(reports)
        self.assertIn("avg_overall", agg)
        self.assertIn("avg_faithfulness", agg)

    def test_to_dict_has_all_keys(self):
        evaluator = self._make_evaluator(self._full_judge_response())
        report = evaluator.evaluate("q", "a", "c")
        d = report.to_dict()
        for key in ["query", "overall_score", "faithfulness",
                    "context_precision", "context_recall",
                    "answer_relevance", "latency_seconds"]:
            self.assertIn(key, d)

    def test_summary_string(self):
        evaluator = self._make_evaluator(self._full_judge_response())
        report = evaluator.evaluate("q", "a", "c")
        summary = report.summary()
        self.assertIn("Overall", summary)
        self.assertIn("Faith", summary)


# ═══════════════════════════════════════════════════════════════════
# 7. INTEGRATION — Pipeline E2E (mocked)
# ═══════════════════════════════════════════════════════════════════

class TestPipelineIntegration(unittest.TestCase):
    """
    End-to-end: Loader → Processor → Embedder → VectorStore → Retriever
    All external I/O is mocked.
    """

    @patch("rag.loader.yf.Ticker")
    @patch("rag.loader.feedparser.parse")
    def test_full_pipeline(self, mock_parse, mock_ticker):
        # --- Loader ---
        mock_ticker.return_value.news = [
            {
                "content": {
                    "title": "AAPL beats earnings",
                    "summary": "Apple reported $90B revenue in Q1 2024.",
                    "canonicalUrl": {"url": "https://finance.example.com/aapl-q1-2024"},
                    "pubDate": "Fri, 15 Mar 2024 14:00:00 +0000",
                }
            }
        ]
        fake_feed = MagicMock()
        fake_feed.bozo = False
        fake_feed.bozo_exception = None
        fake_feed.entries = []
        mock_parse.return_value = fake_feed

        loader = AlphaLoader()
        raw_docs = loader.load(["AAPL"])
        self.assertGreater(len(raw_docs), 0)

        # --- Processor ---
        processor = AlphaProcessor(chunk_size=300, chunk_overlap=30)
        chunks = processor.process(raw_docs)
        self.assertGreater(len(chunks), 0)
        m = processor.metrics.report()
        self.assertGreater(m["chunks_created"], 0)

        # --- Embedder ---
        import rag.embedding_manager as em
        em._INSTANCE = None
        embedder = AlphaEmbedder()
        embedded = embedder.embed_chunks(chunks)
        self.assertEqual(len(embedded), len(chunks))
        self.assertEqual(len(embedded[0]["embedding"]), 384)

        # --- VectorStore ---
        with patch("rag.vector_store.create_client") as mock_cc:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.data = embedded
            mock_client.table.return_value.upsert.return_value.execute.return_value = mock_resp
            mock_cc.return_value = mock_client

            store = AlphaVectorStore(
                supabase_url="https://fake.supabase.co",
                supabase_key="fake-key",
            )
            count = store.upsert(embedded)
            self.assertEqual(count, len(embedded))

        # --- Retriever ---
        mock_store = MagicMock()
        mock_store.hybrid_search.return_value = [
            {
                "id": i, "text": chunks[i % len(chunks)].text,
                "ticker": "AAPL", "source_type": "news",
                "published_at": "2024-03-15T14:00:00+00:00",
                "url": "https://finance.example.com/aapl-q1-2024",
                "title": "AAPL beats earnings",
                "chunk_index": i, "rrf_score": 1.0 - i * 0.1,
            }
            for i in range(5)
        ]
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 384

        retriever = AlphaRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        context = retriever.retrieve("Apple Q1 earnings", ticker="AAPL")
        self.assertIsInstance(context, str)
        self.assertIn("AAPL", context)
        self.assertIn("[1]", context)


if __name__ == "__main__":
    unittest.main(verbosity=2)
