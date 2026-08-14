"""
Tests for: rag/retriever.py
Phase: 3 — RAG Pipeline (5th: depends on embedding_manager + vector_store)

Mocking strategy: AlphaVectorStore and AlphaEmbedder are both injected as
plain MagicMocks (the class explicitly supports DI for this purpose, per
its own docstring warning about get_embedder()'s side effect). No real
Supabase or model loading occurs.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

from rag.retriever import AlphaRetriever


def make_chunk(text="chunk text", rrf_score=0.5, url="https://x.com/1",
               source_type="news", ticker="NVDA", published_at=None):
    if published_at is None:
        published_at = datetime.now(timezone.utc).isoformat()
    return {
        "text": text, "rrf_score": rrf_score, "url": url,
        "source_type": source_type, "ticker": ticker, "published_at": published_at,
    }


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.hybrid_search.return_value = []
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def retriever(mock_store, mock_embedder):
    return AlphaRetriever(vector_store=mock_store, embedder=mock_embedder)


# ---------------------------------------------------------------------------
# __init__ — embedder injection avoids the side-effect-laden default
# ---------------------------------------------------------------------------

class TestInit:
    def test_injected_embedder_is_used_not_get_embedder(self, mock_store, mock_embedder):
        r = AlphaRetriever(vector_store=mock_store, embedder=mock_embedder)
        assert r.embedder is mock_embedder

    def test_default_embedder_calls_get_embedder(self, mock_store, monkeypatch):
        sentinel = MagicMock()
        monkeypatch.setattr("rag.retriever.get_embedder", lambda: sentinel)
        r = AlphaRetriever(vector_store=mock_store)
        assert r.embedder is sentinel


# ---------------------------------------------------------------------------
# _hours_since — timestamp parsing edge cases
# ---------------------------------------------------------------------------

class TestHoursSince:
    def test_recent_timestamp_small_hours(self):
        now = datetime.now(timezone.utc)
        pub = (now - timedelta(hours=2)).isoformat()
        hours = AlphaRetriever._hours_since(pub, now)
        assert hours == pytest.approx(2.0, abs=0.01)

    def test_naive_datetime_assumed_utc(self):
        now = datetime.now(timezone.utc)
        naive_pub = (now - timedelta(hours=5)).replace(tzinfo=None)
        hours = AlphaRetriever._hours_since(naive_pub, now)
        assert hours == pytest.approx(5.0, abs=0.01)

    def test_unparseable_string_returns_720(self):
        now = datetime.now(timezone.utc)
        assert AlphaRetriever._hours_since("not a date", now) == 720.0

    def test_non_string_non_datetime_returns_720(self):
        now = datetime.now(timezone.utc)
        assert AlphaRetriever._hours_since(12345, now) == 720.0

    def test_future_timestamp_clamped_to_zero(self):
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=5)).isoformat()
        assert AlphaRetriever._hours_since(future, now) == 0.0


# ---------------------------------------------------------------------------
# _rerank_by_freshness
# ---------------------------------------------------------------------------

class TestRerankByFreshness:
    def test_fresher_chunk_ranked_above_older_with_same_rrf(self, retriever):
        now = datetime.now(timezone.utc)
        old_chunk = make_chunk(rrf_score=0.5, published_at=(now - timedelta(hours=200)).isoformat())
        new_chunk = make_chunk(rrf_score=0.5, published_at=now.isoformat())

        reranked = retriever._rerank_by_freshness([old_chunk, new_chunk])

        assert reranked[0]["text"] == new_chunk["text"]
        assert reranked[0]["freshness_score"] > reranked[1]["freshness_score"]

    def test_higher_rrf_can_overcome_staleness(self, retriever):
        now = datetime.now(timezone.utc)
        very_old_high_score = make_chunk(
            rrf_score=10.0, published_at=(now - timedelta(hours=1)).isoformat()
        )
        new_low_score = make_chunk(
            rrf_score=0.001, published_at=now.isoformat()
        )
        reranked = retriever._rerank_by_freshness([new_low_score, very_old_high_score])
        assert reranked[0]["rrf_score"] == 10.0

    def test_each_chunk_gets_freshness_score_and_hours_old_fields(self, retriever):
        chunk = make_chunk()
        reranked = retriever._rerank_by_freshness([chunk])
        assert "freshness_score" in reranked[0]
        assert "hours_old" in reranked[0]


# ---------------------------------------------------------------------------
# _diversity_filter
# ---------------------------------------------------------------------------

class TestDiversityFilter:
    def test_max_2_per_url(self, retriever):
        chunks = [make_chunk(url="https://same.com", text=f"t{i}") for i in range(5)]
        result = retriever._diversity_filter(chunks)
        assert len(result) == 2

    def test_max_3_per_source_type(self, retriever):
        chunks = [make_chunk(url=f"https://x.com/{i}", source_type="reddit", text=f"t{i}")
                  for i in range(5)]
        result = retriever._diversity_filter(chunks)
        assert len(result) == 3

    def test_stops_at_stage3_k(self, mock_store, mock_embedder):
        r = AlphaRetriever(vector_store=mock_store, embedder=mock_embedder, stage3_k=2)
        chunks = [make_chunk(url=f"https://x.com/{i}", text=f"t{i}") for i in range(10)]
        result = r._diversity_filter(chunks)
        assert len(result) == 2

    def test_diverse_sources_all_included_up_to_limit(self, retriever):
        chunks = [
            make_chunk(url="https://a.com", source_type="news", text="1"),
            make_chunk(url="https://b.com", source_type="reddit", text="2"),
            make_chunk(url="https://c.com", source_type="rss", text="3"),
        ]
        result = retriever._diversity_filter(chunks)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _apply_token_budget
# ---------------------------------------------------------------------------

class TestApplyTokenBudget:
    def test_chunks_within_budget_all_kept(self, mock_store, mock_embedder):
        r = AlphaRetriever(vector_store=mock_store, embedder=mock_embedder, token_budget=1000)
        chunks = [make_chunk(text="short") for _ in range(3)]
        result = r._apply_token_budget(chunks)
        assert len(result) == 3

    def test_at_least_one_chunk_kept_even_if_over_budget(self, mock_store, mock_embedder):
        r = AlphaRetriever(vector_store=mock_store, embedder=mock_embedder, token_budget=1)
        chunks = [make_chunk(text="x" * 100)]
        result = r._apply_token_budget(chunks)
        assert len(result) == 1  # first chunk always included even alone over budget

    def test_stops_once_budget_exceeded(self, mock_store, mock_embedder):
        # char_budget = token_budget * 4 = 40
        r = AlphaRetriever(vector_store=mock_store, embedder=mock_embedder, token_budget=10)
        chunks = [make_chunk(text="x" * 20), make_chunk(text="x" * 20), make_chunk(text="x" * 20)]
        result = r._apply_token_budget(chunks)
        assert len(result) == 2  # first two fit in 40 chars, third pushes over


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_empty_chunks_returns_placeholder_message(self):
        assert AlphaRetriever._format_context([]) == "No relevant context found."

    def test_includes_index_source_ticker_date_score(self):
        chunk = make_chunk(text="hello", source_type="news", ticker="NVDA",
                           url="https://x.com", published_at="2024-03-15T14:32:00+00:00")
        chunk["freshness_score"] = 0.789
        result = AlphaRetriever._format_context([chunk])
        assert "[1]" in result
        assert "SOURCE: news" in result
        assert "TICKER: NVDA" in result
        assert "SCORE: 0.7890" in result
        assert "hello" in result

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [make_chunk(text="a"), make_chunk(text="b")]
        result = AlphaRetriever._format_context(chunks)
        assert "[1]" in result
        assert "[2]" in result


# ---------------------------------------------------------------------------
# retrieve() / retrieve_raw() — full pipeline orchestration
# ---------------------------------------------------------------------------

class TestRetrievePipeline:
    def test_retrieve_calls_embedder_and_store_with_query(self, retriever, mock_store, mock_embedder):
        mock_store.hybrid_search.return_value = []
        result = retriever.retrieve("NVDA earnings")

        mock_embedder.embed_query.assert_called_once_with("NVDA earnings")
        mock_store.hybrid_search.assert_called_once()
        assert result == "No relevant context found."

    def test_retrieve_returns_formatted_string_with_real_chunk(self, retriever, mock_store):
        mock_store.hybrid_search.return_value = [make_chunk(text="real content")]
        result = retriever.retrieve("query")
        assert "real content" in result
        assert isinstance(result, str)

    def test_retrieve_raw_returns_list_of_dicts_not_string(self, retriever, mock_store):
        mock_store.hybrid_search.return_value = [make_chunk(text="real content")]
        result = retriever.retrieve_raw("query")
        assert isinstance(result, list)
        assert result[0]["text"] == "real content"

    def test_retrieve_passes_ticker_and_days_back_through(self, retriever, mock_store):
        retriever.retrieve("q", ticker="NVDA", days_back=7)
        _, kwargs = mock_store.hybrid_search.call_args
        assert kwargs["ticker"] == "NVDA"
        assert kwargs["days_back"] == 7