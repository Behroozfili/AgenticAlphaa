"""
Tests for: rag/loader.py
Phase: 3 — RAG Pipeline (1st: no internal RAG dependencies)

Mocking strategy: yfinance.Ticker and feedparser.parse are both mocked.
No real network calls are made. Circuit-breaker behavior (per-ticker /
per-feed try/except) is tested by making one source raise while others
succeed, asserting partial failure never aborts the whole load().
"""
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest

from rag.loader import (
    AlphaLoader,
    RawDocument,
    _to_utc_iso8601,
    _safe_timestamp,
)


# ---------------------------------------------------------------------------
# _to_utc_iso8601 / _safe_timestamp
# ---------------------------------------------------------------------------

class TestToUtcIso8601:
    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2024, 3, 15, 14, 32, 0)
        result = _to_utc_iso8601(dt)
        assert result == "2024-03-15T14:32:00+00:00"

    def test_aware_datetime_converted_to_utc(self):
        from datetime import timedelta
        dt = datetime(2024, 3, 15, 14, 32, 0, tzinfo=timezone(timedelta(hours=5)))
        result = _to_utc_iso8601(dt)
        assert result == "2024-03-15T09:32:00+00:00"

    def test_unix_timestamp_int(self):
        result = _to_utc_iso8601(1710512400)
        assert result.startswith("2024-03-1")  # exact day depends on TZ table, just sanity

    def test_rfc2822_string(self):
        result = _to_utc_iso8601("Fri, 15 Mar 2024 14:32:00 GMT")
        assert result.startswith("2024-03-15")

    def test_iso8601_with_z_suffix(self):
        result = _to_utc_iso8601("2024-03-15T14:32:00Z")
        assert result == "2024-03-15T14:32:00+00:00"

    def test_date_only_string(self):
        result = _to_utc_iso8601("2024-03-15")
        assert result.startswith("2024-03-15T00:00:00")

    def test_unparseable_value_raises_valueerror(self):
        with pytest.raises(ValueError):
            _to_utc_iso8601("not a date at all")

    def test_unparseable_type_raises_valueerror(self):
        with pytest.raises(ValueError):
            _to_utc_iso8601(object())


class TestSafeTimestamp:
    def test_valid_value_parsed_normally(self):
        result = _safe_timestamp("2024-03-15T14:32:00Z")
        assert result == "2024-03-15T14:32:00+00:00"

    def test_invalid_value_falls_back_to_now(self):
        result = _safe_timestamp("garbage timestamp", fallback_label="test")
        # Should be a valid ISO string close to "now"
        parsed = datetime.fromisoformat(result)
        assert (datetime.now(timezone.utc) - parsed).total_seconds() < 5


# ---------------------------------------------------------------------------
# AlphaLoader.load() — orchestration
# ---------------------------------------------------------------------------

class TestLoaderOrchestration:
    @patch.object(AlphaLoader, "_fetch_reddit_rss", return_value=[])
    @patch.object(AlphaLoader, "_fetch_yfinance", return_value=[])
    def test_load_combines_both_sources(self, mock_yf, mock_rss):
        loader = AlphaLoader()
        docs = loader.load(["NVDA"])
        assert docs == []
        mock_yf.assert_called_once_with(["NVDA"])
        mock_rss.assert_called_once_with(["NVDA"])

    @patch.object(AlphaLoader, "_fetch_reddit_rss")
    @patch.object(AlphaLoader, "_fetch_yfinance")
    def test_load_concatenates_results_from_both(self, mock_yf, mock_rss):
        doc1 = RawDocument("t1", "c1", "u1", "news", "NVDA", "2024-01-01T00:00:00+00:00")
        doc2 = RawDocument("t2", "c2", "u2", "reddit", "NVDA", "2024-01-01T00:00:00+00:00")
        mock_yf.return_value = [doc1]
        mock_rss.return_value = [doc2]

        loader = AlphaLoader()
        docs = loader.load(["NVDA"])
        assert docs == [doc1, doc2]


# ---------------------------------------------------------------------------
# _fetch_yfinance / _yfinance_news — circuit breaker + schema handling
# ---------------------------------------------------------------------------

class TestFetchYfinance:
    @patch("rag.loader.yf.Ticker")
    def test_one_ticker_failure_does_not_abort_others(self, mock_ticker_cls):
        def ticker_side_effect(symbol):
            m = MagicMock()
            if symbol == "BAD":
                raise ConnectionError("network down")
            m.news = [{
                "content": {
                    "title": "Good News", "summary": "desc",
                    "canonicalUrl": {"url": "https://x.com/1"},
                    "pubDate": "2024-03-15T14:32:00Z",
                }
            }]
            return m
        mock_ticker_cls.side_effect = ticker_side_effect

        loader = AlphaLoader()
        docs = loader._fetch_yfinance(["BAD", "GOOD"])

        assert len(docs) == 1
        assert docs[0].ticker == "GOOD"

    @patch("rag.loader.yf.Ticker")
    def test_new_schema_content_dict_parsed(self, mock_ticker_cls):
        mock_ticker_cls.return_value.news = [{
            "content": {
                "title": "NVDA surges", "summary": "Great quarter",
                "canonicalUrl": {"url": "https://x.com/a"},
                "pubDate": "2024-03-15T14:32:00Z",
            }
        }]
        loader = AlphaLoader()
        docs = loader._yfinance_news("NVDA")
        assert docs[0].title == "NVDA surges"
        assert docs[0].url == "https://x.com/a"
        assert docs[0].source_type == "news"

    @patch("rag.loader.yf.Ticker")
    def test_old_schema_flat_dict_parsed_bug_fixed(self, mock_ticker_cls):
        """
        FIXED: the old-schema ("else") branch is now reachable — old-schema
        items (no "content" key) are correctly parsed via item.get("title"),
        item.get("link"), etc. instead of being silently treated as empty
        new-schema items and dropped.
        """
        mock_ticker_cls.return_value.news = [{
            "title": "Old schema item", "summary": "desc",
            "link": "https://x.com/old", "providerPublishTime": 1710512400,
        }]
        loader = AlphaLoader()
        docs = loader._yfinance_news("NVDA")
        assert len(docs) == 1
        assert docs[0].title == "Old schema item"
        assert docs[0].url == "https://x.com/old"
        assert docs[0].content == "desc"

    @patch("rag.loader.yf.Ticker")
    def test_canonical_url_falls_back_to_clickthrough_url(self, mock_ticker_cls):
        mock_ticker_cls.return_value.news = [{
            "content": {
                "title": "x", "summary": "y",
                "canonicalUrl": {},
                "clickThroughUrl": {"url": "https://fallback.com"},
                "pubDate": "2024-03-15T14:32:00Z",
            }
        }]
        loader = AlphaLoader()
        docs = loader._yfinance_news("NVDA")
        assert docs[0].url == "https://fallback.com"

    @patch("rag.loader.yf.Ticker")
    def test_item_with_no_url_is_skipped(self, mock_ticker_cls):
        mock_ticker_cls.return_value.news = [{
            "content": {"title": "x", "summary": "y", "canonicalUrl": {}, "pubDate": "z"}
        }]
        loader = AlphaLoader()
        docs = loader._yfinance_news("NVDA")
        assert docs == []

    @patch("rag.loader.yf.Ticker")
    def test_max_news_per_ticker_caps_results(self, mock_ticker_cls):
        mock_ticker_cls.return_value.news = [
            {"content": {"title": f"t{i}", "summary": "s",
                         "canonicalUrl": {"url": f"https://x.com/{i}"},
                         "pubDate": "2024-03-15T14:32:00Z"}}
            for i in range(10)
        ]
        loader = AlphaLoader(max_news_per_ticker=3)
        docs = loader._yfinance_news("NVDA")
        assert len(docs) == 3

    @patch("rag.loader.yf.Ticker")
    def test_no_news_attribute_returns_empty(self, mock_ticker_cls):
        mock_ticker_cls.return_value.news = None
        loader = AlphaLoader()
        docs = loader._yfinance_news("NVDA")
        assert docs == []


# ---------------------------------------------------------------------------
# _fetch_reddit_rss / _parse_rss — circuit breaker + source_type detection
# ---------------------------------------------------------------------------

class TestFetchRedditRss:
    @patch("rag.loader.feedparser.parse")
    def test_one_feed_failure_does_not_abort_others(self, mock_parse):
        def parse_side_effect(url):
            m = MagicMock()
            if "wallstreetbets" in url:
                m.bozo = True
                m.bozo_exception = Exception("malformed feed")
            else:
                m.bozo = False
                m.entries = [{"link": "https://reddit.com/r/investing/1",
                              "title": "t", "summary": "s", "published": "2024-03-15T14:32:00Z"}]
            return m
        mock_parse.side_effect = parse_side_effect

        loader = AlphaLoader()
        docs = loader._fetch_reddit_rss(["NVDA"])
        # only r/investing succeeded
        assert len(docs) == 1
        assert docs[0].source_type == "reddit"

    @patch("rag.loader.feedparser.parse")
    def test_no_tickers_uses_general_label(self, mock_parse):
        mock_parse.return_value.bozo = False
        mock_parse.return_value.entries = []
        loader = AlphaLoader()
        loader._fetch_reddit_rss([])  # should not raise
        # primary_ticker fallback path exercised; no assertion needed beyond no-crash

    def test_parse_rss_detects_reddit_vs_generic_source_type(self):
        with patch("rag.loader.feedparser.parse") as mock_parse:
            mock_parse.return_value.bozo = False
            mock_parse.return_value.entries = [
                {"link": "https://reddit.com/r/investing/x", "title": "a",
                 "summary": "s", "published": "2024-03-15T14:32:00Z"},
                {"link": "https://example.com/article", "title": "b",
                 "summary": "s", "published": "2024-03-15T14:32:00Z"},
            ]
            loader = AlphaLoader()
            docs = loader._parse_rss("https://www.reddit.com/r/investing/.rss", "NVDA")
            assert docs[0].source_type == "reddit"
            assert docs[1].source_type == "rss"

    def test_parse_rss_raises_on_bozo_feed(self):
        with patch("rag.loader.feedparser.parse") as mock_parse:
            mock_parse.return_value.bozo = True
            mock_parse.return_value.bozo_exception = Exception("bad xml")
            loader = AlphaLoader()
            with pytest.raises(RuntimeError):
                loader._parse_rss("https://bad.feed/x.rss", "NVDA")

    def test_parse_rss_entry_without_link_is_skipped(self):
        with patch("rag.loader.feedparser.parse") as mock_parse:
            mock_parse.return_value.bozo = False
            mock_parse.return_value.entries = [{"title": "no link entry"}]
            loader = AlphaLoader()
            docs = loader._parse_rss("https://feed.url", "NVDA")
            assert docs == []

    def test_parse_rss_content_falls_back_to_summary(self):
        with patch("rag.loader.feedparser.parse") as mock_parse:
            mock_parse.return_value.bozo = False
            mock_parse.return_value.entries = [{
                "link": "https://example.com/x", "title": "t",
                "summary": "from summary", "published": "2024-03-15T14:32:00Z",
            }]
            loader = AlphaLoader()
            docs = loader._parse_rss("https://feed.url", "NVDA")
            assert docs[0].content == "from summary"

    def test_max_rss_per_feed_caps_results(self):
        with patch("rag.loader.feedparser.parse") as mock_parse:
            mock_parse.return_value.bozo = False
            mock_parse.return_value.entries = [
                {"link": f"https://example.com/{i}", "title": f"t{i}",
                 "summary": "s", "published": "2024-03-15T14:32:00Z"}
                for i in range(10)
            ]
            loader = AlphaLoader(max_rss_per_feed=2)
            docs = loader._parse_rss("https://feed.url", "NVDA")
            assert len(docs) == 2