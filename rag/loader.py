"""
rag/loader.py — AlphaLoader
Multi-source document ingestion with UTC normalization and circuit breaker resilience.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlparse

import feedparser
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------

SourceType = Literal["news", "rss", "reddit"]


@dataclass
class RawDocument:
    title: str
    content: str
    url: str
    source_type: SourceType
    ticker: str
    published_at: str  # UTC ISO-8601, e.g. "2024-03-15T14:32:00+00:00"


# ---------------------------------------------------------------------------
# Timestamp Normalization
# ---------------------------------------------------------------------------

def _to_utc_iso8601(value: object) -> str:
    """
    Convert any common timestamp representation to a UTC ISO-8601 string.
    Raises ValueError when the value cannot be parsed.
    """
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    if isinstance(value, str):
        # RFC-2822 (used by RSS/Atom feeds)
        try:
            return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        # ISO-8601 variants
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                continue

    raise ValueError(f"Cannot parse timestamp: {value!r}")


def _safe_timestamp(value: object, fallback_label: str = "unknown") -> str:
    """Return a UTC ISO-8601 string or the current UTC time on failure."""
    try:
        return _to_utc_iso8601(value)
    except Exception as exc:
        logger.warning("Timestamp parse failed for %s (%s); using utcnow.", fallback_label, exc)
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# AlphaLoader
# ---------------------------------------------------------------------------

class AlphaLoader:
    """
    Fetches documents from yfinance news and Reddit-category RSS feeds.

    Provider-level circuit breakers isolate failures so that one broken
    source never prevents the rest from ingesting.
    """

    REDDIT_FEEDS: dict[str, str] = {
        "r/investing":      "https://www.reddit.com/r/investing/.rss",
        "r/wallstreetbets": "https://www.reddit.com/r/wallstreetbets/.rss",
    }

    def __init__(self, max_news_per_ticker: int = 20, max_rss_per_feed: int = 30) -> None:
        self.max_news_per_ticker = max_news_per_ticker
        self.max_rss_per_feed = max_rss_per_feed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, tickers: list[str]) -> list[RawDocument]:
        """
        Entry point. Returns all successfully fetched documents across
        every source. Partial provider failures are logged but never raised.
        """
        docs: list[RawDocument] = []
        docs.extend(self._fetch_yfinance(tickers))
        docs.extend(self._fetch_reddit_rss(tickers))
        logger.info("AlphaLoader total documents fetched: %d", len(docs))
        return docs

    # ------------------------------------------------------------------
    # yfinance Provider
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, tickers: list[str]) -> list[RawDocument]:
        """Circuit-breaker wraps the entire yfinance provider."""
        docs: list[RawDocument] = []
        for ticker in tickers:
            try:
                t0 = time.perf_counter()
                raw_news = self._yfinance_news(ticker)
                latency = time.perf_counter() - t0
                logger.info(
                    "[yfinance] ticker=%s fetched=%d latency=%.2fs",
                    ticker, len(raw_news), latency,
                )
                docs.extend(raw_news)
            except Exception as exc:
                logger.error("[yfinance] ticker=%s FAILED: %s", ticker, exc)
        return docs

    def _yfinance_news(self, ticker: str) -> list[RawDocument]:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []
        docs: list[RawDocument] = []

        for item in news_items[: self.max_news_per_ticker]:
            # yfinance returns a dict; structure may vary by version
            content_data = item.get("content", {})
            if isinstance(content_data, dict):
                title   = content_data.get("title", "")
                summary = content_data.get("summary", "")
                url     = (
                    content_data.get("canonicalUrl", {}).get("url", "")
                    or content_data.get("clickThroughUrl", {}).get("url", "")
                    or ""
                )
                pub_raw = content_data.get("pubDate", "")
            else:
                # older yfinance schema
                title   = item.get("title", "")
                summary = item.get("summary", "") or item.get("description", "")
                url     = item.get("link", "")
                pub_raw = item.get("providerPublishTime", "")

            if not url:
                continue

            docs.append(RawDocument(
                title=title,
                content=summary,
                url=url,
                source_type="news",
                ticker=ticker,
                published_at=_safe_timestamp(pub_raw, fallback_label=url),
            ))

        return docs

    # ------------------------------------------------------------------
    # Reddit RSS Provider
    # ------------------------------------------------------------------

    def _fetch_reddit_rss(self, tickers: list[str]) -> list[RawDocument]:
        """
        Fetches RSS entries from configured subreddits.
        Tickers are used as metadata only (Reddit feeds are not ticker-specific).
        Circuit-breaker wraps each individual feed.
        """
        docs: list[RawDocument] = []
        primary_ticker = tickers[0] if tickers else "GENERAL"

        for feed_label, feed_url in self.REDDIT_FEEDS.items():
            try:
                t0 = time.perf_counter()
                entries = self._parse_rss(feed_url, primary_ticker)
                latency = time.perf_counter() - t0
                logger.info(
                    "[reddit-rss] feed=%s fetched=%d latency=%.2fs",
                    feed_label, len(entries), latency,
                )
                docs.extend(entries)
            except Exception as exc:
                logger.error("[reddit-rss] feed=%s FAILED: %s", feed_label, exc)

        return docs

    def _parse_rss(self, feed_url: str, ticker: str) -> list[RawDocument]:
        feed = feedparser.parse(feed_url)
        if feed.bozo and feed.bozo_exception:
            raise RuntimeError(f"feedparser error: {feed.bozo_exception}")

        docs: list[RawDocument] = []
        for entry in feed.entries[: self.max_rss_per_feed]:
            url = entry.get("link", "")
            if not url:
                continue

            # Determine if Reddit or generic RSS
            parsed = urlparse(url)
            source_type: SourceType = "reddit" if "reddit.com" in parsed.netloc else "rss"

            pub_raw = entry.get("published", entry.get("updated", ""))
            content = (
                entry.get("summary", "")
                or entry.get("content", [{}])[0].get("value", "")
            )

            docs.append(RawDocument(
                title=entry.get("title", ""),
                content=content,
                url=url,
                source_type=source_type,
                ticker=ticker,
                published_at=_safe_timestamp(pub_raw, fallback_label=url),
            ))

        return docs
