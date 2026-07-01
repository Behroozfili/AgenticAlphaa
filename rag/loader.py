"""
rag/loader.py — AlphaLoader
Multi-source document ingestion with UTC normalization and circuit breaker resilience.
"""

from __future__ import annotations

import logging
import re
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
# Ticker-mention validation
# ---------------------------------------------------------------------------
# DC-6: yfinance's per-ticker news feed and the general-purpose Reddit
# subreddit feeds both surface content that is NOT necessarily about the
# ticker it's being ingested for. yfinance's stock.news for AAPL can include
# broader sector/market articles that are actually about a different company
# (e.g. a Microsoft-focused piece surfaced in Apple's news feed due to sector
# correlation); Reddit's r/investing and r/wallstreetbets are not
# ticker-specific at all — every post there discusses whatever the author
# chose, independent of any particular ingestion batch.
#
# The previous implementation blindly tagged every fetched item with the
# ticker it was fetched *for* (yfinance) or the first ticker in the batch
# (Reddit), regardless of what the item's title/content actually discussed.
# That mislabeling later let the RAG layer's "ticker_filter" correctly
# narrow a query to e.g. AAPL and still surface a chunk that is entirely
# about Microsoft, because the chunk's stored `ticker` metadata was wrong
# from the moment it was ingested — no query-time filter can fix a label
# that was incorrect at write time.
#
# This validator is a simple, low-cost heuristic (word-boundary ticker
# match OR company-name substring match) applied BEFORE a document is
# tagged with a given ticker. It intentionally errs on the side of
# excluding ambiguous content rather than risking another mislabeled
# document in the knowledge base — for a financial RAG system, a smaller
# but accurately-labeled corpus is more valuable than a larger,
# contamination-prone one.

def _mentions_ticker(text: str, ticker: str, company_name: str | None = None) -> bool:
    """
    Return True if `text` plausibly discusses `ticker` — either the ticker
    symbol itself (as a whole word, case-insensitive) or, if provided, the
    company's short name (as a case-insensitive substring, using just the
    first "word" of the name to tolerate suffixes like "Inc.", "Corp",
    "Corporation" that articles often omit or abbreviate differently).
    """
    if not text:
        return False
    if re.search(rf"\b{re.escape(ticker)}\b", text, re.IGNORECASE):
        return True
    if company_name:
        # Use the first token of the company name (e.g. "Apple" from
        # "Apple Inc.", "Microsoft" from "Microsoft Corporation") — matching
        # the full legal name is too strict since articles rarely spell it
        # out verbatim.
        first_word = company_name.split()[0] if company_name.split() else ""
        if len(first_word) >= 3 and re.search(re.escape(first_word), text, re.IGNORECASE):
            return True
    return False


def _get_company_name(ticker: str) -> str | None:
    """
    Best-effort lookup of a ticker's short company name via yfinance, for
    use as a secondary signal in _mentions_ticker(). Returns None (not an
    exception) on any failure — a missing company name just means
    _mentions_ticker() falls back to symbol-only matching, it never blocks
    ingestion outright.
    """
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName")
    except Exception as exc:
        logger.debug("Could not resolve company name for %s: %s", ticker, exc)
        return None


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
        company_name = _get_company_name(ticker)
        skipped_mismatched = 0

        for item in news_items[: self.max_news_per_ticker]:
            
            if "content" in item and isinstance(item["content"], dict):
                
                content_data = item["content"]
                title   = content_data.get("title", "")
                summary = content_data.get("summary", "")
                url     = (
                    content_data.get("canonicalUrl", {}).get("url", "")
                    or content_data.get("clickThroughUrl", {}).get("url", "")
                    or ""
                )
                pub_raw = content_data.get("pubDate", "")
            else:
                
                title   = item.get("title", "")
                summary = item.get("summary", "") or item.get("description", "")
                url     = item.get("link", "")
                pub_raw = item.get("providerPublishTime", "")

        
            if not url:
                logger.debug("Skipping item with empty URL: %s", title)
                continue

            # DC-6: yfinance's per-ticker news feed sometimes surfaces
            # broader sector/market articles that are actually about a
            # different company (sector correlation, "related stocks"
            # sidebars, etc). Blindly tagging every returned item with
            # `ticker` mislabels those in the knowledge base in a way no
            # later query-time filter can correct. Verify the article
            # actually mentions this ticker or its company name before
            # tagging it as such.
            if not _mentions_ticker(f"{title} {summary}", ticker, company_name):
                logger.debug(
                    "[yfinance] Skipping article not actually about %s: %r",
                    ticker, title[:80],
                )
                skipped_mismatched += 1
                continue

            docs.append(RawDocument(
                title=title,
                content=summary,
                url=url,
                source_type="news",
                ticker=ticker,
                published_at=_safe_timestamp(pub_raw, fallback_label=url),
            ))

        if skipped_mismatched:
            logger.info(
                "[yfinance] ticker=%s skipped %d article(s) not actually about this ticker.",
                ticker, skipped_mismatched,
            )
        return docs
    # ------------------------------------------------------------------
    # Reddit RSS Provider
    # ------------------------------------------------------------------

    def _fetch_reddit_rss(self, tickers: list[str]) -> list[RawDocument]:
        """
        Fetches RSS entries from configured subreddits.

        DC-6: r/investing and r/wallstreetbets are general-purpose subreddits
        — no post there is inherently "about" any particular ticker. The
        previous implementation tagged every single post from these feeds
        with `tickers[0]` regardless of content, which silently poisoned the
        knowledge base with e.g. an NVDA-focused Reddit thread stored under
        `ticker="AAPL"` whenever AAPL happened to be first in the ingestion
        batch. A query later filtered to ticker="AAPL" would then legitimately
        (and invisibly) surface content that has nothing to do with Apple.

        Fix: each post is now checked against every ticker in the current
        batch (via _mentions_ticker). A post that mentions N tickers from the
        batch produces N tagged copies (one per mentioned ticker) so it can
        be correctly retrieved under each; a post that mentions none of the
        batch's tickers is dropped rather than mislabeled under an arbitrary
        one. Circuit-breaker wraps each individual feed.
        """
        docs: list[RawDocument] = []
        if not tickers:
            return docs

        company_names = {t: _get_company_name(t) for t in tickers}

        for feed_label, feed_url in self.REDDIT_FEEDS.items():
            try:
                t0 = time.perf_counter()
                entries = self._parse_rss(feed_url, tickers, company_names)
                latency = time.perf_counter() - t0
                logger.info(
                    "[reddit-rss] feed=%s fetched=%d tagged_doc(s) latency=%.2fs",
                    feed_label, len(entries), latency,
                )
                docs.extend(entries)
            except Exception as exc:
                logger.error("[reddit-rss] feed=%s FAILED: %s", feed_label, exc)

        return docs

    def _parse_rss(
        self,
        feed_url: str,
        tickers: list[str],
        company_names: dict[str, str | None],
    ) -> list[RawDocument]:
        feed = feedparser.parse(feed_url)
        if feed.bozo and feed.bozo_exception:
            raise RuntimeError(f"feedparser error: {feed.bozo_exception}")

        docs: list[RawDocument] = []
        skipped_unmatched = 0

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
            title = entry.get("title", "")
            combined_text = f"{title} {content}"

            matched_tickers = [
                t for t in tickers
                if _mentions_ticker(combined_text, t, company_names.get(t))
            ]

            if not matched_tickers:
                skipped_unmatched += 1
                continue

            published_at = _safe_timestamp(pub_raw, fallback_label=url)
            for t in matched_tickers:
                docs.append(RawDocument(
                    title=title,
                    content=content,
                    url=url,
                    source_type=source_type,
                    ticker=t,
                    published_at=published_at,
                ))

        if skipped_unmatched:
            logger.info(
                "[reddit-rss] %s: skipped %d post(s) matching none of %s.",
                feed_url, skipped_unmatched, tickers,
            )
        return docs