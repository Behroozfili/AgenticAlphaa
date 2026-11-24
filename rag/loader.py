"""Temporal, point-in-time safe document loading for Vector/RAG."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterable, Literal, Mapping
from urllib.parse import urlparse, urlsplit, urlunsplit

import feedparser
import yfinance as yf

logger = logging.getLogger(__name__)
SourceType = Literal["news", "rss", "reddit", "filing", "archive"]


class HistoricalSourceUnavailable(RuntimeError):
    """Historical execution has no reproducible archive provider."""


@dataclass
class RawDocument:
    title: str
    content: str
    url: str
    source_type: SourceType
    ticker: str
    published_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    dataset_version: str = "legacy-v1"
    source_id: str | None = None
    revision: str = "1"


@dataclass(frozen=True)
class QuarantinedDocument:
    ticker: str
    source: str
    reason: str
    raw_timestamp: object
    dataset_version: str


ArchiveProvider = Callable[..., Iterable[RawDocument | Mapping[str, Any]]]


def _to_utc_iso8601(value: object) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Cannot parse timestamp: {value!r}")
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            dt = parsedate_to_datetime(raw)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError as exc:
            raise ValueError(f"Cannot parse timestamp: {value!r}") from exc
    raise ValueError(f"Cannot parse timestamp: {value!r}")


def _safe_timestamp(value: object, fallback_label: str = "unknown") -> str | None:
    """Normalize a timestamp or return ``None``; never substitute the current time."""
    try:
        return _to_utc_iso8601(value)
    except (TypeError, ValueError) as exc:
        logger.warning("Timestamp parse failed for %s (%s); quarantining.", fallback_label, exc)
        return None


def _as_datetime(value: object, *, name: str) -> datetime:
    try:
        return datetime.fromisoformat(_to_utc_iso8601(value))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme.lower(), host + port, parts.path.rstrip("/") or "/", parts.query, ""))


def _source_id(url: str, source_type: str) -> str:
    value = f"{source_type.lower()}|{_canonical_url(url)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mentions_ticker(text: str, ticker: str, company_name: str | None = None) -> bool:
    if not text:
        return False
    if re.search(rf"\b{re.escape(ticker)}\b", text, re.IGNORECASE):
        return True
    first_word = company_name.split()[0] if company_name and company_name.split() else ""
    return bool(len(first_word) >= 3 and re.search(re.escape(first_word), text, re.IGNORECASE))


def _get_company_name(ticker: str) -> str | None:
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName")
    except Exception as exc:
        logger.debug("Could not resolve company name for %s: %s", ticker, exc)
        return None


class AlphaLoader:
    REDDIT_FEEDS = {
        "r/investing": "https://www.reddit.com/r/investing/.rss",
        "r/wallstreetbets": "https://www.reddit.com/r/wallstreetbets/.rss",
    }

    def __init__(self, max_news_per_ticker: int = 20, max_rss_per_feed: int = 30, *, archive_provider: ArchiveProvider | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self.max_news_per_ticker = max_news_per_ticker
        self.max_rss_per_feed = max_rss_per_feed
        self.archive_provider = archive_provider
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.quarantine: list[QuarantinedDocument] = []

    def load(self, tickers: list[str], *, start: object | None = None, end: object | None = None, as_of: object | None = None, dataset_version: str = "legacy-v1", historical: bool | None = None) -> list[RawDocument]:
        tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
        requested_tickers = frozenset(tickers)
        if not dataset_version.strip():
            raise ValueError("dataset_version is required")
        historical = historical if historical is not None else any(value is not None for value in (start, end, as_of))
        bounds = self._validate_bounds(start=start, end=end, as_of=as_of, historical=historical)
        self.quarantine = []
        if historical:
            if self.archive_provider is None:
                raise HistoricalSourceUnavailable("Historical mode requires an injected reproducible archive provider; live-web fallback is disabled")
            raw_items = self.archive_provider(tickers=tickers, start=bounds[0].isoformat(), end=bounds[1].isoformat(), as_of=bounds[2].isoformat(), dataset_version=dataset_version)
            docs = [self._coerce_document(item, dataset_version) for item in raw_items]
        else:
            docs = [*self._fetch_yfinance(tickers), *self._fetch_reddit_rss(tickers)]
        eligible: list[RawDocument] = []
        for doc in docs:
            normalized = self._normalize_document(doc, dataset_version, require_complete_times=historical)
            if normalized is None:
                continue
            if historical and normalized.ticker not in requested_tickers:
                self.quarantine.append(QuarantinedDocument(
                    normalized.ticker,
                    normalized.url,
                    "ticker_out_of_scope",
                    normalized.ticker,
                    dataset_version,
                ))
                continue
            if not historical or self._within_bounds(normalized, *bounds):
                eligible.append(normalized)
        logger.info("AlphaLoader eligible=%d quarantined=%d", len(eligible), len(self.quarantine))
        return eligible

    @staticmethod
    def _validate_bounds(*, start: object | None, end: object | None, as_of: object | None, historical: bool) -> tuple[datetime, datetime, datetime]:
        if historical and (start is None or end is None or as_of is None):
            raise ValueError("historical mode requires start, end, and as_of")
        floor = datetime.min.replace(tzinfo=timezone.utc)
        ceiling = datetime.max.replace(tzinfo=timezone.utc)
        start_dt = _as_datetime(start, name="start") if start is not None else floor
        end_dt = _as_datetime(end, name="end") if end is not None else ceiling
        as_of_dt = _as_datetime(as_of, name="as_of") if as_of is not None else ceiling
        if start_dt > end_dt or end_dt > as_of_dt:
            raise ValueError("temporal boundary must satisfy start <= end <= as_of")
        return start_dt, end_dt, as_of_dt

    @staticmethod
    def _coerce_document(item: RawDocument | Mapping[str, Any], dataset_version: str) -> RawDocument:
        if isinstance(item, RawDocument):
            return item
        payload = dict(item)
        payload.setdefault("dataset_version", dataset_version)
        return RawDocument(**payload)

    def _normalize_document(self, doc: RawDocument, dataset_version: str, *, require_complete_times: bool = False) -> RawDocument | None:
        published = _safe_timestamp(doc.published_at, doc.url)
        retrieved = _safe_timestamp(doc.retrieved_at, doc.url) if doc.retrieved_at is not None else (None if require_complete_times else published)
        effective_from = _safe_timestamp(doc.effective_from, doc.url) if doc.effective_from is not None else (None if require_complete_times else published)
        effective_to = _safe_timestamp(doc.effective_to, doc.url) if doc.effective_to is not None else None
        if published is None or retrieved is None or effective_from is None:
            self.quarantine.append(QuarantinedDocument(doc.ticker.upper(), doc.url, "invalid_or_unknown_evidence_time", doc.published_at, dataset_version))
            return None
        if effective_to and _as_datetime(effective_to, name="effective_to") <= _as_datetime(effective_from, name="effective_from"):
            self.quarantine.append(QuarantinedDocument(doc.ticker.upper(), doc.url, "invalid_effective_interval", doc.effective_to, dataset_version))
            return None
        return replace(doc, ticker=doc.ticker.upper(), published_at=published, retrieved_at=retrieved, effective_from=effective_from, effective_to=effective_to, dataset_version=dataset_version, source_id=doc.source_id or _source_id(doc.url, doc.source_type), revision=str(doc.revision), metadata=dict(doc.metadata))

    @staticmethod
    def _within_bounds(doc: RawDocument, start: datetime, end: datetime, as_of: datetime) -> bool:
        published = _as_datetime(doc.published_at, name="published_at")
        retrieved = _as_datetime(doc.retrieved_at, name="retrieved_at")
        effective_from = _as_datetime(doc.effective_from, name="effective_from")
        effective_to = _as_datetime(doc.effective_to, name="effective_to") if doc.effective_to else None
        return start <= published <= end and published <= as_of and retrieved <= as_of and effective_from <= as_of and (effective_to is None or effective_to > as_of)

    def _live_retrieved_at(self) -> str:
        value = self.clock()
        value = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    def _fetch_yfinance(self, tickers: list[str]) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for ticker in tickers:
            try:
                started = time.perf_counter()
                docs.extend(self._yfinance_news(ticker))
                logger.info("[yfinance] ticker=%s latency=%.2fs", ticker, time.perf_counter() - started)
            except Exception as exc:
                logger.error("[yfinance] ticker=%s FAILED: %s", ticker, exc)
        return docs

    def _yfinance_news(self, ticker: str) -> list[RawDocument]:
        news_items = yf.Ticker(ticker).news or []
        company_name = _get_company_name(ticker)
        docs: list[RawDocument] = []
        retrieved_at = self._live_retrieved_at()
        for item in news_items[: self.max_news_per_ticker]:
            if isinstance(item.get("content"), dict):
                data = item["content"]
                title, summary = data.get("title", ""), data.get("summary", "")
                url = data.get("canonicalUrl", {}).get("url", "") or data.get("clickThroughUrl", {}).get("url", "")
                pub_raw = data.get("pubDate", "")
            else:
                title = item.get("title", "")
                summary = item.get("summary", "") or item.get("description", "")
                url, pub_raw = item.get("link", ""), item.get("providerPublishTime", "")
            if not url or not _mentions_ticker(f"{title} {summary}", ticker, company_name):
                continue
            published = _safe_timestamp(pub_raw, url)
            if published is None:
                self.quarantine.append(QuarantinedDocument(ticker, url, "invalid_published_at", pub_raw, "legacy-v1"))
                continue
            docs.append(RawDocument(title, summary, url, "news", ticker, published, retrieved_at=retrieved_at, effective_from=published))
        return docs

    def _fetch_reddit_rss(self, tickers: list[str]) -> list[RawDocument]:
        if not tickers:
            return []
        names = {ticker: _get_company_name(ticker) for ticker in tickers}
        docs: list[RawDocument] = []
        for _, url in self.REDDIT_FEEDS.items():
            try:
                docs.extend(self._parse_rss(url, tickers, names))
            except Exception as exc:
                logger.error("[reddit-rss] feed=%s FAILED: %s", url, exc)
        return docs

    def _parse_rss(self, feed_url: str, tickers: list[str], company_names: dict[str, str | None]) -> list[RawDocument]:
        feed = feedparser.parse(feed_url)
        if feed.bozo and feed.bozo_exception:
            raise RuntimeError(f"feedparser error: {feed.bozo_exception}")
        docs: list[RawDocument] = []
        retrieved_at = self._live_retrieved_at()
        for entry in feed.entries[: self.max_rss_per_feed]:
            url = entry.get("link", "")
            if not url:
                continue
            source_type: SourceType = "reddit" if "reddit.com" in urlparse(url).netloc else "rss"
            content = entry.get("summary", "") or entry.get("content", [{}])[0].get("value", "")
            title = entry.get("title", "")
            matched = [ticker for ticker in tickers if _mentions_ticker(f"{title} {content}", ticker, company_names.get(ticker))]
            if not matched:
                continue
            raw_time = entry.get("published", entry.get("updated", ""))
            published = _safe_timestamp(raw_time, url)
            if published is None:
                self.quarantine.extend(QuarantinedDocument(ticker, url, "invalid_published_at", raw_time, "legacy-v1") for ticker in matched)
                continue
            docs.extend(RawDocument(title, content, url, source_type, ticker, published, retrieved_at=retrieved_at, effective_from=published) for ticker in matched)
        return docs


__all__ = ["AlphaLoader", "ArchiveProvider", "HistoricalSourceUnavailable", "QuarantinedDocument", "RawDocument", "SourceType", "_safe_timestamp", "_to_utc_iso8601"]
