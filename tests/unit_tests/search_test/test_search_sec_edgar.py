"""
Tests for: tools/research_tools/sec_edgar.py  (async, httpx-based)

NOTE: This project has TWO files named sec_edgar.py:
  - tools/research_tools/sec_edgar.py   (this file)          — async, uses `httpx`
  - tools/financial_tools/sec_edgar.py  (separate file)       — sync, uses `requests`
They are tested separately (see test_sec_edgar.py for the financial_tools
one). Make sure your test runner resolves the correct one via package path —
do NOT rely on bare `import sec_edgar`, always use the full dotted path.

Mocking strategy: this module makes all its HTTP calls through
`httpx.AsyncClient(...)` used as an async context manager
(`async with httpx.AsyncClient(...) as client: resp = await client.get(...)`).
So `tools.research_tools.sec_edgar.httpx.AsyncClient` is patched, and the
mock's `__aenter__` return value is what stands in for `client` — its
`.get` is an AsyncMock returning a fake response object with `.json()`,
`.text`, `.status_code`, and `.raise_for_status()`.

A small helper (`make_async_client_mock`) builds this shape once, since
every test needs a slightly different response but the same wiring.
"""
from unittest.mock import patch, MagicMock, AsyncMock
import httpx
import pytest

from tools.research_tools.sec_edgar import (
    sec_edgar_search,
    sec_edgar_filing,
    _resolve_cik,
    _find_latest,
    _html_to_text,
    _extract_sections,
    _sanitize_query,
    _CIK_CACHE,
)
import tools.research_tools.sec_edgar as sec_edgar_module


def make_response(json_data=None, text_data=None, status_code=200, raise_exc=None):
    """Build a fake httpx.Response-like object for a mocked client.get()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text_data if text_data is not None else ""
    if json_data is not None:
        resp.json.return_value = json_data
    if raise_exc:
        resp.raise_for_status.side_effect = raise_exc
    else:
        resp.raise_for_status.return_value = None
    return resp


def make_async_client_mock(responses):
    """
    Build a MagicMock standing in for httpx.AsyncClient(...), configured so
    that `async with httpx.AsyncClient(...) as client` yields an object
    whose `.get` is an AsyncMock returning each of *responses* in order
    (a list) on successive calls, or a single response reused for every call.
    """
    client = MagicMock()
    if isinstance(responses, list):
        client.get = AsyncMock(side_effect=responses)
    else:
        client.get = AsyncMock(return_value=responses)

    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=client)
    context_manager.__aexit__ = AsyncMock(return_value=None)
    return context_manager


@pytest.fixture(autouse=True)
def reset_module_caches():
    """_CIK_CACHE and _TICKER_TO_CIK are module-level singletons shared
    across calls — reset between tests so one test's resolved ticker can't
    silently make another test's mock irrelevant (a real bug class seen
    elsewhere in this project when a module-level cache outlives the
    mock that was supposed to gate a network call)."""
    _CIK_CACHE.clear()
    sec_edgar_module._TICKER_TO_CIK = None
    yield
    _CIK_CACHE.clear()
    sec_edgar_module._TICKER_TO_CIK = None


# ---------------------------------------------------------------------------
# _sanitize_query
# ---------------------------------------------------------------------------

class TestSanitizeQuery:
    def test_strips_bare_year_tokens(self):
        result = _sanitize_query("Apple AI spending 2025 2026")
        assert "2025" not in result
        assert "2026" not in result

    def test_caps_at_max_terms(self):
        result = _sanitize_query("one two three four five six seven eight")
        assert len(result.split()) == 6

    def test_short_query_unchanged(self):
        assert _sanitize_query("Apple earnings") == "Apple earnings"


# ---------------------------------------------------------------------------
# _resolve_cik
# ---------------------------------------------------------------------------

class TestResolveCik:
    @pytest.mark.asyncio
    async def test_happy_path_resolves_ticker(self):
        mock_client_cm = make_async_client_mock(
            make_response(json_data={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
            })
        )
        with patch.object(sec_edgar_module.httpx, "AsyncClient", return_value=mock_client_cm):
            cik = await _resolve_cik("AAPL")
        assert cik == "320193"

    @pytest.mark.asyncio
    async def test_unknown_ticker_returns_none(self):
        mock_client_cm = make_async_client_mock(
            make_response(json_data={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
            })
        )
        with patch.object(sec_edgar_module.httpx, "AsyncClient", return_value=mock_client_cm):
            cik = await _resolve_cik("ZZZZ")
        assert cik is None

    @pytest.mark.asyncio
    async def test_second_call_uses_cache_not_network(self):
        mock_client_cm = make_async_client_mock(
            make_response(json_data={
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
            })
        )
        with patch.object(sec_edgar_module.httpx, "AsyncClient", return_value=mock_client_cm) as mock_cls:
            await _resolve_cik("AAPL")
            await _resolve_cik("AAPL")
        # AsyncClient should only be constructed once — the second
        # resolution must come from _CIK_CACHE, not a second network round-trip.
        assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_dotted_ticker_falls_back_to_hyphenated(self):
        mock_client_cm = make_async_client_mock(
            make_response(json_data={
                "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"}
            })
        )
        with patch.object(sec_edgar_module.httpx, "AsyncClient", return_value=mock_client_cm):
            cik = await _resolve_cik("BRK.B")
        assert cik == "1067983"


# ---------------------------------------------------------------------------
# sec_edgar_search
# ---------------------------------------------------------------------------

class TestSecEdgarSearch:
    @pytest.mark.asyncio
    async def test_happy_path_maps_real_field_names(self):
        """Regression: efts.sec.gov's real response uses display_names/
        root_forms/adsh/period_ending/file_date — NOT entity_name/form_type/
        accession_no/period_of_report, which an earlier version of this
        code assumed (silently returning blank strings for everything but
        filed_at)."""
        search_response = make_response(json_data={
            "hits": {"hits": [{
                "_source": {
                    "display_names": ["APPLE INC (0000320193)"],
                    "root_forms": ["10-Q"],
                    "adsh": "0000320193-26-000013",
                    "period_ending": "2026-03-28",
                    "file_date": "2026-05-01",
                }
            }]}
        })
        with patch.object(sec_edgar_module.httpx, "AsyncClient",
                          return_value=make_async_client_mock(search_response)):
            result = await sec_edgar_search(query="revenue guidance", ticker=None)

        assert result["filings"][0]["company"] == "APPLE INC (0000320193)"
        assert result["filings"][0]["form_type"] == "10-Q"
        assert result["filings"][0]["accession_number"] == "0000320193-26-000013"

    @pytest.mark.asyncio
    async def test_unresolvable_ticker_returns_error_not_unscoped_search(self):
        mock_client_cm = make_async_client_mock(
            make_response(json_data={"0": {"cik_str": 1, "ticker": "AAPL", "title": "x"}})
        )
        with patch.object(sec_edgar_module.httpx, "AsyncClient", return_value=mock_client_cm):
            result = await sec_edgar_search(query="q", ticker="ZZZNOTAREAL")

        assert result["filings"] == []
        assert "CIK not found" in result["error"]

    @pytest.mark.asyncio
    async def test_cik_resolution_exception_returns_error_dict_not_raise(self):
        with patch.object(sec_edgar_module, "_resolve_cik", side_effect=RuntimeError("boom")):
            result = await sec_edgar_search(query="q", ticker="AAPL")
        assert result["filings"] == []
        assert "CIK resolution failed" in result["error"]

    @pytest.mark.asyncio
    async def test_http_status_error_returns_error_dict(self):
        error_response = make_response(status_code=500)
        http_error = httpx.HTTPStatusError("server error", request=MagicMock(), response=error_response)
        error_response.raise_for_status.side_effect = http_error

        with patch.object(sec_edgar_module.httpx, "AsyncClient",
                          return_value=make_async_client_mock(error_response)):
            result = await sec_edgar_search(query="q")

        assert result["filings"] == []
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_max_results_truncates_hits(self):
        many_hits = {"hits": {"hits": [
            {"_source": {"display_names": [f"CO {i}"], "root_forms": ["10-K"],
                        "adsh": f"acc-{i}", "period_ending": "2026-01-01", "file_date": "2026-01-01"}}
            for i in range(10)
        ]}}
        with patch.object(sec_edgar_module.httpx, "AsyncClient",
                          return_value=make_async_client_mock(make_response(json_data=many_hits))):
            result = await sec_edgar_search(query="q", max_results=3)
        assert len(result["filings"]) == 3


# ---------------------------------------------------------------------------
# sec_edgar_filing
# ---------------------------------------------------------------------------

class TestSecEdgarFiling:
    @pytest.mark.asyncio
    async def test_cik_not_found_returns_error(self):
        with patch.object(sec_edgar_module, "_resolve_cik", return_value=None):
            result = await sec_edgar_filing("ZZZZ")
        assert "CIK not found" in result["error"]

    @pytest.mark.asyncio
    async def test_submissions_fetch_failure_returns_error(self):
        with patch.object(sec_edgar_module, "_resolve_cik", return_value="320193"), \
             patch.object(sec_edgar_module, "_get_submissions", return_value=None):
            result = await sec_edgar_filing("AAPL")
        assert "Failed to fetch EDGAR submissions" in result["error"]

    @pytest.mark.asyncio
    async def test_no_matching_form_type_returns_error(self):
        with patch.object(sec_edgar_module, "_resolve_cik", return_value="320193"), \
             patch.object(sec_edgar_module, "_get_submissions",
                          return_value={"name": "Apple Inc.", "filings": {"recent": {
                              "form": ["8-K"], "accessionNumber": ["x"], "filingDate": ["2026-01-01"],
                          }}}):
            result = await sec_edgar_filing("AAPL", form_type="10-K")
        assert "No 10-K found" in result["error"]

    @pytest.mark.asyncio
    async def test_happy_path_returns_parsed_sections(self):
        sample_10q = (
            "Item 1. Financial Statements\n" + "balance sheet stuff " * 50 +
            "Item 2. Management's Discussion and Analysis\n" + "mda content here " * 50 +
            "Item 1A. Risk Factors\n" + "risk factor content " * 50
        )
        with patch.object(sec_edgar_module, "_resolve_cik", return_value="320193"), \
             patch.object(sec_edgar_module, "_get_submissions", return_value={
                 "name": "Apple Inc.",
                 "filings": {"recent": {
                     "form": ["10-Q"], "accessionNumber": ["0000320193-26-000013"],
                     "filingDate": ["2026-05-01"],
                 }},
             }), \
             patch.object(sec_edgar_module, "_fetch_text", return_value=sample_10q):
            result = await sec_edgar_filing("AAPL", form_type="10-Q", sections=["mda"])

        assert "mda" in result["sections"]
        assert "management" in result["sections"]["mda"].lower()


# ---------------------------------------------------------------------------
# _find_latest
# ---------------------------------------------------------------------------

class TestFindLatest:
    def test_finds_matching_form_type(self):
        submissions = {"filings": {"recent": {
            "form": ["10-Q", "8-K", "10-K"],
            "accessionNumber": ["a1", "a2", "a3"],
            "filingDate": ["2026-05-01", "2026-04-01", "2025-11-01"],
        }}}
        result = _find_latest(submissions, "10-K")
        assert result["accessionNumber"] == "a3"

    def test_no_match_returns_none(self):
        submissions = {"filings": {"recent": {
            "form": ["8-K"], "accessionNumber": ["a1"], "filingDate": ["2026-01-01"],
        }}}
        assert _find_latest(submissions, "10-K") is None


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_strips_tags(self):
        result = _html_to_text("<p>Hello <b>World</b></p>")
        assert "<" not in result
        assert "Hello" in result and "World" in result

    def test_drops_script_and_style_content(self):
        result = _html_to_text("<script>evil()</script><p>Real content</p>")
        assert "evil" not in result
        assert "Real content" in result

    def test_unescapes_entities(self):
        result = _html_to_text("Item&nbsp;1.&nbsp;Business &amp; Finance")
        assert "&nbsp;" not in result
        assert "&amp;" not in result
        assert "Business" in result and "Finance" in result


# ---------------------------------------------------------------------------
# _extract_sections
# ---------------------------------------------------------------------------

class TestExtractSections:
    def test_10q_uses_10q_item_numbering(self):
        text = (
            "Item 1. Financial Statements\n" + "fs content " * 30 +
            "Item 2. Management's Discussion\n" + "mda content " * 30
        )
        result = _extract_sections(text, ["all"], max_chars=5000, form_type="10-Q")
        assert "financial_statements" in result
        assert "mda" in result
        assert "risk_factors" not in result  # not present in this sample text

    def test_10k_uses_10k_item_numbering(self):
        text = (
            "Item 1. Business\n" + "business content " * 30 +
            "Item 1A. Risk Factors\n" + "risk content " * 30 +
            "Item 7. Management's Discussion\n" + "mda content " * 30 +
            "Item 8. Financial Statements\n" + "fs content " * 30
        )
        result = _extract_sections(text, ["all"], max_chars=5000, form_type="10-K")
        assert set(result.keys()) == {"business", "risk_factors", "mda", "financial_statements"}

    def test_specific_sections_filter_others_out(self):
        text = (
            "Item 1. Business\n" + "business content " * 30 +
            "Item 1A. Risk Factors\n" + "risk content " * 30
        )
        result = _extract_sections(text, ["risk_factors"], max_chars=5000, form_type="10-K")
        assert "risk_factors" in result
        assert "business" not in result

    def test_max_chars_truncates_section_body(self):
        text = "Item 1. Business\n" + "x" * 10000
        result = _extract_sections(text, ["business"], max_chars=100, form_type="10-K")
        assert len(result["business"]) <= 100

    def test_empty_text_returns_empty_dict(self):
        assert _extract_sections("", ["all"], max_chars=100, form_type="10-K") == {}

    def test_uses_last_occurrence_not_first(self):
        """The first hit is almost always a table-of-contents link; the
        real section body comes later — this locks in that behavior."""
        text = (
            "Item 1A. Risk Factors (see page 12)\n" + "toc filler " * 10 +
            "Item 1A. Risk Factors\n" + "REAL RISK CONTENT HERE " * 20
        )
        result = _extract_sections(text, ["risk_factors"], max_chars=5000, form_type="10-K")
        assert "REAL RISK CONTENT" in result["risk_factors"]
        assert "toc filler" not in result["risk_factors"]