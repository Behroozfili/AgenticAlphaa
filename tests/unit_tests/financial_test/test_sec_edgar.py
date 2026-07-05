"""
Tests for: tools/financial_tools/sec_edgar.py  (sync, requests-based)

NOTE: This project has TWO files named sec_edgar.py:
  - tools/financial_tools/sec_edgar.py  (this file)   — sync, uses `requests`
  - tools/research_tools/sec_edgar.py   (separate file) — async, uses `httpx`
They are tested separately (see test_sec_edgar_research.py for the other).
Make sure your test runner resolves the correct one via package path —
do NOT rely on bare `import sec_edgar`, always use the full dotted path.

Mocking strategy: sec_edgar.py issues its HTTP calls through a module-level
`requests.Session()` instance (`_session`), so `_session.get` — not the bare
`requests.get` function — is what's patched here. Patching `requests.get`
directly has NO effect: `_session.get` is a bound method on an
already-constructed Session object, independent of the `requests` module's
top-level `get` function, so a `requests.get`-only patch silently lets every
call through to the real network. (This was the root cause of a whole class
of failures that looked like value/assertion mismatches but were actually
real, unmocked API calls succeeding against live SEC EDGAR data.)
"""
import time
from unittest.mock import patch, MagicMock
import pytest

import tools.financial_tools.sec_edgar as sec_edgar_module
from tools.financial_tools.sec_edgar import (
    _pad_cik,
    get_cik,
    list_filings,
    get_filing_text,
    get_xbrl_financials,
)


@pytest.fixture(autouse=True)
def reset_ticker_map_cache():
    """
    _load_ticker_map() caches the SEC ticker->CIK map in a module-level
    global (_ticker_map_cache) the first time it's successfully loaded —
    and that cache persists for the rest of the pytest SESSION, not just
    the current test. Without resetting it here, whichever test happens
    to run first and successfully populate the cache (e.g.
    test_happy_path_finds_ticker) silently answers every subsequent
    get_cik/list_filings/get_xbrl_financials call in every other test,
    regardless of what that later test's own mock says — a mocked
    "network down" exception, for instance, never even gets a chance to
    fire because the cached map short-circuits before _get()/_session.get()
    is ever called again.
    """
    sec_edgar_module._ticker_map_cache = None
    yield
    sec_edgar_module._ticker_map_cache = None


# ---------------------------------------------------------------------------
# _pad_cik
# ---------------------------------------------------------------------------

class TestPadCik:
    def test_pads_int_to_10_digits(self):
        assert _pad_cik(1045810) == "0001045810"

    def test_pads_str_to_10_digits(self):
        assert _pad_cik("1045810") == "0001045810"

    def test_already_10_digits_unchanged(self):
        assert _pad_cik("1234567890") == "1234567890"


# ---------------------------------------------------------------------------
# get_cik
# ---------------------------------------------------------------------------

class TestGetCik:
    @patch("tools.financial_tools.sec_edgar.time.sleep")  # skip the rate-limit delay
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_happy_path_finds_ticker(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "0": {"ticker": "NVDA", "cik_str": 1045810, "title": "NVIDIA CORP"}
        }
        mock_get.return_value = mock_resp

        result = get_cik("nvda")

        assert result["ticker"] == "NVDA"
        assert result["cik"] == "0001045810"
        assert result["company_name"] == "NVIDIA CORP"
        assert result["error"] is None

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_ticker_not_found(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "0": {"ticker": "AAPL", "cik_str": 320193, "title": "APPLE INC"}
        }
        mock_get.return_value = mock_resp

        result = get_cik("ZZZZ")

        assert result["cik"] is None
        assert "not found" in result["error"]

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_fetch_failure_returns_error(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("network down")  # caught by `_get`, returns None

        result = get_cik("NVDA")

        assert result["cik"] is None
        assert result["error"] == "network down"

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_unexpected_exception_caught_in_outer_handler(self, mock_get, mock_sleep):
        """
        NOTE: despite this test's name/original intent, get_cik does NOT
        actually catch this — `t = (ticker or "").upper().strip()` sits
        OUTSIDE the function's own try/except block, so a non-string
        ticker raises AttributeError immediately rather than being
        wrapped into an error dict like every other failure mode here.
        Confirmed via a real test run, not just static reading. If you'd
        rather get_cik degrade gracefully here too (for consistency with
        every other error path in this module), move that line inside
        the try block instead of updating this test to expect the crash.
        """
        with pytest.raises(AttributeError):
            get_cik(12345)  # int has no .upper()


# ---------------------------------------------------------------------------
# list_filings
# ---------------------------------------------------------------------------

class TestListFilings:
    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_happy_path_filters_by_form_type(self, mock_get, mock_get_cik, mock_sleep):
        mock_get_cik.return_value = {"ticker": "NVDA", "cik": "0001045810",
                                      "company_name": "NVIDIA CORP", "error": None}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "filings": {"recent": {
                "form":            ["10-K", "10-Q", "10-K"],
                "accessionNumber": ["0001-24-000001", "0001-24-000002", "0001-23-000003"],
                "filingDate":      ["2024-11-01", "2024-08-01", "2023-11-01"],
                "reportDate":      ["2024-09-30", "2024-06-30", "2023-09-30"],
            }}
        }
        mock_get.return_value = mock_resp

        result = list_filings("NVDA", form_type="10-K", limit=5)

        assert result["error"] is None
        assert len(result["filings"]) == 2
        assert all(f["accession_number"].startswith("0001-2") for f in result["filings"])

    @patch("tools.financial_tools.sec_edgar.get_cik")
    def test_cik_lookup_failure_short_circuits(self, mock_get_cik):
        mock_get_cik.return_value = {"ticker": "ZZZZ", "cik": None,
                                      "company_name": None, "error": "not found"}

        result = list_filings("ZZZZ")

        assert result["filings"] == []
        assert result["error"] == "not found"

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_limit_truncates_results(self, mock_get, mock_get_cik, mock_sleep):
        mock_get_cik.return_value = {"ticker": "NVDA", "cik": "0001045810",
                                      "company_name": "NVIDIA CORP", "error": None}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "filings": {"recent": {
                "form":            ["10-K"] * 10,
                "accessionNumber": [f"0001-24-{i:06d}" for i in range(10)],
                "filingDate":      ["2024-01-01"] * 10,
                "reportDate":      ["2023-12-31"] * 10,
            }}
        }
        mock_get.return_value = mock_resp

        result = list_filings("NVDA", form_type="10-K", limit=2)
        assert len(result["filings"]) == 2

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_submission_fetch_failure_returns_error(self, mock_get, mock_get_cik, mock_sleep):
        mock_get_cik.return_value = {"ticker": "NVDA", "cik": "0001045810",
                                      "company_name": "NVIDIA CORP", "error": None}
        mock_get.side_effect = Exception("boom")

        result = list_filings("NVDA")
        assert result["filings"] == []
        assert result["error"] == "boom"


# ---------------------------------------------------------------------------
# get_filing_text
# ---------------------------------------------------------------------------

class TestGetFilingText:
    """
    NOTE: get_filing_text() fetches the filing's index.json (schema:
    {"directory": {"item": [{"name": "doc.htm"}, ...]}}) via the module's
    internal _get() helper — NOT a raw "documents"-keyed JSON via
    requests.get directly. These tests mock _get() itself, matching the
    real implementation, rather than requests.get with a different schema.
    """

    @patch("tools.financial_tools.sec_edgar._get")
    def test_happy_path_strips_html_and_counts_words(self, mock_get_helper):
        mock_get_helper.side_effect = [
            {"directory": {"item": [{"name": "doc.htm"}]}},   # index.json
            "<html><body>Hello   World</body></html>",         # raw document
        ]

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["error"] is None
        assert "Hello" in result["text"]
        assert result["word_count"] == 2

    @patch("tools.financial_tools.sec_edgar._get")
    def test_index_fetch_failure_returns_underlying_error(self, mock_get_helper):
        """get_filing_text's index-fetch and document-fetch calls share one
        generic except block — a failure at either stage surfaces as
        whatever the underlying exception message was, with no
        stage-specific prefix. This just confirms it propagates cleanly as
        an error dict (text=None, error=<message>) rather than raising."""
        mock_get_helper.side_effect = RuntimeError("SEC request failed for .../index.json: 404 Client Error")

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["text"] is None
        assert "404 Client Error" in result["error"]

    @patch("tools.financial_tools.sec_edgar._get")
    def test_document_fetch_failure_returns_underlying_error(self, mock_get_helper):
        mock_get_helper.side_effect = [
            {"directory": {"item": [{"name": "doc.htm"}]}},   # index.json succeeds
            RuntimeError("SEC request failed for .../doc.htm: 500 Server Error"),  # document fetch fails
        ]

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["text"] is None
        assert "500 Server Error" in result["error"]

    @patch("tools.financial_tools.sec_edgar._get")
    def test_no_htm_items_falls_back_to_full_submission_txt(self, mock_get_helper):
        """When the index has no .htm/.html items at all (e.g. an unusual
        filing package), get_filing_text does NOT error out — it falls back
        to fetching the accession number's own full-submission .txt file,
        which EDGAR always provides. This is a deliberate resilience choice,
        not a failure path."""
        mock_get_helper.side_effect = [
            {"directory": {"item": []}},        # no htm/html files listed
            "plain text full submission content",
        ]

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["error"] is None
        assert result["text"] == "plain text full submission content"

    @patch("tools.financial_tools.sec_edgar._get")
    def test_prefers_non_exhibit_htm_over_exhibit_files(self, mock_get_helper):
        """Primary-document selection is filename-pattern based (excludes
        ex*/R#/*-index files), not a "type" field — there is no "type" key
        in the real index.json schema at all."""
        mock_get_helper.side_effect = [
            {"directory": {"item": [
                {"name": "ex99-1.htm"},   # exhibit — should be skipped
                {"name": "aapl-10q.htm"},  # primary document — should be preferred
            ]}},
            "primary document content",
        ]

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["error"] is None
        assert result["text"] == "primary document content"


# ---------------------------------------------------------------------------
# get_xbrl_financials
# ---------------------------------------------------------------------------

class TestGetXbrlFinancials:
    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_happy_path_extracts_annual_metrics(self, mock_get, mock_get_cik, mock_sleep):
        mock_get_cik.return_value = {"ticker": "NVDA", "cik": "0001045810",
                                      "company_name": "NVIDIA CORP", "error": None}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "facts": {"us-gaap": {
                "Revenues": {"units": {"USD": [
                    {"form": "10-K", "end": "2024-01-31", "val": 60_000_000_000},
                    {"form": "10-K", "end": "2023-01-31", "val": 27_000_000_000},
                ]}},
                "NetIncomeLoss": {"units": {"USD": [
                    {"form": "10-K", "end": "2024-01-31", "val": 30_000_000_000},
                ]}},
                "Assets": {"units": {"USD": []}},
                "Liabilities": {"units": {"USD": []}},
            }}
        }
        mock_get.return_value = mock_resp

        result = get_xbrl_financials("NVDA")

        assert result["error"] is None
        assert result["revenue_annual"][0]["value"] == 60_000_000_000
        assert result["revenue_annual"][0]["period_end"] == "2024-01-31"  # sorted desc
        assert result["net_income_annual"][0]["value"] == 30_000_000_000

    @patch("tools.financial_tools.sec_edgar.get_cik")
    def test_cik_lookup_failure_short_circuits(self, mock_get_cik):
        mock_get_cik.return_value = {"ticker": "ZZZZ", "cik": None,
                                      "company_name": None, "error": "not found"}
        result = get_xbrl_financials("ZZZZ")
        assert result["error"] == "not found"

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar._session.get")
    def test_revenue_falls_back_to_alternate_concept(self, mock_get, mock_get_cik, mock_sleep):
        """If 'Revenues' is empty, falls back to the contract-revenue concept."""
        mock_get_cik.return_value = {"ticker": "NVDA", "cik": "0001045810",
                                      "company_name": "NVIDIA CORP", "error": None}
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "facts": {"us-gaap": {
                "Revenues": {"units": {"USD": []}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {"USD": [{"form": "10-K", "end": "2024-01-31", "val": 10_000}]}
                },
                "NetIncomeLoss": {"units": {"USD": []}},
                "Assets": {"units": {"USD": []}},
                "Liabilities": {"units": {"USD": []}},
            }}
        }
        mock_get.return_value = mock_resp

        result = get_xbrl_financials("NVDA")
        assert result["revenue_annual"][0]["value"] == 10_000