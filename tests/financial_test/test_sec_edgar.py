"""
Tests for: tools/financial_tools/sec_edgar.py  (sync, requests-based)

NOTE: This project has TWO files named sec_edgar.py:
  - tools/financial_tools/sec_edgar.py  (this file)   — sync, uses `requests`
  - tools/research_tools/sec_edgar.py   (separate file) — async, uses `httpx`
They are tested separately (see test_sec_edgar_research.py for the other).
Make sure your test runner resolves the correct one via package path —
do NOT rely on bare `import sec_edgar`, always use the full dotted path.

Mocking strategy: requests.get is mocked via patch() on the module's own
`requests` reference (tools.financial_tools.sec_edgar.requests.get), so no
real HTTP calls are made and SEC's rate limits are never hit during tests.
"""
import time
from unittest.mock import patch, MagicMock
import pytest

from tools.financial_tools.sec_edgar import (
    _pad_cik,
    get_cik,
    list_filings,
    get_filing_text,
    get_xbrl_financials,
)


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
    @patch("tools.financial_tools.sec_edgar.requests.get")
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
    @patch("tools.financial_tools.sec_edgar.requests.get")
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
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_fetch_failure_returns_error(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("network down")  # caught by `_get`, returns None

        result = get_cik("NVDA")

        assert result["cik"] is None
        assert "Failed to fetch company tickers index." in result["error"]

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_unexpected_exception_caught_in_outer_handler(self, mock_get, mock_sleep):
        # Force an exception that happens OUTSIDE `_get`'s own try/except,
        # e.g. ticker.upper() on a non-string, to exercise get_cik's own except block.
        result = get_cik(12345)  # int has no .upper()
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# list_filings
# ---------------------------------------------------------------------------

class TestListFilings:
    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar.requests.get")
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
    @patch("tools.financial_tools.sec_edgar.requests.get")
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
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_submission_fetch_failure_returns_error(self, mock_get, mock_get_cik, mock_sleep):
        mock_get_cik.return_value = {"ticker": "NVDA", "cik": "0001045810",
                                      "company_name": "NVIDIA CORP", "error": None}
        mock_get.side_effect = Exception("boom")

        result = list_filings("NVDA")
        assert result["filings"] == []
        assert "Failed to fetch submission data" in result["error"]


# ---------------------------------------------------------------------------
# get_filing_text
# ---------------------------------------------------------------------------

class TestGetFilingText:
    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_happy_path_strips_html_and_counts_words(self, mock_get, mock_sleep):
        idx_resp = MagicMock(status_code=200)
        idx_resp.json.return_value = {
            "documents": [{"type": "10-K", "documentUrl": "/Archives/edgar/data/x/doc.htm"}]
        }
        doc_resp = MagicMock()
        doc_resp.text = "<html><body>Hello   World</body></html>"
        doc_resp.raise_for_status.return_value = None

        mock_get.side_effect = [idx_resp, doc_resp]

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["error"] is None
        assert "Hello" in result["text"]
        assert result["word_count"] == 2

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_index_fetch_non_200_returns_error(self, mock_get, mock_sleep):
        idx_resp = MagicMock(status_code=404)
        mock_get.return_value = idx_resp

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["text"] is None
        assert "Index fetch failed" in result["error"]

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_no_documents_in_index_returns_error(self, mock_get, mock_sleep):
        idx_resp = MagicMock(status_code=200)
        idx_resp.json.return_value = {"documents": []}
        mock_get.return_value = idx_resp

        result = get_filing_text("0001045810-24-000010", "0001045810")

        assert result["text"] is None
        assert "No primary document found" in result["error"]

    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.requests.get")
    def test_falls_back_to_first_document_if_no_type_match(self, mock_get, mock_sleep):
        idx_resp = MagicMock(status_code=200)
        idx_resp.json.return_value = {
            "documents": [{"type": "EX-99.1", "documentUrl": "/Archives/edgar/data/x/ex.htm"}]
        }
        doc_resp = MagicMock()
        doc_resp.text = "plain text content"
        doc_resp.raise_for_status.return_value = None
        mock_get.side_effect = [idx_resp, doc_resp]

        result = get_filing_text("0001045810-24-000010", "0001045810")
        assert result["error"] is None
        assert result["text"] == "plain text content"


# ---------------------------------------------------------------------------
# get_xbrl_financials
# ---------------------------------------------------------------------------

class TestGetXbrlFinancials:
    @patch("tools.financial_tools.sec_edgar.time.sleep")
    @patch("tools.financial_tools.sec_edgar.get_cik")
    @patch("tools.financial_tools.sec_edgar.requests.get")
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
    @patch("tools.financial_tools.sec_edgar.requests.get")
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