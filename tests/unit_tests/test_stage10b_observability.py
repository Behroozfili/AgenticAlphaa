"""Stage 10B compact, local-only lifecycle telemetry contract."""

import time
from unittest.mock import patch

from langsmith_trace import (
    STAGE10B_RUNTIME_IDENTITY,
    emit_stage10b_event,
    stage10b_event,
)


PRIVATE_SENTINEL = "PRIVATE_QUERY_OR_RAW_EVIDENCE"


def _context(ticker: str = "AMD") -> dict:
    return {
        "ticker": ticker,
        "as_of": "2026-08-31T20:00:00+00:00",
        "target": "2026-09-30",
        "horizon_trading_days": 21,
        "analysis_mode": "LIVE",
        "dataset_version": "live-v1",
        "feature_version": "feature-v1",
        "model_version": "model-v1",
        "schema_version": "forecast-v1",
        "calendar_version": "xnys-v1",
    }


def test_runtime_identity_is_exact_for_current_attempt():
    assert dict(STAGE10B_RUNTIME_IDENTITY) == {
        "stage": "10B",
        "stage_revision": 1,
        "attempt": 2,
        "agent": "integration",
        "implementation_version": "stage10b-r1-a2",
    }


def test_event_contains_only_allowlisted_context_and_measured_latency():
    started_at = time.perf_counter()
    event = stage10b_event(
        "stage10b.analysis.completed",
        run_id="opaque-run-one",
        status="DEGRADED",
        context={
            **_context(),
            "query": PRIVATE_SENTINEL,
            "raw_evidence": [PRIVATE_SENTINEL],
            "brain_reasoning": PRIVATE_SENTINEL,
        },
        started_at=started_at,
        evidence_count=7,
        degraded_flags=["graph_coverage_low"],
    )

    assert event["run_id"] == "opaque-run-one"
    assert event["ticker"] == "AMD"
    assert event["horizon"] == 21
    assert event["evidence_count"] == 7
    assert event["degraded_flags"] == ["graph_coverage_low"]
    assert event["latency_ms"] >= 0
    assert not ({"query", "raw_evidence", "brain_reasoning"} & set(event))
    assert PRIVATE_SENTINEL not in repr(event)


def test_failure_uses_stable_code_without_exception_payload():
    event = stage10b_event(
        "stage10b.backtest.failed",
        run_id="opaque-backtest",
        status="FAILED",
        error_code="BACKTEST_EXECUTION_FAILED",
        context={"exception": RuntimeError(PRIVATE_SENTINEL)},
    )
    assert event["error_code"] == "BACKTEST_EXECUTION_FAILED"
    assert "exception" not in event
    assert PRIVATE_SENTINEL not in repr(event)


def test_invalid_or_private_metadata_fails_nonfatally():
    assert emit_stage10b_event(
        "stage10b.analysis.completed",
        run_id="opaque-run",
        status="COMPLETED",
        context=_context(),
        degraded_flags=[PRIVATE_SENTINEL + "\nunsafe"],
    ) is None
    assert emit_stage10b_event(
        "unsupported",
        run_id="opaque-run",
        status="COMPLETED",
    ) is None


def test_sink_failure_is_nonfatal_and_does_not_cross_talk():
    def failing_sink(_payload):
        raise RuntimeError(PRIVATE_SENTINEL)

    first = emit_stage10b_event(
        "stage10b.analysis.completed",
        run_id="run-amd",
        status="COMPLETED",
        context=_context("AMD"),
        sink=failing_sink,
    )
    second = emit_stage10b_event(
        "stage10b.analysis.completed",
        run_id="run-msft",
        status="COMPLETED",
        context=_context("MSFT"),
    )
    assert first is not None and second is not None
    assert (first["run_id"], first["ticker"]) == ("run-amd", "AMD")
    assert (second["run_id"], second["ticker"]) == ("run-msft", "MSFT")
    assert PRIVATE_SENTINEL not in repr(first)


def test_local_event_does_not_construct_external_client():
    with patch("langsmith_trace.Client") as client_type:
        event = emit_stage10b_event(
            "stage10b.backtest.started",
            run_id="opaque-backtest",
            status="STARTED",
        )
    assert event is not None
    client_type.assert_not_called()
