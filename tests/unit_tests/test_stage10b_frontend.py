"""Static smoke checks for the Stage 10B monthly/backtest UI contract."""

from pathlib import Path


HTML = (Path(__file__).resolve().parents[2] / "frontend" / "alpha-agent-app.html").read_text(
    encoding="utf-8"
)


def test_monthly_inputs_and_request_fields_are_present():
    assert 'id="as-of-input"' in HTML
    assert 'id="mode-input"' in HTML
    assert 'id="cutoff-policy-input"' in HTML
    assert "Object.assign(body,{as_of:asOf,horizon:21,mode,cutoff_policy:cutoffPolicy})" in HTML


def test_forecast_card_surfaces_required_compact_fields():
    for label in (
        "Target date", "Positive probability", "Expected return",
        "Expected excess return", "Uncertainty", "Action", "Graph coverage",
    ):
        assert label in HTML
    assert "forecastDisplayState(data)" in HTML
    assert "INSUFFICIENT_DATA" not in HTML or "forecastState" in HTML


def test_dcf_and_21_day_forecast_are_visually_separate():
    assert "DCF fair value is a long-term valuation feature" in HTML
    assert "not the expected 21-day price" in HTML


def test_backtest_ui_uses_async_job_status_and_compact_artifact():
    assert "async function runBacktest()" in HTML
    assert "/api/v1/backtests" in HTML
    assert "job.artifact_url" in HTML
    assert "Future evidence" in HTML
    assert "Preliminary evidence is not investment-performance evidence" in HTML


def test_full_monthly_pipeline_is_visible_and_uses_one_request():
    for node_id in ("ar-manager", "ar-research", "ar-financial", "ar-sentiment", "ar-quant", "ar-monthly_forecast"):
        assert f'id="{node_id}"' in HTML
    for event_type in (
        "feature_start", "feature_complete",
        "monthly_forecast_start", "monthly_forecast_complete", "forecast_error",
    ):
        assert f"case '{event_type}'" in HTML
    assert "new EventSource" in HTML
    assert "fetch(`${API_BASE}/api/v1/analyze`" in HTML
    assert "days_back:30" in HTML


def test_monthly_result_explains_specialist_influence_and_quality():
    for contract_field in (
        "specialist_contributions", "return_adjustments", "decision_drivers",
        "quality_gate_passed", "quality_gate_failures", "mf.narrative",
    ):
        assert contract_field in HTML
    assert "Specialist impact on forecast" in HTML
    assert "Evidence-backed decision drivers" in HTML
    assert "Why ${escapeHtml(mf.action" in HTML


def test_monthly_forecast_loads_verified_runtime_defaults():
    assert "/api/v1/forecast/defaults" in HTML
    assert "async function initializeForecastDefaults()" in HTML
    assert "defaults.active_tickers" in HTML
    assert "defaults.as_of" in HTML
    assert "timeZone:'America/New_York'" in HTML

def test_sentiment_card_uses_actual_compact_summary_fields():
    for field in (
        "fear_greed_score", "fear_greed_label", "fear_greed_confidence",
        "finbert_label", "vader_label", "total_chunks_analyzed",
        "unique_document_count", "quality_passed",
    ):
        assert f"sa.{field}" in HTML
    assert "sa.composite_score??sa.overall_score??sa.score" not in HTML


def test_decision_driver_ui_shows_reason_without_raw_evidence_ids():
    assert "if(driver.reason)" in HTML
    assert "driver.evidence_id" not in HTML
    assert "driver.evidence_ids" not in HTML