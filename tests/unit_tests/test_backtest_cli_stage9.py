import json
from pathlib import Path

from evaluation.backtest_cli import build_report, main
from evaluation.stage3_compare_cli import compare_backtest_artifacts


FIXTURE = Path("tests/fixtures/stage9_walk_forward.json")


def test_cli_builds_offline_preliminary_report():
    report = build_report(FIXTURE, run_id="stage09-cli-test")
    assert report["manifest"]["scientific_report_level"] == "PRELIMINARY"
    assert report["leakage_audit"]["future_record_count"] == 0
    assert "full_model" in report["metrics"]


def test_cli_output_is_machine_readable(tmp_path):
    target = tmp_path / "report.json"
    assert main([str(FIXTURE), "--run-id", "stage09-cli-test", "--output", str(target)]) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["manifest"]["transaction_cost_bps_one_way"] == 10.0


def test_stage3_compare_retains_legacy_path_and_adds_quant_backtest_comparison():
    report = build_report(FIXTURE, run_id="stage09-compare-test")
    result = compare_backtest_artifacts(report, report)
    assert result["identical_eligible_rows"] is True
    assert result["forecast_accuracy_source"] == "deterministic_quant_metrics"
    assert result["ragas_used_for_forecast_accuracy"] is False


def test_cli_injects_local_telemetry_without_changing_scientific_output():
    disabled = build_report(FIXTURE, run_id="stage09-r1-a1")
    events = []
    enabled = build_report(FIXTURE, run_id="stage09-r1-a1",
                           event_sink=lambda event: events.append(dict(event)))
    assert enabled == disabled
    boundaries = [event["boundary"] for event in events]
    assert boundaries[0] == "fixture_materialization"
    assert boundaries[1] == "run"
    assert boundaries[-1] == "run"
    assert enabled["manifest"]["report_hash"] == "213ad73328c244c495d0a465ed1bc7b02fb1ab2348a803e6ab807948f3aad779"
