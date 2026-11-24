from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError
from tests.unit_tests.market_test import context, feature_set, fixture, market_bundle, training_rows
from tools.market_tools.historical_dataset_builder import (
    LABEL_RETURN_ABS_TOLERANCE, MONTH_TO_DATE_CUTOFF_POLICY_VERSION,
    DataReadiness, build_data_readiness,
    build_feature_snapshot, materialize_outcome, labeled_snapshot_from_mapping,
    validate_universe_manifest)


def readiness(rows=None):
    return build_data_readiness(rows=training_rows() if rows is None else rows,
        universe_manifest=fixture()["universe"], run_id="stage07-fixture",
        source_hashes={"fixture": "locked-local"})


def test_data_readiness_passes_before_any_fit_and_has_required_coverage():
    result = readiness()
    assert result.status == "PASS"
    assert result.manifest["ticker_count"] == 1
    assert result.manifest["monthly_cutoff_count"] == 40
    assert result.manifest["eligible_forecast_count"] == 40
    assert result.manifest["news_coverage"] == 1.0
    assert result.manifest["filing_coverage"] == 1.0
    assert result.manifest["graph_coverage"] == 1.0
    assert result.manifest["scientific_report_level"] == "PRELIMINARY"
    assert result.manifest["manifest_hash"] == result.manifest_hash


def test_data_readiness_records_month_to_date_training_policy():
    result = build_data_readiness(
        rows=training_rows(), universe_manifest=fixture()["universe"],
        run_id="stage07-mtd-fixture",
        cutoff_policy_version=MONTH_TO_DATE_CUTOFF_POLICY_VERSION,
    )
    assert result.manifest["cutoff_policy_version"] == "monthly_to_date_v1"


def test_empty_dataset_cannot_claim_data_readiness_pass():
    with pytest.raises(DataUnavailableAsOfError, match="DATA_READINESS"):
        build_data_readiness(rows=[], universe_manifest=fixture()["universe"], run_id="empty")


def test_universe_template_like_unset_values_are_rejected():
    with pytest.raises(ValueError, match="approved universe"):
        validate_universe_manifest({"universe_version": "UNSET", "source": "UNSET", "constituents": [{}]})


def test_snapshot_finalization_after_as_of_is_temporal_leakage():
    ctx = context()
    with pytest.raises(TemporalLeakageError, match="no later than as_of"):
        build_feature_snapshot(context=ctx, features=feature_set(), finalized_at=ctx.as_of + timedelta(seconds=1))


def test_target_is_exact_21st_session_and_label_uses_adjusted_closes():
    ctx = context()
    snapshot = build_feature_snapshot(context=ctx, features=feature_set(), finalized_at=ctx.as_of)
    outcome = materialize_outcome(snapshot=snapshot, market=market_bundle(),
        outcome_materialized_at=f"{ctx.target_date.isoformat()}T23:00:00Z")
    assert snapshot.target_date == "2024-07-30"
    assert outcome.direction_positive in (0, 1)
    assert outcome.ticker_total_return != outcome.benchmark_total_return


def test_outcome_before_target_is_rejected():
    ctx = context()
    snapshot = build_feature_snapshot(context=ctx, features=feature_set(), finalized_at=ctx.as_of)
    with pytest.raises(TemporalLeakageError, match="before target"):
        materialize_outcome(snapshot=snapshot, market=market_bundle(),
            outcome_materialized_at="2024-07-26T23:00:00Z")


def _row_payload(row):
    return {
        "snapshot": row.snapshot.canonical_payload(),
        "ticker_total_return": row.ticker_total_return,
        "benchmark_total_return": row.benchmark_total_return,
        "forward_21d_excess_return": row.forward_21d_excess_return,
        "direction_positive": row.direction_positive,
        "outcome_materialized_at": row.outcome_materialized_at,
    }


def test_fixture_loader_rejects_outcome_before_target_availability():
    payload = _row_payload(training_rows(1)[0])
    payload["outcome_materialized_at"] = payload["snapshot"]["as_of"]
    with pytest.raises(TemporalLeakageError, match="target availability"):
        labeled_snapshot_from_mapping(payload)


@pytest.mark.parametrize("violation", ["excess_return", "direction"])
def test_fixture_loader_rejects_noncanonical_label_identity(violation):
    row = training_rows(1)[0]
    payload = _row_payload(row)
    if violation == "excess_return":
        payload["forward_21d_excess_return"] += LABEL_RETURN_ABS_TOLERANCE * 10
        expected_reason = "inconsistent_excess_return"
    else:
        payload["direction_positive"] = 1 - row.direction_positive
        expected_reason = "inconsistent_direction_label"
    with pytest.raises(ValueError, match=expected_reason):
        labeled_snapshot_from_mapping(payload)


@pytest.mark.parametrize("violation", ["excess_return", "direction"])
def test_data_readiness_fails_closed_for_noncanonical_label_identity(violation):
    rows = list(training_rows())
    row = rows[-1]
    if violation == "excess_return":
        rows[-1] = replace(
            row,
            forward_21d_excess_return=(
                row.forward_21d_excess_return + LABEL_RETURN_ABS_TOLERANCE * 10
            ),
        )
    else:
        rows[-1] = replace(row, direction_positive=1 - row.direction_positive)
    with pytest.raises(DataUnavailableAsOfError, match="DATA_READINESS"):
        readiness(rows)


def test_data_readiness_documents_canonical_label_identity_tolerance():
    contract = readiness().manifest["label_contract"]
    assert contract == {
        "version": "forward_21d_excess_return-v1",
        "excess_return_formula": "ticker_total_return-minus-benchmark_total_return",
        "direction_positive_rule": "1-iff-canonical-excess-return-greater-than-0",
        "excess_return_abs_tolerance": LABEL_RETURN_ABS_TOLERANCE,
        "excess_return_rel_tolerance": 0.0,
    }


def test_data_readiness_rejects_any_out_of_universe_row():
    universe = deepcopy(fixture()["universe"])
    universe["constituents"][0]["ticker"] = "AAPL"
    with pytest.raises(DataUnavailableAsOfError, match="DATA_READINESS"):
        build_data_readiness(
            rows=training_rows(), universe_manifest=universe, run_id="universe-check"
        )


@pytest.mark.parametrize(("field", "value"), [
    ("target_date", "2022-06-01"),
    ("finalized_at", "2022-06-01T20:00:00Z"),
])
def test_data_readiness_rejects_invalid_temporal_row(field, value):
    rows = list(training_rows())
    rows[-1] = replace(rows[-1], snapshot=replace(rows[-1].snapshot, **{field: value}))
    with pytest.raises(DataUnavailableAsOfError, match="DATA_READINESS"):
        readiness(rows)


def test_source_temporal_coverage_is_explicit_not_fabricated():
    coverage = readiness().manifest["source_coverage"]
    assert coverage["market"]["temporal_status"] == "VERIFIED_SNAPSHOT_RANGE"
    for source in ("research_news", "sec_filings", "temporal_graph"):
        assert coverage[source]["date_start"] is None
        assert coverage[source]["date_end"] is None
        assert coverage[source]["temporal_status"] == "UNVERIFIED_REFERENCE_ONLY"


@pytest.mark.parametrize("run_id", [
    "PRIVATE_SENTINEL", " multiline\nrun", " leading", "trailing ", "x" * 129,
])
def test_data_readiness_rejects_nonopaque_run_id(run_id):
    with pytest.raises(ValueError, match="run_id"):
        build_data_readiness(
            rows=training_rows(), universe_manifest=fixture()["universe"], run_id=run_id
        )


def test_data_readiness_rejects_raw_source_hash_metadata():
    with pytest.raises(ValueError, match="evidence"):
        build_data_readiness(
            rows=training_rows(), universe_manifest=fixture()["universe"],
            run_id="source-hash-check",
            source_hashes={"fixture": "PRIVATE_SENTINEL raw fixture body"},
        )
