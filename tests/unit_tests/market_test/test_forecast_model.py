from dataclasses import replace
from types import MappingProxyType

import pytest

from core.error_handler import DataUnavailableAsOfError, TemporalLeakageError
from tests.unit_tests.market_test import feature_set, fixture, training_rows
from tests.unit_tests.market_test.test_historical_dataset_builder import readiness
from tools.market_tools.forecast_model import (ACTION_POLICY_VERSION, apply_action_policy,
    fit_deterministic_model, predict)
from tools.market_tools.historical_dataset_builder import DataReadiness


def test_model_fit_requires_data_readiness_pass():
    blocked = DataReadiness(MappingProxyType({"status": "BLOCKED"}), "blocked")
    with pytest.raises(DataUnavailableAsOfError, match="DATA_READINESS"):
        fit_deterministic_model(rows=training_rows(), readiness=blocked)


def test_model_requires_36_distinct_monthly_cutoffs():
    rows = training_rows(35)
    with pytest.raises(DataUnavailableAsOfError, match="36"):
        fit_deterministic_model(rows=rows, readiness=readiness(rows))


def test_logistic_ridge_artifact_and_forecast_are_deterministic():
    rows = training_rows()
    gate = readiness(rows)
    first = fit_deterministic_model(rows=rows, readiness=gate)
    second = fit_deterministic_model(rows=reversed(rows), readiness=gate)
    assert first.artifact() == second.artifact()
    assert first.model_version == second.model_version
    p1, p2 = predict(first, feature_set()), predict(second, feature_set())
    assert p1 == p2
    assert 0.0 <= p1.probability_positive <= 1.0
    assert p1.confidence_interval[0] <= p1.expected_return_21d <= p1.confidence_interval[1]
    assert p1.action_policy_version == ACTION_POLICY_VERSION


def test_training_fold_rejects_rows_after_train_through():
    rows = training_rows()
    with pytest.raises(TemporalLeakageError, match="after train_through"):
        fit_deterministic_model(rows=rows, readiness=readiness(rows),
            train_through=rows[-2].snapshot.as_of)


@pytest.mark.parametrize(("probability", "excess", "action"), [
    (0.55, 0.02, "BUY"), (0.45, -0.02, "AVOID"),
    (0.80, 0.0, "HOLD"), (0.20, 0.0, "HOLD"), (0.50, 0.03, "HOLD")])
def test_action_policy_v1_is_pre_registered(probability, excess, action):
    assert apply_action_policy(probability, excess) == action


def test_all_missing_optional_column_is_selected_out_not_zero_filled():
    rows = training_rows()
    changed = tuple(replace(row, snapshot=replace(row.snapshot,
        values=MappingProxyType({**row.snapshot.values, "all_missing": None}))) for row in rows)
    model = fit_deterministic_model(rows=changed, readiness=readiness(changed))
    assert "all_missing" not in model.feature_names


def test_model_fit_reproduces_readiness_eligible_row_hash():
    rows = training_rows()
    gate = readiness(rows)
    model = fit_deterministic_model(rows=rows, readiness=gate)
    assert model.eligible_rows_hash == gate.manifest["content_hashes"]["eligible_rows_sha256"]


def test_modified_rows_cannot_fit_under_original_readiness_manifest():
    rows = training_rows()
    gate = readiness(rows)
    changed = list(rows)
    changed[0] = replace(
        changed[0], ticker_total_return=changed[0].ticker_total_return + 0.001
    )
    with pytest.raises(ValueError, match="eligible-row content hash"):
        fit_deterministic_model(rows=changed, readiness=gate)
