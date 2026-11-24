from dataclasses import replace
from types import MappingProxyType

import pytest

from core.error_handler import DataUnavailableAsOfError, VersionMismatchError
from tests.unit_tests.market_test import context, feature_set
from tests.unit_tests.market_test.test_market_server import service
from tools.market_tools.specialist_fusion import SPECIALIST_FUSION_VARIANT_TOKEN


def _fusion_service():
    return replace(
        service(),
        model_variant=f"msft_hybrid_v1+{SPECIALIST_FUSION_VARIANT_TOKEN}",
    )


def _with_value(features, name, value):
    values = dict(features.values)
    values[name] = value
    return replace(features, values=MappingProxyType(values))


def test_each_specialist_changes_final_numeric_forecast_but_not_base_model():
    runtime = _fusion_service()
    features = feature_set()
    baseline = runtime.forecast(
        context=context(), market_features=features, run_id="fusion-baseline"
    )
    counterfactuals = {
        "research": _with_value(features, "research_directional_score", 0.9),
        "financial": _with_value(features, "financial_revenue_growth", 0.80),
        "sentiment": _with_value(features, "sentiment_aggregate_7d", -0.80),
    }
    for family, changed_features in counterfactuals.items():
        changed = runtime.forecast(
            context=context(), market_features=changed_features,
            run_id=f"fusion-counterfactual-{family}",
        )
        assert changed["base_model_expected_return_21d"] == baseline["base_model_expected_return_21d"]
        assert changed["base_model_probability_positive"] == baseline["base_model_probability_positive"]
        assert changed["expected_return_21d"] != baseline["expected_return_21d"]
        assert changed["probability_positive"] != baseline["probability_positive"]


def test_fusion_exposes_auditable_contribution_from_all_specialists():
    result = _fusion_service().forecast(
        context=context(), market_features=feature_set(), run_id="fusion-audit"
    )
    assert set(result["specialist_scores"]) == {"research", "financial", "sentiment"}
    assert set(result["specialist_return_adjustments"]) == {"research", "financial", "sentiment"}
    assert result["specialist_total_return_adjustment"] == pytest.approx(
        sum(result["specialist_return_adjustments"].values())
    )
    assert result["expected_return_21d"] == pytest.approx(
        result["base_model_expected_return_21d"]
        + result["specialist_total_return_adjustment"]
    )


def test_fusion_fails_closed_when_any_specialist_is_missing():
    features = feature_set()
    indicators = dict(features.missing_indicators)
    indicators["sentiment_features__missing"] = True
    missing = replace(features, missing_indicators=MappingProxyType(indicators))
    with pytest.raises(DataUnavailableAsOfError, match="Sentiment"):
        _fusion_service().forecast(
            context=context(), market_features=missing, run_id="fusion-missing"
        )


def test_runtime_rejects_ticker_outside_trained_universe():
    nvda_context = replace(context(), ticker="NVDA")
    nvda_features = replace(feature_set(), ticker="NVDA")
    with pytest.raises(VersionMismatchError, match="outside the model universe"):
        _fusion_service().forecast(
            context=nvda_context, market_features=nvda_features, run_id="nvda-rejected"
        )
