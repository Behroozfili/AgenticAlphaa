from dataclasses import replace

import pytest

from tests.unit_tests.market_test import context, feature_set, market_bundle
from tools.market_tools.feature_builder import build_market_features


def test_fixed_market_features_cover_required_windows_and_domains():
    features = feature_set()
    required = {"momentum_5", "momentum_21", "momentum_63", "volatility_21",
        "volatility_63", "volume_surprise_21", "drawdown_63", "beta_63",
        "relative_strength_21", "relative_strength_63", "research_graph_coverage",
        "financial_valuation_gap", "sentiment_aggregate_30d"}
    assert required <= set(features.values)
    assert not features.missing
    assert set(features.evidence_ids) == {"e1", "e2", "f1", "s1"}


def test_optional_missing_is_null_plus_indicator_not_zero():
    features = build_market_features(context=context(), market=market_bundle())
    assert features.values["financial_valuation_gap"] is None
    assert features.missing_indicators["financial_valuation_gap__missing"] is True
    assert features.numeric_vector()["financial_valuation_gap"] is None


def test_future_target_bar_cannot_change_as_of_features():
    original = market_bundle()
    bars = list(original.ticker.bars)
    bars[-1] = replace(bars[-1], adjusted_close=bars[-1].adjusted_close * 10)
    changed = replace(original, ticker=replace(original.ticker, bars=tuple(bars)))
    before = build_market_features(context=context(), market=original)
    after = build_market_features(context=context(), market=changed)
    assert dict(before.values) == dict(after.values)


def test_feature_mappings_are_immutable():
    features = feature_set()
    with pytest.raises(TypeError):
        features.values["momentum_21"] = 0.0


def test_sector_relative_strength_is_explicitly_missing_without_sector():
    assert feature_set().missing_indicators["sector_relative_strength_21__missing"] is True


@pytest.mark.parametrize("evidence_id", [
    "PRIVATE_SENTINEL raw article text", " evidence-1", "evidence-1 ",
    "https://example.invalid/private", "x" * 161,
])
def test_untrusted_evidence_references_fail_closed_before_state(evidence_id):
    upstream = {"values": {}, "evidence_ids": [evidence_id], "missing": False}
    with pytest.raises(ValueError, match="evidence"):
        build_market_features(
            context=context(), market=market_bundle(), research_features=upstream,
            run_id="privacy-counterexample",
        )


def test_state_slot_contains_only_opaque_evidence_and_truthful_provenance():
    features = feature_set()
    slot = features.to_state_slot()
    assert slot["evidence_ids"] == ["e1", "e2", "f1", "s1"]
    assert slot["values"]["feature_materialized_at"] == features.materialized_at
    assert slot["values"]["schema_version"] == context().schema_version
    assert slot["values"]["calendar_version"] == "XNYS-regular-v1"
    assert "PRIVATE_SENTINEL" not in repr(slot)


def test_private_identity_cannot_enter_market_features_or_state():
    unsafe_context = replace(context(), schema_version="PRIVATE_SENTINEL")
    with pytest.raises(ValueError, match="schema_version"):
        build_market_features(context=unsafe_context, market=market_bundle())
