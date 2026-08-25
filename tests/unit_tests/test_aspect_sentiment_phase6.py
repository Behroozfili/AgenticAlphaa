import pytest

from evaluation.aspect_sentiment import (
    extract_aspect_observations,
    supported_causal_aspects,
    valid_causal_observations,
)

def test_observation_links_document_aspect_polarity_and_span():
    rows = extract_aspect_observations(["Investors worry that valuation is expensive. Revenue growth remains strong."], [{"url": "https://example.com/a", "title": "A"}], [{"label": "Bearish", "bearish_prob": .8, "bullish_prob": .1}])
    valuation = next(row for row in rows if row["aspect"] == "valuation")
    assert valuation["document_url"] == "https://example.com/a"
    assert valuation["polarity"] == "Bearish"
    assert "valuation" in valuation["evidence_span"].lower()
    assert valuation["causal_language_supported"] is False

def test_unsupported_aspect_cannot_be_used_as_causal_explanation():
    rows = extract_aspect_observations(["The weather was calm."], [{}], [{"label": "Neutral"}])
    assert rows == []
    assert supported_causal_aspects(rows) == []

def test_supported_aspects_require_polarity_and_evidence():
    rows = [
        {
            "aspect": "valuation",
            "polarity": None,
            "evidence_span": "Valuation concerns weighed on sentiment.",
            "document_url": "https://example.com/valuation",
            "causal_language_supported": True,
        },
        {
            "aspect": "earnings",
            "polarity": "Bullish",
            "evidence_span": "The earnings beat boosted sentiment.",
            "document_url": "https://example.com/earnings",
            "causal_language_supported": True,
        },
    ]
    assert supported_causal_aspects(rows) == ["earnings"]


def test_legacy_observation_without_explicit_causal_flag_fails_closed():
    legacy = [{
        "aspect": "valuation",
        "polarity": "Bearish",
        "evidence_span": "Valuation concerns weighed on sentiment.",
        "document_url": "https://example.com/legacy",
    }]

    assert valid_causal_observations(legacy) == []
    assert supported_causal_aspects(legacy) == []


@pytest.mark.parametrize(
    "invalid_field",
    ["aspect", "polarity", "evidence_span", "document_url", "causal_language_supported"],
)
def test_causal_observation_requires_every_contract_field(invalid_field):
    row = {
        "aspect": "competition",
        "polarity": "Bearish",
        "evidence_span": "Competition weighed on investor sentiment.",
        "document_url": "https://example.com/competition",
        "causal_language_supported": True,
    }
    row[invalid_field] = False if invalid_field == "causal_language_supported" else None

    assert valid_causal_observations([row]) == []
    assert supported_causal_aspects([row]) == []
