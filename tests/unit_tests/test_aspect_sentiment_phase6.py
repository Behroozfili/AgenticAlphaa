from evaluation.aspect_sentiment import extract_aspect_observations, supported_causal_aspects

def test_observation_links_document_aspect_polarity_and_span():
    rows = extract_aspect_observations(["Investors worry that valuation is expensive. Revenue growth remains strong."], [{"url": "https://example.com/a", "title": "A"}], [{"label": "Bearish", "bearish_prob": .8, "bullish_prob": .1}])
    valuation = next(row for row in rows if row["aspect"] == "valuation")
    assert valuation["document_url"] == "https://example.com/a"
    assert valuation["polarity"] == "Bearish"
    assert "valuation" in valuation["evidence_span"].lower()

def test_unsupported_aspect_cannot_be_used_as_causal_explanation():
    rows = extract_aspect_observations(["The weather was calm."], [{}], [{"label": "Neutral"}])
    assert rows == []
    assert supported_causal_aspects(rows) == []

def test_supported_aspects_require_polarity_and_evidence():
    rows = [{"aspect": "valuation", "polarity": None, "evidence_span": "valuation concern"}, {"aspect": "earnings", "polarity": "Bullish", "evidence_span": "earnings beat"}]
    assert supported_causal_aspects(rows) == ["earnings"]
