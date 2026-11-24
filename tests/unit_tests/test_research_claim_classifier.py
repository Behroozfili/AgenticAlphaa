from agents.research_agent import _research_window_context
from tests.unit_tests.market_test import context
from tools.research_tools import build_research_features
from tools.research_tools.claim_classifier import classify_research_evidence


def _evidence(evidence_id: str, text: str, source: str = "tavily_search") -> dict:
    ctx = context()
    return {
        "evidence_id": evidence_id,
        "source": source,
        "provider": source,
        "ticker": ctx.ticker,
        "timestamp": ctx.as_of.isoformat().replace("+00:00", "Z"),
        "dataset_version": ctx.dataset_version,
        "kind": "evidence",
        "title": text,
        "content_excerpt": text,
        "citation": {"source": source, "url": f"https://example.test/{evidence_id}"},
        "retrieval_origins": ["vector", "graph"] if source == "rag_hybrid_query" else [],
    }


def test_classifier_is_directional_bounded_and_evidence_linked():
    claims = classify_research_evidence([
        _evidence("positive-1", "Microsoft raises guidance after record revenue and strong demand"),
        _evidence("negative-1", "Microsoft lowers guidance amid antitrust investigation and weak demand"),
        _evidence("neutral-1", "Microsoft publishes a routine corporate update"),
    ], context=context())

    by_id = {claim["evidence_id"]: claim for claim in claims}
    assert by_id["positive-1"]["direction"] == "POSITIVE"
    assert by_id["negative-1"]["direction"] == "NEGATIVE"
    assert by_id["neutral-1"]["direction"] == "NEUTRAL"
    assert all(0.0 <= claim["confidence"] <= 1.0 for claim in claims)
    assert all(0.0 <= claim["recency_weight"] <= 1.0 for claim in claims)


def test_research_window_is_narrowed_without_mutating_canonical_market_context():
    canonical = context()
    narrowed = _research_window_context(canonical, {"days_back": 30})

    assert canonical.lookback_days > 30
    assert narrowed.lookback_days == 30
    assert narrowed.as_of == canonical.as_of
    assert narrowed.target == canonical.target

def test_research_features_use_strength_confidence_and_recency_not_raw_counts():
    features = build_research_features([
        _evidence("strong-positive", "raises guidance with record revenue and strong demand", "rag_hybrid_query"),
        _evidence("weak-negative", "product delayed after a routine update", "news_search"),
    ], context=context(), run_id="research-claims-v2")

    assert features.classified_claim_count == 2
    assert features.directional_score > 0.0
    assert features.claim_confidence > 0.0
    assert features.catalysts == ("strong-positive",)
    assert features.risks == ("weak-negative",)
    assert {claim["evidence_id"] for claim in features.claims} == {
        "strong-positive", "weak-negative",
    }
    assert features.vector_coverage is True
    assert features.graph_coverage is True
