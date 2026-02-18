from __future__ import annotations

from bid_scoring.policy.loader import load_policy_bundle
from bid_scoring.retrieval.policy_thresholds import (
    evaluate_retrieval_summary,
    extract_retrieval_thresholds,
)


def test_extract_retrieval_thresholds_from_policy_bundle():
    bundle = load_policy_bundle()

    thresholds = extract_retrieval_thresholds(bundle)

    assert "hybrid" in thresholds
    assert "mrr" in thresholds["hybrid"]


def test_evaluate_retrieval_summary_returns_violations():
    summary = {
        "hybrid": {"mrr": 0.3, "recall_at_5": 0.6},
        "vector": {"mrr": 0.4},
    }
    thresholds = {
        "hybrid": {"mrr": 0.5, "recall_at_5": 0.7},
        "vector": {"mrr": 0.35},
    }

    violations = evaluate_retrieval_summary(summary, thresholds)

    violation_keys = {(item.method, item.metric) for item in violations}
    assert ("hybrid", "mrr") in violation_keys
    assert ("hybrid", "recall_at_5") in violation_keys
    assert ("vector", "mrr") not in violation_keys
