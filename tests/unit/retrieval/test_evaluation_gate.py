from __future__ import annotations

from bid_scoring.retrieval.evaluation_gate import check_metric_thresholds


def test_check_metric_thresholds_passes_when_all_metrics_meet_minimum():
    summary = {
        "hybrid": {"mrr": 0.62, "recall_at_5": 0.85, "ndcg_at_5": 0.79},
    }
    thresholds = {
        "hybrid": {"mrr": 0.60, "recall_at_5": 0.80, "ndcg_at_5": 0.75},
    }

    violations = check_metric_thresholds(summary, thresholds)
    assert violations == []


def test_check_metric_thresholds_reports_below_minimum_values():
    summary = {
        "hybrid": {"mrr": 0.52, "recall_at_5": 0.78, "ndcg_at_5": 0.70},
    }
    thresholds = {
        "hybrid": {"mrr": 0.60, "recall_at_5": 0.80, "ndcg_at_5": 0.75},
    }

    violations = check_metric_thresholds(summary, thresholds)

    assert len(violations) == 3
    assert {v.metric for v in violations} == {"mrr", "recall_at_5", "ndcg_at_5"}


def test_check_metric_thresholds_reports_missing_method_as_violation():
    summary = {"vector": {"mrr": 0.5}}
    thresholds = {"hybrid": {"mrr": 0.60}}

    violations = check_metric_thresholds(summary, thresholds)

    assert len(violations) == 1
    assert violations[0].method == "hybrid"
    assert violations[0].metric == "mrr"
