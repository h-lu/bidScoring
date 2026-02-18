"""Helpers for retrieval evaluation thresholds driven by policy bundles."""

from __future__ import annotations

from bid_scoring.policy.models import PolicyBundle

from .evaluation_gate import ThresholdViolation, check_metric_thresholds


def extract_retrieval_thresholds(bundle: PolicyBundle) -> dict[str, dict[str, float]]:
    """Return retrieval threshold map from policy bundle."""
    return {
        method: dict(metrics)
        for method, metrics in bundle.retrieval.evaluation_thresholds.items()
    }


def evaluate_retrieval_summary(
    summary_by_method: dict[str, dict[str, float]],
    thresholds_by_method: dict[str, dict[str, float]],
) -> list[ThresholdViolation]:
    """Evaluate summary metrics against policy thresholds."""
    return check_metric_thresholds(summary_by_method, thresholds_by_method)
