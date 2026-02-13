"""Threshold gate helpers for retrieval baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdViolation:
    method: str
    metric: str
    actual: float
    minimum: float


def check_metric_thresholds(
    summary_by_method: dict[str, dict[str, float]],
    thresholds_by_method: dict[str, dict[str, float]],
) -> list[ThresholdViolation]:
    """Compare summary metrics against configured minimum thresholds."""
    violations: list[ThresholdViolation] = []

    for method, metric_thresholds in thresholds_by_method.items():
        method_summary = summary_by_method.get(method, {})
        for metric, minimum in metric_thresholds.items():
            actual = float(method_summary.get(metric, 0.0))
            if actual < float(minimum):
                violations.append(
                    ThresholdViolation(
                        method=method,
                        metric=metric,
                        actual=actual,
                        minimum=float(minimum),
                    )
                )

    return violations
