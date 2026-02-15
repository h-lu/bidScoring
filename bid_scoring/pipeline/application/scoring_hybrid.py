from __future__ import annotations

from typing import Any

from .scoring_common import (
    merge_evidence_citations,
    merge_status,
    merge_unique_warnings,
    weighted_score,
    worst_risk_level,
)
from .scoring_types import ScoringProvider, ScoringRequest, ScoringResult


class HybridScoringProvider:
    """Blend two scoring providers with explicit primary weighting."""

    def __init__(
        self,
        primary: ScoringProvider,
        secondary: ScoringProvider,
        primary_weight: float = 0.7,
    ):
        if primary_weight < 0 or primary_weight > 1:
            raise ValueError("primary_weight must be within [0, 1]")
        self._primary = primary
        self._secondary = secondary
        self._primary_weight = float(primary_weight)

    def score(self, request: ScoringRequest) -> ScoringResult:
        primary_result = self._primary.score(request)
        secondary_result = self._secondary.score(request)

        overall_score = weighted_score(
            primary_result.overall_score,
            secondary_result.overall_score,
            self._primary_weight,
        )

        return ScoringResult(
            status=merge_status(primary_result.status, secondary_result.status),
            overall_score=overall_score,
            risk_level=worst_risk_level(
                primary_result.risk_level, secondary_result.risk_level
            ),
            total_risks=max(primary_result.total_risks, secondary_result.total_risks),
            total_benefits=max(
                primary_result.total_benefits, secondary_result.total_benefits
            ),
            chunks_analyzed=max(
                primary_result.chunks_analyzed, secondary_result.chunks_analyzed
            ),
            recommendations=merge_unique_warnings(
                primary_result.recommendations,
                secondary_result.recommendations,
            ),
            evidence_warnings=merge_unique_warnings(
                primary_result.evidence_warnings,
                secondary_result.evidence_warnings,
            ),
            evidence_citations=merge_evidence_citations(
                primary_result.evidence_citations,
                secondary_result.evidence_citations,
            ),
            dimensions=_merge_dimensions(
                primary_result.dimensions,
                secondary_result.dimensions,
                self._primary_weight,
            ),
            warnings=merge_unique_warnings(
                primary_result.warnings,
                secondary_result.warnings,
            ),
            backend_observability={
                "execution_mode": "hybrid",
                "primary": dict(primary_result.backend_observability),
                "secondary": dict(secondary_result.backend_observability),
            },
        )


def _merge_dimensions(
    primary: dict[str, dict[str, Any]],
    secondary: dict[str, dict[str, Any]],
    primary_weight: float,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for key in set(primary) | set(secondary):
        primary_dim = primary.get(key)
        secondary_dim = secondary.get(key)
        if primary_dim is None:
            merged[key] = dict(secondary_dim or {})
            continue
        if secondary_dim is None:
            merged[key] = dict(primary_dim)
            continue

        merged[key] = {
            "score": weighted_score(
                float(primary_dim.get("score", 0.0)),
                float(secondary_dim.get("score", 0.0)),
                primary_weight,
            ),
            "risk_level": worst_risk_level(
                str(primary_dim.get("risk_level", "medium")),
                str(secondary_dim.get("risk_level", "medium")),
            ),
            "chunks_found": max(
                int(primary_dim.get("chunks_found", 0)),
                int(secondary_dim.get("chunks_found", 0)),
            ),
            "summary": str(primary_dim.get("summary", ""))
            or str(secondary_dim.get("summary", "")),
            "evidence_warnings": merge_unique_warnings(
                list(primary_dim.get("evidence_warnings", [])),
                list(secondary_dim.get("evidence_warnings", [])),
            ),
            "evidence_citations": merge_evidence_citations(
                {key: list(primary_dim.get("evidence_citations", []))},
                {key: list(secondary_dim.get("evidence_citations", []))},
            ).get(key, []),
        }
    return merged
