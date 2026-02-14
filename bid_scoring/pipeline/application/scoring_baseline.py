from __future__ import annotations

from typing import Any

from .scoring_common import merge_unique_warnings
from .scoring_types import ScoringProvider, ScoringRequest, ScoringResult


class BidAnalyzerScoringProvider:
    """Adapter: normalize BidAnalyzer report to ScoringResult."""

    def __init__(self, analyzer: Any):
        self._analyzer = analyzer

    def score(self, request: ScoringRequest) -> ScoringResult:
        report = self._analyzer.analyze_version(
            version_id=request.version_id,
            bidder_name=request.bidder_name,
            project_name=request.project_name,
            dimensions=request.dimensions,
        )
        return report_to_scoring_result(report)


class WarningFallbackScoringProvider:
    """Wrap a backend and append warnings for graceful fallback behavior."""

    def __init__(self, fallback: ScoringProvider, warning_codes: list[str]):
        self._fallback = fallback
        self._warning_codes = warning_codes

    def score(self, request: ScoringRequest) -> ScoringResult:
        base = self._fallback.score(request)
        warnings = merge_unique_warnings(base.warnings, self._warning_codes)
        return ScoringResult(
            status=base.status,
            overall_score=base.overall_score,
            risk_level=base.risk_level,
            total_risks=base.total_risks,
            total_benefits=base.total_benefits,
            chunks_analyzed=base.chunks_analyzed,
            recommendations=list(base.recommendations),
            evidence_warnings=list(base.evidence_warnings),
            evidence_citations={
                key: list(value) for key, value in base.evidence_citations.items()
            },
            dimensions=dict(base.dimensions),
            warnings=warnings,
        )


def report_to_scoring_result(report: Any) -> ScoringResult:
    dimensions_payload: dict[str, dict[str, Any]] = {}
    evidence_citations: dict[str, list[dict[str, Any]]] = {}
    for name, result in getattr(report, "dimensions", {}).items():
        citations = list(getattr(result, "evidence_citations", []))
        dimensions_payload[name] = {
            "score": float(getattr(result, "score", 0.0)),
            "risk_level": getattr(result, "risk_level", "medium"),
            "chunks_found": int(getattr(result, "chunks_found", 0)),
            "summary": getattr(result, "summary", ""),
            "evidence_warnings": list(getattr(result, "evidence_warnings", [])),
            "evidence_citations": citations,
        }
        evidence_citations[name] = citations

    evidence_warnings = list(getattr(report, "evidence_warnings", []))
    return ScoringResult(
        status="completed",
        overall_score=float(getattr(report, "overall_score", 0.0)),
        risk_level=getattr(report, "risk_level", "medium"),
        total_risks=int(getattr(report, "total_risks", 0)),
        total_benefits=int(getattr(report, "total_benefits", 0)),
        chunks_analyzed=int(getattr(report, "chunks_analyzed", 0)),
        recommendations=list(getattr(report, "recommendations", [])),
        evidence_warnings=evidence_warnings,
        evidence_citations=evidence_citations,
        dimensions=dimensions_payload,
        warnings=[],
    )
