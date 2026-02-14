from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ScoringRequest:
    """Standard scoring input shared by all scoring backends."""

    version_id: str
    bidder_name: str
    project_name: str
    dimensions: list[str] | None = None


@dataclass(frozen=True)
class ScoringResult:
    """Normalized scoring result contract."""

    status: str
    overall_score: float
    risk_level: str
    total_risks: int
    total_benefits: int
    chunks_analyzed: int
    recommendations: list[str] = field(default_factory=list)
    evidence_warnings: list[str] = field(default_factory=list)
    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "total_risks": self.total_risks,
            "total_benefits": self.total_benefits,
            "chunks_analyzed": self.chunks_analyzed,
            "recommendations": list(self.recommendations),
            "evidence_warnings": list(self.evidence_warnings),
            "dimensions": dict(self.dimensions),
            "warnings": list(self.warnings),
        }


class ScoringProvider(Protocol):
    """Abstraction for pluggable scoring engines."""

    def score(self, request: ScoringRequest) -> ScoringResult: ...


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
        return _report_to_scoring_result(report)


class WarningFallbackScoringProvider:
    """Wrap a backend and append warnings for graceful fallback behavior."""

    def __init__(self, fallback: ScoringProvider, warning_codes: list[str]):
        self._fallback = fallback
        self._warning_codes = warning_codes

    def score(self, request: ScoringRequest) -> ScoringResult:
        base = self._fallback.score(request)
        warnings = _merge_unique_warnings(base.warnings, self._warning_codes)
        return ScoringResult(
            status=base.status,
            overall_score=base.overall_score,
            risk_level=base.risk_level,
            total_risks=base.total_risks,
            total_benefits=base.total_benefits,
            chunks_analyzed=base.chunks_analyzed,
            recommendations=list(base.recommendations),
            evidence_warnings=list(base.evidence_warnings),
            dimensions=dict(base.dimensions),
            warnings=warnings,
        )


class AgentMcpScoringProvider(WarningFallbackScoringProvider):
    """Agent/MCP backend adapter based on normalized BidAnalyzer output."""

    def __init__(self, fallback: ScoringProvider):
        super().__init__(fallback=fallback, warning_codes=[])


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

        overall_score = _weighted_score(
            primary_result.overall_score,
            secondary_result.overall_score,
            self._primary_weight,
        )

        return ScoringResult(
            status=_merge_status(primary_result.status, secondary_result.status),
            overall_score=overall_score,
            risk_level=_worst_risk_level(
                primary_result.risk_level, secondary_result.risk_level
            ),
            total_risks=max(primary_result.total_risks, secondary_result.total_risks),
            total_benefits=max(
                primary_result.total_benefits, secondary_result.total_benefits
            ),
            chunks_analyzed=max(
                primary_result.chunks_analyzed, secondary_result.chunks_analyzed
            ),
            recommendations=_merge_unique_warnings(
                primary_result.recommendations,
                secondary_result.recommendations,
            ),
            evidence_warnings=_merge_unique_warnings(
                primary_result.evidence_warnings,
                secondary_result.evidence_warnings,
            ),
            dimensions=_merge_dimensions(
                primary_result.dimensions,
                secondary_result.dimensions,
                self._primary_weight,
            ),
            warnings=_merge_unique_warnings(
                primary_result.warnings,
                secondary_result.warnings,
            ),
        )


def _report_to_scoring_result(report: Any) -> ScoringResult:
    dimensions_payload: dict[str, dict[str, Any]] = {}
    for name, result in getattr(report, "dimensions", {}).items():
        dimensions_payload[name] = {
            "score": float(getattr(result, "score", 0.0)),
            "risk_level": getattr(result, "risk_level", "medium"),
            "chunks_found": int(getattr(result, "chunks_found", 0)),
            "summary": getattr(result, "summary", ""),
            "evidence_warnings": list(getattr(result, "evidence_warnings", [])),
        }

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
        dimensions=dimensions_payload,
        warnings=[],
    )


def _merge_unique_warnings(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(merged)
    for warning in additions:
        if warning in seen:
            continue
        seen.add(warning)
        merged.append(warning)
    return merged


def _weighted_score(primary: float, secondary: float, primary_weight: float) -> float:
    return round(primary * primary_weight + secondary * (1.0 - primary_weight), 2)


def _merge_status(primary: str, secondary: str) -> str:
    if primary == "completed" and secondary == "completed":
        return "completed"
    if primary == "completed" or secondary == "completed":
        return "partial"
    return primary or secondary or "unknown"


def _worst_risk_level(primary: str, secondary: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    primary_rank = rank.get(primary, 1)
    secondary_rank = rank.get(secondary, 1)
    return primary if primary_rank >= secondary_rank else secondary


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
            "score": _weighted_score(
                float(primary_dim.get("score", 0.0)),
                float(secondary_dim.get("score", 0.0)),
                primary_weight,
            ),
            "risk_level": _worst_risk_level(
                str(primary_dim.get("risk_level", "medium")),
                str(secondary_dim.get("risk_level", "medium")),
            ),
            "chunks_found": max(
                int(primary_dim.get("chunks_found", 0)),
                int(secondary_dim.get("chunks_found", 0)),
            ),
            "summary": str(primary_dim.get("summary", ""))
            or str(secondary_dim.get("summary", "")),
            "evidence_warnings": _merge_unique_warnings(
                list(primary_dim.get("evidence_warnings", [])),
                list(secondary_dim.get("evidence_warnings", [])),
            ),
        }
    return merged
