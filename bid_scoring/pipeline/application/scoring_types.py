from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .question_context import QuestionContext


@dataclass(frozen=True)
class ScoringRequest:
    """Standard scoring input shared by all scoring backends."""

    version_id: str
    bidder_name: str
    project_name: str
    dimensions: list[str] | None = None
    question_context: QuestionContext | None = None


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
    evidence_citations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
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
            "evidence_citations": {
                key: list(value) for key, value in self.evidence_citations.items()
            },
            "dimensions": dict(self.dimensions),
            "warnings": list(self.warnings),
        }


class ScoringProvider(Protocol):
    """Abstraction for pluggable scoring engines."""

    def score(self, request: ScoringRequest) -> ScoringResult: ...


class AgentMcpExecutor(Protocol):
    """Execution contract for agent-driven MCP scoring."""

    def score(self, request: ScoringRequest) -> ScoringResult: ...
