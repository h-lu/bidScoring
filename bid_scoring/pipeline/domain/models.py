from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceWarning:
    """Warning for unverifiable evidence."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CitationAssessment:
    """Assessment result for a single citation verification."""

    status: str
    evidence_status: str
    citation_id: str | None = None
    unit_id: str | None = None
    warnings: list[EvidenceWarning] = field(default_factory=list)

