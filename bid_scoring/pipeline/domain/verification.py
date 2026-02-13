from __future__ import annotations

from typing import Any, Mapping

from .models import CitationAssessment, EvidenceWarning


class CitationVerifier:
    """Converts low-level verification output into warning-first assessment."""

    def assess(self, verification: Mapping[str, Any]) -> CitationAssessment:
        warnings: list[EvidenceWarning] = []

        if verification.get("reason") == "not_found":
            warnings.append(
                EvidenceWarning(
                    code="citation_not_found",
                    message="Citation record not found",
                )
            )
        else:
            if verification.get("hash_ok") is False:
                warnings.append(
                    EvidenceWarning(
                        code="hash_mismatch",
                        message="Evidence hash does not match source unit",
                    )
                )
            if verification.get("anchor_ok") is False:
                warnings.append(
                    EvidenceWarning(
                        code="anchor_mismatch",
                        message="Citation anchor does not match source unit",
                    )
                )

        if warnings:
            return CitationAssessment(
                status="warning",
                evidence_status="unverifiable",
                citation_id=_as_opt_str(verification.get("citation_id")),
                unit_id=_as_opt_str(verification.get("unit_id")),
                warnings=warnings,
            )

        return CitationAssessment(
            status="verified",
            evidence_status="verified",
            citation_id=_as_opt_str(verification.get("citation_id")),
            unit_id=_as_opt_str(verification.get("unit_id")),
            warnings=[],
        )


def _as_opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
