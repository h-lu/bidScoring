from __future__ import annotations

from typing import Any, Mapping

from .scoring_common import merge_unique_warnings


def build_traceability_summary(scoring_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize evidence-citation traceability for scoring outputs."""
    evidence_warnings = _coerce_str_list(scoring_payload.get("evidence_warnings"))
    evidence_citations_raw = scoring_payload.get("evidence_citations")
    evidence_citations = (
        evidence_citations_raw if isinstance(evidence_citations_raw, dict) else {}
    )

    warnings = list(evidence_warnings)
    highlight_ready_chunk_ids: set[str] = set()
    dimension_count = len(evidence_citations)
    dimension_with_traceable = 0
    citation_count_total = 0
    citation_count_traceable = 0

    for citations_raw in evidence_citations.values():
        citations = citations_raw if isinstance(citations_raw, list) else []
        has_traceable = False
        for item in citations:
            citation_count_total += 1
            if not isinstance(item, dict):
                warnings = merge_unique_warnings(warnings, ["citation_payload_invalid"])
                continue
            chunk_id = item.get("chunk_id")
            page_idx = item.get("page_idx")
            bbox = item.get("bbox")

            if not isinstance(chunk_id, str) or not chunk_id:
                warnings = merge_unique_warnings(
                    warnings, ["citation_missing_chunk_id"]
                )
                continue
            if not isinstance(page_idx, int) or page_idx < 0:
                warnings = merge_unique_warnings(
                    warnings, ["citation_missing_page_idx"]
                )
                continue
            if not _is_bbox(bbox):
                warnings = merge_unique_warnings(warnings, ["citation_missing_bbox"])
                continue

            has_traceable = True
            citation_count_traceable += 1
            highlight_ready_chunk_ids.add(chunk_id)

        if has_traceable:
            dimension_with_traceable += 1

    citation_count_untraceable = citation_count_total - citation_count_traceable
    if citation_count_total == 0:
        warnings = merge_unique_warnings(warnings, ["no_evidence_citations"])
    if citation_count_untraceable > 0:
        warnings = merge_unique_warnings(warnings, ["partial_untraceable_citations"])
    if not highlight_ready_chunk_ids:
        warnings = merge_unique_warnings(warnings, ["no_highlight_ready_chunks"])

    status = _resolve_status(
        citation_count_total=citation_count_total,
        citation_count_traceable=citation_count_traceable,
    )
    coverage_ratio = (
        round(citation_count_traceable / citation_count_total, 4)
        if citation_count_total > 0
        else 0.0
    )

    return {
        "status": status,
        "dimension_count": dimension_count,
        "dimension_with_traceable_citations": dimension_with_traceable,
        "citation_count_total": citation_count_total,
        "citation_count_traceable": citation_count_traceable,
        "citation_count_untraceable": citation_count_untraceable,
        "citation_coverage_ratio": coverage_ratio,
        "highlight_ready_chunk_ids": sorted(highlight_ready_chunk_ids),
        "warnings": warnings,
    }


def _resolve_status(*, citation_count_total: int, citation_count_traceable: int) -> str:
    if citation_count_total == 0:
        return "unverifiable"
    if citation_count_traceable == citation_count_total:
        return "verified"
    if citation_count_traceable > 0:
        return "verified_with_warnings"
    return "unverifiable"


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _is_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(v, (int, float)) for v in value)
