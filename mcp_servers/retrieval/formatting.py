"""Result formatting for retrieval MCP responses."""

from __future__ import annotations

from typing import Any, Dict, List

from bid_scoring.retrieval import RetrievalResult


def format_result(
    result: RetrievalResult,
    *,
    include_text: bool,
    max_chars: int | None,
) -> Dict[str, Any]:
    """Format retrieval result for MCP output contract."""
    text = result.text if include_text else ""
    if include_text and max_chars is not None and max_chars >= 0:
        text = text[:max_chars]

    item = {
        "chunk_id": result.chunk_id,
        "page_idx": result.page_idx,
        "source": result.source,
        "score": round(result.score, 6) if result.score is not None else None,
        "vector_score": round(result.vector_score, 6)
        if result.vector_score is not None
        else None,
        "keyword_score": round(result.keyword_score, 6)
        if result.keyword_score is not None
        else None,
        "rerank_score": round(result.rerank_score, 6)
        if result.rerank_score is not None
        else None,
        "text": text,
        "element_type": result.element_type,
        "bbox": result.bbox,
        "coord_system": result.coord_system,
        "evidence_units": format_evidence_units(result.evidence_units),
    }

    if item["evidence_units"]:
        warnings = _collect_evidence_warnings(item["evidence_units"])
        item["evidence_status"] = "verified_with_warnings" if warnings else "verified"
        item["warnings"] = warnings
    else:
        item["evidence_status"] = "unverifiable"
        item["warnings"] = ["missing_evidence_chain"]

    return item


def format_evidence_units(evidence_units: List[Any]) -> List[Dict[str, Any]]:
    """Normalize evidence units for MCP JSON response."""
    items: List[Dict[str, Any]] = []
    for evidence in evidence_units:
        items.append(
            {
                "unit_id": getattr(evidence, "unit_id", None),
                "unit_index": getattr(evidence, "unit_index", None),
                "unit_type": getattr(evidence, "unit_type", None),
                "text": getattr(evidence, "text", None),
                "anchor": getattr(evidence, "anchor_json", None),
                "start_char": getattr(evidence, "start_char", None),
                "end_char": getattr(evidence, "end_char", None),
            }
        )
    return items


def _collect_evidence_warnings(evidence_units: List[Dict[str, Any]]) -> List[str]:
    warnings: set[str] = set()

    for unit in evidence_units:
        anchor = unit.get("anchor")
        if not isinstance(anchor, dict):
            warnings.add("missing_anchor")
            continue

        anchors = anchor.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            warnings.add("missing_anchor")
            continue

        for anchor_item in anchors:
            if not isinstance(anchor_item, dict):
                warnings.add("invalid_anchor_item")
                continue
            if anchor_item.get("page_idx") is None:
                warnings.add("missing_anchor_page_idx")

            bbox = anchor_item.get("bbox")
            if bbox is None:
                warnings.add("missing_anchor_bbox")
            elif not isinstance(bbox, list) or len(bbox) != 4:
                warnings.add("invalid_anchor_bbox")

    return sorted(warnings)
