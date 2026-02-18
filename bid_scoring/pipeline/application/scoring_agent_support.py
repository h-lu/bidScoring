from __future__ import annotations

from typing import Any

from .scoring_common import merge_unique_warnings
from .scoring_types import ScoringResult


def resolve_dimensions(
    selected: list[str] | None,
    *,
    keyword_overrides: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    from mcp_servers.bid_analysis.models import ANALYSIS_DIMENSIONS, AnalysisDimension

    if selected is None:
        selected_names = list(ANALYSIS_DIMENSIONS.keys())
    else:
        selected_names = list(selected)

    resolved: dict[str, Any] = {}
    for name in selected_names:
        dim = ANALYSIS_DIMENSIONS.get(name)
        if dim is None:
            continue
        if keyword_overrides and name in keyword_overrides:
            keywords = sanitize_keywords(keyword_overrides.get(name, []))
            if keywords:
                dim = AnalysisDimension(
                    name=dim.name,
                    display_name=dim.display_name,
                    weight=dim.weight,
                    keywords=keywords,
                    extract_patterns=dim.extract_patterns,
                    risk_thresholds=dim.risk_thresholds,
                )
        resolved[name] = dim
    return resolved


def is_verifiable_item(item: dict[str, Any]) -> bool:
    is_valid, _warnings = evaluate_evidence_item(
        item,
        require_page_idx=False,
        require_bbox=True,
        require_quote=False,
    )
    return is_valid


def evaluate_evidence_item(
    item: dict[str, Any],
    *,
    require_page_idx: bool,
    require_bbox: bool,
    require_quote: bool,
) -> tuple[bool, list[str]]:
    warnings: list[str] = []

    if item.get("evidence_status") not in {"verified", "verified_with_warnings"}:
        warnings.append("unverifiable_evidence_for_scoring")

    if require_bbox:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            warnings.append("missing_bbox")

    if require_page_idx and not isinstance(item.get("page_idx"), int):
        warnings.append("missing_page_idx")

    if require_quote:
        quote = str(item.get("text", "")).strip()
        if not quote:
            warnings.append("missing_quote_text")

    return (len(warnings) == 0), warnings


def extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text"):
                text_val = getattr(item, "text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "".join(parts)
    return ""


def normalize_agent_result(
    *,
    agent_json: dict[str, Any],
    dimensions: dict[str, Any],
    evidence_payload: dict[str, list[dict[str, Any]]],
    dimension_warning_map: dict[str, list[str]],
    evidence_warnings: list[str],
    backend_observability: dict[str, Any] | None = None,
) -> ScoringResult:
    raw_dimensions = agent_json.get("dimensions")
    raw_dimensions = raw_dimensions if isinstance(raw_dimensions, dict) else {}
    normalized_dimensions: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    evidence_citations: dict[str, list[dict[str, Any]]] = {}

    for dim_key in dimensions:
        dim_payload = raw_dimensions.get(dim_key)
        if not isinstance(dim_payload, dict):
            warnings = merge_unique_warnings(
                warnings, [f"agent_mcp_dimension_missing:{dim_key}"]
            )
            dim_payload = {}
        citations = [
            {
                "chunk_id": item.get("chunk_id"),
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
            }
            for item in evidence_payload.get(dim_key, [])
        ]
        evidence_citations[dim_key] = citations
        if not citations:
            warnings = merge_unique_warnings(
                warnings,
                [f"agent_mcp_dimension_no_verifiable_evidence:{dim_key}"],
            )
            normalized_dimensions[dim_key] = {
                "score": 50.0,
                "risk_level": "medium",
                "chunks_found": 0,
                "summary": "证据不足，按中性评分处理",
                "evidence_warnings": merge_unique_warnings(
                    list(dimension_warning_map.get(dim_key, [])),
                    ["agent_mcp_dimension_no_verifiable_evidence"],
                ),
                "evidence_citations": citations,
            }
            continue
        normalized_dimensions[dim_key] = {
            "score": safe_float(dim_payload.get("score"), default=50.0),
            "risk_level": safe_risk_level(dim_payload.get("risk_level")),
            "chunks_found": len(evidence_payload.get(dim_key, [])),
            "summary": str(dim_payload.get("summary", "")),
            "evidence_warnings": list(dimension_warning_map.get(dim_key, [])),
            "evidence_citations": citations,
        }

    return ScoringResult(
        status="completed",
        overall_score=safe_float(agent_json.get("overall_score"), default=50.0),
        risk_level=safe_risk_level(agent_json.get("risk_level")),
        total_risks=safe_int(agent_json.get("total_risks"), default=0),
        total_benefits=safe_int(agent_json.get("total_benefits"), default=0),
        chunks_analyzed=sum(len(items) for items in evidence_payload.values()),
        recommendations=safe_str_list(agent_json.get("recommendations")),
        evidence_warnings=evidence_warnings,
        evidence_citations=evidence_citations,
        dimensions=normalized_dimensions,
        warnings=warnings,
        backend_observability=dict(backend_observability or {}),
    )


def safe_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(0.0, min(100.0, parsed))


def safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_risk_level(value: Any) -> str:
    if isinstance(value, str) and value in {"low", "medium", "high"}:
        return value
    return "medium"


def safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            output.append(item)
    return output


def read_int_env(name: str, default: int) -> int:
    import os

    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed


def sanitize_keywords(values: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(normalized)
    return deduplicated
