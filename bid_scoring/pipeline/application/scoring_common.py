from __future__ import annotations

from typing import Any


def merge_unique_warnings(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(merged)
    for warning in additions:
        if warning in seen:
            continue
        seen.add(warning)
        merged.append(warning)
    return merged


def weighted_score(primary: float, secondary: float, primary_weight: float) -> float:
    return round(primary * primary_weight + secondary * (1.0 - primary_weight), 2)


def merge_status(primary: str, secondary: str) -> str:
    if primary == "completed" and secondary == "completed":
        return "completed"
    if primary == "completed" or secondary == "completed":
        return "partial"
    return primary or secondary or "unknown"


def worst_risk_level(primary: str, secondary: str) -> str:
    rank = {"low": 0, "medium": 1, "high": 2}
    primary_rank = rank.get(primary, 1)
    secondary_rank = rank.get(secondary, 1)
    return primary if primary_rank >= secondary_rank else secondary


def merge_evidence_citations(
    primary: dict[str, list[dict[str, Any]]],
    secondary: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    for key in set(primary) | set(secondary):
        seen: set[tuple[Any, Any, Any]] = set()
        rows: list[dict[str, Any]] = []
        for candidate in [*primary.get(key, []), *secondary.get(key, [])]:
            if not isinstance(candidate, dict):
                continue
            token = (
                candidate.get("chunk_id"),
                candidate.get("page_idx"),
                tuple(candidate.get("bbox", []))
                if isinstance(candidate.get("bbox"), list)
                else candidate.get("bbox"),
            )
            if token in seen:
                continue
            seen.add(token)
            rows.append(
                {
                    "chunk_id": candidate.get("chunk_id"),
                    "page_idx": candidate.get("page_idx"),
                    "bbox": candidate.get("bbox"),
                }
            )
        merged[key] = rows
    return merged
