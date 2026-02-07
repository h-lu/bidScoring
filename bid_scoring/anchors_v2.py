from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def normalize_text(text: str) -> str:
    """Normalize text for stable hashing.

    Notes:
    - Keep this conservative: we only collapse whitespace and trim.
    - Do NOT do lossy normalization (e.g., lowercasing) unless required.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding used for hashing.

    - sorted keys for dicts
    - no insignificant whitespace
    - keep unicode as-is (ensure_ascii=False) for human debugging
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_anchor_json(*, anchors: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the v0.2 anchor_json contract.

    v0.2 shape:
      {"anchors": [ {page_idx, bbox, coord_sys, page_w, page_h, path, source}, ... ]}

    This function is intentionally light-weight; deeper validation belongs in
    higher-level ingestion code (where we have more context).
    """
    normalized: list[dict[str, Any]] = []
    for a in anchors:
        # Ensure a stable set of keys to avoid hash drift due to missing fields.
        normalized.append(
            {
                "page_idx": a.get("page_idx"),
                "bbox": a.get("bbox"),
                "coord_sys": a.get("coord_sys") or "unknown",
                "page_w": a.get("page_w"),
                "page_h": a.get("page_h"),
                "path": a.get("path"),
                "source": a.get("source"),
            }
        )
    return {"anchors": normalized}


def compute_unit_hash(
    *,
    text_norm: str,
    anchor_json: dict[str, Any],
    source_element_id: str | None,
) -> str:
    """Compute a stable unit hash for v0.2 normalized content units."""
    payload = "\n".join(
        [
            text_norm or "",
            canonical_json(anchor_json),
            source_element_id or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

