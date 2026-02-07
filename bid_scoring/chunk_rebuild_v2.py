from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def _as_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def rebuild_chunks_from_units(
    conn,
    *,
    project_id: str,
    version_id: str,
    group_size: int = 2,
    source_prefix: str = "rechunk",
) -> dict[str, int]:
    """Rebuild the index layer (chunks + chunk_unit_spans) from stable content_units.

    Notes:
    - This intentionally does NOT mutate `content_units`.
    - Existing citations referencing deleted chunks should survive, because
      citations.chunk_id is provenance only (ON DELETE SET NULL).
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    stats = {"chunks_deleted": 0, "chunks_inserted": 0, "spans_inserted": 0}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT unit_id::text, unit_index, text_raw, anchor_json
            FROM content_units
            WHERE version_id = %s
            ORDER BY unit_index
            """,
            (version_id,),
        )
        units = cur.fetchall()

        cur.execute("DELETE FROM chunks WHERE version_id = %s", (version_id,))
        stats["chunks_deleted"] = int(cur.rowcount or 0)

        # Group units and write new chunks.
        chunk_index = 0
        for i in range(0, len(units), group_size):
            group = units[i : i + group_size]
            if not group:
                continue

            chunk_id = str(uuid.uuid4())
            source_id = f"{source_prefix}_{group_size}_{chunk_index:04d}"
            chunk_index += 1

            texts: list[str] = []
            bboxes: list[Any] = []
            page_idx = 0
            for _, _, text_raw, anchor_json in group:
                texts.append(text_raw or "")
                anchor_obj = _as_obj(anchor_json) or {}
                anchors = anchor_obj.get("anchors") or []
                if anchors and isinstance(anchors, list):
                    a0 = anchors[0] if anchors else None
                    if a0 and isinstance(a0, dict):
                        if "page_idx" in a0 and page_idx == 0:
                            try:
                                page_idx = int(a0.get("page_idx") or 0)
                            except Exception:
                                page_idx = 0
                        if "bbox" in a0 and a0.get("bbox") is not None:
                            bboxes.append(a0.get("bbox"))

            text_raw = "\n".join(t for t in texts if t)
            text_hash = hashlib.sha256((text_raw or "").encode("utf-8")).hexdigest()
            bbox_json = json.dumps(bboxes, ensure_ascii=False)

            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    project_id,
                    version_id,
                    source_id,
                    chunk_index,
                    page_idx,
                    bbox,
                    element_type,
                    text_raw,
                    text_hash,
                    text_tsv
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    to_tsvector('simple', %s)
                )
                """,
                (
                    chunk_id,
                    project_id,
                    version_id,
                    source_id,
                    chunk_index - 1,
                    page_idx,
                    bbox_json,
                    "rechunk",
                    text_raw,
                    text_hash,
                    text_raw,
                ),
            )
            stats["chunks_inserted"] += 1

            for unit_order, (
                unit_id,
                _unit_index,
                unit_text,
                _anchor_json,
            ) in enumerate(group):
                cur.execute(
                    """
                    INSERT INTO chunk_unit_spans (
                        chunk_id, unit_id, unit_order, start_char, end_char
                    ) VALUES (%s, %s, %s, %s, %s)
                    """,
                    (chunk_id, unit_id, unit_order, 0, len(unit_text or "")),
                )
                stats["spans_inserted"] += 1

    conn.commit()
    return stats
