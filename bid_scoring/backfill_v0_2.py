from __future__ import annotations

import json
from typing import Any

from bid_scoring.anchors_v2 import build_anchor_json, compute_unit_hash, normalize_text


def _as_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _iter_bbox_rects(bbox: Any) -> list[tuple[float, float, float, float]]:
    if bbox is None:
        return []
    bbox = _as_obj(bbox)
    if not isinstance(bbox, list) or not bbox:
        return []

    def _is_num(x: Any) -> bool:
        return isinstance(x, (int, float))

    if len(bbox) == 4 and all(_is_num(x) for x in bbox):
        return [(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))]

    rects: list[tuple[float, float, float, float]] = []
    for item in bbox:
        if isinstance(item, list) and len(item) == 4 and all(_is_num(x) for x in item):
            rects.append(
                (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
            )
    return rects


def backfill_units_from_chunks(
    conn, *, version_id: str | None = None
) -> dict[str, int]:
    """Backfill v0.2 normalized tables from existing v0.1 `chunks`.

    Strategy (compat-first):
    - 1 chunk -> 1 content_unit
    - unit_index := chunk_index
    - source_element_id := chunks.source_id
    - chunk_unit_spans: unit_order=0, full-span chars

    This makes unit-level citations possible without re-parsing the original files.
    """
    stats = {
        "versions_processed": 0,
        "pages_upserted": 0,
        "units_upserted": 0,
        "spans_upserted": 0,
    }

    with conn.cursor() as cur:
        if version_id:
            versions = [version_id]
        else:
            cur.execute(
                "SELECT DISTINCT version_id::text FROM chunks ORDER BY version_id"
            )
            versions = [r[0] for r in cur.fetchall()]

        for ver in versions:
            stats["versions_processed"] += 1

            # document_pages: infer page dims from observed chunk bboxes as a fallback.
            cur.execute(
                """
                SELECT page_idx, bbox
                FROM chunks
                WHERE version_id = %s AND page_idx IS NOT NULL
                """,
                (ver,),
            )
            page_dim_by_idx: dict[int, tuple[float | None, float | None]] = {}
            for page_idx, bbox in cur.fetchall():
                pidx = int(page_idx)
                for _, _, x2, y2 in _iter_bbox_rects(bbox):
                    cur_w, cur_h = page_dim_by_idx.get(pidx, (None, None))
                    page_dim_by_idx[pidx] = (
                        max(cur_w or 0.0, float(x2)),
                        max(cur_h or 0.0, float(y2)),
                    )

            for pidx in sorted(page_dim_by_idx.keys()):
                page_w, page_h = page_dim_by_idx.get(pidx, (None, None))
                cur.execute(
                    """
                    INSERT INTO document_pages (version_id, page_idx, page_w, page_h, coord_sys)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (version_id, page_idx) DO UPDATE SET
                        page_w = COALESCE(EXCLUDED.page_w, document_pages.page_w),
                        page_h = COALESCE(EXCLUDED.page_h, document_pages.page_h),
                        coord_sys = EXCLUDED.coord_sys
                    """,
                    (ver, int(pidx), page_w, page_h, "mineru_bbox_v1"),
                )
                stats["pages_upserted"] += int(cur.rowcount or 0)

            # Backfill units/spans from chunks.
            cur.execute(
                """
                SELECT
                    chunk_id::text,
                    chunk_index,
                    source_id,
                    element_type,
                    text_raw,
                    page_idx,
                    bbox
                FROM chunks
                WHERE version_id = %s
                ORDER BY chunk_index
                """,
                (ver,),
            )

            for (
                chunk_id,
                chunk_index,
                source_id,
                element_type,
                text_raw,
                page_idx,
                bbox,
            ) in cur.fetchall():
                source_element_id = source_id or f"chunk_{int(chunk_index):04d}"
                bbox_obj = _as_obj(bbox) or []
                page_idx_int = int(page_idx or 0)

                anchor_json = build_anchor_json(
                    anchors=[
                        {
                            "page_idx": page_idx_int,
                            "bbox": bbox_obj,
                            "coord_sys": "mineru_bbox_v1",
                            "page_w": None,
                            "page_h": None,
                            "path": None,
                            "source": {"element_id": source_element_id},
                        }
                    ]
                )

                text_raw_str = text_raw or ""
                text_norm = normalize_text(text_raw_str)
                unit_hash = compute_unit_hash(
                    text_norm=text_norm,
                    anchor_json=anchor_json,
                    source_element_id=source_element_id,
                )

                cur.execute(
                    """
                    INSERT INTO content_units (
                        version_id,
                        unit_index,
                        unit_type,
                        text_raw,
                        text_norm,
                        char_count,
                        anchor_json,
                        source_element_id,
                        unit_hash
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (version_id, unit_index) DO UPDATE SET
                        unit_type = EXCLUDED.unit_type,
                        text_raw = EXCLUDED.text_raw,
                        text_norm = EXCLUDED.text_norm,
                        char_count = EXCLUDED.char_count,
                        anchor_json = EXCLUDED.anchor_json,
                        source_element_id = EXCLUDED.source_element_id,
                        unit_hash = EXCLUDED.unit_hash
                    RETURNING unit_id
                    """,
                    (
                        ver,
                        int(chunk_index),
                        str(element_type or "unknown"),
                        text_raw_str,
                        text_norm,
                        len(text_raw_str),
                        json.dumps(anchor_json, ensure_ascii=False),
                        source_element_id,
                        unit_hash,
                    ),
                )
                unit_id = str(cur.fetchone()[0])
                stats["units_upserted"] += 1

                cur.execute(
                    """
                    INSERT INTO chunk_unit_spans (
                        chunk_id, unit_id, unit_order, start_char, end_char
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id, unit_order) DO UPDATE SET
                        unit_id = EXCLUDED.unit_id,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char
                    """,
                    (
                        chunk_id,
                        unit_id,
                        0,
                        0,
                        len(text_raw_str),
                    ),
                )
                stats["spans_upserted"] += 1

    conn.commit()
    return stats
