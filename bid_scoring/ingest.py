"""MineRU content_list 数据入库模块

支持所有 content_list 类型 (除了 page_number):
- text: 文本/标题
- image: 图片 (含 img_path, image_caption, image_footnote)
- table: 表格 (含 img_path, table_body, table_caption, table_footnote)
- list: 列表 (含 list_items, sub_type)
- header: 页眉
- aside_text: 边注
"""

import hashlib
import json
import logging
import re
import uuid
from typing import Any

from psycopg.types.json import Jsonb

from bid_scoring.anchors_v2 import build_anchor_json, compute_unit_hash, normalize_text

logger = logging.getLogger(__name__)


def _hash_text(text: str) -> str:
    """计算文本的 SHA256 哈希"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_html_tags(html: str) -> str:
    """去除 HTML 标签，保留纯文本"""
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_list_field(value: Any) -> list[str]:
    """标准化为字符串列表（兼容 str / list / None）"""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if v]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _iter_bbox_rects(bbox: Any) -> list[tuple[float, float, float, float]]:
    """Return bbox rectangles in (x1, y1, x2, y2) form.

    MineRU bbox may appear as:
    - [x1, y1, x2, y2]
    - [[x1, y1, x2, y2], ...]
    - JSON-encoded string of either form
    """
    if bbox is None:
        return []
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except Exception:
            return []
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


def _extract_text_from_item(item: dict) -> str:
    """从任意类型的 item 中提取可索引文本"""
    item_type = item.get("type", "")

    if item_type == "text":
        return item.get("text", "")

    elif item_type == "table":
        # 表格：使用 caption + footnote + body 的纯文本
        texts = []
        if item.get("table_caption"):
            texts.extend(_normalize_list_field(item["table_caption"]))
        if item.get("table_body"):
            body_text = _strip_html_tags(item["table_body"])
            if body_text:
                texts.append(body_text)
        if item.get("table_footnote"):
            texts.extend(_normalize_list_field(item["table_footnote"]))
        return " ".join(texts)

    elif item_type == "image":
        # 图片：使用 caption + footnote
        texts = []
        if item.get("image_caption"):
            texts.extend(_normalize_list_field(item["image_caption"]))
        if item.get("image_footnote"):
            texts.extend(_normalize_list_field(item["image_footnote"]))
        return " ".join(texts)

    elif item_type == "list":
        # 列表：使用所有 list_items
        list_items = _normalize_list_field(item.get("list_items"))
        return " ".join(list_items) if list_items else ""

    elif item_type in ["header", "aside_text", "page_number", "footer"]:
        return item.get("text", "")

    else:
        # 其他类型，尝试获取 text 字段
        return item.get("text", "")


def _prepare_chunk_data(item: dict, chunk_index: int) -> dict[str, Any] | None:
    """准备单条 chunk 数据，返回字典或 None（跳过）"""
    item_type = item.get("type", "")

    # 跳过 page_number (按用户要求)
    if item_type == "page_number":
        return None

    # ★ FIX: 跳过 header 和 footer，避免污染正文
    if item_type in ["header", "footer"]:
        return None

    # 提取可索引文本
    text_content = _extract_text_from_item(item)
    text_content = text_content.strip()

    # ★ FIX: 处理空标题情况
    # 如果 text_level=1 但文本为空，降级为普通文本（避免创建无效章节）
    text_level = item.get("text_level")
    if text_level == 1 and not text_content:
        logger.warning(
            f"[Ingest] Empty heading at index {chunk_index}, demoting to body text"
        )
        text_level = None

    # 生成 tsvector（如果有文本内容）
    text_tsv = text_content if text_content else None

    # 基础字段
    data = {
        "source_id": f"chunk_{chunk_index:04d}",
        "chunk_index": chunk_index,
        "page_idx": item.get("page_idx", 0),
        "bbox": json.dumps(item.get("bbox", [])),
        "element_type": item_type,
        "text_raw": text_content,
        "text_hash": _hash_text(text_content) if text_content else "",
        "text_tsv": text_tsv,
        # MineRU 特有字段
        "img_path": item.get("img_path"),
        "image_caption": _normalize_list_field(item.get("image_caption")),
        "image_footnote": _normalize_list_field(item.get("image_footnote")),
        "table_body": item.get("table_body"),
        "table_caption": _normalize_list_field(item.get("table_caption")),
        "table_footnote": _normalize_list_field(item.get("table_footnote")),
        "list_items": _normalize_list_field(item.get("list_items")),
        "sub_type": item.get("sub_type"),
        "text_level": text_level,
    }

    return data


def ingest_content_list(
    conn,
    project_id: str,
    document_id: str,
    version_id: str,
    content_list: list[dict],
    document_title: str = "untitled",
    source_type: str = "mineru",
    source_uri: str | None = None,
    parser_version: str | None = None,
    status: str = "ready",
) -> dict:
    """将 MineRU content_list 数据入库

    Args:
        conn: 数据库连接
        project_id: 项目 UUID
        document_id: 文档 UUID
        version_id: 版本 UUID
        content_list: MineRU 解析结果列表
        document_title: 文档标题
        source_type: 来源类型 (默认 mineru)
        source_uri: 源文件路径
        parser_version: 解析器版本
        status: 版本状态

    Returns:
        统计信息字典
    """
    # 准备所有 chunk 数据
    chunks_data = []
    units_data: list[dict[str, Any]] = []
    type_counts = {}

    # v0.2: document_pages fallback.
    # Infer per-page dimensions from observed bbox x2/y2 maxima (even for skipped items),
    # so anchors can carry page_w/page_h for stable UI rendering.
    page_dim_by_idx: dict[int, tuple[float | None, float | None]] = {}
    for item in content_list:
        page_idx = int(item.get("page_idx", 0) or 0)
        for _, _, x2, y2 in _iter_bbox_rects(item.get("bbox")):
            cur_w, cur_h = page_dim_by_idx.get(page_idx, (None, None))
            page_dim_by_idx[page_idx] = (
                max(cur_w or 0.0, float(x2)),
                max(cur_h or 0.0, float(y2)),
            )

    for i, item in enumerate(content_list):
        chunk_data = _prepare_chunk_data(item, i)
        if chunk_data is None:
            type_counts[item.get("type", "unknown")] = (
                type_counts.get(item.get("type", "unknown"), 0) + 1
            )
            continue

        chunks_data.append(chunk_data)

        # v0.2 normalized unit (stable evidence) derived from the same MineRU item.
        source_element_id = item.get("source_element_id") or chunk_data["source_id"]
        page_idx = int(item.get("page_idx", 0) or 0)
        page_w, page_h = page_dim_by_idx.get(page_idx, (None, None))

        anchors: list[dict[str, Any]] = []
        for x1, y1, x2, y2 in _iter_bbox_rects(item.get("bbox")):
            anchors.append(
                {
                    "page_idx": page_idx,
                    "bbox": [x1, y1, x2, y2],
                    "coord_sys": "mineru_bbox_v1",
                    "page_w": page_w,
                    "page_h": page_h,
                    "path": None,
                    "source": {"element_id": source_element_id},
                }
            )

        anchor_json = build_anchor_json(anchors=anchors)
        text_raw = chunk_data.get("text_raw") or ""
        text_norm = normalize_text(text_raw)
        unit_hash = compute_unit_hash(
            text_norm=text_norm,
            anchor_json=anchor_json,
            source_element_id=source_element_id,
        )

        units_data.append(
            {
                "unit_index": int(chunk_data["chunk_index"]),
                "unit_type": str(chunk_data["element_type"]),
                "text_raw": text_raw,
                "text_norm": text_norm,
                "char_count": len(text_raw),
                "anchor_json": anchor_json,
                "source_element_id": source_element_id,
                "unit_hash": unit_hash,
            }
        )

        item_type = item.get("type", "unknown")
        type_counts[item_type] = type_counts.get(item_type, 0) + 1

    # 数据库操作
    with conn.cursor() as cur:
        # 插入项目
        cur.execute(
            "INSERT INTO projects (project_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (project_id, f"project-{project_id[:8]}"),
        )

        # 插入文档
        cur.execute(
            "INSERT INTO documents (doc_id, project_id, title, source_type) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
            (document_id, project_id, document_title, source_type),
        )

        # 插入版本
        cur.execute(
            """
            INSERT INTO document_versions (version_id, doc_id, source_uri, source_hash, parser_version, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (version_id, document_id, source_uri, None, parser_version, status),
        )

        # v0.2: document_pages (fallback inferred from bboxes).
        for pidx in sorted(page_dim_by_idx.keys()):
            page_w, page_h = page_dim_by_idx.get(int(pidx), (None, None))
            cur.execute(
                """
                INSERT INTO document_pages (version_id, page_idx, page_w, page_h, coord_sys)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (version_id, page_idx) DO UPDATE SET
                    page_w = COALESCE(EXCLUDED.page_w, document_pages.page_w),
                    page_h = COALESCE(EXCLUDED.page_h, document_pages.page_h),
                    coord_sys = EXCLUDED.coord_sys
                """,
                (version_id, pidx, page_w, page_h, "mineru_bbox_v1"),
            )

        # v0.2: upsert normalized content units (stable evidence layer).
        # Keep unit_id stable across re-ingestion by upserting on (version_id, unit_index).
        for unit in units_data:
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
                    version_id,
                    unit["unit_index"],
                    unit["unit_type"],
                    unit["text_raw"],
                    unit["text_norm"],
                    unit["char_count"],
                    Jsonb(unit["anchor_json"]),
                    unit["source_element_id"],
                    unit["unit_hash"],
                ),
            )
            unit["unit_id"] = str(cur.fetchone()[0])

        # v0.2: chunks are a rebuildable index layer. Upsert by (version_id, source_id)
        # to keep chunk_id stable across re-ingestion (preserves citation provenance).
        cur.execute(
            """
            DELETE FROM chunk_unit_spans
            WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE version_id = %s)
            """,
            (version_id,),
        )

        seen_source_ids: set[str] = set()

        for data, unit in zip(chunks_data, units_data, strict=False):
            seen_source_ids.add(str(data["source_id"]))
            new_chunk_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, project_id, version_id, source_id, chunk_index, page_idx,
                    bbox, element_type, text_raw, text_hash, text_tsv,
                    img_path, image_caption, image_footnote,
                    table_body, table_caption, table_footnote,
                    list_items, sub_type, text_level
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    to_tsvector('simple', %s),
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (version_id, source_id) DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    chunk_index = EXCLUDED.chunk_index,
                    page_idx = EXCLUDED.page_idx,
                    bbox = EXCLUDED.bbox,
                    element_type = EXCLUDED.element_type,
                    text_raw = EXCLUDED.text_raw,
                    text_hash = EXCLUDED.text_hash,
                    text_tsv = EXCLUDED.text_tsv,
                    img_path = EXCLUDED.img_path,
                    image_caption = EXCLUDED.image_caption,
                    image_footnote = EXCLUDED.image_footnote,
                    table_body = EXCLUDED.table_body,
                    table_caption = EXCLUDED.table_caption,
                    table_footnote = EXCLUDED.table_footnote,
                    list_items = EXCLUDED.list_items,
                    sub_type = EXCLUDED.sub_type,
                    text_level = EXCLUDED.text_level
                RETURNING chunk_id
                """,
                (
                    new_chunk_id,
                    project_id,
                    version_id,
                    data["source_id"],
                    data["chunk_index"],
                    data["page_idx"],
                    data["bbox"],
                    data["element_type"],
                    data["text_raw"],
                    data["text_hash"],
                    (data.get("text_tsv") or ""),
                    data["img_path"],
                    data["image_caption"],
                    data["image_footnote"],
                    data["table_body"],
                    data["table_caption"],
                    data["table_footnote"],
                    data["list_items"],
                    data["sub_type"],
                    data["text_level"],
                ),
            )
            (chunk_id,) = cur.fetchone()

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
                    str(chunk_id),
                    unit["unit_id"],
                    0,
                    0,
                    len(unit.get("text_raw") or ""),
                ),
            )

        # Remove stale chunks that aren't part of this ingestion run.
        if seen_source_ids:
            cur.execute(
                """
                DELETE FROM chunks
                WHERE version_id = %s
                  AND NOT (source_id = ANY(%s))
                """,
                (version_id, sorted(seen_source_ids)),
            )
        else:
            cur.execute("DELETE FROM chunks WHERE version_id = %s", (version_id,))

    conn.commit()

    return {
        "total_chunks": len(chunks_data),
        "total_units": len(units_data),
        "type_counts": type_counts,
        "skipped_page_numbers": type_counts.get("page_number", 0),
    }
