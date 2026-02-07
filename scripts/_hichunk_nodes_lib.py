from __future__ import annotations

from functools import lru_cache
from typing import Any

from psycopg.types.json import Jsonb


def reset_hierarchical_nodes(
    conn, version_id: str | None = None, force: bool = False
) -> bool:
    """Reset/clear hierarchical_nodes table."""
    with conn.cursor() as cur:
        if version_id:
            cur.execute(
                "SELECT COUNT(*) FROM hierarchical_nodes WHERE version_id = %s",
                (version_id,),
            )
        else:
            cur.execute("SELECT COUNT(*) FROM hierarchical_nodes")

        count = cur.fetchone()[0]

        if count == 0:
            print("ℹ️  hierarchical_nodes 表为空，无需重置")
            return True

        if not force:
            scope = f"版本 '{version_id}'" if version_id else "所有版本"
            print(
                f"\n⚠️  警告: 这将删除 {scope} 的 {count} 条 hierarchical_nodes 记录！"
            )
            response = input("确认重置? 输入 'yes' 继续: ")
            if response.lower() != "yes":
                print("❌ 操作已取消")
                return False

        if version_id:
            cur.execute(
                "DELETE FROM hierarchical_nodes WHERE version_id = %s", (version_id,)
            )
        else:
            cur.execute("DELETE FROM hierarchical_nodes")

        conn.commit()

        scope = f"版本 '{version_id}'" if version_id else "所有版本"
        print(f"✅ 已重置 {scope} 的 {count} 条记录")
        return True


def get_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    """Get processing stats for hierarchical nodes building."""
    with conn.cursor() as cur:
        base_query = """
            SELECT 
                COUNT(DISTINCT c.version_id) as total_versions,
                COUNT(DISTINCT hn.version_id) as processed_versions,
                COUNT(DISTINCT c.version_id) FILTER (WHERE hn.version_id IS NULL) as to_process
            FROM chunks c
            LEFT JOIN hierarchical_nodes hn ON c.version_id = hn.version_id
        """

        params: list[Any] = []
        if version_id:
            base_query += " WHERE c.version_id = %s"
            params.append(version_id)

        cur.execute(base_query, params)
        row = cur.fetchone()

        nodes_query = "SELECT COUNT(*) FROM hierarchical_nodes"
        nodes_params: list[Any] = []
        if version_id:
            nodes_query += " WHERE version_id = %s"
            nodes_params.append(version_id)

        cur.execute(nodes_query, nodes_params)
        total_nodes = cur.fetchone()[0]

        return {
            "total_versions": row[0] or 0,
            "processed": row[1] or 0,
            "to_process": row[2] or 0,
            "total_nodes": total_nodes,
        }


def fetch_pending_versions(
    conn,
    batch_size: int = 10,
    version_id: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch versions that need hierarchical nodes built (based on chunks)."""
    with conn.cursor() as cur:
        query = """
            SELECT DISTINCT
                dv.version_id,
                dv.doc_id,
                d.title as document_title
            FROM document_versions dv
            JOIN documents d ON dv.doc_id = d.doc_id
            JOIN chunks c ON dv.version_id = c.version_id
            LEFT JOIN hierarchical_nodes hn ON dv.version_id = hn.version_id
            WHERE hn.version_id IS NULL
        """

        params: list[Any] = []
        if version_id:
            query += " AND dv.version_id = %s"
            params.append(version_id)

        query += " ORDER BY dv.version_id LIMIT %s"
        params.append(batch_size)

        cur.execute(query, params)
        rows = cur.fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            version_id_str = str(row[0])

            cur.execute(
                """
                SELECT 
                    chunk_id,
                    chunk_index,
                    page_idx,
                    bbox,
                    element_type,
                    text_raw,
                    text_level,
                    img_path,
                    image_caption,
                    image_footnote,
                    table_body,
                    table_caption,
                    table_footnote,
                    list_items,
                    sub_type
                FROM chunks 
                WHERE version_id = %s 
                ORDER BY chunk_index
                """,
                (version_id_str,),
            )

            chunks_rows = cur.fetchall()
            content_list = []
            for cr in chunks_rows:
                item = {
                    "chunk_id": str(cr[0]),
                    "index": cr[1],
                    "page_idx": cr[2],
                    "bbox": cr[3],
                    "type": cr[4],
                    "text": cr[5],
                    "text_level": cr[6],
                    "img_path": cr[7],
                    "image_caption": cr[8],
                    "image_footnote": cr[9],
                    "table_body": cr[10],
                    "table_caption": cr[11],
                    "table_footnote": cr[12],
                    "list_items": cr[13],
                    "sub_type": cr[14],
                }
                content_list.append(item)

            result.append(
                {
                    "version_id": version_id_str,
                    "doc_id": str(row[1]),
                    "document_title": row[2],
                    "content_list": content_list,
                }
            )

        return result


def insert_hierarchical_nodes(
    conn,
    version_id: str,
    nodes: list,
    chunk_mapping: dict[int, str],
    show_detail: bool = False,
) -> tuple[int, int]:
    """Insert hierarchical nodes into database (upsert)."""
    if not nodes:
        return 0, 0

    success_count = 0
    fail_count = 0

    nodes_by_level = {3: [], 2: [], 1: [], 0: []}
    leaf_nodes = []

    for node in nodes:
        nodes_by_level[node.level].append(node)
        if node.level == 0:
            leaf_nodes.append(node)

    nodes_by_id = {n.node_id: n for n in nodes if n.node_id}
    leaf_by_id = {n.node_id: n for n in leaf_nodes if n.node_id}

    @lru_cache(maxsize=None)
    def _covered_unit_range(node_id: str) -> tuple[int, int] | None:
        n = nodes_by_id.get(node_id)
        if not n:
            return None
        if n.level == 0:
            source_idx = n.metadata.get("source_index")
            if source_idx is None:
                return None
            i = int(source_idx)
            return (i, i)

        ranges: list[tuple[int, int]] = []
        for child_id in n.children_ids or []:
            r = _covered_unit_range(child_id)
            if r is not None:
                ranges.append(r)
        if not ranges:
            return None
        return (min(r[0] for r in ranges), max(r[1] for r in ranges))

    def prepare_insert_data(node_list):
        data = []
        for node in node_list:
            cov = _covered_unit_range(node.node_id)
            if cov is not None:
                node.metadata["covered_unit_range"] = {"start": cov[0], "end": cov[1]}

            start_chunk_id = None
            end_chunk_id = None

            if node.level == 0:
                source_idx = node.metadata.get("source_index")
                if source_idx is not None and source_idx in chunk_mapping:
                    start_chunk_id = chunk_mapping[source_idx]
                    end_chunk_id = chunk_mapping[source_idx]
            elif node.level == 1:  # paragraph
                child_source_indices = []
                for child_id in node.children_ids:
                    child_node = leaf_by_id.get(child_id)
                    if child_node:
                        source_idx = child_node.metadata.get("source_index")
                        if source_idx is not None:
                            child_source_indices.append(source_idx)

                if child_source_indices:
                    min_idx = min(child_source_indices)
                    max_idx = max(child_source_indices)
                    if min_idx in chunk_mapping:
                        start_chunk_id = chunk_mapping[min_idx]
                    if max_idx in chunk_mapping:
                        end_chunk_id = chunk_mapping[max_idx]

            data.append(
                (
                    node.node_id,
                    version_id,
                    node.parent_id,
                    node.level,
                    node.node_type,
                    node.content,
                    node.children_ids,
                    start_chunk_id,
                    end_chunk_id,
                    Jsonb(node.metadata),
                )
            )
        return data

    try:
        with conn.cursor() as cur:
            for level in [3, 2, 1, 0]:
                level_nodes = nodes_by_level[level]
                if not level_nodes:
                    continue

                insert_data = prepare_insert_data(level_nodes)
                cur.executemany(
                    """
                    INSERT INTO hierarchical_nodes (
                        node_id, version_id, parent_id, level, node_type,
                        content, children_ids, start_chunk_id, end_chunk_id, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (node_id) DO UPDATE SET
                        version_id = EXCLUDED.version_id,
                        parent_id = EXCLUDED.parent_id,
                        level = EXCLUDED.level,
                        node_type = EXCLUDED.node_type,
                        content = EXCLUDED.content,
                        children_ids = EXCLUDED.children_ids,
                        start_chunk_id = EXCLUDED.start_chunk_id,
                        end_chunk_id = EXCLUDED.end_chunk_id,
                        metadata = EXCLUDED.metadata
                    """,
                    insert_data,
                )
                success_count += len(level_nodes)

        conn.commit()

        if show_detail:
            print(f"  ✅ 已插入 {success_count} 个节点")
            for level in range(4):
                count = len(nodes_by_level[level])
                level_name = ["sentence", "paragraph", "section", "document"][level]
                print(f"    - Level {level} ({level_name}): {count}")

    except Exception as e:
        conn.rollback()
        fail_count = len(nodes) - success_count
        print(f"  ❌ 插入失败: {e}")

    return success_count, fail_count


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f}分钟"
    return f"{seconds / 3600:.1f}小时"
