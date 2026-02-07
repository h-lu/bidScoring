from __future__ import annotations

from typing import Any

from bid_scoring.embeddings import embed_texts


DEFAULT_BATCH_SIZE = 100
DEFAULT_LIMIT = 10000
DEFAULT_MAX_TOKENS = 100000


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f}分钟"
    return f"{seconds / 3600:.1f}小时"


def get_chunks_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    with conn.cursor() as cur:
        base_query = """
            SELECT 
                COUNT(*) FILTER (WHERE embedding IS NULL) as null_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as has_count,
                COUNT(*) FILTER (WHERE embedding IS NULL AND text_raw IS NOT NULL AND text_raw != '') as to_process,
                COUNT(*) FILTER (WHERE embedding IS NULL AND (text_raw IS NULL OR text_raw = '')) as empty_text
            FROM chunks
        """

        if version_id:
            base_query += " WHERE version_id = %s"
            cur.execute(base_query, (version_id,))
        else:
            cur.execute(base_query)

        row = cur.fetchone()
        return {
            "null_count": row[0],
            "has_count": row[1],
            "to_process": row[2],
            "empty_text": row[3],
            "total": row[0] + row[1],
        }


def process_chunks_embeddings(
    conn,
    version_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int = DEFAULT_LIMIT,
    client=None,
    model: str | None = None,
    show_detail: bool = False,
) -> tuple[int, int]:
    stats = get_chunks_stats(conn, version_id)
    to_process = min(stats["to_process"], limit)

    if to_process == 0:
        print("  ✅ chunks 表无需处理")
        return 0, 0

    print(f"  待处理: {to_process} 条")

    processed = 0
    success_count = 0
    fail_count = 0
    batch_num = 0

    while processed < to_process:
        with conn.cursor() as cur:
            query = """
                SELECT chunk_id, text_raw
                FROM chunks 
                WHERE embedding IS NULL 
                  AND text_raw IS NOT NULL 
                  AND text_raw != ''
            """
            params: list[Any] = []
            if version_id:
                query += " AND version_id = %s"
                params.append(version_id)
            query += " ORDER BY chunk_id LIMIT %s"
            params.append(batch_size)

            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                break

            batch_num += 1
            ids = [str(r[0]) for r in rows]
            texts = [r[1] for r in rows]

            try:
                vecs = embed_texts(
                    texts, client=client, model=model, show_progress=False
                )
                update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
                cur.executemany(
                    "UPDATE chunks SET embedding = %s WHERE chunk_id = %s", update_data
                )
                conn.commit()

                success_count += len(rows)
                if show_detail:
                    print(f"    批次 {batch_num}: {len(rows)} 条 ✅")

            except Exception as e:
                conn.rollback()
                fail_count += len(rows)
                print(f"    批次 {batch_num}: ❌ {e}")

            processed += len(rows)

    print(f"  完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


def build_contextual_chunks(conn, version_id: str | None = None) -> int:
    """Build contextual_chunks from chunks + hierarchical_nodes."""
    with conn.cursor() as cur:
        if version_id:
            cur.execute(
                "DELETE FROM contextual_chunks WHERE version_id = %s", (version_id,)
            )
        else:
            cur.execute("DELETE FROM contextual_chunks")

        query = """
            SELECT 
                c.chunk_id,
                c.version_id,
                c.text_raw,
                c.page_idx,
                c.element_type
            FROM chunks c
            WHERE c.text_raw IS NOT NULL 
              AND c.text_raw != ''
        """
        params: list[Any] = []
        if version_id:
            query += " AND c.version_id = %s"
            params.append(version_id)

        cur.execute(query, params)
        chunks = cur.fetchall()

        if not chunks:
            return 0

        created_count = 0
        for chunk_id, ver_id, text_raw, page_idx, _elem_type in chunks:
            cur.execute(
                """
                SELECT content, metadata
                FROM hierarchical_nodes
                WHERE version_id = %s
                  AND level = 2
                  AND (metadata->>'page_idx')::int <= %s
                ORDER BY (metadata->>'page_idx')::int DESC
                LIMIT 1
                """,
                (ver_id, page_idx or 0),
            )
            section_row = cur.fetchone()

            section_title = ""
            if section_row:
                section_title = section_row[0] or ""

            context_prefix = f"[{section_title}] " if section_title else ""
            contextualized_text = context_prefix + text_raw

            cur.execute(
                """
                INSERT INTO contextual_chunks (
                    chunk_id, version_id, original_text, context_prefix,
                    contextualized_text, model_name, embedding_model
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    context_prefix = EXCLUDED.context_prefix,
                    contextualized_text = EXCLUDED.contextualized_text
                """,
                (
                    chunk_id,
                    ver_id,
                    text_raw,
                    context_prefix,
                    contextualized_text,
                    "text-embedding-3-small",
                    "text-embedding-3-small",
                ),
            )
            created_count += 1

        conn.commit()
        return created_count


def process_contextual_embeddings(
    conn,
    version_id: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int = DEFAULT_LIMIT,
    client=None,
    model: str | None = None,
    show_detail: bool = False,
) -> tuple[int, int]:
    with conn.cursor() as cur:
        query = """
            SELECT COUNT(*) 
            FROM contextual_chunks 
            WHERE embedding IS NULL
        """
        params: list[Any] = []
        if version_id:
            query += " AND version_id = %s"
            params.append(version_id)

        cur.execute(query, params)
        to_process = cur.fetchone()[0]
        to_process = min(to_process, limit)

        if to_process == 0:
            print("  ✅ contextual_chunks 表无需处理")
            return 0, 0

        print(f"  待处理: {to_process} 条")

    processed = 0
    success_count = 0
    fail_count = 0
    batch_num = 0

    while processed < to_process:
        with conn.cursor() as cur:
            query = """
                SELECT contextual_id, contextualized_text
                FROM contextual_chunks 
                WHERE embedding IS NULL
            """
            params = []
            if version_id:
                query += " AND version_id = %s"
                params.append(version_id)
            query += " ORDER BY contextual_id LIMIT %s"
            params.append(batch_size)

            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                break

            batch_num += 1
            ids = [str(r[0]) for r in rows]
            texts = [r[1] for r in rows]

            try:
                vecs = embed_texts(
                    texts, client=client, model=model, show_progress=False
                )
                update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
                cur.executemany(
                    "UPDATE contextual_chunks SET embedding = %s WHERE contextual_id = %s",
                    update_data,
                )
                conn.commit()

                success_count += len(rows)
                if show_detail:
                    print(f"    批次 {batch_num}: {len(rows)} 条 ✅")

            except Exception as e:
                conn.rollback()
                fail_count += len(rows)
                print(f"    批次 {batch_num}: ❌ {e}")

            processed += len(rows)

    print(f"  完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


def get_hierarchical_stats(conn, version_id: str | None = None) -> dict[str, Any]:
    with conn.cursor() as cur:
        query = """
            SELECT 
                level,
                COUNT(*) FILTER (WHERE embedding IS NULL) as null_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as has_count
            FROM hierarchical_nodes
        """
        params: list[Any] = []
        if version_id:
            query += " WHERE version_id = %s"
            params.append(version_id)

        query += " GROUP BY level ORDER BY level"
        cur.execute(query, params)

        stats: dict[int, dict[str, int]] = {}
        for row in cur.fetchall():
            stats[row[0]] = {
                "null_count": row[1],
                "has_count": row[2],
                "total": row[1] + row[2],
            }
        return stats


def process_hierarchical_embeddings(
    conn,
    version_id: str | None = None,
    levels: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int = DEFAULT_LIMIT,
    client=None,
    model: str | None = None,
    show_detail: bool = False,
) -> dict[int, tuple[int, int]]:
    if levels is None:
        levels = [1, 2]

    level_names = {0: "sentence", 1: "paragraph", 2: "section", 3: "document"}
    results: dict[int, tuple[int, int]] = {}

    for level in levels:
        level_name = level_names.get(level, f"level_{level}")
        print(f"\n  处理 Level {level} ({level_name}):")

        processed = 0
        success_count = 0
        fail_count = 0
        batch_num = 0

        while processed < limit:
            with conn.cursor() as cur:
                query = """
                    SELECT node_id, content
                    FROM hierarchical_nodes 
                    WHERE embedding IS NULL
                      AND level = %s
                """
                params: list[Any] = [level]
                if version_id:
                    query += " AND version_id = %s"
                    params.append(version_id)
                query += " ORDER BY node_id LIMIT %s"
                params.append(batch_size)

                cur.execute(query, params)
                rows = cur.fetchall()

                if not rows:
                    break

                batch_num += 1
                ids = [str(r[0]) for r in rows]
                texts = [r[1] for r in rows]

                try:
                    vecs = embed_texts(
                        texts, client=client, model=model, show_progress=False
                    )
                    update_data = [(vecs[i], ids[i]) for i in range(len(ids))]
                    cur.executemany(
                        "UPDATE hierarchical_nodes SET embedding = %s WHERE node_id = %s",
                        update_data,
                    )
                    conn.commit()

                    success_count += len(rows)
                    if show_detail:
                        print(f"    批次 {batch_num}: {len(rows)} 条 ✅")

                except Exception as e:
                    conn.rollback()
                    fail_count += len(rows)
                    print(f"    批次 {batch_num}: ❌ {e}")

                processed += len(rows)

        print(f"  完成: 成功 {success_count}, 失败 {fail_count}")
        results[level] = (success_count, fail_count)

    return results
