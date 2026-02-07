from __future__ import annotations

import uuid

import psycopg

from bid_scoring.chunk_rebuild_v2 import rebuild_chunks_from_units
from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list


def test_rebuild_chunks_from_units_groups_units_and_writes_spans():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000601"
    document_id = "00000000-0000-0000-0000-000000000603"
    version_id = "00000000-0000-0000-0000-000000000602"

    content_list = [
        {"type": "text", "text": "A", "page_idx": 1, "bbox": [0, 0, 10, 10]},
        {"type": "text", "text": "B", "page_idx": 1, "bbox": [0, 20, 10, 30]},
        {"type": "text", "text": "C", "page_idx": 2, "bbox": [0, 0, 99, 1]},
    ]

    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(rechunk)",
            source_type="mineru",
            content_list=content_list,
        )

        stats = rebuild_chunks_from_units(
            conn, project_id=project_id, version_id=version_id, group_size=2
        )
        assert stats["chunks_inserted"] == 2
        assert stats["spans_inserted"] == 3

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE version_id = %s", (version_id,)
            )
            assert int(cur.fetchone()[0]) == 2

            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunk_unit_spans s
                JOIN chunks c ON c.chunk_id = s.chunk_id
                WHERE c.version_id = %s
                """,
                (version_id,),
            )
            assert int(cur.fetchone()[0]) == 3
