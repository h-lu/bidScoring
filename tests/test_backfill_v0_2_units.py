from __future__ import annotations

import uuid

import json
import psycopg

from bid_scoring.backfill_v0_2 import backfill_units_from_chunks
from bid_scoring.config import load_settings


def test_backfill_creates_units_and_spans_from_existing_chunks():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000301"
    document_id = "00000000-0000-0000-0000-000000000303"
    version_id = "00000000-0000-0000-0000-000000000302"

    chunk_id_1 = uuid.uuid4()
    chunk_id_2 = uuid.uuid4()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Clean up in case the test is re-run locally.
            cur.execute(
                "DELETE FROM chunk_unit_spans WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE version_id = %s)",
                (version_id,),
            )
            cur.execute("DELETE FROM chunks WHERE version_id = %s", (version_id,))
            cur.execute(
                "DELETE FROM content_units WHERE version_id = %s", (version_id,)
            )
            cur.execute(
                "DELETE FROM document_pages WHERE version_id = %s", (version_id,)
            )
            cur.execute(
                "DELETE FROM document_versions WHERE version_id = %s", (version_id,)
            )
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (document_id,))
            cur.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))

            cur.execute(
                "INSERT INTO projects (project_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (project_id, "project-backfill-test"),
            )
            cur.execute(
                "INSERT INTO documents (doc_id, project_id, title, source_type) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (document_id, project_id, "doc-backfill-test", "mineru"),
            )
            cur.execute(
                """
                INSERT INTO document_versions (version_id, doc_id, source_uri, source_hash, parser_version, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (version_id, document_id, None, None, "test", "ready"),
            )

            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, project_id, version_id, source_id, chunk_index, page_idx,
                    bbox, element_type, text_raw, text_hash, text_tsv, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, NULL, NULL
                )
                """,
                (
                    chunk_id_1,
                    project_id,
                    version_id,
                    "chunk_0000",
                    0,
                    1,
                    json.dumps([1, 2, 3, 4]),
                    "text",
                    "A",
                    "hashA",
                ),
            )
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, project_id, version_id, source_id, chunk_index, page_idx,
                    bbox, element_type, text_raw, text_hash, text_tsv, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, NULL, NULL
                )
                """,
                (
                    chunk_id_2,
                    project_id,
                    version_id,
                    "chunk_0001",
                    1,
                    2,
                    json.dumps([10, 20, 30, 40]),
                    "text",
                    "B",
                    "hashB",
                ),
            )

        conn.commit()

        stats = backfill_units_from_chunks(conn, version_id=version_id)
        assert stats["versions_processed"] == 1

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM content_units WHERE version_id = %s",
                (version_id,),
            )
            assert int(cur.fetchone()[0]) == 2

            cur.execute(
                """
                SELECT page_idx, page_w, page_h
                FROM document_pages
                WHERE version_id = %s
                ORDER BY page_idx
                """,
                (version_id,),
            )
            assert cur.fetchall() == [(1, 3, 4), (2, 30, 40)]

            cur.execute(
                "SELECT COUNT(*) FROM chunk_unit_spans WHERE chunk_id IN (%s, %s)",
                (chunk_id_1, chunk_id_2),
            )
            assert int(cur.fetchone()[0]) == 2


def test_backfill_expands_multi_bbox_into_multiple_anchors_and_sets_page_dims():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000311"
    document_id = "00000000-0000-0000-0000-000000000313"
    version_id = "00000000-0000-0000-0000-000000000312"

    chunk_id = uuid.uuid4()

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunk_unit_spans WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE version_id = %s)",
                (version_id,),
            )
            cur.execute("DELETE FROM chunks WHERE version_id = %s", (version_id,))
            cur.execute(
                "DELETE FROM content_units WHERE version_id = %s", (version_id,)
            )
            cur.execute(
                "DELETE FROM document_pages WHERE version_id = %s", (version_id,)
            )
            cur.execute(
                "DELETE FROM document_versions WHERE version_id = %s", (version_id,)
            )
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (document_id,))
            cur.execute("DELETE FROM projects WHERE project_id = %s", (project_id,))

            cur.execute(
                "INSERT INTO projects (project_id, name) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (project_id, "project-backfill-multi-bbox"),
            )
            cur.execute(
                "INSERT INTO documents (doc_id, project_id, title, source_type) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (document_id, project_id, "doc-backfill-multi-bbox", "mineru"),
            )
            cur.execute(
                """
                INSERT INTO document_versions (version_id, doc_id, source_uri, source_hash, parser_version, status)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (version_id, document_id, None, None, "test", "ready"),
            )

            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, project_id, version_id, source_id, chunk_index, page_idx,
                    bbox, element_type, text_raw, text_hash, text_tsv, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, NULL, NULL
                )
                """,
                (
                    chunk_id,
                    project_id,
                    version_id,
                    "chunk_0000",
                    0,
                    0,
                    json.dumps([[0, 0, 10, 10], [0, 20, 10, 30]]),
                    "text",
                    "A",
                    "hashA",
                ),
            )

        conn.commit()

        backfill_units_from_chunks(conn, version_id=version_id)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT anchor_json FROM content_units WHERE version_id = %s LIMIT 1",
                (version_id,),
            )
            anchor_json = cur.fetchone()[0]
            if isinstance(anchor_json, str):
                anchor_json = json.loads(anchor_json)

            anchors = anchor_json["anchors"]
            assert len(anchors) == 2
            assert list(map(int, anchors[0]["bbox"])) == [0, 0, 10, 10]
            assert list(map(int, anchors[1]["bbox"])) == [0, 20, 10, 30]

            assert int(anchors[0]["page_w"]) == 10
            assert int(anchors[0]["page_h"]) == 30

            cur.execute(
                """
                SELECT page_idx, page_w, page_h
                FROM document_pages
                WHERE version_id = %s
                ORDER BY page_idx
                """,
                (version_id,),
            )
            assert cur.fetchall() == [(0, 10, 30)]
