from __future__ import annotations

import json
from pathlib import Path

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list


def _as_dict(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def test_ingest_persists_content_units_and_spans():
    data = json.loads(
        Path("tests/fixtures/sample_content_list.json").read_text(encoding="utf-8")
    )
    dsn = load_settings()["DATABASE_URL"]

    # Dedicated IDs for this test to avoid collisions with other tests.
    project_id = "00000000-0000-0000-0000-000000000101"
    document_id = "00000000-0000-0000-0000-000000000103"
    version_id = "00000000-0000-0000-0000-000000000102"

    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(v0.2 units)",
            source_type="mineru",
            content_list=data,
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM content_units WHERE version_id = %s",
                (version_id,),
            )
            units_count = int(cur.fetchone()[0])
            assert units_count > 0

            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE version_id = %s", (version_id,)
            )
            chunks_count = int(cur.fetchone()[0])
            assert chunks_count > 0

            cur.execute(
                """
                SELECT COUNT(*)
                FROM chunk_unit_spans s
                JOIN chunks c ON c.chunk_id = s.chunk_id
                WHERE c.version_id = %s
                """,
                (version_id,),
            )
            spans_count = int(cur.fetchone()[0])
            assert spans_count == chunks_count

            cur.execute(
                """
                SELECT anchor_json
                FROM content_units
                WHERE version_id = %s
                ORDER BY unit_index
                LIMIT 1
                """,
                (version_id,),
            )
            anchor_json = _as_dict(cur.fetchone()[0])
            assert anchor_json["anchors"][0]["page_idx"] == data[0]["page_idx"]
            assert anchor_json["anchors"][0]["bbox"] == data[0]["bbox"]


def test_ingest_expands_multi_bbox_into_multiple_anchors_and_sets_page_dims():
    dsn = load_settings()["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000111"
    document_id = "00000000-0000-0000-0000-000000000113"
    version_id = "00000000-0000-0000-0000-000000000112"

    content_list = [
        {
            "type": "text",
            "text": "Multi bbox item",
            "page_idx": 0,
            "bbox": [[0, 0, 10, 10], [0, 20, 10, 30]],
        }
    ]

    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(multi bbox)",
            source_type="mineru",
            content_list=content_list,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT anchor_json
                FROM content_units
                WHERE version_id = %s
                ORDER BY unit_index
                LIMIT 1
                """,
                (version_id,),
            )
            anchor_json = _as_dict(cur.fetchone()[0])

        anchors = anchor_json["anchors"]
        assert len(anchors) == 2
        assert anchors[0]["page_idx"] == 0
        assert list(map(int, anchors[0]["bbox"])) == [0, 0, 10, 10]
        assert list(map(int, anchors[1]["bbox"])) == [0, 20, 10, 30]

        # Page dims are inferred from observed bbox x2/y2 maxima.
        assert int(anchors[0]["page_w"]) == 10
        assert int(anchors[0]["page_h"]) == 30
