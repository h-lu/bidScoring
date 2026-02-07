from __future__ import annotations

import json
from pathlib import Path

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list


def test_ingest_infers_document_page_dimensions_from_bboxes():
    data = json.loads(
        Path("tests/fixtures/sample_content_list.json").read_text(encoding="utf-8")
    )
    dsn = load_settings()["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000401"
    document_id = "00000000-0000-0000-0000-000000000403"
    version_id = "00000000-0000-0000-0000-000000000402"

    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(page dims)",
            source_type="mineru",
            content_list=data,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT page_idx, page_w, page_h, coord_sys
                FROM document_pages
                WHERE version_id = %s
                ORDER BY page_idx
                """,
                (version_id,),
            )
            rows = cur.fetchall()

    by_page = {int(r[0]): r for r in rows}
    assert 12 in by_page
    assert 34 in by_page

    # page_w/h are inferred as max(x2)/max(y2) on that page.
    assert float(by_page[12][1]) == 300.0
    assert float(by_page[12][2]) == 240.0
    assert by_page[12][3] == "mineru_bbox_v1"

    assert float(by_page[34][1]) == 360.0
    assert float(by_page[34][2]) == 300.0
    assert by_page[34][3] == "mineru_bbox_v1"
