from __future__ import annotations

import uuid

import psycopg
from psycopg.types.json import Jsonb

from bid_scoring.citations_v2 import compute_evidence_hash
from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list


def test_reingest_preserves_citation_chunk_id_provenance():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000711"
    document_id = "00000000-0000-0000-0000-000000000713"
    version_id = "00000000-0000-0000-0000-000000000712"

    content_list = [
        {
            "type": "text",
            "text": "售后服务：响应时间 2 小时",
            "page_idx": 0,
            "bbox": [0, 0, 100, 10],
        }
    ]

    run_id = str(uuid.uuid4())
    result_id = str(uuid.uuid4())
    citation_id = str(uuid.uuid4())

    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(reingest provenance)",
            source_type="mineru",
            content_list=content_list,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_id::text, cu.unit_id::text, cu.text_raw, cu.unit_hash, cu.anchor_json
                FROM chunks c
                JOIN chunk_unit_spans s ON s.chunk_id = c.chunk_id
                JOIN content_units cu ON cu.unit_id = s.unit_id
                WHERE c.version_id = %s
                ORDER BY c.chunk_index
                LIMIT 1
                """,
                (version_id,),
            )
            chunk_id, unit_id, unit_text, unit_hash, anchor_json = cur.fetchone()

            evidence_hash = compute_evidence_hash(
                quote_text=unit_text or "",
                unit_hash=unit_hash or "",
                anchor_json=anchor_json,
            )

            cur.execute(
                """
                INSERT INTO scoring_runs (run_id, project_id, version_id, dimensions, status)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, project_id, version_id, ["demo"], "done"),
            )
            cur.execute(
                """
                INSERT INTO scoring_results (result_id, run_id, dimension, score, max_score)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (result_id, run_id, "demo", 1.0, 1.0),
            )
            cur.execute(
                """
                INSERT INTO citations (
                    citation_id, result_id, source_id, chunk_id, cited_text,
                    unit_id, quote_text, quote_start_char, quote_end_char,
                    anchor_json, evidence_hash, verified, match_type
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                """,
                (
                    citation_id,
                    result_id,
                    "demo",
                    chunk_id,
                    unit_text,
                    unit_id,
                    unit_text,
                    0,
                    len(unit_text or ""),
                    Jsonb(anchor_json),
                    evidence_hash,
                    True,
                    "exact",
                ),
            )
        conn.commit()

        # Re-ingest the exact same content_list into the same version.
        # This should not delete and recreate chunks, otherwise citation.chunk_id
        # provenance would be nulled by ON DELETE SET NULL.
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(reingest provenance)",
            source_type="mineru",
            content_list=content_list,
        )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id::text FROM citations WHERE citation_id = %s",
                (citation_id,),
            )
            (chunk_id_after_reingest,) = cur.fetchone()
            assert chunk_id_after_reingest == chunk_id
