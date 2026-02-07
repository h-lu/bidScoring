from __future__ import annotations

import uuid

import psycopg
from psycopg.types.json import Jsonb

from bid_scoring.citations_v2 import compute_evidence_hash, verify_citation
from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list


def test_citation_verifies_after_chunk_rebuild():
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000501"
    document_id = "00000000-0000-0000-0000-000000000503"
    version_id = "00000000-0000-0000-0000-000000000502"

    # Minimal content_list.
    content_list = [
        {
            "type": "text",
            "text": "培训时间：2天",
            "page_idx": 1,
            "bbox": [0, 0, 100, 10],
        },
        {"type": "text", "text": "质保：5年", "page_idx": 1, "bbox": [0, 20, 200, 30]},
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
            document_title="示例文档(citations)",
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

            quote_text = unit_text
            quote_start = 0
            quote_end = len(quote_text)
            evidence_hash = compute_evidence_hash(
                quote_text=quote_text, unit_hash=unit_hash, anchor_json=anchor_json
            )

            # Create a minimal scoring run/result to hang the citation off.
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
                    quote_text,
                    unit_id,
                    quote_text,
                    quote_start,
                    quote_end,
                    Jsonb(anchor_json),
                    evidence_hash,
                    True,
                    "exact",
                ),
            )
        conn.commit()

        assert verify_citation(conn, citation_id=citation_id)["ok"] is True

        # Rebuild chunk index layer: deleting chunks must NOT break citations (chunk_id is optional provenance).
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE version_id = %s", (version_id,))
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM citations WHERE citation_id = %s", (citation_id,)
            )
            (chunk_id_after_delete,) = cur.fetchone()
        assert chunk_id_after_delete is None

        assert verify_citation(conn, citation_id=citation_id)["ok"] is True
