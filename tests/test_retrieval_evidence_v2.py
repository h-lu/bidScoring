from __future__ import annotations

import json
from pathlib import Path

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.hybrid_retrieval import HybridRetriever
from bid_scoring.ingest import ingest_content_list


def _as_dict(value):
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def test_fetch_chunks_attaches_unit_evidence():
    data = json.loads(
        Path("tests/fixtures/sample_content_list.json").read_text(encoding="utf-8")
    )
    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    project_id = "00000000-0000-0000-0000-000000000201"
    document_id = "00000000-0000-0000-0000-000000000203"
    version_id = "00000000-0000-0000-0000-000000000202"

    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="示例文档(v0.2 evidence)",
            source_type="mineru",
            content_list=data,
        )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id::text
                FROM chunks
                WHERE version_id = %s
                ORDER BY chunk_index
                LIMIT 2
                """,
                (version_id,),
            )
            chunk_ids = [r[0] for r in cur.fetchall()]

    assert len(chunk_ids) == 2

    retriever = HybridRetriever(version_id=version_id, settings=settings, top_k=2)
    merged_results = [
        (chunk_ids[0], 1.0, {"vector": {"rank": 0, "score": 0.9}}),
        (chunk_ids[1], 0.9, {"vector": {"rank": 1, "score": 0.8}}),
    ]
    results = retriever._fetch_chunks(merged_results)

    assert len(results) == 2
    for r in results:
        assert hasattr(r, "evidence_units")
        assert r.evidence_units
        ev = r.evidence_units[0]
        assert isinstance(ev.unit_id, str)
        anchor_json = _as_dict(ev.anchor_json)
        assert anchor_json["anchors"][0]["page_idx"] in {
            data[0]["page_idx"],
            data[1]["page_idx"],
        }
