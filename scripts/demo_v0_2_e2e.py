#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from bid_scoring.chunk_rebuild_v2 import rebuild_chunks_from_units
from bid_scoring.citations_v2 import compute_evidence_hash, verify_citation
from bid_scoring.config import load_settings
from bid_scoring.hybrid_retrieval import HybridRetriever
from bid_scoring.ingest import ingest_content_list


def _load_content_list(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("content_list must be a JSON array")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v0.2 end-to-end evidence demo (offline)"
    )
    parser.add_argument(
        "--content-list",
        default="data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_A.json",
        help="Path to MineRU-style content_list JSON",
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=["培训", "质保"],
        help="Keywords for ILIKE-based retrieval (offline)",
    )
    parser.add_argument(
        "--rechunk-group-size",
        type=int,
        default=3,
        help="Chunk rebuild group size (simulate index-layer changes)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=200,
        help="Only ingest the first N items from content_list (0 = all)",
    )
    args = parser.parse_args()

    settings = load_settings()
    dsn = settings["DATABASE_URL"]

    content_list_path = Path(args.content_list)
    content_list = _load_content_list(content_list_path)
    if args.max_items and args.max_items > 0:
        content_list = content_list[: int(args.max_items)]

    project_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())

    run_id = str(uuid.uuid4())
    result_id = str(uuid.uuid4())
    citation_id = str(uuid.uuid4())

    print("=== v0.2 E2E Demo ===")
    print(f"content_list: {content_list_path}")
    print(f"project_id:   {project_id}")
    print(f"document_id:  {document_id}")
    print(f"version_id:   {version_id}")
    print()

    with psycopg.connect(dsn) as conn:
        ingest_stats = ingest_content_list(
            conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            document_title="demo-doc",
            source_type="mineru",
            source_uri=str(content_list_path),
            parser_version="demo-v0.2",
            content_list=content_list,
        )
        print("Ingest:", ingest_stats)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM content_units WHERE version_id = %s",
                (version_id,),
            )
            units_count = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE version_id = %s", (version_id,)
            )
            chunks_count = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT page_idx, page_w, page_h, coord_sys
                FROM document_pages
                WHERE version_id = %s
                ORDER BY page_idx
                LIMIT 3
                """,
                (version_id,),
            )
            pages_preview = cur.fetchall()

        print(f"Counts: content_units={units_count} chunks={chunks_count}")
        print("document_pages preview (page_idx, page_w, page_h, coord_sys):")
        for row in pages_preview:
            print(f"  {row}")
        print()

        # Build hierarchical nodes and show covered_unit_range.
        scripts_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(scripts_dir))
        from build_hichunk_nodes import get_chunk_mapping, insert_hierarchical_nodes  # noqa: E402
        from bid_scoring.hichunk import HiChunkBuilder  # noqa: E402

        builder = HiChunkBuilder()
        nodes = builder.build_hierarchy(content_list, "demo-doc")
        chunk_mapping = get_chunk_mapping(conn, version_id)
        insert_hierarchical_nodes(conn, version_id, nodes, chunk_mapping)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT node_type, metadata->'covered_unit_range'
                FROM hierarchical_nodes
                WHERE version_id = %s
                  AND metadata ? 'covered_unit_range'
                ORDER BY level DESC, node_type
                LIMIT 5
                """,
                (version_id,),
            )
            hn_preview = cur.fetchall()

        print("hierarchical_nodes covered_unit_range preview:")
        for row in hn_preview:
            print(f"  {row}")
        print()

        # Offline retrieval: keyword legacy search + fetch (attaches evidence_units).
        retriever = HybridRetriever(version_id=version_id, settings=settings, top_k=5)
        keyword_hits = retriever._keyword_search_legacy(args.keywords)
        merged = []
        for rank, (chunk_id, score) in enumerate(keyword_hits[:5]):
            merged.append(
                (
                    chunk_id,
                    float(score),
                    {"keyword": {"rank": rank, "score": float(score)}},
                )
            )
        results = retriever._fetch_chunks(merged)
        retriever.close()

        if not results:
            raise SystemExit(f"No retrieval results for keywords={args.keywords}")

        top = results[0]
        ev0 = top.evidence_units[0]
        print("Top retrieval result:")
        print(f"  chunk_id: {top.chunk_id}")
        print(f"  page_idx: {top.page_idx}")
        print(f"  evidence unit_id: {ev0.unit_id}")
        print(f"  anchor_json first: {ev0.anchor_json.get('anchors', [{}])[0]}")
        print()

        # Create a citation pinned to unit_id + quote span.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT text_raw, unit_hash, anchor_json
                FROM content_units
                WHERE unit_id = %s
                """,
                (ev0.unit_id,),
            )
            unit_text, unit_hash, unit_anchor = cur.fetchone()

        quote_text = unit_text or ""
        evidence_hash = compute_evidence_hash(
            quote_text=quote_text, unit_hash=unit_hash, anchor_json=unit_anchor
        )

        with conn.cursor() as cur:
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
                    top.chunk_id,
                    quote_text,
                    ev0.unit_id,
                    quote_text,
                    0,
                    len(quote_text),
                    Jsonb(unit_anchor),
                    evidence_hash,
                    True,
                    "exact",
                ),
            )
        conn.commit()

        print(
            "Citation verify (before rechunk):",
            verify_citation(conn, citation_id=citation_id),
        )
        print()

        # Simulate index evolution: rebuild chunks from stable units.
        rechunk_stats = rebuild_chunks_from_units(
            conn,
            project_id=project_id,
            version_id=version_id,
            group_size=int(args.rechunk_group_size),
        )
        print("Rechunk:", rechunk_stats)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id FROM citations WHERE citation_id = %s", (citation_id,)
            )
            (chunk_id_after_rechunk,) = cur.fetchone()
        print(
            "Citation.chunk_id after rechunk (should be NULL):", chunk_id_after_rechunk
        )
        print(
            "Citation verify (after rechunk):",
            verify_citation(conn, citation_id=citation_id),
        )


if __name__ == "__main__":
    main()
