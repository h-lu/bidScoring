#!/usr/bin/env python3
"""
End-to-end evaluation pipeline for hybrid_medical_synthetic dataset.

Steps:
1. Import A/B/C content_list into database with fixed version_ids
2. Build hierarchical nodes for each version
3. Run multi-version retrieval evaluation
4. Generate comprehensive report

Usage:
    uv run python scripts/eval_e2e_full_pipeline.py [--skip-ingest] [--skip-hichunk]
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list
from scripts.build_hichunk_nodes import get_chunk_mapping, insert_hierarchical_nodes
from bid_scoring.hichunk_builder import HiChunkBuilder


def load_content_list(path: Path) -> list[dict]:
    """Load and validate content_list JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("content_list must be a JSON array")
    return data


def ensure_project_and_document(conn: psycopg.Connection, project_id: str, document_id: str, version_id: str, bidder_name: str) -> None:
    """Ensure project, document, and version records exist."""
    with conn.cursor() as cur:
        # Insert project
        cur.execute(
            """
            INSERT INTO projects (project_id, name, status)
            VALUES (%s, %s, %s)
            ON CONFLICT (project_id) DO UPDATE SET name = EXCLUDED.name
            """,
            (project_id, f"eval_project_{bidder_name}", "active"),
        )
        # Insert document
        cur.execute(
            """
            INSERT INTO documents (doc_id, project_id, title, source_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title
            """,
            (document_id, project_id, f"doc_{bidder_name}", "mineru"),
        )
        # Insert version
        cur.execute(
            """
            INSERT INTO document_versions (version_id, doc_id, source_uri, parser_version, status)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (version_id) DO UPDATE SET parser_version = EXCLUDED.parser_version
            """,
            (version_id, document_id, f"synthetic_{bidder_name}", "v0.2", "active"),
        )
    conn.commit()


def ingest_scenario(
    conn: psycopg.Connection,
    content_file: Path,
    version_id: str,
    project_id: str,
    document_id: str,
    bidder_name: str,
) -> dict:
    """Ingest a single scenario (A/B/C) into database."""
    print(f"\n{'='*60}")
    print(f"Ingesting scenario: {bidder_name}")
    print(f"  version_id: {version_id}")
    print(f"  content_file: {content_file}")

    # Load content
    content_list = load_content_list(content_file)
    print(f"  content items: {len(content_list)}")

    # Ensure project/document/version exist
    ensure_project_and_document(conn, project_id, document_id, version_id, bidder_name)

    # Check if already ingested
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE version_id = %s",
            (version_id,),
        )
        existing_chunks = cur.fetchone()[0]

    if existing_chunks > 0:
        print(f"  ⚠️  Already ingested ({existing_chunks} chunks found), skipping...")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM content_units WHERE version_id = %s",
                (version_id,),
            )
            existing_units = cur.fetchone()[0]
        return {
            "chunks": existing_chunks,
            "content_units": existing_units,
            "skipped": True,
        }

    # Ingest content
    stats = ingest_content_list(
        conn,
        project_id=project_id,
        document_id=document_id,
        version_id=version_id,
        document_title=f"synthetic_{bidder_name}",
        source_type="mineru",
        source_uri=str(content_file),
        parser_version="v0.2-eval",
        content_list=content_list,
    )

    # Get final counts
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM content_units WHERE version_id = %s",
            (version_id,),
        )
        units_count = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE version_id = %s",
            (version_id,),
        )
        chunks_count = cur.fetchone()[0]

    print(f"  ✅ Ingested: units={units_count}, chunks={chunks_count}")
    return {
        "chunks": chunks_count,
        "content_units": units_count,
        "ingest_stats": stats,
        "skipped": False,
    }


def build_hichunk_nodes(
    conn: psycopg.Connection,
    content_file: Path,
    version_id: str,
    bidder_name: str,
) -> dict:
    """Build hierarchical nodes for a scenario."""
    print(f"\n  Building hierarchical nodes for {bidder_name}...")

    # Check if already built
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM hierarchical_nodes WHERE version_id = %s",
            (version_id,),
        )
        existing = cur.fetchone()[0]

    if existing > 0:
        print(f"  ⚠️  Already built ({existing} nodes found), skipping...")
        return {"nodes": existing, "skipped": True}

    # Load content and build hierarchy
    content_list = load_content_list(content_file)
    builder = HiChunkBuilder()
    nodes = builder.build_hierarchy(content_list, f"synthetic_{bidder_name}")

    # Get chunk mapping
    chunk_mapping = get_chunk_mapping(conn, version_id)

    # Insert nodes
    insert_hierarchical_nodes(conn, version_id, nodes, chunk_mapping)

    # Get count
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM hierarchical_nodes WHERE version_id = %s",
            (version_id,),
        )
        nodes_count = cur.fetchone()[0]

    print(f"  ✅ Built {nodes_count} hierarchical nodes")
    return {"nodes": nodes_count, "skipped": False}


def run_evaluation(eval_dir: Path, top_k: int = 10) -> dict:
    """Run multi-version evaluation."""
    print(f"\n{'='*60}")
    print("Running retrieval evaluation...")

    from scripts.evaluate_hybrid_search_multiversion import (
        _load_qrels,
        _load_queries,
        _map_source_to_chunk,
        _prepare_chunk_qrels,
        _run_method,
        _evaluate_query,
        QueryMetrics,
    )
    from bid_scoring.retrieval.hybrid import HybridRetriever
    import statistics

    # Load version map
    version_map_path = eval_dir / "version_map.json"
    version_map = json.loads(version_map_path.read_text(encoding="utf-8"))

    # Load queries
    queries_file = eval_dir / "queries.json"
    queries = _load_queries(queries_file)

    settings = load_settings()
    scenarios = sorted(version_map.keys())

    methods = ["vector", "keyword", "hybrid"]
    report: dict = {
        "top_k": top_k,
        "scenarios": {},
        "macro": {},
    }

    def _macro_mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        for scenario in scenarios:
            version_id = str(version_map[scenario])
            qrels_file = eval_dir / f"qrels.source_id.{scenario}.jsonl"

            qrels_rows = _load_qrels(qrels_file)
            source_to_chunk = _map_source_to_chunk(conn, version_id)
            qrels_by_query, unresolved_by_query = _prepare_chunk_qrels(
                qrels_rows, source_to_chunk
            )

            retriever = HybridRetriever(
                version_id=version_id,
                settings=settings,
                top_k=top_k,
            )

            metrics_by_method: dict[str, list[QueryMetrics]] = {m: [] for m in methods}

            print(f"\n  Scenario {scenario} (version_id={version_id}):")

            for query in queries:
                if query.query_id not in qrels_by_query:
                    continue
                qrels_for_query = qrels_by_query[query.query_id]
                unresolved_count = unresolved_by_query.get(query.query_id, 0)

                for method in methods:
                    retrieved_ids, latency_ms = _run_method(
                        retriever, query, method, top_k
                    )
                    row = _evaluate_query(
                        query,
                        method,
                        retrieved_ids,
                        qrels_for_query,
                        relevance_threshold=2,
                        latency_ms=latency_ms,
                        unresolved_count=unresolved_count,
                    )
                    metrics_by_method[method].append(row)

            retriever.close()

            summary = {
                method: {
                    "mrr": _macro_mean([x.mrr for x in rows]),
                    "recall_at_5": _macro_mean([x.recall_at_5 for x in rows]),
                    "precision_at_3": _macro_mean([x.precision_at_3 for x in rows]),
                    "ndcg_at_5": _macro_mean([x.ndcg_at_5 for x in rows]),
                    "ndcg_at_10": _macro_mean([x.ndcg_at_10 for x in rows]),
                    "latency_ms": _macro_mean([x.latency_ms for x in rows]),
                }
                for method, rows in metrics_by_method.items()
            }

            report["scenarios"][scenario] = {
                "version_id": version_id,
                "summary": summary,
            }

            print(f"    hybrid:  MRR={summary['hybrid']['mrr']:.4f} "
                  f"R@5={summary['hybrid']['recall_at_5']:.4f} "
                  f"nDCG@5={summary['hybrid']['ndcg_at_5']:.4f}")

    # Macro-average
    for method in methods:
        report["macro"][method] = {
            key: _macro_mean([
                report["scenarios"][s]["summary"][method][key]
                for s in scenarios
            ])
            for key in ["mrr", "recall_at_5", "precision_at_3", "ndcg_at_5", "ndcg_at_10", "latency_ms"]
        }

    print(f"\n{'='*60}")
    print("Macro-average across all scenarios:")
    for method in methods:
        s = report["macro"][method]
        print(f"  {method:<6} MRR={s['mrr']:.4f} R@5={s['recall_at_5']:.4f} "
              f"P@3={s['precision_at_3']:.4f} nDCG@5={s['ndcg_at_5']:.4f} "
              f"nDCG@10={s['ndcg_at_10']:.4f} Lat={s['latency_ms']:.2f}ms")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation pipeline"
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("data/eval/hybrid_medical_synthetic"),
        help="Directory containing evaluation data",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion (assume already ingested)",
    )
    parser.add_argument(
        "--skip-hichunk",
        action="store_true",
        help="Skip hierarchical node building",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation (only ingest and build)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K for retrieval evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report file",
    )

    args = parser.parse_args()

    # Load configuration
    eval_dir = args.eval_dir
    manifest_path = eval_dir / "multi_version_manifest.json"

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"{'='*60}")
    print(f"Dataset: {manifest['dataset']}")
    print(f"Scenarios: {', '.join(manifest['scenarios'].keys())}")

    # Fixed UUIDs for consistent testing
    project_ids = {
        "A": "11111111-1111-1111-1111-111111111111",
        "B": "11111111-1111-1111-1111-111111111112",
        "C": "11111111-1111-1111-1111-111111111113",
    }
    document_ids = {
        "A": "22222222-2222-2222-2222-222222222221",
        "B": "22222222-2222-2222-2222-222222222222",
        "C": "22222222-2222-2222-2222-222222222223",
    }

    # Load version map
    version_map = json.loads((eval_dir / "version_map.json").read_text())

    settings = load_settings()
    results = {"ingest": {}, "hichunk": {}}

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        # Step 1: Ingest data
        if not args.skip_ingest:
            print(f"\n{'='*60}")
            print("STEP 1: Data Ingestion")

            for scenario in ["A", "B", "C"]:
                scenario_config = manifest["scenarios"][scenario]
                content_file = eval_dir / scenario_config["content_file"]

                results["ingest"][scenario] = ingest_scenario(
                    conn=conn,
                    content_file=content_file,
                    version_id=version_map[scenario],
                    project_id=project_ids[scenario],
                    document_id=document_ids[scenario],
                    bidder_name=scenario_config["bidder_name"],
                )

        # Step 2: Build hierarchical nodes
        if not args.skip_hichunk:
            print(f"\n{'='*60}")
            print("STEP 2: Build Hierarchical Nodes (HiChunk)")

            for scenario in ["A", "B", "C"]:
                scenario_config = manifest["scenarios"][scenario]
                content_file = eval_dir / scenario_config["content_file"]

                results["hichunk"][scenario] = build_hichunk_nodes(
                    conn=conn,
                    content_file=content_file,
                    version_id=version_map[scenario],
                    bidder_name=scenario_config["bidder_name"],
                )

    # Step 3: Run evaluation
    if not args.skip_eval:
        results["evaluation"] = run_evaluation(eval_dir, args.top_k)

    # Save report
    if args.output:
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n{'='*60}")
        print(f"Report saved to: {args.output}")

    print(f"\n{'='*60}")
    print("✅ Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
