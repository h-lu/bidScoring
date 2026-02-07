#!/usr/bin/env python3
"""Evaluate golden retrieval metrics across multiple document versions.

This script runs `scripts/evaluate_hybrid_search_gold.py`-equivalent logic
for multiple version_ids (A/B/C) and outputs per-version and macro-average
metrics.

Inputs:
- version map: scenario -> version_id
- shared queries file
- per-scenario qrels files (source_id based)

Note: qrels are mapped to chunk_id via (version_id, source_id).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_hybrid_search_gold import (
    QueryMetrics,
    _evaluate_query,
    _load_qrels,
    _load_queries,
    _map_source_to_chunk,
    _prepare_chunk_qrels,
    _run_method,
)

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.hybrid_retrieval import HybridRetriever


def _macro_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-version golden retrieval evaluation"
    )
    parser.add_argument(
        "--version-map",
        type=Path,
        required=True,
        help='JSON file mapping scenarios to version_id, e.g. {"A":"uuid",...}',
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("data/eval/hybrid_medical_synthetic/queries.json"),
    )
    parser.add_argument(
        "--qrels-dir",
        type=Path,
        default=Path("data/eval/hybrid_medical_synthetic"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--relevance-threshold",
        type=int,
        default=2,
        help="relevance >= threshold is treated as relevant for recall/mrr/precision",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    version_map = json.loads(args.version_map.read_text(encoding="utf-8"))
    scenarios = sorted(version_map.keys())

    settings = load_settings()
    queries = _load_queries(args.queries_file)

    report: dict[str, object] = {
        "top_k": args.top_k,
        "relevance_threshold": args.relevance_threshold,
        "scenarios": {},
        "macro": {},
    }

    methods = ["vector", "keyword", "hybrid"]

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        for scenario in scenarios:
            version_id = str(version_map[scenario])
            qrels_file = args.qrels_dir / f"qrels.source_id.{scenario}.jsonl"

            qrels_rows = _load_qrels(qrels_file)
            source_to_chunk = _map_source_to_chunk(conn, version_id)
            qrels_by_query, unresolved_by_query = _prepare_chunk_qrels(
                qrels_rows, source_to_chunk
            )

            retriever = HybridRetriever(
                version_id=version_id,
                settings=settings,
                top_k=args.top_k,
            )

            metrics_by_method: dict[str, list[QueryMetrics]] = {m: [] for m in methods}

            if args.verbose:
                print(f"\n=== Scenario {scenario} | version_id={version_id} ===")

            for query in queries:
                if query.query_id not in qrels_by_query:
                    continue
                qrels_for_query = qrels_by_query[query.query_id]
                unresolved_count = unresolved_by_query.get(query.query_id, 0)

                for method in methods:
                    retrieved_ids, latency_ms = _run_method(
                        retriever, query, method, args.top_k
                    )
                    row = _evaluate_query(
                        query,
                        method,
                        retrieved_ids,
                        qrels_for_query,
                        args.relevance_threshold,
                        latency_ms,
                        unresolved_count,
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
                "qrels_file": str(qrels_file),
                "summary": summary,
                "details": {
                    method: [asdict(x) for x in rows]
                    for method, rows in metrics_by_method.items()
                }
                if args.output and args.verbose
                else None,
            }

            print(f"\nScenario {scenario} (version_id={version_id})")
            for method in methods:
                s = summary[method]
                print(
                    f"  {method:<6} MRR={s['mrr']:.4f} R@5={s['recall_at_5']:.4f} "
                    f"nDCG@5={s['ndcg_at_5']:.4f} Lat={s['latency_ms']:.2f}ms"
                )

    # Macro-average across scenarios (per method)
    macro: dict[str, dict[str, float]] = {}
    for method in methods:
        macro[method] = {
            key: _macro_mean(
                [
                    report["scenarios"][s]["summary"][method][key]
                    for s in scenarios
                    if report["scenarios"][s]
                ]
            )
            for key in [
                "mrr",
                "recall_at_5",
                "precision_at_3",
                "ndcg_at_5",
                "ndcg_at_10",
                "latency_ms",
            ]
        }

    report["macro"] = macro

    print("\nMacro-average across scenarios")
    for method in methods:
        s = macro[method]
        print(
            f"  {method:<6} MRR={s['mrr']:.4f} R@5={s['recall_at_5']:.4f} "
            f"nDCG@5={s['ndcg_at_5']:.4f} Lat={s['latency_ms']:.2f}ms"
        )

    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved report to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
