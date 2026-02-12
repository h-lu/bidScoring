#!/usr/bin/env python3
"""Evaluate hybrid retrieval with graded golden qrels.

This script evaluates vector-only, keyword-only, and hybrid retrieval
against manually labeled graded relevance judgments.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bid_scoring.config import load_settings
from bid_scoring.hybrid_retrieval import HybridRetriever
from bid_scoring.retrieval.evaluation_gate import check_metric_thresholds


@dataclass
class QueryDef:
    query_id: str
    query: str
    keywords: list[str]
    query_type: str


@dataclass
class QueryMetrics:
    query_id: str
    query_type: str
    method: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mrr: float
    ndcg_at_3: float
    ndcg_at_5: float
    ndcg_at_10: float
    latency_ms: float
    unresolved_qrels: int


def _load_queries(path: Path) -> list[QueryDef]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    queries: list[QueryDef] = []
    for row in rows:
        queries.append(
            QueryDef(
                query_id=row["query_id"],
                query=row["query"],
                keywords=row.get("keywords", []),
                query_type=row.get("query_type", "unknown"),
            )
        )
    return queries


def _load_qrels(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _map_source_to_chunk(conn, version_id: str) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_id, chunk_id::text
            FROM chunks
            WHERE version_id = %s
            """,
            (version_id,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def _prepare_chunk_qrels(
    qrels_rows: list[dict],
    source_to_chunk: dict[str, str],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    by_query: dict[str, dict[str, int]] = {}
    unresolved: dict[str, int] = {}

    for row in qrels_rows:
        query_id = row["query_id"]
        source_id = row["source_id"]
        relevance = int(row["relevance"])

        chunk_id = source_to_chunk.get(source_id)
        if chunk_id is None:
            unresolved[query_id] = unresolved.get(query_id, 0) + 1
            continue

        per_query = by_query.setdefault(query_id, {})
        # Keep maximum relevance if duplicates exist
        per_query[chunk_id] = max(per_query.get(chunk_id, 0), relevance)

    return by_query, unresolved


def _dcg(relevances: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += (2**rel - 1) / math.log2(i + 2)
    return score


def _ndcg(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gains = [qrels.get(cid, 0) for cid in retrieved_ids[:k]]
    dcg = _dcg(gains, k)
    ideal = sorted(qrels.values(), reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def _precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / k


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def _run_method(
    retriever: HybridRetriever,
    query: QueryDef,
    method: str,
    top_k: int,
) -> tuple[list[str], float]:
    start = time.perf_counter()

    if method == "vector":
        results = retriever._vector_search(query.query)
        retrieved = [row[0] for row in results[:top_k]]
    elif method == "keyword":
        keyword_terms = query.keywords or retriever.extract_keywords_from_query(
            query.query
        )
        results = retriever._keyword_search_fulltext(
            keyword_terms, use_or_semantic=True
        )
        retrieved = [row[0] for row in results[:top_k]]
    elif method == "hybrid":
        results = retriever.retrieve(query.query, keywords=query.keywords)
        retrieved = [row.chunk_id for row in results[:top_k]]
    else:
        raise ValueError(f"unknown method: {method}")

    latency_ms = (time.perf_counter() - start) * 1000
    return retrieved, latency_ms


def _evaluate_query(
    query: QueryDef,
    method: str,
    retrieved_ids: list[str],
    qrels_for_query: dict[str, int],
    relevance_threshold: int,
    latency_ms: float,
    unresolved_count: int,
) -> QueryMetrics:
    relevant_ids = {
        cid for cid, rel in qrels_for_query.items() if rel >= relevance_threshold
    }

    return QueryMetrics(
        query_id=query.query_id,
        query_type=query.query_type,
        method=method,
        recall_at_1=_recall_at_k(retrieved_ids, relevant_ids, 1),
        recall_at_3=_recall_at_k(retrieved_ids, relevant_ids, 3),
        recall_at_5=_recall_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_10=_recall_at_k(retrieved_ids, relevant_ids, 10),
        precision_at_1=_precision_at_k(retrieved_ids, relevant_ids, 1),
        precision_at_3=_precision_at_k(retrieved_ids, relevant_ids, 3),
        precision_at_5=_precision_at_k(retrieved_ids, relevant_ids, 5),
        mrr=_mrr(retrieved_ids, relevant_ids),
        ndcg_at_3=_ndcg(retrieved_ids, qrels_for_query, 3),
        ndcg_at_5=_ndcg(retrieved_ids, qrels_for_query, 5),
        ndcg_at_10=_ndcg(retrieved_ids, qrels_for_query, 10),
        latency_ms=latency_ms,
        unresolved_qrels=unresolved_count,
    )


def _print_summary(metrics_by_method: dict[str, list[QueryMetrics]]) -> None:
    print("\n" + "=" * 96)
    print("黄金标准检索评估报告 (Graded Qrels)")
    print("=" * 96)
    print(
        f"{'Method':<12} {'MRR':>8} {'R@5':>8} {'P@3':>8} {'nDCG@5':>10} {'nDCG@10':>10} {'Lat(ms)':>10}"
    )
    print("-" * 96)

    for method, rows in metrics_by_method.items():
        if not rows:
            continue
        print(
            f"{method:<12}"
            f"{statistics.mean([x.mrr for x in rows]):>8.4f}"
            f"{statistics.mean([x.recall_at_5 for x in rows]):>8.4f}"
            f"{statistics.mean([x.precision_at_3 for x in rows]):>8.4f}"
            f"{statistics.mean([x.ndcg_at_5 for x in rows]):>10.4f}"
            f"{statistics.mean([x.ndcg_at_10 for x in rows]):>10.4f}"
            f"{statistics.mean([x.latency_ms for x in rows]):>10.2f}"
        )

    print("=" * 96)


def _print_by_query_type(metrics_by_method: dict[str, list[QueryMetrics]]) -> None:
    methods = [m for m in ["vector", "keyword", "hybrid"] if m in metrics_by_method]
    if not methods:
        return

    query_types = sorted(
        {
            row.query_type
            for method in methods
            for row in metrics_by_method.get(method, [])
        }
    )

    print("\n按 query_type 分析（nDCG@5 / MRR）")
    print("-" * 72)
    for qtype in query_types:
        parts = []
        for method in methods:
            rows = [r for r in metrics_by_method[method] if r.query_type == qtype]
            if not rows:
                continue
            parts.append(
                f"{method}: {statistics.mean([r.ndcg_at_5 for r in rows]):.3f}/"
                f"{statistics.mean([r.mrr for r in rows]):.3f}"
            )
        if parts:
            print(f"{qtype:<24} {' | '.join(parts)}")


def _build_summary(metrics_by_method: dict[str, list[QueryMetrics]]) -> dict[str, dict]:
    return {
        method: {
            "mrr": statistics.mean([x.mrr for x in rows]) if rows else 0.0,
            "recall_at_5": statistics.mean([x.recall_at_5 for x in rows])
            if rows
            else 0.0,
            "precision_at_3": statistics.mean([x.precision_at_3 for x in rows])
            if rows
            else 0.0,
            "ndcg_at_5": statistics.mean([x.ndcg_at_5 for x in rows]) if rows else 0.0,
            "ndcg_at_10": statistics.mean([x.ndcg_at_10 for x in rows])
            if rows
            else 0.0,
            "latency_ms": statistics.mean([x.latency_ms for x in rows]) if rows else 0.0,
        }
        for method, rows in metrics_by_method.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid search with golden qrels"
    )
    parser.add_argument(
        "--version-id",
        type=str,
        required=True,
        help="document version id in DB",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("data/eval/hybrid_medical_synthetic/queries.json"),
    )
    parser.add_argument(
        "--qrels-file",
        type=Path,
        default=Path("data/eval/hybrid_medical_synthetic/qrels.source_id.jsonl"),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--relevance-threshold",
        type=int,
        default=2,
        help="relevance >= threshold is treated as relevant for recall/mrr/precision",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-query details",
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        help="optional JSON thresholds, e.g. {'hybrid': {'mrr': 0.6, 'recall_at_5': 0.8}}",
    )
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="exit with non-zero code when threshold violations exist",
    )
    args = parser.parse_args()

    settings = load_settings()
    queries = _load_queries(args.queries_file)
    qrels_rows = _load_qrels(args.qrels_file)

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM document_versions WHERE version_id = %s",
                (args.version_id,),
            )
            if cur.fetchone()[0] == 0:
                print(f"Error: version_id {args.version_id} not found")
                return 1

        source_to_chunk = _map_source_to_chunk(conn, args.version_id)

    qrels_by_query, unresolved_by_query = _prepare_chunk_qrels(
        qrels_rows, source_to_chunk
    )

    retriever = HybridRetriever(
        version_id=args.version_id,
        settings=settings,
        top_k=args.top_k,
    )

    methods = ["vector", "keyword", "hybrid"]
    metrics_by_method: dict[str, list[QueryMetrics]] = {m: [] for m in methods}

    print("Start evaluation with golden qrels")
    print(f"version_id: {args.version_id}")
    print(
        f"queries: {len(queries)}, top_k: {args.top_k}, relevance_threshold: {args.relevance_threshold}"
    )

    for i, query in enumerate(queries, start=1):
        if query.query_id not in qrels_by_query:
            print(f"[skip] {query.query_id} has no resolved qrels")
            continue

        qrels_for_query = qrels_by_query[query.query_id]
        unresolved_count = unresolved_by_query.get(query.query_id, 0)

        if unresolved_count > 0:
            print(f"[warn] {query.query_id} unresolved qrels: {unresolved_count}")

        print(
            f"  evaluating {i:02d}/{len(queries)} {query.query_id}: {query.query[:24]}..."
        )
        for method in methods:
            retrieved_ids, latency_ms = _run_method(
                retriever,
                query,
                method,
                args.top_k,
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

            if args.verbose:
                print(
                    f"    - {method:<7} MRR={row.mrr:.3f} R@5={row.recall_at_5:.3f} "
                    f"nDCG@5={row.ndcg_at_5:.3f} Lat={row.latency_ms:.1f}ms"
                )

    retriever.close()

    _print_summary(metrics_by_method)
    _print_by_query_type(metrics_by_method)
    summary = _build_summary(metrics_by_method)

    threshold_violations = []
    if args.thresholds_file:
        thresholds = json.loads(args.thresholds_file.read_text(encoding="utf-8"))
        threshold_violations = check_metric_thresholds(summary, thresholds)
        if threshold_violations:
            print("\nThreshold gate violations:")
            for violation in threshold_violations:
                print(
                    "  - "
                    f"{violation.method}.{violation.metric}: "
                    f"actual={violation.actual:.4f}, minimum={violation.minimum:.4f}"
                )
        else:
            print("\nThreshold gate passed.")

    if args.output:
        payload = {
            "version_id": args.version_id,
            "top_k": args.top_k,
            "relevance_threshold": args.relevance_threshold,
            "summary": summary,
            "threshold_violations": [
                {
                    "method": v.method,
                    "metric": v.metric,
                    "actual": v.actual,
                    "minimum": v.minimum,
                }
                for v in threshold_violations
            ],
            "details": {
                method: [asdict(row) for row in rows]
                for method, rows in metrics_by_method.items()
            },
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nSaved report to: {args.output}")

    if threshold_violations and args.fail_on_thresholds:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
