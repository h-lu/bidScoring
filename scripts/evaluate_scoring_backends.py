from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg

from bid_scoring.pipeline.application import (
    E2EPipelineService,
    E2ERunRequest,
    build_scoring_provider,
)
from bid_scoring.pipeline.application.service import PipelineService
from bid_scoring.pipeline.infrastructure.content_source import AutoContentSource
from bid_scoring.pipeline.infrastructure.postgres_repository import (
    PostgresPipelineRepository,
)

DEFAULT_CONTENT_LIST = Path("data/eval/scoring_compare/content_list.minimal.json")
DEFAULT_THRESHOLDS = Path("data/eval/scoring_compare/thresholds.json")
DEFAULT_BACKENDS = ["analyzer", "agent-mcp", "hybrid"]
_AGENT_DISABLE_ENV = "BID_SCORING_AGENT_MCP_DISABLE"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate analyzer/agent-mcp/hybrid scoring backends."
    )
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument("--content-list", type=Path, default=DEFAULT_CONTENT_LIST)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=list(DEFAULT_BACKENDS),
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["warranty", "delivery"],
        help="Scoring dimensions used for all backends.",
    )
    parser.add_argument(
        "--disable-agent-mcp",
        action="store_true",
        default=True,
        help="Force agent-mcp backend to fallback (stable CI mode).",
    )
    parser.add_argument(
        "--thresholds-file",
        type=Path,
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--fail-on-thresholds", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--project-id", default=str(uuid4()))
    parser.add_argument("--document-id", default=str(uuid4()))
    parser.add_argument("--document-title", default="Scoring Backend Eval")
    parser.add_argument("--bidder-name", default="Eval Bidder")
    parser.add_argument("--project-name", default="Eval Project")
    parser.add_argument("--hybrid-primary-weight", type=float, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.database_url:
        raise RuntimeError("DATABASE_URL is required (set env or pass --database-url).")

    thresholds = _load_json_file(args.thresholds_file)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}

    with psycopg.connect(args.database_url) as conn:
        for backend in args.backends:
            version_id = str(uuid4())
            payload = _run_single_backend(
                conn=conn,
                backend=backend,
                content_list_path=args.content_list,
                request_overrides={
                    "project_id": args.project_id,
                    "document_id": args.document_id,
                    "version_id": version_id,
                    "document_title": args.document_title,
                    "bidder_name": args.bidder_name,
                    "project_name": args.project_name,
                    "dimensions": list(args.dimensions),
                    "build_embeddings": False,
                    "scoring_backend": backend,
                    "hybrid_primary_weight": args.hybrid_primary_weight,
                },
                disable_agent_mcp=args.disable_agent_mcp,
            )
            payloads[backend] = payload
            summaries[backend] = summarize_backend_result(payload)

            if args.output_dir:
                out_path = args.output_dir / f"{backend}.json"
                out_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    violations = evaluate_thresholds(
        summaries=summaries,
        thresholds=thresholds if isinstance(thresholds, dict) else {},
    )
    report = {
        "backends": summaries,
        "violations": violations,
    }

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False))

    if args.fail_on_thresholds and violations:
        return 1
    return 0


def _run_single_backend(
    *,
    conn: Any,
    backend: str,
    content_list_path: Path,
    request_overrides: dict[str, Any],
    disable_agent_mcp: bool,
) -> dict[str, Any]:
    with _scoped_agent_disable(disable_agent_mcp):
        service = E2EPipelineService(
            content_source=AutoContentSource(),
            pipeline_service=PipelineService(
                repository=PostgresPipelineRepository(conn)
            ),
            index_builder=None,
            scoring_provider=build_scoring_provider(
                backend=backend,
                conn=conn,
                hybrid_primary_weight=request_overrides.get("hybrid_primary_weight"),
            ),
        )
        request = E2ERunRequest(
            project_id=request_overrides["project_id"],
            document_id=request_overrides["document_id"],
            version_id=request_overrides["version_id"],
            document_title=request_overrides["document_title"],
            bidder_name=request_overrides["bidder_name"],
            project_name=request_overrides["project_name"],
            dimensions=request_overrides["dimensions"],
            content_list_path=content_list_path,
            build_embeddings=request_overrides["build_embeddings"],
            scoring_backend=request_overrides["scoring_backend"],
            hybrid_primary_weight=request_overrides["hybrid_primary_weight"],
        )
        return service.run(request, conn=conn).as_dict()


def summarize_backend_result(payload: dict[str, Any]) -> dict[str, Any]:
    scoring = payload.get("scoring", {})
    traceability = payload.get("traceability", {})
    timings = payload.get("observability", {}).get("timings_ms", {})
    evidence_citations = scoring.get("evidence_citations", {})
    citation_total = 0
    if isinstance(evidence_citations, dict):
        for rows in evidence_citations.values():
            if isinstance(rows, list):
                citation_total += len(rows)

    return {
        "status": payload.get("status"),
        "overall_score": scoring.get("overall_score"),
        "risk_level": scoring.get("risk_level"),
        "chunks_analyzed": scoring.get("chunks_analyzed"),
        "citation_total": citation_total,
        "traceability_status": traceability.get("status"),
        "coverage_ratio": traceability.get("citation_coverage_ratio"),
        "warnings": list(payload.get("warnings", [])),
        "timings_ms": {
            "scoring": timings.get("scoring"),
            "total": timings.get("total"),
        },
    }


def evaluate_thresholds(
    *,
    summaries: dict[str, dict[str, Any]],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    per_backend = thresholds.get("per_backend", {})
    global_cfg = thresholds.get("global", {})
    relations = thresholds.get("relations", {})

    for backend, summary in summaries.items():
        cfg = per_backend.get(backend, {})
        if cfg:
            expected_status = cfg.get("expected_status")
            if expected_status and summary.get("status") != expected_status:
                violations.append(
                    {
                        "scope": backend,
                        "metric": "status",
                        "actual": summary.get("status"),
                        "expected": expected_status,
                    }
                )

            traceability_ok = cfg.get("allowed_traceability_status")
            if traceability_ok and summary.get("traceability_status") not in set(
                traceability_ok
            ):
                violations.append(
                    {
                        "scope": backend,
                        "metric": "traceability_status",
                        "actual": summary.get("traceability_status"),
                        "expected": traceability_ok,
                    }
                )

            min_citation_total = cfg.get("min_citation_total")
            if (
                isinstance(min_citation_total, int)
                and int(summary.get("citation_total", 0)) < min_citation_total
            ):
                violations.append(
                    {
                        "scope": backend,
                        "metric": "citation_total",
                        "actual": summary.get("citation_total"),
                        "expected_min": min_citation_total,
                    }
                )

            min_coverage = cfg.get("min_coverage_ratio")
            if isinstance(min_coverage, (int, float)):
                actual_cov = float(summary.get("coverage_ratio") or 0.0)
                if actual_cov < float(min_coverage):
                    violations.append(
                        {
                            "scope": backend,
                            "metric": "coverage_ratio",
                            "actual": actual_cov,
                            "expected_min": float(min_coverage),
                        }
                    )

        forbidden_warning_codes = global_cfg.get("forbidden_warning_codes", [])
        warning_set = set(summary.get("warnings", []))
        for code in forbidden_warning_codes:
            if code in warning_set:
                violations.append(
                    {
                        "scope": backend,
                        "metric": "warnings",
                        "actual": sorted(warning_set),
                        "forbidden": code,
                    }
                )

    if relations.get("hybrid_score_between_analyzer_and_agent"):
        analyzer = summaries.get("analyzer", {}).get("overall_score")
        hybrid = summaries.get("hybrid", {}).get("overall_score")
        agent = summaries.get("agent-mcp", {}).get("overall_score")
        if _all_number(analyzer, hybrid, agent):
            low = min(analyzer, agent)
            high = max(analyzer, agent)
            if not (low <= hybrid <= high):
                violations.append(
                    {
                        "scope": "relations",
                        "metric": "hybrid_score_between_analyzer_and_agent",
                        "actual": {
                            "analyzer": analyzer,
                            "hybrid": hybrid,
                            "agent-mcp": agent,
                        },
                    }
                )

    return violations


@contextmanager
def _scoped_agent_disable(disable: bool):
    previous = os.getenv(_AGENT_DISABLE_ENV)
    try:
        if disable:
            os.environ[_AGENT_DISABLE_ENV] = "1"
        yield
    finally:
        if previous is None:
            os.environ.pop(_AGENT_DISABLE_ENV, None)
        else:
            os.environ[_AGENT_DISABLE_ENV] = previous


def _all_number(*values: Any) -> bool:
    return all(isinstance(v, (int, float)) for v in values)


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
