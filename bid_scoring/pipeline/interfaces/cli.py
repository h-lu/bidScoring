from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol, Sequence

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.pipeline.application import (
    E2EPipelineService,
    E2ERunRequest,
    QuestionContextResolver,
    build_scoring_provider,
)
from bid_scoring.pipeline.application.service import PipelineService
from bid_scoring.pipeline.infrastructure.content_source import AutoContentSource
from bid_scoring.pipeline.infrastructure.index_builder import IndexBuilder
from bid_scoring.pipeline.infrastructure.postgres_repository import (
    PostgresPipelineRepository,
)

DEFAULT_PROD_QUESTION_PACK = "cn_medical_v1"
DEFAULT_PROD_QUESTION_OVERLAY = "strict_traceability"


class QuestionContextResolverLike(Protocol):
    def resolve(
        self,
        *,
        question_pack: str | None,
        question_overlay: str | None,
        requested_dimensions: list[str] | None,
    ) -> Any: ...


def _hybrid_weight_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hybrid weight must be a float within [0, 1]"
        ) from exc
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("hybrid weight must be within [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bid-pipeline", description="Evidence-first pipeline CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-id", required=True)
    common.add_argument("--document-id", required=True)
    common.add_argument("--version-id", required=True)
    common.add_argument("--document-title", default="untitled")
    common.add_argument("--source-uri")
    common.add_argument("--parser-version", default="pipeline-v1")

    ingest = sub.add_parser(
        "ingest-content-list",
        help="Ingest MinerU content_list.json",
        parents=[common],
    )
    ingest.add_argument("--content-list", required=True)

    run = sub.add_parser(
        "run-e2e",
        help="Run content-load -> ingest -> embeddings -> scoring",
        parents=[common],
    )
    source_group = run.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--context-list",
        "--context-json",
        "--content-list",
        dest="content_list",
        help="Input pre-parsed context/content JSON directly",
    )
    source_group.add_argument("--mineru-output-dir", help=argparse.SUPPRESS)
    source_group.add_argument(
        "--pdf-path",
        help="Parse PDF directly through MinerU and continue e2e pipeline",
    )
    run.add_argument(
        "--mineru-parser",
        choices=["auto", "cli", "api"],
        default="auto",
        help="Parser backend for --pdf-path (default: auto)",
    )
    run.add_argument("--bidder-name", default="Unknown")
    run.add_argument("--project-name", default="Unknown Project")
    run.add_argument("--dimensions", nargs="+")
    run.add_argument("--scoring-backend", choices=["analyzer", "agent-mcp", "hybrid"], default="hybrid", help=argparse.SUPPRESS)
    run.add_argument(
        "--hybrid-primary-weight",
        type=_hybrid_weight_arg,
        help=argparse.SUPPRESS,
    )
    run.add_argument("--skip-embeddings", action="store_true", help=argparse.SUPPRESS)
    run.add_argument(
        "--question-pack",
        default=DEFAULT_PROD_QUESTION_PACK,
        help=f"Question pack (default: {DEFAULT_PROD_QUESTION_PACK})",
    )
    run.add_argument(
        "--question-overlay",
        default=DEFAULT_PROD_QUESTION_OVERLAY,
        help=f"Question overlay strategy (default: {DEFAULT_PROD_QUESTION_OVERLAY})",
    )
    run.add_argument("--output")
    return parser


def main(
    argv: Sequence[str] | None = None,
    service: Any | None = None,
    question_context_resolver: QuestionContextResolverLike | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "ingest-content-list":
        return _run_ingest(args, service=service)
    if args.command == "run-e2e":
        return _run_e2e(
            args,
            service=service,
            question_context_resolver=question_context_resolver,
        )

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _run_ingest(args: argparse.Namespace, service: Any | None = None) -> int:
    path = Path(args.content_list)
    content_list = json.loads(path.read_text(encoding="utf-8"))

    if service is None:
        settings = load_settings()
        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            repository = PostgresPipelineRepository(conn)
            runtime_service = PipelineService(repository=repository)
            summary = runtime_service.ingest_content_list(
                project_id=args.project_id,
                document_id=args.document_id,
                version_id=args.version_id,
                content_list=content_list,
                document_title=args.document_title,
                source_uri=args.source_uri or str(path),
                parser_version=args.parser_version,
            )
    else:
        summary = service.ingest_content_list(
            project_id=args.project_id,
            document_id=args.document_id,
            version_id=args.version_id,
            content_list=content_list,
            document_title=args.document_title,
            source_uri=args.source_uri or str(path),
            parser_version=args.parser_version,
        )

    print(
        json.dumps(
            {
                "status": summary["status"]
                if isinstance(summary, dict)
                else summary.status,
                "chunks_imported": summary["chunks_imported"]
                if isinstance(summary, dict)
                else summary.chunks_imported,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_e2e(
    args: argparse.Namespace,
    service: Any | None = None,
    question_context_resolver: QuestionContextResolverLike | None = None,
) -> int:
    resolver = question_context_resolver or QuestionContextResolver()
    resolved = resolver.resolve(
        question_pack=args.question_pack,
        question_overlay=args.question_overlay,
        requested_dimensions=list(args.dimensions) if args.dimensions else None,
    )

    request = E2ERunRequest(
        project_id=args.project_id,
        document_id=args.document_id,
        version_id=args.version_id,
        document_title=args.document_title,
        source_uri=args.source_uri,
        parser_version=args.parser_version,
        bidder_name=args.bidder_name,
        project_name=args.project_name,
        dimensions=resolved.dimensions,
        content_list_path=Path(args.content_list) if args.content_list else None,
        mineru_output_dir=Path(args.mineru_output_dir)
        if args.mineru_output_dir
        else None,
        pdf_path=Path(args.pdf_path) if args.pdf_path else None,
        mineru_parser=args.mineru_parser,
        build_embeddings=not args.skip_embeddings,
        scoring_backend=args.scoring_backend,
        hybrid_primary_weight=args.hybrid_primary_weight,
        question_context=resolved.question_context,
        question_pack=args.question_pack,
        question_overlay=args.question_overlay,
    )

    if service is not None:
        result = service.run(request)
        payload = result if isinstance(result, dict) else result.as_dict()
        _emit_payload(payload, output_path=args.output)
        return 0

    settings = load_settings()
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        scoring_provider = build_scoring_provider(
            args.scoring_backend,
            conn,
            hybrid_primary_weight=request.hybrid_primary_weight,
        )
        runtime_service = E2EPipelineService(
            content_source=AutoContentSource(),
            pipeline_service=PipelineService(
                repository=PostgresPipelineRepository(conn)
            ),
            index_builder=IndexBuilder(),
            scoring_provider=scoring_provider,
        )
        result = runtime_service.run(request, conn=conn)
        _emit_payload(result.as_dict(), output_path=args.output)
        return 0


def _emit_payload(payload: dict[str, Any], output_path: str | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False)
    if output_path:
        Path(output_path).write_text(rendered, encoding="utf-8")
        print(
            json.dumps({"status": "written", "output": output_path}, ensure_ascii=False)
        )
        return
    print(rendered)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
