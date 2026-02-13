from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import psycopg

from bid_scoring.config import load_settings
from bid_scoring.pipeline.application.service import PipelineService
from bid_scoring.pipeline.infrastructure.postgres_repository import (
    PostgresPipelineRepository,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bid-pipeline", description="Evidence-first pipeline CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest-content-list", help="Ingest MinerU content_list.json"
    )
    ingest.add_argument("--content-list", required=True)
    ingest.add_argument("--project-id", required=True)
    ingest.add_argument("--document-id", required=True)
    ingest.add_argument("--version-id", required=True)
    ingest.add_argument("--document-title", default="untitled")
    ingest.add_argument("--source-uri")
    ingest.add_argument("--parser-version", default="pipeline-v1")
    return parser


def main(
    argv: Sequence[str] | None = None, service: PipelineService | None = None
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "ingest-content-list":
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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
