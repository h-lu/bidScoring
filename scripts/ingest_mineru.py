import json
import psycopg
import argparse
from pathlib import Path
from bid_scoring.config import load_settings
from bid_scoring.ingest import ingest_content_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--document-title", default="untitled")
    parser.add_argument("--source-type", default="mineru")
    parser.add_argument("--source-uri")
    parser.add_argument("--parser-version")
    parser.add_argument("--status", default="ready")
    args = parser.parse_args()
    dsn = load_settings()["DATABASE_URL"]
    path = Path(args.path)
    content_list = json.loads(path.read_text(encoding="utf-8"))
    with psycopg.connect(dsn) as conn:
        ingest_content_list(
            conn,
            args.project_id,
            args.document_id,
            args.version_id,
            content_list,
            document_title=args.document_title,
            source_type=args.source_type,
            source_uri=args.source_uri,
            parser_version=args.parser_version,
            status=args.status,
        )


if __name__ == "__main__":
    main()
