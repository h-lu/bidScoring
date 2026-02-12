from __future__ import annotations

import hashlib
from typing import Any

from bid_scoring.ingest import ingest_content_list


class PostgresPipelineRepository:
    """Persistence adapter backed by psycopg connections."""

    def __init__(self, conn):
        self.conn = conn

    def persist_content_list(
        self,
        *,
        project_id: str,
        document_id: str,
        version_id: str,
        content_list: list[dict[str, Any]],
        document_title: str,
        source_uri: str | None,
        parser_version: str | None,
    ) -> dict[str, Any]:
        return ingest_content_list(
            conn=self.conn,
            project_id=project_id,
            document_id=document_id,
            version_id=version_id,
            content_list=content_list,
            document_title=document_title,
            source_type="mineru",
            source_uri=source_uri,
            parser_version=parser_version,
            status="ready",
        )

    def record_source_artifact(
        self,
        *,
        version_id: str,
        source_uri: str,
        parser_version: str | None,
        file_sha256: str | None = None,
        page_count: int | None = None,
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_artifacts (
                    artifact_id, version_id, source_uri, file_sha256,
                    parser_version, page_count
                )
                VALUES (gen_random_uuid(), %s, %s, %s, %s, %s)
                ON CONFLICT (version_id, source_uri) DO UPDATE SET
                    file_sha256 = EXCLUDED.file_sha256,
                    parser_version = EXCLUDED.parser_version,
                    page_count = EXCLUDED.page_count
                """,
                (
                    version_id,
                    source_uri,
                    file_sha256,
                    parser_version,
                    page_count,
                ),
            )
        self.conn.commit()

    @staticmethod
    def hash_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

