"""File Registry Module for managing document_files table.

This module provides CRUD operations for tracking files stored in MinIO.
"""

import logging
import uuid
from typing import Any

from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class FileRegistry:
    """Registry for document files stored in MinIO.

    Tracks metadata for all files associated with a document version,
    including original PDFs, parsed content, images, and JSON data.
    """

    def __init__(self, conn):
        """Initialize FileRegistry with a database connection.

        Args:
            conn: psycopg connection object
        """
        self.conn = conn

    def register_file(
        self,
        version_id: str,
        file_type: str,
        file_path: str,
        file_name: str,
        file_size: int | None = None,
        content_type: str | None = None,
        etag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Register a file record in the database.

        Creates a new file record or updates an existing one if the
        (version_id, file_path) combination already exists.

        Args:
            version_id: Document version UUID
            file_type: File type (original_pdf, parsed_zip, markdown, image, json)
            file_path: MinIO object key
            file_name: Original file name
            file_size: File size in bytes
            content_type: MIME type
            etag: MinIO ETag (MD5 hash)
            metadata: Additional metadata as JSON

        Returns:
            The file_id of the registered file
        """
        from psycopg.types.json import Jsonb

        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                INSERT INTO document_files (
                    file_id, version_id, file_type, file_path, file_name,
                    file_size, content_type, etag, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (version_id, file_path) DO UPDATE SET
                    file_type = EXCLUDED.file_type,
                    file_name = EXCLUDED.file_name,
                    file_size = EXCLUDED.file_size,
                    content_type = EXCLUDED.content_type,
                    etag = EXCLUDED.etag,
                    metadata = EXCLUDED.metadata
                RETURNING file_id
                """,
                (
                    str(uuid.uuid4()),
                    version_id,
                    file_type,
                    file_path,
                    file_name,
                    file_size,
                    content_type,
                    etag,
                    Jsonb(metadata) if metadata else None,
                ),
            )
            result = cur.fetchone()
            self.conn.commit()
            return result["file_id"] if result else None

    def get_files_by_version(self, version_id: str) -> list[dict[str, Any]]:
        """Get all files associated with a document version.

        Args:
            version_id: Document version UUID

        Returns:
            List of file records
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM document_files
                WHERE version_id = %s
                ORDER BY created_at
                """,
                (version_id,),
            )
            return cur.fetchall()

    def get_original_pdf(self, version_id: str) -> dict[str, Any] | None:
        """Get the original PDF file for a document version.

        Args:
            version_id: Document version UUID

        Returns:
            File record or None if not found
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM document_files
                WHERE version_id = %s AND file_type = 'original_pdf'
                LIMIT 1
                """,
                (version_id,),
            )
            return cur.fetchone()

    def get_files_by_type(self, version_id: str, file_type: str) -> list[dict[str, Any]]:
        """Get files of a specific type for a document version.

        Args:
            version_id: Document version UUID
            file_type: File type to filter by

        Returns:
            List of file records
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT * FROM document_files
                WHERE version_id = %s AND file_type = %s
                ORDER BY created_at
                """,
                (version_id, file_type),
            )
            return cur.fetchall()

    def delete_file(self, file_id: str) -> bool:
        """Delete a file record.

        Args:
            file_id: File UUID to delete

        Returns:
            True if deleted, False if not found
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_files WHERE file_id = %s",
                (file_id,),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def get_file_by_id(self, file_id: str) -> dict[str, Any] | None:
        """Get a file record by its ID.

        Args:
            file_id: File UUID

        Returns:
            File record or None if not found
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT * FROM document_files WHERE file_id = %s",
                (file_id,),
            )
            return cur.fetchone()
