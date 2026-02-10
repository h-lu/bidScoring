"""Database query utilities for E2E tests."""

from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


def query_chunks_with_bbox(conn, version_id: str) -> list[dict[str, Any]]:
    """Query chunks with bbox for a version.

    Args:
        conn: Database connection
        version_id: Version UUID

    Returns:
        List of chunks with bbox
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, page_idx, bbox, text_raw, element_type
            FROM chunks
            WHERE version_id = %s
            AND bbox IS NOT NULL
            """,
            (version_id,),
        )
        return cur.fetchall()


def query_chunk_with_embedding(conn, version_id: str) -> dict[str, Any] | None:
    """Query a chunk with embedding for verification.

    Args:
        conn: Database connection
        version_id: Version UUID

    Returns:
        Chunk with embedding or None
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, text_raw, embedding
            FROM chunks
            WHERE version_id = %s
            AND embedding IS NOT NULL
            LIMIT 1
            """,
            (version_id,),
        )
        return cur.fetchone()


def query_all_chunks_for_export(conn, version_id: str) -> list[dict[str, Any]]:
    """Query all chunks for export.

    Args:
        conn: Database connection
        version_id: Version UUID

    Returns:
        List of all chunks
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT chunk_id, version_id, project_id, page_idx, bbox,
                   element_type, text_raw, text_hash
            FROM chunks
            WHERE version_id = %s
            ORDER BY page_idx, chunk_id
            """,
            (version_id,),
        )
        return cur.fetchall()


def query_content_units_for_export(conn, version_id: str) -> list[dict[str, Any]]:
    """Query content units for export.

    Args:
        conn: Database connection
        version_id: Version UUID

    Returns:
        List of content units
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT unit_id, version_id, unit_index, text_raw, text_norm,
                   char_count, anchor_json, unit_hash
            FROM content_units
            WHERE version_id = %s
            ORDER BY unit_index
            """,
            (version_id,),
        )
        return cur.fetchall()


def verify_minio_files_registered(conn, version_id: str) -> int:
    """Verify files are registered in database.

    Args:
        conn: Database connection
        version_id: Version UUID

    Returns:
        Count of registered files
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT COUNT(*) as count
            FROM document_files
            WHERE version_id = %s
            """,
            (version_id,),
        )
        result = cur.fetchone()
        return result["count"] if result else 0
