"""Discovery operations for retrieval MCP server."""

from __future__ import annotations

from typing import Any, Dict

from bid_scoring.config import load_settings
from mcp_servers.retrieval.validation import validate_positive_int, validate_version_id


def list_available_versions(
    project_id: str | None = None,
    include_stats: bool = True,
) -> Dict[str, Any]:
    """List available document versions with optional project filter.

    Use this to discover what bidding documents are available for analysis.
    Returns version IDs, bidder names, and document statistics.

    Args:
        project_id: Optional project UUID to filter versions.
        include_stats: Whether to include chunk/page counts for each version.

    Returns:
        Dictionary with list of versions and their metadata.
    """
    import psycopg

    settings = load_settings()
    versions = []

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            if project_id:
                cur.execute(
                    """
                    SELECT v.version_id, v.doc_id, d.title, p.name as project_name,
                           v.source_uri, v.created_at
                    FROM document_versions v
                    JOIN documents d ON v.doc_id = d.doc_id
                    JOIN projects p ON d.project_id = p.project_id
                    WHERE d.project_id = %s
                    ORDER BY v.created_at DESC
                    """,
                    (project_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT v.version_id, v.doc_id, d.title, p.name as project_name,
                           v.source_uri, v.created_at
                    FROM document_versions v
                    JOIN documents d ON v.doc_id = d.doc_id
                    JOIN projects p ON d.project_id = p.project_id
                    ORDER BY v.created_at DESC
                    LIMIT 100
                    """
                )

            rows = cur.fetchall()

            for row in rows:
                version_info = {
                    "version_id": str(row[0]),
                    "doc_id": str(row[1]),
                    "title": row[2],
                    "project_name": row[3],
                    "source_uri": row[4],
                    "created_at": row[5].isoformat() if row[5] else None,
                }

                if include_stats:
                    # Get chunk count
                    cur.execute(
                        "SELECT COUNT(*) FROM chunks WHERE version_id = %s", (row[0],)
                    )
                    version_info["chunk_count"] = cur.fetchone()[0]

                    # Get page count
                    cur.execute(
                        """SELECT COUNT(DISTINCT page_idx) FROM chunks 
                           WHERE version_id = %s AND page_idx IS NOT NULL""",
                        (row[0],),
                    )
                    version_info["page_count"] = cur.fetchone()[0]

                    # Get content unit count (v0.2)
                    cur.execute(
                        "SELECT COUNT(*) FROM content_units WHERE version_id = %s",
                        (row[0],),
                    )
                    version_info["unit_count"] = cur.fetchone()[0]

                versions.append(version_info)

    return {
        "count": len(versions),
        "versions": versions,
    }


def get_document_outline(
    version_id: str,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Get document structure outline (table of contents) for navigation.

    This helps Claude Code understand document organization and locate
    specific sections like "售后服务" or "技术参数".

    Args:
        version_id: UUID of the document version.
        max_depth: Maximum hierarchy depth to return (0=root only).

    Returns:
        Hierarchical outline with section titles, page ranges, and node IDs.
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)
    max_depth = validate_positive_int(max_depth, "max_depth", max_value=10)

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # First check if hierarchical nodes exist
            cur.execute(
                """
                SELECT node_id, parent_id, level, node_type, content,
                       metadata->>'heading' as heading, 
                       metadata->'page_range' as page_range
                FROM hierarchical_nodes
                WHERE version_id = %s AND level <= %s
                ORDER BY level, node_id
                """,
                (version_id, max_depth),
            )

            nodes = cur.fetchall()

            if nodes:
                # Use hierarchical nodes
                outline = []
                for row in nodes:
                    node_info = {
                        "node_id": str(row[0]),
                        "parent_id": str(row[1]) if row[1] else None,
                        "level": row[2],
                        "node_type": row[3],
                        "heading": row[5] or (row[4][:50] if row[4] else None),
                        "page_range": row[6],
                    }
                    outline.append(node_info)

                return {
                    "version_id": version_id,
                    "source": "hierarchical_nodes",
                    "outline": outline,
                }

            # Fallback: use chunks to infer structure
            cur.execute(
                """
                SELECT DISTINCT page_idx, element_type, text_level,
                       CASE WHEN text_level IS NOT NULL AND text_level <= 2 
                            THEN text_raw ELSE NULL END as heading
                FROM chunks
                WHERE version_id = %s AND page_idx IS NOT NULL
                ORDER BY page_idx, text_level NULLS LAST
                """,
                (version_id,),
            )

            chunks = cur.fetchall()
            outline = []

            for row in chunks:
                page_idx, element_type, text_level, heading = row

                if heading and len(heading.strip()) > 0:
                    outline.append(
                        {
                            "page_idx": page_idx,
                            "element_type": element_type,
                            "level": text_level or 0,
                            "heading": heading.strip()[:100],
                        }
                    )

            return {
                "version_id": version_id,
                "source": "chunks",
                "outline": outline,
            }


def get_page_metadata(
    version_id: str,
    page_idx: int | list[int] | None = None,
    include_elements: bool = True,
) -> Dict[str, Any]:
    """Get metadata for specific pages or all pages in a document.

    Useful for understanding page composition before diving into content.

    Args:
        version_id: UUID of the document version.
        page_idx: Specific page number(s), or None for all pages.
        include_elements: Whether to count tables, images, text blocks per page.

    Returns:
        Page dimensions, element counts, and coordinate system info.
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Get document_pages info
            if page_idx is None:
                cur.execute(
                    """
                    SELECT page_idx, page_w, page_h, coord_sys
                    FROM document_pages
                    WHERE version_id = %s
                    ORDER BY page_idx
                    """,
                    (version_id,),
                )
            elif isinstance(page_idx, int):
                cur.execute(
                    """
                    SELECT page_idx, page_w, page_h, coord_sys
                    FROM document_pages
                    WHERE version_id = %s AND page_idx = %s
                    """,
                    (version_id, page_idx),
                )
            else:  # list
                cur.execute(
                    """
                    SELECT page_idx, page_w, page_h, coord_sys
                    FROM document_pages
                    WHERE version_id = %s AND page_idx = ANY(%s)
                    ORDER BY page_idx
                    """,
                    (version_id, page_idx),
                )

            pages = cur.fetchall()

            page_list = []
            for row in pages:
                page_info = {
                    "page_idx": row[0],
                    "width": float(row[1]) if row[1] else None,
                    "height": float(row[2]) if row[2] else None,
                    "coord_system": row[3],
                }

                if include_elements:
                    # Count elements by type
                    cur.execute(
                        """
                        SELECT element_type, COUNT(*)
                        FROM chunks
                        WHERE version_id = %s AND page_idx = %s
                        GROUP BY element_type
                        """,
                        (version_id, row[0]),
                    )
                    element_counts = {t: c for t, c in cur.fetchall()}
                    page_info["element_counts"] = element_counts

                    # Check for tables
                    cur.execute(
                        """
                        SELECT COUNT(*) FROM chunks
                        WHERE version_id = %s AND page_idx = %s
                        AND (table_body IS NOT NULL OR element_type = 'table')
                        """,
                        (version_id, row[0]),
                    )
                    page_info["has_table"] = cur.fetchone()[0] > 0

                    # Check for images
                    cur.execute(
                        """
                        SELECT COUNT(*) FROM chunks
                        WHERE version_id = %s AND page_idx = %s
                        AND img_path IS NOT NULL
                        """,
                        (version_id, row[0]),
                    )
                    page_info["has_images"] = cur.fetchone()[0] > 0

                page_list.append(page_info)

            return {
                "version_id": version_id,
                "pages": page_list,
                "total_pages": len(page_list),
            }
