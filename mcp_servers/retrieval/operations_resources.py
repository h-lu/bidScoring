"""Resource handlers for retrieval MCP server."""

from __future__ import annotations

import json
from typing import Any

from bid_scoring.config import load_settings
from mcp_servers.retrieval.validation import (
    validate_chunk_id,
    validate_unit_id,
    validate_version_id,
)


def get_unit_evidence_resource(unit_id: str) -> str:
    """Get precise evidence content by unit ID.

    Use this resource when you need to:
    - Cite specific evidence in bid analysis
    - Verify exact wording of a commitment
    - Reference audit-grade evidence with hash verification

    URI: evidence://unit/{unit_id}

    Args:
        unit_id: UUID of the content unit.

    Returns:
        JSON string with unit content and verification metadata.
    """
    import psycopg

    unit_id = validate_unit_id(unit_id)
    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version_id, unit_index, unit_type, text_raw,
                       anchor_json, unit_hash, source_element_id
                FROM content_units
                WHERE unit_id = %s
                """,
                (unit_id,),
            )

            row = cur.fetchone()
            if not row:
                return json.dumps(
                    {
                        "error": f"Unit {unit_id} not found",
                        "unit_id": unit_id,
                    }
                )

            (
                version_id,
                unit_index,
                unit_type,
                text_raw,
                anchor_json,
                unit_hash,
                source_element_id,
            ) = row

            result = {
                "unit_id": unit_id,
                "version_id": str(version_id),
                "unit_index": unit_index,
                "unit_type": unit_type,
                "text": text_raw,
                "anchor": anchor_json,
                "unit_hash": unit_hash,
                "source_element_id": source_element_id,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)


def get_chunk_evidence_resource(chunk_id: str) -> str:
    """Get chunk content with surrounding context.

    Use this resource when you need to:
    - Reference a specific paragraph or section
    - View chunk content without additional API calls
    - Maintain context across multiple turns

    URI: evidence://chunk/{chunk_id}

    Args:
        chunk_id: UUID of the chunk.

    Returns:
        JSON string with chunk content and context.
    """
    import psycopg

    chunk_id = validate_chunk_id(chunk_id)
    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.version_id, c.page_idx, c.element_type, c.text_raw,
                    c.text_level, c.source_id, c.table_body, c.img_path, c.bbox,
                    dp.coord_sys
                FROM chunks c
                LEFT JOIN document_pages dp
                    ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                WHERE c.chunk_id = %s
                """,
                (chunk_id,),
            )

            row = cur.fetchone()
            if not row:
                return json.dumps(
                    {
                        "error": f"Chunk {chunk_id} not found",
                        "chunk_id": chunk_id,
                    }
                )

            (
                version_id,
                page_idx,
                element_type,
                text_raw,
                text_level,
                source_id,
                table_body,
                img_path,
                bbox,
                coord_sys,
            ) = row

            result = {
                "chunk_id": chunk_id,
                "version_id": str(version_id),
                "page_idx": page_idx,
                "element_type": element_type,
                "text": text_raw,
                "text_level": text_level,
                "source_id": source_id,
                "has_table": table_body is not None,
                "has_image": img_path is not None,
                "bbox": bbox,
                "coord_system": coord_sys,
            }

            # Get adjacent chunks on same page for context
            if page_idx is not None:
                cur.execute(
                    """
                    SELECT chunk_id, text_raw, element_type, chunk_index
                    FROM chunks
                    WHERE version_id = %s AND page_idx = %s
                    AND chunk_id != %s
                    ORDER BY ABS(chunk_index - (
                        SELECT chunk_index FROM chunks WHERE chunk_id = %s
                    ))
                    LIMIT 3
                    """,
                    (version_id, page_idx, chunk_id, chunk_id),
                )

                adjacent = cur.fetchall()
                result["context_chunks"] = [
                    {
                        "chunk_id": str(r[0]),
                        "text_preview": r[1][:200] if r[1] else None,
                        "element_type": r[2],
                    }
                    for r in adjacent
                ]

            return json.dumps(result, ensure_ascii=False, indent=2)


def get_outline_resource(version_id: str) -> str:
    """Get document structure outline (table of contents).

    Use this resource when you need to:
    - Understand document organization before searching
    - Navigate to specific sections
    - Plan multi-section analysis

    URI: outline://{version_id}

    Args:
        version_id: UUID of the document version.

    Returns:
        JSON string with hierarchical outline.
    """
    import psycopg

    version_id = validate_version_id(version_id)
    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Check for hierarchical nodes
            cur.execute(
                """
                SELECT node_id, parent_id, level, node_type,
                       metadata->>'heading' as heading,
                       metadata->'page_range' as page_range,
                       content
                FROM hierarchical_nodes
                WHERE version_id = %s
                ORDER BY level, node_id
                """,
                (version_id,),
            )

            nodes = cur.fetchall()

            if nodes:
                outline = []
                for row in nodes:
                    node_info = {
                        "node_id": str(row[0]),
                        "parent_id": str(row[1]) if row[1] else None,
                        "level": row[2],
                        "node_type": row[3],
                        "heading": row[4],
                        "page_range": row[5],
                        "content_preview": row[6][:200] if row[6] else None,
                    }
                    outline.append(node_info)

                return json.dumps(
                    {
                        "version_id": version_id,
                        "source": "hierarchical_nodes",
                        "outline": outline,
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Fallback: use chunks
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

            return json.dumps(
                {
                    "version_id": version_id,
                    "source": "chunks",
                    "outline": outline,
                },
                ensure_ascii=False,
                indent=2,
            )


def get_config_limits(retriever_cache_size: int, query_cache_size: int) -> str:
    """Get query configuration limits.

    Use this resource to understand API constraints before making queries.

    URI: config://limits

    Returns:
        JSON string with configuration limits.
    """
    return json.dumps(
        {
            "max_top_k": 100,
            "max_batch_queries": 50,
            "max_context_window": 1000,
            "max_version_ids_compare": 20,
            "max_key_patterns": 50,
            "default_top_k": 10,
            "cache": {
                "retriever_cache_size": retriever_cache_size,
                "query_cache_size": query_cache_size,
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def get_health_status(
    cached_retrievers: int,
    tool_metrics: dict[str, Any],
) -> str:
    """Get server health status.

    Use this resource to check server availability and current state.

    URI: status://health

    Returns:
        JSON string with health status and metrics.
    """
    # Calculate metrics
    total_calls = sum(m.call_count for m in tool_metrics.values())
    total_errors = sum(m.error_count for m in tool_metrics.values())

    return json.dumps(
        {
            "status": "healthy",
            "server": "Bid Scoring Retrieval MCP",
            "cached_retrievers": cached_retrievers,
            "tool_metrics": {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "tool_count": len(tool_metrics),
                "per_tool": {
                    name: {
                        "calls": m.call_count,
                        "errors": m.error_count,
                        "avg_time_ms": round(m.avg_execution_time_ms, 2),
                    }
                    for name, m in tool_metrics.items()
                },
            },
        },
        ensure_ascii=False,
        indent=2,
    )
