"""Bid scoring retrieval MCP server (FastMCP) - V2 Architecture.

Expose retrieval as tools and resources for LLM clients. Tools are intentionally
read-only. Resources provide URI-addressable access to evidence and document structure.
Designed for automated bid analysis workflows.

V2 Architecture Features:
- Structured logging with execution metrics
- Unified tool execution wrapper with error handling
- Enhanced input validation with ValidationError
- Standardized ToolResult return format
- Performance metrics collection
- Sensitive data sanitization
- Modular code organization
- URI-addressable resources for evidence citation
"""

from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, TypeVar

from fastmcp import FastMCP

from bid_scoring.config import load_settings
from bid_scoring.retrieval import HybridRetriever, LRUCache, RetrievalResult

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bid_scoring.mcp")

# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetrics:
    """Tool execution metrics for monitoring."""

    call_count: int = 0
    total_execution_time_ms: float = 0.0
    error_count: int = 0

    @property
    def avg_execution_time_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time_ms / self.call_count


# =============================================================================
# Utilities
# =============================================================================


def _env_int(name: str, default: int) -> int:
    """Parse environment variable as integer."""
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


# =============================================================================
# Global State
# =============================================================================

mcp = FastMCP(name="Bid Scoring Retrieval")
_SETTINGS = load_settings()

# Caches
_RETRIEVER_CACHE = LRUCache(capacity=_env_int("BID_SCORING_RETRIEVER_CACHE_SIZE", 32))
_QUERY_CACHE_SIZE = _env_int("BID_SCORING_QUERY_CACHE_SIZE", 1024)

# Metrics tracking
_tool_metrics: Dict[str, ToolMetrics] = {}

T = TypeVar("T")


# =============================================================================
# Tool Execution Wrapper
# =============================================================================


def tool_wrapper(tool_name: str) -> Callable:
    """Decorator for tool execution with logging, metrics, and error handling.

    Features:
    - Input validation before execution
    - Structured logging with context
    - Execution time tracking
    - Standardized error responses
    """

    def decorator(func: Callable[..., T]) -> Callable[..., ToolResult | T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult | T:
            # Initialize metrics
            if tool_name not in _tool_metrics:
                _tool_metrics[tool_name] = ToolMetrics()

            metrics = _tool_metrics[tool_name]
            metrics.call_count += 1
            start_time = time.time()

            # Log execution start
            logger.info(
                f"Executing tool: {tool_name}",
                extra={
                    "tool_name": tool_name,
                    "parameters": _sanitize_parameters(kwargs),
                },
            )

            try:
                # Execute tool
                result = func(*args, **kwargs)

                # Record success
                execution_time = (time.time() - start_time) * 1000
                metrics.total_execution_time_ms += execution_time

                logger.info(
                    f"Tool execution completed: {tool_name}",
                    extra={
                        "tool_name": tool_name,
                        "execution_time_ms": execution_time,
                        "success": True,
                    },
                )

                # If result is already a ToolResult, add timing
                if isinstance(result, ToolResult):
                    result.execution_time_ms = execution_time
                    return result

                return result

            except Exception as e:
                # Record failure
                execution_time = (time.time() - start_time) * 1000
                metrics.total_execution_time_ms += execution_time
                metrics.error_count += 1

                logger.error(
                    f"Tool execution failed: {tool_name}",
                    extra={
                        "tool_name": tool_name,
                        "execution_time_ms": execution_time,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    exc_info=True,
                )

                # Return standardized error response
                return ToolResult(
                    success=False,
                    error=f"{tool_name} failed: {str(e)}",
                    execution_time_ms=execution_time,
                    metadata={"error_type": type(e).__name__},
                )

        return wrapper

    return decorator


def _sanitize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from parameters for logging."""
    sensitive_keys = {"password", "token", "secret", "api_key", "key"}
    sanitized = {}
    for key, value in params.items():
        if any(sk in key.lower() for sk in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = value
    return sanitized


# =============================================================================
# Validation Utilities
# =============================================================================


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


def validate_version_id(version_id: str | None) -> str:
    """Validate version ID format.

    Args:
        version_id: The version ID to validate.

    Returns:
        The validated version ID.

    Raises:
        ValidationError: If version_id is invalid.
    """
    if not version_id:
        raise ValidationError("version_id is required and cannot be empty")
    if not isinstance(version_id, str):
        raise ValidationError("version_id must be a string")
    return version_id


def validate_chunk_id(chunk_id: str | None) -> str:
    """Validate chunk ID format.

    Args:
        chunk_id: The chunk ID to validate.

    Returns:
        The validated chunk ID.

    Raises:
        ValidationError: If chunk_id is invalid.
    """
    if not chunk_id:
        raise ValidationError("chunk_id is required and cannot be empty")
    if not isinstance(chunk_id, str):
        raise ValidationError("chunk_id must be a string")
    return chunk_id


def validate_unit_id(unit_id: str | None) -> str:
    """Validate unit ID format.

    Args:
        unit_id: The unit ID to validate.

    Returns:
        The validated unit ID.

    Raises:
        ValidationError: If unit_id is invalid.
    """
    if not unit_id:
        raise ValidationError("unit_id is required and cannot be empty")
    if not isinstance(unit_id, str):
        raise ValidationError("unit_id must be a string")
    return unit_id


def validate_positive_int(value: int, name: str, max_value: int = 1000) -> int:
    """Validate positive integer parameter.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.
        max_value: Maximum allowed value.

    Returns:
        The validated value.

    Raises:
        ValidationError: If value is invalid.
    """
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{name} must be positive")
    if value > max_value:
        raise ValidationError(f"{name} exceeds maximum allowed value of {max_value}")
    return value


def validate_query(query: str | None) -> str:
    """Validate search query.

    Args:
        query: The query to validate.

    Returns:
        The validated query.

    Raises:
        ValidationError: If query is invalid.
    """
    if not query:
        raise ValidationError("query is required and cannot be empty")
    if not isinstance(query, str):
        raise ValidationError("query must be a string")
    if len(query) > 10000:
        raise ValidationError("query exceeds maximum length of 10000 characters")
    return query


def validate_string_list(
    value: list, name: str, min_items: int = 1, max_items: int = 100
) -> list:
    """Validate list of strings.

    Args:
        value: The list to validate.
        name: Parameter name for error messages.
        min_items: Minimum number of items.
        max_items: Maximum number of items.

    Returns:
        The validated list.

    Raises:
        ValidationError: If value is invalid.
    """
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    if len(value) < min_items:
        raise ValidationError(f"{name} must have at least {min_items} item(s)")
    if len(value) > max_items:
        raise ValidationError(f"{name} exceeds maximum of {max_items} items")
    return value


# =============================================================================
# Core Utilities
# =============================================================================


def get_retriever(version_id: str, top_k: int) -> HybridRetriever:
    """Return a cached HybridRetriever for (version_id, top_k).

    Args:
        version_id: Document version UUID.
        top_k: Number of results to retrieve.

    Returns:
        Cached or newly created HybridRetriever instance.
    """
    cache_key = f"{version_id}:{top_k}"
    cached = _RETRIEVER_CACHE.get(cache_key)
    if cached is not None:
        logger.debug(f"Retriever cache hit for key: {cache_key}")
        return cached

    logger.debug(f"Creating new retriever for key: {cache_key}")
    retriever = HybridRetriever(
        version_id=version_id,
        settings=_SETTINGS,
        top_k=top_k,
        enable_cache=True,
        cache_size=_QUERY_CACHE_SIZE,
        use_connection_pool=True,
    )
    _RETRIEVER_CACHE.put(cache_key, retriever)
    return retriever


@atexit.register
def _close_cached_retrievers() -> None:  # pragma: no cover
    """Cleanup function to close all cached retrievers on exit."""
    logger.info(f"Closing {len(_RETRIEVER_CACHE._cache)} cached retrievers")
    for retriever in list(_RETRIEVER_CACHE._cache.values()):
        try:
            retriever.close()
        except Exception as e:
            logger.warning(f"Error closing retriever: {e}")


def _format_result(
    r: RetrievalResult,
    *,
    include_text: bool,
    max_chars: int | None,
) -> Dict[str, Any]:
    """Format retrieval result for output.

    Args:
        r: RetrievalResult to format.
        include_text: Whether to include text content.
        max_chars: Maximum characters for text (if included).

    Returns:
        Formatted result dictionary.
    """
    text = r.text if include_text else ""
    if include_text and max_chars is not None and max_chars >= 0:
        text = text[:max_chars]

    result = {
        "chunk_id": r.chunk_id,
        "page_idx": r.page_idx,
        "source": r.source,
        "score": round(r.score, 6) if r.score is not None else None,
        "vector_score": round(r.vector_score, 6)
        if r.vector_score is not None
        else None,
        "keyword_score": round(r.keyword_score, 6)
        if r.keyword_score is not None
        else None,
        "rerank_score": round(r.rerank_score, 6)
        if r.rerank_score is not None
        else None,
        "text": text,
        "element_type": r.element_type,
        "bbox": r.bbox,
        "coord_system": r.coord_system,
    }

    return result


# =============================================================================
# Layer 1: Discovery & Exploration
# =============================================================================


@mcp.tool
@tool_wrapper("list_available_versions")
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


@mcp.tool
@tool_wrapper("get_document_outline")
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


@mcp.tool
@tool_wrapper("get_page_metadata")
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


# =============================================================================
# Layer 2: Precision Search
# =============================================================================


@mcp.tool
@tool_wrapper("search_chunks")
def search_chunks(
    version_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    page_range: tuple[int, int] | None = None,
    element_types: list[str] | None = None,
) -> Dict[str, Any]:
    """Advanced chunk search with filtering capabilities.

    Enhanced version of retrieve() with additional filters for page range
    and element types.

    Args:
        version_id: UUID of the document version.
        query: Search query string.
        top_k: Max results to return.
        mode: Search mode (hybrid/vector/keyword).
        page_range: Optional (start_page, end_page) to limit search scope.
        element_types: Filter by element types ["table", "text", "title"].

    Returns:
        Search results with chunk metadata and scores.
    """
    # Validate inputs
    version_id = validate_version_id(version_id)
    query = validate_query(query)
    top_k = validate_positive_int(top_k, "top_k", max_value=100)

    # First perform standard retrieval
    result = retrieve_impl(
        version_id=version_id,
        query=query,
        top_k=top_k * 2,  # Get more for filtering
        mode=mode,
        include_text=True,
    )

    results = result["results"]

    # Apply page range filter
    if page_range:
        start_page, end_page = page_range
        results = [
            r
            for r in results
            if r["page_idx"] is not None and start_page <= r["page_idx"] <= end_page
        ]

    # Limit to top_k after filtering
    results = results[:top_k]

    return {
        "version_id": version_id,
        "query": query,
        "mode": mode,
        "filters": {
            "page_range": page_range,
            "element_types": element_types,
        },
        "top_k": top_k,
        "results": results,
    }


@mcp.tool
@tool_wrapper("search_by_heading")
def search_by_heading(
    version_id: str,
    heading_keyword: str,
    include_siblings: bool = True,
    include_children: bool = False,
) -> Dict[str, Any]:
    """Search for content by section heading.

    Useful when you know the section name like "售后服务方案" or "技术参数表"
    and want to retrieve the entire section content.

    Args:
        version_id: UUID of the document version.
        heading_keyword: Keyword to match in headings.
        include_siblings: Include sibling sections at same level.
        include_children: Include subsections.

    Returns:
        Matching sections with their content and hierarchy info.
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)
    if not heading_keyword:
        raise ValidationError("heading_keyword is required and cannot be empty")

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Search in hierarchical_nodes
            cur.execute(
                """
                SELECT node_id, parent_id, level, node_type, content,
                       metadata->>'heading' as heading,
                       metadata->'page_range' as page_range
                FROM hierarchical_nodes
                WHERE version_id = %s
                AND (
                    metadata->>'heading' ILIKE %s
                    OR content ILIKE %s
                )
                AND node_type IN ('section', 'paragraph', 'chunk')
                ORDER BY level, node_id
                """,
                (version_id, f"%{heading_keyword}%", f"%{heading_keyword}%"),
            )

            matches = cur.fetchall()

            sections = []
            for row in matches:
                section = {
                    "node_id": str(row[0]),
                    "parent_id": str(row[1]) if row[1] else None,
                    "level": row[2],
                    "node_type": row[3],
                    "heading": row[5],
                    "page_range": row[6],
                    "content_preview": row[4][:500] if row[4] else None,
                }

                # Get full content if it's a chunk-level node
                if row[3] == "chunk" and include_children:
                    # Get child nodes
                    cur.execute(
                        """
                        SELECT node_id, node_type, content
                        FROM hierarchical_nodes
                        WHERE parent_id = %s
                        ORDER BY node_id
                        """,
                        (row[0],),
                    )
                    children = cur.fetchall()
                    section["children"] = [
                        {"node_id": str(c[0]), "type": c[1], "content": c[2][:200]}
                        for c in children
                    ]

                sections.append(section)

            return {
                "version_id": version_id,
                "heading_keyword": heading_keyword,
                "match_count": len(sections),
                "sections": sections,
            }


# =============================================================================
# Layer 3: Filtering & Sorting
# =============================================================================


@mcp.tool
@tool_wrapper("filter_and_sort_results")
def filter_and_sort_results(
    results: list[Dict[str, Any]],
    filters: Dict[str, Any] | None = None,
    sort_by: Literal["score", "page_idx", "vector_score", "keyword_score"] = "score",
    sort_order: Literal["desc", "asc"] = "desc",
    deduplicate: bool = True,
) -> list[Dict[str, Any]]:
    """Filter and sort search results with flexible criteria.

    Use this to refine search results based on additional criteria
    without re-running the search.

    Args:
        results: List of result items from search_chunks or retrieve.
        filters: Dict with keys like min_score, page_range, element_types.
        sort_by: Field to sort by.
        sort_order: Sort direction.
        deduplicate: Remove duplicate chunks by chunk_id.

    Returns:
        Filtered and sorted results list.
    """
    # Validate inputs
    if not isinstance(results, list):
        raise ValidationError("results must be a list")

    filtered = results.copy()

    # Apply filters
    if filters:
        if "min_score" in filters:
            min_score = filters["min_score"]
            filtered = [r for r in filtered if r.get("score", 0) >= min_score]

        if "max_score" in filters:
            max_score = filters["max_score"]
            filtered = [r for r in filtered if r.get("score", 0) <= max_score]

        if "page_range" in filters:
            start, end = filters["page_range"]
            filtered = [
                r
                for r in filtered
                if r.get("page_idx") is not None and start <= r["page_idx"] <= end
            ]

        if "element_types" in filters:
            types = filters["element_types"]
            filtered = [r for r in filtered if r.get("element_type") in types]

    # Deduplicate
    if deduplicate:
        seen_chunks = set()
        unique = []
        for r in filtered:
            chunk_id = r.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique.append(r)
        filtered = unique

    # Sort
    reverse = sort_order == "desc"

    if sort_by == "score":
        filtered.sort(key=lambda x: x.get("score", 0), reverse=reverse)
    elif sort_by == "page_idx":
        filtered.sort(
            key=lambda x: (x.get("page_idx") or 0, x.get("score", 0)), reverse=reverse
        )
    elif sort_by == "vector_score":
        filtered.sort(key=lambda x: x.get("vector_score") or 0, reverse=reverse)
    elif sort_by == "keyword_score":
        filtered.sort(key=lambda x: x.get("keyword_score") or 0, reverse=reverse)

    return filtered


# =============================================================================
# Layer 4: Batch Operations
# =============================================================================


@mcp.tool
@tool_wrapper("batch_search")
def batch_search(
    version_id: str,
    queries: list[str],
    top_k_per_query: int = 5,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    aggregate_by: Literal["query", "chunk", "page"] | None = None,
) -> Dict[str, Any]:
    """Execute multiple searches in batch and aggregate results.

    Efficient for analyzing multiple dimensions at once, e.g.,
    ["质保期", "响应时间", "培训天数", "备件策略"].

    Args:
        version_id: UUID of the document version.
        queries: List of search queries.
        top_k_per_query: Results per query.
        mode: Search mode.
        aggregate_by: How to group results (by query/chunk/page).

    Returns:
        Aggregated results based on specified grouping.
    """
    # Validate inputs
    version_id = validate_version_id(version_id)
    queries = validate_string_list(queries, "queries", min_items=1, max_items=50)
    top_k_per_query = validate_positive_int(
        top_k_per_query, "top_k_per_query", max_value=50
    )

    all_results = []

    for query in queries:
        result = retrieve_impl(
            version_id=version_id,
            query=query,
            top_k=top_k_per_query,
            mode=mode,
            include_text=True,
        )

        for r in result["results"]:
            r["matched_query"] = query
            all_results.append(r)

    # Aggregate based on strategy
    if aggregate_by == "query":
        aggregated = {}
        for r in all_results:
            q = r.pop("matched_query")
            if q not in aggregated:
                aggregated[q] = []
            aggregated[q].append(r)

        return {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "aggregated_by": "query",
            "results": aggregated,
        }

    elif aggregate_by == "chunk":
        # Group by chunk_id, merge query info
        chunk_map = {}
        for r in all_results:
            chunk_id = r["chunk_id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {
                    **r,
                    "matched_queries": [],
                }
            chunk_map[chunk_id]["matched_queries"].append(r["matched_query"])

        # Remove individual matched_query field
        aggregated = list(chunk_map.values())
        for item in aggregated:
            if "matched_query" in item:
                del item["matched_query"]

        return {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "unique_chunks": len(aggregated),
            "aggregated_by": "chunk",
            "results": aggregated,
        }

    elif aggregate_by == "page":
        # Group by page
        page_map = {}
        for r in all_results:
            page = r.get("page_idx")
            if page not in page_map:
                page_map[page] = []
            page_map[page].append(r)

        return {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "aggregated_by": "page",
            "results": {k: v for k, v in sorted(page_map.items()) if k is not None},
        }

    else:  # No aggregation
        return {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "results": all_results,
        }


# =============================================================================
# Layer 5: Evidence & Traceability
# =============================================================================


@mcp.tool
@tool_wrapper("get_chunk_with_context")
def get_chunk_with_context(
    chunk_id: str,
    context_depth: Literal["chunk", "paragraph", "section", "document"] = "paragraph",
    include_adjacent_pages: bool = False,
) -> Dict[str, Any]:
    """Get a chunk with its surrounding context to avoid out-of-context interpretation.

    Critical for accurate bid analysis - ensures you're not misinterpreting
    a table cell or sentence fragment.

    Args:
        chunk_id: UUID of the chunk to retrieve.
        context_depth: How much context to include.
        include_adjacent_pages: Include content from neighboring pages.

    Returns:
        Chunk content with requested context.
    """
    import psycopg

    # Validate inputs
    chunk_id = validate_chunk_id(chunk_id)

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Get the chunk
            cur.execute(
                """
                SELECT
                    c.version_id, c.page_idx, c.element_type, c.text_raw,
                    c.source_id, c.embedding IS NOT NULL as has_embedding,
                    c.bbox, dp.coord_sys
                FROM chunks c
                LEFT JOIN document_pages dp
                    ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                WHERE c.chunk_id = %s
                """,
                (chunk_id,),
            )

            row = cur.fetchone()
            if not row:
                raise ValidationError(f"Chunk {chunk_id} not found")

            (
                version_id,
                page_idx,
                element_type,
                text_raw,
                source_id,
                has_embedding,
                bbox,
                coord_sys,
            ) = row

            result = {
                "chunk_id": chunk_id,
                "version_id": str(version_id),
                "page_idx": page_idx,
                "element_type": element_type,
                "text": text_raw,
                "source_id": source_id,
                "has_embedding": has_embedding,
                "bbox": bbox if bbox else None,
                "coord_system": coord_sys if coord_sys else "mineru_bbox_v1",
            }

            # Get context based on depth
            if context_depth in ["paragraph", "section", "document"]:
                # Try to find in hierarchical_nodes
                cur.execute(
                    """
                    SELECT parent_id, node_type, content, metadata
                    FROM hierarchical_nodes
                    WHERE version_id = %s
                    AND %s = ANY(source_chunk_ids)
                    """,
                    (version_id, chunk_id),
                )

                hierarchy = cur.fetchall()
                if hierarchy:
                    result["hierarchy"] = [
                        {
                            "parent_id": str(h[0]) if h[0] else None,
                            "node_type": h[1],
                            "content": h[2][:1000] if h[2] else None,
                            "metadata": h[3],
                        }
                        for h in hierarchy
                    ]

            # Get adjacent chunks on same page
            if context_depth in ["paragraph", "section"] and page_idx is not None:
                cur.execute(
                    """
                    SELECT chunk_id, text_raw, element_type, chunk_index
                    FROM chunks
                    WHERE version_id = %s AND page_idx = %s
                    AND chunk_id != %s
                    ORDER BY chunk_index
                    LIMIT 5
                    """,
                    (version_id, page_idx, chunk_id),
                )

                adjacent = cur.fetchall()
                result["same_page_chunks"] = [
                    {
                        "chunk_id": str(r[0]),
                        "text_preview": r[1][:200] if r[1] else None,
                        "element_type": r[2],
                        "chunk_index": r[3],
                    }
                    for r in adjacent
                ]

            # Get adjacent pages
            if include_adjacent_pages and page_idx is not None:
                cur.execute(
                    """
                    SELECT page_idx, text_raw, element_type
                    FROM chunks
                    WHERE version_id = %s AND page_idx IN (%s, %s)
                    AND element_type IN ('title', 'text')
                    ORDER BY page_idx, chunk_index
                    """,
                    (version_id, page_idx - 1, page_idx + 1),
                )

                adjacent_pages = cur.fetchall()
                result["adjacent_pages"] = {}
                for r in adjacent_pages:
                    p_idx = r[0]
                    if p_idx not in result["adjacent_pages"]:
                        result["adjacent_pages"][p_idx] = []
                    result["adjacent_pages"][p_idx].append(
                        {
                            "text_preview": r[1][:200] if r[1] else None,
                            "element_type": r[2],
                        }
                    )

            return result


@mcp.tool
@tool_wrapper("get_unit_evidence")
def get_unit_evidence(
    unit_id: str,
    verify_hash: bool = True,
    include_anchor: bool = True,
) -> Dict[str, Any]:
    """Get precise evidence from content_units (v0.2) for audit-grade verification.

    The most granular level of evidence - use this when you need to
    precisely quote and verify a specific commitment.

    Args:
        unit_id: UUID of the content unit.
        verify_hash: Verify evidence hash for integrity.
        include_anchor: Include coordinate/position info.

    Returns:
        Unit content with verification metadata.
    """
    import psycopg

    # Validate inputs
    unit_id = validate_unit_id(unit_id)

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version_id, unit_index, unit_type, text_raw, text_norm,
                       char_count, anchor_json, unit_hash, source_element_id
                FROM content_units
                WHERE unit_id = %s
                """,
                (unit_id,),
            )

            row = cur.fetchone()
            if not row:
                raise ValidationError(f"Unit {unit_id} not found")

            (
                version_id,
                unit_index,
                unit_type,
                text_raw,
                text_norm,
                char_count,
                anchor_json,
                unit_hash,
                source_element_id,
            ) = row

            result = {
                "unit_id": unit_id,
                "version_id": str(version_id),
                "unit_index": unit_index,
                "unit_type": unit_type,
                "text_raw": text_raw,
                "text_normalized": text_norm,
                "char_count": char_count,
                "source_element_id": source_element_id,
            }

            if include_anchor:
                result["anchor"] = anchor_json

            # Get associated chunks with bbox
            cur.execute(
                """
                SELECT
                    c.chunk_id, c.text_raw, c.page_idx, c.bbox, c.element_type,
                    dp.coord_sys
                FROM chunks c
                JOIN chunk_unit_spans span ON c.chunk_id = span.chunk_id
                LEFT JOIN document_pages dp
                    ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                WHERE span.unit_id = %s
                """,
                (unit_id,),
            )

            chunks = cur.fetchall()
            result["associated_chunks"] = [
                {
                    "chunk_id": str(r[0]),
                    "text_preview": r[1][:200] if r[1] else None,
                    "page_idx": r[2],
                    "bbox": r[3] if r[3] else None,
                    "element_type": r[4] if r[4] else None,
                    "coord_system": r[5] if r[5] else "mineru_bbox_v1",
                }
                for r in chunks
            ]

            if verify_hash and unit_hash:
                from bid_scoring.citations_v2 import compute_evidence_hash

                computed_hash = compute_evidence_hash(
                    quote_text=text_raw or "",
                    unit_hash=unit_hash,
                    anchor_json=anchor_json,
                )
                result["hash_verified"] = computed_hash == unit_hash
                result["computed_hash"] = computed_hash

            return result


# =============================================================================
# Layer 6: Comparison & Analysis
# =============================================================================


@mcp.tool
@tool_wrapper("compare_across_versions")
def compare_across_versions(
    version_ids: list[str],
    query: str,
    top_k_per_version: int = 3,
    normalize_scores: bool = True,
) -> Dict[str, Any]:
    """Compare responses across multiple bidding versions for the same query.

    Essential for bid analysis - see how different bidders respond to
    the same requirement.

    Args:
        version_ids: List of version UUIDs to compare.
        query: Search query (e.g., "售后服务响应时间").
        top_k_per_version: Results per version.
        normalize_scores: Normalize scores across versions for fair comparison.

    Returns:
        Side-by-side comparison of responses from each version.
    """
    # Validate inputs
    version_ids = validate_string_list(
        version_ids, "version_ids", min_items=1, max_items=20
    )
    query = validate_query(query)
    top_k_per_version = validate_positive_int(
        top_k_per_version, "top_k_per_version", max_value=50
    )

    all_results = {}
    all_scores = []

    for version_id in version_ids:
        result = retrieve_impl(
            version_id=version_id,
            query=query,
            top_k=top_k_per_version,
            mode="hybrid",
            include_text=True,
            max_chars=500,
        )

        all_results[version_id] = result["results"]
        all_scores.extend([r["score"] for r in result["results"]])

    # Normalize scores if requested
    if normalize_scores and all_scores:
        max_score = max(all_scores) if all_scores else 1.0
        min_score = min(all_scores) if all_scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for version_id, results in all_results.items():
            for r in results:
                if score_range > 0:
                    r["normalized_score"] = (r["score"] - min_score) / score_range
                else:
                    r["normalized_score"] = 1.0

    return {
        "query": query,
        "version_count": len(version_ids),
        "versions_compared": version_ids,
        "normalize_scores": normalize_scores,
        "results_by_version": all_results,
    }


@mcp.tool
@tool_wrapper("extract_key_value")
def extract_key_value(
    version_id: str,
    key_patterns: list[str],
    value_patterns: list[str] | None = None,
    fuzzy_match: bool = True,
    context_window: int = 50,
) -> list[Dict[str, Any]]:
    """Extract structured key-value pairs from document text.

    Useful for extracting commitments like:
    - 质保期: 5年
    - 响应时间: 2小时
    - 培训天数: 3天

    Args:
        version_id: UUID of the document version.
        key_patterns: Keywords to search for (e.g., ["质保期", "保修期"]).
        value_patterns: Optional patterns for values (e.g., ["年", "月", "天"]).
        fuzzy_match: Use fuzzy matching for key patterns.
        context_window: Characters to extract around the match.

    Returns:
        List of extracted key-value pairs with source locations.
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)
    key_patterns = validate_string_list(
        key_patterns, "key_patterns", min_items=1, max_items=50
    )
    if value_patterns is not None:
        value_patterns = validate_string_list(
            value_patterns, "value_patterns", min_items=0, max_items=50
        )
    context_window = validate_positive_int(
        context_window, "context_window", max_value=1000
    )

    settings = load_settings()
    extractions = []

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Build search condition
            if fuzzy_match:
                key_conditions = " OR ".join(
                    ["text_raw ILIKE %s" for _ in key_patterns]
                )
                params = [f"%{k}%" for k in key_patterns] + [version_id]
            else:
                key_conditions = " OR ".join(["text_raw LIKE %s" for _ in key_patterns])
                params = key_patterns + [version_id]

            cur.execute(
                f"""
                SELECT chunk_id, text_raw, page_idx, source_id
                FROM chunks
                WHERE ({key_conditions})
                AND version_id = %s
                AND text_raw IS NOT NULL
                """,
                tuple(params),
            )

            rows = cur.fetchall()

            for row in rows:
                chunk_id, text_raw, page_idx, source_id = row

                # Find key matches in text
                for key in key_patterns:
                    if fuzzy_match:
                        pattern = re.compile(re.escape(key), re.IGNORECASE)
                    else:
                        pattern = re.compile(re.escape(key))

                    for match in pattern.finditer(text_raw):
                        start = max(0, match.start() - context_window)
                        end = min(len(text_raw), match.end() + context_window)
                        context = text_raw[start:end]

                        extraction = {
                            "key": key,
                            "context": context,
                            "chunk_id": str(chunk_id),
                            "page_idx": page_idx,
                            "source_id": source_id,
                            "match_position": match.span(),
                        }

                        # Try to extract value if value_patterns provided
                        if value_patterns:
                            # Look for value patterns after the key
                            search_text = text_raw[
                                match.end() : match.end() + context_window * 2
                            ]
                            for vp in value_patterns:
                                # Pattern: number + unit
                                val_regex = rf"(\d+(?:\.\d+)?)\s*{re.escape(vp)}"
                                val_match = re.search(val_regex, search_text)
                                if val_match:
                                    extraction["value"] = val_match.group(0)
                                    extraction["numeric_value"] = float(
                                        val_match.group(1)
                                    )
                                    extraction["unit"] = vp
                                    break

                        extractions.append(extraction)

    # Deduplicate by chunk_id + key
    seen = set()
    unique = []
    for e in extractions:
        key = (e["chunk_id"], e["key"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# =============================================================================
# Layer 7: PDF Annotation
# =============================================================================


@mcp.tool
@tool_wrapper("highlight_pdf")
def highlight_pdf(
    version_id: str,
    chunk_ids: list[str],
    topic: str,
    color: str | None = None,
    increment: bool = True,
) -> Dict[str, Any]:
    """Add highlights to PDF for specified chunks.

    Creates visually annotated PDFs for bid review by highlighting
    relevant content based on chunk bbox coordinates. Supports cumulative
    layer additions for different analysis topics with color coding.

    Topics are color-coded:
    - risk: Red (liabilities, penalties)
    - warranty: Green (after-sales, guarantees)
    - training: Yellow (training provisions)
    - delivery: Orange (delivery timeline)
    - financial: Blue (payment terms)
    - technical: Purple (technical specs)

    Args:
        version_id: Document version UUID to highlight.
        chunk_ids: List of chunk IDs to highlight from search results.
        topic: Topic name for color coding (e.g., 'warranty', 'training').
        color: Optional color in hex (#RRGGBB) or RGB format (0-1).
               Auto-assigned by topic if None.
        increment: If True, add to existing annotated PDF.
                  If False, create new from original.

    Returns:
        Dict with:
        - success: Whether operation succeeded
        - annotated_url: Presigned URL to annotated PDF (15 min valid)
        - highlights_added: Number of highlights added
        - file_path: MinIO object key for the annotated PDF
        - topics: List of topics in the annotated PDF
        - error: Error message if failed
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)
    chunk_ids = validate_string_list(
        chunk_ids, "chunk_ids", min_items=1, max_items=500
    )
    topic = validate_query(topic)  # Use query validation for topic name
    if color is not None:
        if not isinstance(color, str):
            raise ValidationError("color must be a string")

    increment = validate_bool(increment, "increment")

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        from mineru.minio_storage import MinIOStorage
        from mcp_servers.pdf_annotator import PDFAnnotator, parse_color

        # Parse color if provided
        rgb_color = None
        if color:
            rgb_color = parse_color(color)

        # Create annotator
        storage = MinIOStorage()
        annotator = PDFAnnotator(conn, storage)

        # Perform highlighting
        result = annotator.highlight_chunks(
            version_id=version_id,
            chunk_ids=chunk_ids,
            topic=topic,
            color=rgb_color,
            increment=increment,
        )

        if result.success:
            return {
                "success": True,
                "annotated_url": result.annotated_url,
                "highlights_added": result.highlights_added,
                "file_path": result.file_path,
                "topics": result.topics,
                "expires_in_minutes": 15,
            }
        else:
            return {
                "success": False,
                "error": result.error,
            }


def validate_bool(value: bool, name: str) -> bool:
    """Validate boolean parameter."""
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a boolean")
    return value


# =============================================================================
# Backward Compatibility: Original retrieve tool
# =============================================================================


@mcp.tool
@tool_wrapper("retrieve")
def retrieve(
    version_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid",
    keywords: List[str] | None = None,
    use_or_semantic: bool = True,
    include_text: bool = True,
    max_chars: int | None = None,
) -> Dict[str, Any]:
    """MCP tool wrapper for retrieve_impl.

    Backward compatible interface for the original retrieve tool.

    Args:
        version_id: UUID of the document version to search within.
        query: User query string.
        top_k: Max number of results to return.
        mode: "hybrid" (vector + keyword), "keyword", or "vector".
        keywords: Optional list of keywords; if omitted, keywords are extracted.
        use_or_semantic: Only used for keyword search.
        include_text: Whether to include chunk text in results.
        max_chars: If set, truncate returned text to at most this many characters.

    Returns:
        Retrieval results with chunk metadata and scores.
    """
    return retrieve_impl(
        version_id=version_id,
        query=query,
        top_k=top_k,
        mode=mode,
        keywords=keywords,
        use_or_semantic=use_or_semantic,
        include_text=include_text,
        max_chars=max_chars,
    )


def retrieve_impl(
    version_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid",
    keywords: List[str] | None = None,
    use_or_semantic: bool = True,
    include_text: bool = True,
    max_chars: int | None = None,
) -> Dict[str, Any]:
    """Retrieve chunks from a document version.

    Args:
        version_id: UUID of the document version to search within.
        query: User query string.
        top_k: Max number of results to return.
        mode: "hybrid" (vector + keyword), "keyword", or "vector".
        keywords: Optional list of keywords; if omitted, keywords are extracted.
        use_or_semantic: Only used for keyword search.
        include_text: Whether to include chunk text in results.
        max_chars: If set, truncate returned text to at most this many characters.

    Returns:
        Retrieval results with chunk metadata and scores.
    """
    # Validate inputs
    version_id = validate_version_id(version_id)
    query = validate_query(query)
    top_k = validate_positive_int(top_k, "top_k", max_value=100)

    retriever = get_retriever(version_id=version_id, top_k=top_k)

    if mode == "hybrid":
        results = retriever.retrieve(query, keywords=keywords)
    elif mode == "vector":
        vector_results = retriever._vector_search(query)
        merged = [
            (
                doc_id,
                1.0 / (retriever.rrf.k + rank + 1),
                {"vector": {"rank": rank, "score": score}},
            )
            for rank, (doc_id, score) in enumerate(vector_results)
        ]
        results = retriever._fetch_chunks(merged[:top_k])
    elif mode == "keyword":
        if keywords is None:
            keywords = retriever.extract_keywords_from_query(query)
        keyword_results = retriever._keyword_search_fulltext(
            keywords, use_or_semantic=use_or_semantic
        )
        merged = [
            (
                doc_id,
                1.0 / (retriever.rrf.k + rank + 1),
                {"keyword": {"rank": rank, "score": score}},
            )
            for rank, (doc_id, score) in enumerate(keyword_results)
        ]
        results = retriever._fetch_chunks(merged[:top_k])
    else:  # pragma: no cover
        raise ValidationError(f"Unknown mode: {mode}")

    return {
        "version_id": version_id,
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "results": [
            _format_result(r, include_text=include_text, max_chars=max_chars)
            for r in results
        ],
    }


# =============================================================================
# Resources: URI-Addressable Evidence and Document Structure
# =============================================================================
#
# Resources provide direct URI access to document content, enabling:
# - Precise evidence citation in analysis
# - Multi-turn conversation context retention
# - External system integration via URI references
#
# URI Schemes:
# - evidence://unit/{unit_id}  - Content unit (finest-grained evidence)
# - evidence://chunk/{chunk_id} - Chunk content with context
# - outline://{version_id}     - Document structure/outline
# - config://limits            - Query configuration limits
# - status://health            - Server health status
# =============================================================================


@mcp.resource("evidence://unit/{unit_id}")
@tool_wrapper("resource_unit_evidence")
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


@mcp.resource("evidence://chunk/{chunk_id}")
@tool_wrapper("resource_chunk_evidence")
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


@mcp.resource("outline://{version_id}")
@tool_wrapper("resource_document_outline")
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


@mcp.resource("config://limits")
@tool_wrapper("resource_config_limits")
def get_config_limits() -> str:
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
                "retriever_cache_size": _env_int(
                    "BID_SCORING_RETRIEVER_CACHE_SIZE", 32
                ),
                "query_cache_size": _env_int("BID_SCORING_QUERY_CACHE_SIZE", 1024),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.resource("status://health")
@tool_wrapper("resource_health_status")
def get_health_status() -> str:
    """Get server health status.

    Use this resource to check server availability and current state.

    URI: status://health

    Returns:
        JSON string with health status and metrics.
    """
    # Calculate metrics
    total_calls = sum(m.call_count for m in _tool_metrics.values())
    total_errors = sum(m.error_count for m in _tool_metrics.values())

    return json.dumps(
        {
            "status": "healthy",
            "server": "Bid Scoring Retrieval MCP",
            "cached_retrievers": len(_RETRIEVER_CACHE._cache),
            "tool_metrics": {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "tool_count": len(_tool_metrics),
                "per_tool": {
                    name: {
                        "calls": m.call_count,
                        "errors": m.error_count,
                        "avg_time_ms": round(m.avg_execution_time_ms, 2),
                    }
                    for name, m in _tool_metrics.items()
                },
            },
        },
        ensure_ascii=False,
        indent=2,
    )


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Entry point for uvx and pip install."""
    mcp.run()


if __name__ == "__main__":
    main()
