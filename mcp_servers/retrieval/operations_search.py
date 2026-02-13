"""Search operations for retrieval MCP server."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal

from bid_scoring.config import load_settings

from mcp_servers.retrieval.validation import (
    ValidationError,
    validate_positive_int,
    validate_query,
    validate_string_list,
    validate_version_id,
)


def search_chunks(
    retrieve_fn: Callable[..., Dict[str, Any]],
    version_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    page_range: tuple[int, int] | None = None,
    element_types: list[str] | None = None,
    include_diagnostics: bool = False,
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
    result = retrieve_fn(
        version_id=version_id,
        query=query,
        top_k=top_k * 2,  # Get more for filtering
        mode=mode,
        include_text=True,
        include_diagnostics=include_diagnostics,
    )

    source_results = result["results"]
    results = source_results

    # Apply page range filter
    if page_range:
        start_page, end_page = page_range
        results = [
            r
            for r in results
            if r["page_idx"] is not None and start_page <= r["page_idx"] <= end_page
        ]

    if element_types:
        allowed = set(element_types)
        results = [r for r in results if r.get("element_type") in allowed]

    # Limit to top_k after filtering
    results = results[:top_k]

    response = {
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
    if include_diagnostics:
        response["diagnostics"] = {
            "source_result_count": len(source_results),
            "filtered_result_count": len(results),
            "has_filters": bool(page_range or element_types),
            "source": result.get("diagnostics"),
        }

    return response


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


def batch_search(
    retrieve_fn: Callable[..., Dict[str, Any]],
    version_id: str,
    queries: list[str],
    top_k_per_query: int = 5,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    aggregate_by: Literal["query", "chunk", "page"] | None = None,
    include_diagnostics: bool = False,
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
    per_query_diagnostics: dict[str, Any] = {}

    for query in queries:
        result = retrieve_fn(
            version_id=version_id,
            query=query,
            top_k=top_k_per_query,
            mode=mode,
            include_text=True,
            include_diagnostics=include_diagnostics,
        )
        if include_diagnostics:
            per_query_diagnostics[query] = result.get("diagnostics") or {}

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

        response = {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "aggregated_by": "query",
            "results": aggregated,
        }
        if include_diagnostics:
            response["diagnostics"] = {
                "query_count": len(queries),
                "per_query": per_query_diagnostics,
            }
        return response

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

        response = {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "unique_chunks": len(aggregated),
            "aggregated_by": "chunk",
            "results": aggregated,
        }
        if include_diagnostics:
            response["diagnostics"] = {
                "query_count": len(queries),
                "per_query": per_query_diagnostics,
            }
        return response

    elif aggregate_by == "page":
        # Group by page
        page_map = {}
        for r in all_results:
            page = r.get("page_idx")
            if page not in page_map:
                page_map[page] = []
            page_map[page].append(r)

        response = {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "aggregated_by": "page",
            "results": {k: v for k, v in sorted(page_map.items()) if k is not None},
        }
        if include_diagnostics:
            response["diagnostics"] = {
                "query_count": len(queries),
                "per_query": per_query_diagnostics,
            }
        return response

    else:  # No aggregation
        response = {
            "version_id": version_id,
            "queries": queries,
            "total_results": len(all_results),
            "results": all_results,
        }
        if include_diagnostics:
            response["diagnostics"] = {
                "query_count": len(queries),
                "per_query": per_query_diagnostics,
            }
        return response
