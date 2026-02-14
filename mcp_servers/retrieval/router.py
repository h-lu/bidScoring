"""Route registration for retrieval MCP server.

Keeps FastMCP tool/resource declarations separate from retrieval runtime logic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal

from fastmcp import FastMCP

from mcp_servers.retrieval.execution import _tool_metrics, tool_wrapper
from mcp_servers.retrieval.operations_analysis import (
    analyze_bids_comprehensive as _analyze,
)
from mcp_servers.retrieval.operations_annotation import highlight_pdf as _highlight_pdf
from mcp_servers.retrieval.operations_discovery import (
    get_document_outline as _get_document_outline,
)
from mcp_servers.retrieval.operations_discovery import (
    get_page_metadata as _get_page_metadata,
)
from mcp_servers.retrieval.operations_discovery import (
    list_available_versions as _list_available_versions,
)
from mcp_servers.retrieval.operations_evidence import (
    compare_across_versions as _compare_across_versions,
)
from mcp_servers.retrieval.operations_evidence import (
    extract_key_value as _extract_key_value,
)
from mcp_servers.retrieval.operations_evidence import (
    get_chunk_with_context as _get_chunk_with_context,
)
from mcp_servers.retrieval.operations_evidence import (
    get_unit_evidence as _get_unit_evidence,
)
from mcp_servers.retrieval.operations_resources import (
    get_chunk_evidence_resource as _get_chunk_evidence_resource,
)
from mcp_servers.retrieval.operations_resources import (
    get_config_limits as _get_config_limits,
)
from mcp_servers.retrieval.operations_resources import (
    get_health_status as _get_health_status,
)
from mcp_servers.retrieval.operations_resources import (
    get_outline_resource as _get_outline_resource,
)
from mcp_servers.retrieval.operations_resources import (
    get_unit_evidence_resource as _get_unit_evidence_resource,
)
from mcp_servers.retrieval.operations_search import (
    batch_search as _batch_search,
)
from mcp_servers.retrieval.operations_search import (
    filter_and_sort_results as _filter_and_sort_results,
)
from mcp_servers.retrieval.operations_search import (
    search_by_heading as _search_by_heading,
)
from mcp_servers.retrieval.operations_search import search_chunks as _search_chunks


def register_mcp_routes(
    *,
    mcp: FastMCP,
    retrieve_impl: Callable[..., Dict[str, Any]],
    prepare_highlight_targets_impl: Callable[..., Dict[str, Any]],
    retriever_cache_size: int,
    query_cache_size: int,
    get_cached_retrievers_count: Callable[[], int],
) -> None:
    """Register tool/resource routes on the provided FastMCP app."""

    @mcp.tool
    @tool_wrapper("list_available_versions")
    def list_available_versions(
        project_id: str | None = None,
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        return _list_available_versions(
            project_id=project_id, include_stats=include_stats
        )

    @mcp.tool
    @tool_wrapper("get_document_outline")
    def get_document_outline(version_id: str, max_depth: int = 3) -> Dict[str, Any]:
        return _get_document_outline(version_id=version_id, max_depth=max_depth)

    @mcp.tool
    @tool_wrapper("get_page_metadata")
    def get_page_metadata(
        version_id: str,
        page_idx: int | list[int] | None = None,
        include_elements: bool = True,
    ) -> Dict[str, Any]:
        return _get_page_metadata(
            version_id=version_id,
            page_idx=page_idx,
            include_elements=include_elements,
        )

    @mcp.tool
    @tool_wrapper("search_chunks")
    def search_chunks(
        version_id: str,
        query: str,
        top_k: int = 10,
        mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
        page_range: tuple[int, int] | None = None,
        element_types: list[str] | None = None,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        return _search_chunks(
            retrieve_fn=retrieve_impl,
            version_id=version_id,
            query=query,
            top_k=top_k,
            mode=mode,
            page_range=page_range,
            element_types=element_types,
            include_diagnostics=include_diagnostics,
        )

    @mcp.tool
    @tool_wrapper("search_by_heading")
    def search_by_heading(
        version_id: str,
        heading_keyword: str,
        include_siblings: bool = True,
        include_children: bool = False,
    ) -> Dict[str, Any]:
        return _search_by_heading(
            version_id=version_id,
            heading_keyword=heading_keyword,
            include_siblings=include_siblings,
            include_children=include_children,
        )

    @mcp.tool
    @tool_wrapper("filter_and_sort_results")
    def filter_and_sort_results(
        results: list[Dict[str, Any]],
        filters: Dict[str, Any] | None = None,
        sort_by: Literal[
            "score", "page_idx", "vector_score", "keyword_score"
        ] = "score",
        sort_order: Literal["desc", "asc"] = "desc",
        deduplicate: bool = True,
    ) -> list[Dict[str, Any]]:
        return _filter_and_sort_results(
            results=results,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
            deduplicate=deduplicate,
        )

    @mcp.tool
    @tool_wrapper("batch_search")
    def batch_search(
        version_id: str,
        queries: list[str],
        top_k_per_query: int = 5,
        mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
        aggregate_by: Literal["query", "chunk", "page"] | None = None,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        return _batch_search(
            retrieve_fn=retrieve_impl,
            version_id=version_id,
            queries=queries,
            top_k_per_query=top_k_per_query,
            mode=mode,
            aggregate_by=aggregate_by,
            include_diagnostics=include_diagnostics,
        )

    @mcp.tool
    @tool_wrapper("get_chunk_with_context")
    def get_chunk_with_context(
        chunk_id: str,
        context_depth: Literal[
            "chunk", "paragraph", "section", "document"
        ] = "paragraph",
        include_adjacent_pages: bool = False,
    ) -> Dict[str, Any]:
        return _get_chunk_with_context(
            chunk_id=chunk_id,
            context_depth=context_depth,
            include_adjacent_pages=include_adjacent_pages,
        )

    @mcp.tool
    @tool_wrapper("get_unit_evidence")
    def get_unit_evidence(
        unit_id: str,
        verify_hash: bool = True,
        include_anchor: bool = True,
    ) -> Dict[str, Any]:
        return _get_unit_evidence(
            unit_id=unit_id,
            verify_hash=verify_hash,
            include_anchor=include_anchor,
        )

    @mcp.tool
    @tool_wrapper("compare_across_versions")
    def compare_across_versions(
        version_ids: list[str],
        query: str,
        top_k_per_version: int = 3,
        normalize_scores: bool = True,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        return _compare_across_versions(
            retrieve_fn=retrieve_impl,
            version_ids=version_ids,
            query=query,
            top_k_per_version=top_k_per_version,
            normalize_scores=normalize_scores,
            include_diagnostics=include_diagnostics,
        )

    @mcp.tool
    @tool_wrapper("extract_key_value")
    def extract_key_value(
        version_id: str,
        key_patterns: list[str],
        value_patterns: list[str] | None = None,
        fuzzy_match: bool = True,
        context_window: int = 50,
    ) -> list[Dict[str, Any]]:
        return _extract_key_value(
            version_id=version_id,
            key_patterns=key_patterns,
            value_patterns=value_patterns,
            fuzzy_match=fuzzy_match,
            context_window=context_window,
        )

    @mcp.tool
    @tool_wrapper("highlight_pdf")
    def highlight_pdf(
        version_id: str,
        chunk_ids: list[str],
        topic: str,
        color: str | None = None,
        increment: bool = True,
    ) -> Dict[str, Any]:
        return _highlight_pdf(
            version_id=version_id,
            chunk_ids=chunk_ids,
            topic=topic,
            color=color,
            increment=increment,
        )

    @mcp.tool
    @tool_wrapper("prepare_highlight_targets")
    def prepare_highlight_targets(
        version_id: str,
        query: str,
        top_k: int = 10,
        mode: Literal["hybrid", "keyword", "vector"] = "hybrid",
        keywords: List[str] | None = None,
        use_or_semantic: bool = True,
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        return prepare_highlight_targets_impl(
            version_id=version_id,
            query=query,
            top_k=top_k,
            mode=mode,
            keywords=keywords,
            use_or_semantic=use_or_semantic,
            include_diagnostics=include_diagnostics,
        )

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
        include_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        return retrieve_impl(
            version_id=version_id,
            query=query,
            top_k=top_k,
            mode=mode,
            keywords=keywords,
            use_or_semantic=use_or_semantic,
            include_text=include_text,
            max_chars=max_chars,
            include_diagnostics=include_diagnostics,
        )

    @mcp.resource("evidence://unit/{unit_id}")
    @tool_wrapper("resource_unit_evidence")
    def get_unit_evidence_resource(unit_id: str) -> str:
        return _get_unit_evidence_resource(unit_id=unit_id)

    @mcp.resource("evidence://chunk/{chunk_id}")
    @tool_wrapper("resource_chunk_evidence")
    def get_chunk_evidence_resource(chunk_id: str) -> str:
        return _get_chunk_evidence_resource(chunk_id=chunk_id)

    @mcp.resource("outline://{version_id}")
    @tool_wrapper("resource_outline")
    def get_outline_resource(version_id: str) -> str:
        return _get_outline_resource(version_id=version_id)

    @mcp.resource("config://limits")
    @tool_wrapper("resource_config_limits")
    def get_config_limits() -> str:
        return _get_config_limits(
            retriever_cache_size=retriever_cache_size,
            query_cache_size=query_cache_size,
        )

    @mcp.resource("status://health")
    @tool_wrapper("resource_health_status")
    def get_health_status() -> str:
        return _get_health_status(
            cached_retrievers=get_cached_retrievers_count(),
            tool_metrics=_tool_metrics,
        )

    @mcp.tool
    @tool_wrapper("analyze_bids_comprehensive")
    def analyze_bids_comprehensive(
        version_ids: list[str],
        bidder_names: dict[str, str] | None = None,
        dimensions: list[str] | None = None,
        generate_annotations: bool = False,
    ) -> Dict[str, Any]:
        return _analyze(
            version_ids=version_ids,
            bidder_names=bidder_names,
            dimensions=dimensions,
            generate_annotations=generate_annotations,
        )
