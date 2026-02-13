"""Bid scoring retrieval MCP server (FastMCP) entrypoint."""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any, Dict, List, Literal

from fastmcp import FastMCP

from bid_scoring.config import load_settings
from bid_scoring.retrieval import HybridRetriever, LRUCache
from mcp_servers.retrieval.formatting import format_result
from mcp_servers.retrieval.operations_annotation import highlight_pdf as _highlight_pdf
from mcp_servers.retrieval.operations_annotation import (
    prepare_highlight_targets_for_query as _prepare_highlight_targets_for_query,
)
from mcp_servers.retrieval.router import register_mcp_routes
from mcp_servers.retrieval.validation import ValidationError
from mcp_servers.retrieval.validation import (
    validate_positive_int,
    validate_query,
    validate_version_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bid_scoring.mcp")

__all__ = [
    "mcp",
    "get_retriever",
    "retrieve_impl",
    "prepare_highlight_targets_impl",
    "_highlight_pdf",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


mcp = FastMCP(name="Bid Scoring Retrieval")
_SETTINGS = load_settings()
_RETRIEVER_CACHE_SIZE = _env_int("BID_SCORING_RETRIEVER_CACHE_SIZE", 32)
_QUERY_CACHE_SIZE = _env_int("BID_SCORING_QUERY_CACHE_SIZE", 1024)
_RETRIEVER_CACHE = LRUCache(capacity=_RETRIEVER_CACHE_SIZE)


def get_retriever(version_id: str, top_k: int) -> HybridRetriever:
    cache_key = f"{version_id}:{top_k}"
    cached = _RETRIEVER_CACHE.get(cache_key)
    if cached is not None:
        logger.debug("Retriever cache hit for key: %s", cache_key)
        return cached

    logger.debug("Creating new retriever for key: %s", cache_key)
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
    logger.info("Closing %s cached retrievers", len(_RETRIEVER_CACHE._cache))
    for retriever in list(_RETRIEVER_CACHE._cache.values()):
        try:
            retriever.close()
        except Exception as exc:
            logger.warning("Error closing retriever: %s", exc)


def prepare_highlight_targets_impl(
    version_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid",
    keywords: List[str] | None = None,
    use_or_semantic: bool = True,
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    return _prepare_highlight_targets_for_query(
        retrieve_fn=retrieve_impl,
        version_id=version_id,
        query=query,
        top_k=top_k,
        mode=mode,
        keywords=keywords,
        use_or_semantic=use_or_semantic,
        include_diagnostics=include_diagnostics,
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
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
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
            keywords,
            use_or_semantic=use_or_semantic,
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

    formatted_results = [
        format_result(item, include_text=include_text, max_chars=max_chars)
        for item in results
    ]

    warnings: list[str] = []
    seen: set[str] = set()
    for item in formatted_results:
        for code in item.get("warnings", []):
            if code not in seen:
                seen.add(code)
                warnings.append(code)

    response = {
        "version_id": version_id,
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "warnings": warnings,
        "results": formatted_results,
    }
    if include_diagnostics:
        response["diagnostics"] = _build_retrieval_diagnostics(
            mode=mode,
            query=query,
            top_k=top_k,
            retriever=retriever,
            results=formatted_results,
        )

    return response


def _build_retrieval_diagnostics(
    *,
    mode: str,
    query: str,
    top_k: int,
    retriever: HybridRetriever,
    results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    warning_counts: Dict[str, int] = {}
    vector_hits = 0
    keyword_hits = 0
    hybrid_hits = 0

    for item in results:
        has_vector = item.get("vector_score") is not None
        has_keyword = item.get("keyword_score") is not None
        if has_vector:
            vector_hits += 1
        if has_keyword:
            keyword_hits += 1
        if has_vector and has_keyword:
            hybrid_hits += 1

        for code in item.get("warnings", []):
            warning_counts[code] = warning_counts.get(code, 0) + 1

    return {
        "mode": mode,
        "query_length": len(query),
        "top_k": top_k,
        "result_count": len(results),
        "vector_hits": vector_hits,
        "keyword_hits": keyword_hits,
        "hybrid_hits": hybrid_hits,
        "rrf_k": getattr(getattr(retriever, "rrf", None), "k", None),
        "warning_counts": warning_counts,
    }


register_mcp_routes(
    mcp=mcp,
    retrieve_impl=retrieve_impl,
    prepare_highlight_targets_impl=prepare_highlight_targets_impl,
    retriever_cache_size=_RETRIEVER_CACHE_SIZE,
    query_cache_size=_QUERY_CACHE_SIZE,
    get_cached_retrievers_count=lambda: len(_RETRIEVER_CACHE._cache),
)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
