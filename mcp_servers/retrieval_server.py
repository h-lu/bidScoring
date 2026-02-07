"""Bid scoring retrieval MCP server (FastMCP).

Expose retrieval as tools for LLM clients. Tools are intentionally read-only.
"""

from __future__ import annotations

import atexit
import os
from typing import Any, Dict, List, Literal

from fastmcp import FastMCP

from bid_scoring.config import load_settings
from bid_scoring.retrieval import HybridRetriever, LRUCache, RetrievalResult

mcp = FastMCP(name="Bid Scoring Retrieval")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


_SETTINGS = load_settings()

_RETRIEVER_CACHE = LRUCache(capacity=_env_int("BID_SCORING_RETRIEVER_CACHE_SIZE", 32))
_QUERY_CACHE_SIZE = _env_int("BID_SCORING_QUERY_CACHE_SIZE", 1024)


def get_retriever(version_id: str, top_k: int) -> HybridRetriever:
    """Return a cached HybridRetriever for (version_id, top_k)."""
    cache_key = f"{version_id}:{top_k}"
    cached = _RETRIEVER_CACHE.get(cache_key)
    if cached is not None:
        return cached

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
    for retriever in list(_RETRIEVER_CACHE._cache.values()):
        try:
            retriever.close()
        except Exception:
            pass


def _format_result(
    r: RetrievalResult,
    *,
    include_text: bool,
    max_chars: int | None,
) -> Dict[str, Any]:
    text = r.text if include_text else ""
    if include_text and max_chars is not None and max_chars >= 0:
        text = text[:max_chars]
    return {
        "chunk_id": r.chunk_id,
        "page_idx": r.page_idx,
        "source": r.source,
        "score": r.score,
        "vector_score": r.vector_score,
        "keyword_score": r.keyword_score,
        "rerank_score": r.rerank_score,
        "text": text,
    }


@mcp.tool
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
    """MCP tool wrapper for `retrieve_impl`."""
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
    """
    if not version_id:
        raise ValueError("version_id cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

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
        raise ValueError(f"Unknown mode: {mode}")

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
