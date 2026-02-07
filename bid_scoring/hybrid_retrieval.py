"""Backward-compatible facade for the refactored retrieval package.

New code should import from `bid_scoring.retrieval`.
This module is kept to avoid breaking existing imports.
"""

from bid_scoring.retrieval import (  # noqa: F401
    ColBERTReranker,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_RRF_K,
    HAS_COLBERT_RERANKER,
    HAS_CONNECTION_POOL,
    HAS_RERANKER,
    HybridRetriever,
    LRUCache,
    MAX_SEARCH_WORKERS,
    ReciprocalRankFusion,
    Reranker,
    RetrievalResult,
    build_synonym_index,
    load_retrieval_config,
)

__all__ = [
    "ColBERTReranker",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_HNSW_EF_SEARCH",
    "DEFAULT_RRF_K",
    "HAS_COLBERT_RERANKER",
    "HAS_CONNECTION_POOL",
    "HAS_RERANKER",
    "HybridRetriever",
    "LRUCache",
    "MAX_SEARCH_WORKERS",
    "ReciprocalRankFusion",
    "Reranker",
    "RetrievalResult",
    "build_synonym_index",
    "load_retrieval_config",
]
