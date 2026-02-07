"""Retrieval package.

Public API:
- HybridRetriever
- RetrievalResult
- ReciprocalRankFusion
"""

from .cache import LRUCache
from .config import DEFAULT_CONFIG_PATH, build_synonym_index, load_retrieval_config
from .db import HAS_CONNECTION_POOL
from .hybrid import DEFAULT_HNSW_EF_SEARCH, MAX_SEARCH_WORKERS, HybridRetriever
from .rerankers import HAS_COLBERT_RERANKER, HAS_RERANKER, ColBERTReranker, Reranker
from .rrf import DEFAULT_RRF_K, ReciprocalRankFusion
from .types import RetrievalResult

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
