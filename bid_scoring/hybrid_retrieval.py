"""
Hybrid Retrieval Module for Bid Scoring

Backward-compatible facade for the hybrid retrieval implementation.

v0.2 refactor note:
- Implementation lives in `bid_scoring.retrieval.*` to keep modules small and focused.
- This file keeps the historical import path stable:
  `from bid_scoring.hybrid_retrieval import HybridRetriever`
"""

from __future__ import annotations

from bid_scoring.retrieval import *  # noqa: F403

# Re-export for type checkers / explicitness (avoid relying on star-import only).
from bid_scoring.retrieval import (  # noqa: F401
    ColBERTReranker,
    DEFAULT_CONFIG_PATH,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_RRF_K,
    EvidenceUnit,
    FieldKeywordsDict,
    HAS_COLBERT_RERANKER,
    HAS_CONNECTION_POOL,
    HAS_RERANKER,
    HybridRetriever,
    LRUCache,
    MAX_SEARCH_WORKERS,
    ReciprocalRankFusion,
    RetrievalMetrics,
    RetrievalResult,
    Reranker,
    SynonymIndexDict,
    build_synonym_index,
    load_retrieval_config,
)

