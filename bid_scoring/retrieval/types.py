from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union


@dataclass
class EvidenceUnit:
    """Unit-level evidence attached to a retrieval result (v0.2 contract)."""

    unit_id: str
    unit_index: int
    unit_type: str
    text: str
    anchor_json: Any
    unit_order: int = 0
    start_char: int | None = None
    end_char: int | None = None


@dataclass
class RetrievalResult:
    """Single retrieval result with detailed scoring information."""

    chunk_id: str
    text: str
    page_idx: int
    score: float
    source: str  # "vector", "keyword", or "hybrid"
    vector_score: float | None = None  # Original vector similarity score
    keyword_score: float | None = None  # Original keyword match score
    embedding: List[float] | None = None
    rerank_score: float | None = None  # Cross-encoder/ColBERT rerank score
    evidence_units: List[EvidenceUnit] = field(default_factory=list)


@dataclass
class RetrievalMetrics:
    """Retrieval performance metrics."""

    vector_search_time_ms: float = 0.0
    keyword_search_time_ms: float = 0.0
    rrf_fusion_time_ms: float = 0.0
    fetch_chunks_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0

    vector_results_count: int = 0
    keyword_results_count: int = 0
    final_results_count: int = 0

    cache_hit: bool = False
    query_type: str = "unknown"  # "technical", "long", "standard"

    def to_dict(self) -> Dict[str, Union[float, int, bool, str]]:
        return {
            "vector_search_time_ms": self.vector_search_time_ms,
            "keyword_search_time_ms": self.keyword_search_time_ms,
            "rrf_fusion_time_ms": self.rrf_fusion_time_ms,
            "fetch_chunks_time_ms": self.fetch_chunks_time_ms,
            "rerank_time_ms": self.rerank_time_ms,
            "total_time_ms": self.total_time_ms,
            "vector_results_count": self.vector_results_count,
            "keyword_results_count": self.keyword_results_count,
            "final_results_count": self.final_results_count,
            "cache_hit": self.cache_hit,
            "query_type": self.query_type,
        }


# Public type aliases used by HybridRetriever.
RrfSources = Dict[str, dict]
MergedChunk = Tuple[str, float, RrfSources]
