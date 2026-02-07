"""Shared datatypes for retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


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
    rerank_score: float | None = None  # Reranker score (if enabled)
