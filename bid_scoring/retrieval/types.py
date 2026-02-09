"""Shared datatypes for retrieval.

These types are intentionally lightweight and dependency-free so they can be
used across the retrieval package (vector search, keyword search, fetching,
reranking, MCP formatting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


SourcesDict = Dict[str, Dict[str, Any]]

# (chunk_id, rrf_score, sources)
MergedChunk = Tuple[str, float, SourcesDict]


@dataclass(frozen=True)
class EvidenceUnit:
    """Unit-level evidence span for v0.2 schema (content_units + chunk_unit_spans)."""

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
    vector_score: float | None = None
    keyword_score: float | None = None
    embedding: List[float] | None = None
    rerank_score: float | None = None
    evidence_units: List[EvidenceUnit] = field(default_factory=list)
    # PDF position information (mineru_bbox_v1 format)
    bbox: List[float] | None = None  # [x1, y1, x2, y2] in PDF coordinates
    element_type: str | None = None  # text, table, image, title, etc.
    coord_system: str | None = None  # e.g., "mineru_bbox_v1"


@dataclass(frozen=True)
class RetrievalMetrics:
    """Optional retrieval diagnostics (kept for backwards compatibility)."""

    query: str
    mode: str
    elapsed_ms: float
