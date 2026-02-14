"""Bid Document Analyzer Module.

Provides high-level analysis functions for bid evaluation workflows.
Combines retrieval, insights generation, and scoring into cohesive workflows.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp_servers.bid_analysis.comparison import compare_versions
from mcp_servers.bid_analysis.insight import InsightExtractor
from mcp_servers.bid_analysis.models import (
    ANALYSIS_DIMENSIONS,
    AnalysisDimension,
    BidAnalysisReport,
    DimensionResult,
)
from mcp_servers.bid_analysis.recommendation import RecommendationGenerator
from mcp_servers.bid_analysis.retriever import ChunkRetriever
from mcp_servers.bid_analysis.scorer import DimensionScorer

logger = logging.getLogger(__name__)

__all__ = [
    "ANALYSIS_DIMENSIONS",
    "AnalysisDimension",
    "DimensionResult",
    "BidAnalysisReport",
    "BidAnalyzer",
    "compare_versions",
]


class BidAnalyzer:
    """Analyzes bid documents across multiple dimensions."""

    def __init__(
        self,
        conn,
        retriever: ChunkRetriever | None = None,
        insight_extractor: InsightExtractor | None = None,
        dimension_scorer: DimensionScorer | None = None,
        recommendation_generator: RecommendationGenerator | None = None,
    ):
        """Initialize analyzer with pluggable internal components.

        Args:
            conn: psycopg database connection
            retriever: Chunk retrieval component
            insight_extractor: Insight extraction component
            dimension_scorer: Dimension/overall scoring component
            recommendation_generator: Recommendation generation component
        """
        self.conn = conn
        self._retriever = retriever or ChunkRetriever(conn)
        self._insight_extractor = insight_extractor or InsightExtractor()
        self._dimension_scorer = dimension_scorer or DimensionScorer()
        self._recommendation_generator = (
            recommendation_generator or RecommendationGenerator()
        )

    def analyze_version(
        self,
        version_id: str,
        bidder_name: str = "Unknown",
        project_name: str = "Unknown Project",
        dimensions: list[str] | None = None,
    ) -> BidAnalysisReport:
        """Analyze a bid document across specified dimensions."""
        if dimensions is None:
            dimensions = list(ANALYSIS_DIMENSIONS.keys())

        dimension_results = {}
        total_risks = 0
        total_benefits = 0
        evidence_warnings: list[str] = []
        warning_seen: set[str] = set()

        for dim_name in dimensions:
            dim_config = ANALYSIS_DIMENSIONS.get(dim_name)
            if not dim_config:
                logger.warning("Unknown dimension: %s", dim_name)
                continue

            result = self._analyze_dimension(version_id, dim_config)
            dimension_results[dim_name] = result

            total_risks += len(result.risks)
            total_benefits += len(result.benefits)
            for warn in result.evidence_warnings:
                if warn in warning_seen:
                    continue
                warning_seen.add(warn)
                evidence_warnings.append(warn)

        overall_score = self._calculate_overall_score(dimension_results)
        risk_level = self._determine_risk_level(total_risks, total_benefits)
        recommendations = self._generate_recommendations(dimension_results)
        chunks_analyzed = sum(d.chunks_found for d in dimension_results.values())

        return BidAnalysisReport(
            version_id=version_id,
            bidder_name=bidder_name,
            project_name=project_name,
            dimensions=dimension_results,
            overall_score=overall_score,
            total_risks=total_risks,
            total_benefits=total_benefits,
            risk_level=risk_level,
            recommendations=recommendations,
            evidence_warnings=evidence_warnings,
            chunks_analyzed=chunks_analyzed,
        )

    def _analyze_dimension(
        self,
        version_id: str,
        dimension: AnalysisDimension,
    ) -> DimensionResult:
        """Analyze a single dimension."""
        chunks = self._search_chunks(version_id, dimension.keywords)
        evidence_warnings = self._collect_evidence_warnings(chunks)
        score_chunks = [chunk for chunk in chunks if chunk.get("is_verifiable", True)]

        risks, benefits, info = self._insight_extractor.analyze_chunks(
            score_chunks, dimension.name
        )

        extracted_values = {}
        if dimension.extract_patterns:
            extracted_values = self._extract_values(
                score_chunks, dimension.extract_patterns
            )

        score = self._calculate_dimension_score(
            dimension,
            risks,
            benefits,
            extracted_values,
        )
        risk_level = self._determine_dimension_risk_level(
            dimension,
            risks,
            benefits,
            extracted_values,
        )

        summary_parts = []
        if score_chunks:
            summary_parts.append(f"找到 {len(score_chunks)} 处相关内容")
        if benefits:
            summary_parts.append(f"{len(benefits)} 个优势点")
        if risks:
            summary_parts.append(f"{len(risks)} 个风险点")
        summary = "; ".join(summary_parts) if summary_parts else "未找到相关内容"

        return DimensionResult(
            dimension=dimension.name,
            display_name=dimension.display_name,
            chunks_found=len(score_chunks),
            risks=risks,
            benefits=benefits,
            info=info,
            extracted_values=extracted_values,
            score=score,
            risk_level=risk_level,
            summary=summary,
            evidence_warnings=evidence_warnings,
            evidence_citations=_build_dimension_evidence_citations(score_chunks),
        )

    # Compatibility delegates (preserve previous extension/patch points)
    def _collect_evidence_warnings(self, chunks: list[dict[str, Any]]) -> list[str]:
        return self._retriever.collect_evidence_warnings(chunks)

    def _search_chunks(
        self, version_id: str, keywords: list[str]
    ) -> list[dict[str, Any]]:
        return self._retriever.search_chunks(version_id, keywords)

    def _extract_values(
        self,
        chunks: list[dict[str, Any]],
        patterns: list[str],
    ) -> dict[str, Any]:
        return self._insight_extractor.extract_values(chunks, patterns)

    def _calculate_dimension_score(
        self,
        dimension: AnalysisDimension,
        risks,
        benefits,
        extracted_values: dict[str, Any],
    ) -> float:
        return self._dimension_scorer.calculate_dimension_score(
            dimension,
            risks,
            benefits,
            extracted_values,
        )

    def _determine_dimension_risk_level(
        self,
        dimension: AnalysisDimension,
        risks,
        benefits,
        extracted_values: dict[str, Any],
    ) -> str:
        return self._dimension_scorer.determine_dimension_risk_level(
            dimension,
            risks,
            benefits,
            extracted_values,
        )

    def _calculate_overall_score(
        self,
        dimension_results: dict[str, DimensionResult],
    ) -> float:
        return self._dimension_scorer.calculate_overall_score(dimension_results)

    def _determine_risk_level(
        self,
        total_risks: int,
        total_benefits: int,
    ) -> str:
        return self._dimension_scorer.determine_risk_level(total_risks, total_benefits)

    def _generate_recommendations(
        self,
        dimension_results: dict[str, DimensionResult],
    ) -> list[str]:
        return self._recommendation_generator.generate(dimension_results)


def _build_dimension_evidence_citations(
    score_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for chunk in score_chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = _normalize_chunk_id(chunk.get("chunk_id"))
        page_idx = _normalize_page_idx(chunk.get("page_idx"))
        bbox = _normalize_bbox(chunk.get("bbox"))
        token = (
            chunk_id,
            page_idx,
            tuple(bbox) if isinstance(bbox, list) else bbox,
        )
        if token in seen:
            continue
        seen.add(token)
        citations.append(
            {
                "chunk_id": chunk_id,
                "page_idx": page_idx,
                "bbox": bbox,
            }
        )
    return citations


def _normalize_chunk_id(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _normalize_page_idx(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_bbox(value: Any) -> list[float | int] | None:
    if not isinstance(value, list):
        return None
    normalized: list[float | int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            normalized.append(item)
            continue
        try:
            normalized.append(float(item))
        except (TypeError, ValueError):
            return None
    return normalized
