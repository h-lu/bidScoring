"""Bid Document Analyzer Module.

Provides high-level analysis functions for bid evaluation workflows.
Combines retrieval, insights generation, and scoring into cohesive workflows.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from mcp_servers.annotation_insights import (
    AnnotationInsight,
    analyze_chunk_for_insights,
    RISK_PATTERNS,
    BENEFIT_PATTERNS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Analysis Dimensions Configuration
# =============================================================================

@dataclass
class AnalysisDimension:
    """Configuration for a single analysis dimension."""

    name: str  # e.g., "warranty", "delivery", "training"
    display_name: str  # e.g., "质保售后", "交付响应"
    weight: float  # 0.0-1.0, for scoring
    keywords: list[str]
    extract_patterns: list[str] | None = None
    risk_thresholds: dict[str, Any] | None = None


# Standard analysis dimensions based on industry best practices
ANALYSIS_DIMENSIONS = {
    "warranty": AnalysisDimension(
        name="warranty",
        display_name="质保售后",
        weight=0.25,
        keywords=["质保", "保修", "免费维修", "终身服务", "售后服务"],
        extract_patterns=[r"(\d+)\s*年", r"(\d+)\s*个月"],
        risk_thresholds={
            "excellent": [5, 999],  # 5+ years
            "good": [3, 5],         # 3-5 years
            "medium": [2, 3],       # 2-3 years
            "poor": [0, 2],          # <2 years
        }
    ),
    "delivery": AnalysisDimension(
        name="delivery",
        display_name="交付响应",
        weight=0.25,
        keywords=["交货", "交付", "响应时间", "小时内", "到场", "上门"],
        extract_patterns=[r"(\d+)\s*小时", r"(\d+)\s*天"],
        risk_thresholds={
            "excellent": [0, 2],     # <2 hours
            "good": [2, 8],          # 2-8 hours
            "medium": [8, 24],       # 8-24 hours
            "poor": [24, 999],       # >24 hours
        }
    ),
    "training": AnalysisDimension(
        name="training",
        display_name="培训支持",
        weight=0.20,
        keywords=["培训", "技术指导", "操作培训", "维护培训"],
        extract_patterns=[r"(\d+)\s*[天日]", r"(\d+)\s*人"],
        risk_thresholds={
            "excellent": [5, 999],   # 5+ days
            "good": [3, 5],          # 3-5 days
            "medium": [1, 3],        # 1-3 days
            "poor": [0, 1],          # <1 day
        }
    ),
    "financial": AnalysisDimension(
        name="financial",
        display_name="商务条款",
        weight=0.20,
        keywords=["付款", "预付", "尾款", "保证金", "违约金", "报价"],
        extract_patterns=[r"(\d+)%", r"(\d+)\s*万元"],
        risk_thresholds={
            "high_prepayment": [50, 999],  # >50% prepayment
            "medium_prepayment": [30, 50],  # 30-50% prepayment
            "low_prepayment": [0, 30],       # <30% prepayment
        }
    ),
    "technical": AnalysisDimension(
        name="technical",
        display_name="技术方案",
        weight=0.10,
        keywords=["技术参数", "规格", "性能指标", "技术方案", "功能"],
    ),
    "compliance": AnalysisDimension(
        name="compliance",
        display_name="合规承诺",
        weight=0.10,
        keywords=["承诺", "保证", "必须", "应当", "符合", "满足"],
    ),
}


# =============================================================================
# Analysis Results
# =============================================================================

@dataclass
class DimensionResult:
    """Analysis result for a single dimension."""

    dimension: str
    display_name: str
    chunks_found: int
    risks: list[AnnotationInsight] = field(default_factory=list)
    benefits: list[AnnotationInsight] = field(default_factory=list)
    info: list[AnnotationInsight] = field(default_factory=list)
    extracted_values: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # 0-100
    risk_level: str = "medium"  # "low", "medium", "high"
    summary: str = ""


@dataclass
class BidAnalysisReport:
    """Complete bid analysis report."""

    version_id: str
    bidder_name: str
    project_name: str
    dimensions: dict[str, DimensionResult]

    # Computed metrics
    overall_score: float  # 0-100
    total_risks: int
    total_benefits: int
    risk_level: str  # "low", "medium", "high"

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    chunks_analyzed: int = 0


# =============================================================================
# Main Analyzer
# =============================================================================

class BidAnalyzer:
    """Analyzes bid documents across multiple dimensions.

    Usage:
        analyzer = BidAnalyzer(conn)

        report = analyzer.analyze_version(
            version_id="xxx",
            bidder_name="Company ABC",
            dimensions=["warranty", "delivery", "training"]
        )

        print(report.to_markdown())
    """

    def __init__(self, conn):
        """Initialize analyzer.

        Args:
            conn: psycopg database connection
        """
        self.conn = conn

    def analyze_version(
        self,
        version_id: str,
        bidder_name: str = "Unknown",
        project_name: str = "Unknown Project",
        dimensions: list[str] | None = None,
    ) -> BidAnalysisReport:
        """Analyze a bid document across specified dimensions.

        Args:
            version_id: Document version ID
            bidder_name: Bidder/company name
            project_name: Project name
            dimensions: List of dimension names (default: all)

        Returns:
            Complete BidAnalysisReport
        """
        if dimensions is None:
            dimensions = list(ANALYSIS_DIMENSIONS.keys())

        dimension_results = {}
        total_risks = 0
        total_benefits = 0

        for dim_name in dimensions:
            dim_config = ANALYSIS_DIMENSIONS.get(dim_name)
            if not dim_config:
                logger.warning(f"Unknown dimension: {dim_name}")
                continue

            result = self._analyze_dimension(version_id, dim_config)
            dimension_results[dim_name] = result

            total_risks += len(result.risks)
            total_benefits += len(result.benefits)

        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(dimension_results)

        # Determine risk level
        risk_level = self._determine_risk_level(total_risks, total_benefits)

        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_results)

        # Count chunks analyzed
        chunks_analyzed = sum(
            d.chunks_found for d in dimension_results.values()
        )

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
            chunks_analyzed=chunks_analyzed,
        )

    def _analyze_dimension(
        self,
        version_id: str,
        dimension: AnalysisDimension,
    ) -> DimensionResult:
        """Analyze a single dimension.

        Args:
            version_id: Document version ID
            dimension: Dimension configuration

        Returns:
            DimensionResult with findings
        """
        # Search for chunks matching this dimension
        chunks = self._search_chunks(version_id, dimension.keywords)

        # Analyze each chunk
        risks = []
        benefits = []
        info = []
        extracted_values = {}

        for chunk in chunks:
            text = chunk.get("text_raw", "")

            # Generate insights using annotation_insights
            insights = analyze_chunk_for_insights(text, dimension.name)

            for insight in insights:
                if insight.category == "risk":
                    risks.append(insight)
                elif insight.category == "benefit":
                    benefits.append(insight)
                else:
                    info.append(insight)

        # Extract numerical values
        if dimension.extract_patterns:
            extracted_values = self._extract_values(
                chunks, dimension.extract_patterns
            )

        # Calculate dimension score
        score = self._calculate_dimension_score(
            dimension, risks, benefits, extracted_values
        )

        # Determine risk level for this dimension
        risk_level = self._determine_dimension_risk_level(
            dimension, risks, benefits, extracted_values
        )

        # Generate summary
        summary_parts = []
        if chunks:
            summary_parts.append(f"找到 {len(chunks)} 处相关内容")
        if benefits:
            summary_parts.append(f"{len(benefits)} 个优势点")
        if risks:
            summary_parts.append(f"{len(risks)} 个风险点")

        summary = "; ".join(summary_parts) if summary_parts else "未找到相关内容"

        return DimensionResult(
            dimension=dimension.name,
            display_name=dimension.display_name,
            chunks_found=len(chunks),
            risks=risks,
            benefits=benefits,
            info=info,
            extracted_values=extracted_values,
            score=score,
            risk_level=risk_level,
            summary=summary,
        )

    def _search_chunks(
        self,
        version_id: str,
        keywords: list[str],
    ) -> list[dict[str, Any]]:
        """Search for chunks matching any keyword.

        Args:
            version_id: Document version ID
            keywords: List of keywords to search

        Returns:
            List of matching chunks
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            # Build ILIKE query for each keyword
            conditions = " OR ".join(["text_raw ILIKE %s"] * len(keywords))
            params = [f"%{kw}%" for kw in keywords]

            cur.execute(
                f"""
                SELECT chunk_id, page_idx, text_raw, bbox, element_type
                FROM chunks
                WHERE version_id = %s
                AND ({conditions})
                LIMIT 50
                """,
                [version_id] + params,
            )
            return cur.fetchall()

    def _extract_values(
        self,
        chunks: list[dict[str, Any]],
        patterns: list[str],
    ) -> dict[str, Any]:
        """Extract numerical values using regex patterns.

        Args:
            chunks: List of chunks with text_raw
            patterns: List of regex patterns

        Returns:
            Dict with extracted values
        """
        extracted = {}

        for chunk in chunks:
            text = chunk.get("text_raw", "")

            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # Convert to int and track
                    values = [int(m) for m in matches]
                    if values:
                        key = f"pattern_{pattern}"
                        extracted[key] = {
                            "values": values,
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                        }

        return extracted

    def _calculate_dimension_score(
        self,
        dimension: AnalysisDimension,
        risks: list[AnnotationInsight],
        benefits: list[AnnotationInsight],
        extracted_values: dict[str, Any],
    ) -> float:
        """Calculate score (0-100) for a dimension.

        Base score: 50
        - Each benefit: +10
        - Each low-risk: +5
        - Each medium-risk: -5
        - Each high-risk: -15

        Adjustments based on extracted values (thresholds).
        """
        score = 50.0

        # Benefit/risk adjustments
        for insight in benefits:
            score += 10

        for insight in risks:
            if insight.risk_level == "high":
                score -= 15
            elif insight.risk_level == "medium":
                score -= 5
            else:
                score -= 2

        # Threshold-based adjustments
        if dimension.risk_thresholds and extracted_values:
            thresholds = dimension.risk_thresholds

            # Example logic for warranty/delivery
            if dimension.name == "warranty":
                max_years = max(
                    (v["max"] for v in extracted_values.values()),
                    default=0
                )
                if max_years >= thresholds["excellent"][0]:
                    score += 20
                elif max_years >= thresholds["good"][0]:
                    score += 10
                elif max_years < thresholds["poor"][1]:
                    score -= 20

            elif dimension.name == "delivery":
                min_hours = min(
                    (v["min"] for v in extracted_values.values()),
                    default=999
                )
                if min_hours <= thresholds["excellent"][1]:
                    score += 20
                elif min_hours <= thresholds["good"][1]:
                    score += 10
                elif min_hours > thresholds["poor"][0]:
                    score -= 20

            elif dimension.name == "training":
                days_values = [v["max"] for v in extracted_values.values()]
                total_days = sum(days_values) if days_values else 0
                if total_days >= thresholds["excellent"][0]:
                    score += 20
                elif total_days >= thresholds["good"][0]:
                    score += 10

        return max(0, min(100, score))

    def _determine_dimension_risk_level(
        self,
        dimension: AnalysisDimension,
        risks: list[AnnotationInsight],
        benefits: list[AnnotationInsight],
        extracted_values: dict[str, Any],
    ) -> str:
        """Determine risk level for a dimension."""
        high_risk_count = sum(1 for r in risks if r.risk_level == "high")

        if high_risk_count > 0:
            return "high"
        elif len(risks) > len(benefits):
            return "medium"
        else:
            return "low"

    def _calculate_overall_score(
        self,
        dimension_results: dict[str, DimensionResult],
    ) -> float:
        """Calculate weighted overall score."""
        total_weight = 0.0
        weighted_sum = 0.0

        for dim_name, result in dimension_results.items():
            dim_config = ANALYSIS_DIMENSIONS.get(dim_name)
            if dim_config:
                weight = dim_config.weight
                weighted_sum += result.score * weight
                total_weight += weight

        if total_weight == 0:
            return 50.0

        return weighted_sum / total_weight

    def _determine_risk_level(
        self,
        total_risks: int,
        total_benefits: int,
    ) -> str:
        """Determine overall risk level."""
        ratio = total_benefits / max(1, total_risks)

        if ratio >= 2:
            return "low"
        elif ratio >= 1:
            return "medium"
        else:
            return "high"

    def _generate_recommendations(
        self,
        dimension_results: dict[str, DimensionResult],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Analyze each dimension
        for dim_name, result in dimension_results.items():
            if result.risk_level == "high":
                if dim_name == "warranty":
                    recommendations.append(
                        f"[质保] 质保期偏短，建议确认是否可延长或增加服务承诺"
                    )
                elif dim_name == "delivery":
                    recommendations.append(
                        f"[交付] 响应时间较慢，建议确认紧急情况处理流程"
                    )
                elif dim_name == "training":
                    recommendations.append(
                        f"[培训] 培训时间不足，建议增加实操培训天数"
                    )
                elif dim_name == "financial":
                    recommendations.append(
                        f"[商务] 付款条件较为严格，建议评估资金压力"
                    )

            # Check for missing content
            if result.chunks_found == 0:
                if dim_name == "warranty":
                    recommendations.append(
                        "[质保] 文档未找到质保条款，需补充"
                    )
                elif dim_name == "delivery":
                    recommendations.append(
                        "[交付] 文档未明确交付时间，需确认"
                    )

        # Overall recommendations
        if not recommendations:
            recommendations.append("投标文档整体较为完整，建议按常规流程评审")

        return recommendations


# =============================================================================
# Multi-Version Comparison
# =============================================================================

def compare_versions(
    conn,
    version_ids: list[str],
    bidder_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare multiple bid versions side-by-side.

    Args:
        conn: Database connection
        version_ids: List of version IDs to compare
        bidder_names: Optional mapping of version_id to bidder name

    Returns:
        Comparison results with scores and rankings
    """
    analyzer = BidAnalyzer(conn)
    reports = []

    for vid in version_ids:
        bidder_name = (bidder_names or {}).get(vid, f"Bidder {vid[:8]}")
        report = analyzer.analyze_version(vid, bidder_name=bidder_name)
        reports.append(report)

    # Sort by overall score
    reports.sort(key=lambda r: r.overall_score, reverse=True)

    # Generate comparison table
    comparison = {
        "rankings": [
            {
                "rank": i + 1,
                "version_id": r.version_id,
                "bidder_name": r.bidder_name,
                "overall_score": r.overall_score,
                "risk_level": r.risk_level,
                "total_risks": r.total_risks,
                "total_benefits": r.total_benefits,
            }
            for i, r in enumerate(reports)
        ],
        "dimension_comparison": {},
        "recommendations": [],
    }

    # Dimension-wise comparison
    for dim_name in ANALYSIS_DIMENSIONS:
        dim_data = []
        for r in reports:
            if dim_name in r.dimensions:
                d = r.dimensions[dim_name]
                dim_data.append({
                    "bidder": r.bidder_name,
                    "score": d.score,
                    "risk_level": d.risk_level,
                    "chunks_found": d.chunks_found,
                })

        comparison["dimension_comparison"][dim_name] = sorted(
            dim_data, key=lambda x: x["score"], reverse=True
        )

    # Generate comparison recommendations
    winner = reports[0] if reports else None
    if winner:
        comparison["recommendations"].append(
            f"推荐中标：{winner.bidder_name}（综合评分：{winner.overall_score:.1f}）"
        )

    # Check for close scores (within 5 points)
    if len(reports) >= 2:
        top_score = reports[0].overall_score
        second_score = reports[1].overall_score
        if top_score - second_score < 5:
            comparison["recommendations"].append(
                f"注意：前两名得分接近（差距 {top_score - second_score:.1f}），建议详细对比"
            )

    return comparison
