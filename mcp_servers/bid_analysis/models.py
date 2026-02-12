"""Data models and dimension configuration for bid analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mcp_servers.annotation_insights import AnnotationInsight


@dataclass
class AnalysisDimension:
    """Configuration for a single analysis dimension."""

    name: str
    display_name: str
    weight: float
    keywords: list[str]
    extract_patterns: list[str] | None = None
    risk_thresholds: dict[str, Any] | None = None


ANALYSIS_DIMENSIONS = {
    "warranty": AnalysisDimension(
        name="warranty",
        display_name="质保售后",
        weight=0.25,
        keywords=["质保", "保修", "免费维修", "终身服务", "售后服务"],
        extract_patterns=[r"(\d+)\s*年", r"(\d+)\s*个月"],
        risk_thresholds={
            "excellent": [5, 999],
            "good": [3, 5],
            "medium": [2, 3],
            "poor": [0, 2],
        },
    ),
    "delivery": AnalysisDimension(
        name="delivery",
        display_name="交付响应",
        weight=0.25,
        keywords=["交货", "交付", "响应时间", "小时内", "到场", "上门"],
        extract_patterns=[r"(\d+)\s*小时", r"(\d+)\s*天"],
        risk_thresholds={
            "excellent": [0, 2],
            "good": [2, 8],
            "medium": [8, 24],
            "poor": [24, 999],
        },
    ),
    "training": AnalysisDimension(
        name="training",
        display_name="培训支持",
        weight=0.20,
        keywords=["培训", "技术指导", "操作培训", "维护培训"],
        extract_patterns=[r"(\d+)\s*[天日]", r"(\d+)\s*人"],
        risk_thresholds={
            "excellent": [5, 999],
            "good": [3, 5],
            "medium": [1, 3],
            "poor": [0, 1],
        },
    ),
    "financial": AnalysisDimension(
        name="financial",
        display_name="商务条款",
        weight=0.20,
        keywords=["付款", "预付", "尾款", "保证金", "违约金", "报价"],
        extract_patterns=[r"(\d+)%", r"(\d+)\s*万元"],
        risk_thresholds={
            "high_prepayment": [50, 999],
            "medium_prepayment": [30, 50],
            "low_prepayment": [0, 30],
        },
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
    score: float = 0.0
    risk_level: str = "medium"
    summary: str = ""
    evidence_warnings: list[str] = field(default_factory=list)


@dataclass
class BidAnalysisReport:
    """Complete bid analysis report."""

    version_id: str
    bidder_name: str
    project_name: str
    dimensions: dict[str, DimensionResult]
    overall_score: float
    total_risks: int
    total_benefits: int
    risk_level: str
    recommendations: list[str] = field(default_factory=list)
    evidence_warnings: list[str] = field(default_factory=list)
    analyzed_at: datetime = field(default_factory=datetime.now)
    chunks_analyzed: int = 0
