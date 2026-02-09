"""Generate AI-style insights for PDF annotations in bid document analysis.

This module provides functions to analyze chunk content and generate
meaningful, actionable insights instead of just repeating the text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Risk keywords and their categories
RISK_PATTERNS = {
    "deadline": {
        "keywords": ["天", "小时", "工作日", "日内", "个月内", "周内"],
        "label": "时限承诺",
        "risk_level": "high",
    },
    "financial": {
        "keywords": ["万元", "元", "费用", "免费", "收费", "付款"],
        "label": "财务条款",
        "risk_level": "medium",
    },
    "compliance": {
        "keywords": ["承诺", "保证", "必须", "应当", "不得"],
        "label": "合规承诺",
        "risk_level": "low",
    },
    "vague": {
        "keywords": ["尽力", "尽快", "合理", "适当", "可能"],
        "label": "模糊条款",
        "risk_level": "high",
    },
}

# Benefit keywords for positive identification
BENEFIT_PATTERNS = {
    "warranty": {
        "keywords": ["年保修", "终生维修", "质保期", "免费维修"],
        "label": "质保优势",
    },
    "response": {
        "keywords": ["小时响应", "到场", "2小时", "24小时"],
        "label": "响应优势",
    },
    "training": {
        "keywords": ["培训", "技术指导", "操作培训"],
        "label": "培训支持",
    },
}


@dataclass
class AnnotationInsight:
    """Insight generated for a chunk annotation."""

    category: str  # e.g., "risk", "benefit", "question"
    title: str  # Short title
    content: str  # Full insight text
    risk_level: str | None = None  # "high", "medium", "low"


def analyze_chunk_for_insights(
    text: str,
    topic: str | None = None,
    context: dict[str, Any] | None = None,
) -> list[AnnotationInsight]:
    """Analyze a text chunk and generate annotation insights.

    Args:
        text: The chunk text to analyze
        topic: The topic category (warranty, delivery, etc.)
        context: Additional context like page number

    Returns:
        List of AnnotationInsight objects
    """
    insights = []

    if not text:
        return insights

    # Check for risk patterns
    for risk_type, pattern in RISK_PATTERNS.items():
        for keyword in pattern["keywords"]:
            if keyword in text:
                insights.append(AnnotationInsight(
                    category="risk",
                    title=pattern["label"],
                    content=f"检测到{pattern['label']}关键词：'{keyword}'。需确认具体数值和条件。",
                    risk_level=pattern["risk_level"],
                ))
                break  # Only add one insight per risk type

    # Check for benefit patterns
    for benefit_type, pattern in BENEFIT_PATTERNS.items():
        for keyword in pattern["keywords"]:
            if keyword in text:
                insights.append(AnnotationInsight(
                    category="benefit",
                    title=pattern["label"],
                    content=f"发现{pattern['label']}：'{keyword}'。可作为评分优势。",
                    risk_level=None,
                ))
                break

    # Specific analysis by topic
    if topic == "warranty":
        insights.extend(_analyze_warranty(text))
    elif topic == "delivery":
        insights.extend(_analyze_delivery(text))
    elif topic == "training":
        insights.extend(_analyze_training(text))

    # If no specific insights found, provide a general one
    if not insights:
        insights.append(AnnotationInsight(
            category="info",
            title="内容标记",
            content=f"已标记{topic}相关内容，建议进一步详细评审。",
            risk_level=None,
        ))

    return insights


def _analyze_warranty(text: str) -> list[AnnotationInsight]:
    """Analyze warranty-related content for insights."""
    insights = []

    # Check for warranty period
    years = re.findall(r'(\d+)\s*年', text)
    if years:
        max_years = max(int(y) for y in years)
        if max_years >= 5:
            insights.append(AnnotationInsight(
                category="benefit",
                title="长期质保",
                content=f"承诺{max_years}年质保，优于行业标准。重点强调此优势。",
                risk_level=None,
            ))
        elif max_years < 3:
            insights.append(AnnotationInsight(
                category="risk",
                title="质保期偏短",
                content=f"质保期仅{max_years}年，可能存在竞争劣势。建议确认是否可延长。",
                risk_level="medium",
            ))

    # Check for lifetime warranty
    if "终生" in text or "终身" in text or "永久" in text:
        insights.append(AnnotationInsight(
            category="benefit",
            title="终身质保",
            content="发现终身质保承诺，这是强有力的竞争优势。",
            risk_level=None,
        ))

    # Check for warranty scope
    if "整机" in text:
        insights.append(AnnotationInsight(
            category="info",
            title="质保范围",
            content="承诺整机质保，覆盖范围较广。",
            risk_level=None,
        ))

    return insights


def _analyze_delivery(text: str) -> list[AnnotationInsight]:
    """Analyze delivery/response time content for insights."""
    insights = []

    # Check response time
    hours = re.findall(r'(\d+)\s*小时', text)
    if hours:
        min_hours = min(int(h) for h in hours)
        if min_hours <= 2:
            insights.append(AnnotationInsight(
                category="benefit",
                title="快速响应",
                content=f"承诺{min_hours}小时内响应，响应速度较快。",
                risk_level=None,
            ))
        elif min_hours > 24:
            insights.append(AnnotationInsight(
                category="risk",
                title="响应较慢",
                content=f"响应时间{min_hours}小时，可能影响紧急情况处理。",
                risk_level="medium",
            ))

    # Check for on-site support
    if "现场" in text or "上门" in text:
        insights.append(AnnotationInsight(
            category="info",
            title="现场服务",
            content="包含现场/上门服务承诺。",
            risk_level=None,
        ))

    return insights


def _analyze_training(text: str) -> list[AnnotationInsight]:
    """Analyze training content for insights."""
    insights = []

    # Check for training days
    days = re.findall(r'(\d+)\s*[天日]', text)
    if days:
        total_days = sum(int(d) for d in days)
        if total_days >= 5:
            insights.append(AnnotationInsight(
                category="benefit",
                title="充足培训",
                content=f"提供{total_days}天培训，培训内容较全面。",
                risk_level=None,
            ))

    # Check for training content
    if "操作" in text or "使用" in text:
        insights.append(AnnotationInsight(
            category="info",
            title="操作培训",
            content="包含设备操作培训。",
            risk_level=None,
        ))

    if "维护" in text or "保养" in text:
        insights.append(AnnotationInsight(
            category="info",
            title="维护培训",
            content="包含设备维护保养培训。",
            risk_level=None,
        ))

    return insights


def format_insight_for_annotation(insight: AnnotationInsight) -> str:
    """Format insight for PDF annotation display.

    Args:
        insight: The insight to format

    Returns:
        Formatted string suitable for PDF annotation
    """
    parts = [insight.title]

    if insight.category == "risk":
        parts.insert(0, "⚠️")
    elif insight.category == "benefit":
        parts.insert(0, "✅")
    elif insight.category == "info":
        parts.insert(0, "ℹ️")

    if insight.risk_level:
        risk_map = {"high": "高风险", "medium": "中风险", "low": "低风险"}
        parts.append(f"({risk_map[insight.risk_level]})")

    return " | ".join(parts) + "\n" + insight.content


def generate_annotation_content(
    text: str,
    topic: str,
    max_length: int = 200,
) -> str:
    """Generate annotation content for a chunk.

    This is the main function to use when creating PDF annotations.
    It analyzes the text and generates a meaningful, actionable insight.

    Args:
        text: The chunk text to analyze
        topic: The topic category
        max_length: Maximum length of the annotation

    Returns:
        A formatted string suitable for PDF annotation popup
    """
    insights = analyze_chunk_for_insights(text, topic)

    if not insights:
        return f"【{topic}】{text[:max_length-10]}..."

    # Use the most important insight (prioritize risks, then benefits)
    prioritized = sorted(insights, key=lambda x: (
        0 if x.category == "risk" else 1 if x.category == "benefit" else 2,
        {"high": 0, "medium": 1, "low": 2}.get(x.risk_level or "", 3),
    ))

    best_insight = prioritized[0]
    formatted = format_insight_for_annotation(best_insight)

    # Add original text preview for reference
    preview = text[:50] + "..." if len(text) > 50 else text
    result = f"{formatted}\n\n原文: {preview}"

    if len(result) > max_length:
        result = result[:max_length-3] + "..."

    return result
