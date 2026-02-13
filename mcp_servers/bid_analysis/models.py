"""Data models and dimension configuration for bid analysis."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mcp_servers.annotation_insights import AnnotationInsight

logger = logging.getLogger(__name__)

DEFAULT_SCORING_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "scoring_rules.yaml"
)


@dataclass
class AnalysisDimension:
    """Configuration for a single analysis dimension."""

    name: str
    display_name: str
    weight: float
    keywords: list[str]
    extract_patterns: list[str] | None = None
    risk_thresholds: dict[str, Any] | None = None


def load_analysis_dimensions(
    config_path: str | Path | None = None,
) -> dict[str, AnalysisDimension]:
    """Load analysis dimension rules from YAML with safe fallback."""

    selected_path = _resolve_rules_path(config_path)
    raw = _load_rules_file(selected_path)
    dimensions_payload = raw.get("dimensions")
    if not isinstance(dimensions_payload, dict):
        logger.warning("Invalid scoring rules schema at %s, use defaults", selected_path)
        dimensions_payload = _default_dimensions_payload()

    dimensions = _parse_dimensions(dimensions_payload)
    if dimensions:
        return dimensions

    logger.warning("No valid scoring dimensions found at %s, use defaults", selected_path)
    return _parse_dimensions(_default_dimensions_payload())


def _resolve_rules_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return Path(config_path)
    env_path = os.getenv("BID_SCORING_RULES_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_SCORING_RULES_PATH


def _load_rules_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Scoring rules file missing at %s, use defaults", path)
        return {"dimensions": _default_dimensions_payload()}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.error("Invalid YAML in scoring rules %s: %s", path, exc)
        return {"dimensions": _default_dimensions_payload()}
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read scoring rules %s: %s", path, exc)
        return {"dimensions": _default_dimensions_payload()}

    if not isinstance(payload, dict):
        return {"dimensions": _default_dimensions_payload()}
    return payload


def _parse_dimensions(payload: dict[str, Any]) -> dict[str, AnalysisDimension]:
    dimensions: dict[str, AnalysisDimension] = {}
    for name, cfg in payload.items():
        if not isinstance(cfg, dict):
            logger.warning("Skip invalid dimension '%s': expected object", name)
            continue

        display_name = cfg.get("display_name")
        weight = cfg.get("weight")
        keywords = cfg.get("keywords")
        if not isinstance(display_name, str):
            logger.warning("Skip dimension '%s': missing display_name", name)
            continue
        if not isinstance(weight, (int, float)):
            logger.warning("Skip dimension '%s': invalid weight", name)
            continue
        if not isinstance(keywords, list) or not all(
            isinstance(item, str) for item in keywords
        ):
            logger.warning("Skip dimension '%s': invalid keywords", name)
            continue

        extract_patterns_raw = cfg.get("extract_patterns")
        extract_patterns = None
        if isinstance(extract_patterns_raw, list) and all(
            isinstance(item, str) for item in extract_patterns_raw
        ):
            extract_patterns = extract_patterns_raw

        risk_thresholds_raw = cfg.get("risk_thresholds")
        risk_thresholds = risk_thresholds_raw if isinstance(risk_thresholds_raw, dict) else None

        dimensions[name] = AnalysisDimension(
            name=name,
            display_name=display_name,
            weight=float(weight),
            keywords=keywords,
            extract_patterns=extract_patterns,
            risk_thresholds=risk_thresholds,
        )

    return dimensions


def _default_dimensions_payload() -> dict[str, dict[str, Any]]:
    return {
        "warranty": {
            "display_name": "质保售后",
            "weight": 0.25,
            "keywords": ["质保", "保修", "免费维修", "终身服务", "售后服务"],
            "extract_patterns": [r"(\d+)\s*年", r"(\d+)\s*个月"],
            "risk_thresholds": {
                "excellent": [5, 999],
                "good": [3, 5],
                "medium": [2, 3],
                "poor": [0, 2],
            },
        },
        "delivery": {
            "display_name": "交付响应",
            "weight": 0.25,
            "keywords": ["交货", "交付", "响应时间", "小时内", "到场", "上门"],
            "extract_patterns": [r"(\d+)\s*小时", r"(\d+)\s*天"],
            "risk_thresholds": {
                "excellent": [0, 2],
                "good": [2, 8],
                "medium": [8, 24],
                "poor": [24, 999],
            },
        },
        "training": {
            "display_name": "培训支持",
            "weight": 0.20,
            "keywords": ["培训", "技术指导", "操作培训", "维护培训"],
            "extract_patterns": [r"(\d+)\s*[天日]", r"(\d+)\s*人"],
            "risk_thresholds": {
                "excellent": [5, 999],
                "good": [3, 5],
                "medium": [1, 3],
                "poor": [0, 1],
            },
        },
        "financial": {
            "display_name": "商务条款",
            "weight": 0.20,
            "keywords": ["付款", "预付", "尾款", "保证金", "违约金", "报价"],
            "extract_patterns": [r"(\d+)%", r"(\d+)\s*万元"],
            "risk_thresholds": {
                "high_prepayment": [50, 999],
                "medium_prepayment": [30, 50],
                "low_prepayment": [0, 30],
            },
        },
        "technical": {
            "display_name": "技术方案",
            "weight": 0.10,
            "keywords": ["技术参数", "规格", "性能指标", "技术方案", "功能"],
        },
        "compliance": {
            "display_name": "合规承诺",
            "weight": 0.10,
            "keywords": ["承诺", "保证", "必须", "应当", "符合", "满足"],
        },
    }


ANALYSIS_DIMENSIONS = load_analysis_dimensions()


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
