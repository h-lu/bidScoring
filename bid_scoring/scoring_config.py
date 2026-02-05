"""
评分标准配置管理模块

支持从 YAML/JSON 配置文件加载评分标准，实现评分规则的灵活配置。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ResponseTimeCriteria:
    """响应时间评估标准"""
    excellent: float = 2.0      # 优秀标准（小时）
    standard: float = 24.0      # 标准（小时）
    unit: str = "hours"


@dataclass
class WarrantyCriteria:
    """质保期限评估标准"""
    excellent: float = 5.0      # 优秀标准（年）
    standard: float = 3.0       # 标准（年）
    unit: str = "years"


@dataclass
class OnSiteTimeCriteria:
    """到场时间评估标准"""
    excellent: float = 24.0     # 优秀标准（小时）
    standard: float = 48.0      # 标准（小时）
    unit: str = "hours"


@dataclass
class ServiceLevelCriteria:
    """服务等级评估标准"""
    response_time: ResponseTimeCriteria = field(default_factory=ResponseTimeCriteria)
    warranty_period: WarrantyCriteria = field(default_factory=WarrantyCriteria)
    on_site_time: OnSiteTimeCriteria = field(default_factory=OnSiteTimeCriteria)


@dataclass
class ScoringRuleConfig:
    """评分规则配置"""
    name: str
    min_score: int
    score_range: tuple[float, float]
    description: str


@dataclass
class RequiredFieldConfig:
    """必填字段配置"""
    name: str
    field_name: str
    weight: float = 1.0
    patterns: list[str] | None = None


@dataclass
class DimensionConfig:
    """评分维度配置"""
    weight: float
    description: str
    scoring_rules: list[ScoringRuleConfig]
    required_fields: list[RequiredFieldConfig]
    service_level_criteria: ServiceLevelCriteria | None = None
    scoring_weights: dict[str, float] | None = None


@dataclass
class GeneralConfig:
    """通用配置"""
    passing_threshold: float = 60.0
    confidence_threshold: float = 0.8
    default_conflict_strategy: str = "highest_confidence"
    auto_confirm_high_confidence: bool = True
    high_confidence_threshold: float = 0.9
    manual_review_threshold: float = 0.7


@dataclass
class ScoringStandards:
    """评分标准完整配置"""
    training_plan: DimensionConfig
    after_sales_service: DimensionConfig
    technical_solution: DimensionConfig | None = None
    general: GeneralConfig = field(default_factory=GeneralConfig)
    
    @classmethod
    def from_yaml(cls, filepath: str | Path) -> ScoringStandards:
        """从 YAML 文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str | Path) -> ScoringStandards:
        """从 JSON 文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringStandards:
        """从字典加载配置"""
        return cls(
            training_plan=_parse_dimension_config(data.get('training_plan', {})),
            after_sales_service=_parse_dimension_config(data.get('after_sales_service', {})),
            technical_solution=_parse_dimension_config(data.get('technical_solution')) if 'technical_solution' in data else None,
            general=_parse_general_config(data.get('general', {})),
        )
    
    def get_dimension_config(self, dimension_id: str) -> DimensionConfig | None:
        """获取指定维度的配置"""
        mapping = {
            'training': self.training_plan,
            'training_plan': self.training_plan,
            'after_sales': self.after_sales_service,
            'after_sales_service': self.after_sales_service,
        }
        return mapping.get(dimension_id)


def _parse_dimension_config(data: dict[str, Any] | None) -> DimensionConfig | None:
    """解析维度配置"""
    if data is None:
        return None
    
    # 解析评分规则
    scoring_rules = []
    for rule in data.get('scoring_rules', []):
        # 支持两种格式：min_score（售后）或 min_fields（培训）
        min_score = rule.get('min_score', rule.get('min_fields', 0))
        scoring_rules.append(ScoringRuleConfig(
            name=rule['name'],
            min_score=min_score,
            score_range=tuple(rule['score_range']),
            description=rule['description'],
        ))
    
    # 解析必填字段
    required_fields = [
        RequiredFieldConfig(
            name=field['name'],
            field_name=field['field_name'],
            weight=field.get('weight', 1.0),
            patterns=field.get('patterns'),
        )
        for field in data.get('required_fields', [])
    ]
    
    # 解析服务等级标准（如果有）
    service_level_criteria = None
    if 'service_level_criteria' in data:
        criteria_data = data['service_level_criteria']
        service_level_criteria = ServiceLevelCriteria(
            response_time=ResponseTimeCriteria(**criteria_data.get('response_time', {})),
            warranty_period=WarrantyCriteria(**criteria_data.get('warranty_period', {})),
            on_site_time=OnSiteTimeCriteria(**criteria_data.get('on_site_time', {})),
        )
    
    return DimensionConfig(
        weight=data.get('weight', 10.0),
        description=data.get('description', ''),
        scoring_rules=scoring_rules,
        required_fields=required_fields,
        service_level_criteria=service_level_criteria,
        scoring_weights=data.get('scoring_weights'),
    )


def _parse_general_config(data: dict[str, Any]) -> GeneralConfig:
    """解析通用配置"""
    evidence_validation = data.get('evidence_validation', {})
    return GeneralConfig(
        passing_threshold=data.get('passing_threshold', 60.0),
        confidence_threshold=data.get('confidence_threshold', 0.8),
        default_conflict_strategy=data.get('default_conflict_strategy', 'highest_confidence'),
        auto_confirm_high_confidence=evidence_validation.get('auto_confirm_high_confidence', True),
        high_confidence_threshold=evidence_validation.get('high_confidence_threshold', 0.9),
        manual_review_threshold=evidence_validation.get('manual_review_threshold', 0.7),
    )


# 全局配置实例
_global_config: ScoringStandards | None = None


def load_scoring_config(filepath: str | Path | None = None) -> ScoringStandards:
    """
    加载评分标准配置
    
    Args:
        filepath: 配置文件路径，默认从 config/scoring_standards.yaml 加载
    
    Returns:
        ScoringStandards 实例
    """
    global _global_config
    
    if filepath is None:
        # 默认配置文件路径
        project_root = Path(__file__).parent.parent
        filepath = project_root / 'config' / 'scoring_standards.yaml'
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"评分标准配置文件不存在: {filepath}")
    
    # 根据文件扩展名选择加载方式
    if filepath.suffix in ['.yaml', '.yml']:
        config = ScoringStandards.from_yaml(filepath)
    elif filepath.suffix == '.json':
        config = ScoringStandards.from_json(filepath)
    else:
        raise ValueError(f"不支持的配置文件格式: {filepath.suffix}")
    
    _global_config = config
    return config


def get_scoring_config() -> ScoringStandards:
    """
    获取当前评分标准配置
    
    如果未加载，则自动加载默认配置
    """
    global _global_config
    
    if _global_config is None:
        _global_config = load_scoring_config()
    
    return _global_config


def reload_scoring_config(filepath: str | Path | None = None) -> ScoringStandards:
    """重新加载评分标准配置"""
    global _global_config
    _global_config = None
    return load_scoring_config(filepath)


# 便捷的获取配置函数
def get_training_plan_config() -> DimensionConfig:
    """获取培训方案配置"""
    return get_scoring_config().training_plan


def get_after_sales_config() -> DimensionConfig:
    """获取售后服务配置"""
    return get_scoring_config().after_sales_service


def get_general_config() -> GeneralConfig:
    """获取通用配置"""
    return get_scoring_config().general
