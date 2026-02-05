"""
评分标准配置管理模块 V2 - 使用 Pydantic 验证

改进点：
- 使用 Pydantic BaseModel 进行运行时验证
- 支持不可变配置 (frozen=True)
- 提供友好的错误信息
- 自动生成 JSON Schema
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigurationError(Exception):
    """配置错误基类"""
    pass


class ResponseTimeCriteria(BaseModel):
    """响应时间评估标准"""
    model_config = {"frozen": True}  # 不可变配置
    
    excellent: float = Field(default=2.0, gt=0, description="优秀标准（小时）")
    standard: float = Field(default=24.0, gt=0, description="标准（小时）")
    unit: Literal["hours", "minutes", "days"] = Field(default="hours", description="时间单位")
    
    @field_validator('standard')
    @classmethod
    def standard_must_be_greater_than_excellent(cls, v: float, info) -> float:
        """验证标准值必须大于优秀值"""
        if 'excellent' in info.data and v <= info.data['excellent']:
            raise ValueError(f'standard ({v}) 必须大于 excellent ({info.data["excellent"]})')
        return v


class WarrantyCriteria(BaseModel):
    """质保期限评估标准"""
    model_config = {"frozen": True}
    
    excellent: float = Field(default=5.0, gt=0, description="优秀标准（年）")
    standard: float = Field(default=3.0, gt=0, description="标准（年）")
    unit: Literal["years", "months"] = Field(default="years", description="时间单位")
    
    @field_validator('standard')
    @classmethod
    def standard_must_be_less_than_excellent(cls, v: float, info) -> float:
        """验证标准值必须小于优秀值（质保期越长越好）"""
        if 'excellent' in info.data and v >= info.data['excellent']:
            raise ValueError(f'standard ({v}) 必须小于 excellent ({info.data["excellent"]})')
        return v


class OnSiteTimeCriteria(BaseModel):
    """到场时间评估标准"""
    model_config = {"frozen": True}
    
    excellent: float = Field(default=24.0, gt=0, description="优秀标准（小时）")
    standard: float = Field(default=48.0, gt=0, description="标准（小时）")
    unit: Literal["hours", "minutes", "days"] = Field(default="hours", description="时间单位")
    
    @field_validator('standard')
    @classmethod
    def standard_must_be_greater_than_excellent(cls, v: float, info) -> float:
        """验证标准值必须大于优秀值"""
        if 'excellent' in info.data and v <= info.data['excellent']:
            raise ValueError(f'standard ({v}) 必须大于 excellent ({info.data["excellent"]})')
        return v


class ServiceLevelCriteria(BaseModel):
    """服务等级评估标准"""
    model_config = {"frozen": True}
    
    response_time: ResponseTimeCriteria = Field(default_factory=ResponseTimeCriteria)
    warranty_period: WarrantyCriteria = Field(default_factory=WarrantyCriteria)
    on_site_time: OnSiteTimeCriteria = Field(default_factory=OnSiteTimeCriteria)


class ScoringRuleConfig(BaseModel):
    """评分规则配置"""
    model_config = {"frozen": True}
    
    name: str = Field(..., min_length=1, description="规则名称")
    min_score: int = Field(default=0, ge=0, description="最低得分门槛")
    min_fields: int | None = Field(default=None, ge=0, description="最低字段数（兼容V1）")
    score_range: Tuple[float, float] = Field(..., description="分数范围 (min, max)")
    description: str = Field(..., min_length=1, description="规则描述")
    
    @model_validator(mode='before')
    @classmethod
    def convert_min_fields_to_min_score(cls, data: Any) -> Any:
        """转换 V1 的 min_fields 到 V2 的 min_score"""
        if isinstance(data, dict):
            # 如果提供了 min_fields 但没有 min_score，使用 min_fields
            if 'min_fields' in data and 'min_score' not in data:
                data['min_score'] = data['min_fields']
            # 删除 min_fields 以避免混淆
            data.pop('min_fields', None)
        return data
    
    @field_validator('score_range')
    @classmethod
    def validate_score_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        """验证分数范围"""
        min_score, max_score = v
        if min_score < 0 or max_score < 0:
            raise ValueError('分数不能为负数')
        if min_score > max_score:
            raise ValueError(f'最小分数 ({min_score}) 不能大于最大分数 ({max_score})')
        return v


class RequiredFieldConfig(BaseModel):
    """必填字段配置"""
    model_config = {"frozen": True}
    
    name: str = Field(..., min_length=1, description="字段标识名")
    field_name: str = Field(..., min_length=1, description="字段显示名")
    weight: float = Field(default=1.0, gt=0, description="字段权重")
    patterns: list[str] | None = Field(default=None, description="匹配模式列表")


class DimensionConfig(BaseModel):
    """评分维度配置"""
    model_config = {"frozen": True}
    
    weight: float = Field(default=10.0, gt=0, description="维度权重")
    description: str = Field(default="", description="维度描述")
    scoring_rules: list[ScoringRuleConfig] = Field(default_factory=list, description="评分规则列表")
    required_fields: list[RequiredFieldConfig] = Field(default_factory=list, description="必填字段列表")
    service_level_criteria: ServiceLevelCriteria | None = Field(default=None, description="服务等级标准")
    scoring_weights: dict[str, float] = Field(default_factory=dict, description="各项评分权重")
    
    @field_validator('scoring_rules')
    @classmethod
    def validate_scoring_rules(cls, v: list[ScoringRuleConfig]) -> list[ScoringRuleConfig]:
        """验证评分规则不能为空"""
        if not v:
            raise ValueError('评分规则列表不能为空')
        return v
    
    @field_validator('required_fields')
    @classmethod
    def validate_required_fields(cls, v: list[RequiredFieldConfig]) -> list[RequiredFieldConfig]:
        """验证必填字段不能为空"""
        if not v:
            raise ValueError('必填字段列表不能为空')
        return v


class GeneralConfig(BaseModel):
    """通用配置"""
    model_config = {"frozen": True}
    
    passing_threshold: float = Field(default=60.0, ge=0, le=100, description="通过线（百分比）")
    confidence_threshold: float = Field(default=0.8, ge=0, le=1, description="置信度阈值")
    default_conflict_strategy: str = Field(default="highest_confidence", description="默认冲突解决策略")
    auto_confirm_high_confidence: bool = Field(default=True, description="自动确认高置信度证据")
    high_confidence_threshold: float = Field(default=0.9, ge=0, le=1, description="高置信度阈值")
    manual_review_threshold: float = Field(default=0.7, ge=0, le=1, description="人工审核阈值")


class ScoringStandards(BaseModel):
    """评分标准完整配置"""
    model_config = {"frozen": True}  # 配置加载后不可变
    
    training_plan: DimensionConfig = Field(..., description="培训方案评分配置")
    after_sales_service: DimensionConfig = Field(..., description="售后服务评分配置")
    technical_solution: DimensionConfig | None = Field(default=None, description="技术方案评分配置")
    general: GeneralConfig = Field(default_factory=GeneralConfig, description="通用配置")
    
    @model_validator(mode='after')
    def validate_dimension_weights(self):
        """验证维度权重是否合理"""
        total_weight = self.training_plan.weight + self.after_sales_service.weight
        if self.technical_solution:
            total_weight += self.technical_solution.weight
        
        if total_weight <= 0:
            raise ValueError(f'总权重必须大于0，当前: {total_weight}')
        
        return self
    
    @classmethod
    def from_yaml(cls, filepath: str | Path) -> ScoringStandards:
        """从 YAML 文件加载配置（带验证）"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ConfigurationError(f"配置文件不存在: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML 解析错误: {e}")
        except Exception as e:
            raise ConfigurationError(f"读取配置文件失败: {e}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str | Path) -> ScoringStandards:
        """从 JSON 文件加载配置（带验证）"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise ConfigurationError(f"配置文件不存在: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"JSON 解析错误: {e}")
        except Exception as e:
            raise ConfigurationError(f"读取配置文件失败: {e}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringStandards:
        """从字典加载配置（带验证）"""
        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ConfigurationError(f"配置验证失败: {e}")
    
    def get_dimension_config(self, dimension_id: str) -> DimensionConfig | None:
        """获取指定维度的配置"""
        mapping = {
            'training': self.training_plan,
            'training_plan': self.training_plan,
            'after_sales': self.after_sales_service,
            'after_sales_service': self.after_sales_service,
            'technical': self.technical_solution,
            'technical_solution': self.technical_solution,
        }
        return mapping.get(dimension_id)
    
    def save_schema(self, filepath: str | Path) -> None:
        """保存 JSON Schema 供外部验证使用"""
        schema = self.model_json_schema()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)


# 线程锁保护全局配置
_config_lock = threading.Lock()
_global_config: ScoringStandards | None = None


def load_scoring_config(filepath: str | Path | None = None) -> ScoringStandards:
    """
    加载评分标准配置
    
    Args:
        filepath: 配置文件路径，默认从 config/scoring_standards.yaml 加载
    
    Returns:
        ScoringStandards 实例
    
    Raises:
        ConfigurationError: 配置加载或验证失败
    """
    global _global_config
    
    if filepath is None:
        # 默认配置文件路径
        project_root = Path(__file__).parent.parent
        filepath = project_root / 'config' / 'scoring_standards.yaml'
    
    filepath = Path(filepath)
    
    # 根据文件扩展名选择加载方式
    if filepath.suffix in ['.yaml', '.yml']:
        config = ScoringStandards.from_yaml(filepath)
    elif filepath.suffix == '.json':
        config = ScoringStandards.from_json(filepath)
    else:
        raise ConfigurationError(f"不支持的配置文件格式: {filepath.suffix}，请使用 .yaml 或 .json")
    
    with _config_lock:
        _global_config = config
    
    return config


@lru_cache(maxsize=1)
def get_scoring_config_cached() -> ScoringStandards:
    """
    获取缓存的评分标准配置
    
    使用 lru_cache 确保配置不可变且线程安全
    """
    return load_scoring_config()


def get_scoring_config() -> ScoringStandards:
    """
    获取当前评分标准配置
    
    如果未加载，则自动加载默认配置
    """
    global _global_config
    
    with _config_lock:
        if _global_config is None:
            _global_config = load_scoring_config()
        return _global_config


def reload_scoring_config(filepath: str | Path | None = None) -> ScoringStandards:
    """重新加载评分标准配置"""
    global _global_config
    
    with _config_lock:
        _global_config = None
        get_scoring_config_cached.cache_clear()
    
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


# 向后兼容：保留 V1 的函数签名
try:
    # 尝试导入 V1 模块的符号
    from .scoring_config import (
        ResponseTimeCriteria as _V1ResponseTimeCriteria,
        WarrantyCriteria as _V1WarrantyCriteria,
    )
except ImportError:
    pass
