"""
回标分析评分维度 Schema - Pydantic 实现

提供结构化证据提取、多源冲突解决和评分规则引擎。

设计原则:
    - 使用 Pydantic v2 进行严格的数据验证
    - 策略模式实现可序列化的规则引擎
    - 多层级冲突解决策略
    - 完整的审计追踪支持
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Generic, Literal, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

# =============================================================================
# 基础类型定义
# =============================================================================

T = TypeVar("T")


class BoundingBox(BaseModel):
    """PDF 边界框坐标 (PDF 坐标系)
    
    Attributes:
        x1: 左上角 x 坐标
        y1: 左上角 y 坐标
        x2: 右下角 x 坐标
        y2: 右下角 y 坐标
    """
    model_config = ConfigDict(frozen=True)
    
    x1: float = Field(..., description="左上角 x 坐标")
    y1: float = Field(..., description="左上角 y 坐标")
    x2: float = Field(..., description="右下角 x 坐标")
    y2: float = Field(..., description="右下角 y 坐标")
    
    @field_validator("x2", "y2")
    @classmethod
    def validate_coordinates(cls, v: float, info) -> float:
        """验证坐标顺序正确"""
        if info.field_name == "x2":
            x1 = info.data.get("x1")
            if x1 is not None and v < x1:
                raise ValueError("x2 必须大于等于 x1")
        elif info.field_name == "y2":
            y1 = info.data.get("y1")
            if y1 is not None and v < y1:
                raise ValueError("y2 必须大于等于 y1")
        return v
    
    @computed_field
    @property
    def width(self) -> float:
        """计算宽度"""
        return self.x2 - self.x1
    
    @computed_field
    @property
    def height(self) -> float:
        """计算高度"""
        return self.y2 - self.y1
    
    @computed_field
    @property
    def area(self) -> float:
        """计算面积"""
        return self.width * self.height
    
    def to_dict(self) -> dict[str, float]:
        """转换为字典"""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }
    
    @classmethod
    def from_list(cls, bbox_list: list[float]) -> BoundingBox:
        """从列表创建边界框
        
        Args:
            bbox_list: 包含 [x1, y1, x2, y2] 的列表
        
        Returns:
            BoundingBox 实例。如果列表长度不足4，使用提供的值，
            缺失的坐标会设置为与对应起始坐标相同（确保验证通过）。
        """
        x1 = bbox_list[0] if len(bbox_list) > 0 else 0.0
        y1 = bbox_list[1] if len(bbox_list) > 1 else 0.0
        x2 = bbox_list[2] if len(bbox_list) > 2 else x1  # 确保 x2 >= x1
        y2 = bbox_list[3] if len(bbox_list) > 3 else y1  # 确保 y2 >= y1
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


# =============================================================================
# 枚举类型定义
# =============================================================================

class ValidationStatus(str, Enum):
    """证据验证状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class ConflictResolutionStrategy(str, Enum):
    """冲突解决策略
    
    - HIGHEST_CONFIDENCE: 选择置信度最高的证据
    - FIRST: 选择第一个证据
    - MANUAL: 需要人工审核
    - MAJORITY_VOTE: 多数投票（相同值中最自信的）
    - SOURCE_AUTHORITY: 基于来源权威性加权
    - TEMPORAL_RECENCY: 选择最新的证据
    - WEIGHTED_AVERAGE: 加权平均（仅适用于数值型）
    """
    HIGHEST_CONFIDENCE = "highest_confidence"
    FIRST = "first"
    MANUAL = "manual"
    MAJORITY_VOTE = "majority_vote"
    SOURCE_AUTHORITY = "source_authority"
    TEMPORAL_RECENCY = "temporal_recency"
    WEIGHTED_AVERAGE = "weighted_average"


class ServiceLevel(str, Enum):
    """服务等级评估结果"""
    EXCELLENT = "excellent"
    STANDARD = "standard"
    POOR = "poor"
    UNKNOWN = "unknown"


class CompletenessLevel(str, Enum):
    """完整性评估结果"""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MINIMAL = "minimal"
    EMPTY = "empty"


# =============================================================================
# 证据模型
# =============================================================================

class EvidenceItem(BaseModel):
    """证据项 - 关联到原文的具体位置
    
    这是所有证据类型的基类，提供：
    - 精确的 PDF 位置追踪 (page_idx + bbox)
    - 置信度评估
    - 验证工作流
    - 审计追踪
    
    Example:
        >>> evidence = EvidenceItem(
        ...     field_name="培训时长",
        ...     field_value="2天",
        ...     source_text="培训时长：2天",
        ...     page_idx=67,
        ...     bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
        ...     chunk_id="test-chunk-001",
        ...     confidence=0.95
        ... )
        >>> evidence.is_reliable()
        True
    """
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
    )
    
    # 核心字段
    field_name: str = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="字段名，如 '培训时长'"
    )
    field_value: str = Field(
        ..., 
        min_length=1,
        description="字段值，如 '2天'"
    )
    source_text: str = Field(
        ..., 
        min_length=1,
        description="原文片段"
    )
    
    # 位置信息
    page_idx: int = Field(
        ..., 
        ge=0, 
        description="页码 (从0开始)"
    )
    bbox: BoundingBox = Field(
        ..., 
        description="边界框坐标"
    )
    chunk_id: str = Field(
        ..., 
        min_length=1,
        description="关联的 chunk ID"
    )
    
    # 置信度
    confidence: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="提取置信度 (0-1)"
    )
    
    # 验证状态
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.PENDING,
        description="验证状态"
    )
    validation_notes: str | None = Field(
        default=None,
        max_length=1000,
        description="验证备注"
    )
    
    # 审计字段
    extracted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="提取时间"
    )
    validated_at: datetime | None = Field(
        default=None,
        description="验证时间"
    )
    
    @field_validator("field_value", "source_text")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """验证字段不为空"""
        if not v.strip():
            raise ValueError("字段值不能为空")
        return v.strip()
    
    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, v: str) -> str:
        """验证 chunk_id 格式"""
        if not v.strip():
            raise ValueError("chunk_id 不能为空")
        return v.strip()
    
    def is_reliable(self, threshold: float = 0.8) -> bool:
        """判断证据是否可靠（基于置信度阈值）
        
        Args:
            threshold: 置信度阈值 (默认 0.8)
        
        Returns:
            是否可靠
        """
        return self.confidence >= threshold
    
    def confirm(self, notes: str | None = None) -> None:
        """确认证据有效
        
        Args:
            notes: 确认备注
        """
        self.validation_status = ValidationStatus.CONFIRMED
        self.validation_notes = notes
        self.validated_at = datetime.now(timezone.utc)
    
    def reject(self, reason: str) -> None:
        """拒绝证据
        
        Args:
            reason: 拒绝原因
        """
        self.validation_status = ValidationStatus.REJECTED
        self.validation_notes = reason
        self.validated_at = datetime.now(timezone.utc)
    
    def reset_validation(self) -> None:
        """重置验证状态"""
        self.validation_status = ValidationStatus.PENDING
        self.validation_notes = None
        self.validated_at = None
    
    @computed_field
    @property
    def is_validated(self) -> bool:
        """是否已完成验证"""
        return self.validation_status != ValidationStatus.PENDING


# =============================================================================
# 结构化证据子类
# =============================================================================

class DurationEvidence(EvidenceItem):
    """时长证据（带数值解析）
    
    用于培训时长、质保期限等时间相关字段。
    
    Example:
        >>> evidence = DurationEvidence(
        ...     field_name="培训时长",
        ...     field_value="2天",
        ...     raw_value="培训时长：2天",
        ...     days=2.0,
        ...     page_idx=1,
        ...     bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        ...     chunk_id="chunk-1",
        ...     confidence=0.9
        ... )
    """
    raw_value: str = Field(
        default="",
        description="原始文本值"
    )
    days: float | None = Field(
        default=None,
        ge=0,
        description="天数"
    )
    hours: float | None = Field(
        default=None,
        ge=0,
        description="小时数"
    )
    
    @model_validator(mode="after")
    def validate_has_duration(self) -> DurationEvidence:
        """验证至少有一个时长值"""
        if self.days is None and self.hours is None:
            raise ValueError("必须提供 days 或 hours 至少一个时长值")
        return self
    
    @computed_field
    @property
    def total_hours(self) -> float:
        """总小时数"""
        total = 0.0
        if self.days is not None:
            total += self.days * 24
        if self.hours is not None:
            total += self.hours
        return total


class ResponseTimeEvidence(EvidenceItem):
    """响应时间证据
    
    用于售后服务响应时间提取。
    """
    raw_value: str = Field(default="", description="原始文本值")
    response_hours: float | None = Field(
        default=None,
        ge=0,
        description="响应时间（小时）"
    )
    on_site_hours: float | None = Field(
        default=None,
        ge=0,
        description="到达现场时间（小时）"
    )
    
    @computed_field
    @property
    def is_emergency_response(self) -> bool:
        """是否为紧急响应（2小时内）"""
        return self.response_hours is not None and self.response_hours <= 2


class WarrantyEvidence(EvidenceItem):
    """质保期限证据"""
    raw_value: str = Field(default="", description="原始文本值")
    years: float | None = Field(
        default=None,
        ge=0,
        description="年数"
    )
    months: int | None = Field(
        default=None,
        ge=0,
        le=120,
        description="月数"
    )
    
    @computed_field
    @property
    def total_months(self) -> float:
        """总月数"""
        total = 0.0
        if self.years is not None:
            total += self.years * 12
        if self.months is not None:
            total += self.months
        return total


class ServiceFeeEvidence(EvidenceItem):
    """服务费用证据"""
    raw_value: str = Field(default="", description="原始文本值")
    is_free: bool = Field(default=False, description="是否免费")
    fee_percentage: Decimal | None = Field(
        default=None,
        ge=0,
        le=100,
        description="费用百分比"
    )
    annual_fee: Decimal | None = Field(
        default=None,
        ge=0,
        description="年度费用"
    )


class PersonnelEvidence(EvidenceItem):
    """人员资质证据"""
    raw_value: str = Field(default="", description="原始文本值")
    qualification_level: str | None = Field(
        default=None,
        description="资质等级"
    )
    years_experience: int | None = Field(
        default=None,
        ge=0,
        description="工作年限"
    )
    certification: str | None = Field(
        default=None,
        description="认证证书"
    )


# =============================================================================
# 冲突解决策略实现
# =============================================================================

class ConflictResolver(ABC, BaseModel):
    """冲突解决器基类
    
    所有冲突解决策略必须继承此类，并实现 resolve 方法。
    """
    model_config = ConfigDict(validate_assignment=True)
    
    @abstractmethod
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        """解决冲突
        
        Args:
            candidates: 候选证据列表
        
        Returns:
            选中的证据，或 None（需要人工处理）
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """获取策略名称"""
        pass


class HighestConfidenceResolver(ConflictResolver):
    """最高置信度策略"""
    
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.confidence)
    
    def get_strategy_name(self) -> str:
        return ConflictResolutionStrategy.HIGHEST_CONFIDENCE


class FirstResolver(ConflictResolver):
    """第一个证据策略"""
    
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        return candidates[0] if candidates else None
    
    def get_strategy_name(self) -> str:
        return ConflictResolutionStrategy.FIRST


class MajorityVoteResolver(ConflictResolver):
    """多数投票策略
    
    选择出现次数最多的值，如果有多个相同频次的值，
    选择其中置信度最高的。
    """
    
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        # 统计每个值的出现次数
        value_counts = Counter(e.field_value for e in candidates)
        max_count = max(value_counts.values())
        
        # 找出出现次数最多的值
        majority_values = [
            v for v, c in value_counts.items() if c == max_count
        ]
        
        # 在多数值中选择置信度最高的证据
        majority_candidates = [
            e for e in candidates if e.field_value in majority_values
        ]
        
        return max(majority_candidates, key=lambda e: e.confidence)
    
    def get_strategy_name(self) -> str:
        return ConflictResolutionStrategy.MAJORITY_VOTE


class SourceAuthorityResolver(ConflictResolver):
    """来源权威性策略
    
    基于来源的权威性分数加权选择。
    """
    authority_scores: dict[str, float] = Field(
        default_factory=dict,
        description="来源权威性分数映射"
    )
    
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        if not candidates:
            return None
        
        def get_weighted_score(e: EvidenceItem) -> float:
            authority = self.authority_scores.get(e.chunk_id, 0.5)
            return authority * e.confidence
        
        return max(candidates, key=get_weighted_score)
    
    def get_strategy_name(self) -> str:
        return ConflictResolutionStrategy.SOURCE_AUTHORITY


class TemporalRecencyResolver(ConflictResolver):
    """时间最近策略
    
    选择最新的证据（基于 extracted_at）。
    """
    
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.extracted_at)
    
    def get_strategy_name(self) -> str:
        return ConflictResolutionStrategy.TEMPORAL_RECENCY


class ManualResolver(ConflictResolver):
    """人工审核策略"""
    
    def resolve(self, candidates: list[EvidenceItem]) -> EvidenceItem | None:
        # 始终返回 None，表示需要人工处理
        return None
    
    def get_strategy_name(self) -> str:
        return ConflictResolutionStrategy.MANUAL


# 策略工厂
RESOLVER_REGISTRY: dict[ConflictResolutionStrategy, type[ConflictResolver]] = {
    ConflictResolutionStrategy.HIGHEST_CONFIDENCE: HighestConfidenceResolver,
    ConflictResolutionStrategy.FIRST: FirstResolver,
    ConflictResolutionStrategy.MAJORITY_VOTE: MajorityVoteResolver,
    ConflictResolutionStrategy.SOURCE_AUTHORITY: SourceAuthorityResolver,
    ConflictResolutionStrategy.TEMPORAL_RECENCY: TemporalRecencyResolver,
    ConflictResolutionStrategy.MANUAL: ManualResolver,
}


def create_resolver(
    strategy: ConflictResolutionStrategy,
    **kwargs
) -> ConflictResolver:
    """创建冲突解决器
    
    Args:
        strategy: 解决策略
        **kwargs: 策略特定的参数
    
    Returns:
        冲突解决器实例
    
    Raises:
        ValueError: 未知的策略
    """
    resolver_class = RESOLVER_REGISTRY.get(strategy)
    if resolver_class is None:
        raise ValueError(f"未知的冲突解决策略: {strategy}")
    
    return resolver_class(**kwargs)


# =============================================================================
# 多源证据字段
# =============================================================================

class EvidenceField(BaseModel):
    """支持多源证据的字段
    
    管理同一字段的多个证据候选，提供冲突检测和解决功能。
    
    Example:
        >>> field = EvidenceField(field_name="培训时长")
        >>> field.add_candidate(evidence1)
        >>> field.add_candidate(evidence2)
        >>> selected = field.resolve_conflict(strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
    """
    model_config = ConfigDict(validate_assignment=True)
    
    field_name: str = Field(..., min_length=1, description="字段名")
    candidates: list[EvidenceItem] = Field(
        default_factory=list,
        description="候选证据列表"
    )
    selected: EvidenceItem | None = Field(
        default=None,
        description="选中的证据"
    )
    resolution_strategy: ConflictResolutionStrategy = Field(
        default=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
        description="使用的解决策略"
    )
    
    def add_candidate(self, evidence: EvidenceItem) -> None:
        """添加候选证据
        
        Args:
            evidence: 证据项
        
        Raises:
            ValueError: 字段名不匹配
        """
        if evidence.field_name != self.field_name:
            raise ValueError(
                f"字段名不匹配: 期望 '{self.field_name}', 得到 '{evidence.field_name}'"
            )
        self.candidates.append(evidence)
    
    def has_conflict(self) -> bool:
        """检查是否存在冲突（多个不同值的候选）"""
        if len(self.candidates) <= 1:
            return False
        values = {e.field_value for e in self.candidates}
        return len(values) > 1
    
    def get_unique_values(self) -> set[str]:
        """获取所有唯一的字段值"""
        return {e.field_value for e in self.candidates}
    
    def get_reliable_candidates(self, threshold: float = 0.8) -> list[EvidenceItem]:
        """获取可靠的候选证据
        
        Args:
            threshold: 置信度阈值
        
        Returns:
            可靠的证据列表
        """
        return [e for e in self.candidates if e.is_reliable(threshold)]
    
    def resolve_conflict(
        self,
        strategy: ConflictResolutionStrategy | None = None,
        **resolver_kwargs
    ) -> EvidenceItem | None:
        """解决多源证据冲突
        
        Args:
            strategy: 解决策略（默认使用 self.resolution_strategy）
            **resolver_kwargs: 策略特定的参数
        
        Returns:
            选中的证据，或 None（需要人工处理）
        """
        if not self.candidates:
            return None
        
        effective_strategy = strategy or self.resolution_strategy
        resolver = create_resolver(effective_strategy, **resolver_kwargs)
        
        self.selected = resolver.resolve(self.candidates)
        self.resolution_strategy = effective_strategy
        
        return self.selected
    
    def get_value(self) -> str | None:
        """获取选中证据的字段值"""
        return self.selected.field_value if self.selected else None
    
    def get_confidence(self) -> float | None:
        """获取选中证据的置信度"""
        return self.selected.confidence if self.selected else None
    
    def select_manually(self, evidence: EvidenceItem) -> None:
        """人工选择证据
        
        Args:
            evidence: 选中的证据（必须在 candidates 中）
        """
        if evidence not in self.candidates:
            raise ValueError("选中的证据不在候选列表中")
        self.selected = evidence
        self.resolution_strategy = ConflictResolutionStrategy.MANUAL


# =============================================================================
# 策略模式 - 评分规则引擎
# =============================================================================

class EvaluationStrategy(ABC, BaseModel):
    """评估策略基类
    
    可序列化的评估策略，支持复杂的条件判断。
    """
    model_config = ConfigDict(validate_assignment=True)
    
    @abstractmethod
    def evaluate(self, input_value: Any) -> bool:
        """评估条件"""
        pass
    
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationStrategy:
        """从字典反序列化"""
        pass


class ThresholdStrategy(EvaluationStrategy):
    """阈值比较策略"""
    threshold: float = Field(..., description="阈值")
    operator: Literal[">=", ">", "<=", "<", "==", "!="] = Field(
        default=">=",
        description="比较操作符"
    )
    
    def evaluate(self, input_value: Any) -> bool:
        try:
            value = float(input_value)
        except (TypeError, ValueError):
            return False
        
        ops: dict[str, Callable[[float, float], bool]] = {
            ">=": lambda x, y: x >= y,
            ">": lambda x, y: x > y,
            "<=": lambda x, y: x <= y,
            "<": lambda x, y: x < y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
        }
        return ops[self.operator](value, self.threshold)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "threshold",
            "threshold": self.threshold,
            "operator": self.operator,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdStrategy:
        return cls(
            threshold=data["threshold"],
            operator=data.get("operator", ">="),
        )


class RangeStrategy(EvaluationStrategy):
    """范围策略"""
    min_value: float | None = Field(default=None, description="最小值")
    max_value: float | None = Field(default=None, description="最大值")
    inclusive: bool = Field(default=True, description="是否包含边界")
    
    def evaluate(self, input_value: Any) -> bool:
        try:
            value = float(input_value)
        except (TypeError, ValueError):
            return False
        
        if self.inclusive:
            min_ok = self.min_value is None or value >= self.min_value
            max_ok = self.max_value is None or value <= self.max_value
        else:
            min_ok = self.min_value is None or value > self.min_value
            max_ok = self.max_value is None or value < self.max_value
        
        return min_ok and max_ok
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "range",
            "min_value": self.min_value,
            "max_value": self.max_value,
            "inclusive": self.inclusive,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RangeStrategy:
        return cls(
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            inclusive=data.get("inclusive", True),
        )


class CompositeStrategy(EvaluationStrategy):
    """复合策略（AND/OR 组合）"""
    operator: Literal["AND", "OR"] = Field(..., description="逻辑操作符")
    strategies: list[EvaluationStrategy] = Field(
        default_factory=list,
        description="子策略列表"
    )
    
    def evaluate(self, input_value: Any) -> bool:
        if not self.strategies:
            return True
        
        results = [s.evaluate(input_value) for s in self.strategies]
        
        if self.operator == "AND":
            return all(results)
        else:  # OR
            return any(results)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "composite",
            "operator": self.operator,
            "strategies": [s.to_dict() for s in self.strategies],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompositeStrategy:
        # 递归反序列化子策略
        strategies = []
        for s_data in data.get("strategies", []):
            strategy = deserialize_strategy(s_data)
            if strategy:
                strategies.append(strategy)
        
        return cls(
            operator=data["operator"],
            strategies=strategies,
        )


# 策略反序列化注册表
STRATEGY_REGISTRY: dict[str, type[EvaluationStrategy]] = {
    "threshold": ThresholdStrategy,
    "range": RangeStrategy,
    "composite": CompositeStrategy,
}


def deserialize_strategy(data: dict[str, Any]) -> EvaluationStrategy | None:
    """从字典反序列化策略
    
    Args:
        data: 策略字典
    
    Returns:
        策略实例，或 None（如果类型未知）
    """
    strategy_type = data.get("type")
    strategy_class = STRATEGY_REGISTRY.get(strategy_type)
    
    if strategy_class is None:
        return None
    
    return strategy_class.from_dict(data)


class ScoringRule(BaseModel):
    """评分规则
    
    可序列化的评分规则，支持复杂的条件评估。
    
    Example:
        >>> rule = ScoringRule(
        ...     strategy=ThresholdStrategy(threshold=4, operator=">="),
        ...     score_range=(4.0, 5.0),
        ...     description="培训方案完整"
        ... )
        >>> rule.evaluate(5)
        4.0
    """
    model_config = ConfigDict(validate_assignment=True)
    
    strategy: EvaluationStrategy = Field(..., description="评估策略")
    score_range: tuple[float, float] = Field(
        ...,
        description="分数范围 (min, max)"
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="规则描述"
    )
    weight: float = Field(
        default=1.0,
        gt=0,
        description="规则权重"
    )
    
    @field_validator("score_range")
    @classmethod
    def validate_score_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """验证分数范围"""
        min_score, max_score = v
        if min_score < 0 or max_score < 0:
            raise ValueError("分数不能为负数")
        if min_score > max_score:
            raise ValueError("最小分数不能大于最大分数")
        return v
    
    def evaluate(self, input_value: Any) -> float | None:
        """评估规则
        
        Args:
            input_value: 输入值
        
        Returns:
            如果条件满足返回 score_range 最小值，否则返回 None
        """
        if self.strategy.evaluate(input_value):
            return self.score_range[0]
        return None
    
    def calculate_score(
        self,
        input_value: Any,
        max_input: float | None = None
    ) -> float:
        """基于输入值计算具体分数
        
        Args:
            input_value: 输入值
            max_input: 输入值的最大值（用于归一化）
        
        Returns:
            score_range 范围内的具体分数
        """
        if not self.strategy.evaluate(input_value):
            return 0.0
        
        min_score, max_score = self.score_range
        
        if max_input and max_input > 0:
            try:
                ratio = min(float(input_value) / max_input, 1.0)
                return min_score + (max_score - min_score) * ratio
            except (TypeError, ValueError):
                pass
        
        return min_score
    
    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "strategy": self.strategy.to_dict(),
            "score_range": self.score_range,
            "description": self.description,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoringRule:
        """从字典反序列化"""
        strategy = deserialize_strategy(data["strategy"])
        if strategy is None:
            raise ValueError(f"无法反序列化策略: {data['strategy']}")
        
        return cls(
            strategy=strategy,
            score_range=tuple(data["score_range"]),
            description=data["description"],
            weight=data.get("weight", 1.0),
        )


# =============================================================================
# 评分维度基类
# =============================================================================

class ScoringDimension(ABC, BaseModel):
    """评分维度基类
    
    所有评分维度的抽象基类，定义通用接口。
    
    Attributes:
        dimension_id: 维度唯一标识
        dimension_name: 维度显示名称
        weight: 维度权重
        sequence: 排序序号
        extracted_evidence: 提取的证据列表
        scoring_rules: 评分规则列表
    """
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    
    dimension_id: str = Field(..., min_length=1, description="维度 ID")
    dimension_name: str = Field(..., min_length=1, description="维度名称")
    weight: float = Field(..., gt=0, description="维度权重")
    sequence: int = Field(..., ge=0, description="排序序号")
    extracted_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="提取的证据列表"
    )
    scoring_rules: list[ScoringRule] = Field(
        default_factory=list,
        description="评分规则列表"
    )
    
    # 审计字段
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="创建时间"
    )
    updated_at: datetime | None = Field(
        default=None,
        description="更新时间"
    )
    
    @abstractmethod
    def calculate_score(self) -> float:
        """计算评分
        
        Returns:
            该维度的得分（0 到 weight 之间）
        """
        pass
    
    @abstractmethod
    def evaluate_completeness(self) -> CompletenessLevel:
        """评估完整性
        
        Returns:
            完整性等级
        """
        pass
    
    def add_evidence(self, evidence: EvidenceItem) -> None:
        """添加提取的证据"""
        self.extracted_evidence.append(evidence)
        self.updated_at = datetime.utcnow()
    
    def get_reliable_evidence(
        self,
        threshold: float = 0.8
    ) -> list[EvidenceItem]:
        """获取可靠的证据列表"""
        return [e for e in self.extracted_evidence if e.is_reliable(threshold)]
    
    def get_score_ratio(self) -> float:
        """获取得分率（得分 / 权重）"""
        score = self.calculate_score()
        if self.weight > 0:
            return score / self.weight
        return 0.0
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """是否已完成评估"""
        return self.evaluate_completeness() == CompletenessLevel.COMPLETE


# =============================================================================
# 具体评分维度实现
# =============================================================================

class TrainingPlan(ScoringDimension):
    """培训方案评分维度 - 5分
    
    评估投标方的培训方案完整性，包括：
    - 培训时长
    - 培训计划/内容
    - 培训人员
    - 授课老师资质
    
    支持从配置文件加载评分标准，实现灵活配置
    """
    
    # 字段定义
    training_duration: EvidenceField | None = Field(
        default=None,
        description="培训时长"
    )
    training_schedule: EvidenceField | None = Field(
        default=None,
        description="培训计划"
    )
    training_personnel: EvidenceField | None = Field(
        default=None,
        description="培训人员"
    )
    instructor_qualifications: EvidenceField | None = Field(
        default=None,
        description="授课老师资质"
    )
    
    # 配置引用
    _config: Any | None = None
    
    def model_post_init(self, __context) -> None:
        """初始化后设置评分规则"""
        # 尝试从配置文件加载
        try:
            from bid_scoring.scoring_config import get_training_plan_config
            self._config = get_training_plan_config()
            self._setup_rules_from_config()
        except Exception:
            # 如果配置加载失败，使用默认规则
            self._setup_default_rules()
    
    def _setup_rules_from_config(self) -> None:
        """从配置文件设置评分规则"""
        if not self._config or not self._config.scoring_rules:
            self._setup_default_rules()
            return
        
        self.scoring_rules = []
        for rule_config in self._config.scoring_rules:
            self.scoring_rules.append(ScoringRule(
                strategy=ThresholdStrategy(threshold=rule_config.min_score, operator=">="),
                score_range=rule_config.score_range,
                description=rule_config.description,
            ))
    
    def _setup_default_rules(self) -> None:
        """设置默认评分规则"""
        if not self.scoring_rules:
            self.scoring_rules = [
                ScoringRule(
                    strategy=ThresholdStrategy(threshold=4, operator=">="),
                    score_range=(4.0, 5.0),
                    description="培训方案完整，能满足招标人日常使用及维修需求",
                ),
                ScoringRule(
                    strategy=ThresholdStrategy(threshold=2, operator=">="),
                    score_range=(2.0, 3.5),
                    description="培训方案较全面，基本满足招标人日常使用",
                ),
                ScoringRule(
                    strategy=ThresholdStrategy(threshold=0, operator=">="),
                    score_range=(0.0, 1.5),
                    description="培训方案简单笼统，无法满足招标人日常使用",
                ),
            ]
    
    def _get_required_fields(self) -> list:
        """获取必填字段列表"""
        if self._config and self._config.required_fields:
            return [
                (f.name, getattr(self, f.name, None))
                for f in self._config.required_fields
            ]
        
        # 默认字段
        return [
            ("training_duration", self.training_duration),
            ("training_schedule", self.training_schedule),
            ("training_personnel", self.training_personnel),
            ("instructor_qualifications", self.instructor_qualifications),
        ]
    
    def evaluate_completeness(self) -> CompletenessLevel:
        """评估完整性"""
        fields = self._get_required_fields()
        
        filled_count = sum(
            1 for _, f in fields
            if f is not None and f.get_value() is not None
        )
        
        # 从配置获取阈值
        if self._config and self._config.scoring_rules:
            for rule in self._config.scoring_rules:
                if filled_count >= rule.min_score:
                    if rule.name == "complete":
                        return CompletenessLevel.COMPLETE
                    elif rule.name == "partial":
                        return CompletenessLevel.PARTIAL
                    elif rule.name == "minimal":
                        return CompletenessLevel.MINIMAL
        
        # 默认阈值
        if filled_count >= 4:
            return CompletenessLevel.COMPLETE
        elif filled_count >= 2:
            return CompletenessLevel.PARTIAL
        elif filled_count >= 1:
            return CompletenessLevel.MINIMAL
        else:
            return CompletenessLevel.EMPTY
    
    def calculate_score(self) -> float:
        """基于完整性计算得分"""
        completeness = self.evaluate_completeness()
        
        # 尝试从配置获取分数
        if self._config and self._config.scoring_rules:
            mapping = {
                CompletenessLevel.COMPLETE: "complete",
                CompletenessLevel.PARTIAL: "partial",
                CompletenessLevel.MINIMAL: "minimal",
                CompletenessLevel.EMPTY: "empty",
            }
            rule_name = mapping.get(completeness)
            for rule in self._config.scoring_rules:
                if rule.name == rule_name:
                    return min(rule.score_range[0], self.weight)
        
        # 默认分数
        scores = {
            CompletenessLevel.COMPLETE: 4.5,
            CompletenessLevel.PARTIAL: 2.5,
            CompletenessLevel.MINIMAL: 0.5,
            CompletenessLevel.EMPTY: 0.0,
        }
        
        return min(scores.get(completeness, 0.0), self.weight)


class AfterSalesService(ScoringDimension):
    """售后服务方案评分维度 - 10分
    
    评估投标方的售后服务能力，包括：
    - 服务团队能力
    - 响应时间
    - 质保期限
    - 配件供应期限
    - 质保期后服务费
    
    支持从配置文件加载评分标准，实现灵活配置
    """
    
    # 字段定义
    service_team_capability: EvidenceField | None = Field(
        default=None,
        description="服务团队能力"
    )
    response_time: EvidenceField | None = Field(
        default=None,
        description="响应时间"
    )
    warranty_period: EvidenceField | None = Field(
        default=None,
        description="质保期限"
    )
    parts_supply_period: EvidenceField | None = Field(
        default=None,
        description="配件供应期限"
    )
    post_warranty_service_fee: EvidenceField | None = Field(
        default=None,
        description="质保期后服务费"
    )
    
    # 配置引用
    _config: Any | None = None
    
    def model_post_init(self, __context) -> None:
        """初始化后设置评分规则"""
        # 尝试从配置文件加载
        try:
            from bid_scoring.scoring_config import get_after_sales_config
            self._config = get_after_sales_config()
            self._setup_rules_from_config()
        except Exception:
            # 如果配置加载失败，使用默认规则
            self._setup_default_rules()
    
    def _setup_rules_from_config(self) -> None:
        """从配置文件设置评分规则"""
        if not self._config or not self._config.scoring_rules:
            self._setup_default_rules()
            return
        
        self.scoring_rules = []
        for rule_config in self._config.scoring_rules:
            self.scoring_rules.append(ScoringRule(
                strategy=ThresholdStrategy(threshold=rule_config.min_score, operator=">="),
                score_range=rule_config.score_range,
                description=rule_config.description,
            ))
    
    def _setup_default_rules(self) -> None:
        """设置默认评分规则"""
        if not self.scoring_rules:
            self.scoring_rules = [
                ScoringRule(
                    strategy=ThresholdStrategy(threshold=8, operator=">="),
                    score_range=(8.0, 10.0),
                    description="售后服务方案优秀，响应及时，质保期长",
                ),
                ScoringRule(
                    strategy=RangeStrategy(min_value=4, max_value=7.9),
                    score_range=(4.0, 7.5),
                    description="售后服务方案标准，满足基本需求",
                ),
                ScoringRule(
                    strategy=ThresholdStrategy(threshold=0, operator=">="),
                    score_range=(0.0, 3.5),
                    description="售后服务方案不足",
                ),
            ]
    
    def _get_criteria(self):
        """获取评分标准"""
        if self._config and self._config.service_level_criteria:
            return self._config.service_level_criteria
        # 返回默认标准
        from bid_scoring.scoring_config import ServiceLevelCriteria
        return ServiceLevelCriteria()
    
    def evaluate_completeness(self) -> CompletenessLevel:
        """评估完整性"""
        fields = [
            self.service_team_capability,
            self.response_time,
            self.warranty_period,
            self.parts_supply_period,
            self.post_warranty_service_fee,
        ]
        
        filled_count = sum(
            1 for f in fields
            if f is not None and f.get_value() is not None
        )
        
        if filled_count >= 4:
            return CompletenessLevel.COMPLETE
        elif filled_count >= 2:
            return CompletenessLevel.PARTIAL
        elif filled_count >= 1:
            return CompletenessLevel.MINIMAL
        else:
            return CompletenessLevel.EMPTY
    
    def evaluate_service_level(self) -> ServiceLevel:
        """评估服务等级
        
        基于配置文件中定义的评分标准评估服务等级
        """
        criteria = self._get_criteria()
        score_points = 0
        
        # 响应时间评估
        if self.response_time and self.response_time.selected:
            value = self.response_time.selected.field_value
            response_hours = self._parse_hours(value)
            
            if response_hours is not None:
                if response_hours <= criteria.response_time.excellent:
                    score_points += 2
                elif response_hours <= criteria.response_time.standard:
                    score_points += 1
            else:
                # 文本匹配回退
                if f"{int(criteria.response_time.excellent)}小时" in value:
                    score_points += 2
                elif f"{int(criteria.response_time.standard)}小时" in value:
                    score_points += 1
        
        # 质保期限评估
        if self.warranty_period and self.warranty_period.selected:
            value = self.warranty_period.selected.field_value
            years = self._parse_years(value)
            
            if years is not None:
                if years >= criteria.warranty_period.excellent:
                    score_points += 2
                elif years >= criteria.warranty_period.standard:
                    score_points += 1
            else:
                # 文本匹配回退
                if f"{int(criteria.warranty_period.excellent)}年" in value:
                    score_points += 2
                elif f"{int(criteria.warranty_period.standard)}年" in value:
                    score_points += 1
        
        # 其他字段（根据配置权重）
        weights = self._config.scoring_weights if self._config else {}
        
        if self.parts_supply_period and self.parts_supply_period.selected:
            score_points += weights.get('parts_supply', 1)
        
        if self.post_warranty_service_fee and self.post_warranty_service_fee.selected:
            score_points += weights.get('post_warranty_fee', 1)
        
        # 到场时间评估（如果配置了）
        if weights.get('on_site_time', 0) > 0:
            # 从响应时间中解析到场时间
            if self.response_time and self.response_time.selected:
                value = self.response_time.selected.field_value
                if "到场" in value or "现场" in value:
                    hours = self._parse_hours(value, "到场")
                    if hours is not None and hours <= criteria.on_site_time.excellent:
                        score_points += weights.get('on_site_time', 0)
        
        # 根据总分确定服务等级
        excellent_threshold = sum([
            2,  # 响应时间满分
            2,  # 质保期限满分
            weights.get('parts_supply', 1),
            weights.get('post_warranty_fee', 1),
            weights.get('on_site_time', 0),
        ]) * 0.8  # 80% 为优秀
        
        standard_threshold = sum([
            1,  # 响应时间及格
            1,  # 质保期限及格
            weights.get('parts_supply', 1) * 0.5,
            weights.get('post_warranty_fee', 1) * 0.5,
        ])
        
        if score_points >= excellent_threshold:
            return ServiceLevel.EXCELLENT
        elif score_points >= standard_threshold:
            return ServiceLevel.STANDARD
        elif score_points >= 1:
            return ServiceLevel.POOR
        else:
            return ServiceLevel.UNKNOWN
    
    def _parse_hours(self, text: str, keyword: str | None = None) -> float | None:
        """从文本中解析小时数"""
        import re
        
        # 优先匹配指定关键词附近的数字
        if keyword:
            pattern = rf'(\d+)\s*小时.*{keyword}|{keyword}.*(\d+)\s*小时'
            matches = re.findall(pattern, text)
            if matches:
                for match in matches[0]:
                    if match:
                        return float(match)
        
        # 通用小时匹配
        match = re.search(r'(\d+)\s*小时', text)
        if match:
            return float(match.group(1))
        
        # 匹配工作日
        match = re.search(r'(\d+)\s*个工作日', text)
        if match:
            return float(match.group(1)) * 8  # 转换为小时（假设8小时工作日）
        
        return None
    
    def _parse_years(self, text: str) -> float | None:
        """从文本中解析年数"""
        import re
        
        # 匹配年
        match = re.search(r'(\d+)\s*年', text)
        if match:
            return float(match.group(1))
        
        # 匹配月数并转换
        match = re.search(r'(\d+)\s*个月', text)
        if match:
            months = int(match.group(1))
            return months / 12
        
        # 匹配"60个月"这样的格式
        match = re.search(r'(\d+)\s*个?月', text)
        if match:
            months = int(match.group(1))
            if months >= 12:
                return months / 12
        
        return None
    
    def calculate_score(self) -> float:
        """基于服务等级计算得分"""
        level = self.evaluate_service_level()
        
        # 尝试从配置获取分数
        if self._config and self._config.scoring_rules:
            for rule in self._config.scoring_rules:
                if level.value == rule.name:
                    return min(rule.score_range[0], self.weight)
        
        # 默认分数
        scores = {
            ServiceLevel.EXCELLENT: 9.0,
            ServiceLevel.STANDARD: 5.5,
            ServiceLevel.POOR: 1.5,
            ServiceLevel.UNKNOWN: 0.0,
        }
        
        return min(scores.get(level, 0.0), self.weight)


# =============================================================================
# 评分结果汇总
# =============================================================================

class DimensionScore(BaseModel):
    """维度评分结果"""
    model_config = ConfigDict(frozen=True)
    
    dimension_id: str
    dimension_name: str
    weight: float
    score: float
    completeness: CompletenessLevel
    evidence_count: int


class ScoringResult(BaseModel):
    """完整评分结果"""
    model_config = ConfigDict(validate_assignment=True)
    
    bid_id: str = Field(..., description="投标 ID")
    document_version_id: str = Field(..., description="文档版本 ID")
    dimension_scores: list[DimensionScore] = Field(
        default_factory=list,
        description="各维度评分"
    )
    total_score: float = Field(default=0.0, ge=0, description="总分")
    max_possible_score: float = Field(default=0.0, ge=0, description="最大可能分数")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="评分时间"
    )
    
    @computed_field
    @property
    def score_percentage(self) -> float:
        """得分百分比"""
        if self.max_possible_score > 0:
            return (self.total_score / self.max_possible_score) * 100
        return 0.0
    
    @computed_field
    @property
    def is_passing(self) -> bool:
        """是否通过（默认60%为通过线）"""
        return self.score_percentage >= 60.0


# =============================================================================
# 导出定义
# =============================================================================

__all__ = [
    # 基础
    "BoundingBox",
    # 枚举
    "ValidationStatus",
    "ConflictResolutionStrategy",
    "ServiceLevel",
    "CompletenessLevel",
    # 证据
    "EvidenceItem",
    "DurationEvidence",
    "ResponseTimeEvidence",
    "WarrantyEvidence",
    "ServiceFeeEvidence",
    "PersonnelEvidence",
    # 冲突解决
    "ConflictResolver",
    "HighestConfidenceResolver",
    "FirstResolver",
    "MajorityVoteResolver",
    "SourceAuthorityResolver",
    "TemporalRecencyResolver",
    "ManualResolver",
    "create_resolver",
    "EvidenceField",
    # 评分规则
    "EvaluationStrategy",
    "ThresholdStrategy",
    "RangeStrategy",
    "CompositeStrategy",
    "deserialize_strategy",
    "ScoringRule",
    # 评分维度
    "ScoringDimension",
    "TrainingPlan",
    "AfterSalesService",
    # 结果
    "DimensionScore",
    "ScoringResult",
]

# 延迟导入配置模块（避免循环导入）
def get_scoring_standards():
    """获取评分标准配置（便捷函数）"""
    from bid_scoring.scoring_config import get_scoring_config
    return get_scoring_config()
