"""
回标分析评分维度 Schema 测试

测试策略:
    - 单元测试：每个类和方法的独立测试
    - 集成测试：端到端工作流测试
    - 边界测试：异常输入和边界条件
    - 序列化测试：确保数据可持久化
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from bid_scoring.scoring_schema import (
    # 基础
    BoundingBox,
    # 枚举
    ValidationStatus,
    ConflictResolutionStrategy,
    ServiceLevel,
    CompletenessLevel,
    # 证据
    EvidenceItem,
    DurationEvidence,
    ResponseTimeEvidence,
    WarrantyEvidence,
    ServiceFeeEvidence,
    PersonnelEvidence,
    # 冲突解决
    HighestConfidenceResolver,
    FirstResolver,
    MajorityVoteResolver,
    SourceAuthorityResolver,
    TemporalRecencyResolver,
    ManualResolver,
    create_resolver,
    EvidenceField,
    # 评分规则
    ThresholdStrategy,
    RangeStrategy,
    CompositeStrategy,
    deserialize_strategy,
    ScoringRule,
    # 评分维度
    TrainingPlan,
    AfterSalesService,
    # 结果
    DimensionScore,
    ScoringResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bbox() -> BoundingBox:
    """示例边界框"""
    return BoundingBox(x1=100, y1=200, x2=300, y2=400)


@pytest.fixture
def sample_evidence(sample_bbox) -> EvidenceItem:
    """示例证据"""
    return EvidenceItem(
        field_name="培训时长",
        field_value="2天",
        source_text="培训时长：2天",
        page_idx=67,
        bbox=sample_bbox,
        chunk_id="test-chunk-001",
        confidence=0.95,
    )


@pytest.fixture
def sample_duration_evidence(sample_bbox) -> DurationEvidence:
    """示例时长证据"""
    return DurationEvidence(
        field_name="培训时长",
        field_value="2天",
        source_text="培训时长：2天",
        page_idx=67,
        bbox=sample_bbox,
        chunk_id="chunk-1",
        confidence=0.9,
        raw_value="培训时长：2天",
        days=2.0,
    )


# =============================================================================
# BoundingBox 测试
# =============================================================================

class TestBoundingBox:
    """BoundingBox 测试类"""
    
    def test_creation(self):
        """测试创建边界框"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.x1 == 0
        assert bbox.y1 == 0
        assert bbox.x2 == 100
        assert bbox.y2 == 100
    
    def test_validation_coordinates(self):
        """测试坐标验证"""
        with pytest.raises(ValidationError) as exc_info:
            BoundingBox(x1=100, y1=200, x2=50, y2=400)
        assert "x2 必须大于等于 x1" in str(exc_info.value)
    
    def test_computed_fields(self):
        """测试计算字段"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
    
    def test_immutability(self):
        """测试不可变性"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        with pytest.raises(ValidationError):
            bbox.x1 = 50
    
    def test_from_list(self):
        """测试从列表创建"""
        bbox = BoundingBox.from_list([10, 20, 30, 40])
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 30
        assert bbox.y2 == 40
    
    def test_from_list_short(self):
        """测试从短列表创建 - 缺失坐标使用起始坐标值以确保验证通过"""
        bbox = BoundingBox.from_list([10, 20])
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 10  # 默认等于 x1
        assert bbox.y2 == 20  # 默认等于 y1


# =============================================================================
# EvidenceItem 测试
# =============================================================================

class TestEvidenceItem:
    """EvidenceItem 测试类"""
    
    def test_creation(self, sample_bbox):
        """测试创建证据"""
        evidence = EvidenceItem(
            field_name="培训时长",
            field_value="2天",
            source_text="培训时长：2天",
            page_idx=67,
            bbox=sample_bbox,
            chunk_id="test-chunk-001",
            confidence=0.95,
        )
        assert evidence.field_name == "培训时长"
        assert evidence.confidence == 0.95
        assert evidence.validation_status == ValidationStatus.PENDING
    
    def test_validation_empty_field_name(self, sample_bbox):
        """测试空字段名验证"""
        with pytest.raises(ValidationError):
            EvidenceItem(
                field_name="",
                field_value="2天",
                source_text="text",
                page_idx=1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
            )
    
    def test_validation_empty_field_value(self, sample_bbox):
        """测试空字段值验证"""
        with pytest.raises(ValidationError):
            EvidenceItem(
                field_name="name",
                field_value="   ",
                source_text="text",
                page_idx=1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
            )
    
    def test_validation_confidence_range(self, sample_bbox):
        """测试置信度范围验证"""
        with pytest.raises(ValidationError):
            EvidenceItem(
                field_name="name",
                field_value="value",
                source_text="text",
                page_idx=1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
                confidence=1.5,  # 超过 1.0
            )
    
    def test_validation_page_idx_negative(self, sample_bbox):
        """测试负页码验证"""
        with pytest.raises(ValidationError):
            EvidenceItem(
                field_name="name",
                field_value="value",
                source_text="text",
                page_idx=-1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
            )
    
    def test_is_reliable(self, sample_bbox):
        """测试可靠性判断"""
        high_confidence = EvidenceItem(
            field_name="test",
            field_value="value",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            confidence=0.85,
        )
        assert high_confidence.is_reliable() is True
        assert high_confidence.is_reliable(threshold=0.9) is False
    
    def test_confirm(self, sample_bbox):
        """测试确认证据"""
        evidence = EvidenceItem(
            field_name="test",
            field_value="value",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            confidence=0.9,
        )
        evidence.confirm("已验证")
        assert evidence.validation_status == ValidationStatus.CONFIRMED
        assert evidence.validation_notes == "已验证"
        assert evidence.validated_at is not None
        assert evidence.is_validated is True
    
    def test_reject(self, sample_bbox):
        """测试拒绝证据"""
        evidence = EvidenceItem(
            field_name="test",
            field_value="value",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
        )
        evidence.reject("与原文不符")
        assert evidence.validation_status == ValidationStatus.REJECTED
        assert evidence.validation_notes == "与原文不符"
    
    def test_reset_validation(self, sample_bbox):
        """测试重置验证状态"""
        evidence = EvidenceItem(
            field_name="test",
            field_value="value",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
        )
        evidence.confirm("已验证")
        evidence.reset_validation()
        assert evidence.validation_status == ValidationStatus.PENDING
        assert evidence.validation_notes is None
    
    def test_whitespace_stripping(self, sample_bbox):
        """测试空白字符去除"""
        evidence = EvidenceItem(
            field_name="  name  ",
            field_value="  value  ",
            source_text="  text  ",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="  chunk-1  ",
        )
        assert evidence.field_name == "name"
        assert evidence.field_value == "value"


# =============================================================================
# 结构化证据子类测试
# =============================================================================

class TestDurationEvidence:
    """DurationEvidence 测试类"""
    
    def test_creation(self, sample_bbox):
        """测试创建时长证据"""
        evidence = DurationEvidence(
            field_name="培训时长",
            field_value="2天",
            source_text="培训时长：2天",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            confidence=0.9,
            raw_value="培训时长：2天",
            days=2.0,
        )
        assert evidence.days == 2.0
        assert evidence.total_hours == 48.0
    
    def test_validation_no_duration(self, sample_bbox):
        """测试无时长值验证"""
        with pytest.raises(ValidationError) as exc_info:
            DurationEvidence(
                field_name="培训时长",
                field_value="2天",
                source_text="培训时长：2天",
                page_idx=1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
                # 没有 days 或 hours
            )
        assert "必须提供 days 或 hours 至少一个时长值" in str(exc_info.value)
    
    def test_hours_only(self, sample_bbox):
        """测试仅小时"""
        evidence = DurationEvidence(
            field_name="培训时长",
            field_value="16小时",
            source_text="培训16小时",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            hours=16.0,
        )
        assert evidence.total_hours == 16.0
    
    def test_negative_days_validation(self, sample_bbox):
        """测试负数天数验证"""
        with pytest.raises(ValidationError):
            DurationEvidence(
                field_name="培训时长",
                field_value="-1天",
                source_text="text",
                page_idx=1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
                days=-1.0,
            )


class TestResponseTimeEvidence:
    """ResponseTimeEvidence 测试类"""
    
    def test_emergency_response(self, sample_bbox):
        """测试紧急响应判断"""
        evidence = ResponseTimeEvidence(
            field_name="响应时间",
            field_value="2小时内",
            source_text="2小时内响应",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            response_hours=2.0,
        )
        assert evidence.is_emergency_response is True
    
    def test_non_emergency_response(self, sample_bbox):
        """测试非紧急响应"""
        evidence = ResponseTimeEvidence(
            field_name="响应时间",
            field_value="24小时内",
            source_text="24小时内响应",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            response_hours=24.0,
        )
        assert evidence.is_emergency_response is False


class TestWarrantyEvidence:
    """WarrantyEvidence 测试类"""
    
    def test_total_months(self, sample_bbox):
        """测试总月数计算"""
        evidence = WarrantyEvidence(
            field_name="质保期限",
            field_value="2年",
            source_text="质保2年",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            years=2.0,
        )
        assert evidence.total_months == 24.0
    
    def test_total_months_with_months(self, sample_bbox):
        """测试年+月计算"""
        evidence = WarrantyEvidence(
            field_name="质保期限",
            field_value="2年6个月",
            source_text="质保2年6个月",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            years=2.0,
            months=6,
        )
        assert evidence.total_months == 30.0


# =============================================================================
# 冲突解决策略测试
# =============================================================================

class TestConflictResolvers:
    """冲突解决策略测试类"""
    
    @pytest.fixture
    def candidates(self, sample_bbox) -> list[EvidenceItem]:
        """创建测试候选列表"""
        return [
            EvidenceItem(
                field_name="培训时长",
                field_value="2天",
                source_text="培训时长：2天",
                page_idx=1,
                bbox=sample_bbox,
                chunk_id="chunk-1",
                confidence=0.85,
            ),
            EvidenceItem(
                field_name="培训时长",
                field_value="3天",
                source_text="培训时间为3天",
                page_idx=2,
                bbox=sample_bbox,
                chunk_id="chunk-2",
                confidence=0.75,
            ),
            EvidenceItem(
                field_name="培训时长",
                field_value="2天",
                source_text="培训2天",
                page_idx=3,
                bbox=sample_bbox,
                chunk_id="chunk-3",
                confidence=0.90,
            ),
        ]
    
    def test_highest_confidence_resolver(self, candidates):
        """测试最高置信度策略"""
        resolver = HighestConfidenceResolver()
        selected = resolver.resolve(candidates)
        assert selected is not None
        assert selected.confidence == 0.90
        assert selected.field_value == "2天"
    
    def test_first_resolver(self, candidates):
        """测试第一个策略"""
        resolver = FirstResolver()
        selected = resolver.resolve(candidates)
        assert selected is not None
        assert selected.chunk_id == "chunk-1"
    
    def test_majority_vote_resolver(self, candidates):
        """测试多数投票策略"""
        resolver = MajorityVoteResolver()
        selected = resolver.resolve(candidates)
        assert selected is not None
        # "2天" 出现两次，选择其中置信度最高的 (0.90)
        assert selected.field_value == "2天"
        assert selected.confidence == 0.90
    
    def test_majority_vote_single_candidate(self, sample_bbox):
        """测试多数投票单候选"""
        single = [EvidenceItem(
            field_name="test",
            field_value="value",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            confidence=0.5,
        )]
        resolver = MajorityVoteResolver()
        selected = resolver.resolve(single)
        assert selected is not None
        assert selected.chunk_id == "chunk-1"
    
    def test_source_authority_resolver(self, candidates):
        """测试来源权威性策略"""
        authority_scores = {
            "chunk-1": 0.9,
            "chunk-2": 0.5,
            "chunk-3": 0.8,
        }
        resolver = SourceAuthorityResolver(authority_scores=authority_scores)
        selected = resolver.resolve(candidates)
        assert selected is not None
        # chunk-1: 0.9 * 0.85 = 0.765
        # chunk-2: 0.5 * 0.75 = 0.375
        # chunk-3: 0.8 * 0.90 = 0.720
        assert selected.chunk_id == "chunk-1"
    
    def test_temporal_recency_resolver(self, sample_bbox):
        """测试时间最近策略"""
        old = EvidenceItem(
            field_name="test",
            field_value="old",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-old",
            extracted_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        new = EvidenceItem(
            field_name="test",
            field_value="new",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-new",
            extracted_at=datetime.now(timezone.utc),
        )
        resolver = TemporalRecencyResolver()
        selected = resolver.resolve([old, new])
        assert selected is not None
        assert selected.chunk_id == "chunk-new"
    
    def test_manual_resolver(self, candidates):
        """测试人工策略"""
        resolver = ManualResolver()
        selected = resolver.resolve(candidates)
        assert selected is None
    
    def test_resolver_empty_list(self):
        """测试空列表处理"""
        resolver = HighestConfidenceResolver()
        selected = resolver.resolve([])
        assert selected is None


class TestEvidenceField:
    """EvidenceField 测试类"""
    
    @pytest.fixture
    def field(self) -> EvidenceField:
        """创建测试字段"""
        return EvidenceField(field_name="培训时长")
    
    @pytest.fixture
    def candidate1(self, sample_bbox) -> EvidenceItem:
        return EvidenceItem(
            field_name="培训时长",
            field_value="2天",
            source_text="培训时长：2天",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
            confidence=0.85,
        )
    
    @pytest.fixture
    def candidate2(self, sample_bbox) -> EvidenceItem:
        return EvidenceItem(
            field_name="培训时长",
            field_value="3天",
            source_text="培训时间为3天",
            page_idx=2,
            bbox=sample_bbox,
            chunk_id="chunk-2",
            confidence=0.75,
        )
    
    def test_add_candidate(self, field, candidate1):
        """测试添加候选"""
        field.add_candidate(candidate1)
        assert len(field.candidates) == 1
    
    def test_add_candidate_mismatch(self, field, sample_bbox):
        """测试添加不匹配的候选"""
        wrong = EvidenceItem(
            field_name="错误字段",
            field_value="value",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="chunk-1",
        )
        with pytest.raises(ValueError) as exc_info:
            field.add_candidate(wrong)
        assert "字段名不匹配" in str(exc_info.value)
    
    def test_has_conflict(self, field, candidate1, candidate2):
        """测试冲突检测"""
        field.add_candidate(candidate1)
        field.add_candidate(candidate2)
        assert field.has_conflict() is True
    
    def test_no_conflict(self, field, candidate1, sample_bbox):
        """测试无冲突"""
        same_value = EvidenceItem(
            field_name="培训时长",
            field_value="2天",
            source_text="培训2天",
            page_idx=2,
            bbox=sample_bbox,
            chunk_id="chunk-2",
            confidence=0.9,
        )
        field.add_candidate(candidate1)
        field.add_candidate(same_value)
        assert field.has_conflict() is False
    
    def test_resolve_conflict(self, field, candidate1, candidate2):
        """测试冲突解决"""
        field.add_candidate(candidate1)
        field.add_candidate(candidate2)
        selected = field.resolve_conflict(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        assert selected is not None
        assert selected.confidence == 0.85
        assert field.selected == selected
    
    def test_get_reliable_candidates(self, field, candidate1, sample_bbox):
        """测试获取可靠候选"""
        unreliable = EvidenceItem(
            field_name="培训时长",
            field_value="5天",
            source_text="text",
            page_idx=2,
            bbox=sample_bbox,
            chunk_id="chunk-2",
            confidence=0.5,  # 低于默认阈值 0.8
        )
        field.add_candidate(candidate1)
        field.add_candidate(unreliable)
        reliable = field.get_reliable_candidates()
        assert len(reliable) == 1
        assert reliable[0].confidence == 0.85
    
    def test_select_manually(self, field, candidate1, candidate2):
        """测试人工选择"""
        field.add_candidate(candidate1)
        field.add_candidate(candidate2)
        field.select_manually(candidate2)
        assert field.selected == candidate2
        assert field.resolution_strategy == ConflictResolutionStrategy.MANUAL
    
    def test_select_manually_not_in_candidates(self, field, sample_bbox):
        """测试选择不在候选列表中的证据"""
        external = EvidenceItem(
            field_name="培训时长",
            field_value="10天",
            source_text="text",
            page_idx=1,
            bbox=sample_bbox,
            chunk_id="external",
        )
        with pytest.raises(ValueError) as exc_info:
            field.select_manually(external)
        assert "不在候选列表中" in str(exc_info.value)
    
    def test_get_value(self, field, candidate1):
        """测试获取值"""
        field.add_candidate(candidate1)
        field.resolve_conflict()
        assert field.get_value() == "2天"
    
    def test_get_value_no_selection(self, field):
        """测试未选择时获取值"""
        assert field.get_value() is None


class TestCreateResolver:
    """create_resolver 函数测试"""
    
    def test_create_highest_confidence_resolver(self):
        """测试创建最高置信度解决器"""
        resolver = create_resolver(ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
        assert isinstance(resolver, HighestConfidenceResolver)
    
    def test_create_unknown_strategy(self):
        """测试未知策略"""
        with pytest.raises(ValueError) as exc_info:
            create_resolver("unknown_strategy")  # type: ignore
        assert "未知的冲突解决策略" in str(exc_info.value)
    
    def test_create_with_kwargs(self):
        """测试带参数创建"""
        resolver = create_resolver(
            ConflictResolutionStrategy.SOURCE_AUTHORITY,
            authority_scores={"chunk-1": 0.9}
        )
        assert isinstance(resolver, SourceAuthorityResolver)
        assert resolver.authority_scores == {"chunk-1": 0.9}


# =============================================================================
# 评分规则引擎测试
# =============================================================================

class TestThresholdStrategy:
    """ThresholdStrategy 测试类"""
    
    def test_greater_than_equal(self):
        """测试大于等于"""
        strategy = ThresholdStrategy(threshold=5.0, operator=">=")
        assert strategy.evaluate(5.0) is True
        assert strategy.evaluate(6.0) is True
        assert strategy.evaluate(4.9) is False
    
    def test_less_than(self):
        """测试小于"""
        strategy = ThresholdStrategy(threshold=5.0, operator="<")
        assert strategy.evaluate(4.0) is True
        assert strategy.evaluate(5.0) is False
    
    def test_equal(self):
        """测试等于"""
        strategy = ThresholdStrategy(threshold=5.0, operator="==")
        assert strategy.evaluate(5.0) is True
        assert strategy.evaluate(5) is True
        assert strategy.evaluate(4.0) is False
    
    def test_invalid_input(self):
        """测试无效输入"""
        strategy = ThresholdStrategy(threshold=5.0)
        assert strategy.evaluate("invalid") is False
        assert strategy.evaluate(None) is False
    
    def test_serialization(self):
        """测试序列化"""
        strategy = ThresholdStrategy(threshold=5.0, operator=">=")
        data = strategy.to_dict()
        assert data["type"] == "threshold"
        assert data["threshold"] == 5.0
        assert data["operator"] == ">="
    
    def test_deserialization(self):
        """测试反序列化"""
        data = {"type": "threshold", "threshold": 5.0, "operator": ">="}
        strategy = ThresholdStrategy.from_dict(data)
        assert strategy.threshold == 5.0
        assert strategy.operator == ">="


class TestRangeStrategy:
    """RangeStrategy 测试类"""
    
    def test_inclusive_range(self):
        """测试包含边界的范围"""
        strategy = RangeStrategy(min_value=0, max_value=10, inclusive=True)
        assert strategy.evaluate(0) is True
        assert strategy.evaluate(10) is True
        assert strategy.evaluate(5) is True
        assert strategy.evaluate(-1) is False
        assert strategy.evaluate(11) is False
    
    def test_exclusive_range(self):
        """测试不包含边界的范围"""
        strategy = RangeStrategy(min_value=0, max_value=10, inclusive=False)
        assert strategy.evaluate(0) is False
        assert strategy.evaluate(10) is False
        assert strategy.evaluate(5) is True
    
    def test_open_range(self):
        """测试开放范围"""
        strategy = RangeStrategy(min_value=5, max_value=None)
        assert strategy.evaluate(5) is True
        assert strategy.evaluate(100) is True
        assert strategy.evaluate(4) is False


class TestCompositeStrategy:
    """CompositeStrategy 测试类"""
    
    def test_and_composite(self):
        """测试 AND 组合"""
        strategy = CompositeStrategy(
            operator="AND",
            strategies=[
                ThresholdStrategy(threshold=5, operator=">="),
                ThresholdStrategy(threshold=10, operator="<"),
            ]
        )
        assert strategy.evaluate(7) is True
        assert strategy.evaluate(5) is True
        assert strategy.evaluate(10) is False
        assert strategy.evaluate(4) is False
    
    def test_or_composite(self):
        """测试 OR 组合"""
        strategy = CompositeStrategy(
            operator="OR",
            strategies=[
                ThresholdStrategy(threshold=10, operator=">="),
                ThresholdStrategy(threshold=2, operator="<"),
            ]
        )
        assert strategy.evaluate(10) is True
        assert strategy.evaluate(1) is True
        assert strategy.evaluate(5) is False
    
    def test_empty_strategies(self):
        """测试空策略列表"""
        strategy = CompositeStrategy(operator="AND", strategies=[])
        assert strategy.evaluate(5) is True


class TestScoringRule:
    """ScoringRule 测试类"""
    
    def test_creation(self):
        """测试创建规则"""
        rule = ScoringRule(
            strategy=ThresholdStrategy(threshold=4, operator=">="),
            score_range=(4.0, 5.0),
            description="培训方案完整",
        )
        assert rule.description == "培训方案完整"
        assert rule.weight == 1.0
    
    def test_validation_invalid_score_range(self):
        """测试无效分数范围"""
        with pytest.raises(ValidationError):
            ScoringRule(
                strategy=ThresholdStrategy(threshold=4),
                score_range=(5.0, 4.0),  # 最小大于最大
                description="无效规则",
            )
    
    def test_validation_negative_score(self):
        """测试负分数"""
        with pytest.raises(ValidationError):
            ScoringRule(
                strategy=ThresholdStrategy(threshold=4),
                score_range=(-1.0, 5.0),
                description="无效规则",
            )
    
    def test_evaluate_true(self):
        """测试条件满足"""
        rule = ScoringRule(
            strategy=ThresholdStrategy(threshold=4, operator=">="),
            score_range=(4.0, 5.0),
            description="培训方案完整",
        )
        result = rule.evaluate(5)
        assert result == 4.0
    
    def test_evaluate_false(self):
        """测试条件不满足"""
        rule = ScoringRule(
            strategy=ThresholdStrategy(threshold=4, operator=">="),
            score_range=(4.0, 5.0),
            description="培训方案完整",
        )
        result = rule.evaluate(2)
        assert result is None
    
    def test_calculate_score_with_normalization(self):
        """测试归一化计分"""
        rule = ScoringRule(
            strategy=ThresholdStrategy(threshold=4, operator=">="),
            score_range=(4.0, 5.0),
            description="培训方案完整",
        )
        # 输入 8，最大值 10，应在 range 中占 80%
        score = rule.calculate_score(8, max_input=10)
        assert 4.7 < score < 4.9  # 4.0 + 0.8 = 4.8
    
    def test_calculate_score_false(self):
        """测试条件不满足时计分"""
        rule = ScoringRule(
            strategy=ThresholdStrategy(threshold=4, operator=">="),
            score_range=(4.0, 5.0),
            description="培训方案完整",
        )
        score = rule.calculate_score(2)
        assert score == 0.0
    
    def test_serialization_round_trip(self):
        """测试序列化往返"""
        original = ScoringRule(
            strategy=ThresholdStrategy(threshold=4, operator=">="),
            score_range=(4.0, 5.0),
            description="培训方案完整",
            weight=2.0,
        )
        data = original.to_dict()
        restored = ScoringRule.from_dict(data)
        assert restored.description == original.description
        assert restored.score_range == original.score_range
        assert restored.weight == original.weight


class TestDeserializeStrategy:
    """deserialize_strategy 函数测试"""
    
    def test_deserialize_threshold(self):
        """测试反序列化阈值策略"""
        data = {"type": "threshold", "threshold": 5.0, "operator": ">="}
        strategy = deserialize_strategy(data)
        assert isinstance(strategy, ThresholdStrategy)
    
    def test_deserialize_range(self):
        """测试反序列化范围策略"""
        data = {"type": "range", "min_value": 0, "max_value": 10}
        strategy = deserialize_strategy(data)
        assert isinstance(strategy, RangeStrategy)
    
    def test_deserialize_unknown_type(self):
        """测试未知类型"""
        data = {"type": "unknown"}
        strategy = deserialize_strategy(data)
        assert strategy is None


# =============================================================================
# 评分维度测试
# =============================================================================

class TestTrainingPlan:
    """TrainingPlan 测试类"""
    
    @pytest.fixture
    def plan(self) -> TrainingPlan:
        """创建测试培训方案"""
        return TrainingPlan(
            dimension_id="training",
            dimension_name="培训方案",
            weight=5.0,
            sequence=1,
        )
    
    def test_creation(self, plan):
        """测试创建"""
        assert plan.dimension_id == "training"
        assert plan.weight == 5.0
        assert len(plan.scoring_rules) == 3  # 默认规则
    
    def test_evaluate_completeness_empty(self, plan):
        """测试空完整性评估"""
        assert plan.evaluate_completeness() == CompletenessLevel.EMPTY
    
    def test_evaluate_completeness_partial(self, plan):
        """测试部分完整性"""
        plan.training_duration = EvidenceField(field_name="培训时长")
        plan.training_duration.add_candidate(EvidenceItem(
            field_name="培训时长",
            field_value="2天",
            source_text="培训2天",
            page_idx=1,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            chunk_id="chunk-1",
        ))
        plan.training_duration.resolve_conflict()
        
        plan.training_schedule = EvidenceField(field_name="培训计划")
        plan.training_schedule.add_candidate(EvidenceItem(
            field_name="培训计划",
            field_value="现场授课",
            source_text="现场授课",
            page_idx=1,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            chunk_id="chunk-1",
        ))
        plan.training_schedule.resolve_conflict()
        
        assert plan.evaluate_completeness() == CompletenessLevel.PARTIAL
    
    def test_calculate_score(self, plan):
        """测试计算分数"""
        # 设置完整数据
        for field_name in ["培训时长", "培训计划", "培训人员", "授课老师资质"]:
            field = EvidenceField(field_name=field_name)
            field.add_candidate(EvidenceItem(
                field_name=field_name,
                field_value="有",
                source_text="有",
                page_idx=1,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                chunk_id="chunk-1",
            ))
            field.resolve_conflict()
            
            if field_name == "培训时长":
                plan.training_duration = field
            elif field_name == "培训计划":
                plan.training_schedule = field
            elif field_name == "培训人员":
                plan.training_personnel = field
            elif field_name == "授课老师资质":
                plan.instructor_qualifications = field
        
        score = plan.calculate_score()
        assert score == 4.5
    
    def test_get_score_ratio(self, plan):
        """测试获取得分率"""
        # 准备4个字段
        fields_data = [
            ("training_duration", "培训时长"),
            ("training_schedule", "培训计划"),
            ("training_personnel", "培训人员"),
            ("instructor_qualifications", "授课老师资质"),
        ]
        
        for attr_name, field_name in fields_data:
            field = EvidenceField(field_name=field_name)
            field.add_candidate(EvidenceItem(
                field_name=field_name,
                field_value="有",
                source_text="有",
                page_idx=1,
                bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
                chunk_id="chunk-1",
            ))
            field.resolve_conflict()
            setattr(plan, attr_name, field)
        
        ratio = plan.get_score_ratio()
        assert ratio == 0.9  # 4.5 / 5.0


class TestAfterSalesService:
    """AfterSalesService 测试类"""
    
    @pytest.fixture
    def service(self) -> AfterSalesService:
        """创建测试售后服务"""
        return AfterSalesService(
            dimension_id="after_sales",
            dimension_name="售后服务方案",
            weight=10.0,
            sequence=2,
        )
    
    def test_evaluate_service_level_excellent(self, service):
        """测试优秀服务等级"""
        # 设置 2小时响应
        service.response_time = EvidenceField(field_name="响应时间")
        service.response_time.add_candidate(EvidenceItem(
            field_name="响应时间",
            field_value="2小时内响应，24小时内到达现场",
            source_text="2小时内响应",
            page_idx=1,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            chunk_id="chunk-1",
        ))
        service.response_time.resolve_conflict()
        
        # 设置 5年质保
        service.warranty_period = EvidenceField(field_name="质保期限")
        service.warranty_period.add_candidate(EvidenceItem(
            field_name="质保期限",
            field_value="整机保修5年",
            source_text="整机保修5年",
            page_idx=1,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            chunk_id="chunk-1",
        ))
        service.warranty_period.resolve_conflict()
        
        # 设置配件供应
        service.parts_supply_period = EvidenceField(field_name="配件供应")
        service.parts_supply_period.add_candidate(EvidenceItem(
            field_name="配件供应",
            field_value="长期供应",
            source_text="长期供应",
            page_idx=1,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            chunk_id="chunk-1",
        ))
        service.parts_supply_period.resolve_conflict()
        
        assert service.evaluate_service_level() == ServiceLevel.EXCELLENT
        assert service.calculate_score() == 9.0
    
    def test_evaluate_service_level_unknown(self, service):
        """测试未知服务等级"""
        assert service.evaluate_service_level() == ServiceLevel.UNKNOWN


# =============================================================================
# 评分结果测试
# =============================================================================

class TestScoringResult:
    """ScoringResult 测试类"""
    
    def test_creation(self):
        """测试创建结果"""
        result = ScoringResult(
            bid_id="bid-001",
            document_version_id="version-001",
            total_score=15.0,
            max_possible_score=20.0,
        )
        assert result.bid_id == "bid-001"
        assert result.total_score == 15.0
    
    def test_score_percentage(self):
        """测试得分百分比"""
        result = ScoringResult(
            bid_id="bid-001",
            document_version_id="version-001",
            total_score=15.0,
            max_possible_score=20.0,
        )
        assert result.score_percentage == 75.0
    
    def test_is_passing(self):
        """测试通过判断"""
        passing = ScoringResult(
            bid_id="bid-001",
            document_version_id="version-001",
            total_score=12.0,
            max_possible_score=20.0,  # 60%
        )
        assert passing.is_passing is True
        
        failing = ScoringResult(
            bid_id="bid-001",
            document_version_id="version-001",
            total_score=10.0,
            max_possible_score=20.0,  # 50%
        )
        assert failing.is_passing is False
    
    def test_dimension_scores(self):
        """测试维度分数"""
        scores = [
            DimensionScore(
                dimension_id="training",
                dimension_name="培训方案",
                weight=5.0,
                score=4.5,
                completeness=CompletenessLevel.COMPLETE,
                evidence_count=4,
            ),
        ]
        result = ScoringResult(
            bid_id="bid-001",
            document_version_id="version-001",
            dimension_scores=scores,
        )
        assert len(result.dimension_scores) == 1


# =============================================================================
# 集成测试
# =============================================================================

class TestEndToEndWorkflow:
    """端到端工作流测试"""
    
    def test_complete_scoring_workflow(self):
        """测试完整评分工作流"""
        # 1. 创建证据
        bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
        evidence1 = EvidenceItem(
            field_name="培训时长",
            field_value="2天",
            source_text="培训时长：2天",
            page_idx=67,
            bbox=bbox,
            chunk_id="chunk-1",
            confidence=0.85,
        )
        evidence2 = EvidenceItem(
            field_name="培训时长",
            field_value="3天",
            source_text="培训时间为3天",
            page_idx=68,
            bbox=bbox,
            chunk_id="chunk-2",
            confidence=0.75,
        )
        
        # 2. 创建多源字段
        field = EvidenceField(field_name="培训时长")
        field.add_candidate(evidence1)
        field.add_candidate(evidence2)
        
        # 3. 解决冲突
        selected = field.resolve_conflict(
            strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        )
        assert selected is not None
        assert selected.field_value == "2天"
        
        # 4. 创建评分维度
        plan = TrainingPlan(
            dimension_id="training",
            dimension_name="培训方案",
            weight=5.0,
            sequence=1,
        )
        plan.training_duration = field
        
        # 5. 计算分数
        completeness = plan.evaluate_completeness()
        score = plan.calculate_score()
        
        assert completeness == CompletenessLevel.MINIMAL
        assert score == 0.5
    
    def test_serialization_round_trip(self):
        """测试序列化往返"""
        # 创建完整的数据结构
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        evidence = EvidenceItem(
            field_name="测试",
            field_value="值",
            source_text="原文",
            page_idx=1,
            bbox=bbox,
            chunk_id="chunk-1",
            confidence=0.9,
        )
        
        # Pydantic v2 序列化
        json_str = evidence.model_dump_json()
        restored = EvidenceItem.model_validate_json(json_str)
        
        assert restored.field_name == evidence.field_name
        assert restored.confidence == evidence.confidence
        assert restored.bbox.x1 == evidence.bbox.x1
    
    def test_rule_serialization(self):
        """测试规则序列化"""
        rule = ScoringRule(
            strategy=CompositeStrategy(
                operator="AND",
                strategies=[
                    ThresholdStrategy(threshold=4, operator=">="),
                    RangeStrategy(min_value=0, max_value=10),
                ]
            ),
            score_range=(4.0, 5.0),
            description="复合规则",
        )
        
        data = rule.to_dict()
        restored = ScoringRule.from_dict(data)
        
        assert isinstance(restored.strategy, CompositeStrategy)
        assert len(restored.strategy.strategies) == 2
