# 回标分析评分维度 Schema 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现评分维度数据模型，支持结构化提取、证据链追踪和多源证据冲突解决

**Architecture:** 使用 dataclass 定义评分维度 Schema，实现 EvidenceItem 证据基类和各评分维度子类，包含置信度处理、冲突解决和评分规则引擎

**Tech Stack:** Python 3.14, dataclasses, psycopg, pydantic (可选), pytest

---

## 前置准备

### 已存在文件
- `bid_scoring/citation_rag_pipeline.py` - 包含 BoundingBox 和 HighlightBox
- `docs/plans/2026-02-05-bid-scoring-schema-design.md` - Schema 设计文档

### 需要创建的文件
- `bid_scoring/scoring_schema.py` - 评分维度数据模型
- `tests/test_scoring_schema.py` - 单元测试

---

## Task 1: 创建证据基类 EvidenceItem

**Files:**
- Create: `bid_scoring/scoring_schema.py`
- Test: `tests/test_scoring_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring_schema.py
import pytest
from bid_scoring.scoring_schema import EvidenceItem, BoundingBox

def test_evidence_item_creation():
    """测试 EvidenceItem 创建"""
    bbox = BoundingBox(x1=100, y1=200, x2=300, y2=400)
    evidence = EvidenceItem(
        field_name="培训时长",
        field_value="2天",
        source_text="培训时长：2天",
        page_idx=67,
        bbox=bbox,
        chunk_id="test-chunk-001",
        confidence=0.95
    )
    assert evidence.field_name == "培训时长"
    assert evidence.confidence == 0.95

def test_evidence_item_is_reliable():
    """测试置信度阈值判断"""
    bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    
    high_confidence = EvidenceItem(
        field_name="test",
        field_value="value",
        source_text="text",
        page_idx=1,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.85
    )
    assert high_confidence.is_reliable() is True
    assert high_confidence.is_reliable(threshold=0.9) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_schema.py::test_evidence_item_creation -v`
Expected: FAIL with "module not found" or "EvidenceItem not defined"

**Step 3: Write minimal implementation**

```python
# bid_scoring/scoring_schema.py
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple, Literal
from datetime import datetime

# 复用 citation_rag_pipeline 中的 BoundingBox
from bid_scoring.citation_rag_pipeline import BoundingBox


@dataclass
class EvidenceItem:
    """证据项 - 关联到原文的具体位置"""
    field_name: str           # 字段名，如 "培训时长"
    field_value: str          # 字段值，如 "2天"
    source_text: str          # 原文片段
    page_idx: int            # 页码
    bbox: BoundingBox        # 边界框
    chunk_id: str            # 关联的chunk ID
    confidence: float = 0.0  # 提取置信度 (0-1)
    
    # 验证状态
    validation_status: Literal["confirmed", "pending", "rejected"] = "pending"
    validation_notes: Optional[str] = None
    
    def is_reliable(self, threshold: float = 0.8) -> bool:
        """判断证据是否可靠（基于置信度阈值）"""
        return self.confidence >= threshold
    
    def confirm(self, notes: Optional[str] = None):
        """确认证据有效"""
        self.validation_status = "confirmed"
        if notes:
            self.validation_notes = notes
    
    def reject(self, reason: str):
        """拒绝证据"""
        self.validation_status = "rejected"
        self.validation_notes = reason
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_schema.py::test_evidence_item_creation -v`
Expected: PASS

Run: `pytest tests/test_scoring_schema.py::test_evidence_item_is_reliable -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_scoring_schema.py bid_scoring/scoring_schema.py
git commit -m "feat(scoring): add EvidenceItem base class with confidence and validation"
```

---

## Task 2: 实现多源证据冲突解决

**Files:**
- Modify: `bid_scoring/scoring_schema.py`
- Test: `tests/test_scoring_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring_schema.py (添加)
from bid_scoring.scoring_schema import EvidenceField

def test_evidence_field_conflict_resolution():
    """测试多源证据冲突解决"""
    bbox1 = BoundingBox(x1=100, y1=200, x2=300, y2=400)
    bbox2 = BoundingBox(x1=500, y1=600, x2=700, y2=800)
    
    # 两个证据候选
    evidence1 = EvidenceItem(
        field_name="培训时长",
        field_value="2天",
        source_text="培训时长：2天",
        page_idx=67,
        bbox=bbox1,
        chunk_id="chunk-1",
        confidence=0.85
    )
    evidence2 = EvidenceItem(
        field_name="培训时长",
        field_value="3天",
        source_text="培训时间为3天",
        page_idx=68,
        bbox=bbox2,
        chunk_id="chunk-2",
        confidence=0.75
    )
    
    field = EvidenceField(
        field_name="培训时长",
        candidates=[evidence1, evidence2]
    )
    
    # 默认策略：选择置信度最高的
    selected = field.resolve_conflict(strategy="highest_confidence")
    assert selected.field_value == "2天"  # confidence 0.85 > 0.75
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_schema.py::test_evidence_field_conflict_resolution -v`
Expected: FAIL with "EvidenceField not defined"

**Step 3: Write minimal implementation**

```python
# bid_scoring/scoring_schema.py (添加)

@dataclass
class EvidenceField:
    """支持多源证据的字段"""
    field_name: str
    candidates: List[EvidenceItem] = field(default_factory=list)
    selected: Optional[EvidenceItem] = None
    
    def add_candidate(self, evidence: EvidenceItem):
        """添加候选证据"""
        if evidence.field_name == self.field_name:
            self.candidates.append(evidence)
        else:
            raise ValueError(f"Field name mismatch: {evidence.field_name} != {self.field_name}")
    
    def resolve_conflict(self, strategy: str = "highest_confidence") -> EvidenceItem:
        """
        解决多源证据冲突
        
        Args:
            strategy: 解决策略
                - "highest_confidence": 选择置信度最高的
                - "first": 选择第一个
                - "manual": 人工审核（返回 None，需要人工设置 selected）
        """
        if not self.candidates:
            raise ValueError(f"No candidates for field {self.field_name}")
        
        if strategy == "highest_confidence":
            self.selected = max(self.candidates, key=lambda e: e.confidence)
        elif strategy == "first":
            self.selected = self.candidates[0]
        elif strategy == "manual":
            self.selected = None  # 需要人工设置
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return self.selected
    
    def get_value(self) -> Optional[str]:
        """获取选中证据的字段值"""
        if self.selected:
            return self.selected.field_value
        return None
    
    def has_conflict(self) -> bool:
        """检查是否存在冲突（多个不同值的候选）"""
        if len(self.candidates) <= 1:
            return False
        values = set(e.field_value for e in self.candidates)
        return len(values) > 1
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_schema.py::test_evidence_field_conflict_resolution -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bid_scoring/scoring_schema.py tests/test_scoring_schema.py
git commit -m "feat(scoring): add EvidenceField for multi-source conflict resolution"
```

---

## Task 3: 实现评分规则引擎

**Files:**
- Modify: `bid_scoring/scoring_schema.py`
- Test: `tests/test_scoring_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring_schema.py (添加)
from bid_scoring.scoring_schema import ScoringRule

def test_scoring_rule_evaluation():
    """测试评分规则评估"""
    rule = ScoringRule(
        condition=lambda x: x >= 4,
        score_range=(4.0, 5.0),
        description="培训方案完整"
    )
    
    # 条件满足
    assert rule.evaluate(4) == 4.0  # 返回 range 最小值
    assert rule.evaluate(5) == 4.0
    
    # 条件不满足应返回 None
    assert rule.evaluate(2) is None

def test_scoring_rule_calculate_score():
    """测试评分规则计算具体分数"""
    rule = ScoringRule(
        condition=lambda x: x >= 4,
        score_range=(4.0, 5.0),
        description="培训方案完整"
    )
    
    # 可以基于满足程度计算具体分数
    score = rule.calculate_score(4, max_input=5)
    assert 4.0 <= score <= 5.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_schema.py::test_scoring_rule_evaluation -v`
Expected: FAIL with "ScoringRule not defined"

**Step 3: Write minimal implementation**

```python
# bid_scoring/scoring_schema.py (添加)

@dataclass
class ScoringRule:
    """评分规则"""
    condition: Callable[[any], bool]  # 条件函数
    score_range: Tuple[float, float]  # 分数范围 (min, max)
    description: str                  # 规则描述
    
    def evaluate(self, input_value: any) -> Optional[float]:
        """
        评估规则
        
        Returns:
            如果条件满足返回 score_range 最小值，否则返回 None
        """
        if self.condition(input_value):
            return self.score_range[0]
        return None
    
    def calculate_score(self, input_value: any, max_input: Optional[float] = None) -> float:
        """
        基于输入值计算具体分数
        
        Args:
            input_value: 输入值
            max_input: 输入值的最大值（用于归一化）
        
        Returns:
            score_range 范围内的具体分数
        """
        if not self.condition(input_value):
            return 0.0
        
        min_score, max_score = self.score_range
        
        if max_input and max_input > 0:
            # 基于输入值在范围内的位置计算分数
            ratio = min(input_value / max_input, 1.0)
            return min_score + (max_score - min_score) * ratio
        
        return min_score
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_schema.py::test_scoring_rule_evaluation -v`
Expected: PASS

Run: `pytest tests/test_scoring_schema.py::test_scoring_rule_calculate_score -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bid_scoring/scoring_schema.py tests/test_scoring_schema.py
git commit -m "feat(scoring): add ScoringRule engine for flexible scoring logic"
```

---

## Task 4: 实现评分维度基类

**Files:**
- Modify: `bid_scoring/scoring_schema.py`
- Test: `tests/test_scoring_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring_schema.py (添加)
from bid_scoring.scoring_schema import ScoringDimension
from dataclasses import dataclass

@dataclass
class TestScoringDimension(ScoringDimension):
    """测试用评分维度"""
    test_field: str = ""
    
    def calculate_score(self) -> float:
        if self.test_field:
            return 5.0
        return 0.0

def test_scoring_dimension_base():
    """测试评分维度基类"""
    bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    evidence = EvidenceItem(
        field_name="test",
        field_value="value",
        source_text="text",
        page_idx=1,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.9
    )
    
    dimension = TestScoringDimension(
        dimension_id="test",
        dimension_name="测试维度",
        weight=10.0,
        sequence=1,
        test_field="有值"
    )
    dimension.add_evidence(evidence)
    
    assert dimension.dimension_id == "test"
    assert dimension.weight == 10.0
    assert len(dimension.extracted_evidence) == 1
    assert dimension.calculate_score() == 5.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_schema.py::test_scoring_dimension_base -v`
Expected: FAIL with "ScoringDimension not defined"

**Step 3: Write minimal implementation**

```python
# bid_scoring/scoring_schema.py (添加)
from abc import ABC, abstractmethod


class ScoringDimension(ABC):
    """评分维度基类"""
    
    def __init__(
        self,
        dimension_id: str,
        dimension_name: str,
        weight: float,
        sequence: int,
        extracted_evidence: Optional[List[EvidenceItem]] = None
    ):
        self.dimension_id = dimension_id
        self.dimension_name = dimension_name
        self.weight = weight
        self.sequence = sequence
        self.extracted_evidence = extracted_evidence or []
    
    def add_evidence(self, evidence: EvidenceItem):
        """添加提取的证据"""
        self.extracted_evidence.append(evidence)
    
    def get_reliable_evidence(self, threshold: float = 0.8) -> List[EvidenceItem]:
        """获取可靠的证据列表"""
        return [e for e in self.extracted_evidence if e.is_reliable(threshold)]
    
    @abstractmethod
    def calculate_score(self) -> float:
        """
        计算评分
        
        Returns:
            该维度的得分（0 到 weight 之间）
        """
        pass
    
    def get_score_ratio(self) -> float:
        """获取得分率（得分 / 权重）"""
        score = self.calculate_score()
        if self.weight > 0:
            return score / self.weight
        return 0.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_schema.py::test_scoring_dimension_base -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bid_scoring/scoring_schema.py tests/test_scoring_schema.py
git commit -m "feat(scoring): add ScoringDimension abstract base class"
```

---

## Task 5: 实现培训方案评分维度

**Files:**
- Modify: `bid_scoring/scoring_schema.py`
- Test: `tests/test_scoring_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring_schema.py (添加)
from bid_scoring.scoring_schema import TrainingPlan

def test_training_plan_scoring():
    """测试培训方案评分"""
    bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    
    # 创建培训方案（完整）
    training = TrainingPlan(
        dimension_id="training",
        dimension_name="培训方案",
        weight=5.0,
        sequence=3
    )
    
    # 添加所有字段
    training.training_duration = EvidenceItem(
        field_name="培训时长",
        field_value="2天",
        source_text="培训时长：2天",
        page_idx=67,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.9
    )
    training.training_schedule = EvidenceItem(
        field_name="培训计划",
        field_value="现场授课+实操",
        source_text="培训计划：现场授课+实操",
        page_idx=67,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.85
    )
    training.training_personnel = EvidenceItem(
        field_name="培训人员",
        field_value="工程师",
        source_text="培训人员：工程师",
        page_idx=67,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.8
    )
    training.instructor_qualifications = [
        EvidenceItem(
            field_name="授课老师资质",
            field_value="高级工程师",
            source_text="授课老师资质：高级工程师",
            page_idx=67,
            bbox=bbox,
            chunk_id="chunk-1",
            confidence=0.9
        )
    ]
    
    # 评估完整性
    completeness = training.evaluate_completeness()
    assert completeness == "complete"
    
    # 计算得分
    score = training.calculate_score()
    assert 4.0 <= score <= 5.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_schema.py::test_training_plan_scoring -v`
Expected: FAIL with "TrainingPlan not defined"

**Step 3: Write minimal implementation**

```python
# bid_scoring/scoring_schema.py (添加)

class TrainingPlan(ScoringDimension):
    """培训方案 - 5分"""
    
    def __init__(self, dimension_id: str, dimension_name: str, weight: float, sequence: int):
        super().__init__(dimension_id, dimension_name, weight, sequence)
        self.training_duration: Optional[EvidenceItem] = None
        self.training_schedule: Optional[EvidenceItem] = None
        self.training_personnel: Optional[EvidenceItem] = None
        self.instructor_qualifications: List[EvidenceItem] = []
        
        # 定义评分规则
        self.scoring_rules = [
            ScoringRule(
                condition=lambda x: x >= 4,
                score_range=(4.0, 5.0),
                description="培训方案完整，能满足招标人日常使用及维修需求"
            ),
            ScoringRule(
                condition=lambda x: x >= 2,
                score_range=(2.0, 3.0),
                description="培训方案较全面，基本满足招标人日常使用"
            ),
            ScoringRule(
                condition=lambda x: x >= 0,
                score_range=(0.0, 1.0),
                description="培训方案简单笼统，无法满足招标人日常使用"
            )
        ]
    
    def evaluate_completeness(self) -> str:
        """评估完整性: complete/partial/minimal"""
        filled_fields = sum([
            self.training_duration is not None,
            self.training_schedule is not None,
            self.training_personnel is not None,
            len(self.instructor_qualifications) > 0
        ])
        
        if filled_fields >= 4:
            return "complete"
        elif filled_fields >= 2:
            return "partial"
        else:
            return "minimal"
    
    def calculate_score(self) -> float:
        """基于完整性计算得分"""
        completeness = self.evaluate_completeness()
        
        if completeness == "complete":
            return 4.5
        elif completeness == "partial":
            return 2.5
        else:
            return 0.5
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_schema.py::test_training_plan_scoring -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bid_scoring/scoring_schema.py tests/test_scoring_schema.py
git commit -m "feat(scoring): add TrainingPlan dimension with completeness evaluation"
```

---

## Task 6: 实现售后服务评分维度

**Files:**
- Modify: `bid_scoring/scoring_schema.py`
- Test: `tests/test_scoring_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring_schema.py (添加)
from bid_scoring.scoring_schema import AfterSalesService, ResponseTimeEvidence, WarrantyEvidence

def test_after_sales_service_scoring():
    """测试售后服务评分"""
    bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    
    service = AfterSalesService(
        dimension_id="after_sales",
        dimension_name="售后服务方案",
        weight=10.0,
        sequence=4
    )
    
    # 设置响应时间（优秀）
    service.response_time = ResponseTimeEvidence(
        field_name="响应时间",
        field_value="2小时内响应，24小时内到达现场",
        source_text="2小时内响应，24小时内到达现场",
        page_idx=40,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.9,
        raw_value="2小时内响应，24小时内到达现场",
        hours=2.0,
        on_site_hours=24.0
    )
    
    # 设置质保期限（优秀）
    service.warranty_period = WarrantyEvidence(
        field_name="质保期限",
        field_value="整机保修5年",
        source_text="整机保修5年",
        page_idx=40,
        bbox=bbox,
        chunk_id="chunk-1",
        confidence=0.95,
        raw_value="整机保修5年",
        years=5.0
    )
    
    # 评估服务等级
    level = service.evaluate_service_level()
    assert level == "excellent"  # 响应时间 <= 2h，质保 >= 5年
    
    # 计算得分
    score = service.calculate_score()
    assert 8.0 <= score <= 10.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_schema.py::test_after_sales_service_scoring -v`
Expected: FAIL with "AfterSalesService not defined"

**Step 3: Write minimal implementation**

```python
# bid_scoring/scoring_schema.py (添加)

class DurationEvidence(EvidenceItem):
    """时长证据（带数值解析）"""
    raw_value: str = ""
    days: Optional[float] = None
    hours: Optional[float] = None
    
    def __init__(self, raw_value: str = "", days: Optional[float] = None, 
                 hours: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.raw_value = raw_value
        self.days = days
        self.hours = hours


class ResponseTimeEvidence(EvidenceItem):
    """响应时间证据"""
    raw_value: str = ""
    hours: Optional[float] = None
    on_site_hours: Optional[float] = None
    
    def __init__(self, raw_value: str = "", hours: Optional[float] = None,
                 on_site_hours: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.raw_value = raw_value
        self.hours = hours
        self.on_site_hours = on_site_hours


class WarrantyEvidence(EvidenceItem):
    """质保期限证据"""
    raw_value: str = ""
    years: Optional[float] = None
    months: Optional[int] = None
    
    def __init__(self, raw_value: str = "", years: Optional[float] = None,
                 months: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.raw_value = raw_value
        self.years = years
        self.months = months


class ServiceFeeEvidence(EvidenceItem):
    """服务费用证据"""
    raw_value: str = ""
    is_free: bool = False
    fee_percentage: Optional[float] = None
    
    def __init__(self, raw_value: str = "", is_free: bool = False,
                 fee_percentage: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.raw_value = raw_value
        self.is_free = is_free
        self.fee_percentage = fee_percentage


class AfterSalesService(ScoringDimension):
    """售后服务方案 - 10分"""
    
    def __init__(self, dimension_id: str, dimension_name: str, weight: float, sequence: int):
        super().__init__(dimension_id, dimension_name, weight, sequence)
        self.service_team_capability: Optional[EvidenceItem] = None
        self.response_time: Optional[ResponseTimeEvidence] = None
        self.warranty_period: Optional[WarrantyEvidence] = None
        self.parts_supply_period: Optional[EvidenceItem] = None
        self.post_warranty_service_fee: Optional[ServiceFeeEvidence] = None
        self.labor_cost_standard: Optional[EvidenceItem] = None
    
    def evaluate_service_level(self) -> str:
        """评估服务等级: excellent/standard/poor"""
        score_points = 0
        
        # 响应时间评估
        if self.response_time and self.response_time.hours:
            if self.response_time.hours <= 2:
                score_points += 2
            elif self.response_time.hours <= 8:
                score_points += 1
        
        # 质保期限评估
        if self.warranty_period and self.warranty_period.years:
            if self.warranty_period.years >= 5:
                score_points += 2
            elif self.warranty_period.years >= 3:
                score_points += 1
        
        # 其他评估（简化）
        if self.parts_supply_period:
            score_points += 1
        if self.post_warranty_service_fee:
            score_points += 1
        
        if score_points >= 4:
            return "excellent"
        elif score_points >= 2:
            return "standard"
        else:
            return "poor"
    
    def calculate_score(self) -> float:
        """基于服务等级计算得分"""
        level = self.evaluate_service_level()
        
        if level == "excellent":
            return 9.0
        elif level == "standard":
            return 5.5
        else:
            return 1.5
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring_schema.py::test_after_sales_service_scoring -v`
Expected: PASS

**Step 5: Commit**

```bash
git add bid_scoring/scoring_schema.py tests/test_scoring_schema.py
git commit -m "feat(scoring): add AfterSalesService dimension with service level evaluation"
```

---

## Task 7: 数据库表结构迁移

**Files:**
- Create: `migrations/011_scoring_schema.sql`
- Test: `tests/test_db_schema.py` (可选)

**Step 1: Create migration file**

```sql
-- migrations/011_scoring_schema.sql
-- 评分维度提取结果表

-- 评分维度提取结果表
CREATE TABLE bid_scoring_results (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bid_id UUID,
    document_version_id UUID REFERENCES document_versions(version_id),
    dimension_id TEXT NOT NULL,
    dimension_name TEXT NOT NULL,
    weight FLOAT NOT NULL,
    extracted_score FLOAT,
    final_score FLOAT,
    evaluation_level TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 证据项表
CREATE TABLE scoring_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES bid_scoring_results(result_id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    field_value TEXT,
    source_text TEXT,
    page_idx INTEGER,
    bbox JSONB,
    chunk_id UUID REFERENCES hierarchical_nodes(node_id),
    confidence FLOAT DEFAULT 0.0,
    validation_status TEXT DEFAULT 'pending',
    validation_notes TEXT,
    raw_value TEXT,
    parsed_value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 索引
CREATE INDEX idx_scoring_results_version ON bid_scoring_results(document_version_id);
CREATE INDEX idx_scoring_results_dimension ON bid_scoring_results(dimension_id);
CREATE INDEX idx_evidence_result ON scoring_evidence(result_id);
CREATE INDEX idx_evidence_page ON scoring_evidence(page_idx);
CREATE INDEX idx_evidence_field ON scoring_evidence(field_name);
CREATE INDEX idx_evidence_status ON scoring_evidence(validation_status);

-- 审计日志表
CREATE TABLE scoring_audit_log (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES bid_scoring_results(result_id),
    action TEXT NOT NULL,
    performed_by TEXT,
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    old_value JSONB,
    new_value JSONB
);

-- 注释
COMMENT ON TABLE bid_scoring_results IS '评分维度提取结果';
COMMENT ON TABLE scoring_evidence IS '评分证据项，关联到原文位置';
COMMENT ON TABLE scoring_audit_log IS '评分审计日志';
```

**Step 2: Apply migration**

```bash
cd /Users/wangxq/Documents/投标分析_kimi
psql $DATABASE_URL -f migrations/011_scoring_schema.sql
```

Expected: 所有表和索引创建成功

**Step 3: Verify tables created**

```bash
psql $DATABASE_URL -c "\dt scoring_*"
```

Expected: 显示 scoring_evidence, scoring_audit_log, bid_scoring_results

**Step 4: Commit**

```bash
git add migrations/011_scoring_schema.sql
git commit -m "feat(db): add scoring schema tables with evidence and audit support"
```

---

## 完成总结

**实施计划包含 7 个任务：**
1. ✅ 创建证据基类 EvidenceItem（带置信度和验证状态）
2. ✅ 实现多源证据冲突解决 EvidenceField
3. ✅ 实现评分规则引擎 ScoringRule
4. ✅ 实现评分维度基类 ScoringDimension
5. ✅ 实现培训方案评分维度 TrainingPlan
6. ✅ 实现售后服务评分维度 AfterSalesService
7. ✅ 数据库表结构迁移

**每个任务都包含：**
- 具体文件路径
- 完整的测试代码
- 完整的实现代码
- 验证命令
- 提交命令

**估计总时间：** 2-3 小时
