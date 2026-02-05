# 回标分析评分维度 Schema 使用指南

> **目标读者**: 后端开发者、算法工程师  
> **前置知识**: Python, Pydantic, PostgreSQL  
> **更新时间**: 2026-02-05

---

## 📚 目录

1. [快速开始](#快速开始)
2. [核心概念](#核心概念)
3. [基础用法](#基础用法)
4. [完整示例：投标评分流程](#完整示例投标评分流程)
5. [与 RAG Pipeline 集成](#与-rag-pipeline-集成)
6. [数据库存储](#数据库存储)
7. [最佳实践](#最佳实践)
8. [故障排查](#故障排查)

---

## 快速开始

### 安装依赖

```bash
pip install pydantic
```

### 导入模块

```python
from bid_scoring.scoring_schema import (
    # 基础
    BoundingBox, EvidenceItem, ValidationStatus,
    # 结构化证据
    DurationEvidence, ResponseTimeEvidence, WarrantyEvidence,
    ServiceFeeEvidence, PersonnelEvidence,
    # 冲突解决
    EvidenceField, ConflictResolutionStrategy,
    # 评分规则
    ScoringRule, ThresholdStrategy, RangeStrategy,
    # 评分维度
    TrainingPlan, AfterSalesService, ScoringDimension,
    # 结果
    ScoringResult, DimensionScore, CompletenessLevel,
)
```

### 3 分钟上手

```python
# 1. 创建一个证据
evidence = EvidenceItem(
    field_name="培训时长",
    field_value="2天",
    source_text="培训时长：2天",
    page_idx=67,
    bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
    chunk_id="chunk-001",
    confidence=0.95,
)

# 2. 验证证据是否可靠
if evidence.is_reliable(threshold=0.9):
    print(f"✅ 可信证据: {evidence.field_value}")

# 3. 创建评分维度并计算分数
plan = TrainingPlan(
    dimension_id="training",
    dimension_name="培训方案",
    weight=5.0,
    sequence=1,
)
plan.training_duration = EvidenceField(
    field_name="培训时长",
    candidates=[evidence]
)
plan.training_duration.resolve_conflict()

score = plan.calculate_score()
print(f"📊 培训方案得分: {score}/{plan.weight}")
```

---

## 核心概念

### 概念关系图

```
投标项目 (Bid)
    └── 文档版本 (DocumentVersion)
            └── 评分维度 (ScoringDimension)
                    ├── 证据字段 (EvidenceField)
                    │       ├── 候选证据 1 (EvidenceItem)
                    │       ├── 候选证据 2 (EvidenceItem)
                    │       └── 选中证据 (EvidenceItem)
                    ├── 评分规则 (ScoringRule)
                    └── 计算得分 (Score)
```

### 关键类说明

| 类名 | 用途 | 类比 |
|------|------|------|
| `EvidenceItem` | 单个证据，关联到 PDF 具体位置 | 一条引用 |
| `EvidenceField` | 管理多源证据，解决冲突 | 一个字段的所有候选值 |
| `ScoringDimension` | 评分维度（如培训方案） | 评分表中的一行 |
| `ScoringRule` | 评分规则 | 评分标准 |
| `ScoringResult` | 完整评分结果 | 评分报告 |

---

## 基础用法

### 1. 创建和管理证据

#### 基础证据

```python
from bid_scoring.scoring_schema import EvidenceItem, BoundingBox

# 创建证据
evidence = EvidenceItem(
    field_name="质保期限",           # 字段名
    field_value="5年",              # 字段值
    source_text="整机保修5年",       # 原文
    page_idx=40,                     # 页码
    bbox=BoundingBox(x1=100, y1=200, x2=300, y2=250),  # 位置
    chunk_id="chunk-uuid-001",       # 关联 chunk
    confidence=0.92,                 # 置信度 (0-1)
)

# 验证状态管理
evidence.confirm("已人工核对")       # 确认
evidence.reject("与原文不符")        # 拒绝
evidence.reset_validation()         # 重置

# 检查可靠性
is_reliable = evidence.is_reliable(threshold=0.8)
```

#### 结构化证据（自动解析）

```python
from bid_scoring.scoring_schema import DurationEvidence, ResponseTimeEvidence

# 时长证据（自动计算总小时数）
duration = DurationEvidence(
    field_name="培训时长",
    field_value="2天",
    source_text="培训时长为2天",
    page_idx=10,
    bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
    chunk_id="chunk-1",
    confidence=0.9,
    raw_value="培训时长为2天",
    days=2.0,                        # 解析出 2 天
)
print(duration.total_hours)  # 48.0

# 响应时间证据
response_time = ResponseTimeEvidence(
    field_name="响应时间",
    field_value="2小时内响应",
    source_text="2小时内响应，24小时内到达现场",
    page_idx=20,
    bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
    chunk_id="chunk-2",
    response_hours=2.0,
    on_site_hours=24.0,
)
print(response_time.is_emergency_response)  # True
```

### 2. 处理多源证据冲突

```python
from bid_scoring.scoring_schema import EvidenceField, ConflictResolutionStrategy

# 创建多源字段
field = EvidenceField(field_name="培训时长")

# 添加多个候选证据
field.add_candidate(evidence_2days)    # 置信度 0.85
field.add_candidate(evidence_3days)    # 置信度 0.75
field.add_candidate(evidence_2days_v2) # 置信度 0.90

# 检查是否有冲突
if field.has_conflict():
    print(f"⚠️ 发现冲突: {field.get_unique_values()}")

# 自动解决冲突 - 最高置信度
selected = field.resolve_conflict(
    strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
)

# 或使用其他策略
field.resolve_conflict(strategy=ConflictResolutionStrategy.MAJORITY_VOTE)
field.resolve_conflict(strategy=ConflictResolutionStrategy.FIRST)

# 人工选择
field.select_manually(evidence_2days)

# 获取结果
value = field.get_value()           # 选中的值
confidence = field.get_confidence() # 选中的置信度
```

### 3. 定义评分规则

```python
from bid_scoring.scoring_schema import ScoringRule, ThresholdStrategy, RangeStrategy

# 简单阈值规则
rule = ScoringRule(
    strategy=ThresholdStrategy(threshold=4, operator=">="),
    score_range=(4.0, 5.0),
    description="培训方案完整",
    weight=1.0,
)

# 评估
result = rule.evaluate(5)  # 4.0 (满足条件，返回最低分)
result = rule.evaluate(3)  # None (不满足条件)

# 带归一化的计分
score = rule.calculate_score(8, max_input=10)  # 按比例计算

# 范围规则
range_rule = ScoringRule(
    strategy=RangeStrategy(min_value=2, max_value=4, inclusive=True),
    score_range=(2.0, 4.0),
    description="部分完整",
)

# 复合规则（AND/OR）
from bid_scoring.scoring_schema import CompositeStrategy

composite_rule = ScoringRule(
    strategy=CompositeStrategy(
        operator="AND",
        strategies=[
            ThresholdStrategy(threshold=4, operator=">="),
            RangeStrategy(max_value=10),
        ]
    ),
    score_range=(4.0, 5.0),
    description="复合条件",
)
```

### 4. 使用评分维度

```python
from bid_scoring.scoring_schema import TrainingPlan, AfterSalesService

# ===== 培训方案维度 =====
training = TrainingPlan(
    dimension_id="training",
    dimension_name="培训方案",
    weight=5.0,
    sequence=1,
)

# 设置字段证据
training.training_duration = EvidenceField(...)
training.training_schedule = EvidenceField(...)
training.training_personnel = EvidenceField(...)
training.instructor_qualifications = EvidenceField(...)

# 评估完整性
completeness = training.evaluate_completeness()
# 返回: CompletenessLevel.COMPLETE / PARTIAL / MINIMAL / EMPTY

# 计算得分
score = training.calculate_score()  # 0.5, 2.5, 或 4.5
ratio = training.get_score_ratio()   # 得分 / 权重

# ===== 售后服务维度 =====
service = AfterSalesService(
    dimension_id="after_sales",
    dimension_name="售后服务方案",
    weight=10.0,
    sequence=2,
)

# 设置字段
service.response_time = EvidenceField(...)
service.warranty_period = EvidenceField(...)

# 评估服务等级
level = service.evaluate_service_level()
# 返回: ServiceLevel.EXCELLENT / STANDARD / POOR / UNKNOWN

score = service.calculate_score()  # 9.0, 5.5, 1.5, 或 0.0
```

---

## 完整示例：投标评分流程

```python
"""
完整投标评分流程示例

场景: 对一份投标文件进行评分，包括培训方案和售后服务两个维度
"""

from bid_scoring.scoring_schema import *


def score_bid_document(version_id: str) -> ScoringResult:
    """评分流程主函数"""
    
    # =========================================================
    # 步骤 1: 创建证据（通常从 RAG Pipeline 提取）
    # =========================================================
    
    # 培训时长证据
    training_duration_ev = DurationEvidence(
        field_name="培训时长",
        field_value="2天",
        source_text="培训时长：2天（16小时）",
        page_idx=67,
        bbox=BoundingBox(x1=100, y1=200, x2=300, y2=220),
        chunk_id="chunk-training-1",
        confidence=0.95,
        raw_value="培训时长：2天",
        days=2.0,
        hours=16.0,
    )
    
    # 培训计划证据
    training_schedule_ev = EvidenceItem(
        field_name="培训计划",
        field_value="现场授课+实操演练",
        source_text="培训计划：现场授课+实操演练",
        page_idx=67,
        bbox=BoundingBox(x1=100, y1=230, x2=400, y2=250),
        chunk_id="chunk-training-1",
        confidence=0.92,
    )
    
    # 响应时间证据（多源）
    response_ev1 = ResponseTimeEvidence(
        field_name="响应时间",
        field_value="2小时内响应",
        source_text="2小时内响应，24小时内到达现场",
        page_idx=40,
        bbox=BoundingBox(x1=50, y1=100, x2=300, y2=120),
        chunk_id="chunk-service-1",
        confidence=0.88,
        response_hours=2.0,
        on_site_hours=24.0,
    )
    
    response_ev2 = ResponseTimeEvidence(
        field_name="响应时间",
        field_value="1小时内响应",
        source_text="1小时内响应",
        page_idx=45,
        bbox=BoundingBox(x1=50, y1=200, x2=200, y2=220),
        chunk_id="chunk-service-2",
        confidence=0.75,  # 较低置信度
        response_hours=1.0,
    )
    
    # 质保期限证据
    warranty_ev = WarrantyEvidence(
        field_name="质保期限",
        field_value="5年",
        source_text="整机保修5年",
        page_idx=40,
        bbox=BoundingBox(x1=50, y1=150, x2=200, y2=170),
        chunk_id="chunk-service-1",
        confidence=0.96,
        raw_value="整机保修5年",
        years=5.0,
    )
    
    # =========================================================
    # 步骤 2: 处理多源冲突
    # =========================================================
    
    # 响应时间有多源证据，需要解决冲突
    response_field = EvidenceField(field_name="响应时间")
    response_field.add_candidate(response_ev1)
    response_field.add_candidate(response_ev2)
    
    if response_field.has_conflict():
        print(f"⚠️ 响应时间存在冲突: {response_field.get_unique_values()}")
    
    # 使用最高置信度策略
    selected = response_field.resolve_conflict(
        strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    )
    print(f"✅ 选中响应时间: {selected.field_value} (置信度: {selected.confidence})")
    
    # =========================================================
    # 步骤 3: 创建评分维度
    # =========================================================
    
    # 培训方案维度
    training = TrainingPlan(
        dimension_id="training",
        dimension_name="培训方案",
        weight=5.0,
        sequence=1,
    )
    
    # 构建证据字段
    training.training_duration = EvidenceField(field_name="培训时长")
    training.training_duration.add_candidate(training_duration_ev)
    training.training_duration.resolve_conflict()
    
    training.training_schedule = EvidenceField(field_name="培训计划")
    training.training_schedule.add_candidate(training_schedule_ev)
    training.training_schedule.resolve_conflict()
    
    # 添加更多字段...
    training.training_personnel = EvidenceField(field_name="培训人员")
    training.training_personnel.add_candidate(EvidenceItem(
        field_name="培训人员",
        field_value="高级工程师",
        source_text="由高级工程师授课",
        page_idx=67,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        chunk_id="chunk-training-2",
        confidence=0.85,
    ))
    training.training_personnel.resolve_conflict()
    
    training.instructor_qualifications = EvidenceField(field_name="授课老师资质")
    training.instructor_qualifications.add_candidate(PersonnelEvidence(
        field_name="授课老师资质",
        field_value="高级工程师，10年经验",
        source_text="授课老师：高级工程师，10年以上行业经验",
        page_idx=67,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        chunk_id="chunk-training-3",
        confidence=0.90,
        qualification_level="高级工程师",
        years_experience=10,
    ))
    training.instructor_qualifications.resolve_conflict()
    
    # 售后服务维度
    service = AfterSalesService(
        dimension_id="after_sales",
        dimension_name="售后服务方案",
        weight=10.0,
        sequence=2,
    )
    
    service.response_time = response_field
    
    service.warranty_period = EvidenceField(field_name="质保期限")
    service.warranty_period.add_candidate(warranty_ev)
    service.warranty_period.resolve_conflict()
    
    # 添加其他字段...
    service.parts_supply_period = EvidenceField(field_name="配件供应期限")
    service.parts_supply_period.add_candidate(EvidenceItem(
        field_name="配件供应期限",
        field_value="10年",
        source_text="配件供应期限：10年",
        page_idx=40,
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
        chunk_id="chunk-service-3",
        confidence=0.85,
    ))
    service.parts_supply_period.resolve_conflict()
    
    # =========================================================
    # 步骤 4: 计算评分
    # =========================================================
    
    dimensions = [training, service]
    
    dimension_scores = []
    total_score = 0.0
    max_possible = 0.0
    
    for dim in dimensions:
        score = dim.calculate_score()
        completeness = dim.evaluate_completeness()
        
        dim_score = DimensionScore(
            dimension_id=dim.dimension_id,
            dimension_name=dim.dimension_name,
            weight=dim.weight,
            score=score,
            completeness=completeness,
            evidence_count=len(dim.extracted_evidence),
        )
        
        dimension_scores.append(dim_score)
        total_score += score
        max_possible += dim.weight
        
        print(f"\n📊 {dim.dimension_name}")
        print(f"   完整性: {completeness.value}")
        print(f"   得分: {score}/{dim.weight}")
        print(f"   得分率: {dim.get_score_ratio():.1%}")
    
    # =========================================================
    # 步骤 5: 生成结果
    # =========================================================
    
    result = ScoringResult(
        bid_id="bid-2024-001",
        document_version_id=version_id,
        dimension_scores=dimension_scores,
        total_score=total_score,
        max_possible_score=max_possible,
    )
    
    print(f"\n{'='*50}")
    print(f"📋 评分结果")
    print(f"{'='*50}")
    print(f"总分: {total_score:.1f}/{max_possible:.1f}")
    print(f"得分率: {result.score_percentage:.1f}%")
    print(f"是否通过: {'✅ 是' if result.is_passing else '❌ 否'}")
    
    return result


# 运行示例
if __name__ == "__main__":
    result = score_bid_document(version_id="version-001")
```

---

## 与 RAG Pipeline 集成

```python
"""
将评分 Schema 与现有的 CitationRAGPipeline 集成

目标: 从 RAG 提取的答案自动创建证据
"""

from bid_scoring.citation_rag_pipeline import CitationRAGPipeline, HighlightBox
from bid_scoring.scoring_schema import (
    EvidenceItem, BoundingBox, EvidenceField, 
    TrainingPlan, AfterSalesService
)


def extract_evidence_from_rag(
    version_id: str,
    query: str,
    field_name: str,
) -> EvidenceItem | None:
    """从 RAG Pipeline 提取证据"""
    
    # 执行 RAG 查询
    pipeline = CitationRAGPipeline(version_id=version_id)
    result = pipeline.query(query)
    
    if not result.highlight_boxes:
        return None
    
    # 获取第一个高亮框
    highlight: HighlightBox = result.highlight_boxes[0]
    
    # 转换为 EvidenceItem
    evidence = EvidenceItem(
        field_name=field_name,
        field_value=extract_value_from_answer(result.answer),  # 需要实现提取逻辑
        source_text=highlight.text_preview,
        page_idx=highlight.page_idx,
        bbox=BoundingBox(
            x1=highlight.bbox.x1,
            y1=highlight.bbox.y1,
            x2=highlight.bbox.x2,
            y2=highlight.bbox.y2,
        ),
        chunk_id=highlight.chunk_id,
        confidence=calculate_confidence(result),  # 需要实现置信度计算
    )
    
    return evidence


def auto_score_document(version_id: str) -> dict:
    """自动评分流程"""
    
    # 定义查询模板
    queries = {
        "training": {
            "培训时长": "培训时长是多少？",
            "培训计划": "培训内容包括哪些？",
            "培训人员": "培训对象是谁？",
            "授课老师资质": "授课老师的资质如何？",
        },
        "after_sales": {
            "响应时间": "售后响应时间是多久？",
            "质保期限": "质保期是多长时间？",
            "配件供应": "配件供应期限是多久？",
        }
    }
    
    # 创建评分维度
    training = TrainingPlan(
        dimension_id="training",
        dimension_name="培训方案",
        weight=5.0,
        sequence=1,
    )
    
    # 自动提取培训相关证据
    for field_name, query in queries["training"].items():
        evidence = extract_evidence_from_rag(version_id, query, field_name)
        if evidence:
            field = EvidenceField(field_name=field_name)
            field.add_candidate(evidence)
            field.resolve_conflict()
            
            # 设置到维度
            if field_name == "培训时长":
                training.training_duration = field
            elif field_name == "培训计划":
                training.training_schedule = field
            # ...
    
    # 计算分数
    score = training.calculate_score()
    
    return {
        "dimension": "training",
        "score": score,
        "max_score": training.weight,
        "completeness": training.evaluate_completeness().value,
    }


def extract_value_from_answer(answer: str) -> str:
    """从答案中提取值（简化示例）"""
    # 实际应用中可能需要 LLM 或正则提取
    return answer.strip()[:50]


def calculate_confidence(result) -> float:
    """计算置信度（简化示例）"""
    # 可以基于相似度、引用数量等计算
    return 0.85
```

---

## 数据库存储

### 1. 应用迁移

```bash
# 应用数据库迁移
psql $DATABASE_URL -f migrations/011_scoring_schema.sql
```

### 2. 保存评分结果

```python
import psycopg
import json
from bid_scoring.scoring_schema import ScoringResult, EvidenceItem
from bid_scoring.config import load_settings


def save_scoring_result(result: ScoringResult) -> str:
    """保存评分结果到数据库"""
    
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # 插入评分结果
            cur.execute("""
                INSERT INTO bid_scoring_results (
                    bid_id, document_version_id, dimension_id,
                    dimension_name, weight, extracted_score,
                    final_score, completeness_level, evaluation_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING result_id
            """, (
                result.bid_id,
                result.document_version_id,
                "composite",  # 或具体维度
                "综合评分",
                sum(ds.weight for ds in result.dimension_scores),
                result.total_score,
                result.total_score,  # 最终分数可能经过调整
                "complete" if result.is_passing else "partial",
                json.dumps(result.model_dump()),
            ))
            
            result_id = cur.fetchone()[0]
            conn.commit()
            
    return str(result_id)


def save_evidence(result_id: str, evidence: EvidenceItem) -> str:
    """保存证据到数据库"""
    
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scoring_evidence (
                    result_id, field_name, field_value, source_text,
                    page_idx, bbox, chunk_id, confidence,
                    validation_status, evidence_type, parsed_value
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING evidence_id
            """, (
                result_id,
                evidence.field_name,
                evidence.field_value,
                evidence.source_text,
                evidence.page_idx,
                json.dumps(evidence.bbox.to_dict()),
                evidence.chunk_id,
                evidence.confidence,
                evidence.validation_status.value,
                "base",  # 或具体类型
                None,    # 结构化解析值
            ))
            
            evidence_id = cur.fetchone()[0]
            conn.commit()
            
    return str(evidence_id)
```

### 3. 查询评分结果

```python
def get_bid_score_summary(bid_id: str) -> list[dict]:
    """获取投标评分汇总"""
    
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # 使用视图查询
            cur.execute("""
                SELECT dimension_name, weight, score,
                       completeness_level, evidence_count
                FROM v_scoring_results_summary
                WHERE bid_id = %s
                ORDER BY dimension_id
            """, (bid_id,))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "dimension_name": row[0],
                    "weight": row[1],
                    "score": row[2],
                    "completeness": row[3],
                    "evidence_count": row[4],
                })
                
    return results


def get_evidence_by_page(version_id: str, page_idx: int) -> list[dict]:
    """获取指定页面的所有证据"""
    
    settings = load_settings()
    
    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT field_name, field_value, confidence, bbox
                FROM scoring_evidence se
                JOIN bid_scoring_results sr ON se.result_id = sr.result_id
                WHERE sr.document_version_id = %s
                  AND se.page_idx = %s
                ORDER BY se.confidence DESC
            """, (version_id, page_idx))
            
            return [
                {
                    "field_name": row[0],
                    "field_value": row[1],
                    "confidence": row[2],
                    "bbox": row[3],
                }
                for row in cur.fetchall()
            ]
```

---

## 最佳实践

### 1. 证据验证流程

```python
def validate_evidence_with_human_review(evidence: EvidenceItem) -> bool:
    """
    推荐的人工审核流程：
    1. 高置信度 (>0.9): 自动确认
    2. 中等置信度 (0.7-0.9): 标记待审核
    3. 低置信度 (<0.7): 自动拒绝，需要重新提取
    """
    if evidence.confidence > 0.9:
        evidence.confirm("高置信度，自动确认")
        return True
    elif evidence.confidence > 0.7:
        # 添加到待审核队列
        add_to_review_queue(evidence)
        return False
    else:
        evidence.reject("置信度太低")
        return False
```

### 2. 冲突解决策略选择

```python
def choose_resolution_strategy(field: EvidenceField) -> ConflictResolutionStrategy:
    """
    策略选择建议：
    - 数值型字段: HIGHEST_CONFIDENCE 或 WEIGHTED_AVERAGE
    - 文本型字段: MAJORITY_VOTE 或 HIGHEST_CONFIDENCE
    - 关键字段（如金额）: MANUAL（强制人工审核）
    - 时效性字段: TEMPORAL_RECENCY
    """
    if field.field_name in ["投标金额", "质保期限"]:
        return ConflictResolutionStrategy.MANUAL
    
    if field.has_conflict():
        values = field.get_unique_values()
        # 如果都是数字，尝试加权平均
        if all(can_convert_to_number(v) for v in values):
            return ConflictResolutionStrategy.WEIGHTED_AVERAGE
        else:
            return ConflictResolutionStrategy.MAJORITY_VOTE
    
    return ConflictResolutionStrategy.HIGHEST_CONFIDENCE
```

### 3. 性能优化

```python
# 批量处理证据
def batch_process_evidence(evidences: list[EvidenceItem]) -> None:
    """批量处理证据，减少数据库往返"""
    
    # 先验证所有证据
    valid_evidences = [e for e in evidences if validate_evidence(e)]
    
    # 批量插入
    insert_batch(valid_evidences)


# 缓存评分规则
def get_cached_rules(dimension_id: str) -> list[ScoringRule]:
    """缓存规则配置，避免重复数据库查询"""
    cache_key = f"rules:{dimension_id}"
    
    if cached := cache.get(cache_key):
        return [ScoringRule.from_dict(r) for r in cached]
    
    rules = load_rules_from_db(dimension_id)
    cache.set(cache_key, [r.to_dict() for r in rules], ttl=3600)
    
    return rules
```

### 4. 错误处理

```python
from pydantic import ValidationError


def safe_create_evidence(data: dict) -> EvidenceItem | None:
    """安全创建证据"""
    try:
        return EvidenceItem(**data)
    except ValidationError as e:
        # 记录错误，返回 None
        logger.error(f"证据创建失败: {e}")
        return None


def safe_resolve_conflict(field: EvidenceField) -> EvidenceItem | None:
    """安全解决冲突"""
    try:
        return field.resolve_conflict()
    except ValueError as e:
        # 冲突解决失败（如无候选）
        logger.warning(f"冲突解决失败: {e}")
        return None
```

---

## 故障排查

### 常见问题

#### 1. ValidationError: x2 必须大于等于 x1

**原因**: BoundingBox 坐标顺序错误

**解决**:
```python
# 错误
bbox = BoundingBox(x1=100, y1=200, x2=50, y2=400)

# 正确
bbox = BoundingBox(x1=50, y1=200, x2=100, y2=400)
```

#### 2. ValidationError: 必须提供 days 或 hours 至少一个时长值

**原因**: DurationEvidence 没有提供任何时长值

**解决**:
```python
# 错误
evidence = DurationEvidence(field_value="2天", ...)

# 正确
evidence = DurationEvidence(
    field_value="2天",
    days=2.0,  # 或 hours=48.0
    ...
)
```

#### 3. ValueError: 字段名不匹配

**原因**: EvidenceField 和 EvidenceItem 的 field_name 不一致

**解决**:
```python
field = EvidenceField(field_name="培训时长")
evidence = EvidenceItem(field_name="培训时长", ...)  # 必须相同
field.add_candidate(evidence)
```

#### 4. 冲突解决返回 None

**原因**: 使用 MANUAL 策略或没有候选

**解决**:
```python
selected = field.resolve_conflict(strategy=ConflictResolutionStrategy.MANUAL)
if selected is None:
    # 需要人工处理
    send_to_manual_review(field)
```

---

## 下一步

- [ ] 实现前端可视化界面
- [ ] 添加更多评分维度（如价格、技术方案）
- [ ] 集成机器学习模型自动提取证据
- [ ] 实现批量评分 API
- [ ] 添加评分报告导出功能

---

**更多示例代码请参考**: `tests/test_scoring_schema.py`
