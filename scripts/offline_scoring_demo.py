#!/usr/bin/env python3
"""
回标分析评分 Schema - 离线数据集成测试

直接从数据库查询已处理的文本内容，无需调用 OpenAI API。

Usage:
    cd /Users/wangxq/Documents/投标分析_kimi
    source .venv/bin/activate
    python scripts/offline_scoring_demo.py
"""

import os
import sys
import re
from datetime import datetime, timezone

sys.path.insert(0, '/Users/wangxq/Documents/投标分析_kimi')

import psycopg
from dotenv import load_dotenv
load_dotenv(override=True)

from bid_scoring.scoring_schema import (
    BoundingBox, EvidenceItem, DurationEvidence, ResponseTimeEvidence, WarrantyEvidence,
    EvidenceField, ConflictResolutionStrategy,
    TrainingPlan, AfterSalesService, ScoringResult, DimensionScore,
    CompletenessLevel, ServiceLevel,
)
from bid_scoring.config import load_settings


# 文档版本 ID
VERSION_ID = "9a5a0214-3b98-4a64-9194-a01648479f7a"


def extract_from_chunks(
    conn,
    version_id: str,
    keywords: list[str],
    field_name: str,
    evidence_type: str = "base",
) -> list[EvidenceItem]:
    """
    从数据库 chunks 中提取证据
    
    Args:
        conn: 数据库连接
        version_id: 文档版本 ID
        keywords: 关键词列表
        field_name: 字段名
        evidence_type: 证据类型
    
    Returns:
        提取的证据列表
    """
    evidences = []
    
    with conn.cursor() as cur:
        # 构建查询条件
        conditions = []
        params = [version_id]
        
        for keyword in keywords:
            conditions.append("content_for_embedding ILIKE %s")
            params.append(f'%{keyword}%')
        
        query = f"""
            SELECT node_id, heading, page_range, content_for_embedding
            FROM hierarchical_nodes
            WHERE version_id = %s 
              AND level = 2
              AND ({' OR '.join(conditions)})
            ORDER BY 
                CASE 
                    WHEN content_for_embedding ILIKE %s THEN 1
                    ELSE 2
                END,
                char_count DESC
            LIMIT 5
        """
        # 添加优先级排序参数
        params.append(f'%{keywords[0]}%')
        
        cur.execute(query, params)
        
        for row in cur.fetchall():
            node_id, heading, page_range, content = row
            
            # 构建证据
            evidence_data = {
                "field_name": field_name,
                "field_value": extract_value_from_content(content, field_name),
                "source_text": content[:200],
                "page_idx": page_range[0] if page_range else 0,
                "bbox": BoundingBox(x1=0, y1=0, x2=100, y2=100),  # 简化边界框
                "chunk_id": str(node_id),
                "confidence": calculate_confidence(content, keywords),
            }
            
            # 根据类型创建证据
            if evidence_type == "duration":
                parsed = parse_duration(content)
                if parsed.get("days") or parsed.get("hours"):
                    evidence = DurationEvidence(
                        **evidence_data,
                        raw_value=content[:100],
                        **parsed,
                    )
                else:
                    # 回退到基础证据
                    evidence = EvidenceItem(**evidence_data)
            elif evidence_type == "response_time":
                parsed = parse_response_time(content)
                evidence = ResponseTimeEvidence(
                    **evidence_data,
                    raw_value=content[:100],
                    **parsed,
                )
            elif evidence_type == "warranty":
                parsed = parse_warranty(content)
                evidence = WarrantyEvidence(
                    **evidence_data,
                    raw_value=content[:100],
                    **parsed,
                )
            else:
                evidence = EvidenceItem(**evidence_data)
            
            evidences.append(evidence)
    
    return evidences


def extract_value_from_content(content: str, field_name: str) -> str:
    """从内容中提取值"""
    # 简化处理：取前50个字符作为值
    value = content.strip()[:50]
    return value


def calculate_confidence(content: str, keywords: list[str]) -> float:
    """计算置信度"""
    base_confidence = 0.75
    
    # 根据关键词匹配程度调整
    content_lower = content.lower()
    matches = sum(1 for k in keywords if k.lower() in content_lower)
    
    if matches >= 2:
        base_confidence += 0.1
    if matches >= 3:
        base_confidence += 0.05
    
    # 根据内容长度调整（较长内容通常信息更丰富）
    if len(content) > 200:
        base_confidence += 0.05
    
    return min(base_confidence, 0.95)


def parse_duration(text: str) -> dict:
    """解析时长信息"""
    result = {"days": None, "hours": None}
    
    # 匹配天数
    match = re.search(r'(\d+)\s*[天日]', text)
    if match:
        result["days"] = float(match.group(1))
    
    # 匹配小时数
    match = re.search(r'(\d+)\s*小时', text)
    if match:
        result["hours"] = float(match.group(1))
    
    return result


def parse_response_time(text: str) -> dict:
    """解析响应时间"""
    result = {"response_hours": None, "on_site_hours": None}
    
    # 匹配响应时间
    match = re.search(r'(\d+)\s*小时[内]?', text)
    if match and match.group(1):
        result["response_hours"] = float(match.group(1))
    
    # 匹配到场时间 (使用独立的变量)
    match2 = re.search(r'(\d+)\s*小时.*到场|现场', text)
    if match2 and match2.group(1):
        result["on_site_hours"] = float(match2.group(1))
    
    # 如果没有明确的小时数，根据关键词判断
    if result["response_hours"] is None and ("即时" in text or "立即" in text or "马上" in text):
        result["response_hours"] = 1.0
    
    return result


def parse_warranty(text: str) -> dict:
    """解析质保期限"""
    result = {"years": None, "months": None}
    
    # 匹配年数
    match = re.search(r'(\d+)\s*年', text)
    if match:
        result["years"] = float(match.group(1))
    
    # 匹配月数
    match = re.search(r'(\d+)\s*个月', text)
    if match:
        result["months"] = int(match.group(1))
    
    # 匹配 "60个月" 这样的格式
    match = re.search(r'(\d+)\s*个?月', text)
    if match and not result["months"]:
        months = int(match.group(1))
        if months > 12:
            result["years"] = months / 12
        else:
            result["months"] = months
    
    return result


def score_training_plan(conn, version_id: str) -> TrainingPlan:
    """评分：培训方案"""
    print("\n" + "="*70)
    print("📚 评分维度: 培训方案")
    print("="*70)
    
    plan = TrainingPlan(
        dimension_id="training",
        dimension_name="培训方案",
        weight=5.0,
        sequence=1,
    )
    
    # 定义查询配置
    training_configs = {
        "training_duration": {
            "keywords": ["培训", "天数", "小时", "时长"],
            "field_name": "培训时长",
            "type": "duration",
        },
        "training_schedule": {
            "keywords": ["培训内容", "培训课程", "培训方式", "现场授课"],
            "field_name": "培训计划",
            "type": "base",
        },
        "training_personnel": {
            "keywords": ["培训人员", "培训对象", "使用人员", "管理人员"],
            "field_name": "培训人员",
            "type": "base",
        },
        "instructor_qualifications": {
            "keywords": ["讲师", "授课老师", "培训师资", "工程师"],
            "field_name": "授课老师资质",
            "type": "base",
        },
    }
    
    # 提取每个字段的证据
    for attr, config in training_configs.items():
        print(f"\n🔍 查找: {config['field_name']}")
        
        evidences = extract_from_chunks(
            conn=conn,
            version_id=version_id,
            keywords=config["keywords"],
            field_name=config["field_name"],
            evidence_type=config["type"],
        )
        
        if evidences:
            print(f"   ✓ 找到 {len(evidences)} 个相关段落")
            
            field = EvidenceField(field_name=config["field_name"])
            
            for ev in evidences:
                field.add_candidate(ev)
                print(f"     - {ev.field_value[:40]}... (置信度: {ev.confidence:.2f})")
            
            # 解决冲突
            field.resolve_conflict(strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
            setattr(plan, attr, field)
        else:
            print("   ⚠️ 未找到相关内容")
    
    # 计算评分
    completeness = plan.evaluate_completeness()
    score = plan.calculate_score()
    
    print(f"\n📊 培训方案评分:")
    print(f"   完整性: {completeness.value}")
    print(f"   得分: {score}/{plan.weight}")
    print(f"   得分率: {plan.get_score_ratio():.1%}")
    
    return plan


def score_after_sales_service(conn, version_id: str) -> AfterSalesService:
    """评分：售后服务方案"""
    print("\n" + "="*70)
    print("🔧 评分维度: 售后服务方案")
    print("="*70)
    
    service = AfterSalesService(
        dimension_id="after_sales",
        dimension_name="售后服务方案",
        weight=10.0,
        sequence=2,
    )
    
    # 定义查询配置
    service_configs = {
        "response_time": {
            "keywords": ["响应时间", "响应", "到达现场", "上门"],
            "field_name": "响应时间",
            "type": "response_time",
        },
        "warranty_period": {
            "keywords": ["质保", "保修", "保修期", "质量保证"],
            "field_name": "质保期限",
            "type": "warranty",
        },
        "parts_supply_period": {
            "keywords": ["配件", "耗材", "供应", "备件"],
            "field_name": "配件供应期限",
            "type": "base",
        },
        "post_warranty_service_fee": {
            "keywords": ["过保", "质保期后", "保修期后", "服务费用"],
            "field_name": "质保期后服务费",
            "type": "base",
        },
    }
    
    # 提取每个字段的证据
    for attr, config in service_configs.items():
        print(f"\n🔍 查找: {config['field_name']}")
        
        evidences = extract_from_chunks(
            conn=conn,
            version_id=version_id,
            keywords=config["keywords"],
            field_name=config["field_name"],
            evidence_type=config["type"],
        )
        
        if evidences:
            print(f"   ✓ 找到 {len(evidences)} 个相关段落")
            
            field = EvidenceField(field_name=config["field_name"])
            
            for ev in evidences:
                field.add_candidate(ev)
                print(f"     - {ev.field_value[:40]}... (置信度: {ev.confidence:.2f})")
                
                # 打印结构化解析结果
                if isinstance(ev, (DurationEvidence, ResponseTimeEvidence, WarrantyEvidence)):
                    if hasattr(ev, 'total_hours') and ev.total_hours:
                        print(f"       解析: {ev.total_hours}小时")
                    if hasattr(ev, 'response_hours') and ev.response_hours:
                        print(f"       解析: 响应{ev.response_hours}小时")
                    if hasattr(ev, 'total_months') and ev.total_months:
                        print(f"       解析: {ev.total_months}个月 ({ev.total_months/12:.1f}年)")
            
            field.resolve_conflict(strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE)
            setattr(service, attr, field)
        else:
            print("   ⚠️ 未找到相关内容")
    
    # 计算评分
    completeness = service.evaluate_completeness()
    service_level = service.evaluate_service_level()
    score = service.calculate_score()
    
    print(f"\n📊 售后服务评分:")
    print(f"   完整性: {completeness.value}")
    print(f"   服务等级: {service_level.value}")
    print(f"   得分: {score}/{service.weight}")
    print(f"   得分率: {service.get_score_ratio():.1%}")
    
    return service


def generate_final_report(
    dimensions: list,
    version_id: str,
) -> ScoringResult:
    """生成最终评分报告"""
    
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
    
    result = ScoringResult(
        bid_id="bid-253135-妙生",
        document_version_id=version_id,
        dimension_scores=dimension_scores,
        total_score=total_score,
        max_possible_score=max_possible,
    )
    
    return result


def main():
    """主函数"""
    print("="*70)
    print("🎯 回标分析评分系统 - 离线数据集成测试")
    print("="*70)
    print(f"\n文档版本: {VERSION_ID}")
    print(f"投标方: 上海妙生科贸有限公司")
    print(f"项目: 共聚焦显微镜")
    print(f"时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    settings = load_settings()
    
    try:
        # 连接数据库
        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            print("\n✅ 数据库连接成功")
            
            # 执行评分
            training = score_training_plan(conn, VERSION_ID)
            after_sales = score_after_sales_service(conn, VERSION_ID)
            
            # 生成报告
            result = generate_final_report([training, after_sales], VERSION_ID)
            
            # 打印最终报告
            print("\n" + "="*70)
            print("📋 最终评分报告")
            print("="*70)
            
            for ds in result.dimension_scores:
                print(f"\n{ds.dimension_name}")
                print(f"  权重: {ds.weight}分")
                print(f"  得分: {ds.score:.1f}分")
                print(f"  得分率: {ds.score/ds.weight:.1%}")
                print(f"  完整性: {ds.completeness.value}")
                print(f"  证据数: {ds.evidence_count}")
            
            print("\n" + "-"*70)
            print(f"总分: {result.total_score:.1f}/{result.max_possible_score:.1f}")
            print(f"得分率: {result.score_percentage:.1f}%")
            print(f"评审结果: {'✅ 通过' if result.is_passing else '❌ 未通过'}")
            print("-"*70)
            
            # 保存结果
            output_file = "/tmp/scoring_result_offline.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            print(f"\n💾 详细结果已保存到: {output_file}")
            
    except Exception as e:
        print(f"\n❌ 评分过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
