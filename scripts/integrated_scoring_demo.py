#!/usr/bin/env python3
"""
回标分析评分 Schema - 真实数据集成测试

使用真实投标文件数据进行端到端评分演示。

Usage:
    cd /Users/wangxq/Documents/投标分析_kimi
    source .venv/bin/activate
    python scripts/integrated_scoring_demo.py
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, '/Users/wangxq/Documents/投标分析_kimi')

from dotenv import load_dotenv
load_dotenv(override=True)

from bid_scoring.citation_rag_pipeline import CitationRAGPipeline, HighlightBox
from bid_scoring.scoring_schema import (
    BoundingBox, EvidenceItem, DurationEvidence, ResponseTimeEvidence, WarrantyEvidence,
    EvidenceField, ConflictResolutionStrategy,
    TrainingPlan, AfterSalesService, ScoringResult, DimensionScore,
    CompletenessLevel, ServiceLevel,
)


# 文档版本 ID
VERSION_ID = "9a5a0214-3b98-4a64-9194-a01648479f7a"


class BidScoringService:
    """投标评分服务 - 集成 RAG 和评分 Schema"""
    
    def __init__(self, version_id: str):
        self.version_id = version_id
        self.rag_pipeline = CitationRAGPipeline(version_id=version_id, top_k=5)
        self.extracted_evidence: list[EvidenceItem] = []
    
    def query_and_extract(
        self,
        query: str,
        field_name: str,
        evidence_type: str = "base",
    ) -> Optional[EvidenceItem]:
        """
        查询文档并提取证据
        
        Args:
            query: 查询问题
            field_name: 字段名
            evidence_type: 证据类型 (base/duration/response_time/warranty)
        
        Returns:
            提取的证据，如果没有找到则返回 None
        """
        print(f"\n🔍 查询: {query}")
        
        # 执行 RAG 查询
        result = self.rag_pipeline.query(query, temperature=0.3)
        
        if not result.highlight_boxes:
            print("   ⚠️ 未找到相关内容")
            return None
        
        print(f"   ✓ 找到 {len(result.highlight_boxes)} 个相关区域")
        print(f"   💡 答案: {result.answer[:200]}...")
        
        # 取第一个高亮框作为主证据
        highlight: HighlightBox = result.highlight_boxes[0]
        
        # 构建证据
        evidence_data = {
            "field_name": field_name,
            "field_value": self._extract_value(result.answer, field_name),
            "source_text": highlight.text_preview,
            "page_idx": highlight.page_idx,
            "bbox": BoundingBox(
                x1=highlight.bbox.x1,
                y1=highlight.bbox.y1,
                x2=highlight.bbox.x2,
                y2=highlight.bbox.y2,
            ),
            "chunk_id": highlight.chunk_id,
            "confidence": self._calculate_confidence(result),
        }
        
        # 根据类型创建不同证据
        if evidence_type == "duration":
            parsed = self._parse_duration(result.answer)
            evidence = DurationEvidence(
                **evidence_data,
                raw_value=result.answer[:100],
                **parsed,
            )
        elif evidence_type == "response_time":
            parsed = self._parse_response_time(result.answer)
            evidence = ResponseTimeEvidence(
                **evidence_data,
                raw_value=result.answer[:100],
                **parsed,
            )
        elif evidence_type == "warranty":
            parsed = self._parse_warranty(result.answer)
            evidence = WarrantyEvidence(
                **evidence_data,
                raw_value=result.answer[:100],
                **parsed,
            )
        else:
            evidence = EvidenceItem(**evidence_data)
        
        self.extracted_evidence.append(evidence)
        
        print(f"   ✅ 提取证据: {evidence.field_value}")
        print(f"      位置: 第{evidence.page_idx}页")
        print(f"      置信度: {evidence.confidence:.2f}")
        
        return evidence
    
    def _extract_value(self, answer: str, field_name: str) -> str:
        """从答案中提取值（简化版）"""
        # 实际应用中可以使用 LLM 进行更精确的提取
        lines = answer.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:
                return line
        return answer[:50]
    
    def _calculate_confidence(self, result) -> float:
        """计算置信度"""
        # 基于引用数量和检索相似度计算
        base_confidence = 0.85
        
        # 如果有多个引用，增加置信度
        if len(result.citations) > 1:
            base_confidence += 0.05
        
        # 如果答案明确，增加置信度
        if "无法" not in result.answer and "根据" in result.answer:
            base_confidence += 0.05
        
        return min(base_confidence, 0.98)
    
    def _parse_duration(self, text: str) -> dict:
        """解析时长信息"""
        result = {"days": None, "hours": None}
        
        # 简单的规则匹配
        if "天" in text or "日" in text:
            # 尝试提取数字
            import re
            match = re.search(r'(\d+)\s*[天日]', text)
            if match:
                result["days"] = float(match.group(1))
        
        if "小时" in text or "h" in text.lower():
            import re
            match = re.search(r'(\d+)\s*[小时h]', text.lower())
            if match:
                result["hours"] = float(match.group(1))
        
        return result
    
    def _parse_response_time(self, text: str) -> dict:
        """解析响应时间"""
        result = {"response_hours": None, "on_site_hours": None}
        
        import re
        
        # 提取响应时间
        match = re.search(r'(\d+)\s*小时[内]?', text)
        if match:
            result["response_hours"] = float(match.group(1))
        
        # 提取到场时间
        match = re.search(r'(\d+)\s*小时.*到场|现场', text)
        if match:
            result["on_site_hours"] = float(match.group(1))
        
        return result
    
    def _parse_warranty(self, text: str) -> dict:
        """解析质保期限"""
        result = {"years": None, "months": None}
        
        import re
        
        # 提取年数
        match = re.search(r'(\d+)\s*年', text)
        if match:
            result["years"] = float(match.group(1))
        
        # 提取月数
        match = re.search(r'(\d+)\s*个月', text)
        if match:
            result["months"] = int(match.group(1))
        
        return result
    
    def score_training_plan(self) -> TrainingPlan:
        """评分：培训方案"""
        print("\n" + "="*60)
        print("📚 评分维度: 培训方案")
        print("="*60)
        
        plan = TrainingPlan(
            dimension_id="training",
            dimension_name="培训方案",
            weight=5.0,
            sequence=1,
        )
        
        # 定义查询
        training_queries = {
            "training_duration": {
                "query": "培训时长是多少？培训天数或小时数",
                "field_name": "培训时长",
                "type": "duration",
            },
            "training_schedule": {
                "query": "培训内容包括哪些？培训计划和方式",
                "field_name": "培训计划",
                "type": "base",
            },
            "training_personnel": {
                "query": "培训对象是谁？培训人员要求",
                "field_name": "培训人员",
                "type": "base",
            },
            "instructor_qualifications": {
                "query": "授课老师的资质如何？讲师要求",
                "field_name": "授课老师资质",
                "type": "base",
            },
        }
        
        # 提取每个字段的证据
        for attr, config in training_queries.items():
            evidence = self.query_and_extract(
                query=config["query"],
                field_name=config["field_name"],
                evidence_type=config["type"],
            )
            
            if evidence:
                field = EvidenceField(field_name=config["field_name"])
                field.add_candidate(evidence)
                field.resolve_conflict()
                setattr(plan, attr, field)
        
        # 计算评分
        completeness = plan.evaluate_completeness()
        score = plan.calculate_score()
        
        print(f"\n📊 培训方案评分结果:")
        print(f"   完整性: {completeness.value}")
        print(f"   得分: {score}/{plan.weight}")
        print(f"   得分率: {plan.get_score_ratio():.1%}")
        
        return plan
    
    def score_after_sales_service(self) -> AfterSalesService:
        """评分：售后服务方案"""
        print("\n" + "="*60)
        print("🔧 评分维度: 售后服务方案")
        print("="*60)
        
        service = AfterSalesService(
            dimension_id="after_sales",
            dimension_name="售后服务方案",
            weight=10.0,
            sequence=2,
        )
        
        # 定义查询
        service_queries = {
            "response_time": {
                "query": "售后服务响应时间是多久？多久响应，多久到达现场",
                "field_name": "响应时间",
                "type": "response_time",
            },
            "warranty_period": {
                "query": "质保期限是多长时间？保修期多久",
                "field_name": "质保期限",
                "type": "warranty",
            },
            "parts_supply_period": {
                "query": "配件供应期限是多久？耗材供应",
                "field_name": "配件供应期限",
                "type": "base",
            },
            "post_warranty_service_fee": {
                "query": "质保期后的服务费用是多少？过保后收费标准",
                "field_name": "质保期后服务费",
                "type": "base",
            },
        }
        
        # 提取每个字段的证据
        for attr, config in service_queries.items():
            evidence = self.query_and_extract(
                query=config["query"],
                field_name=config["field_name"],
                evidence_type=config["type"],
            )
            
            if evidence:
                field = EvidenceField(field_name=config["field_name"])
                field.add_candidate(evidence)
                field.resolve_conflict()
                setattr(service, attr, field)
        
        # 计算评分
        completeness = service.evaluate_completeness()
        service_level = service.evaluate_service_level()
        score = service.calculate_score()
        
        print(f"\n📊 售后服务评分结果:")
        print(f"   完整性: {completeness.value}")
        print(f"   服务等级: {service_level.value}")
        print(f"   得分: {score}/{service.weight}")
        print(f"   得分率: {service.get_score_ratio():.1%}")
        
        return service
    
    def generate_final_report(
        self,
        dimensions: list,
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
            document_version_id=self.version_id,
            dimension_scores=dimension_scores,
            total_score=total_score,
            max_possible_score=max_possible,
        )
        
        return result


def main():
    """主函数"""
    print("="*70)
    print("🎯 回标分析评分系统 - 真实数据集成测试")
    print("="*70)
    print(f"\n文档版本: {VERSION_ID}")
    print(f"投标方: 上海妙生科贸有限公司")
    print(f"项目: 共聚焦显微镜")
    print(f"时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建评分服务
    service = BidScoringService(version_id=VERSION_ID)
    
    try:
        # 执行评分
        training = service.score_training_plan()
        after_sales = service.score_after_sales_service()
        
        # 生成报告
        result = service.generate_final_report([training, after_sales])
        
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
        
        # 打印提取的所有证据摘要
        print("\n📎 提取的证据摘要")
        print("="*70)
        for i, ev in enumerate(service.extracted_evidence, 1):
            print(f"\n{i}. {ev.field_name}")
            print(f"   值: {ev.field_value}")
            print(f"   位置: 第{ev.page_idx}页")
            print(f"   置信度: {ev.confidence:.2f}")
            print(f"   来源: {ev.source_text[:50]}...")
        
        # 保存结果到文件
        output_file = "/tmp/scoring_result.json"
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
