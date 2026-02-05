#!/usr/bin/env python3
"""
混合检索评分测试脚本

结合向量检索和关键词检索的优势:
1. 向量检索: 找语义相关内容
2. 关键词检索: 找精确匹配内容
3. RRF融合: 合并排序结果

Usage:
    python scripts/test_scoring_hybrid.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import traceback
from datetime import datetime, timezone
from typing import Any

import psycopg
from bid_scoring.config import load_settings
from bid_scoring.llm import LLMClient
from bid_scoring.scoring_schema import (
    AfterSalesService, BoundingBox, EvidenceField, EvidenceItem, TrainingPlan,
)
from bid_scoring.scoring_config import load_scoring_config
from bid_scoring.hybrid_retrieval import HybridRetriever


VERSION_ID = "83420a7c-b27b-480f-9427-565c47d2b53c"


class HybridScoringTester:
    """使用混合检索的评分测试器"""
    
    def __init__(self, version_id: str, top_k: int = 5):
        self.version_id = version_id
        self.top_k = top_k
        self.settings = load_settings()
        self.dsn = self.settings["DATABASE_URL"]
        self.llm = LLMClient(self.settings)
        self.retriever = HybridRetriever(
            version_id=version_id,
            settings=self.settings,
            top_k=top_k
        )
        self.results: dict[str, Any] = {}
        
    def extract_keywords(self, field_name: str) -> list[str]:
        """从字段名提取关键词"""
        # 字段特定的关键词映射
        keyword_map = {
            "培训时长": ["培训", "时长", "天数", "小时", "工作日"],
            "培训计划": ["培训", "计划", "内容", "课程", "安排"],
            "培训对象": ["培训", "对象", "人员", "受训", "用户"],
            "授课老师资质": ["授课", "老师", "讲师", "资质", "资格", "认证"],
            "响应时间": ["响应", "时间", "小时", "到达", "现场"],
            "质保期限": ["质保", "保修", "期限", "年", "月"],
            "配件供应期限": ["配件", "供应", "备件", "耗材", "期限"],
            "质保期后服务费": ["质保", "服务", "费用", "收费", "过保", "价格"],
        }
        return keyword_map.get(field_name, [field_name])
    
    def extract_evidence(
        self, 
        field_name: str, 
        query: str
    ) -> EvidenceItem | None:
        """使用混合检索提取证据"""
        
        # 1. 提取关键词
        keywords = self.extract_keywords(field_name)
        print(f"   关键词: {', '.join(keywords)}")
        
        # 2. 混合检索
        results = self.retriever.retrieve(query, keywords=keywords)
        
        if not results:
            print(f"   ⚠️  混合检索未返回结果")
            return None
        
        print(f"   ✓ 向量检索+关键词检索共找到 {len(results)} 个结果")
        
        # 3. 使用LLM从多个结果中提取最佳答案
        contexts = []
        for i, r in enumerate(results[:3]):
            text_clean = r.text.replace('\n', ' ').strip()[:400]
            contexts.append(f"[第{r.page_idx}页] {text_clean}...")
        
        context_text = "\n\n".join(contexts)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "你是专业的投标文档分析助手。"
                    "基于提供的多个检索结果，提取最准确的信息。"
                    "如果信息不存在或矛盾，明确回答'未找到'。"
                    "只回答提取的信息，不要解释。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"【需要提取的信息】\n{field_name}\n\n"
                    f"【检索到的文档内容】\n{context_text}\n\n"
                    f"请从上述内容中提取'{field_name}'的具体信息。"
                    f"如果没有明确信息，请回答'未找到'。"
                )
            }
        ]
        
        try:
            llm_response = self.llm.complete(messages, temperature=0.1)
            field_value = llm_response.strip()
            
            if not field_value or "未找到" in field_value or len(field_value) < 3:
                print(f"   ⚠️  LLM返回无效结果")
                return None
                
        except Exception as e:
            print(f"   ❌ LLM调用失败: {e}")
            return None
        
        # 4. 创建证据对象
        best_result = results[0]
        return EvidenceItem(
            field_name=field_name,
            field_value=field_value,
            source_text=best_result.text[:500],
            page_idx=best_result.page_idx,
            bbox=BoundingBox(x1=0, y1=0, x2=0, y2=0),
            chunk_id=best_result.chunk_id,
            confidence=0.85  # Hybrid confidence
        )
    
    def test_training_plan(self) -> TrainingPlan:
        """测试培训方案评分"""
        print("\n" + "="*70)
        print("📚 培训方案评分测试 (混合检索)")
        print("="*70)
        
        training = TrainingPlan(
            dimension_id="training",
            dimension_name="培训方案",
            weight=5.0,
            sequence=1
        )
        
        fields_config = [
            ("training_duration", "培训时长", "培训时长是多少天或小时"),
            ("training_schedule", "培训计划", "培训计划和培训内容包括哪些"),
            ("training_personnel", "培训对象", "培训对象和人员要求"),
            ("instructor_qualifications", "授课老师资质", "授课老师和讲师资质要求"),
        ]
        
        found_count = 0
        training_evidence = []
        
        for attr, field_name, query in fields_config:
            print(f"\n🔍 提取: {field_name}")
            evidence = self.extract_evidence(field_name, query)
            
            if evidence:
                print(f"   ✅ 成功提取 (第{evidence.page_idx}页)")
                print(f"   📝 内容: {evidence.field_value[:80]}...")
                
                training_evidence.append({
                    "field_name": evidence.field_name,
                    "field_value": evidence.field_value,
                    "page_idx": evidence.page_idx,
                    "chunk_id": evidence.chunk_id,
                })
                
                field = EvidenceField(field_name=field_name)
                field.add_candidate(evidence)
                field.resolve_conflict()
                setattr(training, attr, field)
                found_count += 1
            else:
                print(f"   ❌ 未能提取")
        
        completeness = training.evaluate_completeness()
        score = training.calculate_score()
        
        print(f"\n📊 培训方案: {score:.1f}/{training.weight}分 ({completeness.value})")
        
        self.results["training"] = {
            "dimension": "培训方案",
            "weight": training.weight,
            "score": score,
            "completeness": completeness.value,
            "found_fields": found_count,
            "total_fields": 4,
            "evidence": training_evidence
        }
        return training
    
    def test_after_sales_service(self) -> AfterSalesService:
        """测试售后服务评分"""
        print("\n" + "="*70)
        print("🔧 售后服务方案评分测试 (混合检索)")
        print("="*70)
        
        service = AfterSalesService(
            dimension_id="after_sales",
            dimension_name="售后服务方案",
            weight=10.0,
            sequence=2
        )
        
        fields_config = [
            ("response_time", "响应时间", "售后服务响应时间多久到达现场"),
            ("warranty_period", "质保期限", "质保期限保修期多长时间"),
            ("parts_supply_period", "配件供应期限", "配件供应耗材备件期限"),
            ("post_warranty_service_fee", "质保期后服务费", "质保期后过保服务费用收费标准"),
        ]
        
        found_count = 0
        service_evidence = []
        
        for attr, field_name, query in fields_config:
            print(f"\n🔍 提取: {field_name}")
            evidence = self.extract_evidence(field_name, query)
            
            if evidence:
                print(f"   ✅ 成功提取 (第{evidence.page_idx}页)")
                print(f"   📝 内容: {evidence.field_value[:80]}...")
                
                service_evidence.append({
                    "field_name": evidence.field_name,
                    "field_value": evidence.field_value,
                    "page_idx": evidence.page_idx,
                    "chunk_id": evidence.chunk_id,
                })
                
                field = EvidenceField(field_name=field_name)
                field.add_candidate(evidence)
                field.resolve_conflict()
                setattr(service, attr, field)
                found_count += 1
            else:
                print(f"   ❌ 未能提取")
        
        completeness = service.evaluate_completeness()
        service_level = service.evaluate_service_level()
        score = service.calculate_score()
        
        print(f"\n📊 售后服务: {score:.1f}/{service.weight}分 ({service_level.value})")
        
        self.results["after_sales"] = {
            "dimension": "售后服务方案",
            "weight": service.weight,
            "score": score,
            "completeness": completeness.value,
            "service_level": service_level.value,
            "found_fields": found_count,
            "total_fields": 4,
            "evidence": service_evidence
        }
        return service
    
    def generate_report(self) -> dict:
        """生成完整报告"""
        print("\n" + "="*70)
        print("📋 混合检索评分报告")
        print("="*70)
        
        training = self.results.get("training", {})
        after_sales = self.results.get("after_sales", {})
        
        total_score = training.get("score", 0) + after_sales.get("score", 0)
        total_weight = training.get("weight", 0) + after_sales.get("weight", 0)
        percentage = (total_score / total_weight * 100) if total_weight > 0 else 0
        
        print(f"\n📚 培训方案: {training.get('score', 0):.1f}/{training.get('weight', 5)}分")
        print(f"   找到: {training.get('found_fields', 0)}/{training.get('total_fields', 4)} 字段")
        
        print(f"\n🔧 售后服务: {after_sales.get('score', 0):.1f}/{after_sales.get('weight', 10)}分")
        print(f"   找到: {after_sales.get('found_fields', 0)}/{after_sales.get('total_fields', 4)} 字段")
        
        print(f"\n{'-'*70}")
        print(f"总分: {total_score:.1f}/{total_weight}分 ({percentage:.1f}%)")
        passing = "✅ 通过" if percentage >= 60 else "❌ 未通过"
        print(f"结果: {passing}")
        print(f"{'-'*70}")
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version_id": self.version_id,
            "retrieval_method": "hybrid",
            "total_score": total_score,
            "total_weight": total_weight,
            "percentage": percentage,
            "passed": percentage >= 60,
            "dimensions": self.results
        }


def main():
    print("="*70)
    print("🎯 混合检索评分测试")
    print("="*70)
    print(f"文档版本: {VERSION_ID}")
    print(f"时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        tester = HybridScoringTester(version_id=VERSION_ID, top_k=5)
        tester.test_training_plan()
        tester.test_after_sales_service()
        report = tester.generate_report()
        
        output_file = Path("scoring_report_hybrid.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 报告已保存: {output_file}")
        
        return 0 if report["passed"] else 1
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
