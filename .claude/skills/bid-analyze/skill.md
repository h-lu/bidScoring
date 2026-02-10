---
name: bid-analyze
description: Comprehensive bid document analysis for tender evaluation. Analyzes bidding documents across 6 dimensions with scoring, ranking, and risk assessment. Use for comparing multiple bidders, evaluating bid quality, generating ranking reports, and extracting warranty terms.
---

# 投标分析 Skill

## 快速开始

调用 `analyze_bids_comprehensive` MCP工具分析投标文档。

```python
analyze_bids_comprehensive(
    version_ids=["uuid1", "uuid2"],  # 文档版本ID列表
    bidder_names={"uuid1": "公司A"}, # 可选：投标人名称
    dimensions=["warranty", "delivery"], # 可选：分析维度
    generate_annotations=True         # 可选：生成标注PDF
)
```

## MCP工具

### analyze_bids_comprehensive

一站式投标分析工具。

**输入:**
- `version_ids` (必填): 文档版本ID列表
- `bidder_names` (可选): 投标人名称映射
- `dimensions` (可选): 分析维度，默认全部6个
- `generate_annotations` (可选): 生成标注PDF

**输出:**
- `rankings`: 排名结果
- `dimension_comparison`: 维度对比表
- `recommendations`: 建议列表
- `annotated_urls`: PDF标注链接

## 分析维度

| 维度 | 权重 | 评分标准 |
|------|------|----------|
| warranty | 25% | ≥5年优秀, <3年关注 |
| delivery | 25% | <2小时优秀, >24小时差 |
| training | 20% | ≥5天优秀, <3天关注 |
| financial | 20% | 预付款>50%高风险 |
| technical | 10% | 技术参数符合度 |
| compliance | 10% | 强制要求符合度 |

## 评分规则

基础50分 + 优势加分 - 风险减分，限制在 [0, 100]。

## 参考资源

- **详细评分规则**: [prompt.md](prompt.md)
- **使用示例**: [examples.md](examples.md)
