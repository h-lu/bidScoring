# Usage Examples

## 1) 单文档评分
用户输入：
> 用 bid-analyze 评估 version_id=xxxx，输出 6 维评分和证据。

期望执行：
1. 调 `list_available_versions` / `get_document_outline`
2. 每维度至少两轮检索
3. 关键证据调 `get_chunk_with_context`
4. 输出 JSON（含 evidence + warnings）

## 2) 多投标方对比
用户输入：
> 对比 A/B/C 三家，给排序和依据。

期望执行：
1. 每家先完成单文档评分
2. 调 `compare_across_versions` 做关键条款横向核验
3. 输出每家评分、差异点、推荐顺位

## 3) 证据不足示例
当 `training` 找不到有效证据时：
- `training.score = 50`
- `training.warnings` 包含 `evidence_insufficient:training`
- 顶层 `warnings` 同步记录

## 4) 输出最小示例
```json
{
  "overall_score": 76.5,
  "risk_level": "medium",
  "total_risks": 4,
  "total_benefits": 7,
  "recommendations": ["先澄清付款节点和违约责任"],
  "dimensions": [
    {
      "key": "warranty",
      "score": 88,
      "risk_level": "low",
      "reasoning": "质保和响应SLA条款完整",
      "evidence": [
        {
          "version_id": "...",
          "chunk_id": "...",
          "page_idx": 17,
          "bbox": [76, 156, 1116, 186],
          "quote": "2小时响应，24小时到场"
        }
      ],
      "warnings": []
    }
  ],
  "warnings": []
}
```
