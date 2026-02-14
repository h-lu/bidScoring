# Usage Examples

## 1) 单文档评分（推荐）
用户：
> 用 bid-analyze skill 评估 version_id=xxxx 的投标文件，输出6维评分和证据。

Agent 预期行为：
1. `get_document_outline(version_id)`
2. 对每个维度执行 `search_chunks`（至少两组查询）
3. 对关键 chunk 执行 `get_chunk_with_context`
4. 必要时 `extract_key_value`
5. 输出 JSON 评分 + warnings

## 2) 多投标方对比评分
用户：
> 对比 version A/B/C，按同一标准评分并给推荐顺位。

Agent 预期行为：
1. 每个 version 先做单文档证据采集
2. `compare_across_versions(version_ids, query)` 做关键条款横向核验
3. 输出每家评分、差异点、推荐顺位

## 3) 证据不足处理
用户：
> 文档里找不到培训条款时怎么处理？

Agent 预期行为：
- `training.score = 50`
- 在 `training.warnings` 与顶层 `warnings` 加入 `evidence_insufficient:training`
- 不拒绝回答，继续完成其余维度评分

## 4) 输出（精简示例）
```json
{
  "overall_score": 78.4,
  "risk_level": "medium",
  "recommendations": ["先澄清付款节点和违约责任"],
  "dimensions": [
    {
      "key": "warranty",
      "score": 88,
      "risk_level": "low",
      "reasoning": "质保期与响应SLA条款完整",
      "evidence": [
        {
          "version_id": "...",
          "chunk_id": "...",
          "page_idx": 17,
          "bbox": [76,156,1116,186],
          "quote": "2小时响应，24小时到场"
        }
      ],
      "warnings": []
    }
  ],
  "warnings": []
}
```
