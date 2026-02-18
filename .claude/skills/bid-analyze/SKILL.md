---
name: bid-analyze
description: 在投标评审场景使用，要求证据优先评分并强制总控与专职代理协作。
---

# 投标分析 Skill

## 协作模式
本 skill 强制使用多代理协作：
1. `bid-team-orchestrator`（总控）
2. `bid-team-evidence`
3. `bid-team-scoring`
4. `bid-team-traceability`

执行顺序：
1. 取证
2. 评分
3. 追溯审核
4. 汇总最终 JSON

## 硬规则
1. 未取证前禁止评分。
2. 禁止使用 MCP 证据之外的外部事实。
3. 证据不足时该维度给 `50` 分并加 warning。
4. 每条被采纳结论必须可追溯到 PDF 字段：
   - `version_id`
   - `chunk_id`
   - `page_idx`
   - `bbox`
   - `quote`

## 参考文件
1. `workflow.md`
2. `rubric.md`
3. `prompt.md`
4. `examples.md`
