# 工作流

## 阶段一：取证
1. 校验 `version_id`。
2. 生成各维度查询词。
3. 通过 MCP 检索并核验证据。
4. 产出 `evidence_pack`。

## 阶段二：评分
1. 消费 `evidence_pack`。
2. 应用 rubric 和策略规则。
3. 产出 `scoring_pack`。

## 阶段三：追溯审核
1. 审计 `scoring_pack` 的引用完整性。
2. 校验高亮可用性。
3. 产出 `traceability_pack`。

## 参数类型防错
1. `page_idx`: int or int[].
2. `page_range`: [start, end] array.
3. 数组参数禁止字符串化。
