# Workflow

## 阶段 1：Retrieval

1. 明确要评估的维度
2. 逐维度调用 `retrieve_dimension_evidence`
3. 汇总证据与来源位置

## 阶段 2：Scoring

1. 依据 `rubric.md` 计算维度分
2. 聚合总分并给风险等级
3. 对证据不足维度保持中性分

## 阶段 3：Traceability

1. 检查每条引用是否包含 `chunk_id/page_idx/bbox`
2. 统计 `citation_coverage_ratio`
3. 输出 `warnings` 与 `highlight_ready_chunk_ids`
