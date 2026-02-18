# Evidence Workflow

## 1) Scope Validation
1. 确认任务类型：单文档评分 / 多文档对比。
2. 确认输入 `version_id` 是否存在（可调用 `list_available_versions`）。
3. 读取目录结构（`get_document_outline`）定位重点章节。

## 2) Evidence Collection (Per Dimension)
维度固定为：
- `warranty`
- `delivery`
- `training`
- `financial`
- `technical`
- `compliance`

每个维度至少执行 2 轮不同关键词检索，并保持“检索 -> 上下文核验 -> 结构化提取”的顺序。

优先流程（若工具可用）：
1. 调用 `retrieve_dimension_evidence` 一次拿基础证据。
2. 对关键结果抽样调用 `get_chunk_with_context` 做上下文核验。

兼容流程（若无 `retrieve_dimension_evidence`）：
1. 调用 `search_chunks` 检索候选 chunk。
2. 调用 `get_chunk_with_context` 校验前后文。
3. 对时效/比例/金额等字段调用 `extract_key_value`。

## 3) Evidence Quality Gate
每条可用证据必须满足：
1. 有 `chunk_id`
2. 有 `page_idx`
3. 有 `bbox`
4. 引用片段 `quote` 与结论语义一致

不满足则移除该证据，并在 `warnings` 记录：
- `untraceable_evidence:<dimension>`
- `evidence_conflict:<dimension>`
- `evidence_insufficient:<dimension>`

## 4) Cross-Bid Comparison (Optional)
多投标方时：
1. 先完成每家单独取证与评分。
2. 再调用 `compare_across_versions` 做关键条款横向核验。
3. 输出排序时必须附可定位证据，禁止仅给主观判断。

## 5) Scoring and Output
1. 按 `rubric.md` 计算维度分、总分、风险等级。
2. 生成严格 JSON（字段见 `prompt.md`）。
3. 若证据不足，不拒答：维度记 `50` 并写 warning。

## 6) MCP 参数类型防错（重点）
调用工具时必须传“真实类型”，不要把数组/对象包成字符串。

常见正确写法：
1. `get_page_metadata`
   - 正确：`page_idx: 3`
   - 正确：`page_idx: [0, 1, 2, 3]`
   - 错误：`page_idx: "[0, 1, 2, 3]"`
2. `search_chunks`
   - 正确：`page_range: [3, 8]`（MCP 会按 tuple 解析）
   - 错误：`page_range: "[3, 8]"`
3. `element_types`
   - 正确：`element_types: ["text", "table"]`
   - 错误：`element_types: "[\"text\",\"table\"]"`

若出现 `validation error`：
1. 不换工具，不改任务语义。
2. 仅修正参数类型后重试同一调用。
