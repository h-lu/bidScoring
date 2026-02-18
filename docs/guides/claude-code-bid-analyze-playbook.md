# Claude Code 可直接使用的投标评审 Playbook（含可粘贴 Prompt）

## 1. 目的

本文件提供一套“开箱可用”的方法，让你在 Claude Code 里直接运行：

1. 连接 `bid-scoring` MCP
2. 触发 `bid-analyze` skill
3. 按证据链完成评分（可定位到 PDF 原始位置）

---

## 2. 前置条件

1. 本仓库已可运行，数据库已初始化并有数据。
2. Claude Code 已安装。
3. 本机有 `uv`。
4. 环境变量可用：
   - `DATABASE_URL`
   - `OPENAI_API_KEY`（如果要用向量/混合检索）

---

## 3. 在 Claude Code 安装 MCP（推荐项目级）

在项目根目录执行：

```bash
claude mcp add-json bid-scoring '{
  "type":"stdio",
  "command":"uv",
  "args":["run","fastmcp","run","mcp_servers/retrieval_server.py","-t","stdio"],
  "env":{
    "DATABASE_URL":"${DATABASE_URL}",
    "OPENAI_API_KEY":"${OPENAI_API_KEY}"
  }
}' --scope project
```

验证：

```bash
claude mcp list
claude mcp get bid-scoring
```

看到 `bid-scoring` 即成功。

---

## 4. 先做一次连通性检查（建议）

在 Claude Code 会话里先发一条简单请求：

```text
请先调用 MCP 工具 list_available_versions，列出当前可用的 version_id 和标题。
```

如果能返回版本列表，说明 MCP 通了。

---

## 5. 单文档评分：可直接粘贴 Prompt

将下面整段原样粘贴到 Claude Code（替换 `<VERSION_ID>`）：

```text
你现在使用 bid-analyze skill 执行投标评审。目标：对 version_id=<VERSION_ID> 做6维评分，并严格基于可定位证据输出结论。

硬性要求：
1) 必须先调用 MCP 工具探索证据，再评分，禁止直接凭经验总结。
2) 每个维度至少进行两次检索（不同查询词），并对关键结果调用 get_chunk_with_context 做上下文核验。
3) 能结构化提取的条款（如质保年限、响应时间、付款比例、交付周期）必须调用 extract_key_value。
4) 每条结论必须绑定证据：chunk_id + page_idx + bbox + quote。
5) 若某维度证据不足：该维度打 50 分，并在 warnings 中标注 evidence_insufficient:<dimension>，但不要拒答。
6) 不得使用文档外知识，不得杜撰。

执行步骤：
A. 调用 list_available_versions 确认版本存在；
B. 调用 get_document_outline 了解章节分布；
C. 按维度执行检索和核验（warranty/delivery/training/financial/technical/compliance）；
D. 输出最终 JSON（不要输出额外解释文本）。

输出 JSON Schema（严格遵守）：
{
  "version_id": "string",
  "overall_score": 0,
  "risk_level": "low|medium|high",
  "total_risks": 0,
  "total_benefits": 0,
  "recommendations": ["string"],
  "dimensions": [
    {
      "key": "warranty|delivery|training|financial|technical|compliance",
      "score": 0,
      "risk_level": "low|medium|high",
      "reasoning": "string",
      "evidence": [
        {
          "chunk_id": "string",
          "page_idx": 0,
          "bbox": [0,0,0,0],
          "quote": "string"
        }
      ],
      "warnings": ["string"]
    }
  ],
  "warnings": ["string"]
}
```

---

## 6. 多投标方对比：可直接粘贴 Prompt

将下面整段粘贴到 Claude Code（替换 `<A/B/C_VERSION_ID>`）：

```text
你现在使用 bid-analyze skill 对比三家投标文件，version_id 分别是：
- A: <A_VERSION_ID>
- B: <B_VERSION_ID>
- C: <C_VERSION_ID>

目标：按同一口径完成评分、证据对齐和推荐排序。

硬性要求：
1) 每家都必须先检索后评分；证据不足维度按50分并标warning，不拒答。
2) 每家每个维度至少两次检索，并做关键上下文核验。
3) 关键条款必须使用 compare_across_versions 做横向核验（至少：质保、交付、付款、合规）。
4) 每个结论必须给可定位证据（version_id/chunk_id/page_idx/bbox/quote）。
5) 不得杜撰。

输出 JSON（严格）：
{
  "ranking": ["A","B","C"],
  "scoring_standard": "同一评分口径说明",
  "bidders": [
    {
      "name": "A|B|C",
      "version_id": "string",
      "overall_score": 0,
      "risk_level": "low|medium|high",
      "dimensions": [
        {
          "key": "warranty|delivery|training|financial|technical|compliance",
          "score": 0,
          "risk_level": "low|medium|high",
          "reasoning": "string",
          "evidence": [
            {
              "version_id": "string",
              "chunk_id": "string",
              "page_idx": 0,
              "bbox": [0,0,0,0],
              "quote": "string"
            }
          ],
          "warnings": ["string"]
        }
      ],
      "warnings": ["string"]
    }
  ],
  "cross_bid_findings": ["string"],
  "warnings": ["string"]
}
```

---

## 7. 推荐使用姿势（高成功率）

1. 先发“连通性检查 prompt”（第 4 节）。
2. 再发“单文档评分 prompt”（第 5 节）。
3. 结果回看重点：
   - 是否每维都有 `evidence[]`
   - `chunk_id/page_idx/bbox` 是否齐全
   - `warnings` 是否合理
4. 若证据太少：
   - 要求“对低置信维度追加 2 轮检索并重评”。

---

## 8. 常见问题

### Q1: Claude 没有调用工具，直接给结论

在下一条消息强制：

```text
你上一轮没有按要求调用 MCP 工具。请先调用工具完成检索证据链，再输出评分 JSON。
```

### Q2: 返回了评分但没有 bbox

在下一条消息强制：

```text
请仅保留可定位证据；每条 evidence 必须包含 chunk_id/page_idx/bbox，缺失则移除并在 warnings 标注。
```

### Q3: 分数看起来异常

在下一条消息要求：

```text
请输出每个维度的“加分证据/扣分证据”及对应引用，说明最终分数形成过程。
```

### Q4: MCP 参数校验报错（validation error）

这类报错通常是“把数组写成了字符串”。

高频错误与修复：

1. `get_page_metadata.page_idx`
   - 错误：`page_idx: "[0, 1, 2, 3]"`
   - 正确：`page_idx: [0, 1, 2, 3]`
2. `search_chunks.page_range`
   - 错误：`page_range: "[3, 8]"`
   - 正确：`page_range: [3, 8]`
3. `search_chunks.element_types`
   - 错误：`element_types: "[\"text\",\"table\"]"`
   - 正确：`element_types: ["text", "table"]`

可直接给 Claude 的纠错提示：

```text
你上一轮 MCP 调用参数类型错误。请按工具签名重试：
- 数组/列表请传原生数组，不要加引号
- page_idx 用 int 或 int[]
- page_range 用 [start, end]
仅修正参数类型并重试同一工具调用。
```
