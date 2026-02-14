---
name: bid-analyze
description: Use when Claude Code needs to evaluate bidding documents with evidence-first scoring, by iteratively calling retrieval MCP tools before generating scores.
---

# Bid Analyze Skill (MCP Agentic)

## Purpose
让 Claude Code 先用 MCP 工具探索文档，再基于证据评分。禁止把整份文档一次性塞给模型后直接打分。

## Hard Rules
1. 先检索后评分：必须先调用 MCP 工具收集证据，再输出分数。
2. 事实可定位：每个结论必须绑定 `chunk_id + page_idx + bbox`。
3. 证据不足不拒答：给出 `warning`，该维度按中性分 `50`。
4. 不得杜撰：不得使用文档外知识补全事实。

## Required MCP Workflow
1. `list_available_versions` / 使用用户提供的 `version_id`。
2. `get_document_outline` 快速定位章节。
3. 按维度循环检索：
   - `search_chunks`（每个维度至少 2 个查询词）
   - `get_chunk_with_context`（核实上下文，避免断章取义）
   - `extract_key_value`（提取时效/年限/比例等关键值）
4. 多投标方时，用 `compare_across_versions` 生成横向对比证据。
5. 依据 `prompt.md` 的评分规则计算维度分和总分。

## Output Contract
输出必须同时包含：
- `summary`: 总分、风险等级、核心建议
- `dimensions[]`: 每个维度的 `score/risk_level/reasoning`
- `evidence[]`: 每条证据的 `version_id/chunk_id/page_idx/bbox/quote`
- `warnings[]`: 证据缺失、定位失败、信息冲突

## Update Points
- 调整评分标准：编辑 `prompt.md` 的“维度权重/阈值/加减分规则”。
- 调整 Agent 行为：编辑本文件的 `Required MCP Workflow`。
- 调整输出样式：编辑 `examples.md`。
