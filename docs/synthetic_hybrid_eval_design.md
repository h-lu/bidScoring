# 混合检索模拟评测集设计（医疗器械投标场景）

## 目标
构建一套贴近真实投标文件结构的模拟数据，并提供可执行黄金标准（qrels），用于评估 `vector / keyword / hybrid` 三类检索在真实业务问题上的差异。

## 数据来源约束
设计对齐以下真实结构：
- `data/clean/..._content_list.json`（MineRU 输出）
- 数据库 `chunks` 表字段：`source_id/chunk_index/page_idx/element_type/text_raw/list_items/table_body/...`
- 入库规则：`header/footer/page_number` 被跳过，不进入可检索正文

## 产物结构
目录：`data/eval/hybrid_medical_synthetic`

1. `content_list.synthetic_bidder_A.json` / `content_list.synthetic_bidder_B.json` / `content_list.synthetic_bidder_C.json`
- 模拟三家供应商（A/B/C）投标文件，字段结构一致但措辞/品牌/联系方式不同
- 覆盖 `text/list/table/image/aside_text/header/page_number`
- 章节包含：资格、技术、售后、培训、商务、业绩
- `content_list.synthetic_bidder.json` 作为 A 场景的兼容别名（方便单版本快速跑通）

2. `queries.json`
- 查询元数据（跨版本共享）：`query_id/query/keywords/query_type/intent/expected_answer/edge_focus`
- 重点 query_type：`keyword_critical/factual/semantic/negation/numeric_reasoning/conflict_resolution`

3. `qrels.source_id.A.jsonl` / `qrels.source_id.B.jsonl` / `qrels.source_id.C.jsonl`
- 黄金标注按 `source_id` 给分（每个 scenario 一份，便于后续引入 scenario 差异标注）
- `qrels.source_id.jsonl` 作为 A 场景的兼容别名
- relevance 分级：
  - `3`: 直接证据（主承诺）
  - `2`: 支持证据（同义或辅助）
  - `1`: 弱相关（上下文）
  - `0`: 干扰项（hard negative）

4. `multi_version_manifest.json`
- 记录 A/B/C 的文件名与统计信息，用于集成和自动化

## 关键边界场景
- 冲突条款：同一主题出现“主承诺”和“模板条款”
- 否定表达：例如“不包含第三方插件升级费用”
- 数值归一：`5年` vs `60个月`、`2小时` vs OCR 变形写法
- OCR 噪声：`响 应 时 间 2 小 时`
- 中英混合：`Service SLA: first response within 2 hours`
- 表格证据：训练计划、付款节点、业绩数量等

## 评估口径
使用 `scripts/evaluate_hybrid_search_gold.py`：
- 主指标：`MRR`、`Recall@K`、`Precision@K`、`nDCG@K`
- 其中 `nDCG` 使用 graded relevance（0-3）
- Recall/MRR 默认将 `relevance >= 2` 视为“相关”

## 与现有系统集成
1. 生成数据：
```bash
uv run python scripts/generate_synthetic_hybrid_eval_data.py \
  --output-dir data/eval/hybrid_medical_synthetic \
  --all-scenarios
```

2. 导入 content_list（示例 UUID 可替换，A/B/C 分别用不同 version_id）：
```bash
# A
uv run python scripts/ingest_mineru.py \
  --path data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_A.json \
  --project-id 11111111-1111-1111-1111-111111111111 \
  --document-id 22222222-2222-2222-2222-222222222222 \
  --version-id 33333333-3333-3333-3333-333333333333 \
  --document-title "模拟投标文件-医疗器械(A)"

# B
uv run python scripts/ingest_mineru.py \
  --path data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_B.json \
  --project-id 11111111-1111-1111-1111-111111111111 \
  --document-id 22222222-2222-2222-2222-222222222222 \
  --version-id 44444444-4444-4444-4444-444444444444 \
  --document-title "模拟投标文件-医疗器械(B)"

# C
uv run python scripts/ingest_mineru.py \
  --path data/eval/hybrid_medical_synthetic/content_list.synthetic_bidder_C.json \
  --project-id 11111111-1111-1111-1111-111111111111 \
  --document-id 22222222-2222-2222-2222-222222222222 \
  --version-id 55555555-5555-5555-5555-555555555555 \
  --document-title "模拟投标文件-医疗器械(C)"
```

3. 生成向量（否则 vector/hybrid 会退化为 0 分）：  
```bash
uv run python scripts/build_all_embeddings.py \
  --version-id 33333333-3333-3333-3333-333333333333 \
  --skip-contextual --skip-hierarchical

uv run python scripts/build_all_embeddings.py \
  --version-id 44444444-4444-4444-4444-444444444444 \
  --skip-contextual --skip-hierarchical

uv run python scripts/build_all_embeddings.py \
  --version-id 55555555-5555-5555-5555-555555555555 \
  --skip-contextual --skip-hierarchical
```

4. 单版本黄金评测（示例：A）：
```bash
uv run python scripts/evaluate_hybrid_search_gold.py \
  --version-id 33333333-3333-3333-3333-333333333333 \
  --queries-file data/eval/hybrid_medical_synthetic/queries.json \
  --qrels-file data/eval/hybrid_medical_synthetic/qrels.source_id.A.jsonl \
  --top-k 10 \
  --output data/eval/hybrid_medical_synthetic/eval_report.json
```

5. 跨版本基线评测（A/B/C）：
```bash
cat > /tmp/synthetic_version_map.json <<'JSON'
{
  "A": "33333333-3333-3333-3333-333333333333",
  "B": "44444444-4444-4444-4444-444444444444",
  "C": "55555555-5555-5555-5555-555555555555"
}
JSON

uv run python scripts/evaluate_hybrid_search_multiversion.py \
  --version-map /tmp/synthetic_version_map.json \
  --queries-file data/eval/hybrid_medical_synthetic/queries.json \
  --qrels-dir data/eval/hybrid_medical_synthetic \
  --top-k 10 \
  --output data/eval/hybrid_medical_synthetic/multiversion_report.json
```

## 扩展建议
- 每月新增 5~10 条失败查询到 `queries.json` 与 `qrels`
- 为每个 query 增加 `hard_negative`，专测误召回
- 增加多版本（不同供应商）模拟，比较跨版本鲁棒性
