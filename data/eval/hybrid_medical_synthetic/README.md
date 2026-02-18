# hybrid_medical_synthetic 数据集说明

## 1. 数据内容

- `content_list.synthetic_bidder_*.json`：合成投标文档内容
- `queries.json`：检索查询集
- `qrels.source_id.*.jsonl`：黄金相关性标注
- `retrieval_baseline.thresholds.json`：检索阈值基线
- `eval_summary.json`：检索评测输出（由脚本生成）

## 2. 生成评测 summary

```bash
uv run python scripts/evaluate_hybrid_search_gold.py \
  --version-id <VERSION_UUID> \
  --queries-file data/eval/hybrid_medical_synthetic/queries.json \
  --qrels-file data/eval/hybrid_medical_synthetic/qrels.source_id.A.jsonl \
  --thresholds-file data/eval/hybrid_medical_synthetic/retrieval_baseline.thresholds.json \
  --output data/eval/hybrid_medical_synthetic/eval_summary.json
```
