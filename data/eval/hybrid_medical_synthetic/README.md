# Synthetic Medical Hybrid Retrieval Eval Set

该目录包含 A/B/C 三个供应商版本的模拟投标评测集。

## 产物
- queries.json（跨版本共享）
- content_list.synthetic_bidder_A|B|C.json
- qrels.source_id.A|B|C.jsonl
- multi_version_manifest.json

## 场景概览
- A: 上海澄研医疗科技有限公司 / 1247 chunks / 45 qrels
- B: 苏州启衡生物仪器有限公司 / 1247 chunks / 45 qrels
- C: 杭州赛泓精密医疗设备有限公司 / 1247 chunks / 45 qrels

## relevance 标准
- 3: 主证据
- 2: 支持证据
- 1: 弱相关
- 0: 干扰项

## 评估建议
1. 将 A/B/C content_list 分别入库为不同 version_id。
2. 使用 scripts/evaluate_hybrid_search_multiversion.py 做跨版本基线。
3. 使用 scripts/evaluate_hybrid_search_gold.py + `retrieval_baseline.thresholds.json` 做门禁：

```bash
uv run python scripts/evaluate_hybrid_search_gold.py \
  --version-id <VERSION_ID> \
  --queries-file data/eval/hybrid_medical_synthetic/queries.json \
  --qrels-file data/eval/hybrid_medical_synthetic/qrels.source_id.jsonl \
  --thresholds-file data/eval/hybrid_medical_synthetic/retrieval_baseline.thresholds.json \
  --fail-on-thresholds
```
