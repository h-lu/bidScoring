# scoring_compare/runs 目录说明

本目录存放评分运行结果与对比报告。

## 1. 命名建议

- 单次运行：`YYYY-MM-DD-run-prod-<backend>-<case>.json`
- 对比报告：`YYYY-MM-DD-compare-<baseline>-vs-<candidate>.json`

## 2. 对比命令

```bash
uv run python scripts/compare_scoring_runs.py \
  --baseline data/eval/scoring_compare/runs/<baseline>.json \
  --candidate data/eval/scoring_compare/runs/<candidate>.json \
  --output data/eval/scoring_compare/runs/<report>.json
```

## 3. 重点关注指标

- `overall_score`
- `coverage_ratio`
- `citation_count_total`
- `chunks_analyzed`
- `dimension_scores` 差异
