# 评分运行归档

该目录用于保存 `bid-pipeline run-prod` 的真实输出 JSON，便于后续回归对比。

建议命名：

- `<YYYY-MM-DD>-run-prod-<backend>-<dataset>.json`

推荐配套生成对比报告：

```bash
uv run python scripts/compare_scoring_runs.py \
  --baseline data/eval/scoring_compare/runs/<baseline>.json \
  --candidate data/eval/scoring_compare/runs/<candidate>.json \
  --output data/eval/scoring_compare/runs/<compare-report>.json
```
