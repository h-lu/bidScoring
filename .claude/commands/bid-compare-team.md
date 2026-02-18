# /bid-compare-team

用途：比较两个运行结果（baseline/candidate）并输出差异报告。

建议输入：
- baseline 结果文件路径
- candidate 结果文件路径

推荐命令：

```bash
uv run python scripts/compare_scoring_runs.py \
  --baseline data/eval/scoring_compare/runs/<baseline>.json \
  --candidate data/eval/scoring_compare/runs/<candidate>.json \
  --output data/eval/scoring_compare/runs/<report>.json
```
