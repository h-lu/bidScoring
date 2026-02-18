# 评分运行对比说明（2026-02-16）

## 1. 目的

记录一次基线与候选配置的对比方法，便于后续复跑。

## 2. 输入文件

- 基线结果：`data/eval/scoring_compare/runs/2026-02-14-run-prod-hybrid-synthetic-bidder-A.json`
- 候选结果：`data/eval/scoring_compare/runs/2026-02-14-run-prod-hybrid-synthetic-bidder-A-candidate2.json`
- 对比报告：`data/eval/scoring_compare/runs/2026-02-14-compare-baseline-vs-candidate2.json`

## 3. 关键信息

- 总分变化：`+3.5`
- 追溯覆盖率变化：`0.0`
- 引用总数变化：`0`
- 分维度变化：主要提升来自 `compliance/delivery/training/warranty/financial`

## 4. 复跑命令

```bash
uv run python scripts/compare_scoring_runs.py \
  --baseline data/eval/scoring_compare/runs/<baseline>.json \
  --candidate data/eval/scoring_compare/runs/<candidate>.json \
  --output data/eval/scoring_compare/runs/<compare-report>.json
```
