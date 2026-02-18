# Claude 评分结果对比（2026-02-16）

## 1. 对比对象

- Baseline  
  `data/eval/claude_code_runs/2026-02-15-203118-claude-full-6d/summary.json`  
  `version_id=3ea2532c-6209-41bb-a91f-efc5d5cd9712`
- Candidate  
  `data/eval/claude_code_runs/2026-02-16-092024-claude-full-6d-v2/summary.json`  
  `version_id=5d872d53-cbd5-4e78-baba-972059cc27e1`
- 机器对比结果  
  `data/eval/claude_code_runs/compare/2026-02-16-claude-v1-v2-compare.json`

## 2. 总体结论

Candidate 相比 Baseline：

- `overall_score`: `+3`（`63 -> 66`）
- `risk_level`: `medium -> low`
- `total_risks`: `-2`（`2 -> 0`）
- `total_benefits`: `+2`（`5 -> 7`）

## 3. 分维度变化

| 维度 | Baseline | Candidate | Delta | Candidate Risk |
|---|---:|---:|---:|---|
| warranty | 80 | 65 | -15 | low |
| delivery | 65 | 60 | -5 | low |
| training | 60 | 60 | 0 | low |
| financial | 50 | 55 | +5 | medium |
| technical | 75 | 60 | -15 | low |
| compliance | 65 | 60 | -5 | low |

## 4. 告警变化

- Baseline warnings（2 条）：
  - `evidence_insufficient:financial:缺失付款方式、付款节点、结算周期等关键财务条款`
  - `training_plan_incomplete:未明确每批培训人数上限与具体课时数`
- Candidate warnings：空

## 5. 如何解读这份对比

1. 这是不同 `version_id` 的横向对比，反映的是两份投标文本差异，不是“同一文档调参前后”回归。
2. 如果要评估提示词/技能优化是否有效，应固定同一 `version_id` 多次跑，并对比输出稳定性与证据质量。
3. 当前结果可用于快速筛查：Candidate 总体更稳（风险更低），但 `warranty` 和 `technical` 评分明显更保守。
