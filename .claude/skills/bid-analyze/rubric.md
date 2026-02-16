# Scoring Rubric

## Policy Source
评分单源配置：`config/agent_scoring_policy.yaml`

## Dimension Weights
| key | weight |
|---|---:|
| warranty | 0.25 |
| delivery | 0.25 |
| training | 0.20 |
| financial | 0.20 |
| technical | 0.10 |
| compliance | 0.10 |

## Score Update Rule
1. 维度默认分：`50`
2. 优势证据：每条 `+5 ~ +15`
3. 风险证据：每条 `-20 ~ -5`
4. 维度分裁剪到 `[0, 100]`
5. 总分：`sum(score_i * weight_i)`

## Risk Level Rule
- `high`: 存在重大合规/履约风险，或高风险证据明显多于优势
- `medium`: 风险与优势并存，关键条款需澄清
- `low`: 证据完整，主要条款清晰且风险可控

## Mandatory Warning Policy
遇到任一情况必须写 warning：
1. 证据不足（缺失维度关键信息）
2. 证据不可定位（缺少 `chunk_id/page_idx/bbox`）
3. 证据冲突（同一维度出现互斥条款）

## Recommendation Rules
建议项应直接绑定风险证据，不写空泛建议：
- `financial` 风险高 -> 优先建议澄清付款节点/违约责任
- `delivery` 风险高 -> 优先建议澄清交付里程碑与验收条件
- `compliance` 风险高 -> 优先建议补齐资质/证书/合规说明
