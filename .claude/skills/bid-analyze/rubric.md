# 评分规则（Rubric）

策略来源：
1. `config/policy/packs/cn_medical_v1/base.yaml`
2. `config/policy/packs/cn_medical_v1/overlays/strict_traceability.yaml`
3. （可选运行时产物）`artifacts/policy/cn_medical_v1/strict_traceability/runtime_policy.json`

维度与权重：
1. `warranty`: 0.25
2. `delivery`: 0.25
3. `training`: 0.20
4. `financial`: 0.20
5. `technical`: 0.10
6. `compliance`: 0.10

评分方法：
1. 每个维度基线分：`50`
2. 优势证据加分：`+5 ~ +15`
3. 风险证据减分：`-20 ~ -5`
4. 分数裁剪到 `[0, 100]`
5. 按权重计算总分
