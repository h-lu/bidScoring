# Claude Agent Team 协作模式

## 1. 模式

- `single-pass`：单轮汇总，速度优先
- `staged`：分 retrieval/scoring/traceability 多阶段

当前推荐 `staged`，便于审计和复盘。

## 2. 交互规范

- 每个阶段输出结构化中间结果
- 阶段间只传必要字段，避免信息泄漏与漂移
- 失败阶段必须输出 machine-readable warning code

## 3. 质量门禁

- 无可定位证据不得输出高置信结论
- 缺失关键字段时必须降级并告警
