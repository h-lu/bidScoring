# 项目全景手册

## 1. 一句话

这是一个以证据追溯为核心的投标分析系统：每个评分结论都要可定位到文档原文。

## 2. 已完成能力

- Evidence-first 主链路完整可跑
- 评分后端三轨并存，`hybrid` 作为生产默认
- 策略系统完成配置化重构（pack/overlay/artifact）
- 检索与评分门禁脚本可执行
- `.claude` skill 与策略一致性校验已接入

## 3. 当前生产流程

1. `run-prod` 接收输入（`context-json` 或 `pdf-path`）
2. 内容入库并建立 chunk/unit 映射
3. 构建 embeddings
4. 执行评分（默认 `hybrid`）
5. 输出 `scoring/traceability/observability`

## 4. 配置设计原则

- 所有评分策略与检索阈值可配置
- 提示词不再手工散落，统一由策略生成约束
- 运行时可直接切换 `pack/overlay` 或指定 artifact

## 5. 当前优先路线

### P0
- 用真实业务样本构建评测集
- 将检索门禁和评分门禁纳入 CI 必过项

### P1
- 做维度级 retrieval override A/B
- 加强 agent 失败归因和降级可观测性

### P2
- 策略变更影响报告自动化
- 评标人员可读的策略差异说明

## 6. 关键命令清单

```bash
uv run bid-pipeline run-prod --help
uv run python scripts/build_policy_artifacts.py --pack cn_medical_v1 --overlay strict_traceability
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
uv run python scripts/evaluate_retrieval_policy_gate.py --summary-file data/eval/hybrid_medical_synthetic/eval_summary.json --policy-pack cn_medical_v1 --policy-overlay strict_traceability --fail-on-violations
```
