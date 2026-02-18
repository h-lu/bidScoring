# Policy-as-Config Refactor Plan (Archived)

## 1. 目标

把分散在代码与提示词中的评分策略统一为可配置策略包，并提供可编译、可校验、可审计能力。

## 2. 结果

该计划已完成并合入主线，核心产物如下：

- 策略包目录：`config/policy/packs/cn_medical_v1/*`
- 加载/校验：`bid_scoring/policy/loader.py`
- 编译器：`bid_scoring/policy/compiler.py`
- 编译脚本：`scripts/build_policy_artifacts.py`
- 一致性门禁：`scripts/check_skill_policy_sync.py`
- 检索策略门禁：`scripts/evaluate_retrieval_policy_gate.py`

## 3. 验证基线

- `tests/unit/policy/*`
- `tests/unit/pipeline/test_scoring_agent_policy.py`
- `tests/unit/test_skill_policy_sync.py`

## 4. 当前默认配置

- pack: `cn_medical_v1`
- overlay: `strict_traceability`

## 5. 归档说明

本文件从“执行计划”转为“完成记录”。后续策略演进请新增新的计划文件。
