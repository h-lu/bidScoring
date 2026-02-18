# Claude Skill 编写最佳实践

## 1. 目标

让 skill 在长期迭代中保持“可维护、可校验、可复用”。

## 2. 结构建议

- `SKILL.md`：用途、输入输出、约束
- `workflow.md`：阶段步骤与工具调用顺序
- `rubric.md`：评分规则与阈值
- `prompt.md`：由策略编译器生成，避免手工漂移
- `examples.md`：最小可复用样例

## 3. 与策略同步

必须定期运行：

```bash
uv run python scripts/check_skill_policy_sync.py --fail-on-violations
```

## 4. 常见错误

- 手工改 prompt，导致与 policy 不一致
- 在 skill 中写死已过时参数
- 只给结论不绑定证据引用
