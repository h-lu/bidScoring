# Examples

## 示例 1：单文档评分

输入：
- `version_id=<VERSION_UUID>`
- `dimensions=[financial,compliance,delivery]`

执行：
1. 按维度调用 `retrieve_dimension_evidence`
2. 汇总可定位证据
3. 输出 `scoring_pack`

预期：
- 每个维度都包含证据引用
- 若证据不足，维度给 `50` 并写入 warning

## 示例 2：争议条款复核

输入：
- 同一文档
- 聚焦 `compliance`

执行：
1. 检索法规/资质相关条款
2. 验证引用位置可定位
3. 给出风险等级和改进建议
