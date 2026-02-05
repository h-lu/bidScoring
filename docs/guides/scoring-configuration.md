# 评分标准配置指南

> 支持从 YAML/JSON 配置文件加载评分标准，实现灵活的评分规则定制。

---

## 📁 配置文件位置

默认配置文件：`config/scoring_standards.yaml`

---

## ⚙️ 配置结构

### 1. 售后服务评分标准 (`after_sales_service`)

```yaml
after_sales_service:
  weight: 10.0                    # 维度权重
  description: "售后服务方案评分"
  
  # 服务等级评估标准
  service_level_criteria:
    response_time:                # 响应时间标准
      excellent: 2                # 优秀：≤2小时
      standard: 24                # 标准：≤24小时
      unit: "hours"
    
    warranty_period:              # 质保期限标准
      excellent: 5                # 优秀：≥5年
      standard: 3                 # 标准：≥3年
      unit: "years"
    
    on_site_time:                 # 到场时间标准
      excellent: 24               # 优秀：≤24小时
      standard: 48                # 标准：≤48小时
      unit: "hours"
  
  # 各项评分权重
  scoring_weights:
    response_time: 2              # 响应时间权重
    warranty_period: 2            # 质保期限权重
    parts_supply: 1               # 配件供应权重
    post_warranty_fee: 1          # 质保期后费用权重
    on_site_time: 2               # 到场时间权重
  
  # 评分规则
  scoring_rules:
    - name: "excellent"
      min_score: 5                # 最低得分
      score_range: [8.0, 10.0]    # 分数范围
      description: "售后服务方案优秀"
    - name: "standard"
      min_score: 3
      score_range: [4.0, 7.5]
      description: "售后服务方案标准"
    - name: "poor"
      min_score: 1
      score_range: [0.0, 3.5]
      description: "售后服务方案不足"
```

### 2. 培训方案评分标准 (`training_plan`)

```yaml
training_plan:
  weight: 5.0
  description: "培训方案评分"
  
  # 评分规则（按填写字段数）
  scoring_rules:
    - name: "complete"
      min_fields: 4               # 4个字段都填写
      score_range: [4.0, 5.0]
      description: "培训方案完整"
    - name: "partial"
      min_fields: 2               # 填写2-3个字段
      score_range: [2.0, 3.5]
      description: "培训方案较全面"
    - name: "minimal"
      min_fields: 1               # 填写1个字段
      score_range: [0.0, 1.5]
      description: "培训方案简单"
  
  # 必填字段
  required_fields:
    - name: "training_duration"
      field_name: "培训时长"
      weight: 1.0
    - name: "training_schedule"
      field_name: "培训计划"
      weight: 1.0
    # ...
```

### 3. 通用配置 (`general`)

```yaml
general:
  passing_threshold: 60.0         # 通过线（百分比）
  confidence_threshold: 0.8       # 置信度阈值
  default_conflict_strategy: "highest_confidence"
  
  evidence_validation:
    auto_confirm_high_confidence: true
    high_confidence_threshold: 0.9
    manual_review_threshold: 0.7
```

---

## 🚀 使用方法

### 方法 1：使用默认配置

```python
from bid_scoring.scoring_schema import AfterSalesService, TrainingPlan

# 自动加载默认配置
service = AfterSalesService(
    dimension_id="after_sales",
    dimension_name="售后服务方案",
    weight=10.0,
    sequence=1,
)

# 配置会自动生效
score = service.calculate_score()
```

### 方法 2：加载自定义配置

```python
from bid_scoring.scoring_config import load_scoring_config
from bid_scoring.scoring_schema import AfterSalesService

# 加载自定义配置文件
config = load_scoring_config("/path/to/custom_scoring.yaml")

# 创建评分维度（自动使用新配置）
service = AfterSalesService(
    dimension_id="after_sales",
    dimension_name="售后服务方案",
    weight=10.0,
    sequence=1,
)
```

### 方法 3：动态修改配置

```python
from bid_scoring.scoring_config import get_scoring_config

# 获取当前配置
config = get_scoring_config()

# 修改标准（仅影响当前进程）
config.after_sales_service.service_level_criteria.response_time.excellent = 4  # 改为4小时

# 重新评估
service = AfterSalesService(...)
```

---

## 📝 配置示例：适配不同招标要求

### 示例 1：严格的响应时间要求

```yaml
# strict_scoring.yaml
after_sales_service:
  service_level_criteria:
    response_time:
      excellent: 1        # 1小时内响应
      standard: 4         # 4小时内响应
    warranty_period:
      excellent: 5
      standard: 3
  scoring_weights:
    response_time: 3      # 提高响应时间权重
    warranty_period: 1
```

### 示例 2：宽松的质量标准

```yaml
# relaxed_scoring.yaml
after_sales_service:
  service_level_criteria:
    response_time:
      excellent: 8        # 8小时内响应
      standard: 24
    warranty_period:
      excellent: 3        # 3年即可
      standard: 1
  scoring_rules:
    - name: "excellent"
      min_score: 3        # 降低优秀门槛
      score_range: [8.0, 10.0]
```

### 示例 3：工作日响应模式

```yaml
# business_day_scoring.yaml
after_sales_service:
  required_fields:
    - name: "response_time"
      field_name: "响应时间"
      patterns:
        - "{value}个工作日内"    # 支持工作日模式
        - "{value}小时"
```

---

## 🔧 高级用法

### 从招标文件自动生成配置

```python
import yaml

# 解析招标文件，提取评分标准
def extract_scoring_standards_from_bid_document(doc_text: str) -> dict:
    """从招标文件文本中提取评分标准"""
    standards = {
        "after_sales_service": {
            "service_level_criteria": {
                "response_time": {
                    "excellent": extract_hours(doc_text, "响应时间"),
                    "standard": 24
                }
            }
        }
    }
    return standards

# 保存为配置文件
config = extract_scoring_standards_from_bid_document(bid_text)
with open("extracted_scoring.yaml", "w") as f:
    yaml.dump(config, f)
```

### 多项目配置管理

```python
# 不同项目使用不同配置
PROJECT_CONFIGS = {
    "medical_equipment": "config/scoring_medical.yaml",
    "lab_instruments": "config/scoring_lab.yaml",
    "office_supplies": "config/scoring_office.yaml",
}

def score_bid(project_type: str, version_id: str):
    config_path = PROJECT_CONFIGS.get(project_type)
    load_scoring_config(config_path)
    
    # 执行评分...
```

---

## ✅ 配置验证

```python
from bid_scoring.scoring_config import load_scoring_config

# 加载并验证配置
try:
    config = load_scoring_config("config/scoring_standards.yaml")
    print(f"✅ 配置加载成功")
    print(f"售后服务权重: {config.after_sales_service.weight}")
    print(f"响应时间优秀标准: {config.after_sales_service.service_level_criteria.response_time.excellent}小时")
except FileNotFoundError as e:
    print(f"❌ 配置文件不存在: {e}")
except ValueError as e:
    print(f"❌ 配置格式错误: {e}")
```

---

## 📚 配置文件模板

完整模板请参考：`config/scoring_standards.yaml`

---

**通过配置文件，你可以：**
- ✅ 根据不同的招标文件灵活调整评分标准
- ✅ 支持不同行业的特殊要求
- ✅ 无需修改代码即可更新评分规则
- ✅ 版本化管理评分标准变更
