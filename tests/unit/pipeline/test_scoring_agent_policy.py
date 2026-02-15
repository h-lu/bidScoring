from __future__ import annotations

from pathlib import Path

from bid_scoring.pipeline.application.scoring_agent_policy import (
    load_agent_scoring_policy,
)


def test_load_agent_scoring_policy_from_yaml(tmp_path: Path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(
        """
constraints:
  - 必须基于证据
risk_rules:
  high: 高风险定义
  medium: 中风险定义
  low: 低风险定义
output:
  schema_hint: "{overall_score,dimensions}"
workflow:
  tool_calling_required: true
""".strip(),
        encoding="utf-8",
    )

    policy = load_agent_scoring_policy(policy_file)

    assert policy.constraints == ["必须基于证据"]
    assert policy.risk_rules["high"] == "高风险定义"
    assert policy.output_schema_hint == "{overall_score,dimensions}"
    assert policy.tool_calling_required is True
    assert "retrieve_dimension_evidence" in policy.tool_calling_system_prompt()


def test_load_agent_scoring_policy_falls_back_to_default(tmp_path: Path):
    missing = tmp_path / "missing.yaml"

    policy = load_agent_scoring_policy(missing)

    assert policy.tool_calling_required is True
    assert "risk_level" in policy.bulk_system_prompt()
