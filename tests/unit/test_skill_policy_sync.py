from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_skill_policy_sync.py"
)
_SPEC = importlib.util.spec_from_file_location("check_skill_policy_sync", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

check_sync = _MODULE.check_sync
_load_yaml = _MODULE._load_yaml


def test_skill_prompt_and_policy_are_in_sync():
    root = Path(__file__).resolve().parents[2]
    policy = _load_yaml(root / "config" / "agent_scoring_policy.yaml")
    prompt = (root / ".claude" / "skills" / "bid-analyze" / "prompt.md").read_text(
        encoding="utf-8"
    )

    violations = check_sync(policy=policy, prompt=prompt)

    assert violations == []


def test_sync_check_detects_missing_required_tool():
    policy = {
        "workflow": {
            "tool_calling_required": True,
            "required_tools": ["retrieve_dimension_evidence"],
        },
        "scoring": {"baseline_score": 50},
        "risk_rules": {
            "high": "高",
            "medium": "中",
            "low": "低",
        },
        "output": {"schema_hint": "{overall_score,dimensions}"},
    }
    prompt = "先调用 MCP 工具，基线 `50`，风险 high=高 medium=中 low=低"

    violations = check_sync(policy=policy, prompt=prompt)

    assert "missing_required_tool:retrieve_dimension_evidence" in violations
    assert "missing_output_schema_hint" in violations
