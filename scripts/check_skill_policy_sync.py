from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY = Path("config/agent_scoring_policy.yaml")
DEFAULT_SKILL_PROMPT = Path(".claude/skills/bid-analyze/prompt.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate sync between agent scoring policy and Claude skill prompt."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--skill-prompt", type=Path, default=DEFAULT_SKILL_PROMPT)
    parser.add_argument("--fail-on-violations", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = _load_yaml(args.policy)
    prompt = args.skill_prompt.read_text(encoding="utf-8")
    violations = check_sync(policy=policy, prompt=prompt)

    payload = {
        "policy": str(args.policy),
        "skill_prompt": str(args.skill_prompt),
        "violations": violations,
        "ok": len(violations) == 0,
    }
    print(json.dumps(payload, ensure_ascii=False))

    if args.fail_on_violations and violations:
        return 1
    return 0


def check_sync(*, policy: dict[str, Any], prompt: str) -> list[str]:
    violations: list[str] = []

    workflow = policy.get("workflow") if isinstance(policy.get("workflow"), dict) else {}
    scoring = policy.get("scoring") if isinstance(policy.get("scoring"), dict) else {}
    risk_rules = policy.get("risk_rules") if isinstance(policy.get("risk_rules"), dict) else {}
    output = policy.get("output") if isinstance(policy.get("output"), dict) else {}

    if "config/agent_scoring_policy.yaml" not in prompt:
        violations.append("missing_policy_file_reference")

    if workflow.get("tool_calling_required") is True:
        required_hint = "先调用 MCP 工具"
        if required_hint not in prompt:
            violations.append("missing_tool_calling_requirement")

    required_tools = workflow.get("required_tools") if isinstance(workflow.get("required_tools"), list) else []
    for tool_name in required_tools:
        if isinstance(tool_name, str) and tool_name and tool_name not in prompt:
            violations.append(f"missing_required_tool:{tool_name}")

    baseline_score = scoring.get("baseline_score")
    if isinstance(baseline_score, (int, float)):
        token = f"`{int(baseline_score)}`"
        if token not in prompt:
            violations.append("missing_baseline_score")

    for key in ("high", "medium", "low"):
        rule = risk_rules.get(key)
        if isinstance(rule, str) and rule.strip() and rule.strip() not in prompt:
            violations.append(f"missing_risk_rule:{key}")

    schema_hint = output.get("schema_hint")
    if isinstance(schema_hint, str) and schema_hint.strip():
        # Require top-level schema hint in prompt so users see canonical contract.
        if schema_hint.strip() not in prompt:
            violations.append("missing_output_schema_hint")

    return sorted(set(violations))


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
