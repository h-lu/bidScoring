from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bid_scoring.policy import (
    PolicyLoadError,
    load_policy_bundle,
    load_policy_bundle_from_artifact,
)

DEFAULT_SKILL_PROMPT = Path(".claude/skills/bid-analyze/prompt.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate sync between policy bundle/artifact and Claude skill prompt."
    )
    parser.add_argument("--policy-pack", default="cn_medical_v1")
    parser.add_argument("--policy-overlay", default="strict_traceability")
    parser.add_argument("--policy-packs-root", type=Path)
    parser.add_argument("--policy-artifact", type=Path)
    parser.add_argument("--skill-prompt", type=Path, default=DEFAULT_SKILL_PROMPT)
    parser.add_argument("--fail-on-violations", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy = _load_policy(
        policy_pack=args.policy_pack,
        policy_overlay=args.policy_overlay,
        policy_packs_root=args.policy_packs_root,
        policy_artifact=args.policy_artifact,
    )
    prompt = args.skill_prompt.read_text(encoding="utf-8")
    violations = check_sync(policy=policy, prompt=prompt)

    payload = {
        "policy_pack": args.policy_pack,
        "policy_overlay": args.policy_overlay,
        "policy_artifact": str(args.policy_artifact) if args.policy_artifact else None,
        "skill_prompt": str(args.skill_prompt),
        "violations": violations,
        "ok": len(violations) == 0,
    }
    print(json.dumps(payload, ensure_ascii=False))

    if args.fail_on_violations and violations:
        return 1
    return 0


def _load_policy(
    *,
    policy_pack: str,
    policy_overlay: str | None,
    policy_packs_root: Path | None,
    policy_artifact: Path | None,
) -> dict[str, Any]:
    try:
        if policy_artifact is not None:
            return load_policy_bundle_from_artifact(policy_artifact).as_runtime_dict()
        return load_policy_bundle(
            pack_id=policy_pack,
            overlay_name=policy_overlay,
            packs_root=policy_packs_root,
        ).as_runtime_dict()
    except PolicyLoadError as exc:
        raise SystemExit(f"Failed to load policy: {exc}") from exc


def check_sync(*, policy: dict[str, Any], prompt: str) -> list[str]:
    violations: list[str] = []

    workflow = policy.get("workflow") if isinstance(policy.get("workflow"), dict) else {}
    scoring = policy.get("scoring") if isinstance(policy.get("scoring"), dict) else {}
    risk_rules = policy.get("risk_rules") if isinstance(policy.get("risk_rules"), dict) else {}
    output = policy.get("output") if isinstance(policy.get("output"), dict) else {}
    meta = policy.get("meta") if isinstance(policy.get("meta"), dict) else {}

    pack_id = meta.get("pack_id")
    if isinstance(pack_id, str) and pack_id.strip():
        expected_reference = f"config/policy/packs/{pack_id.strip()}"
        if expected_reference not in prompt:
            violations.append("missing_policy_pack_reference")

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
        if schema_hint.strip() not in prompt:
            violations.append("missing_output_schema_hint")

    return sorted(set(violations))


if __name__ == "__main__":
    raise SystemExit(main())
