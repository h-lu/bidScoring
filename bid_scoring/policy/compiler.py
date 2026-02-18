from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .loader import load_policy_bundle


@dataclass(frozen=True)
class CompiledPolicyArtifacts:
    policy_hash: str
    runtime_policy_path: Path
    agent_prompt_path: Path


def compile_policy_artifacts(
    *,
    pack_id: str | None = None,
    overlay_name: str | None = None,
    output_root: Path | None = None,
    packs_root: Path | None = None,
) -> CompiledPolicyArtifacts:
    bundle = load_policy_bundle(
        pack_id=pack_id,
        overlay_name=overlay_name,
        packs_root=packs_root,
    )

    runtime_payload = bundle.as_runtime_dict()
    runtime_payload["meta"]["policy_hash"] = None
    policy_hash = _compute_policy_hash(runtime_payload)
    runtime_payload["meta"]["policy_hash"] = policy_hash

    root = output_root or (Path(__file__).resolve().parents[2] / "artifacts")
    overlay_key = bundle.meta.overlay or "base"
    artifact_dir = root / "policy" / bundle.meta.pack_id / overlay_key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    runtime_policy_path = artifact_dir / "runtime_policy.json"
    runtime_policy_path.write_text(
        json.dumps(runtime_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    agent_prompt_path = artifact_dir / "agent_prompt.md"
    agent_prompt_path.write_text(
        render_agent_prompt(runtime_payload),
        encoding="utf-8",
    )

    return CompiledPolicyArtifacts(
        policy_hash=policy_hash,
        runtime_policy_path=runtime_policy_path,
        agent_prompt_path=agent_prompt_path,
    )


def render_agent_prompt(runtime_payload: dict) -> str:
    workflow = runtime_payload["workflow"]
    scoring = runtime_payload["scoring"]
    risk_rules = runtime_payload["risk_rules"]
    output = runtime_payload["output"]

    required_tools = ", ".join(workflow["required_tools"])
    constraints = "\n".join([f"- {item}" for item in runtime_payload["constraints"]])

    return (
        "# 提示词模板（Prompt）\n\n"
        "此文件由 policy 编译器自动生成，请勿手工改动。\n\n"
        f"策略来源：`config/policy/packs/{runtime_payload['meta']['pack_id']}`\n\n"
        "系统提示词核心：\n"
        "1. 使用 `bid-team-orchestrator`。\n"
        "2. 严格执行协作阶段：retrieval -> scoring -> traceability。\n"
        f"3. 必须先调用 MCP 工具（尤其是 {required_tools}），再输出评分。\n"
        "4. 仅输出严格 JSON。\n\n"
        "约束：\n"
        f"{constraints}\n\n"
        "Baseline:\n"
        f"- 默认基线分：`{int(scoring['baseline_score'])}`\n\n"
        "Risk rules:\n"
        f"- `high`: {risk_rules['high']}\n"
        f"- `medium`: {risk_rules['medium']}\n"
        f"- `low`: {risk_rules['low']}\n\n"
        "输出契约提示：\n"
        f"`{output['schema_hint']}`\n"
    )


def _compute_policy_hash(runtime_payload: dict) -> str:
    rendered = json.dumps(
        runtime_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
