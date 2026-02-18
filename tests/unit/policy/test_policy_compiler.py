from __future__ import annotations

import json
from pathlib import Path

from bid_scoring.policy.compiler import (
    CompiledPolicyArtifacts,
    compile_policy_artifacts,
)


def test_compile_policy_artifacts_outputs_runtime_json_and_prompt(tmp_path: Path):
    artifacts = compile_policy_artifacts(
        pack_id="cn_medical_v1",
        overlay_name="strict_traceability",
        output_root=tmp_path,
    )

    assert isinstance(artifacts, CompiledPolicyArtifacts)
    assert artifacts.runtime_policy_path.exists()
    assert artifacts.agent_prompt_path.exists()
    assert artifacts.policy_hash

    runtime_payload = json.loads(
        artifacts.runtime_policy_path.read_text(encoding="utf-8")
    )
    assert runtime_payload["meta"]["pack_id"] == "cn_medical_v1"
    assert runtime_payload["meta"]["overlay"] == "strict_traceability"
    assert runtime_payload["meta"]["policy_hash"] == artifacts.policy_hash
    assert runtime_payload["workflow"]["tool_calling_required"] is True
    assert (
        "retrieve_dimension_evidence" in runtime_payload["workflow"]["required_tools"]
    )
    assert "evaluation_thresholds" in runtime_payload["retrieval"]

    prompt_text = artifacts.agent_prompt_path.read_text(encoding="utf-8")
    assert "仅输出严格 JSON" in prompt_text
    assert "retrieve_dimension_evidence" in prompt_text
    assert runtime_payload["output"]["schema_hint"] in prompt_text


def test_compile_policy_artifacts_is_deterministic(tmp_path: Path):
    first = compile_policy_artifacts(
        pack_id="cn_medical_v1",
        overlay_name="strict_traceability",
        output_root=tmp_path,
    )
    second = compile_policy_artifacts(
        pack_id="cn_medical_v1",
        overlay_name="strict_traceability",
        output_root=tmp_path,
    )

    assert first.policy_hash == second.policy_hash
