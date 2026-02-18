from __future__ import annotations

import json
from pathlib import Path

from bid_scoring.pipeline.application.scoring_agent_policy import (
    load_agent_scoring_policy,
)


def test_load_agent_scoring_policy_from_runtime_artifact(tmp_path: Path):
    policy_file = tmp_path / "runtime_policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "meta": {
                    "pack_id": "cn_medical_v1",
                    "overlay": "strict_traceability",
                    "version": "2026-02-18",
                    "policy_hash": "abc",
                },
                "constraints": ["必须基于证据"],
                "workflow": {
                    "tool_calling_required": True,
                    "max_turns_default": 8,
                    "required_tools": ["retrieve_dimension_evidence"],
                },
                "scoring": {
                    "baseline_score": 50,
                    "min_score": 0,
                    "max_score": 100,
                    "positive_evidence_delta_range": [5, 15],
                    "risk_evidence_delta_range": [-20, -5],
                },
                "risk_rules": {
                    "high": "高风险定义",
                    "medium": "中风险定义",
                    "low": "低风险定义",
                },
                "output": {
                    "strict_json": True,
                    "schema_hint": "{overall_score,dimensions}",
                },
                "retrieval": {
                    "default_mode": "hybrid",
                    "default_top_k": 8,
                    "dimension_overrides": {
                        "warranty": {"mode": "keyword", "top_k": 6}
                    },
                    "evaluation_thresholds": {
                        "hybrid": {"mrr": 0.58, "recall_at_5": 0.72}
                    },
                },
                "evidence_gate": {
                    "default_min_citations": 1,
                    "require_page_idx": True,
                    "require_bbox": True,
                    "require_quote": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    policy = load_agent_scoring_policy(policy_file)

    assert policy.constraints == ["必须基于证据"]
    assert policy.risk_rules["high"] == "高风险定义"
    assert policy.output_schema_hint == "{overall_score,dimensions}"
    assert policy.tool_calling_required is True
    assert policy.max_turns_default == 8
    assert policy.retrieval_default_mode == "hybrid"
    assert policy.retrieval_default_top_k == 8
    assert policy.retrieval_evaluation_thresholds["hybrid"]["mrr"] == 0.58
    assert policy.evidence_require_quote is True
    assert "retrieve_dimension_evidence" in policy.tool_calling_system_prompt()
    assert policy.resolve_dimension_defaults(
        "warranty",
        fallback_mode=policy.retrieval_default_mode,
        fallback_top_k=policy.retrieval_default_top_k,
    ) == ("keyword", 6)
    assert policy.resolve_dimension_defaults(
        "delivery",
        fallback_mode="hybrid",
        fallback_top_k=8,
    ) == ("hybrid", 8)


def test_load_agent_scoring_policy_falls_back_to_default_pack():
    policy = load_agent_scoring_policy()

    assert policy.tool_calling_required is True
    assert policy.retrieval_default_mode in {"hybrid", "keyword", "vector"}
    assert policy.retrieval_default_top_k >= 1
    assert "risk_level" in policy.bulk_system_prompt()
