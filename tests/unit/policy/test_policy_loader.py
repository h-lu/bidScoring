from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bid_scoring.policy.loader import (
    PolicyLoadError,
    load_policy_bundle,
    load_policy_bundle_from_artifact,
)


def test_load_policy_bundle_from_default_pack():
    bundle = load_policy_bundle()

    assert bundle.meta.pack_id == "cn_medical_v1"
    assert bundle.scoring.baseline_score == 50
    assert bundle.workflow.tool_calling_required is True
    assert "retrieve_dimension_evidence" in bundle.workflow.required_tools
    assert bundle.output.strict_json is True


def test_load_policy_bundle_with_overlay_merge(tmp_path: Path):
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "demo"
    overlay_dir = pack_dir / "overlays"
    overlay_dir.mkdir(parents=True)

    (pack_dir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "pack_id": "demo",
                "version": "1.0.0",
                "base_file": "base.yaml",
                "default_overlay": "strict.yaml",
                "overlays": ["strict.yaml", "fast.yaml"],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (pack_dir / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "constraints": ["仅基于证据"],
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
                "risk_rules": {"high": "高", "medium": "中", "low": "低"},
                "output": {
                    "strict_json": True,
                    "schema_hint": "{overall_score,dimensions}",
                },
                "retrieval": {
                    "default_mode": "hybrid",
                    "default_top_k": 8,
                    "dimension_overrides": {},
                    "evaluation_thresholds": {
                        "hybrid": {"mrr": 0.55, "recall_at_5": 0.7}
                    },
                },
                "evidence_gate": {
                    "default_min_citations": 1,
                    "require_page_idx": True,
                    "require_bbox": True,
                    "require_quote": True,
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (overlay_dir / "strict.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "strict",
                "policy": {
                    "workflow": {"max_turns_default": 12},
                    "retrieval": {"default_top_k": 12},
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (overlay_dir / "fast.yaml").write_text(
        yaml.safe_dump(
            {"name": "fast", "policy": {"workflow": {"max_turns_default": 4}}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    bundle = load_policy_bundle(
        pack_id="demo",
        overlay_name="strict",
        packs_root=packs_dir,
    )

    assert bundle.meta.overlay == "strict"
    assert bundle.workflow.max_turns_default == 12
    assert bundle.retrieval.default_top_k == 12
    assert bundle.retrieval.evaluation_thresholds["hybrid"]["mrr"] == 0.55
    assert bundle.scoring.baseline_score == 50


def test_policy_bundle_validates_required_fields(tmp_path: Path):
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "broken"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(
        "pack_id: broken\nversion: 1.0.0\nbase_file: base.yaml\noverlays: []\n",
        encoding="utf-8",
    )
    (pack_dir / "base.yaml").write_text("constraints: []\n", encoding="utf-8")

    with pytest.raises(PolicyLoadError):
        load_policy_bundle(pack_id="broken", packs_root=packs_dir)


def test_load_policy_bundle_from_artifact(tmp_path: Path):
    artifact_path = tmp_path / "runtime_policy.json"
    artifact_path.write_text(
        """
{
  "meta": {"pack_id": "cn_medical_v1", "overlay": "strict_traceability", "version": "2026-02-18", "policy_hash": "abc"},
  "constraints": ["必须仅基于证据"],
  "workflow": {"tool_calling_required": true, "max_turns_default": 8, "required_tools": ["retrieve_dimension_evidence"]},
  "scoring": {"baseline_score": 50, "min_score": 0, "max_score": 100, "positive_evidence_delta_range": [5, 15], "risk_evidence_delta_range": [-20, -5]},
  "risk_rules": {"high": "高", "medium": "中", "low": "低"},
  "output": {"strict_json": true, "schema_hint": "{overall_score,dimensions}"},
  "retrieval": {"default_mode": "hybrid", "default_top_k": 8, "dimension_overrides": {}, "evaluation_thresholds": {"hybrid": {"mrr": 0.56}}},
  "evidence_gate": {"default_min_citations": 1, "require_page_idx": true, "require_bbox": true, "require_quote": true}
}
""".strip(),
        encoding="utf-8",
    )

    bundle = load_policy_bundle_from_artifact(artifact_path)
    assert bundle.meta.pack_id == "cn_medical_v1"
    assert bundle.output.schema_hint == "{overall_score,dimensions}"
    assert bundle.retrieval.evaluation_thresholds["hybrid"]["mrr"] == 0.56


def test_load_policy_bundle_rejects_invalid_retrieval_thresholds(tmp_path: Path):
    packs_dir = tmp_path / "packs"
    pack_dir = packs_dir / "broken_thresholds"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.yaml").write_text(
        "pack_id: broken_thresholds\nversion: 1.0.0\nbase_file: base.yaml\noverlays: []\n",
        encoding="utf-8",
    )
    (pack_dir / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "constraints": ["仅基于证据"],
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
                "risk_rules": {"high": "高", "medium": "中", "low": "低"},
                "output": {
                    "strict_json": True,
                    "schema_hint": "{overall_score,dimensions}",
                },
                "retrieval": {
                    "default_mode": "hybrid",
                    "default_top_k": 8,
                    "dimension_overrides": {},
                    "evaluation_thresholds": {"hybrid": {"mrr": -0.1}},
                },
                "evidence_gate": {
                    "default_min_citations": 1,
                    "require_page_idx": True,
                    "require_bbox": True,
                    "require_quote": True,
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PolicyLoadError):
        load_policy_bundle(pack_id="broken_thresholds", packs_root=packs_dir)
