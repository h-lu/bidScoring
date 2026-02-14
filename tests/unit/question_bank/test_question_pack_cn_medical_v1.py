from __future__ import annotations

import re
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_question_bank_pack_layout_and_manifest_contract():
    root = _repo_root()
    pack_dir = root / "config" / "question_bank" / "packs" / "cn_medical_v1"

    assert (
        root / "config" / "question_bank" / "schema" / "question_bank.schema.json"
    ).exists()
    assert pack_dir.exists()

    manifest_path = pack_dir / "manifest.yaml"
    assert manifest_path.exists()

    manifest = _load_yaml(manifest_path)
    assert isinstance(manifest, dict)
    assert manifest.get("pack_id") == "cn_medical_v1"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(manifest.get("version", "")))

    dimension_files = manifest.get("dimension_files")
    assert isinstance(dimension_files, list)
    assert set(dimension_files) == {
        "warranty.yaml",
        "delivery.yaml",
        "training.yaml",
        "financial.yaml",
        "technical.yaml",
        "compliance.yaml",
    }

    overlay_files = manifest.get("overlays")
    assert isinstance(overlay_files, list)
    assert set(overlay_files) == {
        "strict_traceability.yaml",
        "fast_eval.yaml",
    }


def test_question_bank_contains_12_active_questions_with_traceability_requirements():
    root = _repo_root()
    pack_dir = root / "config" / "question_bank" / "packs" / "cn_medical_v1"
    scoring_rules = _load_yaml(root / "config" / "scoring_rules.yaml")
    scoring_dimensions = set((scoring_rules or {}).get("dimensions", {}).keys())

    manifest = _load_yaml(pack_dir / "manifest.yaml")
    dimensions_dir = pack_dir / "dimensions"

    all_ids: set[str] = set()
    active_questions = []

    per_dimension_counts: dict[str, int] = {}
    for filename in manifest["dimension_files"]:
        path = dimensions_dir / filename
        assert path.exists(), f"missing dimension file: {filename}"

        payload = _load_yaml(path)
        assert isinstance(payload, dict)
        dimension = payload.get("dimension")
        assert isinstance(dimension, str)
        assert dimension in scoring_dimensions

        questions = payload.get("questions")
        assert isinstance(questions, list)
        assert len(questions) >= 2

        count_active = 0
        for question in questions:
            assert isinstance(question, dict)
            qid = question.get("id")
            assert isinstance(qid, str) and qid
            assert qid not in all_ids, f"duplicate question id: {qid}"
            all_ids.add(qid)

            assert question.get("dimension") == dimension
            assert (
                isinstance(question.get("question"), str)
                and question["question"].strip()
            )
            assert (
                isinstance(question.get("intent"), str) and question["intent"].strip()
            )

            keywords = question.get("keywords")
            assert isinstance(keywords, list) and keywords
            assert all(isinstance(item, str) and item.strip() for item in keywords)

            evidence = question.get("evidence_requirements")
            assert isinstance(evidence, dict)
            assert int(evidence.get("min_citations", 0)) >= 1
            assert evidence.get("require_page_idx") is True
            assert evidence.get("require_bbox") is True

            warning_policy = question.get("warning_policy")
            assert isinstance(warning_policy, dict)
            assert warning_policy.get("on_missing_evidence") == "warn"
            assert warning_policy.get("on_partial_untraceable") == "warn"

            status = question.get("status")
            assert status in {"active", "deprecated"}
            if status == "active":
                count_active += 1
                active_questions.append(question)

        per_dimension_counts[dimension] = count_active

    assert len(active_questions) == 12
    assert per_dimension_counts == {
        "warranty": 2,
        "delivery": 2,
        "training": 2,
        "financial": 2,
        "technical": 2,
        "compliance": 2,
    }
