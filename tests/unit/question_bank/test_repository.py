from __future__ import annotations

from pathlib import Path

import pytest

from bid_scoring.question_bank.repository import QuestionBankRepository


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_repository_load_pack_with_overlay():
    repository = QuestionBankRepository(base_dir=_repo_root() / "config" / "question_bank")

    pack = repository.load_pack("cn_medical_v1", overlay_name="strict_traceability")

    assert pack.pack_id == "cn_medical_v1"
    assert pack.version == "1.0.0"
    assert pack.selected_overlay == "strict_traceability"
    assert len(pack.questions) == 12
    assert set(pack.questions_by_dimension.keys()) == {
        "warranty",
        "delivery",
        "training",
        "financial",
        "technical",
        "compliance",
    }
    assert "warranty" in pack.keywords_by_dimension
    assert "质保" in pack.keywords_by_dimension["warranty"]


def test_repository_rejects_unknown_overlay():
    repository = QuestionBankRepository(base_dir=_repo_root() / "config" / "question_bank")

    with pytest.raises(ValueError, match="overlay"):
        repository.load_pack("cn_medical_v1", overlay_name="missing_overlay")
