from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import LoadedQuestionPack, QuestionDefinition
from .validator import QuestionBankValidationError, QuestionBankValidator


class QuestionBankRepository:
    """Filesystem-backed repository for question packs."""

    def __init__(self, *, base_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self._base_dir = base_dir or (root / "config" / "question_bank")
        self._schema_path = self._base_dir / "schema" / "question_bank.schema.json"
        self._validator = QuestionBankValidator(schema_path=self._schema_path)

    def load_pack(
        self,
        pack_id: str,
        *,
        overlay_name: str | None = None,
    ) -> LoadedQuestionPack:
        pack_dir = self._base_dir / "packs" / pack_id
        if not pack_dir.exists():
            raise QuestionBankValidationError(f"question pack not found: {pack_id}")

        manifest_path = pack_dir / "manifest.yaml"
        manifest = self._load_yaml(manifest_path)
        errors = self._validator.validate_manifest(manifest)

        dimension_files = (
            list(manifest.get("dimension_files", []))
            if isinstance(manifest, dict)
            else []
        )
        scoring_dimensions = self._load_scoring_dimensions()
        questions: list[QuestionDefinition] = []
        question_ids: list[str] = []

        for filename in dimension_files:
            if not isinstance(filename, str):
                errors.append(
                    f"manifest.dimension_files entry must be string: {filename!r}"
                )
                continue
            dimension_name = Path(filename).stem
            path = pack_dir / "dimensions" / filename
            if not path.exists():
                errors.append(f"missing dimension file: {filename}")
                continue
            payload = self._load_yaml(path)
            errors.extend(
                self._validator.validate_dimension_payload(
                    payload=payload,
                    expected_dimension=dimension_name,
                    scoring_dimensions=scoring_dimensions,
                )
            )
            if not isinstance(payload, dict):
                continue
            for item in payload.get("questions", []):
                if not isinstance(item, dict):
                    continue
                question_ids.append(str(item.get("id", "")))
                if item.get("status") != "active":
                    continue
                questions.append(_to_question_definition(item))

        errors.extend(self._validator.validate_unique_ids(question_ids=question_ids))

        selected_overlay = self._resolve_overlay_name(
            manifest=manifest,
            requested=overlay_name,
        )
        overlay_policy = self._load_overlay_policy(
            pack_dir=pack_dir,
            overlay_name=selected_overlay,
            errors=errors,
        )

        if errors:
            raise QuestionBankValidationError("; ".join(errors))

        return LoadedQuestionPack(
            pack_id=pack_id,
            version=str(manifest.get("version", "")),
            description=str(manifest.get("description", "")),
            selected_overlay=selected_overlay,
            overlay_policy=overlay_policy,
            questions=questions,
        )

    def _resolve_overlay_name(
        self, *, manifest: Any, requested: str | None
    ) -> str | None:
        if not isinstance(manifest, dict):
            return None
        overlays = manifest.get("overlays")
        overlays = overlays if isinstance(overlays, list) else []
        default_overlay = manifest.get("default_overlay")
        if requested:
            normalized = _normalize_overlay_filename(requested)
            if normalized not in overlays:
                raise ValueError(f"overlay '{requested}' not declared in manifest")
            return Path(normalized).stem
        if isinstance(default_overlay, str):
            return Path(default_overlay).stem
        return None

    def _load_overlay_policy(
        self,
        *,
        pack_dir: Path,
        overlay_name: str | None,
        errors: list[str],
    ) -> dict[str, Any]:
        if overlay_name is None:
            return {}
        path = pack_dir / "overlays" / f"{overlay_name}.yaml"
        if not path.exists():
            errors.append(f"overlay file missing: {overlay_name}.yaml")
            return {}
        payload = self._load_yaml(path)
        errors.extend(self._validator.validate_overlay(payload))
        if not isinstance(payload, dict):
            return {}
        policy = payload.get("policy")
        return policy if isinstance(policy, dict) else {}

    def _load_scoring_dimensions(self) -> set[str]:
        root = Path(__file__).resolve().parents[2]
        scoring_rules_path = root / "config" / "scoring_rules.yaml"
        payload = self._load_yaml(scoring_rules_path)
        dimensions = payload.get("dimensions") if isinstance(payload, dict) else None
        if not isinstance(dimensions, dict):
            return set()
        return {str(name) for name in dimensions.keys()}

    @staticmethod
    def _load_yaml(path: Path) -> Any:
        if not path.exists():
            raise QuestionBankValidationError(f"file not found: {path}")
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise QuestionBankValidationError(f"invalid yaml: {path}") from exc
        return payload if payload is not None else {}


def _normalize_overlay_filename(value: str) -> str:
    return value if value.endswith(".yaml") else f"{value}.yaml"


def _to_question_definition(payload: dict[str, Any]) -> QuestionDefinition:
    return QuestionDefinition(
        id=str(payload.get("id", "")),
        dimension=str(payload.get("dimension", "")),
        question=str(payload.get("question", "")),
        intent=str(payload.get("intent", "")),
        keywords=[
            str(item) for item in payload.get("keywords", []) if isinstance(item, str)
        ],
        expected_answer_type=str(payload.get("expected_answer_type", "")),
        scoring_rule=dict(payload.get("scoring_rule", {}))
        if isinstance(payload.get("scoring_rule"), dict)
        else {},
        evidence_requirements=dict(payload.get("evidence_requirements", {}))
        if isinstance(payload.get("evidence_requirements"), dict)
        else {},
        warning_policy=dict(payload.get("warning_policy", {}))
        if isinstance(payload.get("warning_policy"), dict)
        else {},
        status=str(payload.get("status", "")),
    )
