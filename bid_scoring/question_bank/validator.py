from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class QuestionBankValidationError(ValueError):
    """Raised when question bank payload fails validation."""


class QuestionBankValidator:
    """Validate question-pack files with deterministic business constraints."""

    _REQUIRED_QUESTION_KEYS = {
        "id",
        "dimension",
        "question",
        "intent",
        "keywords",
        "expected_answer_type",
        "scoring_rule",
        "evidence_requirements",
        "warning_policy",
        "status",
    }
    _ALLOWED_ANSWER_TYPES = {"number", "duration", "boolean", "enum", "text"}
    _ALLOWED_STATUS = {"active", "deprecated"}

    def __init__(self, *, schema_path: Path) -> None:
        self._schema_path = schema_path
        self._schema = self._load_schema(schema_path)

    def validate_manifest(self, payload: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ["manifest must be an object"]

        if (
            not isinstance(payload.get("pack_id"), str)
            or not payload["pack_id"].strip()
        ):
            errors.append("manifest.pack_id is required")
        if (
            not isinstance(payload.get("version"), str)
            or not payload["version"].strip()
        ):
            errors.append("manifest.version is required")
        if (
            not isinstance(payload.get("dimension_files"), list)
            or not payload["dimension_files"]
        ):
            errors.append("manifest.dimension_files must be non-empty list")
        if not isinstance(payload.get("overlays"), list):
            errors.append("manifest.overlays must be list")
        return errors

    def validate_dimension_payload(
        self,
        *,
        payload: Any,
        expected_dimension: str | None = None,
        scoring_dimensions: set[str] | None = None,
    ) -> list[str]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ["dimension payload must be an object"]

        dimension = payload.get("dimension")
        if not isinstance(dimension, str) or not dimension.strip():
            errors.append("dimension is required")
        if expected_dimension is not None and dimension != expected_dimension:
            errors.append(
                f"dimension mismatch: expected '{expected_dimension}', got '{dimension}'"
            )
        if scoring_dimensions is not None and isinstance(dimension, str):
            if dimension not in scoring_dimensions:
                errors.append(f"unknown dimension in scoring rules: {dimension}")

        questions = payload.get("questions")
        if not isinstance(questions, list) or not questions:
            errors.append("questions must be non-empty list")
            return errors

        for idx, item in enumerate(questions):
            prefix = f"questions[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            missing = self._REQUIRED_QUESTION_KEYS - set(item.keys())
            if missing:
                errors.append(f"{prefix} missing keys: {sorted(missing)}")

            if item.get("dimension") != dimension:
                errors.append(f"{prefix}.dimension must equal '{dimension}'")
            if not isinstance(item.get("id"), str) or not item["id"].strip():
                errors.append(f"{prefix}.id is required")
            if item.get("expected_answer_type") not in self._ALLOWED_ANSWER_TYPES:
                errors.append(
                    f"{prefix}.expected_answer_type must be one of "
                    f"{sorted(self._ALLOWED_ANSWER_TYPES)}"
                )
            if item.get("status") not in self._ALLOWED_STATUS:
                errors.append(
                    f"{prefix}.status must be one of {sorted(self._ALLOWED_STATUS)}"
                )

            keywords = item.get("keywords")
            if not isinstance(keywords, list) or not keywords:
                errors.append(f"{prefix}.keywords must be non-empty list")

            evidence = item.get("evidence_requirements")
            if not isinstance(evidence, dict):
                errors.append(f"{prefix}.evidence_requirements must be object")
            else:
                if int(evidence.get("min_citations", 0)) < 1:
                    errors.append(
                        f"{prefix}.evidence_requirements.min_citations must be >= 1"
                    )
                if evidence.get("require_page_idx") is not True:
                    errors.append(
                        f"{prefix}.evidence_requirements.require_page_idx must be true"
                    )
                if evidence.get("require_bbox") is not True:
                    errors.append(
                        f"{prefix}.evidence_requirements.require_bbox must be true"
                    )

            warning_policy = item.get("warning_policy")
            if not isinstance(warning_policy, dict):
                errors.append(f"{prefix}.warning_policy must be object")
            else:
                if warning_policy.get("on_missing_evidence") != "warn":
                    errors.append(
                        f"{prefix}.warning_policy.on_missing_evidence must be 'warn'"
                    )
                if warning_policy.get("on_partial_untraceable") != "warn":
                    errors.append(
                        f"{prefix}.warning_policy.on_partial_untraceable must be 'warn'"
                    )
        return errors

    def validate_unique_ids(self, *, question_ids: list[str]) -> list[str]:
        seen: set[str] = set()
        errors: list[str] = []
        for qid in question_ids:
            if qid in seen:
                errors.append(f"duplicate question id: {qid}")
            seen.add(qid)
        return errors

    def validate_overlay(self, payload: Any) -> list[str]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ["overlay must be an object"]
        if not isinstance(payload.get("name"), str) or not payload["name"].strip():
            errors.append("overlay.name is required")
        if not isinstance(payload.get("policy"), dict):
            errors.append("overlay.policy must be an object")
        return errors

    def _load_schema(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise QuestionBankValidationError(f"schema file not found: {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise QuestionBankValidationError(f"invalid schema json: {path}") from exc
        if not isinstance(parsed, dict):
            raise QuestionBankValidationError("schema root must be object")
        return parsed
