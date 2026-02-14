from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuestionDefinition:
    id: str
    dimension: str
    question: str
    intent: str
    keywords: list[str]
    expected_answer_type: str
    scoring_rule: dict[str, Any]
    evidence_requirements: dict[str, Any]
    warning_policy: dict[str, Any]
    status: str


@dataclass(frozen=True)
class LoadedQuestionPack:
    pack_id: str
    version: str
    description: str
    selected_overlay: str | None
    overlay_policy: dict[str, Any]
    questions: list[QuestionDefinition]

    @property
    def questions_by_dimension(self) -> dict[str, list[QuestionDefinition]]:
        grouped: dict[str, list[QuestionDefinition]] = {}
        for item in self.questions:
            grouped.setdefault(item.dimension, []).append(item)
        return grouped

    @property
    def keywords_by_dimension(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        for item in self.questions:
            key = item.dimension
            if key not in grouped:
                grouped[key] = []
                seen[key] = set()
            for keyword in item.keywords:
                if keyword in seen[key]:
                    continue
                seen[key].add(keyword)
                grouped[key].append(keyword)
        return grouped
