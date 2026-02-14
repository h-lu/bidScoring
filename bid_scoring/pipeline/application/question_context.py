from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bid_scoring.question_bank.models import LoadedQuestionPack
from bid_scoring.question_bank.repository import QuestionBankRepository


@dataclass(frozen=True)
class QuestionContext:
    pack_id: str
    overlay: str | None
    question_count: int
    dimensions: list[str]
    keywords_by_dimension: dict[str, list[str]]


@dataclass(frozen=True)
class ResolvedQuestionContext:
    dimensions: list[str] | None
    question_context: QuestionContext | None


class QuestionPackLoader(Protocol):
    def load_pack(
        self,
        pack_id: str,
        *,
        overlay_name: str | None = None,
    ) -> LoadedQuestionPack: ...


class QuestionContextResolver:
    """Resolve question-pack inputs into strongly-typed context for pipeline."""

    def __init__(self, loader: QuestionPackLoader | None = None) -> None:
        self._loader = loader or QuestionBankRepository()

    def resolve(
        self,
        *,
        question_pack: str | None,
        question_overlay: str | None,
        requested_dimensions: list[str] | None,
    ) -> ResolvedQuestionContext:
        if not question_pack:
            return ResolvedQuestionContext(
                dimensions=list(requested_dimensions) if requested_dimensions else None,
                question_context=None,
            )

        loaded = self._loader.load_pack(
            question_pack,
            overlay_name=question_overlay,
        )
        available = loaded.keywords_by_dimension

        resolved_dimensions: list[str]
        if requested_dimensions is None:
            resolved_dimensions = list(available.keys())
        else:
            resolved_dimensions = list(requested_dimensions)
            missing = [name for name in resolved_dimensions if name not in available]
            if missing:
                raise ValueError(
                    "Selected dimensions not found in question pack: "
                    + ",".join(missing)
                )

        context = QuestionContext(
            pack_id=loaded.pack_id,
            overlay=loaded.selected_overlay,
            question_count=len(loaded.questions),
            dimensions=resolved_dimensions,
            keywords_by_dimension={
                name: list(available.get(name, [])) for name in resolved_dimensions
            },
        )
        return ResolvedQuestionContext(
            dimensions=resolved_dimensions,
            question_context=context,
        )
