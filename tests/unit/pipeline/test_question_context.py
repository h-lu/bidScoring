from __future__ import annotations

from bid_scoring.pipeline.application.question_context import QuestionContextResolver


def test_question_context_resolver_returns_none_when_pack_missing():
    resolver = QuestionContextResolver()

    resolved = resolver.resolve(
        question_pack=None,
        question_overlay=None,
        requested_dimensions=["warranty"],
    )

    assert resolved.question_context is None
    assert resolved.dimensions == ["warranty"]


def test_question_context_resolver_loads_pack_defaults():
    resolver = QuestionContextResolver()

    resolved = resolver.resolve(
        question_pack="cn_medical_v1",
        question_overlay="strict_traceability",
        requested_dimensions=None,
    )

    assert resolved.question_context is not None
    assert resolved.question_context.pack_id == "cn_medical_v1"
    assert resolved.question_context.overlay == "strict_traceability"
    assert resolved.question_context.question_count == 12
    assert resolved.dimensions == [
        "warranty",
        "delivery",
        "training",
        "financial",
        "technical",
        "compliance",
    ]
    assert "质保" in resolved.question_context.keywords_by_dimension["warranty"]


def test_question_context_resolver_restricts_to_requested_dimensions():
    resolver = QuestionContextResolver()

    resolved = resolver.resolve(
        question_pack="cn_medical_v1",
        question_overlay="strict_traceability",
        requested_dimensions=["warranty", "delivery"],
    )

    assert resolved.dimensions == ["warranty", "delivery"]
    assert set(resolved.question_context.keywords_by_dimension.keys()) == {
        "warranty",
        "delivery",
    }
