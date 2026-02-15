from __future__ import annotations

from pathlib import Path

from bid_scoring.pipeline.application.e2e_service import (
    E2EPipelineService,
    E2ERunRequest,
    LoadedContent,
)
from bid_scoring.pipeline.application.question_context import (
    QuestionContext,
    QuestionContextResolver,
)
from bid_scoring.pipeline.application.scoring_provider import ScoringResult


class _ContentSource:
    def __init__(self, events: list[str]):
        self._events = events

    def load(self, request: E2ERunRequest) -> LoadedContent:
        self._events.append("load")
        return LoadedContent(
            content_list=[{"type": "text", "text": "hello", "bbox": [0, 0, 1, 1]}],
            source_uri=str(request.content_list_path),
            parser_version="context-list-v1",
            warnings=["mineru_bypassed"],
        )


class _PipelineService:
    def __init__(self, events: list[str]):
        self._events = events

    def ingest_content_list(self, **kwargs):
        self._events.append("ingest")
        return {"status": "completed", "chunks_imported": 1}


class _IndexBuilder:
    def __init__(self, events: list[str]):
        self._events = events

    def build_embeddings(self, *, version_id, conn):
        self._events.append("embed")
        return {"status": "completed", "succeeded": 1, "failed": 0}


class _ScoringProvider:
    def __init__(self, events: list[str]):
        self._events = events
        self.last_request = None

    def score(self, request):
        self.last_request = request
        self._events.append("score")
        return ScoringResult(
            status="completed",
            overall_score=88.0,
            risk_level="low",
            total_risks=1,
            total_benefits=3,
            chunks_analyzed=6,
            recommendations=["ok"],
            evidence_warnings=["missing_bbox"],
            evidence_citations={
                "warranty": [
                    {
                        "chunk_id": "chunk-1",
                        "page_idx": 0,
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "chunk_id": "chunk-2",
                        "page_idx": 1,
                        "bbox": None,
                    },
                ]
            },
            dimensions={},
        )


class _ScoringProviderWithAgentMeta(_ScoringProvider):
    def score(self, request):
        base = super().score(request)
        return ScoringResult(
            status=base.status,
            overall_score=base.overall_score,
            risk_level=base.risk_level,
            total_risks=base.total_risks,
            total_benefits=base.total_benefits,
            chunks_analyzed=base.chunks_analyzed,
            recommendations=list(base.recommendations),
            evidence_warnings=list(base.evidence_warnings),
            evidence_citations=dict(base.evidence_citations),
            dimensions=dict(base.dimensions),
            warnings=list(base.warnings),
            backend_observability={
                "execution_mode": "tool-calling",
                "turns": 2,
                "tool_call_count": 3,
                "tool_names": ["retrieve_dimension_evidence"],
                "trace_id": "agent-mcp-test-trace",
            },
        )


class _Resolver:
    def __init__(self):
        self.calls = []

    def resolve(self, *, question_pack, question_overlay, requested_dimensions):
        self.calls.append(
            {
                "question_pack": question_pack,
                "question_overlay": question_overlay,
                "requested_dimensions": requested_dimensions,
            }
        )
        return type(
            "Resolved",
            (),
            {
                "dimensions": ["warranty"],
                "question_context": QuestionContext(
                    pack_id="cn_medical_v1",
                    overlay="strict_traceability",
                    question_count=12,
                    dimensions=["warranty"],
                    keywords_by_dimension={"warranty": ["质保", "保修"]},
                ),
            },
        )()


def _request(tmp_path: Path, fixed_ids: dict[str, str]) -> E2ERunRequest:
    return E2ERunRequest(
        project_id=fixed_ids["project_id"],
        document_id=fixed_ids["document_id"],
        version_id=fixed_ids["version_id"],
        content_list_path=tmp_path / "content_list.json",
        bidder_name="A公司",
        project_name="示例项目",
    )


def test_e2e_service_runs_all_stages_in_order(tmp_path: Path, fixed_ids):
    events: list[str] = []
    service = E2EPipelineService(
        content_source=_ContentSource(events),
        pipeline_service=_PipelineService(events),
        index_builder=_IndexBuilder(events),
        scoring_provider=_ScoringProvider(events),
    )

    result = service.run(_request(tmp_path, fixed_ids), conn=object())

    assert events == ["load", "ingest", "embed", "score"]
    assert result.status == "completed"
    assert result.ingest["chunks_imported"] == 1
    assert result.embeddings["succeeded"] == 1
    assert result.scoring["overall_score"] == 88.0
    assert "mineru_bypassed" in result.warnings
    assert "missing_bbox" in result.warnings
    assert "citation_missing_bbox" in result.warnings
    assert "partial_untraceable_citations" in result.warnings
    assert result.traceability["status"] == "verified_with_warnings"
    assert result.traceability["citation_count_total"] == 2
    assert result.traceability["citation_count_traceable"] == 1
    assert result.traceability["highlight_ready_chunk_ids"] == ["chunk-1"]
    assert result.observability["timings_ms"]["total"] >= 0


def test_e2e_service_can_skip_embeddings(tmp_path: Path, fixed_ids):
    events: list[str] = []
    service = E2EPipelineService(
        content_source=_ContentSource(events),
        pipeline_service=_PipelineService(events),
        index_builder=_IndexBuilder(events),
        scoring_provider=_ScoringProvider(events),
    )

    request = _request(tmp_path, fixed_ids)
    request.build_embeddings = False
    result = service.run(request, conn=object())

    assert events == ["load", "ingest", "score"]
    assert result.embeddings["status"] == "skipped"
    assert "traceability" in result.as_dict()
    assert "observability" in result.as_dict()


def test_e2e_service_emits_question_bank_observability(tmp_path: Path, fixed_ids):
    events: list[str] = []
    scoring_provider = _ScoringProvider(events)
    service = E2EPipelineService(
        content_source=_ContentSource(events),
        pipeline_service=_PipelineService(events),
        index_builder=_IndexBuilder(events),
        scoring_provider=scoring_provider,
    )

    request = _request(tmp_path, fixed_ids)
    request.question_context = QuestionContext(
        pack_id="cn_medical_v1",
        overlay="strict_traceability",
        question_count=12,
        dimensions=["warranty"],
        keywords_by_dimension={"warranty": ["质保", "保修"]},
    )

    result = service.run(request, conn=object())

    assert result.observability["question_bank"]["pack_id"] == "cn_medical_v1"
    assert result.observability["question_bank"]["overlay"] == "strict_traceability"
    assert result.observability["question_bank"]["question_count"] == 12
    assert scoring_provider.last_request.question_context is not None
    assert scoring_provider.last_request.dimensions == ["warranty"]
    assert scoring_provider.last_request.question_context.keywords_by_dimension == {
        "warranty": ["质保", "保修"]
    }


def test_e2e_service_with_resolved_question_context(tmp_path: Path, fixed_ids):
    events: list[str] = []
    scoring_provider = _ScoringProvider(events)
    service = E2EPipelineService(
        content_source=_ContentSource(events),
        pipeline_service=_PipelineService(events),
        index_builder=_IndexBuilder(events),
        scoring_provider=scoring_provider,
    )

    resolved = QuestionContextResolver().resolve(
        question_pack="cn_medical_v1",
        question_overlay="strict_traceability",
        requested_dimensions=["warranty", "delivery"],
    )
    request = _request(tmp_path, fixed_ids)
    request.dimensions = resolved.dimensions
    request.question_context = resolved.question_context

    result = service.run(request, conn=object())

    assert scoring_provider.last_request.question_context is not None
    assert scoring_provider.last_request.question_context.pack_id == "cn_medical_v1"
    assert scoring_provider.last_request.question_context.dimensions == [
        "warranty",
        "delivery",
    ]
    assert result.observability["question_bank"]["pack_id"] == "cn_medical_v1"


def test_e2e_service_can_resolve_question_context_via_injected_resolver(
    tmp_path: Path, fixed_ids
):
    events: list[str] = []
    resolver = _Resolver()
    scoring_provider = _ScoringProvider(events)
    service = E2EPipelineService(
        content_source=_ContentSource(events),
        pipeline_service=_PipelineService(events),
        index_builder=_IndexBuilder(events),
        scoring_provider=scoring_provider,
        question_context_resolver=resolver,
    )

    request = _request(tmp_path, fixed_ids)
    request.question_pack = "cn_medical_v1"
    request.question_overlay = "strict_traceability"
    request.question_context = None
    request.dimensions = None

    result = service.run(request, conn=object())

    assert len(resolver.calls) == 1
    assert resolver.calls[0]["question_pack"] == "cn_medical_v1"
    assert scoring_provider.last_request.question_context is not None
    assert scoring_provider.last_request.question_context.pack_id == "cn_medical_v1"
    assert scoring_provider.last_request.dimensions == ["warranty"]
    assert result.observability["question_bank"]["pack_id"] == "cn_medical_v1"


def test_e2e_service_emits_agent_observability_when_available(
    tmp_path: Path, fixed_ids
):
    events: list[str] = []
    service = E2EPipelineService(
        content_source=_ContentSource(events),
        pipeline_service=_PipelineService(events),
        index_builder=_IndexBuilder(events),
        scoring_provider=_ScoringProviderWithAgentMeta(events),
    )

    result = service.run(_request(tmp_path, fixed_ids), conn=object())

    assert result.observability["agent"]["execution_mode"] == "tool-calling"
    assert result.observability["agent"]["turns"] == 2
    assert result.observability["agent"]["tool_call_count"] == 3
    assert result.observability["agent"]["trace_id"] == "agent-mcp-test-trace"
