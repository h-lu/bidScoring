from __future__ import annotations

from pathlib import Path

from bid_scoring.pipeline.application.e2e_service import (
    E2EPipelineService,
    E2ERunRequest,
    LoadedContent,
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

    def score(self, request):
        _ = request
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
            dimensions={},
        )


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
    assert result.warnings == ["mineru_bypassed", "missing_bbox"]


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
