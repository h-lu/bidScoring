from __future__ import annotations

from bid_scoring.pipeline.application.service import PipelineService


class _Repo:
    def __init__(self):
        self.persist_calls = []
        self.artifact_calls = []

    def persist_content_list(self, **kwargs):  # pragma: no cover
        self.persist_calls.append(kwargs)
        return {"total_chunks": 7}

    def record_source_artifact(self, **kwargs):  # pragma: no cover
        self.artifact_calls.append(kwargs)


def test_ingest_content_list_records_artifact_and_returns_summary(fixed_ids):
    repo = _Repo()
    service = PipelineService(repository=repo)

    summary = service.ingest_content_list(
        project_id=fixed_ids["project_id"],
        document_id=fixed_ids["document_id"],
        version_id=fixed_ids["version_id"],
        content_list=[{"type": "text", "text": "x"}],
        document_title="demo",
        source_uri="minio://demo/content_list.json",
        parser_version="pipeline-v1",
    )

    assert summary.status == "completed"
    assert summary.chunks_imported == 7
    assert len(repo.persist_calls) == 1
    assert len(repo.artifact_calls) == 1
