from __future__ import annotations

import json
from pathlib import Path

from bid_scoring.pipeline.interfaces import cli


class _FakeService:
    def __init__(self):
        self.calls = []

    def ingest_content_list(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "completed", "chunks_imported": 2}


def test_cli_ingest_content_list_invokes_pipeline_service(tmp_path: Path, fixed_ids):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(
        json.dumps(
            [{"type": "text", "text": "hello", "page_idx": 0, "bbox": [0, 0, 1, 1]}]
        ),
        encoding="utf-8",
    )

    service = _FakeService()
    code = cli.main(
        [
            "ingest-content-list",
            "--content-list",
            str(content_path),
            "--project-id",
            fixed_ids["project_id"],
            "--document-id",
            fixed_ids["document_id"],
            "--version-id",
            fixed_ids["version_id"],
            "--document-title",
            "demo",
        ],
        service=service,
    )

    assert code == 0
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["project_id"] == fixed_ids["project_id"]
    assert call["document_id"] == fixed_ids["document_id"]
    assert call["version_id"] == fixed_ids["version_id"]
    assert len(call["content_list"]) == 1
