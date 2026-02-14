from __future__ import annotations

import json
from pathlib import Path

from bid_scoring.pipeline.interfaces import cli


class _FakeService:
    def __init__(self):
        self.calls = []
        self.run_calls = []

    def ingest_content_list(self, **kwargs):
        self.calls.append(kwargs)
        return {"status": "completed", "chunks_imported": 2}

    def run(self, request):
        self.run_calls.append(request)
        return {
            "status": "completed",
            "warnings": ["mineru_bypassed"],
            "ingest": {"chunks_imported": 1},
            "scoring": {"overall_score": 75.0, "risk_level": "low"},
        }


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


def test_cli_run_e2e_supports_context_list(tmp_path: Path, fixed_ids, capsys):
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
            "run-e2e",
            "--context-list",
            str(content_path),
            "--project-id",
            fixed_ids["project_id"],
            "--document-id",
            fixed_ids["document_id"],
            "--version-id",
            fixed_ids["version_id"],
            "--document-title",
            "demo",
            "--bidder-name",
            "A公司",
            "--project-name",
            "示例项目",
        ],
        service=service,
    )

    assert code == 0
    assert len(service.run_calls) == 1
    request = service.run_calls[0]
    assert request.content_list_path == content_path
    assert request.project_id == fixed_ids["project_id"]
    assert request.bidder_name == "A公司"

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["warnings"] == ["mineru_bypassed"]


def test_cli_run_e2e_supports_content_list_alias(tmp_path: Path, fixed_ids):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "hello"}]), "utf-8")

    service = _FakeService()
    code = cli.main(
        [
            "run-e2e",
            "--content-list",
            str(content_path),
            "--project-id",
            fixed_ids["project_id"],
            "--document-id",
            fixed_ids["document_id"],
            "--version-id",
            fixed_ids["version_id"],
        ],
        service=service,
    )

    assert code == 0
    assert len(service.run_calls) == 1
    assert service.run_calls[0].content_list_path == content_path


def test_cli_run_e2e_accepts_scoring_backend_option(tmp_path: Path, fixed_ids):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "hello"}]), "utf-8")

    service = _FakeService()
    code = cli.main(
        [
            "run-e2e",
            "--context-list",
            str(content_path),
            "--project-id",
            fixed_ids["project_id"],
            "--document-id",
            fixed_ids["document_id"],
            "--version-id",
            fixed_ids["version_id"],
            "--scoring-backend",
            "agent-mcp",
        ],
        service=service,
    )

    assert code == 0
    assert service.run_calls[0].scoring_backend == "agent-mcp"


def test_cli_run_e2e_accepts_hybrid_weight_option(tmp_path: Path, fixed_ids):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "hello"}]), "utf-8")

    service = _FakeService()
    code = cli.main(
        [
            "run-e2e",
            "--context-list",
            str(content_path),
            "--project-id",
            fixed_ids["project_id"],
            "--document-id",
            fixed_ids["document_id"],
            "--version-id",
            fixed_ids["version_id"],
            "--scoring-backend",
            "hybrid",
            "--hybrid-primary-weight",
            "0.85",
        ],
        service=service,
    )

    assert code == 0
    assert service.run_calls[0].scoring_backend == "hybrid"
    assert service.run_calls[0].hybrid_primary_weight == 0.85


def test_cli_run_e2e_supports_pdf_path_with_service(tmp_path: Path, fixed_ids):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    service = _FakeService()
    code = cli.main(
        [
            "run-e2e",
            "--pdf-path",
            str(pdf_path),
            "--project-id",
            fixed_ids["project_id"],
            "--document-id",
            fixed_ids["document_id"],
            "--version-id",
            fixed_ids["version_id"],
            "--mineru-parser",
            "api",
        ],
        service=service,
    )

    assert code == 0
    assert len(service.run_calls) == 1
    assert service.run_calls[0].pdf_path == pdf_path
    assert service.run_calls[0].mineru_parser == "api"
