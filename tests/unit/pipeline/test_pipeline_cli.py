from __future__ import annotations

import json
from dataclasses import dataclass
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
            "traceability": {
                "status": "verified",
                "highlight_ready_chunk_ids": ["chunk-1"],
            },
            "observability": {"timings_ms": {"total": 12}},
        }


@dataclass
class _Resolved:
    dimensions: list[str] | None
    question_context: object | None


class _FakeResolver:
    def __init__(self, resolved: _Resolved):
        self._resolved = resolved
        self.calls = []

    def resolve(
        self,
        *,
        question_pack: str | None,
        question_overlay: str | None,
        requested_dimensions: list[str] | None,
    ):
        self.calls.append(
            {
                "question_pack": question_pack,
                "question_overlay": question_overlay,
                "requested_dimensions": requested_dimensions,
            }
        )
        return self._resolved


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
    assert output["traceability"]["status"] == "verified"
    assert output["observability"]["timings_ms"]["total"] == 12


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
    assert service.run_calls[0].scoring_backend == "hybrid"
    assert service.run_calls[0].question_context is not None
    assert service.run_calls[0].question_context.pack_id == "cn_medical_v1"
    assert service.run_calls[0].question_context.overlay == "strict_traceability"


def test_cli_run_e2e_supports_context_json_alias(tmp_path: Path, fixed_ids):
    content_path = tmp_path / "context_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "hello"}]), "utf-8")

    service = _FakeService()
    code = cli.main(
        [
            "run-e2e",
            "--context-json",
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
    assert service.run_calls[0].content_list_path == content_path


def test_cli_run_prod_uses_context_json_with_production_defaults(
    tmp_path: Path, fixed_ids
):
    content_path = tmp_path / "context_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "hello"}]), "utf-8")

    service = _FakeService()
    code = cli.main(
        [
            "run-prod",
            "--context-json",
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
    request = service.run_calls[0]
    assert request.content_list_path == content_path
    assert request.scoring_backend == "hybrid"
    assert request.build_embeddings is True
    assert request.question_context is not None
    assert request.question_context.pack_id == "cn_medical_v1"
    assert request.question_context.overlay == "strict_traceability"


def test_cli_run_prod_supports_pdf_path(tmp_path: Path, fixed_ids):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    service = _FakeService()
    code = cli.main(
        [
            "run-prod",
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
    request = service.run_calls[0]
    assert request.pdf_path == pdf_path
    assert request.mineru_parser == "api"
    assert request.scoring_backend == "hybrid"


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


def test_cli_run_e2e_accepts_question_pack_options(tmp_path: Path, fixed_ids):
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
            "--question-pack",
            "cn_medical_v1",
            "--question-overlay",
            "strict_traceability",
        ],
        service=service,
    )

    assert code == 0
    assert service.run_calls[0].question_context is not None
    assert service.run_calls[0].question_context.pack_id == "cn_medical_v1"
    assert service.run_calls[0].question_context.overlay == "strict_traceability"
    assert service.run_calls[0].question_context.question_count == 12
    assert service.run_calls[0].dimensions == [
        "warranty",
        "delivery",
        "training",
        "financial",
        "technical",
        "compliance",
    ]
    assert "warranty" in service.run_calls[0].question_context.keywords_by_dimension
    assert (
        "质保"
        in service.run_calls[0].question_context.keywords_by_dimension["warranty"]
    )


def test_cli_run_e2e_question_pack_respects_dimensions_subset(
    tmp_path: Path, fixed_ids
):
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
            "--question-pack",
            "cn_medical_v1",
            "--question-overlay",
            "strict_traceability",
            "--dimensions",
            "warranty",
            "delivery",
        ],
        service=service,
    )

    assert code == 0
    assert service.run_calls[0].dimensions == ["warranty", "delivery"]
    assert service.run_calls[0].question_context is not None
    assert service.run_calls[0].question_context.dimensions == ["warranty", "delivery"]
    assert set(service.run_calls[0].question_context.keywords_by_dimension.keys()) == {
        "warranty",
        "delivery",
    }


def test_cli_run_e2e_supports_injected_question_context_resolver(
    tmp_path: Path, fixed_ids
):
    content_path = tmp_path / "content_list.json"
    content_path.write_text(json.dumps([{"type": "text", "text": "hello"}]), "utf-8")

    fake_context = object()
    resolver = _FakeResolver(
        _Resolved(
            dimensions=["warranty"],
            question_context=fake_context,
        )
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
            "--question-pack",
            "cn_medical_v1",
        ],
        service=service,
        question_context_resolver=resolver,
    )

    assert code == 0
    assert len(resolver.calls) == 1
    assert resolver.calls[0]["question_pack"] == "cn_medical_v1"
    assert service.run_calls[0].dimensions == ["warranty"]
    assert service.run_calls[0].question_context is fake_context
