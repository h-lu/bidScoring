from __future__ import annotations

import json
from pathlib import Path

from bid_scoring.pipeline.application.e2e_service import E2ERunRequest
from bid_scoring.pipeline.infrastructure.content_source import PdfMinerUAdapter


def test_pdf_mineru_adapter_loads_content_from_parsed_output(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    output_dir = tmp_path / "mineru-output"
    output_dir.mkdir(parents=True, exist_ok=True)
    content_path = output_dir / "content_list.json"
    content_path.write_text(
        json.dumps([{"type": "text", "text": "from-mineru"}]),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_parse(_pdf_path, *, parser_mode=None):
        captured["mode"] = parser_mode
        return output_dir

    adapter = PdfMinerUAdapter(parse_pdf_fn=_fake_parse)
    loaded = adapter.load(
        E2ERunRequest(
            project_id="p1",
            document_id="d1",
            version_id="v1",
            pdf_path=pdf_path,
            mineru_parser="api",
        )
    )

    assert loaded.content_list == [{"type": "text", "text": "from-mineru"}]
    assert loaded.source_uri.endswith("content_list.json")
    assert loaded.parser_version == "pipeline-v1"
    assert loaded.warnings == []
    assert captured["mode"] == "api"


def test_pdf_mineru_adapter_adds_empty_warning_when_no_content(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%test")

    output_dir = tmp_path / "mineru-output"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = PdfMinerUAdapter(parse_pdf_fn=lambda _pdf_path, *, parser_mode=None: output_dir)
    loaded = adapter.load(
        E2ERunRequest(
            project_id="p1",
            document_id="d1",
            version_id="v1",
            pdf_path=pdf_path,
        )
    )

    assert loaded.content_list == []
    assert loaded.warnings == ["empty_content_list"]
