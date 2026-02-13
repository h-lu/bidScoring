from __future__ import annotations

import json
from pathlib import Path

from bid_scoring.pipeline.infrastructure.mineru_adapter import (
    load_content_list_from_output,
)


def test_load_content_list_from_output_reads_json_list(tmp_path: Path):
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [{"type": "text", "text": "ok"}]
    (output_dir / "content_list.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_content_list_from_output(output_dir)
    assert loaded == payload


def test_load_content_list_from_output_returns_empty_when_missing(tmp_path: Path):
    loaded = load_content_list_from_output(tmp_path)
    assert loaded == []
