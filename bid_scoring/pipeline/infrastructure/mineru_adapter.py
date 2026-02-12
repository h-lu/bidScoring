from __future__ import annotations

import json
from pathlib import Path


def load_content_list_from_output(output_dir: Path) -> list[dict]:
    """Load MinerU `content_list.json` from an output directory."""
    content_list_path = output_dir / "content_list.json"
    if not content_list_path.exists():
        return []

    data = json.loads(content_list_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("content_list.json must be a JSON list")
    return data

