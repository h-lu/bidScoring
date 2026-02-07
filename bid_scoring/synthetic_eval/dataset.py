from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .content_builder import ContentBuilder
from .labels import build_qrels, build_queries
from .profiles import DEFAULT_SCENARIOS, SCENARIO_PROFILES


DEFAULT_OUTPUT_DIR = Path("data/eval/hybrid_medical_synthetic")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _source_id(index: int) -> str:
    return f"chunk_{index:04d}"


def _assert_dataset(
    content: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> None:
    if len(content) < 1200:
        raise ValueError("content_list too small (expected >= 1200)")
    if len(queries) < 20:
        raise ValueError("query count must be >= 20")
    if not qrels:
        raise ValueError("qrels empty")

    source_ids = {_source_id(i) for i in range(len(content))}
    content_by_sid = {_source_id(i): item for i, item in enumerate(content)}
    for row in qrels:
        sid = row["source_id"]
        if sid not in source_ids:
            raise ValueError(f"qrels source missing: {sid}")
        if content_by_sid[sid]["type"] in {"header", "page_number", "footer"}:
            raise ValueError(f"qrels labels skipped type: {sid}")

    qids = {q["query_id"] for q in queries}
    qrels_qids = {r["query_id"] for r in qrels}
    missing = qids - qrels_qids
    if missing:
        raise ValueError(f"queries without qrels: {sorted(missing)}")


def _render_readme(manifest: dict[str, Any]) -> str:
    scenarios = manifest["scenarios"]
    lines = [
        "# Synthetic Medical Hybrid Retrieval Eval Set",
        "",
        "该目录包含 A/B/C 三个供应商版本的模拟投标评测集。",
        "",
        "## 产物",
        "- queries.json（跨版本共享）",
        "- content_list.synthetic_bidder_A|B|C.json",
        "- qrels.source_id.A|B|C.jsonl",
        "- multi_version_manifest.json",
        "",
        "## 场景概览",
    ]
    for name in sorted(scenarios):
        item = scenarios[name]
        lines.append(
            f"- {name}: {item['bidder_name']} / {item['content_count']} chunks / {item['qrel_count']} qrels"
        )
    lines.extend(
        [
            "",
            "## relevance 标准",
            "- 3: 主证据",
            "- 2: 支持证据",
            "- 1: 弱相关",
            "- 0: 干扰项",
            "",
            "## 评估建议",
            "1. 将 A/B/C content_list 分别入库为不同 version_id。",
            "2. 使用 scripts/evaluate_hybrid_search_multiversion.py 做跨版本基线。",
        ]
    )
    return "\n".join(lines) + "\n"


def generate(
    output_dir: Path,
    *,
    scenario: str = "A",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    profile = SCENARIO_PROFILES[scenario]
    builder = ContentBuilder(profile)
    content, anchors = builder.build()

    # Per-scenario qrels (version_tag keeps qrels self-describing).
    queries = build_queries(version_tag=profile.version_tag)
    qrels = build_qrels(anchors=anchors, content=content, version_tag=profile.version_tag)
    _assert_dataset(content, queries, qrels)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / f"content_list.synthetic_bidder_{scenario}.json", content)
    _write_jsonl(output_dir / f"qrels.source_id.{scenario}.jsonl", qrels)

    if scenario == "A":
        # Backwards-compatible convenience files (single-version).
        _write_json(output_dir / "content_list.synthetic_bidder.json", content)
        _write_json(output_dir / "queries.json", build_queries())
        _write_jsonl(output_dir / "qrels.source_id.jsonl", qrels)

    return content, queries, qrels


def generate_all(
    output_dir: Path,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "synthetic_medical_hybrid_eval",
        "shared_queries_file": "queries.json",
        "scenarios": {},
    }

    shared_queries = build_queries()
    _write_json(output_dir / "queries.json", shared_queries)

    for scenario in scenarios:
        content, _, qrels = generate(output_dir, scenario=scenario)
        profile = SCENARIO_PROFILES[scenario]
        manifest["scenarios"][scenario] = {
            "version_tag": profile.version_tag,
            "bidder_name": profile.bidder_name,
            "content_file": f"content_list.synthetic_bidder_{scenario}.json",
            "qrels_file": f"qrels.source_id.{scenario}.jsonl",
            "content_count": len(content),
            "qrel_count": len(qrels),
            "query_count": len(shared_queries),
        }

    _write_json(output_dir / "multi_version_manifest.json", manifest)
    (output_dir / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    return manifest


def validate(
    output_dir: Path,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
) -> dict[str, tuple[int, int, int]]:
    shared_queries = _read_json(output_dir / "queries.json")
    summary: dict[str, tuple[int, int, int]] = {}

    for scenario in scenarios:
        content = _read_json(output_dir / f"content_list.synthetic_bidder_{scenario}.json")
        qrels = _read_jsonl(output_dir / f"qrels.source_id.{scenario}.jsonl")
        _assert_dataset(content, shared_queries, qrels)
        summary[scenario] = (len(content), len(shared_queries), len(qrels))

    return summary
