from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_synthetic_hybrid_eval_data import generate_all, validate


def _build_assets(tmp_path: Path):
    out_dir = tmp_path / "hybrid_eval_assets"
    generate_all(out_dir)
    summary = validate(out_dir)
    return out_dir, summary


def _load_qrels(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(__import__("json").loads(line))
    return rows


def test_generate_and_validate_roundtrip(tmp_path: Path):
    _, summary = _build_assets(tmp_path)
    assert set(summary) == {"A", "B", "C"}
    for _, (content_count, query_count, qrel_count) in summary.items():
        # Total content_list includes header/page_number noise.
        # Real tender docs are typically ~1000+ chunks; we keep it similar.
        assert content_count >= 1200
        assert query_count >= 20
        assert qrel_count >= 40


def test_ingestable_chunk_count_is_close_to_real_docs(tmp_path: Path):
    out_dir, _ = _build_assets(tmp_path)

    for scenario in ["A", "B", "C"]:
        content = json.loads(
            (out_dir / f"content_list.synthetic_bidder_{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        ingestable = [x for x in content if x["type"] not in {"header", "page_number", "footer"}]
        assert len(ingestable) >= 950


def test_content_list_type_coverage(tmp_path: Path):
    out_dir, _ = _build_assets(tmp_path)
    for scenario in ["A", "B", "C"]:
        content = json.loads(
            (out_dir / f"content_list.synthetic_bidder_{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        types = {item["type"] for item in content}
        expected = {
            "text",
            "list",
            "table",
            "image",
            "aside_text",
            "header",
            "page_number",
        }
        assert expected.issubset(types)


def test_queries_and_qrels_alignment(tmp_path: Path):
    out_dir, _ = _build_assets(tmp_path)
    queries = json.loads((out_dir / "queries.json").read_text(encoding="utf-8"))
    query_ids = {q["query_id"] for q in queries}

    for scenario in ["A", "B", "C"]:
        content = json.loads(
            (out_dir / f"content_list.synthetic_bidder_{scenario}.json").read_text(
                encoding="utf-8"
            )
        )
        qrels = _load_qrels(out_dir / f"qrels.source_id.{scenario}.jsonl")

        qrels_query_ids = {r["query_id"] for r in qrels}
        assert query_ids.issubset(qrels_query_ids)

        source_ids = {f"chunk_{i:04d}" for i in range(len(content))}
        content_by_sid = {f"chunk_{i:04d}": item for i, item in enumerate(content)}

        for row in qrels:
            sid = row["source_id"]
            assert sid in source_ids
            assert content_by_sid[sid]["type"] not in {
                "header",
                "page_number",
                "footer",
            }


def test_every_query_has_authoritative_evidence(tmp_path: Path):
    out_dir, _ = _build_assets(tmp_path)
    for scenario in ["A", "B", "C"]:
        qrels = _load_qrels(out_dir / f"qrels.source_id.{scenario}.jsonl")
        by_query: dict[str, list[dict]] = {}
        for row in qrels:
            by_query.setdefault(row["query_id"], []).append(row)

        for query_id, rows in by_query.items():
            assert any(r["relevance"] == 3 for r in rows), (
                f"{query_id} missing relevance=3"
            )


def test_edge_tags_coverage(tmp_path: Path):
    out_dir, _ = _build_assets(tmp_path)
    all_tags = set()
    for scenario in ["A", "B", "C"]:
        qrels = _load_qrels(out_dir / f"qrels.source_id.{scenario}.jsonl")
        all_tags |= {tag for row in qrels for tag in row.get("edge_tags", [])}

    required_tags = {
        "conflict_clause",
        "negation",
        "alias_term",
        "ocr_noise",
        "table_evidence",
        "numeric_normalization",
        "template_vs_commitment",
    }
    assert required_tags.issubset(all_tags)


def test_query_type_coverage(tmp_path: Path):
    out_dir, _ = _build_assets(tmp_path)
    queries = json.loads((out_dir / "queries.json").read_text(encoding="utf-8"))
    query_types = {q["query_type"] for q in queries}

    required_types = {
        "keyword_critical",
        "factual",
        "semantic",
        "negation",
        "numeric_reasoning",
        "conflict_resolution",
    }
    assert required_types.issubset(query_types)
