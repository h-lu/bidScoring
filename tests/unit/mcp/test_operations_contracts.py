from __future__ import annotations

from mcp_servers.retrieval.operations_annotation import (
    prepare_highlight_targets_for_query,
)
from mcp_servers.retrieval.operations_evidence import compare_across_versions
from mcp_servers.retrieval.operations_search import batch_search, search_chunks


def test_search_chunks_applies_element_type_filter():
    def _fake_retrieve_fn(**_kwargs):
        return {
            "results": [
                {
                    "chunk_id": "c1",
                    "page_idx": 1,
                    "element_type": "table",
                    "score": 0.8,
                },
                {
                    "chunk_id": "c2",
                    "page_idx": 1,
                    "element_type": "text",
                    "score": 0.7,
                },
            ]
        }

    result = search_chunks(
        retrieve_fn=_fake_retrieve_fn,
        version_id="33333333-3333-3333-3333-333333333333",
        query="测试",
        top_k=10,
        mode="hybrid",
        element_types=["table"],
    )

    assert len(result["results"]) == 1
    assert result["results"][0]["chunk_id"] == "c1"


def test_compare_across_versions_ignores_none_scores_when_normalizing():
    by_version = {
        "v1": [{"chunk_id": "a", "score": None}, {"chunk_id": "b", "score": 0.5}],
        "v2": [{"chunk_id": "c", "score": 0.8}],
    }

    def _fake_retrieve_fn(version_id: str, **_kwargs):
        return {"results": [item.copy() for item in by_version[version_id]]}

    result = compare_across_versions(
        retrieve_fn=_fake_retrieve_fn,
        version_ids=["v1", "v2"],
        query="售后",
        top_k_per_version=3,
        normalize_scores=True,
    )

    row_none = result["results_by_version"]["v1"][0]
    row_scored = result["results_by_version"]["v2"][0]

    assert row_none["score"] is None
    assert row_none["normalized_score"] is None
    assert row_scored["normalized_score"] == 1.0


def test_search_chunks_can_include_diagnostics():
    def _fake_retrieve_fn(**_kwargs):
        return {
            "results": [{"chunk_id": "c1", "page_idx": 1, "element_type": "text"}],
            "diagnostics": {"result_count": 1},
        }

    result = search_chunks(
        retrieve_fn=_fake_retrieve_fn,
        version_id="33333333-3333-3333-3333-333333333333",
        query="测试",
        top_k=10,
        mode="hybrid",
        include_diagnostics=True,
    )

    assert result["diagnostics"]["source_result_count"] == 1
    assert result["diagnostics"]["filtered_result_count"] == 1


def test_batch_search_can_include_per_query_diagnostics():
    def _fake_retrieve_fn(query: str, **_kwargs):
        return {
            "results": [{"chunk_id": f"c-{query}", "page_idx": 1, "score": 0.8}],
            "diagnostics": {"result_count": 1, "vector_hits": 1},
        }

    result = batch_search(
        retrieve_fn=_fake_retrieve_fn,
        version_id="33333333-3333-3333-3333-333333333333",
        queries=["q1", "q2"],
        top_k_per_query=2,
        mode="hybrid",
        include_diagnostics=True,
    )

    assert result["diagnostics"]["query_count"] == 2
    assert result["diagnostics"]["per_query"]["q1"]["result_count"] == 1


def test_compare_across_versions_can_include_per_version_diagnostics():
    def _fake_retrieve_fn(version_id: str, **_kwargs):
        return {
            "results": [{"chunk_id": f"c-{version_id}", "score": 0.8}],
            "diagnostics": {"result_count": 1, "hybrid_hits": 1},
        }

    result = compare_across_versions(
        retrieve_fn=_fake_retrieve_fn,
        version_ids=["v1", "v2"],
        query="售后",
        top_k_per_version=3,
        normalize_scores=True,
        include_diagnostics=True,
    )

    assert result["diagnostics"]["version_count"] == 2
    assert result["diagnostics"]["per_version"]["v1"]["hybrid_hits"] == 1


def test_prepare_highlight_targets_for_query_filters_non_factual_items():
    def _fake_retrieve_fn(**_kwargs):
        return {
            "warnings": ["missing_evidence_chain"],
            "results": [
                {
                    "chunk_id": "chunk-ok",
                    "bbox": [1, 2, 3, 4],
                    "evidence_status": "verified",
                    "warnings": [],
                },
                {
                    "chunk_id": "chunk-no-bbox",
                    "bbox": None,
                    "evidence_status": "verified",
                    "warnings": [],
                },
                {
                    "chunk_id": "chunk-unverified",
                    "bbox": [1, 2, 3, 4],
                    "evidence_status": "unverifiable",
                    "warnings": ["missing_evidence_chain"],
                },
            ],
        }

    result = prepare_highlight_targets_for_query(
        retrieve_fn=_fake_retrieve_fn,
        version_id="33333333-3333-3333-3333-333333333333",
        query="测试",
        top_k=5,
        mode="hybrid",
    )

    assert result["chunk_ids"] == ["chunk-ok"]
    assert result["included_count"] == 1
    assert result["excluded_count"] == 2
    assert "missing_chunk_bbox" in result["warnings"]
    assert "unverifiable_evidence_for_highlight" in result["warnings"]
    assert "missing_evidence_chain" in result["warnings"]


def test_prepare_highlight_targets_for_query_can_include_diagnostics():
    def _fake_retrieve_fn(**_kwargs):
        return {
            "warnings": [],
            "results": [
                {
                    "chunk_id": "chunk-ok",
                    "bbox": [1, 2, 3, 4],
                    "evidence_status": "verified",
                    "warnings": [],
                }
            ],
            "diagnostics": {"result_count": 1},
        }

    result = prepare_highlight_targets_for_query(
        retrieve_fn=_fake_retrieve_fn,
        version_id="33333333-3333-3333-3333-333333333333",
        query="测试",
        top_k=5,
        mode="hybrid",
        include_diagnostics=True,
    )

    assert result["diagnostics"]["retrieval"]["result_count"] == 1
    assert result["diagnostics"]["gate"]["included_count"] == 1
