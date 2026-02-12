from __future__ import annotations

from mcp_servers.retrieval.operations_evidence import compare_across_versions
from mcp_servers.retrieval.operations_search import search_chunks


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
