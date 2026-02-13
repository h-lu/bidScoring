from __future__ import annotations

from bid_scoring.retrieval.types import EvidenceUnit, RetrievalResult
from mcp_servers import retrieval_server as srv
from mcp_servers.retrieval.operations_annotation import (
    prepare_highlight_targets_from_results,
)


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, _query, keywords=None):  # pragma: no cover
        return self._results


def test_prepare_highlight_targets_filters_unverifiable_and_missing_bbox():
    results = [
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
    ]

    gate = prepare_highlight_targets_from_results(results)

    assert gate["chunk_ids"] == ["chunk-ok"]
    assert "missing_chunk_bbox" in gate["warnings"]
    assert "unverifiable_evidence_for_highlight" in gate["warnings"]


def test_traceability_chain_retrieve_to_highlight_filters_non_factual_items(
    monkeypatch,
):
    retrieval_results = [
        RetrievalResult(
            chunk_id="chunk-ok",
            text="text ok",
            page_idx=1,
            score=0.9,
            source="hybrid",
            bbox=[1, 2, 3, 4],
            evidence_units=[
                EvidenceUnit(
                    unit_id="unit-1",
                    unit_index=1,
                    unit_type="text",
                    text="text ok",
                    anchor_json={"anchors": [{"page_idx": 1, "bbox": [1, 2, 3, 4]}]},
                )
            ],
        ),
        RetrievalResult(
            chunk_id="chunk-no-bbox",
            text="text no bbox",
            page_idx=2,
            score=0.8,
            source="hybrid",
            bbox=None,
            evidence_units=[
                EvidenceUnit(
                    unit_id="unit-2",
                    unit_index=2,
                    unit_type="text",
                    text="text no bbox",
                    anchor_json={"anchors": [{"page_idx": 2, "bbox": [1, 2, 3, 4]}]},
                )
            ],
        ),
    ]
    monkeypatch.setattr(
        srv,
        "get_retriever",
        lambda version_id, top_k: _FakeRetriever(retrieval_results),
    )

    retrieved = srv.retrieve_impl(
        version_id="33333333-3333-3333-3333-333333333333",
        query="test query",
        top_k=10,
        mode="hybrid",
    )
    gate = prepare_highlight_targets_from_results(retrieved["results"])

    captured: dict[str, object] = {}

    def _fake_highlight_pdf(**kwargs):
        captured.update(kwargs)
        return {"success": True, "highlights_added": len(kwargs["chunk_ids"])}

    monkeypatch.setattr(srv, "_highlight_pdf", _fake_highlight_pdf)

    highlight_resp = srv._highlight_pdf(
        version_id="33333333-3333-3333-3333-333333333333",
        chunk_ids=gate["chunk_ids"],
        topic="risk",
    )

    assert captured["chunk_ids"] == ["chunk-ok"]
    assert highlight_resp["success"] is True
    assert "missing_chunk_bbox" in gate["warnings"]


def test_prepare_highlight_targets_tool_uses_traceability_gate(monkeypatch):
    retrieval_results = [
        RetrievalResult(
            chunk_id="chunk-ok",
            text="text ok",
            page_idx=1,
            score=0.9,
            source="hybrid",
            bbox=[1, 2, 3, 4],
            evidence_units=[
                EvidenceUnit(
                    unit_id="unit-1",
                    unit_index=1,
                    unit_type="text",
                    text="text ok",
                    anchor_json={"anchors": [{"page_idx": 1, "bbox": [1, 2, 3, 4]}]},
                )
            ],
        ),
        RetrievalResult(
            chunk_id="chunk-bad",
            text="text bad",
            page_idx=2,
            score=0.8,
            source="hybrid",
            bbox=None,
            evidence_units=[],
        ),
    ]
    monkeypatch.setattr(
        srv,
        "get_retriever",
        lambda version_id, top_k: _FakeRetriever(retrieval_results),
    )

    response = srv.prepare_highlight_targets_impl(
        version_id="33333333-3333-3333-3333-333333333333",
        query="测试问题",
        top_k=5,
        mode="hybrid",
        include_diagnostics=True,
    )

    assert response["chunk_ids"] == ["chunk-ok"]
    assert response["included_count"] == 1
    assert response["excluded_count"] == 1
    assert "missing_chunk_bbox" in response["warnings"]
    assert "missing_evidence_chain" in response["warnings"]
    assert response["diagnostics"]["gate"]["included_count"] == 1
