from __future__ import annotations

from types import SimpleNamespace

from bid_scoring.retrieval.types import EvidenceUnit, RetrievalResult
from mcp_servers import retrieval_server as srv


class _FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.rrf = SimpleNamespace(k=60)

    def retrieve(self, _query, keywords=None):  # pragma: no cover
        return self._results


def test_retrieve_impl_reports_unverifiable_warning_when_evidence_missing(monkeypatch):
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            text="text A",
            page_idx=1,
            score=0.9,
            source="hybrid",
            evidence_units=[],
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            text="text B",
            page_idx=2,
            score=0.8,
            source="hybrid",
            evidence_units=[
                EvidenceUnit(
                    unit_id="unit-2",
                    unit_index=2,
                    unit_type="text",
                    text="text B",
                    anchor_json={"anchors": [{"page_idx": 2, "bbox": [1, 2, 3, 4]}]},
                )
            ],
        ),
    ]
    monkeypatch.setattr(
        srv, "get_retriever", lambda version_id, top_k: _FakeRetriever(results)
    )

    response = srv.retrieve_impl(
        version_id="33333333-3333-3333-3333-333333333333",
        query="test query",
        top_k=10,
        mode="hybrid",
    )

    assert response["warnings"] == ["missing_evidence_chain"]
    assert response["results"][0]["evidence_status"] == "unverifiable"
    assert response["results"][1]["evidence_status"] == "verified"
    assert response["results"][1]["evidence_units"][0]["unit_id"] == "unit-2"


def test_retrieve_impl_can_include_diagnostics(monkeypatch):
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            text="text A",
            page_idx=1,
            score=0.9,
            source="hybrid",
            vector_score=0.8,
            keyword_score=1.0,
            evidence_units=[],
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            text="text B",
            page_idx=2,
            score=0.7,
            source="vector",
            vector_score=0.6,
            keyword_score=None,
            evidence_units=[],
        ),
    ]
    monkeypatch.setattr(
        srv, "get_retriever", lambda version_id, top_k: _FakeRetriever(results)
    )

    response = srv.retrieve_impl(
        version_id="33333333-3333-3333-3333-333333333333",
        query="test query",
        top_k=10,
        mode="hybrid",
        include_diagnostics=True,
    )

    assert response["diagnostics"]["result_count"] == 2
    assert response["diagnostics"]["keyword_hits"] == 1
    assert response["diagnostics"]["vector_hits"] == 2
