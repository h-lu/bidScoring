from __future__ import annotations

from bid_scoring.pipeline.application.traceability import build_traceability_summary


def test_build_traceability_summary_counts_traceable_and_untraceable_citations():
    summary = build_traceability_summary(
        {
            "evidence_warnings": ["missing_bbox"],
            "evidence_citations": {
                "warranty": [
                    {
                        "chunk_id": "chunk-1",
                        "page_idx": 0,
                        "bbox": [0, 0, 10, 10],
                    },
                    {
                        "chunk_id": "chunk-2",
                        "page_idx": 1,
                        "bbox": None,
                    },
                ],
                "delivery": [
                    {
                        "chunk_id": None,
                        "page_idx": 4,
                        "bbox": [0, 0, 3, 3],
                    }
                ],
            },
        }
    )

    assert summary["status"] == "verified_with_warnings"
    assert summary["dimension_count"] == 2
    assert summary["citation_count_total"] == 3
    assert summary["citation_count_traceable"] == 1
    assert summary["citation_count_untraceable"] == 2
    assert summary["highlight_ready_chunk_ids"] == ["chunk-1"]
    assert "missing_bbox" in summary["warnings"]
    assert "citation_missing_bbox" in summary["warnings"]
    assert "citation_missing_chunk_id" in summary["warnings"]


def test_build_traceability_summary_marks_unverifiable_when_no_citations():
    summary = build_traceability_summary(
        {
            "evidence_warnings": [],
            "evidence_citations": {},
        }
    )

    assert summary["status"] == "unverifiable"
    assert summary["dimension_count"] == 0
    assert summary["citation_count_total"] == 0
    assert summary["citation_count_traceable"] == 0
    assert summary["citation_coverage_ratio"] == 0.0
    assert summary["highlight_ready_chunk_ids"] == []
    assert "no_evidence_citations" in summary["warnings"]
    assert "no_highlight_ready_chunks" in summary["warnings"]
