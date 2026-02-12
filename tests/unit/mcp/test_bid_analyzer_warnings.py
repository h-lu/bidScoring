from __future__ import annotations

from mcp_servers.bid_analyzer import ANALYSIS_DIMENSIONS, BidAnalyzer


def test_analyze_dimension_marks_missing_bbox_as_warning(monkeypatch):
    analyzer = BidAnalyzer(conn=object())

    monkeypatch.setattr(
        analyzer,
        "_search_chunks",
        lambda _version_id, _keywords: [
            {"chunk_id": "c1", "text_raw": "x", "bbox": None, "element_type": "text"}
        ],
    )

    result = analyzer._analyze_dimension(
        "33333333-3333-3333-3333-333333333333", ANALYSIS_DIMENSIONS["warranty"]
    )

    assert result.evidence_warnings == ["missing_bbox"]

