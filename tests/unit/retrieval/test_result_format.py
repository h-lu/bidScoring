from __future__ import annotations

from bid_scoring.retrieval.types import EvidenceUnit, RetrievalResult
from mcp_servers.retrieval.formatting import format_result


def test_format_result_includes_unit_evidence_chain():
    result = RetrievalResult(
        chunk_id="chunk-1",
        text="sample text",
        page_idx=3,
        score=0.8,
        source="hybrid",
        evidence_units=[
            EvidenceUnit(
                unit_id="unit-1",
                unit_index=1,
                unit_type="text",
                text="quoted",
                anchor_json={"anchors": [{"page_idx": 3, "bbox": [1, 2, 3, 4]}]},
                start_char=0,
                end_char=6,
            )
        ],
    )

    formatted = format_result(result, include_text=True, max_chars=None)

    assert formatted["evidence_status"] == "verified"
    assert formatted["warnings"] == []
    assert len(formatted["evidence_units"]) == 1
    assert formatted["evidence_units"][0]["unit_id"] == "unit-1"


def test_format_result_marks_warning_when_evidence_chain_missing():
    result = RetrievalResult(
        chunk_id="chunk-2",
        text="missing evidence",
        page_idx=4,
        score=0.4,
        source="keyword",
        evidence_units=[],
    )

    formatted = format_result(result, include_text=True, max_chars=None)

    assert formatted["evidence_status"] == "unverifiable"
    assert formatted["warnings"] == ["missing_evidence_chain"]
    assert formatted["evidence_units"] == []
