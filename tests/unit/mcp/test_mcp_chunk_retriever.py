from __future__ import annotations

from mcp_servers.bid_analysis.retriever import McpChunkRetriever


def test_mcp_chunk_retriever_marks_verifiable_chunks_and_collects_warnings():
    def _fake_retrieve(**kwargs):
        _ = kwargs
        return {
            "warnings": ["missing_evidence_chain"],
            "results": [
                {
                    "chunk_id": "c-ok",
                    "page_idx": 1,
                    "bbox": [1, 2, 3, 4],
                    "element_type": "text",
                    "text": "质保 5 年",
                    "evidence_status": "verified",
                    "warnings": [],
                },
                {
                    "chunk_id": "c-no-evidence",
                    "page_idx": 2,
                    "bbox": [2, 3, 4, 5],
                    "element_type": "text",
                    "text": "无证据链",
                    "evidence_status": "unverifiable",
                    "warnings": ["missing_evidence_chain"],
                },
                {
                    "chunk_id": "c-no-bbox",
                    "page_idx": 3,
                    "bbox": None,
                    "element_type": "text",
                    "text": "缺失 bbox",
                    "evidence_status": "verified",
                    "warnings": [],
                },
            ],
        }

    retriever = McpChunkRetriever(retrieve_fn=_fake_retrieve)
    chunks = retriever.search_chunks(
        version_id="33333333-3333-3333-3333-333333333333",
        keywords=["质保"],
    )

    assert len(chunks) == 3
    assert chunks[0]["is_verifiable"] is True
    assert chunks[1]["is_verifiable"] is False
    assert chunks[2]["is_verifiable"] is False
    assert chunks[0]["text_raw"] == "质保 5 年"

    warnings = retriever.collect_evidence_warnings(chunks)
    assert "missing_evidence_chain" in warnings
    assert "missing_bbox" in warnings
    assert "unverifiable_evidence_for_scoring" in warnings
