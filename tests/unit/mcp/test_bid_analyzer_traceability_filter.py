from __future__ import annotations

from mcp_servers.annotation_insights import AnnotationInsight
from mcp_servers.bid_analyzer import ANALYSIS_DIMENSIONS, BidAnalyzer


class _Retriever:
    def search_chunks(self, version_id: str, keywords: list[str]):
        _ = (version_id, keywords)
        return [
            {
                "chunk_id": "c-unverifiable",
                "page_idx": 1,
                "text_raw": "风险内容",
                "bbox": None,
                "element_type": "text",
                "evidence_status": "unverifiable",
                "warnings": ["missing_evidence_chain"],
                "is_verifiable": False,
            },
            {
                "chunk_id": "c-verifiable",
                "page_idx": 2,
                "text_raw": "优势内容",
                "bbox": [1, 2, 3, 4],
                "element_type": "text",
                "evidence_status": "verified",
                "warnings": [],
                "is_verifiable": True,
            },
        ]

    @staticmethod
    def collect_evidence_warnings(chunks):
        _ = chunks
        return ["missing_evidence_chain"]


class _Insight:
    def __init__(self) -> None:
        self.received_chunk_ids: list[str] = []

    def analyze_chunks(self, chunks, dimension_name):
        _ = dimension_name
        self.received_chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        return (
            [AnnotationInsight("risk", "r", "r", risk_level="medium")],
            [],
            [],
        )

    @staticmethod
    def extract_values(chunks, patterns):
        _ = (chunks, patterns)
        return {}


class _Scorer:
    @staticmethod
    def calculate_dimension_score(dimension, risks, benefits, extracted_values):
        _ = (dimension, benefits, extracted_values)
        return 50.0 - 5.0 * len(risks)

    @staticmethod
    def determine_dimension_risk_level(dimension, risks, benefits, extracted_values):
        _ = (dimension, benefits, extracted_values)
        return "high" if risks else "low"

    @staticmethod
    def calculate_overall_score(dimension_results):
        return sum(item.score for item in dimension_results.values())

    @staticmethod
    def determine_risk_level(total_risks, total_benefits):
        _ = total_benefits
        return "high" if total_risks else "low"


def test_bid_analyzer_filters_unverifiable_chunks_for_scoring():
    insight = _Insight()
    analyzer = BidAnalyzer(
        conn=object(),
        retriever=_Retriever(),
        insight_extractor=insight,
        dimension_scorer=_Scorer(),
    )

    result = analyzer._analyze_dimension(
        "33333333-3333-3333-3333-333333333333", ANALYSIS_DIMENSIONS["warranty"]
    )

    assert insight.received_chunk_ids == ["c-verifiable"]
    assert result.chunks_found == 1
    assert result.evidence_warnings == ["missing_evidence_chain"]
    assert result.evidence_citations == [
        {"chunk_id": "c-verifiable", "page_idx": 2, "bbox": [1, 2, 3, 4]}
    ]
