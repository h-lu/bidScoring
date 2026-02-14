from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from mcp_servers.annotation_insights import AnnotationInsight
from mcp_servers.bid_analyzer import BidAnalyzer


class _Retriever:
    def search_chunks(self, version_id: str, keywords: list[str]):
        _ = (version_id, keywords)
        return [
            {
                "chunk_id": "c1",
                "page_idx": 0,
                "text_raw": "质保5年",
                "bbox": [0, 0, 1, 1],
                "element_type": "text",
            }
        ]

    @staticmethod
    def collect_evidence_warnings(chunks):
        _ = chunks
        return []


class _Insight:
    def analyze_chunks(self, chunks, dimension_name):
        _ = (chunks, dimension_name)
        return (
            [AnnotationInsight("risk", "r", "r", risk_level="medium")],
            [AnnotationInsight("benefit", "b", "b", risk_level=None)],
            [],
        )

    @staticmethod
    def extract_values(chunks, patterns):
        _ = (chunks, patterns)
        return {"pattern_years": {"values": [5], "min": 5, "max": 5, "count": 1}}


class _Scorer:
    @staticmethod
    def calculate_dimension_score(dimension, risks, benefits, extracted_values):
        _ = (dimension, risks, benefits, extracted_values)
        return 77.0

    @staticmethod
    def determine_dimension_risk_level(dimension, risks, benefits, extracted_values):
        _ = (dimension, risks, benefits, extracted_values)
        return "medium"

    @staticmethod
    def calculate_overall_score(dimension_results):
        _ = dimension_results
        return 77.0

    @staticmethod
    def determine_risk_level(total_risks, total_benefits):
        _ = (total_risks, total_benefits)
        return "medium"


class _Recommender:
    @staticmethod
    def generate(dimension_results):
        _ = dimension_results
        return ["custom-recommendation"]


def test_bid_analyzer_supports_component_injection():
    analyzer = BidAnalyzer(
        conn=object(),
        retriever=_Retriever(),
        insight_extractor=_Insight(),
        dimension_scorer=_Scorer(),
        recommendation_generator=_Recommender(),
    )

    report = analyzer.analyze_version(
        version_id="33333333-3333-3333-3333-333333333333",
        bidder_name="A公司",
        dimensions=["warranty"],
    )

    assert report.overall_score == 77.0
    assert report.risk_level == "medium"
    assert report.recommendations == ["custom-recommendation"]
    assert report.dimensions["warranty"].score == 77.0
    assert report.dimensions["warranty"].evidence_citations == [
        {"chunk_id": "c1", "page_idx": 0, "bbox": [0, 0, 1, 1]}
    ]


def test_bid_analyzer_normalizes_citation_payload_for_json():
    class _UuidRetriever:
        def search_chunks(self, version_id: str, keywords: list[str]):
            _ = (version_id, keywords)
            return [
                {
                    "chunk_id": UUID("11111111-1111-1111-1111-111111111111"),
                    "page_idx": Decimal("2"),
                    "text_raw": "交付时间 3 个月",
                    "bbox": [Decimal("1.0"), Decimal("2.0"), 3, 4],
                    "element_type": "text",
                }
            ]

        @staticmethod
        def collect_evidence_warnings(chunks):
            _ = chunks
            return []

    analyzer = BidAnalyzer(
        conn=object(),
        retriever=_UuidRetriever(),
        insight_extractor=_Insight(),
        dimension_scorer=_Scorer(),
        recommendation_generator=_Recommender(),
    )

    report = analyzer.analyze_version(
        version_id="33333333-3333-3333-3333-333333333333",
        bidder_name="A公司",
        dimensions=["delivery"],
    )
    citations = report.dimensions["delivery"].evidence_citations
    assert citations == [
        {
            "chunk_id": "11111111-1111-1111-1111-111111111111",
            "page_idx": 2,
            "bbox": [1.0, 2.0, 3, 4],
        }
    ]


def test_bid_analyzer_accepts_question_keyword_overrides():
    class _CaptureRetriever:
        def __init__(self):
            self.calls: list[list[str]] = []

        def search_chunks(self, version_id: str, keywords: list[str]):
            _ = version_id
            self.calls.append(list(keywords))
            return []

        @staticmethod
        def collect_evidence_warnings(chunks):
            _ = chunks
            return []

    retriever = _CaptureRetriever()
    analyzer = BidAnalyzer(
        conn=object(),
        retriever=retriever,
        insight_extractor=_Insight(),
        dimension_scorer=_Scorer(),
        recommendation_generator=_Recommender(),
    )

    _ = analyzer.analyze_version(
        version_id="33333333-3333-3333-3333-333333333333",
        bidder_name="A公司",
        dimensions=["warranty"],
        question_dimension_keywords={"warranty": ["售后承诺", "保修时长"]},
    )

    assert retriever.calls == [["售后承诺", "保修时长"]]
