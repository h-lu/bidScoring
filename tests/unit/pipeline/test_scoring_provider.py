from __future__ import annotations

from bid_scoring.pipeline.application.scoring_provider import (
    HybridScoringProvider,
    BidAnalyzerScoringProvider,
    ScoringResult,
    ScoringRequest,
    WarningFallbackScoringProvider,
)


class _Analyzer:
    def analyze_version(
        self,
        *,
        version_id: str,
        bidder_name: str,
        project_name: str,
        dimensions: list[str] | None,
    ):
        _ = (version_id, bidder_name, project_name, dimensions)
        return type(
            "Report",
            (),
            {
                "overall_score": 82.5,
                "risk_level": "medium",
                "total_risks": 3,
                "total_benefits": 7,
                "chunks_analyzed": 18,
                "recommendations": ["r1"],
                "evidence_warnings": ["missing_bbox"],
                "dimensions": {},
            },
        )()


def test_bid_analyzer_scoring_provider_outputs_standard_payload():
    provider = BidAnalyzerScoringProvider(analyzer=_Analyzer())
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
            dimensions=["warranty"],
        )
    )

    payload = result.as_dict()
    assert payload["status"] == "completed"
    assert payload["overall_score"] == 82.5
    assert payload["risk_level"] == "medium"
    assert payload["evidence_warnings"] == ["missing_bbox"]


def test_warning_fallback_provider_appends_warning_codes():
    provider = WarningFallbackScoringProvider(
        fallback=BidAnalyzerScoringProvider(analyzer=_Analyzer()),
        warning_codes=["scoring_backend_agent_mcp_not_implemented"],
    )
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert result.warnings == ["scoring_backend_agent_mcp_not_implemented"]


class _Provider:
    def __init__(self, result: ScoringResult):
        self._result = result
        self.calls = 0

    def score(self, request: ScoringRequest) -> ScoringResult:
        _ = request
        self.calls += 1
        return self._result


def test_hybrid_scoring_provider_blends_results():
    agent = _Provider(
        ScoringResult(
            status="completed",
            overall_score=90.0,
            risk_level="low",
            total_risks=1,
            total_benefits=4,
            chunks_analyzed=8,
            recommendations=["agent-r1"],
            evidence_warnings=["missing_anchor_bbox"],
            dimensions={
                "warranty": {
                    "score": 90.0,
                    "risk_level": "low",
                    "chunks_found": 6,
                    "summary": "agent",
                    "evidence_warnings": ["missing_anchor_bbox"],
                }
            },
            warnings=[],
        )
    )
    analyzer = _Provider(
        ScoringResult(
            status="completed",
            overall_score=60.0,
            risk_level="medium",
            total_risks=3,
            total_benefits=2,
            chunks_analyzed=10,
            recommendations=["baseline-r1"],
            evidence_warnings=["missing_bbox"],
            dimensions={
                "warranty": {
                    "score": 60.0,
                    "risk_level": "medium",
                    "chunks_found": 10,
                    "summary": "baseline",
                    "evidence_warnings": ["missing_bbox"],
                }
            },
            warnings=["scoring_backend_unknown"],
        )
    )

    provider = HybridScoringProvider(
        primary=agent,
        secondary=analyzer,
        primary_weight=0.7,
    )
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )

    assert agent.calls == 1
    assert analyzer.calls == 1
    assert result.overall_score == 81.0
    assert result.risk_level == "medium"
    assert result.total_risks == 3
    assert result.total_benefits == 4
    assert result.chunks_analyzed == 10
    assert "agent-r1" in result.recommendations
    assert "baseline-r1" in result.recommendations
    assert result.dimensions["warranty"]["score"] == 81.0
    assert result.dimensions["warranty"]["risk_level"] == "medium"
    assert "missing_anchor_bbox" in result.evidence_warnings
    assert "missing_bbox" in result.evidence_warnings
    assert "scoring_backend_unknown" in result.warnings
