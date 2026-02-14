from __future__ import annotations

from bid_scoring.pipeline.application.scoring_provider import (
    AgentMcpScoringProvider,
    HybridScoringProvider,
    BidAnalyzerScoringProvider,
    OpenAIMcpAgentExecutor,
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


class _Executor:
    def __init__(
        self, result: ScoringResult | None = None, exc: Exception | None = None
    ):
        self._result = result
        self._exc = exc
        self.calls = 0

    def score(self, request: ScoringRequest) -> ScoringResult:
        _ = request
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        assert self._result is not None
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
            evidence_citations={
                "warranty": [
                    {
                        "chunk_id": "agent-chunk-1",
                        "page_idx": 1,
                        "bbox": [1, 2, 3, 4],
                    }
                ]
            },
            dimensions={
                "warranty": {
                    "score": 90.0,
                    "risk_level": "low",
                    "chunks_found": 6,
                    "summary": "agent",
                    "evidence_warnings": ["missing_anchor_bbox"],
                    "evidence_citations": [
                        {
                            "chunk_id": "agent-chunk-1",
                            "page_idx": 1,
                            "bbox": [1, 2, 3, 4],
                        }
                    ],
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
            evidence_citations={
                "warranty": [
                    {
                        "chunk_id": "baseline-chunk-1",
                        "page_idx": 2,
                        "bbox": [2, 3, 4, 5],
                    }
                ]
            },
            dimensions={
                "warranty": {
                    "score": 60.0,
                    "risk_level": "medium",
                    "chunks_found": 10,
                    "summary": "baseline",
                    "evidence_warnings": ["missing_bbox"],
                    "evidence_citations": [
                        {
                            "chunk_id": "baseline-chunk-1",
                            "page_idx": 2,
                            "bbox": [2, 3, 4, 5],
                        }
                    ],
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
    assert len(result.evidence_citations["warranty"]) == 2
    assert len(result.dimensions["warranty"]["evidence_citations"]) == 2


def test_agent_mcp_scoring_provider_prefers_executor_result():
    executor = _Executor(
        result=ScoringResult(
            status="completed",
            overall_score=92.0,
            risk_level="low",
            total_risks=1,
            total_benefits=5,
            chunks_analyzed=12,
            recommendations=["agent"],
            evidence_warnings=[],
            dimensions={},
            warnings=[],
        )
    )
    fallback = _Provider(
        ScoringResult(
            status="completed",
            overall_score=60.0,
            risk_level="medium",
            total_risks=3,
            total_benefits=2,
            chunks_analyzed=6,
            recommendations=["fallback"],
            evidence_warnings=[],
            dimensions={},
            warnings=[],
        )
    )
    provider = AgentMcpScoringProvider(executor=executor, fallback=fallback)

    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert executor.calls == 1
    assert fallback.calls == 0
    assert result.overall_score == 92.0


def test_agent_mcp_scoring_provider_falls_back_and_marks_warning():
    executor = _Executor(exc=RuntimeError("agent failed"))
    fallback = _Provider(
        ScoringResult(
            status="completed",
            overall_score=61.0,
            risk_level="medium",
            total_risks=3,
            total_benefits=2,
            chunks_analyzed=6,
            recommendations=["fallback"],
            evidence_warnings=[],
            dimensions={},
            warnings=[],
        )
    )
    provider = AgentMcpScoringProvider(executor=executor, fallback=fallback)

    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert executor.calls == 1
    assert fallback.calls == 1
    assert result.overall_score == 61.0
    assert "scoring_backend_agent_mcp_fallback" in result.warnings


def test_openai_mcp_agent_executor_parses_json_and_filters_unverifiable_evidence():
    class _Message:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.message = _Message(content)

    class _Response:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, **kwargs):
            _ = kwargs
            return _Response(
                """
                {
                  "overall_score": 86,
                  "risk_level": "medium",
                  "total_risks": 2,
                  "total_benefits": 4,
                  "recommendations": ["优先核验商务条款"],
                  "dimensions": {
                    "warranty": {"score": 90, "risk_level": "low", "summary": "质保承诺明确"},
                    "delivery": {"score": 70, "risk_level": "medium", "summary": "响应时效一般"}
                  }
                }
                """
            )

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self):
            self.chat = _Chat()

    def _fake_retrieve(**kwargs):
        if kwargs["keywords"][0] == "质保":
            return {
                "warnings": ["missing_evidence_chain"],
                "results": [
                    {
                        "chunk_id": "c-ok",
                        "page_idx": 1,
                        "bbox": [1, 2, 3, 4],
                        "text": "免费质保 5 年",
                        "evidence_status": "verified",
                        "warnings": [],
                    },
                    {
                        "chunk_id": "c-bad",
                        "page_idx": 2,
                        "bbox": None,
                        "text": "无法定位",
                        "evidence_status": "unverifiable",
                        "warnings": ["missing_bbox"],
                    },
                ],
            }
        return {
            "warnings": [],
            "results": [
                {
                    "chunk_id": "c-delivery",
                    "page_idx": 5,
                    "bbox": [2, 2, 8, 8],
                    "text": "4小时内响应",
                    "evidence_status": "verified",
                    "warnings": [],
                }
            ],
        }

    executor = OpenAIMcpAgentExecutor(
        retrieve_fn=_fake_retrieve,
        client=_Client(),
        model="test-model",
    )

    result = executor.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
            dimensions=["warranty", "delivery"],
        )
    )

    assert result.overall_score == 86.0
    assert result.chunks_analyzed == 2
    assert result.dimensions["warranty"]["chunks_found"] == 1
    assert result.dimensions["delivery"]["chunks_found"] == 1
    assert len(result.evidence_citations["warranty"]) == 1
    assert result.evidence_citations["warranty"][0]["chunk_id"] == "c-ok"
    assert len(result.dimensions["warranty"]["evidence_citations"]) == 1
    assert "missing_evidence_chain" in result.evidence_warnings
    assert "missing_bbox" in result.evidence_warnings
    assert "unverifiable_evidence_for_scoring" in result.evidence_warnings
