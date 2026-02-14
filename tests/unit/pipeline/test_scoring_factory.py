from __future__ import annotations

from bid_scoring.pipeline.application.scoring_factory import build_scoring_provider
from bid_scoring.pipeline.application.scoring_provider import (
    ScoringRequest,
    ScoringResult,
)


class _Analyzer:
    def analyze_version(
        self,
        *,
        version_id: str,
        bidder_name: str,
        project_name: str,
        dimensions: list[str] | None = None,
    ):
        _ = (version_id, bidder_name, project_name, dimensions)
        return type(
            "Report",
            (),
            {
                "overall_score": 80.0,
                "risk_level": "medium",
                "total_risks": 1,
                "total_benefits": 2,
                "chunks_analyzed": 3,
                "recommendations": [],
                "evidence_warnings": [],
                "dimensions": {},
            },
        )()


class _AgentExecutor:
    def score(self, request: ScoringRequest) -> ScoringResult:
        _ = request
        return ScoringResult(
            status="completed",
            overall_score=92.0,
            risk_level="low",
            total_risks=1,
            total_benefits=3,
            chunks_analyzed=4,
            recommendations=["agent"],
            evidence_warnings=[],
            dimensions={},
            warnings=[],
        )


def test_scoring_factory_returns_analyzer_provider(monkeypatch):
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.build_bid_analyzer",
        lambda conn, backend: _Analyzer(),
    )
    provider = build_scoring_provider("analyzer", conn=object())
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert result.status == "completed"
    assert result.warnings == []


def test_scoring_factory_agent_mcp_returns_real_provider(monkeypatch):
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.build_bid_analyzer",
        lambda conn, backend: _Analyzer(),
    )
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.OpenAIMcpAgentExecutor",
        _AgentExecutor,
    )
    provider = build_scoring_provider("agent-mcp", conn=object())
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert result.status == "completed"
    assert result.overall_score == 92.0
    assert "scoring_backend_agent_mcp_fallback" not in result.warnings


def test_scoring_factory_hybrid_builds_dual_backends(monkeypatch):
    captured_backends: list[str] = []

    def _fake_build_bid_analyzer(conn, backend):
        _ = conn
        captured_backends.append(backend)
        return _Analyzer()

    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.build_bid_analyzer",
        _fake_build_bid_analyzer,
    )
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.OpenAIMcpAgentExecutor",
        _AgentExecutor,
    )
    provider = build_scoring_provider("hybrid", conn=object())
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert set(captured_backends) == {"analyzer", "agent-mcp"}
    assert result.status == "completed"
    assert "scoring_backend_hybrid_not_implemented" not in result.warnings


def test_scoring_factory_hybrid_applies_custom_weight(monkeypatch):
    class _ByBackendAnalyzer:
        def __init__(self, score: float):
            self._score = score

        def analyze_version(
            self,
            *,
            version_id: str,
            bidder_name: str,
            project_name: str,
            dimensions: list[str] | None = None,
        ):
            _ = (version_id, bidder_name, project_name, dimensions)
            return type(
                "Report",
                (),
                {
                    "overall_score": self._score,
                    "risk_level": "medium",
                    "total_risks": 1,
                    "total_benefits": 1,
                    "chunks_analyzed": 1,
                    "recommendations": [],
                    "evidence_warnings": [],
                    "dimensions": {},
                },
            )()

    def _fake_build_bid_analyzer(conn, backend):
        _ = conn
        return _ByBackendAnalyzer(100.0 if backend == "agent-mcp" else 0.0)

    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.build_bid_analyzer",
        _fake_build_bid_analyzer,
    )
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.OpenAIMcpAgentExecutor",
        _AgentExecutor,
    )
    provider = build_scoring_provider(
        "hybrid",
        conn=object(),
        hybrid_primary_weight=0.9,
    )
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert result.overall_score == 82.8


def test_scoring_factory_hybrid_reads_weight_from_env(monkeypatch):
    class _ByBackendAnalyzer:
        def __init__(self, score: float):
            self._score = score

        def analyze_version(
            self,
            *,
            version_id: str,
            bidder_name: str,
            project_name: str,
            dimensions: list[str] | None = None,
        ):
            _ = (version_id, bidder_name, project_name, dimensions)
            return type(
                "Report",
                (),
                {
                    "overall_score": self._score,
                    "risk_level": "medium",
                    "total_risks": 1,
                    "total_benefits": 1,
                    "chunks_analyzed": 1,
                    "recommendations": [],
                    "evidence_warnings": [],
                    "dimensions": {},
                },
            )()

    monkeypatch.setenv("BID_SCORING_HYBRID_PRIMARY_WEIGHT", "0.8")
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.build_bid_analyzer",
        lambda conn, backend: _ByBackendAnalyzer(
            100.0 if backend == "agent-mcp" else 0.0
        ),
    )
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.OpenAIMcpAgentExecutor",
        _AgentExecutor,
    )
    provider = build_scoring_provider("hybrid", conn=object())
    result = provider.score(
        ScoringRequest(
            version_id="33333333-3333-3333-3333-333333333333",
            bidder_name="A公司",
            project_name="示例项目",
        )
    )
    assert result.overall_score == 73.6


def test_scoring_factory_uses_analyzer_fallback_when_agent_disabled(monkeypatch):
    captured_backends: list[str] = []

    def _fake_build_bid_analyzer(conn, backend):
        _ = conn
        captured_backends.append(backend)
        return _Analyzer()

    monkeypatch.setenv("BID_SCORING_AGENT_MCP_DISABLE", "1")
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.build_bid_analyzer",
        _fake_build_bid_analyzer,
    )
    monkeypatch.setattr(
        "bid_scoring.pipeline.application.scoring_factory.OpenAIMcpAgentExecutor",
        _AgentExecutor,
    )

    _ = build_scoring_provider("agent-mcp", conn=object())
    _ = build_scoring_provider("hybrid", conn=object())

    assert "agent-mcp" not in captured_backends
    assert captured_backends.count("analyzer") >= 2
