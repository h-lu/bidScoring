"""Application layer for the evidence-first pipeline."""

from .e2e_service import (
    E2EPipelineService,
    E2ERunRequest,
    E2ERunResult,
    LoadedContent,
)
from .scoring_factory import build_scoring_provider
from .scoring_provider import (
    AgentMcpScoringProvider,
    BidAnalyzerScoringProvider,
    HybridScoringProvider,
    OpenAIMcpAgentExecutor,
    ScoringProvider,
    ScoringRequest,
    ScoringResult,
    WarningFallbackScoringProvider,
)
from .service import (
    CitationEvaluationSummary,
    IngestSummary,
    PipelineRepository,
    PipelineService,
)

__all__ = [
    "E2EPipelineService",
    "E2ERunRequest",
    "E2ERunResult",
    "LoadedContent",
    "build_scoring_provider",
    "AgentMcpScoringProvider",
    "BidAnalyzerScoringProvider",
    "HybridScoringProvider",
    "OpenAIMcpAgentExecutor",
    "ScoringProvider",
    "ScoringRequest",
    "ScoringResult",
    "WarningFallbackScoringProvider",
    "CitationEvaluationSummary",
    "IngestSummary",
    "PipelineRepository",
    "PipelineService",
]
