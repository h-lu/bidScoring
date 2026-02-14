"""Backward-compatible scoring provider exports.

This module remains as a stable import surface while implementation details
are split into SRP-focused modules.
"""

from .scoring_agent import AgentMcpScoringProvider, OpenAIMcpAgentExecutor
from .scoring_baseline import BidAnalyzerScoringProvider, WarningFallbackScoringProvider
from .scoring_hybrid import HybridScoringProvider
from .scoring_types import (
    AgentMcpExecutor,
    ScoringProvider,
    ScoringRequest,
    ScoringResult,
)

__all__ = [
    "AgentMcpExecutor",
    "ScoringProvider",
    "ScoringRequest",
    "ScoringResult",
    "BidAnalyzerScoringProvider",
    "WarningFallbackScoringProvider",
    "AgentMcpScoringProvider",
    "OpenAIMcpAgentExecutor",
    "HybridScoringProvider",
]
