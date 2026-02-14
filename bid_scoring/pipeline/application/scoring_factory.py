from __future__ import annotations

import logging
import os
from typing import Any

from mcp_servers.bid_analysis.factory import build_bid_analyzer

from .scoring_provider import (
    AgentMcpScoringProvider,
    BidAnalyzerScoringProvider,
    HybridScoringProvider,
    OpenAIMcpAgentExecutor,
    ScoringProvider,
    WarningFallbackScoringProvider,
)

logger = logging.getLogger(__name__)
_DEFAULT_HYBRID_PRIMARY_WEIGHT = 0.7
_AGENT_DISABLE_ENV = "BID_SCORING_AGENT_MCP_DISABLE"


def build_scoring_provider(
    backend: str,
    conn: Any,
    *,
    hybrid_primary_weight: float | None = None,
) -> ScoringProvider:
    """Create scoring provider by backend name."""
    if backend == "analyzer":
        analyzer = build_bid_analyzer(conn, backend="analyzer")
        baseline = BidAnalyzerScoringProvider(analyzer=analyzer)
        return baseline
    if backend == "agent-mcp":
        analyzer_backend = "analyzer" if _is_agent_mcp_disabled() else "agent-mcp"
        analyzer = build_bid_analyzer(conn, backend=analyzer_backend)
        baseline = BidAnalyzerScoringProvider(analyzer=analyzer)
        return AgentMcpScoringProvider(
            executor=OpenAIMcpAgentExecutor(),
            fallback=baseline,
        )
    if backend == "hybrid":
        baseline_analyzer = build_bid_analyzer(conn, backend="analyzer")
        if _is_agent_mcp_disabled():
            mcp_analyzer = baseline_analyzer
        else:
            mcp_analyzer = build_bid_analyzer(conn, backend="agent-mcp")
        resolved_weight = _resolve_hybrid_weight(hybrid_primary_weight)
        return HybridScoringProvider(
            primary=AgentMcpScoringProvider(
                executor=OpenAIMcpAgentExecutor(),
                fallback=BidAnalyzerScoringProvider(analyzer=mcp_analyzer),
            ),
            secondary=BidAnalyzerScoringProvider(analyzer=baseline_analyzer),
            primary_weight=resolved_weight,
        )

    analyzer = build_bid_analyzer(conn, backend="analyzer")
    baseline = BidAnalyzerScoringProvider(analyzer=analyzer)
    return WarningFallbackScoringProvider(
        fallback=baseline,
        warning_codes=["scoring_backend_unknown"],
    )


def _resolve_hybrid_weight(explicit_weight: float | None) -> float:
    if explicit_weight is not None:
        return _validate_hybrid_weight(explicit_weight)

    env_value = os.getenv("BID_SCORING_HYBRID_PRIMARY_WEIGHT")
    if env_value is None or env_value == "":
        return _DEFAULT_HYBRID_PRIMARY_WEIGHT

    try:
        parsed = float(env_value)
    except ValueError:
        logger.warning(
            "Invalid BID_SCORING_HYBRID_PRIMARY_WEIGHT='%s', fallback=%s",
            env_value,
            _DEFAULT_HYBRID_PRIMARY_WEIGHT,
        )
        return _DEFAULT_HYBRID_PRIMARY_WEIGHT

    return _validate_hybrid_weight(parsed)


def _validate_hybrid_weight(weight: float) -> float:
    if 0.0 <= weight <= 1.0:
        return float(weight)
    logger.warning(
        "Out-of-range hybrid weight=%s, fallback=%s",
        weight,
        _DEFAULT_HYBRID_PRIMARY_WEIGHT,
    )
    return _DEFAULT_HYBRID_PRIMARY_WEIGHT


def _is_agent_mcp_disabled() -> bool:
    value = (os.getenv(_AGENT_DISABLE_ENV) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}
