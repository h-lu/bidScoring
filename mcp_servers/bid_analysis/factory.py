from __future__ import annotations

from typing import Any

from mcp_servers.bid_analysis.insight import InsightExtractor
from mcp_servers.bid_analysis.recommendation import RecommendationGenerator
from mcp_servers.bid_analysis.retriever import ChunkRetriever, McpChunkRetriever
from mcp_servers.bid_analysis.scorer import DimensionScorer
from mcp_servers.bid_analyzer import BidAnalyzer


def build_bid_analyzer(conn: Any, backend: str = "analyzer") -> BidAnalyzer:
    """Build BidAnalyzer with explicit internal component wiring.

    backend is reserved for future alternative component graphs.
    """
    retriever = ChunkRetriever(conn)
    if backend in {"agent-mcp", "hybrid"}:
        retriever = McpChunkRetriever()

    return BidAnalyzer(
        conn=conn,
        retriever=retriever,
        insight_extractor=InsightExtractor(),
        dimension_scorer=DimensionScorer(),
        recommendation_generator=RecommendationGenerator(),
    )
