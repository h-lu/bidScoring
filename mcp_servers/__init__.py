"""MCP server entrypoints for this repository."""

from mcp_servers.annotation_insights import (
    AnnotationInsight,
    analyze_chunk_for_insights,
    generate_annotation_content,
)
from mcp_servers.bid_analyzer import (
    ANALYSIS_DIMENSIONS,
    BidAnalyzer,
    BidAnalysisReport,
    DimensionResult,
    compare_versions,
)
from mcp_servers.pdf_annotator import (
    PDFAnnotator,
    HighlightResult,
    HighlightRequest,
    parse_color,
    TOPIC_COLORS,
)

__all__ = [
    # Annotation insights
    "AnnotationInsight",
    "analyze_chunk_for_insights",
    "generate_annotation_content",
    # Bid analyzer
    "ANALYSIS_DIMENSIONS",
    "BidAnalyzer",
    "BidAnalysisReport",
    "DimensionResult",
    "compare_versions",
    # PDF annotator
    "PDFAnnotator",
    "HighlightResult",
    "HighlightRequest",
    "parse_color",
    "TOPIC_COLORS",
]
