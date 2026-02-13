"""Shared building blocks for retrieval MCP server."""

from .execution import ToolMetrics, ToolResult, tool_wrapper
from .formatting import format_evidence_units, format_result
from .validation import (
    ValidationError,
    validate_bool,
    validate_chunk_id,
    validate_positive_int,
    validate_query,
    validate_string_list,
    validate_unit_id,
    validate_version_id,
)

__all__ = [
    "ToolMetrics",
    "ToolResult",
    "tool_wrapper",
    "format_result",
    "format_evidence_units",
    "ValidationError",
    "validate_bool",
    "validate_chunk_id",
    "validate_positive_int",
    "validate_query",
    "validate_string_list",
    "validate_unit_id",
    "validate_version_id",
]
