from __future__ import annotations

import pytest

from bid_scoring.retrieval.types import RetrievalResult


def test_validation_module_preserves_query_contract():
    from mcp_servers.retrieval.validation import ValidationError, validate_query

    assert validate_query("投标报价") == "投标报价"
    with pytest.raises(ValidationError):
        validate_query("")


def test_execution_module_returns_tool_result_on_error():
    from mcp_servers.retrieval.execution import ToolResult, tool_wrapper

    @tool_wrapper("explode")
    def _explode():
        raise RuntimeError("boom")

    result = _explode()
    assert isinstance(result, ToolResult)
    assert result.success is False
    assert result.error == "explode failed: boom"


def test_formatting_module_emits_warning_for_missing_evidence():
    from mcp_servers.retrieval.formatting import format_result

    result = RetrievalResult(
        chunk_id="chunk-legacy",
        text="legacy",
        page_idx=1,
        score=0.5,
        source="hybrid",
        evidence_units=[],
    )

    formatted = format_result(result, include_text=True, max_chars=None)
    assert formatted["warnings"] == ["missing_evidence_chain"]
