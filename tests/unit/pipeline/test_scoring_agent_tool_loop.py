from __future__ import annotations

import json

from bid_scoring.pipeline.application.scoring_provider import OpenAIMcpAgentExecutor
from bid_scoring.pipeline.application.scoring_types import ScoringRequest


class _Message:
    def __init__(self, content: str | None = None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Response:
    def __init__(self, message: _Message):
        self.choices = [_Choice(message)]


class _ToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = _ToolFunction(name=name, arguments=arguments)


class _Completions:
    def __init__(self, responses: list[_Response]):
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected extra completion call")
        return self._responses.pop(0)


class _Chat:
    def __init__(self, completions: _Completions):
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions):
        self.chat = _Chat(completions)


def _request() -> ScoringRequest:
    return ScoringRequest(
        version_id="33333333-3333-3333-3333-333333333333",
        bidder_name="A公司",
        project_name="示例项目",
        dimensions=["warranty"],
    )


def test_openai_mcp_agent_executor_defaults_to_tool_calling(monkeypatch):
    monkeypatch.delenv("BID_SCORING_AGENT_MCP_EXECUTION_MODE", raising=False)

    completions = _Completions(
        responses=[
            _Response(
                _Message(
                    json.dumps(
                        {
                            "overall_score": 90,
                            "risk_level": "low",
                            "total_risks": 0,
                            "total_benefits": 1,
                            "recommendations": [],
                            "dimensions": {
                                "warranty": {
                                    "score": 90,
                                    "risk_level": "low",
                                    "summary": "模型直接输出",
                                }
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
        ]
    )

    executor = OpenAIMcpAgentExecutor(
        retrieve_fn=lambda **kwargs: {"warnings": [], "results": []},
        client=_Client(completions),
        model="test-model",
    )

    result = executor.score(_request())

    assert completions.calls[0]["tools"][0]["function"]["name"] == "retrieve_dimension_evidence"
    assert "agent_mcp_dimension_no_verifiable_evidence:warranty" in result.warnings
    assert result.dimensions["warranty"]["score"] == 50.0


def test_openai_mcp_agent_executor_tool_loop_collects_evidence():
    tool_call = _ToolCall(
        call_id="call_1",
        name="retrieve_dimension_evidence",
        arguments=json.dumps(
            {
                "dimension": "warranty",
                "query": "质保期 响应时间",
                "top_k": 2,
                "mode": "hybrid",
                "keywords": ["质保", "响应"],
            },
            ensure_ascii=False,
        ),
    )
    completions = _Completions(
        responses=[
            _Response(_Message(content="", tool_calls=[tool_call])),
            _Response(
                _Message(
                    json.dumps(
                        {
                            "overall_score": 88,
                            "risk_level": "low",
                            "total_risks": 1,
                            "total_benefits": 3,
                            "recommendations": ["保持条款一致性"],
                            "dimensions": {
                                "warranty": {
                                    "score": 88,
                                    "risk_level": "low",
                                    "summary": "质保条款明确",
                                }
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            ),
        ]
    )

    captured_retrieve: list[dict[str, object]] = []

    def _fake_retrieve(**kwargs):
        captured_retrieve.append(kwargs)
        return {
            "warnings": ["missing_evidence_chain"],
            "results": [
                {
                    "chunk_id": "chunk-1",
                    "page_idx": 8,
                    "bbox": [1, 2, 3, 4],
                    "text": "原厂免费质保5年",
                    "evidence_status": "verified",
                    "warnings": [],
                }
            ],
        }

    executor = OpenAIMcpAgentExecutor(
        retrieve_fn=_fake_retrieve,
        client=_Client(completions),
        model="test-model",
        execution_mode="tool-calling",
        max_turns=4,
    )

    result = executor.score(_request())

    assert len(captured_retrieve) == 1
    assert captured_retrieve[0]["query"] == "质保期 响应时间"
    assert captured_retrieve[0]["top_k"] == 2
    assert result.overall_score == 88.0
    assert result.dimensions["warranty"]["score"] == 88.0
    assert result.chunks_analyzed == 1
    assert result.evidence_citations["warranty"][0]["chunk_id"] == "chunk-1"
    assert "agent_mcp_dimension_no_verifiable_evidence:warranty" not in result.warnings
    assert any(
        isinstance(msg, dict) and msg.get("role") == "tool"
        for msg in completions.calls[1]["messages"]
    )
