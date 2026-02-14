from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from bid_scoring.config import load_settings

from .scoring_agent_support import (
    normalize_agent_result,
    read_int_env,
    resolve_dimensions,
)
from .scoring_agent_tool_loop import run_tool_calling_loop
from .scoring_common import merge_unique_warnings
from .scoring_types import (
    AgentMcpExecutor,
    ScoringProvider,
    ScoringRequest,
    ScoringResult,
)

logger = logging.getLogger(__name__)


class AgentMcpScoringProvider:
    """Run agent scoring first, fallback to baseline scoring on failure."""

    def __init__(
        self,
        *,
        executor: AgentMcpExecutor,
        fallback: ScoringProvider,
        fallback_warning_code: str = "scoring_backend_agent_mcp_fallback",
    ) -> None:
        self._executor = executor
        self._fallback = fallback
        self._fallback_warning_code = fallback_warning_code

    def score(self, request: ScoringRequest) -> ScoringResult:
        try:
            return self._executor.score(request)
        except Exception as exc:
            logger.warning("Agent/MCP scoring failed, fallback to baseline: %s", exc)
            fallback = self._fallback.score(request)
            warnings = merge_unique_warnings(
                fallback.warnings,
                [self._fallback_warning_code],
            )
            return ScoringResult(
                status=fallback.status,
                overall_score=fallback.overall_score,
                risk_level=fallback.risk_level,
                total_risks=fallback.total_risks,
                total_benefits=fallback.total_benefits,
                chunks_analyzed=fallback.chunks_analyzed,
                recommendations=list(fallback.recommendations),
                evidence_warnings=list(fallback.evidence_warnings),
                evidence_citations={
                    key: list(value)
                    for key, value in fallback.evidence_citations.items()
                },
                dimensions=dict(fallback.dimensions),
                warnings=warnings,
            )


class OpenAIMcpAgentExecutor:
    """Score bids with LLM over retrieval MCP evidence."""

    def __init__(
        self,
        *,
        retrieve_fn: Any | None = None,
        client: OpenAI | Any | None = None,
        model: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        max_chars: int | None = None,
        execution_mode: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        self._retrieve_fn = retrieve_fn or _default_retrieve_fn
        self._client = client
        self._model = model or os.getenv("BID_SCORING_AGENT_MCP_MODEL", "gpt-5-mini")

        resolved_top_k = top_k
        if resolved_top_k is None:
            resolved_top_k = read_int_env("BID_SCORING_AGENT_MCP_TOP_K", 8)
        self._top_k = max(1, int(resolved_top_k))

        self._mode = mode or os.getenv("BID_SCORING_AGENT_MCP_MODE", "hybrid")

        resolved_max_chars = max_chars
        if resolved_max_chars is None:
            resolved_max_chars = read_int_env("BID_SCORING_AGENT_MCP_MAX_CHARS", 320)
        self._max_chars = max(64, int(resolved_max_chars))

        self._execution_mode = _resolve_execution_mode(execution_mode)

        resolved_max_turns = max_turns
        if resolved_max_turns is None:
            resolved_max_turns = read_int_env("BID_SCORING_AGENT_MCP_MAX_TURNS", 8)
        self._max_turns = max(1, int(resolved_max_turns))

    def score(self, request: ScoringRequest) -> ScoringResult:
        if _is_agent_mcp_disabled():
            raise RuntimeError("agent_mcp_disabled")

        dimensions = resolve_dimensions(
            request.dimensions,
            keyword_overrides=(
                request.question_context.keywords_by_dimension
                if request.question_context is not None
                else None
            ),
        )
        if not dimensions:
            raise RuntimeError("No valid scoring dimensions")

        if self._execution_mode == "bulk":
            agent_json, evidence_payload, dimension_warning_map, evidence_warnings = (
                self._run_bulk_mode(request=request, dimensions=dimensions)
            )
        else:
            agent_json, evidence_payload, dimension_warning_map, evidence_warnings = (
                self._run_tool_calling_mode(request=request, dimensions=dimensions)
            )

        return normalize_agent_result(
            agent_json=agent_json,
            dimensions=dimensions,
            evidence_payload=evidence_payload,
            dimension_warning_map=dimension_warning_map,
            evidence_warnings=evidence_warnings,
        )

    def _run_bulk_mode(
        self,
        *,
        request: ScoringRequest,
        dimensions: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, list[dict[str, Any]]],
        dict[str, list[str]],
        list[str],
    ]:
        evidence_payload: dict[str, list[dict[str, Any]]] = {}
        evidence_warnings: list[str] = []
        dimension_warning_map: dict[str, list[str]] = {}

        for dim_key, dim in dimensions.items():
            response = self._retrieve_fn(
                version_id=request.version_id,
                query=" ".join(dim.keywords),
                top_k=self._top_k,
                mode=self._mode,
                keywords=dim.keywords,
                include_text=True,
                max_chars=self._max_chars,
                include_diagnostics=False,
            )
            merged_dimension_warnings = merge_unique_warnings(
                list(response.get("warnings", [])),
                [],
            )
            raw_results = response.get("results", [])
            verifiable_items: list[dict[str, Any]] = []
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                merged_dimension_warnings = merge_unique_warnings(
                    merged_dimension_warnings,
                    list(item.get("warnings", [])),
                )
                if item.get("evidence_status") not in {
                    "verified",
                    "verified_with_warnings",
                }:
                    merged_dimension_warnings = merge_unique_warnings(
                        merged_dimension_warnings,
                        ["unverifiable_evidence_for_scoring"],
                    )
                    if item.get("bbox") is None:
                        merged_dimension_warnings = merge_unique_warnings(
                            merged_dimension_warnings,
                            ["missing_bbox"],
                        )
                    continue
                bbox = item.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    merged_dimension_warnings = merge_unique_warnings(
                        merged_dimension_warnings,
                        ["missing_bbox"],
                    )
                    continue
                verifiable_items.append(
                    {
                        "chunk_id": item.get("chunk_id"),
                        "page_idx": item.get("page_idx"),
                        "bbox": bbox,
                        "text": str(item.get("text", "")),
                    }
                )

            evidence_payload[dim_key] = verifiable_items
            dimension_warning_map[dim_key] = merged_dimension_warnings
            evidence_warnings = merge_unique_warnings(
                evidence_warnings,
                merged_dimension_warnings,
            )

        agent_json = self._call_agent_bulk(
            request=request,
            dimensions=dimensions,
            evidence_payload=evidence_payload,
        )
        return agent_json, evidence_payload, dimension_warning_map, evidence_warnings

    def _run_tool_calling_mode(
        self,
        *,
        request: ScoringRequest,
        dimensions: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        dict[str, list[dict[str, Any]]],
        dict[str, list[str]],
        list[str],
    ]:
        client = self._client or _build_openai_client()
        payload = {
            "version_id": request.version_id,
            "bidder_name": request.bidder_name,
            "project_name": request.project_name,
            "dimensions": {
                key: {
                    "display_name": dim.display_name,
                    "keywords": dim.keywords,
                }
                for key, dim in dimensions.items()
            },
        }
        loop_result = run_tool_calling_loop(
            client=client,
            model=self._model,
            request_payload=payload,
            dimensions=dimensions,
            retrieve_fn=self._retrieve_fn,
            default_top_k=self._top_k,
            default_mode=self._mode,
            max_chars=self._max_chars,
            max_turns=self._max_turns,
        )
        return (
            loop_result.agent_json,
            loop_result.evidence_payload,
            loop_result.dimension_warning_map,
            loop_result.evidence_warnings,
        )

    def _call_agent_bulk(
        self,
        *,
        request: ScoringRequest,
        dimensions: dict[str, Any],
        evidence_payload: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        client = self._client or _build_openai_client()
        payload = {
            "version_id": request.version_id,
            "bidder_name": request.bidder_name,
            "project_name": request.project_name,
            "dimensions": {
                key: {
                    "display_name": dim.display_name,
                    "keywords": dim.keywords,
                    "evidence": evidence_payload.get(key, []),
                }
                for key, dim in dimensions.items()
            },
        }
        response = client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是评标专家。必须仅基于给定证据评分，禁止使用外部知识和杜撰。"
                        "若证据不足，请明确说明“证据不足”。"
                        "输出严格 JSON："
                        "{overall_score,risk_level,total_risks,total_benefits,"
                        "recommendations,dimensions}。"
                        "dimensions 是对象，key 为维度名，"
                        "value 含 score/risk_level/summary。"
                        "risk_level 仅允许 low/medium/high。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
        )
        content = _extract_message_content(response)
        if not content:
            raise RuntimeError("agent_mcp_empty_response")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("agent_mcp_invalid_json") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("agent_mcp_invalid_payload")
        return parsed


def _build_openai_client() -> OpenAI:
    settings = load_settings()
    api_key = settings.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for agent-mcp scoring")
    timeout = settings.get("OPENAI_TIMEOUT") or 30
    max_retries = settings.get("OPENAI_MAX_RETRIES") or 2
    return OpenAI(
        api_key=api_key,
        base_url=settings.get("OPENAI_BASE_URL"),
        timeout=timeout,
        max_retries=max_retries,
    )


def _default_retrieve_fn(**kwargs: Any) -> dict[str, Any]:
    from mcp_servers.retrieval_server import retrieve_impl

    return retrieve_impl(**kwargs)


def _is_agent_mcp_disabled() -> bool:
    value = (os.getenv("BID_SCORING_AGENT_MCP_DISABLE") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_execution_mode(explicit_mode: str | None) -> str:
    raw = explicit_mode or os.getenv("BID_SCORING_AGENT_MCP_EXECUTION_MODE")
    mode = (raw or "tool-calling").strip().lower()
    if mode in {"tool-calling", "bulk"}:
        return mode
    logger.warning(
        "Unknown BID_SCORING_AGENT_MCP_EXECUTION_MODE='%s', fallback='tool-calling'",
        raw,
    )
    return "tool-calling"


def _extract_message_content(response: Any) -> str:
    from .scoring_agent_support import extract_message_content

    return extract_message_content(response)
