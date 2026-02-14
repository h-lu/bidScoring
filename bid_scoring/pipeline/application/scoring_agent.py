from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from bid_scoring.config import load_settings

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
    """Score bids with LLM over evidence retrieved via retrieval MCP."""

    def __init__(
        self,
        *,
        retrieve_fn: Any | None = None,
        client: OpenAI | Any | None = None,
        model: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        max_chars: int | None = None,
    ) -> None:
        self._retrieve_fn = retrieve_fn or _default_retrieve_fn
        self._client = client
        self._model = model or os.getenv("BID_SCORING_AGENT_MCP_MODEL", "gpt-4o-mini")
        resolved_top_k = top_k
        if resolved_top_k is None:
            resolved_top_k = _read_int_env("BID_SCORING_AGENT_MCP_TOP_K", 8)
        self._top_k = max(1, int(resolved_top_k))
        self._mode = mode or os.getenv("BID_SCORING_AGENT_MCP_MODE", "hybrid")
        resolved_max_chars = max_chars
        if resolved_max_chars is None:
            resolved_max_chars = _read_int_env("BID_SCORING_AGENT_MCP_MAX_CHARS", 320)
        self._max_chars = max(64, int(resolved_max_chars))

    def score(self, request: ScoringRequest) -> ScoringResult:
        dimensions = _resolve_dimensions(request.dimensions)
        if not dimensions:
            raise RuntimeError("No valid scoring dimensions")

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
                item_warnings = list(item.get("warnings", []))
                merged_dimension_warnings = merge_unique_warnings(
                    merged_dimension_warnings,
                    item_warnings,
                )

                if not _is_verifiable_item(item):
                    merged_dimension_warnings = merge_unique_warnings(
                        merged_dimension_warnings,
                        ["unverifiable_evidence_for_scoring"],
                    )
                    if item.get("bbox") is None:
                        merged_dimension_warnings = merge_unique_warnings(
                            merged_dimension_warnings, ["missing_bbox"]
                        )
                    continue

                verifiable_items.append(
                    {
                        "chunk_id": item.get("chunk_id"),
                        "page_idx": item.get("page_idx"),
                        "bbox": item.get("bbox"),
                        "text": str(item.get("text", "")),
                    }
                )

            evidence_payload[dim_key] = verifiable_items
            dimension_warning_map[dim_key] = merged_dimension_warnings
            evidence_warnings = merge_unique_warnings(
                evidence_warnings,
                merged_dimension_warnings,
            )

        agent_json = self._call_agent(
            request=request,
            dimensions=dimensions,
            evidence_payload=evidence_payload,
        )
        return _normalize_agent_result(
            agent_json=agent_json,
            dimensions=dimensions,
            evidence_payload=evidence_payload,
            dimension_warning_map=dimension_warning_map,
            evidence_warnings=evidence_warnings,
        )

    def _call_agent(
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
                        "你是评标专家。必须仅基于给定证据评分；"
                        "禁止杜撰。输出严格 JSON："
                        "{overall_score,risk_level,total_risks,total_benefits,"
                        "recommendations,dimensions}。"
                        "dimensions 是对象，key 为维度名，value 含 score/risk_level/summary。"
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


def _resolve_dimensions(
    selected: list[str] | None,
) -> dict[str, Any]:
    from mcp_servers.bid_analysis.models import ANALYSIS_DIMENSIONS

    if selected is None:
        return dict(ANALYSIS_DIMENSIONS)
    resolved: dict[str, Any] = {}
    for name in selected:
        dim = ANALYSIS_DIMENSIONS.get(name)
        if dim is None:
            continue
        resolved[name] = dim
    return resolved


def _is_verifiable_item(item: dict[str, Any]) -> bool:
    if item.get("evidence_status") not in {"verified", "verified_with_warnings"}:
        return False
    bbox = item.get("bbox")
    return isinstance(bbox, list) and len(bbox) == 4


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


def _extract_message_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text"):
                text_val = getattr(item, "text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "".join(parts)
    return ""


def _normalize_agent_result(
    *,
    agent_json: dict[str, Any],
    dimensions: dict[str, Any],
    evidence_payload: dict[str, list[dict[str, Any]]],
    dimension_warning_map: dict[str, list[str]],
    evidence_warnings: list[str],
) -> ScoringResult:
    raw_dimensions = agent_json.get("dimensions")
    raw_dimensions = raw_dimensions if isinstance(raw_dimensions, dict) else {}
    normalized_dimensions: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    evidence_citations: dict[str, list[dict[str, Any]]] = {}

    for dim_key in dimensions:
        dim_payload = raw_dimensions.get(dim_key)
        if not isinstance(dim_payload, dict):
            warnings = merge_unique_warnings(
                warnings, [f"agent_mcp_dimension_missing:{dim_key}"]
            )
            dim_payload = {}
        citations = [
            {
                "chunk_id": item.get("chunk_id"),
                "page_idx": item.get("page_idx"),
                "bbox": item.get("bbox"),
            }
            for item in evidence_payload.get(dim_key, [])
        ]
        evidence_citations[dim_key] = citations
        normalized_dimensions[dim_key] = {
            "score": _safe_float(dim_payload.get("score"), default=50.0),
            "risk_level": _safe_risk_level(dim_payload.get("risk_level")),
            "chunks_found": len(evidence_payload.get(dim_key, [])),
            "summary": str(dim_payload.get("summary", "")),
            "evidence_warnings": list(dimension_warning_map.get(dim_key, [])),
            "evidence_citations": citations,
        }

    return ScoringResult(
        status="completed",
        overall_score=_safe_float(agent_json.get("overall_score"), default=50.0),
        risk_level=_safe_risk_level(agent_json.get("risk_level")),
        total_risks=_safe_int(agent_json.get("total_risks"), default=0),
        total_benefits=_safe_int(agent_json.get("total_benefits"), default=0),
        chunks_analyzed=sum(len(items) for items in evidence_payload.values()),
        recommendations=_safe_str_list(agent_json.get("recommendations")),
        evidence_warnings=evidence_warnings,
        evidence_citations=evidence_citations,
        dimensions=normalized_dimensions,
        warnings=warnings,
    )


def _safe_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(0.0, min(100.0, parsed))


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_risk_level(value: Any) -> str:
    if isinstance(value, str) and value in {"low", "medium", "high"}:
        return value
    return "medium"


def _safe_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            output.append(item)
    return output


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return default
    return parsed
