from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .scoring_common import merge_unique_warnings
from .scoring_agent_support import (
    extract_message_content,
    is_verifiable_item,
    safe_int,
    sanitize_keywords,
)

_RETRIEVE_TOOL_NAME = "retrieve_dimension_evidence"


@dataclass(frozen=True)
class ToolLoopResult:
    agent_json: dict[str, Any]
    evidence_payload: dict[str, list[dict[str, Any]]]
    dimension_warning_map: dict[str, list[str]]
    evidence_warnings: list[str]
    turns: int
    tool_call_count: int
    tool_names: list[str]


def run_tool_calling_loop(
    *,
    client: Any,
    model: str,
    request_payload: dict[str, Any],
    dimensions: dict[str, Any],
    retrieve_fn: Callable[..., dict[str, Any]],
    default_top_k: int,
    default_mode: str,
    max_chars: int,
    max_turns: int,
    system_prompt: str,
) -> ToolLoopResult:
    evidence_payload: dict[str, list[dict[str, Any]]] = {key: [] for key in dimensions}
    dimension_warning_map: dict[str, list[str]] = {key: [] for key in dimensions}
    evidence_warnings: list[str] = []

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(request_payload, ensure_ascii=False),
        },
    ]

    turns = 0
    tool_call_count = 0
    tool_name_seen: set[str] = set()

    for _ in range(max_turns):
        turns += 1
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            tools=[_build_retrieve_tool_schema()],
            tool_choice="auto",
            messages=messages,
        )
        parsed_tool_calls = _parse_tool_calls_from_response(response)
        assistant_content = extract_message_content(response)

        if parsed_tool_calls:
            tool_call_count += len(parsed_tool_calls)
            for tool_call in parsed_tool_calls:
                function_payload = tool_call.get("function", {})
                tool_name = function_payload.get("name")
                if isinstance(tool_name, str) and tool_name:
                    tool_name_seen.add(tool_name)
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content or "",
                    "tool_calls": parsed_tool_calls,
                }
            )
            for call in parsed_tool_calls:
                tool_output = _execute_retrieve_tool(
                    call=call,
                    dimensions=dimensions,
                    retrieve_fn=retrieve_fn,
                    version_id=str(request_payload.get("version_id", "")),
                    default_top_k=default_top_k,
                    default_mode=default_mode,
                    max_chars=max_chars,
                    evidence_payload=evidence_payload,
                    dimension_warning_map=dimension_warning_map,
                    evidence_warnings=evidence_warnings,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(tool_output, ensure_ascii=False),
                    }
                )
            continue

        if not assistant_content:
            raise RuntimeError("agent_mcp_empty_response")
        try:
            parsed = json.loads(assistant_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("agent_mcp_invalid_json") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("agent_mcp_invalid_payload")
        return ToolLoopResult(
            agent_json=parsed,
            evidence_payload=evidence_payload,
            dimension_warning_map=dimension_warning_map,
            evidence_warnings=evidence_warnings,
            turns=turns,
            tool_call_count=tool_call_count,
            tool_names=sorted(tool_name_seen),
        )

    raise RuntimeError("agent_mcp_tool_loop_max_turns_exceeded")


def _build_retrieve_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _RETRIEVE_TOOL_NAME,
            "description": "按维度检索可定位证据，返回 chunk/page/bbox/text。",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dimension": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "mode": {"type": "string", "enum": ["hybrid", "keyword", "vector"]},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["dimension"],
            },
        },
    }


def _parse_tool_calls_from_response(response: Any) -> list[dict[str, Any]]:
    choices = getattr(response, "choices", None)
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    if message is None:
        return []
    raw_calls = getattr(message, "tool_calls", None) or []

    parsed: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls):
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", "{}")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, str):
            arguments = "{}"
        parsed.append(
            {
                "id": str(getattr(call, "id", f"call_{index + 1}")),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return parsed


def _execute_retrieve_tool(
    *,
    call: dict[str, Any],
    dimensions: dict[str, Any],
    retrieve_fn: Callable[..., dict[str, Any]],
    version_id: str,
    default_top_k: int,
    default_mode: str,
    max_chars: int,
    evidence_payload: dict[str, list[dict[str, Any]]],
    dimension_warning_map: dict[str, list[str]],
    evidence_warnings: list[str],
) -> dict[str, Any]:
    function_payload = call.get("function", {})
    name = function_payload.get("name")
    if name != _RETRIEVE_TOOL_NAME:
        return {"status": "error", "error": "unknown_tool"}

    arguments_raw = function_payload.get("arguments", "{}")
    try:
        args = json.loads(arguments_raw)
    except json.JSONDecodeError:
        return {"status": "error", "error": "invalid_tool_arguments"}

    if not isinstance(args, dict):
        return {"status": "error", "error": "invalid_tool_arguments"}

    dimension = str(args.get("dimension", "")).strip()
    if dimension not in dimensions:
        return {
            "status": "error",
            "error": "unknown_dimension",
            "available_dimensions": list(dimensions.keys()),
        }
    dim_config = dimensions[dimension]

    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        query = " ".join(dim_config.keywords)

    top_k_raw = args.get("top_k")
    top_k = (
        min(max(1, safe_int(top_k_raw, default=default_top_k)), 50)
        if top_k_raw is not None
        else default_top_k
    )

    mode = args.get("mode")
    if mode not in {"hybrid", "keyword", "vector"}:
        mode = default_mode

    keywords = args.get("keywords")
    if isinstance(keywords, list):
        normalized_keywords = sanitize_keywords(keywords)
        if not normalized_keywords:
            normalized_keywords = list(dim_config.keywords)
    else:
        normalized_keywords = list(dim_config.keywords)

    response = retrieve_fn(
        version_id=version_id,
        query=query,
        top_k=top_k,
        mode=mode,
        keywords=normalized_keywords,
        include_text=True,
        max_chars=max_chars,
        include_diagnostics=False,
    )

    merged_dimension_warnings = merge_unique_warnings(
        list(dimension_warning_map.get(dimension, [])),
        list(response.get("warnings", [])),
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
        if not is_verifiable_item(item):
            merged_dimension_warnings = merge_unique_warnings(
                merged_dimension_warnings, ["unverifiable_evidence_for_scoring"]
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

    evidence_payload[dimension] = _merge_unique_items(
        evidence_payload.get(dimension, []),
        verifiable_items,
    )
    dimension_warning_map[dimension] = merged_dimension_warnings
    evidence_warnings[:] = merge_unique_warnings(
        list(evidence_warnings),
        merged_dimension_warnings,
    )

    return {
        "status": "ok",
        "dimension": dimension,
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "warnings": merged_dimension_warnings,
        "evidence_count": len(evidence_payload.get(dimension, [])),
        "evidence": evidence_payload.get(dimension, []),
    }


def _merge_unique_items(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = list(existing)
    seen: set[tuple[Any, Any, Any]] = set()
    for item in output:
        seen.add(_item_key(item))
    for item in additions:
        key = _item_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _item_key(item: dict[str, Any]) -> tuple[Any, Any, Any]:
    bbox = item.get("bbox")
    bbox_token = tuple(bbox) if isinstance(bbox, list) else None
    return (item.get("chunk_id"), item.get("page_idx"), bbox_token)
