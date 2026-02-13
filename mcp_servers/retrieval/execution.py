"""Execution wrapper and metrics for retrieval MCP tools/resources."""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, TypeVar

logger = logging.getLogger("bid_scoring.mcp")


@dataclass
class ToolResult:
    """Standardized tool execution result."""

    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolMetrics:
    """Tool execution metrics for monitoring."""

    call_count: int = 0
    total_execution_time_ms: float = 0.0
    error_count: int = 0

    @property
    def avg_execution_time_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_execution_time_ms / self.call_count


_tool_metrics: Dict[str, ToolMetrics] = {}
T = TypeVar("T")


def tool_wrapper(tool_name: str) -> Callable:
    """Decorate tools/resources with metrics and standardized error handling."""

    def decorator(func: Callable[..., T]) -> Callable[..., ToolResult | T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> ToolResult | T:
            if tool_name not in _tool_metrics:
                _tool_metrics[tool_name] = ToolMetrics()

            metrics = _tool_metrics[tool_name]
            metrics.call_count += 1
            start_time = time.time()

            logger.info(
                f"Executing tool: {tool_name}",
                extra={
                    "tool_name": tool_name,
                    "parameters": _sanitize_parameters(kwargs),
                },
            )

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                metrics.total_execution_time_ms += execution_time

                logger.info(
                    f"Tool execution completed: {tool_name}",
                    extra={
                        "tool_name": tool_name,
                        "execution_time_ms": execution_time,
                        "success": True,
                    },
                )

                if isinstance(result, ToolResult):
                    result.execution_time_ms = execution_time
                    return result
                return result

            except Exception as exc:  # pragma: no cover - exercised via contract tests
                execution_time = (time.time() - start_time) * 1000
                metrics.total_execution_time_ms += execution_time
                metrics.error_count += 1

                logger.error(
                    f"Tool execution failed: {tool_name}",
                    extra={
                        "tool_name": tool_name,
                        "execution_time_ms": execution_time,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )

                return ToolResult(
                    success=False,
                    error=f"{tool_name} failed: {exc}",
                    execution_time_ms=execution_time,
                    metadata={"error_type": type(exc).__name__},
                )

        return wrapper

    return decorator


def _sanitize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_keys = {"password", "token", "secret", "api_key", "key"}
    sanitized = {}
    for key, value in params.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = value
    return sanitized
