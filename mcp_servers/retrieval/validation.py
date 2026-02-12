"""Validation helpers for retrieval MCP server inputs."""

from __future__ import annotations


class ValidationError(ValueError):
    """Raised when input validation fails."""


def validate_version_id(version_id: str | None) -> str:
    """Validate version identifier."""
    if not version_id:
        raise ValidationError("version_id is required and cannot be empty")
    if not isinstance(version_id, str):
        raise ValidationError("version_id must be a string")
    return version_id


def validate_chunk_id(chunk_id: str | None) -> str:
    """Validate chunk identifier."""
    if not chunk_id:
        raise ValidationError("chunk_id is required and cannot be empty")
    if not isinstance(chunk_id, str):
        raise ValidationError("chunk_id must be a string")
    return chunk_id


def validate_unit_id(unit_id: str | None) -> str:
    """Validate evidence unit identifier."""
    if not unit_id:
        raise ValidationError("unit_id is required and cannot be empty")
    if not isinstance(unit_id, str):
        raise ValidationError("unit_id must be a string")
    return unit_id


def validate_positive_int(value: int, name: str, max_value: int = 1000) -> int:
    """Validate positive integer parameter."""
    if not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    if value <= 0:
        raise ValidationError(f"{name} must be positive")
    if value > max_value:
        raise ValidationError(f"{name} exceeds maximum allowed value of {max_value}")
    return value


def validate_query(query: str | None) -> str:
    """Validate search query."""
    if not query:
        raise ValidationError("query is required and cannot be empty")
    if not isinstance(query, str):
        raise ValidationError("query must be a string")
    if len(query) > 10000:
        raise ValidationError("query exceeds maximum length of 10000 characters")
    return query


def validate_string_list(
    value: list, name: str, min_items: int = 1, max_items: int = 100
) -> list:
    """Validate list of strings."""
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list")
    if len(value) < min_items:
        raise ValidationError(f"{name} must have at least {min_items} item(s)")
    if len(value) > max_items:
        raise ValidationError(f"{name} exceeds maximum of {max_items} items")
    return value


def validate_bool(value: bool, name: str) -> bool:
    """Validate boolean parameter."""
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a boolean")
    return value
