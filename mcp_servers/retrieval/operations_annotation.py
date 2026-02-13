"""PDF annotation operations for retrieval MCP server."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal

from bid_scoring.config import load_settings
from mcp_servers.retrieval.validation import (
    ValidationError,
    validate_bool,
    validate_positive_int,
    validate_query,
    validate_string_list,
    validate_version_id,
)


def highlight_pdf(
    version_id: str,
    chunk_ids: list[str],
    topic: str,
    color: str | None = None,
    increment: bool = True,
) -> Dict[str, Any]:
    """Add highlights to PDF for specified chunks.

    Creates visually annotated PDFs for bid review by highlighting
    relevant content based on chunk bbox coordinates. Supports cumulative
    layer additions for different analysis topics with color coding.

    Topics are color-coded:
    - risk: Red (liabilities, penalties)
    - warranty: Green (after-sales, guarantees)
    - training: Yellow (training provisions)
    - delivery: Orange (delivery timeline)
    - financial: Blue (payment terms)
    - technical: Purple (technical specs)

    Args:
        version_id: Document version UUID to highlight.
        chunk_ids: List of chunk IDs to highlight from search results.
        topic: Topic name for color coding (e.g., 'warranty', 'training').
        color: Optional color in hex (#RRGGBB) or RGB format (0-1).
               Auto-assigned by topic if None.
        increment: If True, add to existing annotated PDF.
                  If False, create new from original.

    Returns:
        Dict with:
        - success: Whether operation succeeded
        - annotated_url: Presigned URL to annotated PDF (15 min valid)
        - highlights_added: Number of highlights added
        - file_path: MinIO object key for the annotated PDF
        - topics: List of topics in the annotated PDF
        - error: Error message if failed
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)
    chunk_ids = validate_string_list(chunk_ids, "chunk_ids", min_items=1, max_items=500)
    topic = validate_query(topic)  # Use query validation for topic name
    if color is not None:
        if not isinstance(color, str):
            raise ValidationError("color must be a string")

    increment = validate_bool(increment, "increment")

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        from mineru.minio_storage import MinIOStorage
        from mcp_servers.pdf_annotator import PDFAnnotator, parse_color

        # Parse color if provided
        rgb_color = None
        if color:
            rgb_color = parse_color(color)

        # Create annotator
        storage = MinIOStorage()
        annotator = PDFAnnotator(conn, storage)

        # Perform highlighting
        result = annotator.highlight_chunks(
            version_id=version_id,
            chunk_ids=chunk_ids,
            topic=topic,
            color=rgb_color,
            increment=increment,
        )

        if result.success:
            return {
                "success": True,
                "annotated_url": result.annotated_url,
                "highlights_added": result.highlights_added,
                "file_path": result.file_path,
                "topics": result.topics,
                "expires_in_minutes": 15,
            }
        else:
            return {
                "success": False,
                "error": result.error,
            }


def prepare_highlight_targets_from_results(
    results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build safe highlight targets from retrieval results.

    Policy:
    - include items only when `chunk_id` exists, `bbox` exists, and evidence is factual
      (`verified` or `verified_with_warnings`)
    - do not reject execution, return warnings for excluded items
    """
    chunk_ids: list[str] = []
    warnings: set[str] = set()
    seen: set[str] = set()
    excluded = 0

    for item in results:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            warnings.add("missing_chunk_id")
            excluded += 1
            continue

        if item.get("bbox") is None:
            warnings.add("missing_chunk_bbox")
            excluded += 1
            continue

        evidence_status = item.get("evidence_status")
        if evidence_status not in {"verified", "verified_with_warnings"}:
            warnings.add("unverifiable_evidence_for_highlight")
            excluded += 1
            continue

        for code in item.get("warnings", []):
            warnings.add(code)

        if chunk_id not in seen:
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)

    return {
        "chunk_ids": chunk_ids,
        "warnings": sorted(warnings),
        "included_count": len(chunk_ids),
        "excluded_count": excluded,
    }


def prepare_highlight_targets_for_query(
    retrieve_fn: Callable[..., Dict[str, Any]],
    version_id: str,
    query: str,
    top_k: int = 10,
    mode: Literal["hybrid", "vector", "keyword"] = "hybrid",
    keywords: list[str] | None = None,
    use_or_semantic: bool = True,
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Retrieve by query and return factual highlight targets with warnings.

    This operation is warning-only:
    - no hard reject for unverifiable items
    - unsafe candidates are filtered out from highlight targets
    """
    version_id = validate_version_id(version_id)
    query = validate_query(query)
    top_k = validate_positive_int(top_k, "top_k", max_value=100)

    retrieved = retrieve_fn(
        version_id=version_id,
        query=query,
        top_k=top_k,
        mode=mode,
        keywords=keywords,
        use_or_semantic=use_or_semantic,
        include_text=False,
        include_diagnostics=include_diagnostics,
    )

    gate = prepare_highlight_targets_from_results(retrieved.get("results", []))
    merged_warning_codes = sorted(
        set(retrieved.get("warnings", [])) | set(gate.get("warnings", []))
    )

    response: Dict[str, Any] = {
        "version_id": version_id,
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "chunk_ids": gate["chunk_ids"],
        "warnings": merged_warning_codes,
        "included_count": gate["included_count"],
        "excluded_count": gate["excluded_count"],
    }
    if include_diagnostics:
        response["diagnostics"] = {
            "retrieval": retrieved.get("diagnostics"),
            "gate": {
                "included_count": gate["included_count"],
                "excluded_count": gate["excluded_count"],
                "warning_count": len(merged_warning_codes),
            },
        }

    return response
