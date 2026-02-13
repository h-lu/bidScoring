"""Evidence and cross-version operations for retrieval MCP server."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Literal

from bid_scoring.config import load_settings
from mcp_servers.retrieval.validation import (
    ValidationError,
    validate_chunk_id,
    validate_positive_int,
    validate_query,
    validate_string_list,
    validate_unit_id,
    validate_version_id,
)


def get_chunk_with_context(
    chunk_id: str,
    context_depth: Literal["chunk", "paragraph", "section", "document"] = "paragraph",
    include_adjacent_pages: bool = False,
) -> Dict[str, Any]:
    """Get a chunk with its surrounding context to avoid out-of-context interpretation.

    Critical for accurate bid analysis - ensures you're not misinterpreting
    a table cell or sentence fragment.

    Args:
        chunk_id: UUID of the chunk to retrieve.
        context_depth: How much context to include.
        include_adjacent_pages: Include content from neighboring pages.

    Returns:
        Chunk content with requested context.
    """
    import psycopg

    # Validate inputs
    chunk_id = validate_chunk_id(chunk_id)

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Get the chunk
            cur.execute(
                """
                SELECT
                    c.version_id, c.page_idx, c.element_type, c.text_raw,
                    c.source_id, c.embedding IS NOT NULL as has_embedding,
                    c.bbox, dp.coord_sys
                FROM chunks c
                LEFT JOIN document_pages dp
                    ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                WHERE c.chunk_id = %s
                """,
                (chunk_id,),
            )

            row = cur.fetchone()
            if not row:
                raise ValidationError(f"Chunk {chunk_id} not found")

            (
                version_id,
                page_idx,
                element_type,
                text_raw,
                source_id,
                has_embedding,
                bbox,
                coord_sys,
            ) = row

            result = {
                "chunk_id": chunk_id,
                "version_id": str(version_id),
                "page_idx": page_idx,
                "element_type": element_type,
                "text": text_raw,
                "source_id": source_id,
                "has_embedding": has_embedding,
                "bbox": bbox if bbox else None,
                "coord_system": coord_sys if coord_sys else "mineru_bbox_v1",
            }

            # Get context based on depth
            if context_depth in ["paragraph", "section", "document"]:
                # Try to find in hierarchical_nodes
                cur.execute(
                    """
                    SELECT parent_id, node_type, content, metadata
                    FROM hierarchical_nodes
                    WHERE version_id = %s
                    AND %s = ANY(source_chunk_ids)
                    """,
                    (version_id, chunk_id),
                )

                hierarchy = cur.fetchall()
                if hierarchy:
                    result["hierarchy"] = [
                        {
                            "parent_id": str(h[0]) if h[0] else None,
                            "node_type": h[1],
                            "content": h[2][:1000] if h[2] else None,
                            "metadata": h[3],
                        }
                        for h in hierarchy
                    ]

            # Get adjacent chunks on same page
            if context_depth in ["paragraph", "section"] and page_idx is not None:
                cur.execute(
                    """
                    SELECT chunk_id, text_raw, element_type, chunk_index
                    FROM chunks
                    WHERE version_id = %s AND page_idx = %s
                    AND chunk_id != %s
                    ORDER BY chunk_index
                    LIMIT 5
                    """,
                    (version_id, page_idx, chunk_id),
                )

                adjacent = cur.fetchall()
                result["same_page_chunks"] = [
                    {
                        "chunk_id": str(r[0]),
                        "text_preview": r[1][:200] if r[1] else None,
                        "element_type": r[2],
                        "chunk_index": r[3],
                    }
                    for r in adjacent
                ]

            # Get adjacent pages
            if include_adjacent_pages and page_idx is not None:
                cur.execute(
                    """
                    SELECT page_idx, text_raw, element_type
                    FROM chunks
                    WHERE version_id = %s AND page_idx IN (%s, %s)
                    AND element_type IN ('title', 'text')
                    ORDER BY page_idx, chunk_index
                    """,
                    (version_id, page_idx - 1, page_idx + 1),
                )

                adjacent_pages = cur.fetchall()
                result["adjacent_pages"] = {}
                for r in adjacent_pages:
                    p_idx = r[0]
                    if p_idx not in result["adjacent_pages"]:
                        result["adjacent_pages"][p_idx] = []
                    result["adjacent_pages"][p_idx].append(
                        {
                            "text_preview": r[1][:200] if r[1] else None,
                            "element_type": r[2],
                        }
                    )

            return result


def get_unit_evidence(
    unit_id: str,
    verify_hash: bool = True,
    include_anchor: bool = True,
) -> Dict[str, Any]:
    """Get precise evidence from content_units (v0.2) for audit-grade verification.

    The most granular level of evidence - use this when you need to
    precisely quote and verify a specific commitment.

    Args:
        unit_id: UUID of the content unit.
        verify_hash: Verify evidence hash for integrity.
        include_anchor: Include coordinate/position info.

    Returns:
        Unit content with verification metadata.
    """
    import psycopg

    # Validate inputs
    unit_id = validate_unit_id(unit_id)

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT version_id, unit_index, unit_type, text_raw, text_norm,
                       char_count, anchor_json, unit_hash, source_element_id
                FROM content_units
                WHERE unit_id = %s
                """,
                (unit_id,),
            )

            row = cur.fetchone()
            if not row:
                raise ValidationError(f"Unit {unit_id} not found")

            (
                version_id,
                unit_index,
                unit_type,
                text_raw,
                text_norm,
                char_count,
                anchor_json,
                unit_hash,
                source_element_id,
            ) = row

            result = {
                "unit_id": unit_id,
                "version_id": str(version_id),
                "unit_index": unit_index,
                "unit_type": unit_type,
                "text_raw": text_raw,
                "text_normalized": text_norm,
                "char_count": char_count,
                "source_element_id": source_element_id,
            }

            if include_anchor:
                result["anchor"] = anchor_json

            # Get associated chunks with bbox
            cur.execute(
                """
                SELECT
                    c.chunk_id, c.text_raw, c.page_idx, c.bbox, c.element_type,
                    dp.coord_sys
                FROM chunks c
                JOIN chunk_unit_spans span ON c.chunk_id = span.chunk_id
                LEFT JOIN document_pages dp
                    ON c.version_id = dp.version_id AND c.page_idx = dp.page_idx
                WHERE span.unit_id = %s
                """,
                (unit_id,),
            )

            chunks = cur.fetchall()
            result["associated_chunks"] = [
                {
                    "chunk_id": str(r[0]),
                    "text_preview": r[1][:200] if r[1] else None,
                    "page_idx": r[2],
                    "bbox": r[3] if r[3] else None,
                    "element_type": r[4] if r[4] else None,
                    "coord_system": r[5] if r[5] else "mineru_bbox_v1",
                }
                for r in chunks
            ]

            if verify_hash and unit_hash:
                from bid_scoring.anchors_v2 import compute_unit_hash

                recomputed_unit_hash = compute_unit_hash(
                    text_norm=text_norm or "",
                    anchor_json=anchor_json or {"anchors": []},
                    source_element_id=source_element_id,
                )
                result["hash_verified"] = recomputed_unit_hash == unit_hash
                result["computed_unit_hash"] = recomputed_unit_hash

            return result


def compare_across_versions(
    retrieve_fn: Callable[..., Dict[str, Any]],
    version_ids: list[str],
    query: str,
    top_k_per_version: int = 3,
    normalize_scores: bool = True,
    include_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Compare responses across multiple bidding versions for the same query.

    Essential for bid analysis - see how different bidders respond to
    the same requirement.

    Args:
        version_ids: List of version UUIDs to compare.
        query: Search query (e.g., "售后服务响应时间").
        top_k_per_version: Results per version.
        normalize_scores: Normalize scores across versions for fair comparison.

    Returns:
        Side-by-side comparison of responses from each version.
    """
    # Validate inputs
    version_ids = validate_string_list(
        version_ids, "version_ids", min_items=1, max_items=20
    )
    query = validate_query(query)
    top_k_per_version = validate_positive_int(
        top_k_per_version, "top_k_per_version", max_value=50
    )

    all_results = {}
    all_scores = []
    per_version_diagnostics: dict[str, Any] = {}

    for version_id in version_ids:
        result = retrieve_fn(
            version_id=version_id,
            query=query,
            top_k=top_k_per_version,
            mode="hybrid",
            include_text=True,
            max_chars=500,
            include_diagnostics=include_diagnostics,
        )

        all_results[version_id] = result["results"]
        if include_diagnostics:
            per_version_diagnostics[version_id] = result.get("diagnostics") or {}
        all_scores.extend(
            [
                score
                for score in (r.get("score") for r in result["results"])
                if isinstance(score, (int, float))
            ]
        )

    # Normalize scores if requested
    if normalize_scores and all_scores:
        max_score = max(all_scores) if all_scores else 1.0
        min_score = min(all_scores) if all_scores else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        for version_id, results in all_results.items():
            for r in results:
                score = r.get("score")
                if not isinstance(score, (int, float)):
                    r["normalized_score"] = None
                elif score_range > 0:
                    r["normalized_score"] = (score - min_score) / score_range
                else:
                    r["normalized_score"] = 1.0

    response = {
        "query": query,
        "version_count": len(version_ids),
        "versions_compared": version_ids,
        "normalize_scores": normalize_scores,
        "results_by_version": all_results,
    }
    if include_diagnostics:
        response["diagnostics"] = {
            "version_count": len(version_ids),
            "per_version": per_version_diagnostics,
            "score_sample_size": len(all_scores),
        }
    return response


def extract_key_value(
    version_id: str,
    key_patterns: list[str],
    value_patterns: list[str] | None = None,
    fuzzy_match: bool = True,
    context_window: int = 50,
) -> list[Dict[str, Any]]:
    """Extract structured key-value pairs from document text.

    Useful for extracting commitments like:
    - 质保期: 5年
    - 响应时间: 2小时
    - 培训天数: 3天

    Args:
        version_id: UUID of the document version.
        key_patterns: Keywords to search for (e.g., ["质保期", "保修期"]).
        value_patterns: Optional patterns for values (e.g., ["年", "月", "天"]).
        fuzzy_match: Use fuzzy matching for key patterns.
        context_window: Characters to extract around the match.

    Returns:
        List of extracted key-value pairs with source locations.
    """
    import psycopg

    # Validate inputs
    version_id = validate_version_id(version_id)
    key_patterns = validate_string_list(
        key_patterns, "key_patterns", min_items=1, max_items=50
    )
    if value_patterns is not None:
        value_patterns = validate_string_list(
            value_patterns, "value_patterns", min_items=0, max_items=50
        )
    context_window = validate_positive_int(
        context_window, "context_window", max_value=1000
    )

    settings = load_settings()
    extractions = []

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Build search condition
            if fuzzy_match:
                key_conditions = " OR ".join(
                    ["text_raw ILIKE %s" for _ in key_patterns]
                )
                params = [f"%{k}%" for k in key_patterns] + [version_id]
            else:
                key_conditions = " OR ".join(["text_raw LIKE %s" for _ in key_patterns])
                params = key_patterns + [version_id]

            cur.execute(
                f"""
                SELECT chunk_id, text_raw, page_idx, source_id
                FROM chunks
                WHERE ({key_conditions})
                AND version_id = %s
                AND text_raw IS NOT NULL
                """,
                tuple(params),
            )

            rows = cur.fetchall()

            for row in rows:
                chunk_id, text_raw, page_idx, source_id = row

                # Find key matches in text
                for key in key_patterns:
                    if fuzzy_match:
                        pattern = re.compile(re.escape(key), re.IGNORECASE)
                    else:
                        pattern = re.compile(re.escape(key))

                    for match in pattern.finditer(text_raw):
                        start = max(0, match.start() - context_window)
                        end = min(len(text_raw), match.end() + context_window)
                        context = text_raw[start:end]

                        extraction = {
                            "key": key,
                            "context": context,
                            "chunk_id": str(chunk_id),
                            "page_idx": page_idx,
                            "source_id": source_id,
                            "match_position": match.span(),
                        }

                        # Try to extract value if value_patterns provided
                        if value_patterns:
                            # Look for value patterns after the key
                            search_text = text_raw[
                                match.end() : match.end() + context_window * 2
                            ]
                            for vp in value_patterns:
                                # Pattern: number + unit
                                val_regex = rf"(\d+(?:\.\d+)?)\s*{re.escape(vp)}"
                                val_match = re.search(val_regex, search_text)
                                if val_match:
                                    extraction["value"] = val_match.group(0)
                                    extraction["numeric_value"] = float(
                                        val_match.group(1)
                                    )
                                    extraction["unit"] = vp
                                    break

                        extractions.append(extraction)

    # Deduplicate by chunk_id + key
    seen = set()
    unique = []
    for e in extractions:
        key = (e["chunk_id"], e["key"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique
