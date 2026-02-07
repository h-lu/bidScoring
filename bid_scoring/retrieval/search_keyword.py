from __future__ import annotations

import logging
from typing import List, Tuple

import psycopg

logger = logging.getLogger(__name__)


def keyword_search_fulltext(
    retriever: object,
    keywords: List[str],
    use_or_semantic: bool = True,
) -> List[Tuple[str, float]]:
    """PostgreSQL full-text search (tsvector + GIN) keyword matching."""
    if not keywords:
        return []

    cleaned_keywords = [k.strip().replace('"', " ") for k in keywords if k.strip()]
    if not cleaned_keywords:
        return []

    joiner = " OR " if use_or_semantic else " "
    ts_query = joiner.join(cleaned_keywords)

    try:
        with retriever._get_connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '30s'")

                cur.execute(
                    "SELECT querytree(websearch_to_tsquery('simple', %s))",
                    (ts_query,),
                )
                querytree_result = cur.fetchone()
                querytree_text = (
                    str(querytree_result[0]).strip()
                    if querytree_result and querytree_result[0] is not None
                    else ""
                )
                if querytree_text in {"", "T"}:
                    logger.debug(
                        "Skip fulltext search due to non-indexable querytree: %s",
                        querytree_text,
                    )
                    return []

                cur.execute(
                    """
                    WITH q AS (
                        SELECT websearch_to_tsquery('simple', %s) AS tsq
                    )
                    SELECT
                        chunk_id::text,
                        ts_rank_cd(textsearch, q.tsq, 32) as rank
                    FROM chunks, q
                    WHERE version_id = %s
                      AND textsearch @@ q.tsq
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (
                        ts_query,
                        retriever.version_id,  # type: ignore[attr-defined]
                        retriever.top_k * 2,  # type: ignore[attr-defined]
                    ),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]
    except psycopg.Error as e:
        logger.error("Fulltext search failed: %s", e, exc_info=True)
        logger.warning("Falling back to legacy keyword search")
        return keyword_search_legacy(retriever, keywords)


def keyword_search_legacy(retriever: object, keywords: List[str]) -> List[Tuple[str, float]]:
    """Legacy ILIKE keyword search (fallback)."""
    if not keywords:
        return []

    try:
        with retriever._get_connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                conditions = " OR ".join(["text_raw ILIKE %s"] * len(keywords))
                match_scores = " + ".join(
                    ["CASE WHEN text_raw ILIKE %s THEN 1 ELSE 0 END" for _ in keywords]
                )
                keyword_patterns = [f"%{k}%" for k in keywords]
                params = (
                    keyword_patterns
                    + [retriever.version_id]  # type: ignore[attr-defined]
                    + keyword_patterns
                    + [retriever.top_k * 2]  # type: ignore[attr-defined]
                )
                cur.execute(
                    f"""
                    SELECT chunk_id::text,
                           ({match_scores}) as match_count
                    FROM chunks
                    WHERE version_id = %s
                      AND ({conditions})
                    ORDER BY match_count DESC
                    LIMIT %s
                    """,
                    params,
                )
                return [(row[0], float(row[1] or 0)) for row in cur.fetchall()]
    except Exception as e:
        logger.error(
            "Keyword search failed with keywords %s: %s", keywords, e, exc_info=True
        )
        return []

