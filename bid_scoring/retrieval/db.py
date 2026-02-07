"""Database helpers for hybrid retrieval."""

from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from psycopg import sql

logger = logging.getLogger(__name__)

# Connection pool support - graceful fallback if not installed
try:
    from psycopg_pool import ConnectionPool

    HAS_CONNECTION_POOL = True
except ImportError:  # pragma: no cover
    HAS_CONNECTION_POOL = False
    ConnectionPool = None  # type: ignore[assignment]


def vector_search(
    *,
    get_connection,
    version_id: str,
    query_embedding: list[float],
    top_k: int,
    hnsw_ef_search: int,
) -> List[Tuple[str, float]]:
    """Vector similarity search using cosine similarity.

    `get_connection` must return a context manager yielding a psycopg Connection.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '30s'")
            cur.execute(
                sql.SQL("SET hnsw.ef_search = {}").format(
                    sql.Literal(int(hnsw_ef_search))
                )
            )
            cur.execute(
                """
                SELECT chunk_id::text,
                       1 - (embedding <=> %s::vector) as similarity
                FROM chunks
                WHERE version_id = %s
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, version_id, query_embedding, top_k * 2),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]


def keyword_search_fulltext(
    *,
    get_connection,
    version_id: str,
    ts_query: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    """Keyword search using PostgreSQL full-text search."""
    with get_connection() as conn:
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
                (ts_query, version_id, top_k * 2),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]


def keyword_search_legacy(
    *,
    get_connection,
    version_id: str,
    keywords: Sequence[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    """Legacy keyword search using ILIKE as fallback."""
    if not keywords:
        return []

    with get_connection() as conn:
        with conn.cursor() as cur:
            conditions = " OR ".join(["text_raw ILIKE %s"] * len(keywords))
            match_scores = " + ".join(
                ["CASE WHEN text_raw ILIKE %s THEN 1 ELSE 0 END" for _ in keywords]
            )
            keyword_patterns = [f"%{k}%" for k in keywords]
            params = keyword_patterns + [version_id] + keyword_patterns + [top_k * 2]
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


def fetch_chunks_by_id(
    *,
    get_connection,
    chunk_ids: Sequence[str],
) -> dict[str, tuple[str, str, int]]:
    """Fetch chunks for a list of ids.

    Returns mapping chunk_id -> (chunk_id, text_raw, page_idx).
    """
    if not chunk_ids:
        return {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id::text, text_raw, page_idx
                FROM chunks
                WHERE chunk_id = ANY(%s::uuid[])
                """,
                (list(chunk_ids),),
            )
            return {
                row[0]: (row[0], row[1] or "", row[2] or 0) for row in cur.fetchall()
            }
