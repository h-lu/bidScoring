from __future__ import annotations

import logging
from typing import List, Tuple

import psycopg
from psycopg import sql

from bid_scoring.embeddings import embed_single_text

logger = logging.getLogger(__name__)


def vector_search(retriever: object, query: str) -> List[Tuple[str, float]]:
    """Vector similarity search using cosine similarity."""
    try:
        query_emb = embed_single_text(query)

        # retriever is expected to expose:
        # - version_id: str
        # - top_k: int
        # - _hnsw_ef_search: int
        # - _get_connection(): contextmanager returning a psycopg connection
        with retriever._get_connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '30s'")
                cur.execute(
                    sql.SQL("SET hnsw.ef_search = {}").format(
                        sql.Literal(int(retriever._hnsw_ef_search))  # type: ignore[attr-defined]
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
                    (
                        query_emb,
                        retriever.version_id,  # type: ignore[attr-defined]
                        query_emb,
                        retriever.top_k * 2,  # type: ignore[attr-defined]
                    ),
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]
    except Exception as e:
        logger.error(
            "Vector search failed for query '%s...': %s", query[:50], e, exc_info=True
        )
        return []

