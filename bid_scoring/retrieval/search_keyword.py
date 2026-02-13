from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


def keyword_search_fulltext(
    retriever: object,
    keywords: List[str],
    use_or_semantic: bool = True,
) -> List[Tuple[str, float]]:
    """Keyword search with textsearch-first strategy.

    Strategy:
    - CJK keywords: use trigram/ILIKE path directly
    - Non-CJK keywords: prefer PostgreSQL textsearch ranking (`ts_rank_cd`)
      with fallback to trigram/ILIKE for robustness

    This keeps Chinese retrieval stable while improving ranking quality for
    English/alphanumeric technical queries.

    Requires:
        - pg_trgm extension: CREATE EXTENSION pg_trgm;
        - GIN index: CREATE INDEX idx_chunks_text_raw_trgm
                     ON chunks USING gin(text_raw gin_trgm_ops);

    Performance: GIN index makes ILIKE queries efficient even for large tables.
    See: migrations/003_add_pg_trgm_index_for_keyword_search.sql
    """
    if not keywords:
        return []

    cleaned_keywords = [k.strip().replace('"', " ") for k in keywords if k.strip()]
    if not cleaned_keywords:
        return []

    # Chinese text is not tokenized well by simple textsearch config.
    if any(_contains_cjk(keyword) for keyword in cleaned_keywords):
        return keyword_search_legacy(retriever, cleaned_keywords)

    ranked = _keyword_search_textsearch(
        retriever,
        cleaned_keywords,
        use_or_semantic=use_or_semantic,
    )
    if ranked:
        return ranked

    # Fallback when textsearch column/index not available or no matches.
    return keyword_search_legacy(retriever, cleaned_keywords)


def _contains_cjk(text: str) -> bool:
    return re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF]", text) is not None


def _build_tsquery_expression(
    keywords: List[str],
    *,
    use_or_semantic: bool,
) -> str:
    terms: List[str] = []
    for keyword in keywords:
        sanitized = re.sub(r"[':|&!()<>]", " ", keyword).strip()
        if not sanitized:
            continue
        tokens = [token for token in sanitized.split() if token]
        if not tokens:
            continue
        terms.append(" & ".join(tokens))

    if not terms:
        return ""

    separator = " | " if use_or_semantic else " & "
    return separator.join(terms)


def _keyword_search_textsearch(
    retriever: object,
    keywords: List[str],
    use_or_semantic: bool = True,
) -> List[Tuple[str, float]]:
    tsquery_expr = _build_tsquery_expression(
        keywords,
        use_or_semantic=use_or_semantic,
    )
    if not tsquery_expr:
        return []

    try:
        with retriever._get_connection() as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH q AS (
                        SELECT to_tsquery('simple', %s) AS query
                    )
                    SELECT c.chunk_id::text,
                           ts_rank_cd(c.textsearch, q.query, 32) AS rank
                    FROM chunks c
                    CROSS JOIN q
                    WHERE c.version_id = %s
                      AND c.textsearch @@ q.query
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (
                        tsquery_expr,
                        retriever.version_id,  # type: ignore[attr-defined]
                        retriever.top_k * 2,  # type: ignore[attr-defined]
                    ),
                )
                return [(row[0], float(row[1] or 0.0)) for row in cur.fetchall()]
    except Exception as exc:
        logger.debug(
            "Textsearch keyword path unavailable, fallback to trigram/ILIKE: %s",
            exc,
            exc_info=True,
        )
        return []


def keyword_search_legacy(
    retriever: object, keywords: List[str]
) -> List[Tuple[str, float]]:
    """Legacy ILIKE keyword search with pg_trgm GIN index support."""
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
