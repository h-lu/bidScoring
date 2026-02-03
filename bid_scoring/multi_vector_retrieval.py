"""Multi-Vector Retriever with Parent-Child Association

Implements parent-child chunk retrieval where:
- Child chunks (small, specific) are searched
- Parent chunks (large, contextual) are returned

Retrieval modes:
- child: Search and return child chunks only
- parent: Search parent chunks directly
- hierarchical: Traverse hierarchical structure
- hybrid: Combine BM25 and vector search (default)

References:
- https://arxiv.org/abs/2406.14657 (Multi-Vector Retrieval)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from bid_scoring.config import load_settings
from bid_scoring.embeddings import embed_single_text
from bid_scoring.search import rrf_fuse

logger = logging.getLogger(__name__)


class MultiVectorRetriever:
    """Retriever supporting parent-child chunk relationships.

    Searches child chunks (small, specific) and returns parent chunks
    (large, contextual) for better retrieval quality.
    """

    # Supported retrieval modes
    RETRIEVAL_MODES = ["child", "parent", "hierarchical", "hybrid"]

    def __init__(
        self,
        dsn: Optional[str] = None,
        embedding_dim: int = 1536,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        """Initialize the multi-vector retriever.

        Args:
            dsn: Database connection string (defaults to config)
            embedding_dim: Dimension of embeddings (default 1536 for text-embedding-3-small)
            bm25_weight: Weight for BM25 scores in RRF fusion (default 0.4)
            vector_weight: Weight for vector scores in RRF fusion (default 0.6)
            rrf_k: RRF constant for rank fusion (default 60)
        """
        self.dsn = dsn or load_settings()["DATABASE_URL"]
        self.embedding_dim = embedding_dim
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

    def _get_connection(self) -> psycopg.Connection:
        """Get a database connection."""
        return psycopg.connect(self.dsn)

    def _bm25_search(
        self,
        conn: psycopg.Connection,
        query: str,
        version_id: Optional[str] = None,
        top_k: int = 10,
        search_parents: bool = False,
    ) -> List[Tuple[str, float]]:
        """Perform BM25 text search on chunks.

        Args:
            conn: Database connection
            query: Search query
            version_id: Optional version ID to filter by
            top_k: Number of results to return
            search_parents: If True, search parent chunks; else search child chunks

        Returns:
            List of (chunk_id, score) tuples
        """
        with conn.cursor() as cur:
            if search_parents:
                # Search parent chunks via multi_vector_mappings
                if version_id:
                    cur.execute(
                        """
                        SELECT c.chunk_id, ts_rank_cd(c.text_tsv, query, 32) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.parent_chunk_id
                        CROSS JOIN plainto_tsquery('simple', %s) query
                        WHERE c.text_tsv @@ query
                          AND mvm.version_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, version_id, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.chunk_id, ts_rank_cd(c.text_tsv, query, 32) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.parent_chunk_id
                        CROSS JOIN plainto_tsquery('simple', %s) query
                        WHERE c.text_tsv @@ query
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, top_k),
                    )
            else:
                # Search child chunks
                if version_id:
                    cur.execute(
                        """
                        SELECT c.chunk_id, ts_rank_cd(c.text_tsv, query, 32) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.child_chunk_id
                        CROSS JOIN plainto_tsquery('simple', %s) query
                        WHERE c.text_tsv @@ query
                          AND mvm.version_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, version_id, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.chunk_id, ts_rank_cd(c.text_tsv, query, 32) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.child_chunk_id
                        CROSS JOIN plainto_tsquery('simple', %s) query
                        WHERE c.text_tsv @@ query
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, top_k),
                    )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def _vector_search(
        self,
        conn: psycopg.Connection,
        query_embedding: List[float],
        version_id: Optional[str] = None,
        top_k: int = 10,
        search_parents: bool = False,
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search on chunks.

        Args:
            conn: Database connection
            query_embedding: Query vector embedding
            version_id: Optional version ID to filter by
            top_k: Number of results to return
            search_parents: If True, search parent chunks; else search child chunks

        Returns:
            List of (chunk_id, score) tuples
        """
        with conn.cursor() as cur:
            if search_parents:
                # Search parent chunks via multi_vector_mappings
                if version_id:
                    cur.execute(
                        """
                        SELECT c.chunk_id, 1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.parent_chunk_id
                        WHERE c.embedding IS NOT NULL
                          AND mvm.version_id = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, version_id, query_embedding, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.chunk_id, 1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.parent_chunk_id
                        WHERE c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, query_embedding, top_k),
                    )
            else:
                # Search child chunks
                if version_id:
                    cur.execute(
                        """
                        SELECT c.chunk_id, 1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.child_chunk_id
                        WHERE c.embedding IS NOT NULL
                          AND mvm.version_id = %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, version_id, query_embedding, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT c.chunk_id, 1 - (c.embedding <=> %s::vector) AS score
                        FROM chunks c
                        JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.child_chunk_id
                        WHERE c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, query_embedding, top_k),
                    )
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def _get_parent_chunks(
        self,
        conn: psycopg.Connection,
        child_chunk_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Get parent chunks for given child chunk IDs.

        Args:
            conn: Database connection
            child_chunk_ids: List of child chunk IDs

        Returns:
            List of parent chunk dictionaries
        """
        if not child_chunk_ids:
            return []

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (p.chunk_id)
                    p.chunk_id,
                    p.text_raw,
                    p.page_idx,
                    p.element_type,
                    c.chunk_id as child_chunk_id,
                    mvm.relationship,
                    mvm.metadata
                FROM chunks c
                JOIN multi_vector_mappings mvm ON c.chunk_id = mvm.child_chunk_id
                JOIN chunks p ON mvm.parent_chunk_id = p.chunk_id
                WHERE c.chunk_id = ANY(%s)
                  AND mvm.parent_chunk_id IS NOT NULL
                """,
                (child_chunk_ids,),
            )
            rows = cur.fetchall()

        parents = []
        seen_parent_ids = set()

        for row in rows:
            parent_id = row[0]
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parents.append({
                    "chunk_id": parent_id,
                    "text": row[1],
                    "page_idx": row[2],
                    "element_type": row[3],
                    "source_child_chunk_id": row[4],
                    "relationship": row[5],
                    "metadata": row[6] or {},
                })

        return parents

    def _get_chunk_details(
        self,
        conn: psycopg.Connection,
        chunk_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Get full details for chunk IDs.

        Args:
            conn: Database connection
            chunk_ids: List of chunk IDs

        Returns:
            List of chunk dictionaries
        """
        if not chunk_ids:
            return []

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, text_raw, page_idx, element_type, embedding IS NOT NULL as has_embedding
                FROM chunks
                WHERE chunk_id = ANY(%s)
                ORDER BY array_position(%s::uuid[], chunk_id)
                """,
                (chunk_ids, chunk_ids),
            )
            rows = cur.fetchall()

        return [
            {
                "chunk_id": row[0],
                "text": row[1],
                "page_idx": row[2],
                "element_type": row[3],
                "has_embedding": row[4],
            }
            for row in rows
        ]

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank results using a simple scoring mechanism.

        Uses BM25-like scoring based on term frequency in the text.
        In production, this could use a cross-encoder or LLM-based reranker.

        Args:
            query: Search query
            results: List of result dictionaries with 'text' field
            top_k: Number of top results to return

        Returns:
            Reranked list of results
        """
        if not results:
            return []

        # Simple term-based scoring
        query_terms = query.lower().split()

        scored_results = []
        for result in results:
            text = (result.get("text") or "").lower()
            if not text:
                score = 0.0
            else:
                # Count term occurrences
                score = sum(text.count(term) for term in query_terms)
                # Normalize by text length
                score = score / (len(text.split()) + 1)
                # Boost results with original query terms
                if query.lower() in text:
                    score += 1.0

            scored_results.append((result, score))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return [r[0] for r in scored_results[:top_k]]

    async def retrieve(
        self,
        query: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        rerank: bool = True,
        version_id: Optional[str] = None,
        return_parents: bool = True,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks using multi-vector retrieval.

        Args:
            query: Search query
            retrieval_mode: One of 'child', 'parent', 'hierarchical', 'hybrid'
            top_k: Number of results to return
            rerank: Whether to apply reranking
            version_id: Optional version ID to filter by
            return_parents: If True, return parent chunks; if False, return child chunks

        Returns:
            List of chunk dictionaries with metadata

        Raises:
            ValueError: If retrieval_mode is invalid
        """
        if retrieval_mode not in self.RETRIEVAL_MODES:
            raise ValueError(
                f"Invalid retrieval_mode: {retrieval_mode}. "
                f"Must be one of: {self.RETRIEVAL_MODES}"
            )

        try:
            with self._get_connection() as conn:
                # Determine if we should search parents or children
                search_parents = retrieval_mode == "parent"

                if retrieval_mode == "hybrid":
                    # Hybrid search: BM25 + Vector
                    query_embedding = embed_single_text(query)

                    # Perform both searches
                    bm25_results = self._bm25_search(
                        conn, query, version_id, top_k * 2, search_parents=False
                    )
                    vector_results = self._vector_search(
                        conn, query_embedding, version_id, top_k * 2, search_parents=False
                    )

                    # Fuse results using RRF
                    fused_ids = rrf_fuse(
                        bm25_results,
                        vector_results,
                        k=self.rrf_k,
                        bm25_weight=self.bm25_weight,
                        vector_weight=self.vector_weight,
                    )

                    # Get parent chunks if requested
                    if return_parents:
                        results = self._get_parent_chunks(conn, fused_ids)
                    else:
                        results = self._get_chunk_details(conn, fused_ids)

                elif retrieval_mode == "child":
                    # Child-only search: use hybrid on child chunks
                    query_embedding = embed_single_text(query)

                    bm25_results = self._bm25_search(
                        conn, query, version_id, top_k * 2, search_parents=False
                    )
                    vector_results = self._vector_search(
                        conn, query_embedding, version_id, top_k * 2, search_parents=False
                    )

                    fused_ids = rrf_fuse(
                        bm25_results,
                        vector_results,
                        k=self.rrf_k,
                        bm25_weight=self.bm25_weight,
                        vector_weight=self.vector_weight,
                    )

                    if return_parents:
                        results = self._get_parent_chunks(conn, fused_ids)
                    else:
                        results = self._get_chunk_details(conn, fused_ids)

                elif retrieval_mode == "parent":
                    # Parent-only search: search parents directly
                    query_embedding = embed_single_text(query)

                    bm25_results = self._bm25_search(
                        conn, query, version_id, top_k * 2, search_parents=True
                    )
                    vector_results = self._vector_search(
                        conn, query_embedding, version_id, top_k * 2, search_parents=True
                    )

                    fused_ids = rrf_fuse(
                        bm25_results,
                        vector_results,
                        k=self.rrf_k,
                        bm25_weight=self.bm25_weight,
                        vector_weight=self.vector_weight,
                    )

                    results = self._get_chunk_details(conn, fused_ids)

                else:  # hierarchical
                    # Hierarchical mode: search hierarchical_nodes table
                    results = self._hierarchical_search(conn, query, version_id, top_k)

                # Apply reranking if enabled
                if rerank and results:
                    results = self._rerank_results(query, results, top_k)
                else:
                    results = results[:top_k]

                return results

        except Exception as e:
            logger.error(f"Multi-vector retrieval failed: {e}")
            # Fallback: return empty list
            return []

    def _hierarchical_search(
        self,
        conn: psycopg.Connection,
        query: str,
        version_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search using hierarchical nodes structure.

        Args:
            conn: Database connection
            query: Search query
            version_id: Optional version ID to filter by
            top_k: Number of results to return

        Returns:
            List of hierarchical node dictionaries
        """
        # Use BM25 on hierarchical_nodes content
        with conn.cursor() as cur:
            if version_id:
                cur.execute(
                    """
                    SELECT 
                        hn.node_id,
                        hn.content,
                        hn.node_type,
                        hn.level,
                        hn.metadata,
                        ts_rank_cd(
                            to_tsvector('simple', hn.content),
                            plainto_tsquery('simple', %s),
                            32
                        ) AS score
                    FROM hierarchical_nodes hn
                    WHERE hn.version_id = %s
                      AND to_tsvector('simple', hn.content) @@ plainto_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, version_id, query, top_k * 2),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        hn.node_id,
                        hn.content,
                        hn.node_type,
                        hn.level,
                        hn.metadata,
                        ts_rank_cd(
                            to_tsvector('simple', hn.content),
                            plainto_tsquery('simple', %s),
                            32
                        ) AS score
                    FROM hierarchical_nodes hn
                    WHERE to_tsvector('simple', hn.content) @@ plainto_tsquery('simple', %s)
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query, query, top_k * 2),
                )
            rows = cur.fetchall()

        return [
            {
                "node_id": row[0],
                "text": row[1],
                "node_type": row[2],
                "level": row[3],
                "metadata": row[4] or {},
                "score": float(row[5]),
                "is_hierarchical": True,
            }
            for row in rows
        ]


class FallbackRetriever:
    """Fallback retriever using basic search.py implementation.

    Used when MultiVectorRetriever fails or tables are not populated.
    """

    def __init__(self, dsn: Optional[str] = None):
        """Initialize fallback retriever.

        Args:
            dsn: Database connection string
        """
        self.dsn = dsn or load_settings()["DATABASE_URL"]

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        version_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Basic retrieval without parent-child relationship.

        Args:
            query: Search query
            top_k: Number of results to return
            version_id: Optional version ID to filter by

        Returns:
            List of chunk dictionaries
        """
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                if version_id:
                    cur.execute(
                        """
                        SELECT chunk_id, text_raw, page_idx, element_type,
                               ts_rank_cd(text_tsv, query, 32) AS score
                        FROM chunks
                        CROSS JOIN plainto_tsquery('simple', %s) query
                        WHERE text_tsv @@ query
                          AND version_id = %s
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, version_id, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT chunk_id, text_raw, page_idx, element_type,
                               ts_rank_cd(text_tsv, query, 32) AS score
                        FROM chunks
                        CROSS JOIN plainto_tsquery('simple', %s) query
                        WHERE text_tsv @@ query
                        ORDER BY score DESC
                        LIMIT %s
                        """,
                        (query, top_k),
                    )
                rows = cur.fetchall()

        return [
            {
                "chunk_id": row[0],
                "text": row[1],
                "page_idx": row[2],
                "element_type": row[3],
                "score": float(row[4]),
                "is_fallback": True,
            }
            for row in rows
        ]


async def retrieve_with_fallback(
    query: str,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    rerank: bool = True,
    version_id: Optional[str] = None,
    return_parents: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience function for retrieval with automatic fallback.

    Tries MultiVectorRetriever first, falls back to basic search if needed.

    Args:
        query: Search query
        retrieval_mode: One of 'child', 'parent', 'hierarchical', 'hybrid'
        top_k: Number of results to return
        rerank: Whether to apply reranking
        version_id: Optional version ID to filter by
        return_parents: If True, return parent chunks

    Returns:
        List of chunk dictionaries
    """
    retriever = MultiVectorRetriever()

    try:
        results = await retriever.retrieve(
            query=query,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            rerank=rerank,
            version_id=version_id,
            return_parents=return_parents,
        )

        if results:
            return results

        # If no results, try fallback
        logger.warning("MultiVectorRetriever returned no results, using fallback")
        fallback = FallbackRetriever()
        return await fallback.retrieve(query, top_k, version_id)

    except Exception as e:
        logger.error(f"MultiVectorRetriever failed: {e}, using fallback")
        fallback = FallbackRetriever()
        return await fallback.retrieve(query, top_k, version_id)
