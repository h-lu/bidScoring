"""
Hybrid Retrieval Module for Bid Scoring

Combines vector similarity search with keyword matching using
Reciprocal Rank Fusion (RRF) for optimal retrieval performance.

References:
- DeepMind: "On the Theoretical Limitations of Embedding-Based Retrieval"
- Cormack et al.: "Reciprocal Rank Fusion outperforms Condorcet"
- Assembled Blog: "Better RAG results with Reciprocal Rank Fusion and hybrid search"
- LangChain Hybrid Search Best Practices
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import psycopg

from bid_scoring.embeddings import embed_single_text

logger = logging.getLogger(__name__)

# DeepMind recommended value for RRF damping constant.
# This value balances the influence of top-ranked items vs. deep-ranked items.
DEFAULT_RRF_K = 60

# Maximum number of parallel search workers
MAX_SEARCH_WORKERS = 2


@dataclass
class RetrievalResult:
    """Single retrieval result with detailed scoring information."""

    chunk_id: str
    text: str
    page_idx: int
    score: float
    source: str  # "vector", "keyword", or "hybrid"
    vector_score: float | None = None  # Original vector similarity score
    keyword_score: float | None = None  # Original keyword match score
    embedding: List[float] | None = None


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for merging ranked lists.

    RRF formula: score = sum(1 / (k + rank)) for each list
    where k is a constant (default 60) to dampen the impact of ranking

    Reference:
        Cormack, V., & Clarke, C. (2009). "Reciprocal Rank Fusion outperforms
        Condorcet and individual Rank Learning Methods"
    """

    def __init__(self, k: int = DEFAULT_RRF_K):
        self.k = k

    def fuse(
        self,
        vector_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float, Dict[str, dict]]]:
        """
        Merge vector and keyword search results using RRF.

        Args:
            vector_results: List of (chunk_id, similarity_score) from vector search
            keyword_results: List of (chunk_id, match_count) from keyword search

        Returns:
            Merged list of (chunk_id, rrf_score, source_scores) sorted by RRF score descending.
            source_scores contains original rank and score from each source.
        """
        scores: Dict[str, float] = {}
        sources: Dict[str, Dict[str, dict]] = {}

        # Process vector search results
        for rank, (doc_id, orig_score) in enumerate(vector_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = {}
            scores[doc_id] += 1.0 / (self.k + rank + 1)
            sources[doc_id]["vector"] = {"rank": rank, "score": orig_score}

        # Process keyword search results
        for rank, (doc_id, orig_score) in enumerate(keyword_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = {}
            scores[doc_id] += 1.0 / (self.k + rank + 1)
            sources[doc_id]["keyword"] = {"rank": rank, "score": orig_score}

        # Sort by RRF score descending and include source information
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, sources[doc_id]) for doc_id, score in sorted_results]


class HybridRetriever:
    """
    Hybrid retriever combining vector and keyword search with RRF fusion.

    This implementation follows industry best practices:
    - Parallel execution of vector and keyword searches
    - Proper error handling with logging
    - RRF (Reciprocal Rank Fusion) for result merging
    - Detailed score tracking for debugging

    Usage:
        retriever = HybridRetriever(version_id="xxx", settings=settings)
        results = retriever.retrieve("培训时长", keywords=["培训", "时长", "天数"])
    """

    def __init__(
        self,
        version_id: str,
        settings: dict,
        top_k: int = 10,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            version_id: Document version ID to search within
            settings: Configuration dictionary containing DATABASE_URL
            top_k: Number of top results to return
            rrf_k: RRF damping constant (default 60 as per DeepMind research)

        Raises:
            ValueError: If version_id is empty or top_k is not positive
        """
        if not version_id:
            raise ValueError("version_id cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.version_id = version_id
        self.settings = settings
        self.top_k = top_k
        self.rrf = ReciprocalRankFusion(k=rrf_k)

    def _vector_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search using cosine similarity.

        Args:
            query: Search query text

        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        try:
            query_emb = embed_single_text(query)

            with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
                with conn.cursor() as cur:
                    # Use cosine similarity: 1 - (embedding <=> query) gives similarity in [0, 1]
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
                        (query_emb, self.version_id, query_emb, self.top_k * 2),
                    )
                    return [(row[0], float(row[1])) for row in cur.fetchall()]
        except Exception as e:
            logger.error(
                f"Vector search failed for query '{query[:50]}...': {e}",
                exc_info=True,
            )
            return []

    def _keyword_search(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """
        Perform keyword search using ILIKE pattern matching.

        Args:
            keywords: List of keywords to search for

        Returns:
            List of (chunk_id, match_count) tuples
        """
        if not keywords:
            return []

        try:
            with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
                with conn.cursor() as cur:
                    # Build ILIKE conditions for WHERE clause
                    conditions = " OR ".join(["text_raw ILIKE %s"] * len(keywords))

                    # Build match score calculation (count of matching keywords)
                    match_scores = " + ".join(
                        [
                            "CASE WHEN text_raw ILIKE %s THEN 1 ELSE 0 END"
                            for _ in keywords
                        ]
                    )

                    # Build parameters: patterns for match_scores, version_id,
                    # patterns for conditions, limit
                    keyword_patterns = [f"%{k}%" for k in keywords]
                    params = (
                        keyword_patterns  # For match_scores
                        + [self.version_id]  # For version_id
                        + keyword_patterns  # For conditions
                        + [self.top_k * 2]  # For LIMIT
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
                f"Keyword search failed with keywords {keywords}: {e}", exc_info=True
            )
            return []

    def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract keywords from natural language query with field-specific expansion.

        This method uses:
        1. Stopword filtering for Chinese
        2. Field-specific keyword dictionaries for synonym expansion
        3. Alphanumeric token extraction

        Args:
            query: Natural language query string

        Returns:
            List of extracted keywords with synonyms expanded
        """
        # Chinese stopwords (common)
        stopwords = {
            "的",
            "了",
            "是",
            "在",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "一个",
            "上",
            "也",
            "很",
            "到",
            "说",
            "要",
            "去",
            "你",
            "会",
            "着",
            "没有",
            "看",
            "好",
            "自己",
            "这",
            "那",
            "多少",  # Filter this as it's a question word
            "什么",
        }

        # Field-specific keyword expansion dictionaries
        # These map core concepts to their synonyms for better recall
        field_keywords = {
            "培训": ["培训", "训练", "教学", "指导", "教程"],
            "时长": ["时长", "时间", "天数", "小时", "工作日", "周期", "期限"],
            "计划": ["计划", "安排", "课程", "大纲", "内容", "方案"],
            "对象": ["对象", "人员", "受训", "用户", "学员", "参与者"],
            "老师": ["老师", "讲师", "授课", "教师", "专家", "培训师"],
            "资质": ["资质", "资格", "认证", "证书", "背景", "经验"],
            "响应": ["响应", "反应", "回复", "到达", "到场", "时效"],
            "质保": ["质保", "保修", "质量保证", "保修期", "维护期"],
            "配件": ["配件", "备件", "耗材", "零件", "部件", "组件"],
            "服务": ["服务", "支持", "维护", "售后", "保障"],
            "费用": ["费用", "收费", "价格", "成本", "金额", "预算"],
        }

        # Extract field keywords that appear in the query
        expanded = set()
        for key, synonyms in field_keywords.items():
            if key in query:
                expanded.update(synonyms)

        # Add alphanumeric tokens (e.g., API, SLA, 128GB)
        for token in re.findall(r"[A-Za-z0-9]+", query):
            if token not in stopwords and len(token) >= 2:
                expanded.add(token)

        return list(expanded)

    def retrieve(
        self, query: str, keywords: List[str] | None = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using hybrid search with parallel execution.

        This method:
        1. Runs vector search and keyword search in parallel
        2. Merges results using RRF (Reciprocal Rank Fusion)
        3. Fetches full chunk data for top results

        Args:
            query: Natural language query for vector search
            keywords: Optional keywords for keyword search.
                     If None, keywords will be auto-extracted from query.

        Returns:
            List of RetrievalResult sorted by RRF relevance score
        """
        # Auto-extract keywords if not provided
        if keywords is None:
            keywords = self.extract_keywords_from_query(query)
            logger.debug(f"Auto-extracted keywords: {keywords}")

        # Run searches in parallel for better performance
        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            vector_future = executor.submit(self._vector_search, query)
            keyword_future = executor.submit(self._keyword_search, keywords)

            vector_results = vector_future.result()
            keyword_results = keyword_future.result()

        logger.debug(
            f"Vector search returned {len(vector_results)} results, "
            f"Keyword search returned {len(keyword_results)} results"
        )

        # Merge using RRF
        if keyword_results:
            merged = self.rrf.fuse(vector_results, keyword_results)
        else:
            # Fallback to vector-only results with empty source info
            merged = [
                (
                    doc_id,
                    1.0 / (self.rrf.k + rank),
                    {"vector": {"rank": rank, "score": score}},
                )
                for rank, (doc_id, score) in enumerate(vector_results)
            ]

        # Fetch full documents with scores
        merged_with_scores = merged[: self.top_k]
        return self._fetch_chunks(merged_with_scores)

    def _fetch_chunks(
        self,
        merged_results: List[Tuple[str, float, Dict[str, dict]]],
    ) -> List[RetrievalResult]:
        """
        Fetch full chunk data by IDs with detailed source information.

        Args:
            merged_results: List of (chunk_id, rrf_score, source_scores) from RRF fusion

        Returns:
            List of RetrievalResult with complete chunk data
        """
        if not merged_results:
            return []

        # Extract chunk IDs and create scores lookup
        chunk_ids = [doc_id for doc_id, _, _ in merged_results]
        scores_dict = {
            doc_id: (rrf_score, sources)
            for doc_id, rrf_score, sources in merged_results
        }

        try:
            with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT chunk_id::text, text_raw, page_idx, embedding
                        FROM chunks
                        WHERE chunk_id = ANY(%s::uuid[])
                        """,
                        (chunk_ids,),
                    )

                    rows = {row[0]: row for row in cur.fetchall()}

                    # Maintain order from merged results
                    results = []
                    for chunk_id in chunk_ids:
                        if chunk_id in rows:
                            row = rows[chunk_id]
                            rrf_score, sources = scores_dict[chunk_id]

                            # Determine source type
                            source_types = list(sources.keys())
                            if len(source_types) == 2:
                                source = "hybrid"
                            elif "vector" in source_types:
                                source = "vector"
                            elif "keyword" in source_types:
                                source = "keyword"
                            else:
                                source = "unknown"

                            # Extract original scores if available
                            vector_score = sources.get("vector", {}).get("score")
                            keyword_score = sources.get("keyword", {}).get("score")

                            results.append(
                                RetrievalResult(
                                    chunk_id=row[0],
                                    text=row[1] or "",
                                    page_idx=row[2] or 0,
                                    score=rrf_score,
                                    source=source,
                                    vector_score=vector_score,
                                    keyword_score=keyword_score,
                                    embedding=row[3] if row[3] else None,
                                )
                            )

                    return results
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}", exc_info=True)
            return []
