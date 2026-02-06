"""
Hybrid Retrieval Module for Bid Scoring

Combines vector similarity search with keyword matching using
Reciprocal Rank Fusion (RRF) for optimal retrieval performance.

References:
- DeepMind: "On the Theoretical Limitations of Embedding-Based Retrieval"
- Cormack et al.: "Reciprocal Rank Fusion outperforms Condorcet"
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
import re
import psycopg
from bid_scoring.embeddings import embed_single_text


# DeepMind recommended value for RRF damping
DEFAULT_RRF_K = 60


@dataclass
class RetrievalResult:
    """Single retrieval result"""

    chunk_id: str
    text: str
    page_idx: int
    score: float
    source: str  # "vector", "keyword", or "hybrid"
    embedding: List[float] | None = None


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for merging ranked lists.

    RRF formula: score = sum(1 / (k + rank)) for each list
    where k is a constant (default 60) to dampen the impact of ranking
    """

    def __init__(self, k: int = DEFAULT_RRF_K):
        self.k = k

    def fuse(self, *result_lists: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Merge multiple ranked lists using RRF.

        Args:
            *result_lists: Variable number of (id, score) lists

        Returns:
            Merged list sorted by RRF score descending
        """
        scores: Dict[str, float] = {}

        for results in result_lists:
            for rank, (doc_id, _) in enumerate(results):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (self.k + rank + 1)

        # Sort by RRF score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """
    Hybrid retriever combining vector and keyword search.

    Usage:
        retriever = HybridRetriever(version_id="xxx", settings=settings)
        results = retriever.retrieve("培训时长", keywords=["培训", "时长", "天数"])
    """

    def __init__(
        self,
        version_id: str,
        settings: dict,
        top_k: int = 10,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        # Input validation
        if not version_id:
            raise ValueError("version_id cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if vector_weight < 0 or keyword_weight < 0:
            raise ValueError("weights must be non-negative")

        self.version_id = version_id
        self.settings = settings
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf = ReciprocalRankFusion(k=rrf_k)

    def _vector_search(self, query: str) -> List[Tuple[str, float]]:
        """Perform vector similarity search"""
        try:
            query_emb = embed_single_text(query)

            with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
                with conn.cursor() as cur:
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
        except Exception:
            # Return empty list on error
            return []

    def _keyword_search(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """Perform keyword search using ILIKE"""
        if not keywords:
            return []

        try:
            with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
                with conn.cursor() as cur:
                    # Build ILIKE conditions
                    conditions = " OR ".join(["text_raw ILIKE %s"] * len(keywords))

                    # Count keyword matches as a simple relevance score
                    match_scores = " + ".join(
                        [
                            "CASE WHEN text_raw ILIKE %s THEN 1 ELSE 0 END"
                            for _ in keywords
                        ]
                    )

                    # 正确的参数顺序: match_scores 的 %s -> version_id -> conditions 的 %s -> limit
                    params = [f"%{k}%" for k in keywords]  # match_scores
                    params = params + [self.version_id]  # version_id
                    params = params + [f"%{k}%" for k in keywords]  # conditions
                    params = params + [self.top_k * 2]  # limit

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
            # Return empty list on error
            print(f"Keyword search error: {e}")
            return []

    def extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract keywords from natural language query.

        Uses simple heuristics:
        1. Remove common stopwords
        2. Keep nouns and key terms
        3. Split compound terms

        Args:
            query: Natural language query string

        Returns:
            List of extracted keywords
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
        }

        # Split and filter - extract 2+ character words
        words = []
        # Simple approach: scan for field-specific keywords
        for i in range(len(query) - 1):
            bigram = query[i : i + 2]
            if len(bigram) == 2 and bigram not in stopwords:
                words.append(bigram)

        # Field-specific keyword expansion
        field_keywords = {
            "培训": ["培训", "训练", "教学", "指导"],
            "时长": ["时长", "时间", "天数", "小时", "工作日", "周期"],
            "计划": ["计划", "安排", "课程", "大纲", "内容"],
            "对象": ["对象", "人员", "受训", "用户", "学员"],
            "老师": ["老师", "讲师", "授课", "教师", "专家"],
            "资质": ["资质", "资格", "认证", "证书", "背景"],
            "响应": ["响应", "反应", "回复", "到达", "到场"],
            "质保": ["质保", "保修", "质量保证", "保修期"],
            "配件": ["配件", "备件", "耗材", "零件", "部件"],
            "服务": ["服务", "支持", "维护", "售后"],
            "费用": ["费用", "收费", "价格", "成本", "金额"],
        }

        # Expand with synonyms
        expanded = set(words)
        for key, synonyms in field_keywords.items():
            if key in query:
                expanded.update(synonyms)

        # Add alphanumeric tokens (e.g., API, SLA, 128GB)
        for token in re.findall(r"[A-Za-z0-9]+", query):
            if token not in stopwords:
                expanded.add(token)

        return list(expanded)

    def retrieve(
        self, query: str, keywords: List[str] | None = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using hybrid search.

        Args:
            query: Natural language query for vector search
            keywords: Optional keywords for keyword search

        Returns:
            List of RetrievalResult sorted by relevance
        """
        # Run searches
        vector_results = self._vector_search(query)

        keyword_results = []
        if keywords:
            keyword_results = self._keyword_search(keywords)

        # Merge using RRF
        if keyword_results:
            merged = self.rrf.fuse(vector_results, keyword_results)
        else:
            merged = vector_results

        # Fetch full documents with scores
        merged_with_scores = merged[: self.top_k]
        return self._fetch_chunks(merged_with_scores, vector_results, keyword_results)

    def _fetch_chunks(
        self,
        merged_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
    ) -> List[RetrievalResult]:
        """Fetch full chunk data by IDs"""
        if not merged_results:
            return []

        # Extract chunk IDs and create scores lookup
        chunk_ids = [doc_id for doc_id, _ in merged_results]
        scores_dict = dict(merged_results)

        # Determine source type
        source = "vector"
        if keyword_results:
            source = "hybrid" if vector_results else "keyword"

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
                            results.append(
                                RetrievalResult(
                                    chunk_id=row[0],
                                    text=row[1] or "",
                                    page_idx=row[2] or 0,
                                    score=scores_dict.get(chunk_id, 0.0),
                                    source=source,
                                    embedding=row[3] if row[3] else None,
                                )
                            )

                    return results
        except Exception:
            # Return empty list on error
            return []
