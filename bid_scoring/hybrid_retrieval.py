"""
Hybrid Retrieval Module for Bid Scoring

Combines vector similarity search with keyword matching using
Reciprocal Rank Fusion (RRF) for optimal retrieval performance.

References:
- DeepMind: "On the Theoretical Limitations of Embedding-Based Retrieval"
- Cormack et al.: "Reciprocal Rank Fusion outperforms Condorcet"
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import psycopg
from bid_scoring.embeddings import embed_single_text


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    chunk_id: str
    text: str
    page_idx: int
    score: float
    source: str  # "vector" or "keyword"
    embedding: List[float] | None = None


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for merging ranked lists.
    
    RRF formula: score = sum(1 / (k + rank)) for each list
    where k is a constant (default 60) to dampen the impact of ranking
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self, 
        *result_lists: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
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
        rrf_k: int = 60
    ):
        self.version_id = version_id
        self.settings = settings
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.rrf = ReciprocalRankFusion(k=rrf_k)
    
    def _vector_search(self, query: str) -> List[Tuple[str, float]]:
        """Perform vector similarity search"""
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
                    (query_emb, self.version_id, query_emb, self.top_k * 2)
                )
                return [(row[0], float(row[1])) for row in cur.fetchall()]
    
    def _keyword_search(
        self, 
        keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """Perform keyword search using ILIKE"""
        if not keywords:
            return []
        
        with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                # Build ILIKE conditions
                conditions = " OR ".join(
                    ["text_raw ILIKE %s"] * len(keywords)
                )
                params = [self.version_id] + [f"%{k}%" for k in keywords]
                
                cur.execute(
                    f"""
                    SELECT chunk_id::text, 
                           ts_rank(text_tsv, plainto_tsquery('chinese', %s)) as rank
                    FROM chunks
                    WHERE version_id = %s 
                      AND ({conditions})
                    ORDER BY rank DESC
                    LIMIT %s
                    """,
                    (" ".join(keywords), *params, self.top_k * 2)
                )
                return [(row[0], float(row[1] or 0)) for row in cur.fetchall()]
    
    def retrieve(
        self, 
        query: str, 
        keywords: List[str] | None = None
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
        
        # Fetch full documents
        return self._fetch_chunks([doc_id for doc_id, _ in merged[:self.top_k]])
    
    def _fetch_chunks(self, chunk_ids: List[str]) -> List[RetrievalResult]:
        """Fetch full chunk data by IDs"""
        if not chunk_ids:
            return []
        
        with psycopg.connect(self.settings["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chunk_id::text, text_raw, page_idx, embedding
                    FROM chunks
                    WHERE chunk_id = ANY(%s::uuid[])
                    """,
                    (chunk_ids,)
                )
                
                rows = {row[0]: row for row in cur.fetchall()}
                
                # Maintain order from merged results
                results = []
                for chunk_id in chunk_ids:
                    if chunk_id in rows:
                        row = rows[chunk_id]
                        results.append(RetrievalResult(
                            chunk_id=row[0],
                            text=row[1] or "",
                            page_idx=row[2] or 0,
                            score=0.0,  # Will be set by RRF
                            source="hybrid",
                            embedding=row[3] if row[3] else None
                        ))
                
                return results
