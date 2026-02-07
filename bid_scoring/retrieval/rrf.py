"""Reciprocal Rank Fusion (RRF) implementation."""

from __future__ import annotations

from typing import Dict, List, Tuple

# DeepMind recommended value for RRF damping constant.
# This value balances the influence of top-ranked items vs. deep-ranked items.
DEFAULT_RRF_K = 60


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for merging ranked lists.

    RRF formula: score = sum(weight / (k + rank)) for each list.
    """

    def __init__(
        self,
        k: int = DEFAULT_RRF_K,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
    ):
        self.k = k
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def fuse(
        self,
        vector_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float, Dict[str, dict]]]:
        """Merge vector and keyword search results using RRF.

        Args:
            vector_results: List of (chunk_id, similarity_score) from vector search
            keyword_results: List of (chunk_id, match_score) from keyword search

        Returns:
            List of (chunk_id, rrf_score, source_scores) sorted by rrf_score desc.
        """
        scores: Dict[str, float] = {}
        sources: Dict[str, Dict[str, dict]] = {}

        for rank, (doc_id, orig_score) in enumerate(vector_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = {}
            scores[doc_id] += self.vector_weight / (self.k + rank + 1)
            sources[doc_id]["vector"] = {"rank": rank, "score": orig_score}

        for rank, (doc_id, orig_score) in enumerate(keyword_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = {}
            scores[doc_id] += self.keyword_weight / (self.k + rank + 1)
            sources[doc_id]["keyword"] = {"rank": rank, "score": orig_score}

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, sources[doc_id]) for doc_id, score in sorted_results]
