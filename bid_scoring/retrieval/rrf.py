from __future__ import annotations

from typing import Dict, List, Tuple

from .types import RrfSources


# DeepMind recommended value for RRF damping constant.
DEFAULT_RRF_K = 60


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) for merging ranked lists."""

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
    ) -> List[Tuple[str, float, RrfSources]]:
        scores: Dict[str, float] = {}
        sources: Dict[str, RrfSources] = {}

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
