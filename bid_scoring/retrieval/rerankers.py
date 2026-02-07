"""Optional rerankers used by hybrid retrieval.

Keep this module import-safe: all heavyweight deps are behind ImportError guards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import RetrievalResult

logger = logging.getLogger(__name__)

# Cross-encoder reranker support - graceful fallback if not installed
try:
    from sentence_transformers import CrossEncoder

    HAS_RERANKER = True
except ImportError:  # pragma: no cover
    HAS_RERANKER = False
    CrossEncoder = None  # type: ignore[assignment]

# ColBERT reranker support - graceful fallback if not installed
try:
    from ragatouille import RAGPretrainedModel

    HAS_COLBERT_RERANKER = True
except ImportError:  # pragma: no cover
    HAS_COLBERT_RERANKER = False
    RAGPretrainedModel = None  # type: ignore[assignment]


class Reranker:
    """Lightweight cross-encoder reranker adapter."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        if not HAS_RERANKER:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: uv add sentence-transformers"
            )

        self._model_name = model_name
        self._device = device
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> "CrossEncoder":
        if self._model is None:
            logger.debug("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_n: int = 5,
    ) -> List[RetrievalResult]:
        if not results:
            return results

        max_rerank = min(len(results), top_n * 2)
        candidates = results[:max_rerank]
        pairs = [(query, r.text) for r in candidates]

        try:
            model = self._load_model()
            scores = model.predict(pairs)

            scored_results = list(zip(candidates, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(
                "Reranked %d results, top score: %.4f",
                len(candidates),
                float(scored_results[0][1]),
            )

            return [r for r, _ in scored_results[:top_n]]
        except Exception as e:  # pragma: no cover
            logger.error("Reranking failed: %s", e, exc_info=True)
            return results[:top_n]

    def rerank_with_scores(
        self, query: str, results: List[RetrievalResult]
    ) -> List[Tuple[RetrievalResult, float]]:
        if not results:
            return []

        pairs = [(query, r.text) for r in results]
        model = self._load_model()
        scores = model.predict(pairs)
        out: List[Tuple[RetrievalResult, float]] = []
        for r, score in zip(results, scores):
            out.append((r, float(score)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out


class ColBERTReranker:
    """ColBERT (late interaction) based reranker adapter."""

    DEFAULT_MODEL = "colbert-ir/colbertv2.0"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        if not HAS_COLBERT_RERANKER:
            raise ImportError(
                "ragatouille is required for ColBERT reranking. "
                "Install with: uv add ragatouille"
            )
        self._model_name = model_name
        self._device = device
        self._model: Optional[Any] = None

    def _load_model(self) -> Any:
        if self._model is None:
            logger.debug("Loading ColBERT reranker model: %s", self._model_name)
            self._model = RAGPretrainedModel.from_pretrained(self._model_name)
        return self._model

    def rerank(
        self, query: str, results: List[RetrievalResult], top_n: int = 5
    ) -> List[RetrievalResult]:
        if not results:
            return results

        max_rerank = min(len(results), top_n * 2)
        candidates = results[:max_rerank]
        documents = [r.text for r in candidates]

        if not any(documents):
            return candidates[:top_n]

        try:
            model = self._load_model()
            ranked = model.rerank(
                query=query, documents=documents, k=min(top_n, len(candidates))
            )

            if not ranked:
                return candidates[:top_n]

            text_to_indices: Dict[str, List[int]] = {}
            for idx, item in enumerate(candidates):
                text_to_indices.setdefault(item.text, []).append(idx)

            used: Set[int] = set()
            ordered: List[RetrievalResult] = []
            for row in ranked:
                content = row.get("content", "") if isinstance(row, dict) else ""
                score = row.get("score") if isinstance(row, dict) else None
                candidate_indices = text_to_indices.get(content, [])
                target_idx = next((i for i in candidate_indices if i not in used), None)
                if target_idx is None:
                    continue
                selected = candidates[target_idx]
                if score is not None:
                    selected.rerank_score = float(score)
                ordered.append(selected)
                used.add(target_idx)

            for idx, item in enumerate(candidates):
                if idx not in used:
                    ordered.append(item)

            return ordered[:top_n]
        except Exception as e:  # pragma: no cover
            logger.error("ColBERT reranking failed: %s", e, exc_info=True)
            return results[:top_n]
