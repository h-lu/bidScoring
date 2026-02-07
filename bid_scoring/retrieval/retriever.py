from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import psycopg

from .cache import LRUCache
from .config import FieldKeywordsDict, SynonymIndexDict, build_synonym_index, load_retrieval_config
from .fetch import fetch_chunks
from . import rerankers as _rerankers
from .rrf import DEFAULT_RRF_K, ReciprocalRankFusion
from .search_keyword import keyword_search_fulltext, keyword_search_legacy
from .search_vector import vector_search
from .types import MergedChunk, RetrievalMetrics, RetrievalResult


logger = logging.getLogger(__name__)

# Default pgvector is 40; 100 is a decent recall-oriented baseline.
DEFAULT_HNSW_EF_SEARCH = 100

# Keep small: vector+keyword in parallel.
MAX_SEARCH_WORKERS = 2


try:
    from psycopg_pool import ConnectionPool

    HAS_CONNECTION_POOL = True
except ImportError:  # pragma: no cover
    HAS_CONNECTION_POOL = False
    ConnectionPool = None


class HybridRetriever:
    """Hybrid retriever: vector search + keyword search + RRF fusion."""

    def __init__(
        self,
        version_id: str,
        settings: dict,
        top_k: int = 10,
        rrf_k: int = DEFAULT_RRF_K,
        config_path: str | None = None,
        extra_stopwords: Set[str] | None = None,
        extra_field_keywords: Dict[str, List[str]] | None = None,
        use_connection_pool: bool = True,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        enable_cache: bool = False,
        cache_size: int = 1000,
        use_or_semantic: bool = True,
        enable_rerank: bool = False,
        rerank_backend: Literal["cross_encoder", "colbert"] = "cross_encoder",
        rerank_model: str | None = None,
        rerank_top_n: int | None = None,
        enable_dynamic_weights: bool = False,
        enable_metrics: bool = False,
    ):
        if not version_id:
            raise ValueError("version_id cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.version_id = version_id
        self.settings = settings
        self.top_k = top_k

        self._hnsw_ef_search = max(DEFAULT_HNSW_EF_SEARCH, hnsw_ef_search, top_k * 2)
        self._use_or_semantic = use_or_semantic

        self._default_vector_weight = vector_weight
        self._default_keyword_weight = keyword_weight
        self.rrf = ReciprocalRankFusion(
            k=rrf_k, vector_weight=vector_weight, keyword_weight=keyword_weight
        )

        # Reranker
        self._enable_rerank = enable_rerank
        self._rerank_backend = rerank_backend
        self._rerank_top_n = rerank_top_n or top_k
        self._reranker: object | None = None
        if enable_rerank:
            if rerank_backend == "cross_encoder":
                model_name = rerank_model or _rerankers.Reranker.DEFAULT_MODEL
                if _rerankers.HAS_RERANKER:
                    self._reranker = _rerankers.Reranker(model_name=model_name)
                else:
                    logger.warning(
                        "Reranking enabled (cross_encoder) but sentence-transformers not installed. "
                        "Install with: uv add sentence-transformers"
                    )
                    self._enable_rerank = False
            elif rerank_backend == "colbert":
                model_name = rerank_model or _rerankers.ColBERTReranker.DEFAULT_MODEL
                if _rerankers.HAS_COLBERT_RERANKER:
                    self._reranker = _rerankers.ColBERTReranker(model_name=model_name)
                else:
                    logger.warning(
                        "Reranking enabled (colbert) but ragatouille not installed. "
                        "Install with: uv add ragatouille"
                    )
                    self._enable_rerank = False
            else:
                logger.warning(
                    "Unknown rerank_backend '%s' (expected 'cross_encoder'|'colbert'); disabling rerank.",
                    rerank_backend,
                )
                self._enable_rerank = False

        # Optional features kept for backwards compatibility (not heavily used yet).
        self._enable_dynamic_weights = enable_dynamic_weights
        self._enable_metrics = enable_metrics
        self._metrics_history: List[RetrievalMetrics] = []

        self._cache: LRUCache | None = LRUCache(cache_size) if enable_cache else None

        # Connection pool
        self._pool: ConnectionPool | None = None
        if use_connection_pool and HAS_CONNECTION_POOL:
            try:
                self._pool = ConnectionPool(
                    settings["DATABASE_URL"],
                    min_size=pool_min_size,
                    max_size=pool_max_size,
                    max_idle=300,
                    max_lifetime=3600,
                    open=True,
                )
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "Failed to initialize connection pool: %s. Using direct connections.",
                    e,
                )
        elif use_connection_pool and not HAS_CONNECTION_POOL:
            logger.warning("psycopg-pool not installed. Install with: uv add psycopg-pool")

        # Config
        config = load_retrieval_config(config_path)

        self._stopwords: Set[str] = set(config.get("stopwords", []))
        if extra_stopwords:
            self._stopwords.update(extra_stopwords)

        self._field_keywords: FieldKeywordsDict = dict(config.get("field_keywords", {}))
        if extra_field_keywords:
            for key, synonyms in extra_field_keywords.items():
                all_synonyms = [key] + [s for s in synonyms if s != key]
                if key in self._field_keywords:
                    existing = set(self._field_keywords[key])
                    self._field_keywords[key].extend(
                        [s for s in all_synonyms if s not in existing]
                    )
                else:
                    self._field_keywords[key] = list(all_synonyms)

        for key in list(self._field_keywords.keys()):
            if key not in self._field_keywords[key]:
                self._field_keywords[key].insert(0, key)

        self._synonym_index: SynonymIndexDict = build_synonym_index(self._field_keywords)

    def _get_connection(self):
        if self._pool:
            return self._pool.connection()
        return psycopg.connect(self.settings["DATABASE_URL"])

    def close(self) -> None:
        if self._pool:
            self._pool.close()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def stopwords(self) -> Set[str]:
        return self._stopwords.copy()

    @property
    def field_keywords(self) -> Dict[str, List[str]]:
        return self._field_keywords.copy()

    def add_stopwords(self, words: Set[str]) -> None:
        self._stopwords.update(words)

    def add_field_keywords(self, keywords: Dict[str, List[str]]) -> None:
        for key, synonyms in keywords.items():
            all_synonyms = [key] + [s for s in synonyms if s != key]
            if key in self._field_keywords:
                existing = set(self._field_keywords[key])
                self._field_keywords[key].extend([s for s in all_synonyms if s not in existing])
            else:
                self._field_keywords[key] = list(all_synonyms)
        self._synonym_index = build_synonym_index(self._field_keywords)

    def _analyze_query_type(self, query: str) -> str:
        technical_pattern = r"\b[A-Z]{2,}\b|\b\d+[A-Za-z]+\b"
        technical_matches = len(re.findall(technical_pattern, query))
        has_chinese_tech_terms = any(term in query for term in self._field_keywords.keys())
        if technical_matches >= 2 or has_chinese_tech_terms:
            return "technical"
        if len(query) > 50:
            return "long"
        return "standard"

    def _adjust_weights_for_query(self, query: str) -> Tuple[float, float]:
        if not self._enable_dynamic_weights:
            return self._default_vector_weight, self._default_keyword_weight
        query_type = self._analyze_query_type(query)
        if query_type == "technical":
            return 0.7, 1.3
        if query_type == "long":
            return 1.3, 0.7
        return self._default_vector_weight, self._default_keyword_weight

    def _generate_cache_key(self, query: str, keywords: List[str] | None) -> str:
        key_data = f"{self.version_id}:{query}:{keywords}:{self.top_k}"
        return sha256(key_data.encode()).hexdigest()

    def clear_cache(self) -> None:
        if self._cache:
            self._cache.clear()

    def get_cache_stats(self) -> dict:
        if not self._cache:
            return {"enabled": False, "size": 0, "capacity": 0}
        return {
            "enabled": True,
            "size": len(self._cache._cache),
            "capacity": self._cache.capacity,
        }

    # Delegated DB operations (kept as methods for compatibility / easy mocking).
    def _vector_search(self, query: str):
        return vector_search(self, query)

    def _keyword_search_fulltext(self, keywords: List[str], use_or_semantic: bool = True):
        return keyword_search_fulltext(self, keywords, use_or_semantic=use_or_semantic)

    def _keyword_search_legacy(self, keywords: List[str]):
        return keyword_search_legacy(self, keywords)

    def extract_keywords_from_query(self, query: str) -> List[str]:
        expanded: Set[str] = set()

        for term, key in self._synonym_index.items():
            if term in query:
                expanded.update(self._field_keywords[key])

        for token in re.findall(r"[A-Za-z0-9]+", query):
            if token not in self._stopwords and len(token) >= 2:
                expanded.add(token)
                if token in self._synonym_index:
                    key = self._synonym_index[token]
                    expanded.update(self._field_keywords[key])

        return list(expanded)

    def retrieve(self, query: str, keywords: List[str] | None = None) -> List[RetrievalResult]:
        if keywords is None:
            keywords = self.extract_keywords_from_query(query)

        cache_key: str | None = None
        if self._cache:
            cache_key = self._generate_cache_key(query, keywords)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            vector_future = executor.submit(self._vector_search, query)
            keyword_future = executor.submit(
                self._keyword_search_fulltext, keywords, self._use_or_semantic
            )
            vector_results = vector_future.result()
            keyword_results = keyword_future.result()

        if keyword_results:
            merged = self.rrf.fuse(vector_results, keyword_results)
        else:
            merged = [
                (
                    doc_id,
                    1.0 / (self.rrf.k + rank + 1),
                    {"vector": {"rank": rank, "score": score}},
                )
                for rank, (doc_id, score) in enumerate(vector_results)
            ]

        merged_with_scores: List[MergedChunk] = merged[: self.top_k]
        results = fetch_chunks(self, merged_with_scores)

        if self._enable_rerank and self._reranker and results:
            results = self._reranker.rerank(query, results, self._rerank_top_n)

        if self._cache and cache_key is not None:
            self._cache.put(cache_key, results)

        return results

    async def retrieve_async(
        self,
        query: str,
        keywords: List[str] | None = None,
        use_cache: bool = True,
    ) -> List[RetrievalResult]:
        cache_key: str | None = None
        if use_cache and self._cache:
            cache_key = self._generate_cache_key(query, keywords)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        if keywords is None:
            loop = asyncio.get_event_loop()
            keywords = await loop.run_in_executor(None, self.extract_keywords_from_query, query)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            vector_future = loop.run_in_executor(executor, self._vector_search, query)
            keyword_future = loop.run_in_executor(
                executor, self._keyword_search_fulltext, keywords, self._use_or_semantic
            )
            vector_results, keyword_results = await asyncio.gather(vector_future, keyword_future)

        if keyword_results:
            merged = self.rrf.fuse(vector_results, keyword_results)
        else:
            merged = [
                (
                    doc_id,
                    1.0 / (self.rrf.k + rank + 1),
                    {"vector": {"rank": rank, "score": score}},
                )
                for rank, (doc_id, score) in enumerate(vector_results)
            ]

        merged_with_scores: List[MergedChunk] = merged[: self.top_k]
        results = await loop.run_in_executor(None, fetch_chunks, self, merged_with_scores)

        if self._enable_rerank and self._reranker and results:
            results = await loop.run_in_executor(
                None, self._reranker.rerank, query, results, self._rerank_top_n
            )

        if use_cache and self._cache and cache_key is not None:
            self._cache.put(cache_key, results)

        return results

    async def close_async(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.close)
