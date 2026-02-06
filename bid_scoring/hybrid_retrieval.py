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

import asyncio
import logging
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psycopg
import yaml

from bid_scoring.embeddings import embed_single_text

# Reranker support - graceful fallback if not installed
try:
    from sentence_transformers import CrossEncoder

    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False
    CrossEncoder = None

# Connection pool support - graceful fallback if not installed
try:
    from psycopg_pool import ConnectionPool

    HAS_CONNECTION_POOL = True
except ImportError:
    HAS_CONNECTION_POOL = False
    ConnectionPool = None

# Type alias for field keywords dictionary
FieldKeywordsDict = Dict[str, List[str]]
SynonymIndexDict = Dict[str, str]  # synonym -> key mapping for bidirectional lookup

logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "retrieval_config.yaml"


class LRUCache:
    """Simple LRU Cache implementation using OrderedDict.

    This cache stores key-value pairs with a fixed capacity.
    When the capacity is exceeded, the least recently accessed item
    is evicted to make room for the new item.

    Attributes:
        capacity: Maximum number of items to store in the cache
    """

    def __init__(self, capacity: int = 1000):
        """Initialize the LRU cache.

        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """Get a value from the cache.

        Args:
            key: Cache key to look up

        Returns:
            The cached value, or None if not found
        """
        if key not in self._cache:
            return None
        # Move to end to mark as recently used
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self._cache:
            # Update existing value and mark as recently used
            self._cache.move_to_end(key)
        self._cache[key] = value
        # Evict oldest item if over capacity
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()


class Reranker:
    """轻量级 Cross-Encoder 重排序器。

    基于 FlashRank 理念，使用轻量级 cross-encoder 模型
    对初步检索结果进行精排，提升检索质量。

    References:
        - FlashRank: https://github.com/PrithivirajDamodaran/FlashRank
        - LangChain Contextual Compression

    Example:
        >>> reranker = Reranker()
        >>> results = reranker.rerank("查询", retrieval_results, top_n=5)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        """初始化重排序器。

        Args:
            model_name: Cross-encoder 模型名称
            device: 运行设备 ("cpu" 或 "cuda")

        Raises:
            ImportError: 如果 sentence-transformers 未安装
        """
        if not HAS_RERANKER:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: uv add sentence-transformers"
            )

        self._model_name = model_name
        self._device = device
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """延迟加载模型。"""
        if self._model is None:
            logger.debug(f"Loading reranker model: {self._model_name}")
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_n: int = 5,
    ) -> List["RetrievalResult"]:
        """对检索结果进行重排序。

        Args:
            query: 原始查询
            results: 初步检索结果列表
            top_n: 返回前 N 个结果

        Returns:
            重排序后的结果列表
        """
        if not results:
            return results

        # 限制重排序数量，避免过长的计算时间
        max_rerank = min(len(results), top_n * 2)
        candidates = results[:max_rerank]

        # 构建 query-document 对
        pairs = [(query, r.text) for r in candidates]

        try:
            model = self._load_model()
            scores = model.predict(pairs)

            # 按重排序分数排序
            scored_results = list(zip(candidates, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            logger.debug(
                f"Reranked {len(candidates)} results, "
                f"top score: {scored_results[0][1]:.4f}"
            )

            return [r for r, _ in scored_results[:top_n]]

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # 降级：返回原始排序的前 top_n 个
            return results[:top_n]

    def rerank_with_scores(
        self,
        query: str,
        results: List["RetrievalResult"],
    ) -> List[Tuple["RetrievalResult", float]]:
        """对检索结果进行重排序并返回分数。

        Args:
            query: 原始查询
            results: 初步检索结果列表

        Returns:
            (结果, 重排序分数) 的列表
        """
        if not results:
            return []

        pairs = [(query, r.text) for r in results]

        try:
            model = self._load_model()
            scores = model.predict(pairs)
            return list(zip(results, scores))
        except Exception as e:
            logger.error(f"Reranking with scores failed: {e}", exc_info=True)
            return [(r, r.score) for r in results]


def load_retrieval_config(config_path: str | Path | None = None) -> dict:
    """Load retrieval configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.
                    If None, uses the default path.

    Returns:
        Configuration dictionary containing stopwords and field_keywords.
        Returns empty configuration if file not found or invalid.

    Example:
        >>> config = load_retrieval_config()
        >>> print(config["stopwords"][:5])
        ['的', '了', '是', '在', '我']
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(f"Config file not found: {path}. Using empty configuration.")
        return {"stopwords": [], "field_keywords": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # Ensure required keys exist
        config.setdefault("stopwords", [])
        config.setdefault("field_keywords", {})

        logger.debug(f"Loaded retrieval config from {path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file {path}: {e}")
        return {"stopwords": [], "field_keywords": {}}
    except Exception as e:
        logger.error(f"Failed to load config file {path}: {e}")
        return {"stopwords": [], "field_keywords": {}}


def _build_synonym_index(field_keywords: FieldKeywordsDict) -> SynonymIndexDict:
    """Build a bidirectional synonym index for fast lookup.

    This creates a mapping from each synonym (including the key itself)
    back to the original key, enabling bidirectional keyword expansion.

    For example, with field_keywords = {"MRI": ["磁共振", "核磁共振"]}
    The synonym index will be:
        {"MRI": "MRI", "磁共振": "MRI", "核磁共振": "MRI"}

    This allows queries containing "核磁共振" to expand to all MRI-related terms.

    Args:
        field_keywords: Dictionary mapping keys to their synonym lists

    Returns:
        Dictionary mapping each term to its canonical key

    Performance:
        O(N * M) where N = number of keys, M = average synonyms per key
        Memory: O(total number of unique terms)
    """
    synonym_index: SynonymIndexDict = {}
    for key, synonyms in field_keywords.items():
        # Map the key itself to itself for consistent lookup
        synonym_index[key] = key
        # Map each synonym to the key
        for synonym in synonyms:
            # If synonym already exists, the first occurrence wins
            # This is deterministic behavior
            if synonym not in synonym_index:
                synonym_index[synonym] = key
            else:
                # Log warning about synonym collision
                existing_key = synonym_index[synonym]
                if existing_key != key:
                    logger.debug(
                        f"Synonym '{synonym}' maps to both '{existing_key}' and '{key}', "
                        f"using first mapping"
                    )
    return synonym_index


# DeepMind recommended value for RRF damping constant.
# This value balances the influence of top-ranked items vs. deep-ranked items.
DEFAULT_RRF_K = 60

# Default HNSW search expansion factor for better recall.
# Default pgvector is 40, we use 100 for better recall.
DEFAULT_HNSW_EF_SEARCH = 100

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
    rerank_score: float | None = None  # Cross-encoder rerank score (if reranking enabled)


@dataclass
class RetrievalMetrics:
    """检索性能指标。

    用于收集和分析检索操作的性能数据，包括各环节耗时、
    缓存命中率、返回结果数量等。

    Example:
        >>> metrics = RetrievalMetrics()
        >>> metrics.vector_search_time_ms = 45.2
        >>> print(metrics.total_time_ms)
    """

    # 各环节耗时 (毫秒)
    vector_search_time_ms: float = 0.0
    keyword_search_time_ms: float = 0.0
    rrf_fusion_time_ms: float = 0.0
    fetch_chunks_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # 结果统计
    vector_results_count: int = 0
    keyword_results_count: int = 0
    final_results_count: int = 0

    # 缓存和特征
    cache_hit: bool = False
    query_type: str = "unknown"  # "technical", "long", "standard"

    def to_dict(self) -> Dict[str, Union[float, int, bool, str]]:
        """转换为字典格式。"""
        return {
            "vector_search_time_ms": self.vector_search_time_ms,
            "keyword_search_time_ms": self.keyword_search_time_ms,
            "rrf_fusion_time_ms": self.rrf_fusion_time_ms,
            "fetch_chunks_time_ms": self.fetch_chunks_time_ms,
            "rerank_time_ms": self.rerank_time_ms,
            "total_time_ms": self.total_time_ms,
            "vector_results_count": self.vector_results_count,
            "keyword_results_count": self.keyword_results_count,
            "final_results_count": self.final_results_count,
            "cache_hit": self.cache_hit,
            "query_type": self.query_type,
        }


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for merging ranked lists.

    RRF formula: score = sum(weight / (k + rank)) for each list
    where k is a constant (default 60) to dampen the impact of ranking
    and weight allows adjusting the influence of each source

    Reference:
        Cormack, V., & Clarke, C. (2009). "Reciprocal Rank Fusion outperforms
        Condorcet and individual Rank Learning Methods"
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

        # Process vector search results with weight
        for rank, (doc_id, orig_score) in enumerate(vector_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = {}
            scores[doc_id] += self.vector_weight / (self.k + rank + 1)
            sources[doc_id]["vector"] = {"rank": rank, "score": orig_score}

        # Process keyword search results with weight
        for rank, (doc_id, orig_score) in enumerate(keyword_results):
            if doc_id not in scores:
                scores[doc_id] = 0.0
                sources[doc_id] = {}
            scores[doc_id] += self.keyword_weight / (self.k + rank + 1)
            sources[doc_id]["keyword"] = {"rank": rank, "score": orig_score}

        # Sort by RRF score descending and include source information
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, sources[doc_id]) for doc_id, score in sorted_results]


class HybridRetriever:
    """
    Hybrid retriever combining vector and keyword search with RRF fusion.

    This implementation follows industry best practices (from context7/pgvector):
    - PostgreSQL full-text search with tsvector + GIN index (10-50x faster than ILIKE)
    - Parallel execution of vector and keyword searches
    - RRF (Reciprocal Rank Fusion) for result merging
    - Detailed score tracking for debugging
    - Configurable stopwords and field keywords from external files

    References:
        - https://github.com/pgvector/pgvector#hybrid-search
        - https://docs.paradedb.com/blog/hybrid-search

    Usage:
        # Basic usage with default config
        retriever = HybridRetriever(version_id="xxx", settings=settings)
        results = retriever.retrieve("培训时长")

        # With custom config file
        retriever = HybridRetriever(
            version_id="xxx",
            settings=settings,
            config_path="/path/to/custom_config.yaml"
        )

        # With extra stopwords and field keywords
        retriever = HybridRetriever(
            version_id="xxx",
            settings=settings,
            extra_stopwords={"自定义", "停用词"},
            extra_field_keywords={"新技术": ["AI", "人工智能", "机器学习"]}
        )
    """

    def __init__(
        self,
        version_id: str,
        settings: dict,
        top_k: int = 10,
        rrf_k: int = DEFAULT_RRF_K,
        config_path: str | Path | None = None,
        extra_stopwords: Set[str] | None = None,
        extra_field_keywords: Dict[str, List[str]] | None = None,
        use_connection_pool: bool = True,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        hnsw_ef_search: int = 100,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0,
        enable_cache: bool = False,
        cache_size: int = 1000,
        use_or_semantic: bool = True,
        enable_rerank: bool = False,
        rerank_model: str = Reranker.DEFAULT_MODEL,
        rerank_top_n: int | None = None,
        enable_dynamic_weights: bool = False,
        enable_metrics: bool = False,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            version_id: Document version ID to search within
            settings: Configuration dictionary containing DATABASE_URL
            top_k: Number of top results to return
            rrf_k: RRF damping constant (default 60 as per DeepMind research)
            config_path: Path to YAML configuration file with stopwords and field_keywords.
                        If None, uses default path (data/retrieval_config.yaml).
            extra_stopwords: Additional stopwords to filter out during keyword extraction.
                           These are merged with config file stopwords.
            extra_field_keywords: Additional field keywords for synonym expansion.
                                These are merged with config file keywords.
            use_connection_pool: Whether to use connection pooling (default True)
            pool_min_size: Minimum connections in pool (default 2)
            pool_max_size: Maximum connections in pool (default 10)
            hnsw_ef_search: HNSW search expansion factor for better recall (default 100)
                            Higher values improve recall at the cost of performance.
                            Default pgvector value is 40, we use 100 for better recall.
            vector_weight: Weight for vector search results in RRF (default 1.0)
            keyword_weight: Weight for keyword search results in RRF (default 1.0)
            enable_cache: Whether to enable query result caching (default False)
            cache_size: Maximum number of cached query results (default 1000)
            use_or_semantic: Whether to use OR semantic for full-text search (default True)
                            - True: Match any keyword (higher recall)
                            - False: Match all keywords (higher precision)
            enable_rerank: Whether to enable cross-encoder reranking (default False)
            rerank_model: Cross-encoder model name for reranking
            rerank_top_n: Number of results to return after reranking
            enable_dynamic_weights: Whether to enable dynamic weight adjustment (default False)
            enable_metrics: Whether to collect retrieval metrics (default False)

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
        # 动态调整 ef_search：确保 ef_search >= top_k * 2，最小值为 100
        # 参考 pgvector 最佳实践：ef_search 应大于返回的最近邻数量
        self._hnsw_ef_search = max(100, hnsw_ef_search, top_k * 2)
        self._use_or_semantic = use_or_semantic
        self._default_vector_weight = vector_weight
        self._default_keyword_weight = keyword_weight
        self.rrf = ReciprocalRankFusion(
            k=rrf_k, vector_weight=vector_weight, keyword_weight=keyword_weight
        )

        # Initialize reranker
        self._enable_rerank = enable_rerank
        self._rerank_top_n = rerank_top_n or top_k
        self._reranker: Optional[Reranker] = None
        if enable_rerank:
            if HAS_RERANKER:
                self._reranker = Reranker(model_name=rerank_model)
                logger.debug(f"Initialized reranker with model: {rerank_model}")
            else:
                logger.warning(
                    "Reranking enabled but sentence-transformers not installed. "
                    "Install with: uv add sentence-transformers"
                )
                self._enable_rerank = False

        # Initialize dynamic weight adjustment
        self._enable_dynamic_weights = enable_dynamic_weights

        # Initialize metrics collection
        self._enable_metrics = enable_metrics
        self._metrics_history: List[RetrievalMetrics] = []

        # Initialize query result cache
        self._cache: LRUCache | None = LRUCache(cache_size) if enable_cache else None

        # Initialize connection pool
        self._pool: ConnectionPool | None = None
        if use_connection_pool and HAS_CONNECTION_POOL:
            try:
                self._pool = ConnectionPool(
                    settings["DATABASE_URL"],
                    min_size=pool_min_size,
                    max_size=pool_max_size,
                    max_idle=300,  # 5 minutes
                    max_lifetime=3600,  # 1 hour
                    open=True,  # Explicitly open the pool
                )
                logger.debug(
                    f"Initialized connection pool (min={pool_min_size}, max={pool_max_size})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize connection pool: {e}. "
                    "Using direct connections."
                )
        elif use_connection_pool and not HAS_CONNECTION_POOL:
            logger.warning(
                "psycopg-pool not installed. Install with: uv add psycopg-pool"
            )

        # Load configuration from file
        config = load_retrieval_config(config_path)

        # Initialize stopwords: config file + extra
        self._stopwords: Set[str] = set(config.get("stopwords", []))
        if extra_stopwords:
            self._stopwords.update(extra_stopwords)
            logger.debug(f"Added {len(extra_stopwords)} extra stopwords")

        # Initialize field keywords: config file + extra (extra takes precedence)
        self._field_keywords: FieldKeywordsDict = dict(config.get("field_keywords", {}))
        if extra_field_keywords:
            for key, synonyms in extra_field_keywords.items():
                # Ensure key itself is in the synonyms list for consistency
                all_synonyms = [key] + [s for s in synonyms if s != key]
                if key in self._field_keywords:
                    # Merge synonyms, avoid duplicates
                    existing = set(self._field_keywords[key])
                    new_synonyms = [s for s in all_synonyms if s not in existing]
                    self._field_keywords[key].extend(new_synonyms)
                    logger.debug(
                        f"Extended field keyword '{key}' with {len(new_synonyms)} new synonyms"
                    )
                else:
                    self._field_keywords[key] = list(all_synonyms)
                    logger.debug(
                        f"Added new field keyword '{key}' with {len(all_synonyms)} synonyms"
                    )

        # Ensure all existing field keywords have the key itself in their synonyms
        for key in list(self._field_keywords.keys()):
            if key not in self._field_keywords[key]:
                self._field_keywords[key].insert(0, key)

        # Build bidirectional synonym index for fast lookup
        # This enables queries containing synonyms to expand to all related terms
        self._synonym_index: SynonymIndexDict = _build_synonym_index(
            self._field_keywords
        )
        logger.debug(
            f"Built synonym index with {len(self._synonym_index)} terms "
            f"from {len(self._field_keywords)} keyword groups"
        )

    def _get_connection(self):
        """获取数据库连接（使用连接池或直接连接）"""
        if self._pool:
            return self._pool.connection()
        return psycopg.connect(self.settings["DATABASE_URL"])

    def close(self) -> None:
        """关闭连接池和资源"""
        if self._pool:
            self._pool.close()
            logger.debug("Connection pool closed")
            self._pool = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，自动关闭资源"""
        self.close()
        return False

    @property
    def stopwords(self) -> Set[str]:
        """Get the current set of stopwords."""
        return self._stopwords.copy()

    @property
    def field_keywords(self) -> Dict[str, List[str]]:
        """Get the current field keywords mapping."""
        return self._field_keywords.copy()

    def add_stopwords(self, words: Set[str]) -> None:
        """Add additional stopwords at runtime.

        Args:
            words: Set of words to add as stopwords
        """
        self._stopwords.update(words)
        logger.debug(f"Added {len(words)} stopwords at runtime")

    def add_field_keywords(self, keywords: Dict[str, List[str]]) -> None:
        """Add additional field keywords at runtime.

        This method updates both the field keywords dictionary and rebuilds
        the synonym index to include the new mappings.

        Args:
            keywords: Dictionary mapping core concepts to synonym lists
        """
        for key, synonyms in keywords.items():
            # Ensure key itself is in the synonyms list
            all_synonyms = [key] + [s for s in synonyms if s != key]
            if key in self._field_keywords:
                existing = set(self._field_keywords[key])
                new_synonyms = [s for s in all_synonyms if s not in existing]
                self._field_keywords[key].extend(new_synonyms)
            else:
                self._field_keywords[key] = list(all_synonyms)

        # Rebuild synonym index to include new keywords
        # This ensures bidirectional lookup works for newly added terms
        self._synonym_index = _build_synonym_index(self._field_keywords)
        logger.debug(
            f"Added {len(keywords)} field keyword entries at runtime, "
            f"synonym index now has {len(self._synonym_index)} terms"
        )

    def _analyze_query_type(self, query: str) -> str:
        """分析查询类型，用于动态权重调整。

        Args:
            query: 查询字符串

        Returns:
            查询类型: "technical", "long", 或 "standard"
        """
        # 检测技术术语（如 API, SLA, 128GB 等）
        technical_pattern = r"\b[A-Z]{2,}\b|\b\d+[A-Za-z]+\b"
        technical_matches = len(re.findall(technical_pattern, query))

        # 检测中文专业术语
        has_chinese_tech_terms = any(
            term in query for term in self._field_keywords.keys()
        )

        if technical_matches >= 2 or has_chinese_tech_terms:
            return "technical"
        elif len(query) > 50:
            return "long"
        else:
            return "standard"

    def _adjust_weights_for_query(self, query: str) -> Tuple[float, float]:
        """根据查询特征动态调整 RRF 权重。

        不同类型的查询适合不同的权重配置：
        - 技术查询: 增加关键词搜索权重（精确匹配更重要）
        - 长查询: 增加向量搜索权重（语义理解更重要）
        - 标准查询: 使用默认平衡权重

        Args:
            query: 查询字符串

        Returns:
            (vector_weight, keyword_weight) 元组
        """
        if not self._enable_dynamic_weights:
            return self._default_vector_weight, self._default_keyword_weight

        query_type = self._analyze_query_type(query)

        if query_type == "technical":
            # 技术查询：增加关键词搜索权重
            return 0.7, 1.3
        elif query_type == "long":
            # 长查询：增加向量搜索权重
            return 1.3, 0.7
        else:
            # 标准查询：平衡权重
            return self._default_vector_weight, self._default_keyword_weight

    def _generate_cache_key(self, query: str, keywords: List[str] | None) -> str:
        """Generate a cache key for a query.

        The cache key is a SHA256 hash of the combined query parameters:
        version_id, query text, keywords, and top_k. This ensures that
        identical queries produce identical keys while different parameters
        produce different keys.

        Args:
            query: The search query text
            keywords: Optional list of keywords

        Returns:
            SHA256 hex digest string (64 characters)
        """
        key_data = f"{self.version_id}:{query}:{keywords}:{self.top_k}"
        return sha256(key_data.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear all cached query results.

        This is a no-op if caching is not enabled.
        """
        if self._cache:
            self._cache.clear()
            logger.debug("Query result cache cleared")

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with keys:
                - enabled: Whether caching is enabled
                - size: Current number of cached items
                - capacity: Maximum cache capacity
        """
        if not self._cache:
            return {"enabled": False, "size": 0, "capacity": 0}
        return {
            "enabled": True,
            "size": len(self._cache._cache),
            "capacity": self._cache.capacity,
        }

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

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Set query timeout to prevent long-running queries
                    # 30 seconds should be sufficient for most searches
                    cur.execute("SET LOCAL statement_timeout = '30s'")

                    # Set HNSW search expansion factor for better recall
                    # Reference: https://github.com/pgvector/pgvector#hnsw
                    cur.execute("SET hnsw.ef_search = %s", (self._hnsw_ef_search,))

                    # Use cosine similarity: 1 - (embedding <=> query)
                    # gives similarity in [0, 1]
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

    def _keyword_search_fulltext(
        self, keywords: List[str], use_or_semantic: bool = True
    ) -> List[Tuple[str, float]]:
        """
        使用 PostgreSQL 全文搜索进行关键词匹配（基于 tsvector 和 GIN 索引）。

        相比 ILIKE 的优势：
        1. 利用 GIN 索引，查询速度提升 10-50 倍（来自 context7 最佳实践）
        2. 支持中文分词和语义匹配
        3. ts_rank_cd 提供更精确的相关性评分（cover density 算法适合短文本）

        SQL 参考：
            https://github.com/pgvector/pgvector#hybrid-search
            https://docs.paradedb.com/blog/hybrid-search

        Args:
            keywords: 关键词列表
            use_or_semantic: 是否使用 OR 语义（默认 True）
                - True: 匹配任一关键词（提高召回率）
                - False: 匹配所有关键词（提高精确率）

        Returns:
            List of (chunk_id, rank_score) tuples，rank_score 为相关性分数
        """
        if not keywords:
            return []

        # 使用 websearch 语法构建查询，提升容错性（context7 推荐）
        # OR 语义: "A OR B"
        # AND 语义: "A B"（空格在 websearch 中表示 AND）
        cleaned_keywords = [k.strip().replace('"', " ") for k in keywords if k.strip()]
        if not cleaned_keywords:
            return []
        joiner = " OR " if use_or_semantic else " "
        ts_query = joiner.join(cleaned_keywords)

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 设置查询超时保护
                    cur.execute("SET LOCAL statement_timeout = '30s'")

                    # 先检查 querytree，避免不可索引查询浪费一次大扫描
                    # querytree 返回 "T" 或空字符串表示没有可索引词元
                    cur.execute(
                        "SELECT querytree(websearch_to_tsquery('simple', %s))",
                        (ts_query,),
                    )
                    querytree_result = cur.fetchone()
                    querytree_text = (
                        str(querytree_result[0]).strip()
                        if querytree_result and querytree_result[0] is not None
                        else ""
                    )
                    if querytree_text in {"", "T"}:
                        logger.debug(
                            "Skip fulltext search due to non-indexable querytree: %s",
                            querytree_text,
                        )
                        return []

                    # 使用 websearch_to_tsquery 提升用户输入兼容性
                    # 使用 ts_rank_cd(..., 32) 对 rank 做归一化，便于与向量结果融合
                    cur.execute(
                        """
                        WITH q AS (
                            SELECT websearch_to_tsquery('simple', %s) AS tsq
                        )
                        SELECT
                            chunk_id::text,
                            ts_rank_cd(textsearch, q.tsq, 32) as rank
                        FROM chunks, q
                        WHERE version_id = %s
                          AND textsearch @@ q.tsq
                        ORDER BY rank DESC
                        LIMIT %s
                        """,
                        (ts_query, self.version_id, self.top_k * 2),
                    )
                    return [(row[0], float(row[1])) for row in cur.fetchall()]
        except psycopg.Error as e:
            logger.error(f"Fulltext search failed: {e}", exc_info=True)
            # 降级到旧的关键词搜索
            logger.warning("Falling back to legacy keyword search")
            return self._keyword_search_legacy(keywords)

    def _keyword_search_legacy(self, keywords: List[str]) -> List[Tuple[str, float]]:
        """
        遗留的关键词搜索方法（ILIKE 方式），作为降级方案保留。

        Args:
            keywords: List of keywords to search for

        Returns:
            List of (chunk_id, match_count) tuples
        """
        if not keywords:
            return []

        try:
            with self._get_connection() as conn:
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
        Extract keywords from natural language query with bidirectional synonym expansion.

        This method uses:
        1. Stopword filtering for Chinese (configurable via config file)
        2. Bidirectional synonym expansion - matches both keys and synonyms in query
        3. Alphanumeric token extraction

        The bidirectional expansion allows queries containing synonyms to match
        and expand to all related terms. For example:
        - Query "核磁共振参数" will expand to all MRI-related terms
        - Query "多层CT维修" will expand to all CT-related terms

        Args:
            query: Natural language query string

        Returns:
            List of extracted keywords with synonyms expanded
        """
        expanded = set()

        # Method 1: Check if any synonym (including keys) appears in query
        # This enables bidirectional lookup: synonym -> key -> all synonyms
        for term, key in self._synonym_index.items():
            if term in query:
                # Add all synonyms for the matched key
                expanded.update(self._field_keywords[key])

        # Method 2: Add alphanumeric tokens (e.g., API, SLA, 128GB)
        for token in re.findall(r"[A-Za-z0-9]+", query):
            if token not in self._stopwords and len(token) >= 2:
                expanded.add(token)
                # Also check if this token is in synonym index
                if token in self._synonym_index:
                    key = self._synonym_index[token]
                    expanded.update(self._field_keywords[key])

        return list(expanded)

    def retrieve(
        self, query: str, keywords: List[str] | None = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using hybrid search with parallel execution.

        This method:
        1. Checks cache for existing results (if caching enabled)
        2. Runs vector search and keyword search in parallel
        3. Merges results using RRF (Reciprocal Rank Fusion)
        4. Fetches full chunk data for top results
        5. Stores results in cache (if caching enabled)

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

        # Check cache if enabled
        if self._cache:
            cache_key = self._generate_cache_key(query, keywords)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result

        # Run searches in parallel for better performance
        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            vector_future = executor.submit(self._vector_search, query)
            # 使用新的全文搜索方法（基于 tsvector + GIN 索引）
            # 传递 use_or_semantic 参数控制 OR/AND 语义
            keyword_future = executor.submit(
                self._keyword_search_fulltext, keywords, self._use_or_semantic
            )

            vector_results = vector_future.result()
            keyword_results = keyword_future.result()

        logger.debug(
            f"Vector search returned {len(vector_results)} results, "
            f"Fulltext search returned {len(keyword_results)} results"
        )

        # Merge using RRF
        if keyword_results:
            merged = self.rrf.fuse(vector_results, keyword_results)
        else:
            # Fallback to vector-only results with empty source info
            merged = [
                (
                    doc_id,
                    1.0 / (self.rrf.k + rank + 1),
                    {"vector": {"rank": rank, "score": score}},
                )
                for rank, (doc_id, score) in enumerate(vector_results)
            ]

        # Fetch full documents with scores
        merged_with_scores = merged[: self.top_k]
        results = self._fetch_chunks(merged_with_scores)

        # Store in cache if enabled
        if self._cache:
            self._cache.put(cache_key, results)
            logger.debug(f"Cached results for query: {query[:50]}...")

        return results

    async def retrieve_async(
        self,
        query: str,
        keywords: List[str] | None = None,
        use_cache: bool = True,
    ) -> List[RetrievalResult]:
        """
        Async version of retrieve() using ThreadPoolExecutor.

        This method provides the same functionality as retrieve() but runs
        the I/O-bound operations in a thread pool for async compatibility.

        Args:
            query: Natural language query for vector search
            keywords: Optional keywords for keyword search.
                     If None, keywords will be auto-extracted from query.
            use_cache: Whether to use query result caching (default True).
                      Only effective if caching is enabled on retriever.

        Returns:
            List of RetrievalResult sorted by RRF relevance score

        Example:
            retriever = HybridRetriever(version_id="xxx", settings=settings)
            results = await retriever.retrieve_async("培训时长")
        """
        # Check cache if enabled
        if use_cache and self._cache:
            cache_key = self._generate_cache_key(query, keywords)
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for async query: {query[:50]}...")
                return cached_result

        # Auto-extract keywords if not provided
        if keywords is None:
            # Keyword extraction is CPU-bound, run in executor
            loop = asyncio.get_event_loop()
            keywords = await loop.run_in_executor(
                None, self.extract_keywords_from_query, query
            )
            logger.debug(f"Auto-extracted keywords (async): {keywords}")

        # Run searches in thread pool for I/O-bound operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
            vector_future = loop.run_in_executor(executor, self._vector_search, query)
            keyword_future = loop.run_in_executor(
                executor, self._keyword_search_fulltext, keywords, self._use_or_semantic
            )

            vector_results, keyword_results = await asyncio.gather(
                vector_future, keyword_future
            )

        logger.debug(
            f"Vector search returned {len(vector_results)} results, "
            f"Fulltext search returned {len(keyword_results)} results (async)"
        )

        # Merge using RRF (CPU-bound, but fast enough to run directly)
        if keyword_results:
            merged = self.rrf.fuse(vector_results, keyword_results)
        else:
            # Fallback to vector-only results with empty source info
            merged = [
                (
                    doc_id,
                    1.0 / (self.rrf.k + rank + 1),
                    {"vector": {"rank": rank, "score": score}},
                )
                for rank, (doc_id, score) in enumerate(vector_results)
            ]

        # Fetch full documents with scores (I/O-bound)
        merged_with_scores = merged[: self.top_k]
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._fetch_chunks, merged_with_scores
        )

        # Store in cache if enabled
        if use_cache and self._cache:
            self._cache.put(cache_key, results)
            logger.debug(f"Cached async results for query: {query[:50]}...")

        return results

    async def close_async(self) -> None:
        """
        Async close method for cleanup.

        This method provides an async interface to close resources,
        useful when using the retriever in async contexts.

        Example:
            retriever = HybridRetriever(...)
            try:
                results = await retriever.retrieve_async("query")
            finally:
                await retriever.close_async()
        """
        # Run close() in executor to make it async-friendly
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.close)

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
            with self._get_connection() as conn:
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
