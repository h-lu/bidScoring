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
from pathlib import Path
from typing import Dict, List, Set, Tuple

import psycopg
import yaml

from bid_scoring.embeddings import embed_single_text

# Type alias for field keywords dictionary
FieldKeywordsDict = Dict[str, List[str]]
SynonymIndexDict = Dict[str, str]  # synonym -> key mapping for bidirectional lookup

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "retrieval_config.yaml"


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
    - Configurable stopwords and field keywords from external files

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
