import pytest
from bid_scoring.hybrid_retrieval import (
    HybridRetriever,
    ReciprocalRankFusion,
    DEFAULT_RRF_K,
)


def test_reciprocal_rank_fusion_basic():
    """Test RRF merging of two result lists"""
    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]
    keyword_results = [("chunk_2", 1.0), ("chunk_4", 0.9), ("chunk_1", 0.8)]

    rrf = ReciprocalRankFusion(k=60)
    merged = rrf.fuse(vector_results, keyword_results)

    assert len(merged) == 4  # All unique chunks
    # Top result should be in both lists (chunk_1 or chunk_2)
    assert merged[0][0] in ["chunk_1", "chunk_2"]
    # Verify structure: (doc_id, rrf_score, sources_dict)
    assert len(merged[0]) == 3
    assert isinstance(merged[0][2], dict)  # sources info


def test_hybrid_retriever_initialization():
    """Test HybridRetriever can be initialized with settings"""
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test-version", settings=settings, top_k=5)

    assert retriever.version_id == "test-version"
    assert retriever.top_k == 5


def test_rrf_empty_results():
    """Test RRF with empty result lists"""
    rrf = ReciprocalRankFusion(k=60)

    # Both empty
    merged = rrf.fuse([], [])
    assert merged == []

    # One empty (vector only)
    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8)]
    merged = rrf.fuse(vector_results, [])
    assert len(merged) == 2
    assert merged[0][0] == "chunk_1"
    # Verify sources contains vector info
    assert "vector" in merged[0][2]


def test_rrf_single_result():
    """Test RRF with single result lists"""
    rrf = ReciprocalRankFusion(k=60)

    vector_results = [("chunk_1", 0.9)]
    keyword_results = [("chunk_1", 0.8)]

    merged = rrf.fuse(vector_results, keyword_results)
    assert len(merged) == 1
    assert merged[0][0] == "chunk_1"
    # Score should be 1/(60+0+1) + 1/(60+0+1) = 2/61
    assert merged[0][1] == pytest.approx(2 / 61, rel=1e-6)
    # Verify both sources are tracked
    assert "vector" in merged[0][2]
    assert "keyword" in merged[0][2]


def test_rrf_identical_lists():
    """Test RRF with identical result lists"""
    rrf = ReciprocalRankFusion(k=60)

    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]
    keyword_results = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]

    merged = rrf.fuse(vector_results, keyword_results)
    assert len(merged) == 3
    # Same order as input since ranks are identical
    assert merged[0][0] == "chunk_1"
    assert merged[1][0] == "chunk_2"
    assert merged[2][0] == "chunk_3"
    # Scores should be doubled
    assert merged[0][1] == pytest.approx(2 / 61, rel=1e-6)


def test_rrf_completely_different_lists():
    """Test RRF with no overlapping results"""
    rrf = ReciprocalRankFusion(k=60)

    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8)]
    keyword_results = [("chunk_3", 0.9), ("chunk_4", 0.8)]

    merged = rrf.fuse(vector_results, keyword_results)
    assert len(merged) == 4
    # First items from each list should have same score
    assert merged[0][1] == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_tracks_source_scores():
    """Test that RRF tracks original scores from each source"""
    rrf = ReciprocalRankFusion(k=60)

    vector_results = [("chunk_1", 0.95), ("chunk_2", 0.85)]
    keyword_results = [("chunk_1", 5.0), ("chunk_3", 3.0)]

    merged = rrf.fuse(vector_results, keyword_results)

    # Find chunk_1 in results
    chunk_1_result = next(r for r in merged if r[0] == "chunk_1")
    sources = chunk_1_result[2]

    # Verify original scores are preserved
    assert sources["vector"]["score"] == 0.95
    assert sources["vector"]["rank"] == 0
    assert sources["keyword"]["score"] == 5.0
    assert sources["keyword"]["rank"] == 0


def test_default_rrf_k_constant():
    """Test that DEFAULT_RRF_K is defined and equals 60"""
    assert DEFAULT_RRF_K == 60

    # Verify RRF uses the default
    rrf = ReciprocalRankFusion(k=DEFAULT_RRF_K)
    assert rrf.k == 60


def test_input_validation_empty_version_id():
    """Test that empty version_id raises ValueError"""
    from bid_scoring.config import load_settings

    settings = load_settings()
    with pytest.raises(ValueError, match="version_id cannot be empty"):
        HybridRetriever(version_id="", settings=settings)


def test_input_validation_invalid_top_k():
    """Test that invalid top_k raises ValueError"""
    from bid_scoring.config import load_settings

    settings = load_settings()
    with pytest.raises(ValueError, match="top_k must be positive"):
        HybridRetriever(version_id="test", settings=settings, top_k=0)

    with pytest.raises(ValueError, match="top_k must be positive"):
        HybridRetriever(version_id="test", settings=settings, top_k=-1)


def test_input_validation_rrf_k():
    """Test that custom rrf_k is accepted"""
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, rrf_k=40)
    assert retriever.rrf.k == 40


def test_keyword_extraction():
    """Test keyword extraction from Chinese queries"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Test training-related query
    keywords = retriever.extract_keywords_from_query("培训时长是多少天")
    assert "培训" in keywords
    assert "时长" in keywords

    # Test service-related query
    keywords = retriever.extract_keywords_from_query("售后服务响应时间")
    assert "服务" in keywords


def test_keyword_extraction_includes_alphanumeric_tokens():
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )
    keywords = retriever.extract_keywords_from_query("API响应时间")
    assert "API" in keywords


def test_keyword_extraction_filters_stopwords():
    """Test that question words like '多少' are filtered"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )
    keywords = retriever.extract_keywords_from_query("培训时长是多少")
    # "多少" should be filtered as a stopword
    assert "多少" not in keywords
    # But field keywords should still be present
    assert "培训" in keywords
    assert "时长" in keywords


def test_config_loading_from_file():
    """Test loading configuration from default YAML file"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, load_retrieval_config

    # Test loading default config
    config = load_retrieval_config()
    assert "stopwords" in config
    assert "field_keywords" in config
    assert len(config["stopwords"]) > 0
    assert len(config["field_keywords"]) > 0

    # Test retriever with default config
    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )
    assert len(retriever.stopwords) > 0
    assert len(retriever.field_keywords) > 0


def test_extra_stopwords():
    """Test adding extra stopwords via constructor"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5,
        extra_stopwords={"自定义", "测试词"},
    )

    # Verify extra stopwords were added
    assert "自定义" in retriever.stopwords
    assert "测试词" in retriever.stopwords

    # Verify the stopword is filtered in keyword extraction
    keywords = retriever.extract_keywords_from_query("自定义测试")
    assert "自定义" not in keywords


def test_extra_field_keywords():
    """Test adding extra field keywords via constructor"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5,
        extra_field_keywords={
            "云原生": ["Kubernetes", "K8s", "容器", "Docker"],
            "AI": ["人工智能", "深度学习", "机器学习"],
        },
    )

    # Verify extra field keywords were added
    assert "云原生" in retriever.field_keywords
    assert "AI" in retriever.field_keywords
    assert "Kubernetes" in retriever.field_keywords["云原生"]

    # Test keyword expansion with extra keywords
    keywords = retriever.extract_keywords_from_query("云原生架构")
    assert "Kubernetes" in keywords
    assert "容器" in keywords


def test_merge_field_keywords():
    """Test that extra field keywords merge with config file keywords"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5,
        extra_field_keywords={
            "培训": ["新词1", "新词2"],  # Should merge with existing "培训"
        },
    )

    # Verify existing synonyms are preserved
    assert "培训" in retriever.field_keywords
    assert "训练" in retriever.field_keywords["培训"]  # From config file
    assert "新词1" in retriever.field_keywords["培训"]  # From extra


def test_runtime_add_stopwords():
    """Test adding stopwords at runtime"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Add stopwords at runtime
    retriever.add_stopwords({"临时停用词", "另一个"})

    assert "临时停用词" in retriever.stopwords
    assert "另一个" in retriever.stopwords


def test_runtime_add_field_keywords():
    """Test adding field keywords at runtime"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Add field keywords at runtime
    retriever.add_field_keywords({"区块链": ["Blockchain", "分布式账本", "智能合约"]})

    assert "区块链" in retriever.field_keywords
    assert "Blockchain" in retriever.field_keywords["区块链"]

    # Test expansion works with runtime-added keywords
    keywords = retriever.extract_keywords_from_query("区块链技术")
    assert "Blockchain" in keywords


def test_custom_config_path():
    """Test loading from a custom config file path"""
    import tempfile
    import os
    from bid_scoring.hybrid_retrieval import HybridRetriever, load_retrieval_config

    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            'stopwords:\n  - "测试停用词"\nfield_keywords:\n  测试概念:\n    - "同义词1"\n    - "同义词2"\n'
        )
        temp_path = f.name

    try:
        # Test loading custom config
        config = load_retrieval_config(temp_path)
        assert "测试停用词" in config["stopwords"]
        assert "测试概念" in config["field_keywords"]

        # Test retriever with custom config
        retriever = HybridRetriever(
            version_id="test",
            settings={"DATABASE_URL": "postgresql://test"},
            top_k=5,
            config_path=temp_path,
        )
        assert "测试停用词" in retriever.stopwords
        assert "测试概念" in retriever.field_keywords
    finally:
        os.unlink(temp_path)


def test_missing_config_file():
    """Test handling of missing config file"""
    from bid_scoring.hybrid_retrieval import load_retrieval_config

    # Should return empty config without error
    config = load_retrieval_config("/nonexistent/path/config.yaml")
    assert config["stopwords"] == []
    assert config["field_keywords"] == {}


def test_properties_return_copies():
    """Test that stopwords and field_keywords properties return copies"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Get copies
    stopwords = retriever.stopwords
    field_keywords = retriever.field_keywords

    # Modify the copies
    stopwords.add("新词")
    field_keywords["新键"] = ["值"]

    # Original should be unchanged
    assert "新词" not in retriever.stopwords
    assert "新键" not in retriever.field_keywords


def test_bidirectional_synonym_expansion():
    """Test that synonyms in query can expand to all related terms"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Query with synonym (not key) should expand to all related terms
    keywords = retriever.extract_keywords_from_query("核磁共振设备")
    assert "MRI" in keywords  # key
    assert "核磁" in keywords  # synonym
    assert "磁共振" in keywords  # synonym
    assert "MR" in keywords  # synonym

    # Query with another synonym
    keywords = retriever.extract_keywords_from_query("计算机断层扫描仪")
    assert "CT" in keywords
    assert "螺旋CT" in keywords
    assert "多层CT" in keywords

    # Query with abbreviation
    keywords = retriever.extract_keywords_from_query("数字X光机")
    assert "DR" in keywords
    assert "数字化摄影" in keywords
    assert "X光机" in keywords


def test_synonym_index_rebuild_on_add():
    """Test that synonym index is rebuilt when adding field keywords at runtime"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Add new field keyword with synonyms at runtime
    retriever.add_field_keywords(
        {"人工智能": ["AI", "Artificial Intelligence", "智能算法"]}
    )

    # Query with synonym should expand to all terms
    keywords = retriever.extract_keywords_from_query("AI技术")
    assert "人工智能" in keywords
    assert "智能算法" in keywords
    assert "Artificial Intelligence" in keywords

    # Query with another synonym
    keywords = retriever.extract_keywords_from_query("智能算法应用")
    assert "人工智能" in keywords
    assert "AI" in keywords


# =============================================================================
# Fulltext Search Tests (New from Task 2)
# =============================================================================


def test_fulltext_search_method_exists():
    """Test that _keyword_search_fulltext method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Verify method exists
    assert hasattr(retriever, "_keyword_search_fulltext")
    assert hasattr(retriever, "_keyword_search_legacy")


def test_fulltext_search_empty_keywords():
    """Test fulltext search with empty keywords returns empty list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Empty keywords should return empty list
    result = retriever._keyword_search_fulltext([])
    assert result == []


def test_legacy_keyword_search_empty_keywords():
    """Test legacy keyword search with empty keywords returns empty list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Empty keywords should return empty list
    result = retriever._keyword_search_legacy([])
    assert result == []


def test_keyword_extraction_for_fulltext():
    """Test keyword extraction produces valid input for fulltext search"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    # Test that keywords are extracted correctly
    keywords = retriever.extract_keywords_from_query("培训时长和服务响应")

    # Should contain field keywords
    assert "培训" in keywords
    assert "时长" in keywords
    assert "服务" in keywords
    assert "响应" in keywords


def test_fulltext_search_integration():
    """Integration test for fulltext search with real database"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    # Extract keywords
    keywords = retriever.extract_keywords_from_query("培训")
    assert "培训" in keywords

    # Fulltext search should return results
    results = retriever._keyword_search_fulltext(["培训"])

    # Verify results format
    for chunk_id, score in results:
        assert isinstance(chunk_id, str)
        assert isinstance(score, float)
        assert score >= 0  # ts_rank_cd returns non-negative values


# =============================================================================
# Connection Pool Tests (New from Task 3)
# =============================================================================


def test_connection_pool_initialization():
    """Test connection pool initialization with default settings"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, HAS_CONNECTION_POOL
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=True
    )

    if HAS_CONNECTION_POOL:
        # Should have pool if psycopg-pool is installed
        assert retriever._pool is not None
    else:
        # Should be None if psycopg-pool not installed
        assert retriever._pool is None


def test_connection_pool_disabled():
    """Test that connection pool can be disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=False
    )

    # Pool should be None when disabled
    assert retriever._pool is None


def test_get_connection_method_exists():
    """Test that _get_connection method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    assert hasattr(retriever, "_get_connection")
    assert callable(getattr(retriever, "_get_connection"))


def test_close_method_exists():
    """Test that close method exists and works"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=True
    )

    assert hasattr(retriever, "close")

    # Close should not raise error
    retriever.close()

    # Pool should be None after close
    assert retriever._pool is None


def test_context_manager():
    """Test that HybridRetriever works as context manager"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Test context manager
    with HybridRetriever(
        version_id="test", settings=settings, top_k=5, use_connection_pool=True
    ) as retriever:
        assert retriever.version_id == "test"
        # Pool should exist inside context

    # Pool should be closed after exiting context


def test_custom_pool_sizes():
    """Test custom pool size configuration"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Should accept custom pool sizes
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        use_connection_pool=True,
        pool_min_size=1,
        pool_max_size=5,
    )

    # Just verify initialization doesn't fail
    assert retriever.version_id == "test"


# =============================================================================
# HNSW Index Parameter Tests (New from Task 4)
# =============================================================================


def test_hnsw_ef_search_configuration():
    """Test that hnsw_ef_search parameter is properly configured"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Test with default value (100)
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)
    assert retriever._hnsw_ef_search == 100

    # Test with custom value
    retriever_custom = HybridRetriever(
        version_id="test", settings=settings, top_k=5, hnsw_ef_search=200
    )
    assert retriever_custom._hnsw_ef_search == 200


# =============================================================================
# RRF Weight Parameter Tests (New from Task 5)
# =============================================================================


def test_rrf_weight_default_values():
    """Test that RRF weights default to 1.0"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    rrf = ReciprocalRankFusion(k=60)
    assert rrf.vector_weight == 1.0
    assert rrf.keyword_weight == 1.0


def test_rrf_weight_custom_values():
    """Test that RRF accepts custom weight values"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    rrf = ReciprocalRankFusion(k=60, vector_weight=2.0, keyword_weight=0.5)
    assert rrf.vector_weight == 2.0
    assert rrf.keyword_weight == 0.5


def test_rrf_with_equal_weights():
    """Test RRF with equal weights produces same result as no weights"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]
    keyword_results = [("chunk_2", 1.0), ("chunk_4", 0.9), ("chunk_1", 0.8)]

    # Without explicit weights (defaults to 1.0)
    rrf_default = ReciprocalRankFusion(k=60)
    merged_default = rrf_default.fuse(vector_results, keyword_results)

    # With explicit equal weights
    rrf_equal = ReciprocalRankFusion(k=60, vector_weight=1.0, keyword_weight=1.0)
    merged_equal = rrf_equal.fuse(vector_results, keyword_results)

    # Results should be identical
    assert len(merged_default) == len(merged_equal)
    for i in range(len(merged_default)):
        assert merged_default[i][0] == merged_equal[i][0]
        assert merged_default[i][1] == pytest.approx(merged_equal[i][1], rel=1e-6)


def test_rrf_with_vector_weight_boost():
    """Test that increasing vector weight boosts vector-only results"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    # chunk_1 is only in vector results, chunk_2 is only in keyword results
    vector_results = [("chunk_1", 0.9)]
    keyword_results = [("chunk_2", 1.0)]

    # Equal weights - both should have same score
    rrf_equal = ReciprocalRankFusion(k=60, vector_weight=1.0, keyword_weight=1.0)
    merged_equal = rrf_equal.fuse(vector_results, keyword_results)

    # chunk_1 and chunk_2 should have equal scores with equal weights
    chunk_1_score_equal = next(
        score for doc_id, score, _ in merged_equal if doc_id == "chunk_1"
    )
    chunk_2_score_equal = next(
        score for doc_id, score, _ in merged_equal if doc_id == "chunk_2"
    )
    assert chunk_1_score_equal == pytest.approx(chunk_2_score_equal, rel=1e-6)

    # Boost vector weight - chunk_1 should now have higher score
    rrf_vector = ReciprocalRankFusion(k=60, vector_weight=2.0, keyword_weight=1.0)
    merged_vector = rrf_vector.fuse(vector_results, keyword_results)

    chunk_1_score_boosted = next(
        score for doc_id, score, _ in merged_vector if doc_id == "chunk_1"
    )
    chunk_2_score_boosted = next(
        score for doc_id, score, _ in merged_vector if doc_id == "chunk_2"
    )

    # chunk_1 score should be higher with vector weight = 2.0
    assert chunk_1_score_boosted > chunk_2_score_boosted
    # Verify exact calculation: vector weight 2.0 gives 2x the RRF contribution
    assert chunk_1_score_boosted == pytest.approx(chunk_1_score_equal * 2.0, rel=1e-6)


def test_rrf_with_keyword_weight_boost():
    """Test that increasing keyword weight boosts keyword-only results"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    vector_results = [("chunk_1", 0.9)]
    keyword_results = [("chunk_2", 1.0)]

    # Equal weights
    rrf_equal = ReciprocalRankFusion(k=60, vector_weight=1.0, keyword_weight=1.0)
    merged_equal = rrf_equal.fuse(vector_results, keyword_results)

    chunk_2_score_equal = next(
        score for doc_id, score, _ in merged_equal if doc_id == "chunk_2"
    )

    # Boost keyword weight
    rrf_keyword = ReciprocalRankFusion(k=60, vector_weight=1.0, keyword_weight=3.0)
    merged_keyword = rrf_keyword.fuse(vector_results, keyword_results)

    chunk_1_score = next(
        score for doc_id, score, _ in merged_keyword if doc_id == "chunk_1"
    )
    chunk_2_score = next(
        score for doc_id, score, _ in merged_keyword if doc_id == "chunk_2"
    )

    # chunk_2 should now be higher than chunk_1
    assert chunk_2_score > chunk_1_score
    # chunk_2 score should be 3x the equal weight score
    assert chunk_2_score == pytest.approx(chunk_2_score_equal * 3.0, rel=1e-6)


def test_rrf_weighted_score_calculation():
    """Test exact weighted score calculation"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    vector_results = [("chunk_1", 0.9)]  # rank 0
    keyword_results = [("chunk_1", 0.8)]  # rank 0

    # With k=60, vector_weight=2.0, keyword_weight=0.5
    # Expected score: 2.0/(60+0+1) + 0.5/(60+0+1) = 2.5/61
    rrf = ReciprocalRankFusion(k=60, vector_weight=2.0, keyword_weight=0.5)
    merged = rrf.fuse(vector_results, keyword_results)

    assert len(merged) == 1
    assert merged[0][0] == "chunk_1"
    expected_score = 2.5 / 61
    assert merged[0][1] == pytest.approx(expected_score, rel=1e-6)


def test_rrf_weight_affects_ranking():
    """Test that different weights can change the ranking order"""
    from bid_scoring.hybrid_retrieval import ReciprocalRankFusion

    # chunk_1 appears in both (strong in vector, weak in keyword)
    # chunk_2 appears only in keyword (strong)
    vector_results = [("chunk_1", 0.95)]  # rank 0
    keyword_results = [
        ("chunk_2", 1.0),
        ("chunk_1", 0.5),
    ]  # chunk_2 rank 0, chunk_1 rank 1

    # With equal weights
    rrf_equal = ReciprocalRankFusion(k=60, vector_weight=1.0, keyword_weight=1.0)
    merged_equal = rrf_equal.fuse(vector_results, keyword_results)

    # chunk_1: 1/61 + 1/62 = ~0.0325
    # chunk_2: 0 + 1/61 = ~0.0164
    # chunk_1 should be first
    assert merged_equal[0][0] == "chunk_1"

    # With zero vector weight, only keyword matters
    rrf_no_vector = ReciprocalRankFusion(k=60, vector_weight=0.0, keyword_weight=1.0)
    merged_no_vector = rrf_no_vector.fuse(vector_results, keyword_results)

    # chunk_1: 0 + 1/62 = ~0.0161
    # chunk_2: 0 + 1/61 = ~0.0164
    # chunk_2 should now be first
    assert merged_no_vector[0][0] == "chunk_2"


def test_hybrid_retriever_passes_weights_to_rrf():
    """Test that HybridRetriever passes weights to ReciprocalRankFusion"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Test with custom weights
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        rrf_k=40,
        vector_weight=2.0,
        keyword_weight=0.5,
    )

    # Verify weights are passed to RRF
    assert retriever.rrf.k == 40
    assert retriever.rrf.vector_weight == 2.0
    assert retriever.rrf.keyword_weight == 0.5


def test_hybrid_retriever_default_weights():
    """Test that HybridRetriever uses default weights of 1.0"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    # Verify default weights
    assert retriever.rrf.vector_weight == 1.0
    assert retriever.rrf.keyword_weight == 1.0


# =============================================================================
# Query Caching Tests (New from Task 6)
# =============================================================================


def test_lru_cache_basic_operations():
    """Test LRUCache basic get/put operations"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=3)

    # Test put and get
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test non-existent key
    assert cache.get("key2") is None

    # Test update existing key
    cache.put("key1", "value1_updated")
    assert cache.get("key1") == "value1_updated"


def test_lru_cache_eviction():
    """Test LRUCache eviction when capacity is exceeded"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1 (least recently used)

    # key1 should be evicted
    assert cache.get("key1") is None
    # key2 and key3 should still exist
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_lru_cache_access_order():
    """Test that accessing item updates its LRU order"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    # Access key1 to make it more recently used
    cache.get("key1")

    # Add key3 - should evict key2 (now least recently used)
    cache.put("key3", "value3")

    assert cache.get("key1") == "value1"  # Should still exist
    assert cache.get("key2") is None  # Should be evicted
    assert cache.get("key3") == "value3"


def test_lru_cache_clear():
    """Test LRUCache clear operation"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=5)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert len(cache._cache) == 0


def test_lru_cache_capacity_property():
    """Test LRUCache capacity property"""
    from bid_scoring.hybrid_retrieval import LRUCache

    cache = LRUCache(capacity=100)
    assert cache.capacity == 100


def test_query_caching_enabled():
    """Test that cache stores and returns results when enabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, LRUCache
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Create retriever with cache enabled
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    # Verify cache is initialized
    assert retriever._cache is not None
    assert isinstance(retriever._cache, LRUCache)
    assert retriever._cache.capacity == 100


def test_query_caching_disabled():
    """Test that cache is None when disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    # Create retriever with cache disabled (default)
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    # Verify cache is None
    assert retriever._cache is None


def test_query_caching_default_disabled():
    """Test that caching is disabled by default"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    # Cache should be None by default
    assert retriever._cache is None


def test_clear_cache_method():
    """Test clear_cache method clears the cache"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    # Add something to cache directly
    retriever._cache.put("test_key", "test_value")
    assert retriever._cache.get("test_key") == "test_value"

    # Clear cache
    retriever.clear_cache()

    # Cache should be empty
    assert retriever._cache.get("test_key") is None
    assert len(retriever._cache._cache) == 0


def test_clear_cache_when_disabled():
    """Test clear_cache doesn't error when cache is disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    # Should not raise error
    retriever.clear_cache()


def test_get_cache_stats_enabled():
    """Test get_cache_stats when cache is enabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    stats = retriever.get_cache_stats()

    assert stats["enabled"] is True
    assert stats["size"] == 0
    assert stats["capacity"] == 100

    # Add something to cache
    retriever._cache.put("key", "value")
    stats = retriever.get_cache_stats()
    assert stats["size"] == 1


def test_get_cache_stats_disabled():
    """Test get_cache_stats when cache is disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    stats = retriever.get_cache_stats()

    assert stats["enabled"] is False
    assert stats["size"] == 0
    assert stats["capacity"] == 0


def test_cache_key_generation():
    """Test cache key generation is deterministic"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )

    # Same inputs should produce same key
    key1 = retriever._generate_cache_key("test query", ["kw1", "kw2"])
    key2 = retriever._generate_cache_key("test query", ["kw1", "kw2"])
    assert key1 == key2

    # Different inputs should produce different keys
    key3 = retriever._generate_cache_key("test query", ["kw1"])
    assert key1 != key3

    key4 = retriever._generate_cache_key("different query", ["kw1", "kw2"])
    assert key1 != key4

    # Different version_id should produce different key
    retriever2 = HybridRetriever(
        version_id="v2",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )
    key5 = retriever2._generate_cache_key("test query", ["kw1", "kw2"])
    assert key1 != key5

    # Different top_k should produce different key
    retriever3 = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=5,
        enable_cache=True,
    )
    key6 = retriever3._generate_cache_key("test query", ["kw1", "kw2"])
    assert key1 != key6


def test_cache_key_with_none_keywords():
    """Test cache key generation with None keywords"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )

    # Should work with None keywords
    key = retriever._generate_cache_key("test query", None)
    assert isinstance(key, str)
    assert len(key) == 64  # SHA256 hex digest length


def test_cache_key_is_sha256():
    """Test that cache key is a valid SHA256 hash"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()

    retriever = HybridRetriever(
        version_id="v1",
        settings=settings,
        top_k=10,
        enable_cache=True,
    )

    key = retriever._generate_cache_key("test query", ["kw1"])

    # SHA256 hex digest should be 64 characters
    assert len(key) == 64
    # Should only contain hex characters
    assert all(c in "0123456789abcdef" for c in key)


# =============================================================================
# Async Retrieval Tests (New from Task 7)
# =============================================================================


@pytest.mark.asyncio
async def test_retrieve_async_method_exists():
    """Test that retrieve_async method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    assert hasattr(retriever, "retrieve_async")
    assert callable(getattr(retriever, "retrieve_async"))


@pytest.mark.asyncio
async def test_close_async_method_exists():
    """Test that close_async method exists"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test", settings={"DATABASE_URL": "postgresql://test"}, top_k=5
    )

    assert hasattr(retriever, "close_async")
    assert callable(getattr(retriever, "close_async"))


@pytest.mark.asyncio
async def test_close_async_closes_resources():
    """Test that close_async properly closes resources"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        use_connection_pool=True,
    )

    # Close should not raise error
    await retriever.close_async()

    # Pool should be None after close
    assert retriever._pool is None


@pytest.mark.asyncio
async def test_retrieve_async_basic():
    """Test basic async retrieval with real database"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训时长")
        assert isinstance(results, list)
        # Results should be RetrievalResult objects
        for result in results:
            assert hasattr(result, "chunk_id")
            assert hasattr(result, "text")
            assert hasattr(result, "score")
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_with_keywords():
    """Test async retrieval with explicit keywords"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训时长", keywords=["培训", "时长"])
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_returns_list():
    """Test that retrieve_async returns a list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训")
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_with_cache():
    """Test async retrieval with caching enabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    try:
        # First call
        results1 = await retriever.retrieve_async("培训时长", use_cache=True)

        # Second call should hit cache
        results2 = await retriever.retrieve_async("培训时长", use_cache=True)

        # Results should be identical (from cache)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.chunk_id == r2.chunk_id
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_without_cache():
    """Test async retrieval with cache disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=True,
        cache_size=100,
    )

    try:
        # Call with use_cache=False
        results = await retriever.retrieve_async("培训时长", use_cache=False)
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_cache_disabled_retriever():
    """Test async retrieval when retriever has cache disabled"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5,
        enable_cache=False,
    )

    try:
        results = await retriever.retrieve_async("培训时长", use_cache=True)
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_same_params_as_sync():
    """Test that async version accepts same parameters as sync"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        # Test with all parameters
        results = await retriever.retrieve_async(
            query="培训时长",
            keywords=["培训", "时长"],
            use_cache=False,
        )
        assert isinstance(results, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_concurrent_calls():
    """Test that multiple async calls can run concurrently"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    from bid_scoring.config import load_settings
    import asyncio

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        # Run multiple queries concurrently
        queries = ["培训", "服务", "响应"]
        tasks = [retriever.retrieve_async(q) for q in queries]
        results = await asyncio.gather(*tasks)

        # All should return lists
        for result_list in results:
            assert isinstance(result_list, list)
    finally:
        await retriever.close_async()


@pytest.mark.asyncio
async def test_retrieve_async_result_format():
    """Test that async results have correct format"""
    from bid_scoring.hybrid_retrieval import HybridRetriever, RetrievalResult
    from bid_scoring.config import load_settings

    settings = load_settings()
    retriever = HybridRetriever(version_id="test", settings=settings, top_k=5)

    try:
        results = await retriever.retrieve_async("培训时长")

        for result in results:
            assert isinstance(result, RetrievalResult)
            assert isinstance(result.chunk_id, str)
            assert isinstance(result.text, str)
            assert isinstance(result.score, float)
            assert isinstance(result.page_idx, int)
            assert result.source in ["vector", "keyword", "hybrid"]
    finally:
        await retriever.close_async()
