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
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5
    )
    
    # Verify method exists
    assert hasattr(retriever, '_keyword_search_fulltext')
    assert hasattr(retriever, '_keyword_search_legacy')


def test_fulltext_search_empty_keywords():
    """Test fulltext search with empty keywords returns empty list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5
    )
    
    # Empty keywords should return empty list
    result = retriever._keyword_search_fulltext([])
    assert result == []


def test_legacy_keyword_search_empty_keywords():
    """Test legacy keyword search with empty keywords returns empty list"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5
    )
    
    # Empty keywords should return empty list
    result = retriever._keyword_search_legacy([])
    assert result == []


def test_keyword_extraction_for_fulltext():
    """Test keyword extraction produces valid input for fulltext search"""
    from bid_scoring.hybrid_retrieval import HybridRetriever

    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5
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
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        top_k=5
    )
    
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
