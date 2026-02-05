import pytest
from bid_scoring.hybrid_retrieval import (
    HybridRetriever, 
    ReciprocalRankFusion, 
    DEFAULT_RRF_K
)


def test_reciprocal_rank_fusion_basic():
    """Test RRF merging of two result lists"""
    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8), ("chunk_3", 0.7)]
    keyword_results = [("chunk_2", 1.0), ("chunk_4", 0.9), ("chunk_1", 0.8)]
    
    rrf = ReciprocalRankFusion(k=60)
    merged = rrf.fuse(vector_results, keyword_results)
    
    assert len(merged) == 4  # All unique chunks
    assert merged[0][0] in ["chunk_1", "chunk_2"]  # Top results should be in both


def test_hybrid_retriever_initialization():
    """Test HybridRetriever can be initialized with settings"""
    from bid_scoring.config import load_settings
    
    settings = load_settings()
    retriever = HybridRetriever(
        version_id="test-version",
        settings=settings,
        top_k=5
    )
    
    assert retriever.version_id == "test-version"
    assert retriever.top_k == 5


def test_rrf_empty_results():
    """Test RRF with empty result lists"""
    rrf = ReciprocalRankFusion(k=60)
    
    # Both empty
    merged = rrf.fuse([], [])
    assert merged == []
    
    # One empty
    vector_results = [("chunk_1", 0.9), ("chunk_2", 0.8)]
    merged = rrf.fuse(vector_results, [])
    assert len(merged) == 2
    assert merged[0][0] == "chunk_1"


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


def test_rrf_multiple_lists():
    """Test RRF with more than two lists"""
    rrf = ReciprocalRankFusion(k=60)
    
    list1 = [("chunk_1", 0.9), ("chunk_2", 0.8)]
    list2 = [("chunk_1", 0.85), ("chunk_3", 0.75)]
    list3 = [("chunk_1", 0.8), ("chunk_4", 0.7)]
    
    merged = rrf.fuse(list1, list2, list3)
    assert len(merged) == 4
    # chunk_1 appears in all lists
    assert merged[0][0] == "chunk_1"


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


def test_input_validation_negative_weights():
    """Test that negative weights raise ValueError"""
    from bid_scoring.config import load_settings
    
    settings = load_settings()
    with pytest.raises(ValueError, match="weights must be non-negative"):
        HybridRetriever(version_id="test", settings=settings, vector_weight=-0.1)
    
    with pytest.raises(ValueError, match="weights must be non-negative"):
        HybridRetriever(version_id="test", settings=settings, keyword_weight=-0.1)


def test_input_validation_valid_parameters():
    """Test that valid parameters don't raise errors"""
    from bid_scoring.config import load_settings
    
    settings = load_settings()
    # Zero weights should be valid
    retriever = HybridRetriever(
        version_id="test",
        settings=settings,
        vector_weight=0,
        keyword_weight=0
    )
    assert retriever.vector_weight == 0
    assert retriever.keyword_weight == 0


def test_keyword_extraction():
    """Test keyword extraction from Chinese queries"""
    from bid_scoring.hybrid_retrieval import HybridRetriever
    
    retriever = HybridRetriever(
        version_id="test",
        settings={"DATABASE_URL": "postgresql://test"},
        top_k=5
    )
    
    # Test training-related query
    keywords = retriever.extract_keywords_from_query("培训时长是多少天")
    assert "培训" in keywords
    assert "时长" in keywords
    
    # Test service-related query  
    keywords = retriever.extract_keywords_from_query("售后服务响应时间")
    assert "服务" in keywords
    assert "响应" in keywords or "响应" in str(keywords)
