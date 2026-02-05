import pytest
from bid_scoring.hybrid_retrieval import HybridRetriever, ReciprocalRankFusion


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
