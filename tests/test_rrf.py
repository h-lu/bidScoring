# tests/test_rrf.py
from bid_scoring.search import rrf_fuse


def test_rrf_fuse_prefers_top_ranks():
    """Test that RRF fuses results and top ranks are preferred."""
    bm25 = [("a", 1), ("b", 2), ("c", 3)]
    vec = [("c", 1), ("a", 2), ("d", 3)]
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.4, vector_weight=0.6)
    assert fused[0] in {"a", "c"}


def test_rrf_fuse_empty_bm25():
    """Test RRF with empty BM25 results."""
    bm25 = []
    vec = [("a", 1), ("b", 2)]
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.4, vector_weight=0.6)
    assert fused == ["a", "b"]


def test_rrf_fuse_empty_vector():
    """Test RRF with empty vector results."""
    bm25 = [("a", 1), ("b", 2)]
    vec = []
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.4, vector_weight=0.6)
    assert fused == ["a", "b"]


def test_rrf_fuse_both_empty():
    """Test RRF with both empty results."""
    bm25 = []
    vec = []
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.4, vector_weight=0.6)
    assert fused == []


def test_rrf_fuse_equal_weights():
    """Test RRF with equal weights."""
    bm25 = [("a", 1), ("b", 2)]
    vec = [("a", 1), ("b", 2)]
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.5, vector_weight=0.5)
    # Both should be present, "a" should be first since it has rank 0 in both
    assert "a" in fused
    assert "b" in fused
    assert fused[0] == "a"


def test_rrf_fuse_single_result():
    """Test RRF with single result in each list."""
    bm25 = [("x", 1)]
    vec = [("y", 1)]
    fused = rrf_fuse(bm25, vec, k=60, bm25_weight=0.4, vector_weight=0.6)
    assert len(fused) == 2
    # "y" should have higher score due to higher vector weight
    assert fused[0] == "y"
