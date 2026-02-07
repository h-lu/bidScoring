import pytest

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
