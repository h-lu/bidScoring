# bid_scoring/search.py


def rrf_fuse(bm25_results, vector_results, k=60, bm25_weight=0.4, vector_weight=0.6):
    """
    Fuse BM25 and vector search results using Reciprocal Rank Fusion (RRF).

    RRF formula: score = sum(weight / (k + rank))

    Args:
        bm25_results: List of (doc_id, score) tuples from BM25 search
        vector_results: List of (doc_id, score) tuples from vector search
        k: RRF constant (default 60)
        bm25_weight: Weight for BM25 results (default 0.4)
        vector_weight: Weight for vector results (default 0.6)

    Returns:
        List of doc_ids sorted by fused RRF score (descending)
    """
    scores = {}
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank + 1)
    return [doc for doc, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
