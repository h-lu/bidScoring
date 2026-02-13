from __future__ import annotations

from bid_scoring.retrieval.hybrid import HybridRetriever


def test_retrieve_prefers_keyword_side_for_technical_queries():
    retriever = HybridRetriever(
        version_id="33333333-3333-3333-3333-333333333333",
        settings={"DATABASE_URL": "postgresql://unused"},
        top_k=2,
        use_connection_pool=False,
        enable_dynamic_weights=True,
    )

    retriever._vector_search = lambda _query: [("v1", 0.9), ("v2", 0.8)]  # type: ignore[method-assign]
    retriever._keyword_search_fulltext = lambda _keywords, _use_or_semantic: [  # type: ignore[method-assign]
        ("v2", 3.0),
        ("v1", 2.0),
    ]

    captured = {}

    def _capture_fetch(merged):
        captured["merged"] = merged
        return merged

    retriever._fetch_chunks = _capture_fetch  # type: ignore[method-assign]

    retriever.retrieve("CT MRI detector spec")

    assert captured["merged"][0][0] == "v2"


def test_retrieve_uses_default_weights_when_dynamic_disabled():
    retriever = HybridRetriever(
        version_id="33333333-3333-3333-3333-333333333333",
        settings={"DATABASE_URL": "postgresql://unused"},
        top_k=2,
        use_connection_pool=False,
        enable_dynamic_weights=False,
    )

    retriever._vector_search = lambda _query: [("v1", 0.9), ("v2", 0.8)]  # type: ignore[method-assign]
    retriever._keyword_search_fulltext = lambda _keywords, _use_or_semantic: [  # type: ignore[method-assign]
        ("v2", 3.0),
        ("v1", 2.0),
    ]

    captured = {}

    def _capture_fetch(merged):
        captured["merged"] = merged
        return merged

    retriever._fetch_chunks = _capture_fetch  # type: ignore[method-assign]

    retriever.retrieve("CT MRI detector spec")

    assert captured["merged"][0][0] == "v1"
