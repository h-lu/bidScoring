from __future__ import annotations


def test_mcp_server_exports_mcp_instance():
    import fastmcp
    import mcp_servers.retrieval_server as srv

    assert hasattr(srv, "mcp")
    assert isinstance(srv.mcp, fastmcp.FastMCP)


def test_retrieve_tool_formats_results(monkeypatch):
    import mcp_servers.retrieval_server as srv
    from bid_scoring.retrieval import RetrievalResult

    fake_results = [
        RetrievalResult(
            chunk_id="c1",
            text="hello world",
            page_idx=1,
            score=0.5,
            source="hybrid",
            vector_score=0.9,
            keyword_score=1.2,
        )
    ]

    class FakeRetriever:
        def __init__(self):
            class _RRF:
                k = 60

            self.rrf = _RRF()

        def retrieve(self, query, keywords=None):
            return fake_results

        def _vector_search(self, query):
            return [("c1", 0.99)]

        def _fetch_chunks(self, merged):
            return fake_results

        def extract_keywords_from_query(self, query):
            return ["培训"]

        def _keyword_search_fulltext(self, keywords, use_or_semantic=True):
            return [("c1", 1.0)]

    monkeypatch.setattr(srv, "get_retriever", lambda version_id, top_k: FakeRetriever())

    out = srv.retrieve_impl(
        version_id="v1",
        query="q",
        top_k=10,
        mode="hybrid",
        include_text=True,
        max_chars=5,
    )

    assert out["version_id"] == "v1"
    assert out["mode"] == "hybrid"
    assert out["results"][0]["chunk_id"] == "c1"
    assert out["results"][0]["text"] == "hello"

    out2 = srv.retrieve_impl(
        version_id="v1",
        query="q",
        top_k=10,
        mode="hybrid",
        include_text=False,
    )
    assert out2["results"][0]["text"] == ""
