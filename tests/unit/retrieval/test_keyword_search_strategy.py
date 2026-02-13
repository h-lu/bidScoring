from __future__ import annotations

from bid_scoring.retrieval import search_keyword as sk


def test_keyword_search_fulltext_prefers_tsquery_for_non_cjk(monkeypatch):
    called = {"fts": 0, "legacy": 0}

    def _fake_fts(_retriever, _keywords, use_or_semantic):
        _ = use_or_semantic
        called["fts"] += 1
        return [("chunk-fts", 0.9)]

    def _fake_legacy(_retriever, _keywords):
        called["legacy"] += 1
        return [("chunk-legacy", 1.0)]

    monkeypatch.setattr(sk, "_keyword_search_textsearch", _fake_fts, raising=False)
    monkeypatch.setattr(sk, "keyword_search_legacy", _fake_legacy)

    result = sk.keyword_search_fulltext(
        retriever=object(),
        keywords=["delivery", "response"],
        use_or_semantic=True,
    )

    assert result == [("chunk-fts", 0.9)]
    assert called == {"fts": 1, "legacy": 0}


def test_keyword_search_fulltext_falls_back_to_legacy_when_fts_empty(monkeypatch):
    called = {"fts": 0, "legacy": 0}

    def _fake_fts(_retriever, _keywords, use_or_semantic):
        _ = use_or_semantic
        called["fts"] += 1
        return []

    def _fake_legacy(_retriever, _keywords):
        called["legacy"] += 1
        return [("chunk-legacy", 1.0)]

    monkeypatch.setattr(sk, "_keyword_search_textsearch", _fake_fts, raising=False)
    monkeypatch.setattr(sk, "keyword_search_legacy", _fake_legacy)

    result = sk.keyword_search_fulltext(
        retriever=object(),
        keywords=["delivery", "response"],
        use_or_semantic=True,
    )

    assert result == [("chunk-legacy", 1.0)]
    assert called == {"fts": 1, "legacy": 1}
