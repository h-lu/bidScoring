# tests/test_embeddings.py
from types import SimpleNamespace
from bid_scoring.embeddings import embed_texts
from bid_scoring.config import load_settings


class FakeClient:
    def __init__(self, dim: int):
        self.dim = dim
        self.embeddings = self

    def create(self, model, input):
        data = [SimpleNamespace(embedding=[0.0] * self.dim) for _ in input]
        return SimpleNamespace(data=data)


def test_embed_texts_uses_client(monkeypatch):
    monkeypatch.setenv("OPENAI_EMBEDDING_DIM", "1536")
    dim = load_settings()["OPENAI_EMBEDDING_DIM"]
    vecs = embed_texts(["a", "b"], client=FakeClient(dim), model="text-embedding-3-small")
    assert len(vecs) == 2
    assert len(vecs[0]) == dim
