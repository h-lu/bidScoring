from bid_scoring.pipeline.infrastructure.index_builder import IndexBuilder


class _Embedder:
    def process_version(self, *, version_id, conn):  # pragma: no cover
        return {"version_id": version_id, "succeeded": 3, "failed": 0}


def test_index_builder_delegates_to_embedding_service():
    builder = IndexBuilder(embedder=_Embedder())
    result = builder.build_embeddings(version_id="ver-1", conn=object())
    assert result["version_id"] == "ver-1"
    assert result["succeeded"] == 3
