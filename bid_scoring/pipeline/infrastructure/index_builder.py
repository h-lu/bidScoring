from __future__ import annotations

from typing import Any

from bid_scoring.embeddings_batch import EmbeddingBatchService


class IndexBuilder:
    """Builds derivative index artifacts (embeddings in M1)."""

    def __init__(self, embedder: EmbeddingBatchService | None = None):
        self.embedder = embedder or EmbeddingBatchService()

    def build_embeddings(self, *, version_id: str, conn: Any) -> dict[str, Any]:
        return self.embedder.process_version(version_id=version_id, conn=conn)

