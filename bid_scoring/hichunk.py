"""HiChunk Builder - 层次化文档分块构建器.

Backward-compatible facade.
Implementation is split into smaller modules to keep files under the 500 LOC limit.
"""

from __future__ import annotations

from bid_scoring.hichunk_builder import HiChunkBuilder
from bid_scoring.hichunk_node import HiChunkNode

__all__ = ["HiChunkBuilder", "HiChunkNode"]

