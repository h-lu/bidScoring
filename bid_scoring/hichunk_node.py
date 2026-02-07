from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HiChunkNode:
    """In-memory representation of a row in `hierarchical_nodes`."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 0
    node_type: str = "sentence"
    content: str = ""
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    start_chunk_id: Optional[str] = None
    end_chunk_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.level <= 3:
            raise ValueError(f"Level must be 0-3, got {self.level}")
        valid_types = {"sentence", "paragraph", "section", "document"}
        if self.node_type not in valid_types:
            raise ValueError(f"Invalid node_type: {self.node_type}")

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "level": self.level,
            "node_type": self.node_type,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "start_chunk_id": self.start_chunk_id,
            "end_chunk_id": self.end_chunk_id,
            "metadata": self.metadata,
        }

    def add_child(self, child_id: str) -> None:
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def set_parent(self, parent_id: str) -> None:
        self.parent_id = parent_id

