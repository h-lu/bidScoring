from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class EvidenceLocator(Protocol):
    """Abstract locator for evidence rendering."""

    def locate(self, anchor_json: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class TextBBoxLocator:
    """Current locator implementation using text bbox anchors."""

    def locate(self, anchor_json: dict[str, Any]) -> dict[str, Any]:
        return {
            "locator_type": "text_bbox",
            "anchor": anchor_json,
        }


@dataclass
class ImageRegionLocator:
    """Future extension point for image-region evidence."""

    def locate(self, anchor_json: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError("Image region evidence is not implemented yet")
