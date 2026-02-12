from __future__ import annotations

import pytest

from bid_scoring.pipeline.domain.locator import ImageRegionLocator, TextBBoxLocator


def test_text_bbox_locator_returns_anchor_payload():
    locator = TextBBoxLocator()
    anchor = {"anchors": [{"page_idx": 1, "bbox": [1, 2, 3, 4]}]}
    located = locator.locate(anchor)
    assert located["locator_type"] == "text_bbox"
    assert located["anchor"] == anchor


def test_image_region_locator_reserved_for_future_extension():
    locator = ImageRegionLocator()
    with pytest.raises(NotImplementedError):
        locator.locate({"anchors": []})

