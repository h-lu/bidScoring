from __future__ import annotations

from bid_scoring.anchors_v2 import build_anchor_json, compute_unit_hash, normalize_text


def test_build_anchor_json_shape():
    anchor_json = build_anchor_json(
        anchors=[
            {
                "page_idx": 12,
                "bbox": [100, 200, 300, 240],
                "coord_sys": "mineru_bbox_v1",
                "page_w": None,
                "page_h": None,
                "path": None,
                "source": {"element_id": "chunk_0000"},
            }
        ]
    )
    assert "anchors" in anchor_json
    assert anchor_json["anchors"][0]["page_idx"] == 12
    assert anchor_json["anchors"][0]["bbox"] == [100, 200, 300, 240]


def test_compute_unit_hash_is_stable():
    text_norm = normalize_text("A  B")
    anchor_json = build_anchor_json(
        anchors=[
            {
                "page_idx": 1,
                "bbox": [1, 2, 3, 4],
                "coord_sys": "mineru_bbox_v1",
                "page_w": None,
                "page_h": None,
                "path": None,
                "source": {"element_id": "x"},
            }
        ]
    )
    h1 = compute_unit_hash(
        text_norm=text_norm, anchor_json=anchor_json, source_element_id="x"
    )
    h2 = compute_unit_hash(
        text_norm=text_norm, anchor_json=anchor_json, source_element_id="x"
    )
    assert h1 == h2
