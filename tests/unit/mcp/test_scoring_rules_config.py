from __future__ import annotations

import textwrap

from mcp_servers.bid_analysis.models import load_analysis_dimensions


def test_load_analysis_dimensions_from_yaml(tmp_path):
    config_path = tmp_path / "scoring_rules.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            dimensions:
              warranty:
                display_name: 质保售后
                weight: 0.6
                keywords: [质保, 保修]
                extract_patterns: ['(\\d+)\\s*年']
                risk_thresholds:
                  excellent: [5, 999]
              delivery:
                display_name: 交付响应
                weight: 0.4
                keywords: [交付, 响应]
            """
        ).strip(),
        encoding="utf-8",
    )

    dimensions = load_analysis_dimensions(config_path)

    assert set(dimensions.keys()) == {"warranty", "delivery"}
    assert dimensions["warranty"].weight == 0.6
    assert dimensions["delivery"].keywords == ["交付", "响应"]
