from __future__ import annotations

import re
from typing import Any

from mcp_servers.annotation_insights import (
    AnnotationInsight,
    analyze_chunk_for_insights,
)


class InsightExtractor:
    """Extract risks/benefits/info and numeric values from chunks."""

    def analyze_chunks(
        self,
        chunks: list[dict[str, Any]],
        dimension_name: str,
    ) -> tuple[
        list[AnnotationInsight], list[AnnotationInsight], list[AnnotationInsight]
    ]:
        risks: list[AnnotationInsight] = []
        benefits: list[AnnotationInsight] = []
        info: list[AnnotationInsight] = []

        for chunk in chunks:
            text = chunk.get("text_raw", "")
            insights = analyze_chunk_for_insights(text, dimension_name)
            for insight in insights:
                if insight.category == "risk":
                    risks.append(insight)
                elif insight.category == "benefit":
                    benefits.append(insight)
                else:
                    info.append(insight)

        return risks, benefits, info

    @staticmethod
    def extract_values(
        chunks: list[dict[str, Any]],
        patterns: list[str],
    ) -> dict[str, Any]:
        extracted = {}

        for chunk in chunks:
            text = chunk.get("text_raw", "")

            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    values = [int(m) for m in matches]
                    if values:
                        key = f"pattern_{pattern}"
                        extracted[key] = {
                            "values": values,
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                        }

        return extracted
