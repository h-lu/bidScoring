"""Comparison helpers for multi-version bid evaluation."""

from __future__ import annotations

from typing import Any

from mcp_servers.bid_analysis.models import ANALYSIS_DIMENSIONS


def compare_versions(
    conn,
    version_ids: list[str],
    bidder_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compare multiple bid versions side-by-side."""
    from mcp_servers.bid_analyzer import BidAnalyzer

    analyzer = BidAnalyzer(conn)
    reports = []

    for version_id in version_ids:
        bidder_name = (bidder_names or {}).get(version_id, f"Bidder {version_id[:8]}")
        report = analyzer.analyze_version(version_id, bidder_name=bidder_name)
        reports.append(report)

    reports.sort(key=lambda report: report.overall_score, reverse=True)

    comparison = {
        "rankings": [
            {
                "rank": i + 1,
                "version_id": report.version_id,
                "bidder_name": report.bidder_name,
                "overall_score": report.overall_score,
                "risk_level": report.risk_level,
                "total_risks": report.total_risks,
                "total_benefits": report.total_benefits,
            }
            for i, report in enumerate(reports)
        ],
        "dimension_comparison": {},
        "recommendations": [],
    }

    for dim_name in ANALYSIS_DIMENSIONS:
        dim_data = []
        for report in reports:
            if dim_name not in report.dimensions:
                continue
            result = report.dimensions[dim_name]
            dim_data.append(
                {
                    "bidder": report.bidder_name,
                    "score": result.score,
                    "risk_level": result.risk_level,
                    "chunks_found": result.chunks_found,
                }
            )

        comparison["dimension_comparison"][dim_name] = sorted(
            dim_data, key=lambda item: item["score"], reverse=True
        )

    winner = reports[0] if reports else None
    if winner:
        comparison["recommendations"].append(
            f"推荐中标：{winner.bidder_name}（综合评分：{winner.overall_score:.1f}）"
        )

    if len(reports) >= 2:
        top_score = reports[0].overall_score
        second_score = reports[1].overall_score
        if top_score - second_score < 5:
            comparison["recommendations"].append(
                f"注意：前两名得分接近（差距 {top_score - second_score:.1f}），建议详细对比"
            )

    return comparison
