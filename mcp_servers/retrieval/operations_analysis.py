"""High-level bid analysis operations for retrieval MCP server."""

from __future__ import annotations

from typing import Any, Dict

from bid_scoring.config import load_settings
from mcp_servers.retrieval.validation import validate_string_list


def analyze_bids_comprehensive(
    version_ids: list[str],
    bidder_names: dict[str, str] | None = None,
    dimensions: list[str] | None = None,
    generate_annotations: bool = False,
) -> Dict[str, Any]:
    """Comprehensive bid analysis across multiple documents and dimensions.

    This is the primary tool for bid analysis workflows. It orchestrates
    the complete analysis pipeline: document validation, dimension analysis,
    scoring, ranking, and optional PDF annotation generation.

    Args:
        version_ids: List of document version UUIDs to analyze.
        bidder_names: Optional mapping of version_id to bidder name.
                      If not provided, bidder names will be inferred from
                      document titles or content.
        dimensions: List of dimension names to analyze.
                   If not provided, analyzes all 6 dimensions:
                   ["warranty", "delivery", "training", "financial",
                    "technical", "compliance"]
        generate_annotations: Whether to generate annotated PDFs
                             with highlights for risks/benefits.

    Returns:
        Comprehensive analysis results including:
        - rankings: Sorted list of bidders by score
        - dimension_comparison: Side-by-side dimension scores
        - recommendations: Actionable recommendations
        - annotated_urls: PDF annotation links (if requested)
        - summary: High-level summary of findings
    """
    import psycopg

    # Validate inputs
    version_ids = validate_string_list(
        version_ids, "version_ids", min_items=1, max_items=20
    )

    if dimensions is None:
        dimensions = [
            "warranty",
            "delivery",
            "training",
            "financial",
            "technical",
            "compliance",
        ]
    else:
        dimensions = validate_string_list(
            dimensions, "dimensions", min_items=1, max_items=6
        )

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        from mcp_servers.bid_analyzer import (
            BidAnalyzer,
            ANALYSIS_DIMENSIONS,
        )

        analyzer = BidAnalyzer(conn)

        # Get project info for each version
        project_info = {}
        with conn.cursor() as cur:
            for vid in version_ids:
                cur.execute(
                    """
                    SELECT p.name as project_name, p.project_id, d.title
                    FROM document_versions v
                    JOIN documents d ON v.doc_id = d.doc_id
                    JOIN projects p ON d.project_id = p.project_id
                    WHERE v.version_id = %s
                    """,
                    (vid,),
                )
                row = cur.fetchone()
                if row:
                    project_info[vid] = {
                        "project_name": row[0],
                        "project_id": str(row[1]),
                        "document_title": row[2],
                    }

        # Infer bidder names if not provided
        if bidder_names is None:
            bidder_names = {}
            for vid in version_ids:
                info = project_info.get(vid, {})
                title = info.get("document_title", "")
                # Try to extract company name from title
                if title:
                    # Simple extraction: look for company-like patterns
                    import re

                    company_patterns = [
                        r"([^\s]+(?:科技|生物|实业|贸易|有限公司|公司))",
                        r"([^\s]+有限公司)",
                        r"([A-Z][a-zA-Z]+\s+(?:Technology|Bio|Science|Co|Ltd|Inc))",
                    ]
                    for pattern in company_patterns:
                        match = re.search(pattern, title)
                        if match:
                            bidder_names[vid] = match.group(1)
                            break
                if vid not in bidder_names:
                    bidder_names[vid] = f"投标方{vid[:8]}"

        # Analyze each version
        reports = []
        for vid in version_ids:
            info = project_info.get(vid, {})
            report = analyzer.analyze_version(
                version_id=vid,
                bidder_name=bidder_names.get(vid, "Unknown"),
                project_name=info.get("project_name", "Unknown Project"),
                dimensions=dimensions,
            )
            reports.append(report)

        # Sort by overall score
        reports.sort(key=lambda r: r.overall_score, reverse=True)

        # Build rankings
        rankings = [
            {
                "rank": i + 1,
                "version_id": r.version_id,
                "bidder_name": r.bidder_name,
                "overall_score": round(r.overall_score, 1),
                "risk_level": r.risk_level,
                "total_risks": r.total_risks,
                "total_benefits": r.total_benefits,
                "chunks_analyzed": r.chunks_analyzed,
                "evidence_warnings": r.evidence_warnings,
            }
            for i, r in enumerate(reports)
        ]

        # Build dimension comparison
        dimension_comparison = {}
        for dim_name in dimensions:
            dim_config = ANALYSIS_DIMENSIONS.get(dim_name)
            if not dim_config:
                continue

            dim_data = []
            for r in reports:
                if dim_name in r.dimensions:
                    d = r.dimensions[dim_name]
                    dim_data.append(
                        {
                            "bidder": r.bidder_name,
                            "score": round(d.score, 1),
                            "risk_level": d.risk_level,
                            "chunks_found": d.chunks_found,
                            "risks_count": len(d.risks),
                            "benefits_count": len(d.benefits),
                        }
                    )

            dimension_comparison[dim_name] = {
                "display_name": dim_config.display_name,
                "weight": dim_config.weight,
                "bidders": sorted(dim_data, key=lambda x: x["score"], reverse=True),
            }

        # Generate recommendations
        recommendations = []

        if not reports:
            recommendations.append("无法生成分析：没有有效的文档数据")
        else:
            winner = reports[0]
            recommendations.append(
                f"推荐中标：{winner.bidder_name}（综合评分：{winner.overall_score:.1f}，风险等级：{winner.risk_level}）"
            )

            # Check for close scores
            if len(reports) >= 2:
                top_score = reports[0].overall_score
                second_score = reports[1].overall_score
                if top_score - second_score < 5:
                    recommendations.append(
                        f"注意：前两名得分接近（差距 {top_score - second_score:.1f}），建议详细对比"
                    )

            # Collect dimension-specific recommendations
            all_recommendations = []
            for r in reports:
                all_recommendations.extend(r.recommendations)

            # Add top concerns
            high_risk_bidders = [
                r.bidder_name for r in reports if r.risk_level == "high"
            ]
            if high_risk_bidders:
                recommendations.append(
                    f"风险提示：{', '.join(high_risk_bidders)} 存在较高风险，建议重点审查"
                )

        # Generate annotated PDFs if requested
        annotated_urls = {}
        if generate_annotations:
            from mineru.minio_storage import MinIOStorage
            from mcp_servers.pdf_annotator import PDFAnnotator

            storage = MinIOStorage()
            annotator = PDFAnnotator(conn, storage)

            for report in reports:
                vid = report.version_id

                # Collect chunk_ids for each topic
                risk_chunks = []
                benefit_chunks = []

                for dim_result in report.dimensions.values():
                    for insight in dim_result.risks:
                        if hasattr(insight, "chunk_id"):
                            risk_chunks.append(insight.chunk_id)
                    for insight in dim_result.benefits:
                        if hasattr(insight, "chunk_id"):
                            benefit_chunks.append(insight.chunk_id)

                # Highlight risks (red) and benefits (green)
                if risk_chunks:
                    result = annotator.highlight_chunks(
                        version_id=vid,
                        chunk_ids=risk_chunks[:100],  # Limit to 100
                        topic="risk",
                        increment=False,
                    )
                    if result.success:
                        if vid not in annotated_urls:
                            annotated_urls[vid] = {
                                "bidder_name": report.bidder_name,
                                "topics": {},
                            }
                        annotated_urls[vid]["topics"]["risk"] = result.annotated_url

                if benefit_chunks:
                    result = annotator.highlight_chunks(
                        version_id=vid,
                        chunk_ids=benefit_chunks[:100],  # Limit to 100
                        topic="benefit",
                        increment=bool(
                            risk_chunks
                        ),  # Add to existing if risks were highlighted
                    )
                    if result.success:
                        if vid not in annotated_urls:
                            annotated_urls[vid] = {
                                "bidder_name": report.bidder_name,
                                "topics": {},
                            }
                        annotated_urls[vid]["topics"]["benefit"] = result.annotated_url

        # Build summary
        summary = {
            "total_bidders": len(reports),
            "winner": rankings[0]["bidder_name"] if rankings else None,
            "winner_score": rankings[0]["overall_score"] if rankings else None,
            "avg_score": round(sum(r.overall_score for r in reports) / len(reports), 1)
            if reports
            else 0,
            "high_risk_count": sum(1 for r in reports if r.risk_level == "high"),
            "dimensions_analyzed": len(dimensions),
            "versions_with_evidence_warnings": sum(
                1 for r in reports if r.evidence_warnings
            ),
        }

        return {
            "rankings": rankings,
            "dimension_comparison": dimension_comparison,
            "recommendations": recommendations,
            "annotated_urls": annotated_urls,
            "summary": summary,
            "version_ids": version_ids,
        }
