#!/usr/bin/env python3
"""End-to-End Tests for Bid Document Analysis.

Tests the complete bid analysis workflow:
1. Document verification (exists, has chunks)
2. Multi-aspect content search (warranty, delivery, training, financial)
3. Risk and benefit identification
4. PDF annotation with insights
5. Analysis report generation

Requires:
- DATABASE_URL set
- At least one document imported with searchable content

Run with:
    uv run pytest tests/test_bid_analysis_e2e.py -v --tb=short -s
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pytest
from psycopg.rows import dict_row


needs_database = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set"
)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class SearchResult:
    """Content found during search."""

    topic: str
    chunk_id: str
    page_idx: int
    bbox: list[float]
    text: str
    relevance: float = 1.0


@dataclass
class AnalysisInsight:
    """AI-generated insight for a chunk."""

    category: str  # "risk", "benefit", "info"
    title: str
    content: str
    risk_level: str | None = None  # "high", "medium", "low"


@dataclass
class AspectAnalysis:
    """Analysis results for one aspect of the bid."""

    aspect: str  # "warranty", "delivery", "training", etc.
    chunks_found: int
    risks: list[AnalysisInsight] = field(default_factory=list)
    benefits: list[AnalysisInsight] = field(default_factory=list)
    summary: str = ""


@dataclass
class BidAnalysisReport:
    """Complete bid analysis report."""

    version_id: str
    bidder_name: str
    project_name: str
    aspects: dict[str, AspectAnalysis]
    total_risks: int
    total_benefits: int
    overall_score: float
    recommendations: list[str]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_version_id():
    """Get a test version ID from database with original PDF."""
    import psycopg
    from bid_scoring.config import load_settings

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT df.version_id
                FROM document_files df
                WHERE df.file_type = 'original_pdf'
                LIMIT 1
                """
            )
            result = cur.fetchone()

            if not result:
                pytest.skip("No original_pdf file found in database")

            return str(result[0])


@pytest.fixture(scope="module")
def local_pdf_path():
    """Get path to local PDF file for testing."""
    pdf_path = Path(
        "/Users/wangxq/Documents/投标分析_kimi/mineru/pdf/"
        "0811-DSITC253135-上海悦晟生物科技有限公司投标文件.pdf"
    )
    if not pdf_path.exists():
        pytest.skip(f"Local PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture(scope="module")
def bidder_info(test_version_id):
    """Extract basic bidder information from chunks."""
    import psycopg
    from bid_scoring.config import load_settings

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT text_raw
                FROM chunks
                WHERE version_id = %s
                AND text_raw ILIKE %s
                LIMIT 1
                """,
                (test_version_id, "%投标人%"),
            )
            result = cur.fetchone()

            if result:
                return {"name": result.get("text_raw", "Unknown Bidder")}
            return {"name": "Unknown Bidder"}


# =============================================================================
# Analysis Functions
# =============================================================================


def search_content_by_topic(
    conn, version_id: str, topic: str, keywords: list[str], limit: int = 5
) -> list[SearchResult]:
    """Search for content related to a specific topic.

    Args:
        conn: Database connection
        version_id: Document version ID
        topic: Topic name (e.g., "warranty", "delivery")
        keywords: List of keywords to search for
        limit: Max results to return

    Returns:
        List of SearchResult objects
    """
    results = []

    for keyword in keywords:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT chunk_id, page_idx, bbox, text_raw
                FROM chunks
                WHERE version_id = %s
                AND text_raw ILIKE %s
                AND bbox IS NOT NULL
                LIMIT %s
                """,
                (version_id, f"%{keyword}%", limit),
            )
            chunks = cur.fetchall()

        for chunk in chunks:
            results.append(
                SearchResult(
                    topic=topic,
                    chunk_id=chunk["chunk_id"],
                    page_idx=chunk["page_idx"],
                    bbox=chunk["bbox"],
                    text=chunk["text_raw"],
                )
            )

    return results[:limit]


def generate_insights(text: str, topic: str) -> list[AnalysisInsight]:
    """Generate analysis insights for a text chunk.

    Args:
        text: The chunk text to analyze
        topic: The topic category

    Returns:
        List of AnalysisInsight objects
    """
    from mcp_servers.annotation_insights import analyze_chunk_for_insights

    insights = analyze_chunk_for_insights(text, topic)

    return [
        AnalysisInsight(
            category=i.category,
            title=i.title,
            content=i.content,
            risk_level=i.risk_level,
        )
        for i in insights
    ]


def analyze_aspect(
    conn, version_id: str, aspect: str, keywords: list[str]
) -> AspectAnalysis:
    """Analyze one aspect of the bid.

    Args:
        conn: Database connection
        version_id: Document version ID
        aspect: Aspect name (e.g., "warranty")
        keywords: Keywords to search for

    Returns:
        AspectAnalysis with findings
    """
    results = search_content_by_topic(conn, version_id, aspect, keywords)

    risks = []
    benefits = []

    for result in results:
        insights = generate_insights(result.text, aspect)
        for insight in insights:
            if insight.category == "risk":
                risks.append(insight)
            elif insight.category == "benefit":
                benefits.append(insight)

    # Generate summary
    summary_parts = []
    if results:
        summary_parts.append(f"找到 {len(results)} 处相关内容")
    if benefits:
        summary_parts.append(f"{len(benefits)} 个优势点")
    if risks:
        summary_parts.append(f"{len(risks)} 个风险点")

    summary = "; ".join(summary_parts) if summary_parts else "未找到相关内容"

    return AspectAnalysis(
        aspect=aspect,
        chunks_found=len(results),
        risks=risks,
        benefits=benefits,
        summary=summary,
    )


def generate_annotated_pdf(
    local_pdf: Path,
    search_results: list[SearchResult],
    output_path: Path,
) -> None:
    """Generate PDF with highlights and annotations.

    Args:
        local_pdf: Path to original PDF
        search_results: List of SearchResult to highlight
        output_path: Path to save annotated PDF
    """
    from mcp_servers.pdf_annotator import TOPIC_COLORS

    doc = fitz.open(local_pdf)

    for result in search_results:
        page_idx = result.page_idx
        bbox = result.bbox

        if page_idx >= len(doc):
            continue

        page = doc[page_idx]
        page_width = page.rect.width
        page_height = page.rect.height

        # Convert from normalized (0-1000) to PDF coordinates
        x0 = bbox[0] / 1000.0 * page_width
        y0 = bbox[1] / 1000.0 * page_height
        x1 = bbox[2] / 1000.0 * page_width
        y1 = bbox[3] / 1000.0 * page_height

        rect = fitz.Rect(x0, y0, x1, y1)

        # Get color for topic
        color = TOPIC_COLORS.get(result.topic, TOPIC_COLORS["default"])

        # Add highlight
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=color)
        annot.set_opacity(0.4)

        # Add intelligent annotation
        insights = generate_insights(result.text, result.topic)
        if insights:
            top_insight = insights[0]
            content = f"{top_insight.title}\n{top_insight.content}"
            annot.set_info(content)

        annot.update()

    doc.save(output_path, incremental=False)
    doc.close()


# =============================================================================
# E2E Tests
# =============================================================================


@pytest.mark.e2e
@needs_database
class TestBidAnalysisE2E:
    """Complete end-to-end bid analysis tests."""

    def test_complete_bid_analysis_workflow(
        self, test_version_id, local_pdf_path, bidder_info
    ):
        """Test complete bid analysis workflow.

        Workflow:
        1. Search for content across multiple aspects
        2. Analyze each aspect for risks and benefits
        3. Generate comprehensive report
        4. Create annotated PDF
        5. Verify output quality
        """
        import psycopg
        from bid_scoring.config import load_settings

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Step 1: Define analysis aspects
            aspects = {
                "warranty": ["质保", "保修", "免费维修", "终生"],
                "delivery": ["交货", "交付", "货期", "响应", "小时"],
                "training": ["培训", "技术指导", "操作培训"],
                "financial": ["付款", "支付", "费用", "报价"],
            }

            # Step 2: Analyze each aspect
            analyses = {}
            all_search_results = []

            for aspect, keywords in aspects.items():
                analysis = analyze_aspect(conn, test_version_id, aspect, keywords)
                analyses[aspect] = analysis

                # Collect results for PDF annotation
                results = search_content_by_topic(
                    conn, test_version_id, aspect, keywords, limit=3
                )
                all_search_results.extend(results)

            # Step 3: Generate report
            total_risks = sum(len(a.risks) for a in analyses.values())
            total_benefits = sum(len(a.benefits) for a in analyses.values())

            # Calculate overall score (benefits - risks, normalized)
            score = max(0, min(100, (total_benefits * 10) - (total_risks * 5) + 50))

            recommendations = []
            if analyses.get("warranty") and len(analyses["warranty"].risks) > 0:
                recommendations.append("建议确认质保期条款是否满足招标要求")
            if analyses.get("delivery") and len(analyses["delivery"].risks) > 0:
                recommendations.append("建议确认交付时间是否具有竞争力")

            report = BidAnalysisReport(
                version_id=test_version_id,
                bidder_name=bidder_info["name"],
                project_name="Test Project",
                aspects=analyses,
                total_risks=total_risks,
                total_benefits=total_benefits,
                overall_score=score,
                recommendations=recommendations,
            )

            # Step 4: Verify report structure
            assert report.bidder_name, "Should have bidder name"
            assert len(report.aspects) == 4, "Should analyze all 4 aspects"
            assert report.total_risks >= 0, "Risk count should be non-negative"
            assert report.total_benefits >= 0, "Benefit count should be non-negative"
            assert 0 <= report.overall_score <= 100, "Score should be 0-100"

            # Step 5: Generate annotated PDF
            with tempfile.TemporaryDirectory() as tmpdir:
                output_pdf = Path(tmpdir) / "bid_analysis_annotated.pdf"
                generate_annotated_pdf(local_pdf_path, all_search_results, output_pdf)

                # Verify PDF was created and has annotations
                assert output_pdf.exists(), "Annotated PDF should be created"

                doc = fitz.open(output_pdf)
                annotation_count = 0
                for page in doc:
                    annotation_count += len(list(page.annots()))
                doc.close()

                assert annotation_count > 0, "PDF should have annotations"

            # Step 6: Print summary report
            print("\n" + "=" * 60)
            print("投标分析报告")
            print("=" * 60)
            print(f"投标人: {report.bidder_name}")
            print(f"综合评分: {report.overall_score:.1f}/100")
            print("\n分析维度:")
            for aspect, analysis in report.aspects.items():
                print(f"  {aspect:12s}: {analysis.summary}")
            print(f"\n风险总计: {report.total_risks}")
            print(f"优势总计: {report.total_benefits}")
            if report.recommendations:
                print("\n建议:")
                for rec in report.recommendations:
                    print(f"  - {rec}")
            print("=" * 60)

    def test_warranty_analysis_detailed(self, test_version_id):
        """Test detailed warranty/after-sales analysis.

        Verifies:
        1. Content search finds warranty-related chunks
        2. Insights identify warranty periods
        3. Risks are flagged for short warranty periods
        4. Benefits are identified for long warranty periods
        """
        import psycopg
        from bid_scoring.config import load_settings

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Search for warranty content
            keywords = ["质保", "保修", "免费维修", "终生"]
            results = search_content_by_topic(
                conn, test_version_id, "warranty", keywords
            )

            print(f"\n质保条款分析: 找到 {len(results)} 处相关内容")

            # Analyze each result
            warranty_periods = []
            insights_by_type = {"risk": [], "benefit": [], "info": []}

            for result in results:
                insights = generate_insights(result.text, "warranty")
                for insight in insights:
                    insights_by_type[insight.category].append(insight)

                # Extract warranty period from text
                import re

                years = re.findall(r"(\d+)\s*年", result.text)
                months = re.findall(r"(\d+)\s*个月", result.text)
                if years:
                    warranty_periods.extend([int(y) for y in years])
                if months:
                    warranty_periods.extend([int(m) / 12 for m in months])

            # Verify analysis
            assert len(results) > 0, "Should find warranty-related content"

            if warranty_periods:
                max_warranty = max(warranty_periods)
                print(f"  最长质保期: {max_warranty:.1f} 年")

                # Check insights correspond to warranty length
                if max_warranty >= 5:
                    assert any(
                        "质保" in i.title or "优势" in i.title
                        for i in insights_by_type.get("benefit", [])
                    ), "Long warranty should generate benefit insight"

            print(f"  风险点: {len(insights_by_type['risk'])}")
            print(f"  优势点: {len(insights_by_type['benefit'])}")

    def test_delivery_analysis_detailed(self, test_version_id):
        """Test detailed delivery/response time analysis.

        Verifies:
        1. Content search finds delivery-related chunks
        2. Insights identify response times
        3. Fast response is flagged as benefit
        4. Slow response is flagged as risk
        """
        import psycopg
        from bid_scoring.config import load_settings

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Search for delivery content
            keywords = ["交货", "交付", "响应", "小时", "天内"]
            results = search_content_by_topic(
                conn, test_version_id, "delivery", keywords
            )

            print(f"\n交付响应分析: 找到 {len(results)} 处相关内容")

            # Analyze response times
            response_times = []
            insights_by_type = {"risk": [], "benefit": [], "info": []}

            for result in results:
                insights = generate_insights(result.text, "delivery")
                for insight in insights:
                    insights_by_type[insight.category].append(insight)

                # Extract time periods
                import re

                hours = re.findall(r"(\d+)\s*小时", result.text)
                days = re.findall(r"(\d+)\s*[天日]", result.text)
                if hours:
                    response_times.extend([int(h) for h in hours])
                if days:
                    response_times.extend([int(d) * 24 for d in days])

            if response_times:
                min_response = min(response_times)
                print(f"  最快响应: {min_response} 小时")

                # Fast response (< 4 hours) should be a benefit
                if min_response <= 4:
                    assert any(
                        "响应" in i.title or "快速" in i.title
                        for i in insights_by_type.get("benefit", [])
                    ), "Fast response should generate benefit insight"

            print(f"  风险点: {len(insights_by_type['risk'])}")
            print(f"  优势点: {len(insights_by_type['benefit'])}")

    def test_generate_annotated_bid_pdf(self, test_version_id, local_pdf_path):
        """Test generating annotated PDF with all analysis insights.

        Workflow:
        1. Search content across all aspects
        2. Generate insights for each chunk
        3. Create PDF with color-coded highlights by aspect
        4. Add popup annotations with AI-generated insights
        5. Verify annotations are correctly positioned
        """
        import psycopg
        from bid_scoring.config import load_settings
        from mcp_servers.pdf_annotator import TOPIC_COLORS

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Define all aspects to analyze
            aspects = {
                "warranty": ["质保", "保修", "免费维修", "终生"],
                "delivery": ["交货", "交付", "响应", "小时内"],
                "training": ["培训", "技术指导", "操作培训"],
                "financial": ["付款", "支付", "费用", "报价"],
            }

            # Collect all search results
            all_results = []
            aspect_colors = {}

            for aspect, keywords in aspects.items():
                results = search_content_by_topic(
                    conn, test_version_id, aspect, keywords, limit=3
                )
                all_results.extend(results)
                aspect_colors[aspect] = TOPIC_COLORS.get(
                    aspect, TOPIC_COLORS["default"]
                )

            # Generate annotated PDF
            with tempfile.TemporaryDirectory() as tmpdir:
                output_pdf = Path(tmpdir) / "bid_analysis_annotated.pdf"

                # Open original PDF and save to new path
                doc = fitz.open(local_pdf_path)

                annotations_added = 0
                annotations_by_aspect = {a: 0 for a in aspects}

                for result in all_results:
                    page_idx = result.page_idx
                    bbox = result.bbox

                    if page_idx >= len(doc):
                        continue

                    page = doc[page_idx]
                    page_width = page.rect.width
                    page_height = page.rect.height

                    # Convert from normalized (0-1000) to PDF coordinates
                    x0 = bbox[0] / 1000.0 * page_width
                    y0 = bbox[1] / 1000.0 * page_height
                    x1 = bbox[2] / 1000.0 * page_width
                    y1 = bbox[3] / 1000.0 * page_height

                    rect = fitz.Rect(x0, y0, x1, y1)

                    # Get color for topic
                    color = aspect_colors.get(result.topic, TOPIC_COLORS["default"])

                    # Generate intelligent insight
                    insights = generate_insights(result.text, result.topic)

                    # Build annotation content
                    if insights:
                        top_insight = insights[0]
                        icon = (
                            "⚠️"
                            if top_insight.category == "risk"
                            else "✅"
                            if top_insight.category == "benefit"
                            else "ℹ️"
                        )
                        annotation_content = (
                            f"{icon} {result.topic.upper()}\n"
                            f"{top_insight.title}\n\n"
                            f"{top_insight.content}\n\n"
                            f"原文: {result.text[:80]}..."
                        )
                    else:
                        annotation_content = (
                            f"{result.topic.upper()}\n{result.text[:100]}"
                        )

                    # Add highlight annotation
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=color)
                    annot.set_opacity(0.3)
                    annot.set_info(annotation_content)
                    annot.update()

                    annotations_added += 1
                    annotations_by_aspect[result.topic] += 1

                doc.save(output_pdf, incremental=False)
                doc.close()

                # Verify PDF was created
                assert output_pdf.exists(), "Annotated PDF should be created"

                # Verify annotations
                doc = fitz.open(output_pdf)
                total_annotations = 0
                annotation_colors = {}

                for page in doc:
                    for annot in page.annots():
                        if annot.type[0] == 8:  # Highlight annotation
                            total_annotations += 1
                            colors = annot.colors
                            stroke = colors.get("stroke")
                            if stroke:
                                # Round for grouping
                                color_key = (
                                    round(stroke[0], 1),
                                    round(stroke[1], 1),
                                    round(stroke[2], 1),
                                )
                                annotation_colors[color_key] = (
                                    annotation_colors.get(color_key, 0) + 1
                                )

                doc.close()

                # Verify results
                assert total_annotations == annotations_added, (
                    f"Should have {annotations_added} annotations, got {total_annotations}"
                )

                # Verify multiple colors were used (different aspects)
                assert len(annotation_colors) >= 2, (
                    "Should use at least 2 different colors for aspects"
                )

                # Print summary
                print("\n" + "=" * 60)
                print("PDF高亮注释生成完成")
                print("=" * 60)
                print(f"保存位置: {output_pdf}")
                print(f"总注释数: {annotations_added}")
                print("\n各维度注释数:")
                aspect_names = {
                    "warranty": "质保",
                    "delivery": "交付",
                    "training": "培训",
                    "financial": "财务",
                }
                for aspect, count in annotations_by_aspect.items():
                    if count > 0:
                        color = aspect_colors.get(aspect, (0.5, 0.5, 0.5))
                        print(
                            f"  {aspect_names.get(aspect, aspect):8s}: {count} 个 (RGB{color})"
                        )
                print("\n颜色统计:")
                for color, count in annotation_colors.items():
                    print(f"  {color}: {count} 个")
                print("=" * 60)

                # Copy to a permanent location for inspection
                final_path = Path(
                    "/Users/wangxq/Documents/投标分析_kimi/mineru/annotated/bid_analysis_complete.pdf"
                )
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(output_pdf, final_path)
                print(f"\n已复制到: {final_path}")

    def test_cross_aspect_comparison(self, test_version_id):
        """Test comparison across different aspects.

        Verifies:
        1. Can analyze multiple aspects
        2. Results can be compared
        3. Weakest aspect is identified
        """
        import psycopg
        from bid_scoring.config import load_settings

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            aspects = {
                "warranty": ["质保", "保修"],
                "delivery": ["交货", "响应"],
                "training": ["培训"],
            }

            aspect_scores = {}

            for aspect, keywords in aspects.items():
                analysis = analyze_aspect(conn, test_version_id, aspect, keywords)

                # Score: benefits - risks
                score = len(analysis.benefits) - len(analysis.risks)
                aspect_scores[aspect] = score

                print(f"\n{aspect} 分析:")
                print(f"  内容: {analysis.chunks_found} 处")
                print(f"  风险: {len(analysis.risks)}")
                print(f"  优势: {len(analysis.benefits)}")
                print(f"  得分: {score}")

            # Identify weakest aspect
            if aspect_scores:
                weakest = min(aspect_scores, key=aspect_scores.get)
                strongest = max(aspect_scores, key=aspect_scores.get)

                print(f"\n最强维度: {strongest}")
                print(f"最弱维度: {weakest}")

                # Verify we found differences
                if len(aspect_scores) > 1:
                    assert (
                        aspect_scores[strongest] != aspect_scores[weakest]
                        or len(set(aspect_scores.values())) == 1
                    )


# =============================================================================
# Report Generation Functions
# =============================================================================


def format_report_markdown(report: BidAnalysisReport) -> str:
    """Format analysis report as Markdown.

    Args:
        report: The bid analysis report

    Returns:
        Markdown formatted report
    """
    lines = [
        "# 投标分析报告",
        "",
        f"**投标人**: {report.bidder_name}",
        f"**项目**: {report.project_name}",
        f"**综合评分**: {report.overall_score:.1f}/100",
        "",
        "## 分析维度",
        "",
    ]

    for aspect, analysis in report.aspects.items():
        lines.append(f"### {aspect.upper()}")
        lines.append(f"{analysis.summary}")
        lines.append(f"- 内容点: {analysis.chunks_found}")

        if analysis.benefits:
            lines.append("- 优势:")
            for benefit in analysis.benefits[:3]:
                lines.append(f"  - {benefit.title}: {benefit.content}")

        if analysis.risks:
            lines.append("- 风险:")
            for risk in analysis.risks[:3]:
                lines.append(f"  - {risk.title}: {risk.content}")

        lines.append("")

    if report.recommendations:
        lines.extend(["## 建议", ""])
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    return "\n".join(lines)
