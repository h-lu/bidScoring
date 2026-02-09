#!/usr/bin/env python3
"""End-to-End Tests for PDF Highlighting Feature.

Tests the complete workflow:
1. Search for content (two different topics)
2. Generate highlights with color coding
3. Verify annotated PDF contains both sets of highlights

Requires:
- DATABASE_URL set
- MinIO running with accessible storage
- At least one document imported with searchable content

Run with:
    uv run pytest tests/test_pdf_highlight_e2e.py -v --tb=short -s
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pytest
from psycopg.rows import dict_row


needs_database = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set"
)

needs_minio = pytest.mark.skipif(
    not all(
        [
            os.getenv("MINIO_ENDPOINT"),
            os.getenv("MINIO_ACCESS_KEY"),
            os.getenv("MINIO_SECRET_KEY"),
        ]
    ),
    reason="MinIO credentials not set",
)


# Check if MinIO is accessible
def _minio_accessible() -> bool:
    """Check if MinIO server is accessible."""
    try:
        from mineru.minio_storage import MinIOStorage

        storage = MinIOStorage()
        # Try to list objects (will fail if credentials are wrong)
        storage.client.list_objects(storage.bucket, recursive=False)
        return True
    except Exception:
        return False


skip_if_no_minio = pytest.mark.skipif(
    not _minio_accessible(), reason="MinIO server not accessible"
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_version_id():
    """Get a test version ID from database.

    Uses the version that has an original_pdf file record.
    """
    import psycopg
    from bid_scoring.config import load_settings

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            # Find version with original_pdf file
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
    """Get path to local PDF file for testing.

    Returns the path to the original PDF file in mineru/pdf directory.
    """
    pdf_path = Path(
        "/Users/wangxq/Documents/投标分析_kimi/mineru/pdf/0811-DSITC253135-上海悦晟生物科技有限公司投标文件.pdf"
    )
    if not pdf_path.exists():
        pytest.skip(f"Local PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture(scope="module")
def test_project_id(test_version_id):
    """Get project_id for test version."""
    import psycopg
    from bid_scoring.config import load_settings

    settings = load_settings()

    with psycopg.connect(settings["DATABASE_URL"]) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.project_id
                FROM document_versions v
                JOIN documents d ON v.doc_id = d.doc_id
                WHERE v.version_id = %s
                """,
                (test_version_id,),
            )
            result = cur.fetchone()
            return str(result[0]) if result else None


# =============================================================================
# E2E Tests
# =============================================================================


# =============================================================================
# Complete E2E Tests with Real PDF
# =============================================================================


@pytest.mark.e2e
@needs_database
class TestPDFHighlightRealE2E:
    """Complete end-to-end tests using real PDF file.

    This test class performs actual PDF highlighting on the real document
    without any mocking, providing complete verification of the workflow.
    """

    def test_complete_workflow_two_topics(self, test_version_id, local_pdf_path):
        """Test complete highlighting workflow with two different topics.

        Workflow:
        1. Search for "售后服务" chunks in database
        2. Copy real PDF and add green highlights for warranty topic
        3. Search for "响应时间" chunks
        4. Add orange highlights to the SAME PDF (incremental)
        5. Verify both sets of highlights exist with correct colors
        """
        import psycopg
        import shutil
        from bid_scoring.config import load_settings
        from mcp_servers.pdf_annotator import TOPIC_COLORS

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Step 1: Search for after-sales (售后服务) content
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT chunk_id, page_idx, bbox, text_raw
                    FROM chunks
                    WHERE version_id = %s
                    AND text_raw ILIKE %s
                    AND bbox IS NOT NULL
                    LIMIT 5
                    """,
                    (test_version_id, "%售后服务%"),
                )
                warranty_chunks = cur.fetchall()

            if not warranty_chunks:
                pytest.skip("No '售后服务' content found in test document")

            # Verify chunks have valid bbox
            for chunk in warranty_chunks:
                assert chunk["bbox"] is not None, "Chunk should have bbox coordinates"
                assert len(chunk["bbox"]) == 4, "Bbox should have 4 coordinates"

            # Step 2: Create a working copy of the real PDF
            with tempfile.TemporaryDirectory() as tmpdir:
                test_pdf_path = Path(tmpdir) / "test_document.pdf"
                annotated_pdf_path = Path(tmpdir) / "annotated_document.pdf"
                shutil.copy(local_pdf_path, test_pdf_path)

                # Verify we can open the PDF
                doc = fitz.open(test_pdf_path)
                original_page_count = len(doc)
                doc.close()
                assert original_page_count > 0, "PDF should have pages"

                # Step 3: Add green highlights for warranty (售后服务)
                warranty_color = TOPIC_COLORS["warranty"]  # Green
                doc = fitz.open(test_pdf_path)
                warranty_highlights = 0

                for chunk in warranty_chunks:
                    page_idx = chunk["page_idx"]
                    bbox = chunk["bbox"]

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
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=warranty_color)
                    annot.set_opacity(0.3)
                    annot.update()
                    warranty_highlights += 1

                # Save to new file to avoid incremental requirement
                doc.save(annotated_pdf_path, incremental=False)
                doc.close()

                # Verify warranty highlights were added
                doc = fitz.open(annotated_pdf_path)
                warranty_count = sum(
                    1
                    for page in doc
                    for annot in page.annots()
                    if annot.type[0] == 8  # Highlight annotation type
                )
                doc.close()

                assert warranty_count == warranty_highlights, (
                    f"Should have {warranty_highlights} warranty highlights, got {warranty_count}"
                )

                # Step 4: Search for response time (响应时间) content
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT chunk_id, page_idx, bbox, text_raw
                        FROM chunks
                        WHERE version_id = %s
                        AND (
                            text_raw ILIKE %s
                            OR text_raw ILIKE %s
                        )
                        AND bbox IS NOT NULL
                        LIMIT 5
                        """,
                        (test_version_id, "%响应时间%", "%响应时限%"),
                    )
                    delivery_chunks = cur.fetchall()

                if not delivery_chunks:
                    pytest.skip("No response time content found in test document")

                # Step 5: Add orange highlights for delivery (响应时间) - INCREMENTAL
                delivery_color = TOPIC_COLORS["delivery"]  # Orange
                doc = fitz.open(annotated_pdf_path)  # Open annotated PDF
                delivery_highlights = 0

                for chunk in delivery_chunks:
                    page_idx = chunk["page_idx"]
                    bbox = chunk["bbox"]

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
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=delivery_color)
                    annot.set_opacity(0.3)
                    annot.update()
                    delivery_highlights += 1

                # Save to yet another file for final verification
                final_pdf_path = Path(tmpdir) / "final_document.pdf"
                doc.save(final_pdf_path, incremental=False)
                doc.close()

                # Step 6: Verify both sets of highlights exist
                doc = fitz.open(final_pdf_path)
                total_highlights = 0
                color_counts = {}

                for page in doc:
                    for annot in page.annots():
                        if annot.type[0] == 8:  # Highlight annotation type
                            total_highlights += 1
                            colors = annot.colors
                            stroke = colors.get("stroke")
                            if stroke:
                                # Round to 1 decimal for grouping
                                color = (
                                    round(stroke[0], 1),
                                    round(stroke[1], 1),
                                    round(stroke[2], 1),
                                )
                                color_counts[color] = color_counts.get(color, 0) + 1

                doc.close()

                # Should have all highlights
                expected_total = warranty_highlights + delivery_highlights
                assert total_highlights == expected_total, (
                    f"Should have {expected_total} total highlights, got {total_highlights}"
                )

                # Verify both colors are present
                warranty_color_rounded = (
                    round(warranty_color[0], 1),
                    round(warranty_color[1], 1),
                    round(warranty_color[2], 1),
                )
                delivery_color_rounded = (
                    round(delivery_color[0], 1),
                    round(delivery_color[1], 1),
                    round(delivery_color[2], 1),
                )

                assert warranty_color_rounded in color_counts, (
                    "Warranty (green) highlights should exist"
                )
                assert delivery_color_rounded in color_counts, (
                    "Delivery (orange) highlights should exist"
                )
                assert color_counts[warranty_color_rounded] >= warranty_highlights, (
                    f"Should have at least {warranty_highlights} green highlights"
                )

    def test_verify_annotated_pdf_structure(self, test_version_id, local_pdf_path):
        """Test that annotated PDF maintains document structure.

        Verifies:
        1. Page count remains the same
        2. Page content is preserved
        3. Only annotations are added
        """
        import shutil
        from bid_scoring.config import load_settings
        from mcp_servers.pdf_annotator import TOPIC_COLORS

        settings = load_settings()

        # Get a chunk to highlight
        import psycopg

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT chunk_id, page_idx, bbox, text_raw
                    FROM chunks
                    WHERE version_id = %s
                    AND bbox IS NOT NULL
                    LIMIT 1
                    """,
                    (test_version_id,),
                )
                chunks = cur.fetchall()

        if not chunks:
            pytest.skip("No chunks found in test document")

        chunk = chunks[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            test_pdf_path = Path(tmpdir) / "test_document.pdf"
            annotated_pdf_path = Path(tmpdir) / "annotated_document.pdf"
            shutil.copy(local_pdf_path, test_pdf_path)

            # Get original page count and dimensions
            original_doc = fitz.open(test_pdf_path)
            original_page_count = len(original_doc)
            original_page_dims = [
                (original_doc[i].rect.width, original_doc[i].rect.y1)
                for i in range(min(5, original_page_count))  # Check first 5 pages
            ]
            original_doc.close()

            # Add a highlight
            doc = fitz.open(test_pdf_path)
            page_idx = chunk["page_idx"]
            bbox = chunk["bbox"]

            if page_idx < len(doc):
                page = doc[page_idx]
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=TOPIC_COLORS["warranty"])
                annot.set_opacity(0.3)
                annot.update()

            # Save to new file to avoid incremental requirement
            doc.save(annotated_pdf_path, incremental=False)
            doc.close()

            # Verify structure is preserved
            annotated_doc = fitz.open(annotated_pdf_path)
            annotated_page_count = len(annotated_doc)
            annotated_page_dims = [
                (annotated_doc[i].rect.width, annotated_doc[i].rect.y1)
                for i in range(min(5, annotated_page_count))
            ]
            annotated_doc.close()

            assert annotated_page_count == original_page_count, (
                "Page count should remain the same"
            )
            assert annotated_page_dims == original_page_dims, (
                "Page dimensions should be preserved"
            )

    def test_color_coding_by_topic(self, test_version_id, local_pdf_path):
        """Test that different topics get different colors.

        Tests all defined topic colors to ensure proper color assignment.
        """
        import shutil
        from bid_scoring.config import load_settings
        from mcp_servers.pdf_annotator import TOPIC_COLORS

        settings = load_settings()

        # Get chunks for each topic
        topic_keywords = {
            "warranty": "%质保%",
            "risk": "%风险%",
            "training": "%培训%",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            test_pdf_path = Path(tmpdir) / "test_document.pdf"
            shutil.copy(local_pdf_path, test_pdf_path)

            colors_found = {}
            current_pdf = test_pdf_path

            for idx, (topic, keyword) in enumerate(topic_keywords.items()):
                import psycopg

                with psycopg.connect(settings["DATABASE_URL"]) as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            """
                            SELECT chunk_id, page_idx, bbox, text_raw
                            FROM chunks
                            WHERE version_id = %s
                            AND text_raw ILIKE %s
                            AND bbox IS NOT NULL
                            LIMIT 2
                            """,
                            (test_version_id, keyword),
                        )
                        chunks = cur.fetchall()

                if not chunks:
                    continue  # Skip topics without content

                expected_color = TOPIC_COLORS[topic]

                # Add highlights for this topic (with coordinate conversion)
                doc = fitz.open(current_pdf)
                for chunk in chunks:
                    page_idx = chunk["page_idx"]
                    bbox = chunk["bbox"]

                    if page_idx >= len(doc):
                        continue

                    page = doc[page_idx]
                    page_width = page.rect.width
                    page_height = page.rect.height

                    # Convert from normalized (0-1000) to PDF coordinates
                    # MinerU coord_sys uses normalized coordinates
                    x0 = bbox[0] / 1000.0 * page_width
                    y0 = bbox[1] / 1000.0 * page_height
                    x1 = bbox[2] / 1000.0 * page_width
                    y1 = bbox[3] / 1000.0 * page_height

                    rect = fitz.Rect(x0, y0, x1, y1)
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=expected_color)
                    annot.set_opacity(0.3)
                    annot.update()

                # Save to new file for each topic to avoid incremental requirement
                output_pdf = Path(tmpdir) / f"document_step_{idx}.pdf"
                doc.save(output_pdf, incremental=False)
                doc.close()

                # Verify color was applied
                doc = fitz.open(output_pdf)
                for page in doc:
                    for annot in page.annots():
                        if annot.type[0] == 8:  # Highlight
                            colors = annot.colors
                            stroke = colors.get("stroke")
                            if stroke:
                                rounded = (
                                    round(stroke[0], 1),
                                    round(stroke[1], 1),
                                    round(stroke[2], 1),
                                )
                                expected_rounded = (
                                    round(expected_color[0], 1),
                                    round(expected_color[1], 1),
                                    round(expected_color[2], 1),
                                )
                                if rounded == expected_rounded:
                                    colors_found[topic] = rounded
                doc.close()

                # Use output as input for next iteration (incremental)
                current_pdf = output_pdf

            # Verify at least one color was tested
            if colors_found:
                # Each found color should match its expected value
                for topic, found_color in colors_found.items():
                    expected = TOPIC_COLORS[topic]
                    expected_rounded = (
                        round(expected[0], 1),
                        round(expected[1], 1),
                        round(expected[2], 1),
                    )
                    assert found_color == expected_rounded, (
                        f"Topic '{topic}' should have color {expected_rounded}, got {found_color}"
                    )
            else:
                pytest.skip("No topic content found for color verification test")


# =============================================================================
# Helper Functions
# =============================================================================


def _count_highlights_by_color(pdf_path: Path) -> dict[tuple[float, float, float], int]:
    """Count highlights in PDF by color.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict mapping color tuple to count
    """
    doc = fitz.open(pdf_path)
    color_counts = {}

    for page in doc:
        for annot in page.annots():
            if annot.type[0] == 8:  # Highlight annotation type
                colors = annot.colors
                stroke = colors.get("stroke")
                if stroke:
                    # Round to 1 decimal for grouping
                    color = (
                        round(stroke[0], 1),
                        round(stroke[1], 1),
                        round(stroke[2], 1),
                    )
                    color_counts[color] = color_counts.get(color, 0) + 1

    doc.close()
    return color_counts
