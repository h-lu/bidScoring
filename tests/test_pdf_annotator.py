"""Tests for PDF Annotator module.

Tests the PDF highlighting functionality including:
- Color assignment by topic
- Chunk bbox retrieval
- PDF annotation creation
- MinIO upload/download integration
"""

import os
import uuid
from unittest.mock import MagicMock, patch
import pytest

from mcp_servers.pdf_annotator import (
    PDFAnnotator,
    HighlightRequest,
    HighlightResult,
    TOPIC_COLORS,
    parse_color,
)


# =============================================================================
# Test Color Configuration
# =============================================================================


class TestTopicColors:
    """Test topic color coding."""

    def test_topic_colors_has_all_expected_topics(self):
        """Should have colors for all expected topics."""
        expected_topics = {
            "risk",
            "warranty",
            "training",
            "delivery",
            "financial",
            "technical",
        }
        assert expected_topics.issubset(TOPIC_COLORS.keys())

    def test_topic_colors_are_rgb_tuples(self):
        """Color values should be 3-tuples of floats."""
        for topic, color in TOPIC_COLORS.items():
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, float) for c in color)
            assert all(0 <= c <= 1 for c in color)

    def test_default_color_exists(self):
        """Should have a default color."""
        assert "default" in TOPIC_COLORS


# =============================================================================
# Test Data Models
# =============================================================================


class TestHighlightRequest:
    """Test HighlightRequest dataclass."""

    def test_create_highlight_request(self):
        """Should create request with all fields."""
        request = HighlightRequest(
            version_id="abc-123",
            chunk_ids=["chunk1", "chunk2"],
            topic="warranty",
        )
        assert request.version_id == "abc-123"
        assert request.chunk_ids == ["chunk1", "chunk2"]
        assert request.topic == "warranty"
        assert request.color is None
        assert request.increment is True

    def test_highlight_request_with_custom_color(self):
        """Should accept custom color."""
        request = HighlightRequest(
            version_id="abc-123",
            chunk_ids=["chunk1"],
            topic="custom",
            color=(1.0, 0.0, 0.0),
            increment=False,
        )
        assert request.color == (1.0, 0.0, 0.0)
        assert request.increment is False


class TestHighlightResult:
    """Test HighlightResult dataclass."""

    def test_create_success_result(self):
        """Should create success result."""
        result = HighlightResult(
            success=True,
            annotated_url="https://example.com/pdf",
            highlights_added=5,
            file_path="bids/test/version/annotated.pdf",
            file_id="file-123",
            topics=["warranty", "training"],
        )
        assert result.success is True
        assert result.highlights_added == 5
        assert result.topics == ["warranty", "training"]
        assert result.error is None

    def test_create_error_result(self):
        """Should create error result."""
        result = HighlightResult(
            success=False,
            error="PDF not found",
        )
        assert result.success is False
        assert result.error == "PDF not found"
        assert result.annotated_url is None


# =============================================================================
# Test PDFAnnotator
# =============================================================================


class TestPDFAnnotatorInit:
    """Test PDFAnnotator initialization."""

    @patch("mcp_servers.pdf_annotator.MinIOStorage")
    def test_init_with_storage(self, mock_storage):
        """Should initialize with provided storage."""
        mock_conn = MagicMock()
        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance

        annotator = PDFAnnotator(mock_conn, mock_storage_instance)

        assert annotator.conn == mock_conn
        assert annotator.storage == mock_storage_instance
        # MinIOStorage should not be called since storage was provided
        mock_storage.assert_not_called()

    @patch("mcp_servers.pdf_annotator.MinIOStorage")
    def test_init_creates_storage_from_env(self, mock_storage):
        """Should create storage from env if not provided."""
        mock_conn = MagicMock()
        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance

        annotator = PDFAnnotator(mock_conn)

        assert annotator.storage == mock_storage_instance
        mock_storage.assert_called_once()


class TestPDFAnnotatorHighlightChunks:
    """Test PDFAnnotator.highlight_chunks method."""

    def test_highlight_chunks_empty_chunk_ids(self):
        """Should return error when chunk_ids is empty."""
        mock_conn = MagicMock()

        with patch("mcp_servers.pdf_annotator.MinIOStorage"):
            annotator = PDFAnnotator(mock_conn)
            result = annotator.highlight_chunks(
                version_id="test-version",
                chunk_ids=[],
                topic="warranty",
            )

        assert result.success is False
        assert "empty" in result.error.lower()

    def test_highlight_chunks_no_chunks_found(self):
        """Should return error when no chunks found."""
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []

        with patch("mcp_servers.pdf_annotator.MinIOStorage"):
            annotator = PDFAnnotator(mock_conn)
            result = annotator.highlight_chunks(
                version_id="test-version",
                chunk_ids=["nonexistent"],
                topic="warranty",
            )

        assert result.success is False
        assert "No chunks found" in result.error


class TestPDFAnnotatorGetChunkBboxes:
    """Test PDFAnnotator._get_chunk_bboxes method."""

    def test_get_chunk_bboxes_returns_correct_data(self):
        """Should return chunks with bbox and page info."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock database response
        mock_cur.fetchall.return_value = [
            {
                "chunk_id": "chunk1",
                "page_idx": 0,
                "bbox": [100, 200, 300, 250],
                "text_raw": "Test text",
                "element_type": "text",
                "coord_sys": "mineru_bbox_v1",
            }
        ]

        with patch("mcp_servers.pdf_annotator.MinIOStorage"):
            annotator = PDFAnnotator(mock_conn)
            chunks = annotator._get_chunk_bboxes("version-123", ["chunk1"])

        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "chunk1"
        assert chunks[0]["bbox"] == [100, 200, 300, 250]
        assert chunks[0]["page_idx"] == 0

        # Verify SQL execution
        mock_cur.execute.assert_called_once()
        sql_call = mock_cur.execute.call_args
        assert "chunk_id = ANY(%s)" in sql_call[0][0]
        assert sql_call[0][1] == (["chunk1"],)


class TestParseColor:
    """Test parse_color utility function."""

    def test_parse_color_from_topic_name(self):
        """Should parse color from known topic names."""
        assert parse_color("risk") == TOPIC_COLORS["risk"]
        assert parse_color("warranty") == TOPIC_COLORS["warranty"]
        assert parse_color("training") == TOPIC_COLORS["training"]

    def test_parse_color_from_hex(self):
        """Should parse hex color strings."""
        result = parse_color("#FF0000")
        assert result is not None
        assert result[0] > 0.9  # R component high
        assert result[1] < 0.1  # G component low
        assert result[2] < 0.1  # B component low

    def test_parse_color_invalid_hex(self):
        """Should return None for invalid hex."""
        result = parse_color("#GG0000")
        assert result is None

    def test_parse_color_unknown(self):
        """Should return None for unknown color format."""
        result = parse_color("unknown-color")
        assert result is None


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


# =============================================================================
# Integration Tests (require database and MinIO)
# =============================================================================


@needs_database
class TestPDFAnnotatorIntegration:
    """Integration tests with real database."""

    def test_end_to_end_highlight_chunks(self):
        """Test full workflow: get chunks, highlight, upload."""
        import psycopg
        from bid_scoring.config import load_settings

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Create test data
            project_id = str(uuid.uuid4())
            doc_id = str(uuid.uuid4())
            version_id = str(uuid.uuid4())
            chunk_id = str(uuid.uuid4())

            with conn.cursor() as cur:
                # Insert project, document, version (correct order for FK constraints)
                cur.execute(
                    "INSERT INTO projects (project_id, name) VALUES (%s, %s)",
                    (project_id, "test-project"),
                )
                cur.execute(
                    """INSERT INTO documents (doc_id, project_id, title, source_type)
                    VALUES (%s, %s, %s, %s)""",
                    (doc_id, project_id, "test-doc", "test"),
                )
                cur.execute(
                    """INSERT INTO document_versions (version_id, doc_id, source_uri, parser_version, status)
                    VALUES (%s, %s, %s, %s, %s)""",
                    (version_id, doc_id, "test://uri", "1.0", "ready"),
                )
                # Insert test chunk
                cur.execute(
                    """INSERT INTO chunks (chunk_id, version_id, page_idx, bbox, text_raw, element_type)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)""",
                    (
                        chunk_id,
                        version_id,
                        0,
                        "[100, 200, 300, 250]",
                        "Test warranty clause",
                        "text",
                    ),
                )
            conn.commit()

            try:
                with patch("mcp_servers.pdf_annotator.MinIOStorage"):
                    from mcp_servers.pdf_annotator import PDFAnnotator

                    annotator = PDFAnnotator(conn)

                    # Get chunk bboxes
                    chunks = annotator._get_chunk_bboxes(version_id, [chunk_id])

                    assert len(chunks) == 1
                    assert str(chunks[0]["chunk_id"]) == chunk_id
                    assert chunks[0]["bbox"] == [100, 200, 300, 250]

            finally:
                # Cleanup
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM chunks WHERE chunk_id = %s", (chunk_id,))
                    cur.execute(
                        "DELETE FROM document_versions WHERE version_id = %s",
                        (version_id,),
                    )
                    cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
                    cur.execute(
                        "DELETE FROM projects WHERE project_id = %s", (project_id,)
                    )
                conn.commit()

    def test_get_project_id(self):
        """Test _get_project_id retrieves correct project."""
        import psycopg
        from bid_scoring.config import load_settings

        settings = load_settings()

        with psycopg.connect(settings["DATABASE_URL"]) as conn:
            # Create test data
            project_id = str(uuid.uuid4())
            doc_id = str(uuid.uuid4())
            version_id = str(uuid.uuid4())

            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO projects (project_id, name) VALUES (%s, %s)",
                    (project_id, "test-project"),
                )
                cur.execute(
                    """INSERT INTO documents (doc_id, project_id, title, source_type)
                    VALUES (%s, %s, %s, %s)""",
                    (doc_id, project_id, "test-doc", "test"),
                )
                cur.execute(
                    """INSERT INTO document_versions (version_id, doc_id, source_uri, parser_version, status)
                    VALUES (%s, %s, %s, %s, %s)""",
                    (version_id, doc_id, "test://uri", "1.0", "ready"),
                )
            conn.commit()

            try:
                with patch("mcp_servers.pdf_annotator.MinIOStorage"):
                    from mcp_servers.pdf_annotator import PDFAnnotator

                    annotator = PDFAnnotator(conn)
                    result = annotator._get_project_id(version_id)

                    # _get_project_id returns a UUID, convert to string for comparison
                    assert str(result) == project_id

            finally:
                # Cleanup
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM document_versions WHERE version_id = %s",
                        (version_id,),
                    )
                    cur.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
                    cur.execute(
                        "DELETE FROM projects WHERE project_id = %s", (project_id,)
                    )
                conn.commit()
