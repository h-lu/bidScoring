"""Tests for processing coordinator."""

import uuid
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

needs_all = pytest.mark.skipif(
    not all([os.getenv("DATABASE_URL"), os.getenv("OPENAI_API_KEY")]),
    reason="Required credentials not set"
)


class TestProcessingCoordinator:
    """Test ProcessingCoordinator class."""

    @patch("mineru.coordinator.MinIOStorage")
    @patch("mineru.coordinator.create_storage_from_env")
    def test_init_creates_storage(self, mock_create_storage, mock_storage_class):
        """Should initialize with MinIO storage."""
        from mineru.coordinator import ProcessingCoordinator

        mock_storage = MagicMock()
        mock_create_storage.return_value = mock_storage

        coord = ProcessingCoordinator()

        mock_create_storage.assert_called_once()
        assert coord.storage == mock_storage

    @patch("mineru.coordinator.create_storage_from_env")
    @patch("mineru.coordinator.EmbeddingBatchService")
    def test_process_pdf_workflow(self, mock_embed_service, mock_storage):
        """Should execute full PDF processing workflow."""
        from mineru.coordinator import ProcessingCoordinator

        # Setup mocks
        mock_storage_inst = MagicMock()
        mock_storage.return_value = mock_storage_inst
        mock_storage_inst.upload_directory.return_value = [
            {"object_key": "bids/proj/version/files/original/test.pdf", "size": 1000},
            {"object_key": "bids/proj/version/files/parsed/full.md", "size": 500},
        ]

        mock_embed_inst = MagicMock()
        mock_embed_service.return_value = mock_embed_inst
        mock_embed_inst.process_version.return_value = {
            "total_processed": 10,
            "succeeded": 10,
            "failed": 0,
        }

        # Mock database
        mock_conn = MagicMock()

        # Create temp directory with content_list.json
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            content_list = [
                {"type": "text", "text": "Hello", "page_idx": 0, "bbox": [0, 0, 100, 100]},
                {"type": "image", "img_path": "images/img1.png", "page_idx": 0},
            ]
            (output_dir / "content_list.json").write_text(json.dumps(content_list))

            coord = ProcessingCoordinator()

            # Mock the ingest function
            with patch("mineru.coordinator.ingest_content_list") as mock_ingest:
                mock_ingest.return_value = {"total_chunks": 2}

                result = coord.process_existing_output(
                    output_dir=output_dir,
                    project_id=str(uuid.uuid4()),
                    document_id=str(uuid.uuid4()),
                    version_id=str(uuid.uuid4()),
                    conn=mock_conn,
                )

                # Verify workflow steps
                mock_storage_inst.upload_directory.assert_called()
                mock_ingest.assert_called_once()
                mock_embed_inst.process_version.assert_called_once()

    @patch("mineru.coordinator.create_storage_from_env")
    def test_generate_ids(self, mock_storage):
        """Should generate UUIDs for project/document/version."""
        from mineru.coordinator import ProcessingCoordinator

        coord = ProcessingCoordinator()

        project_id = coord.generate_project_id("test-project")
        document_id = coord.generate_document_id(project_id, "test.pdf")
        version_id = coord.generate_version_id()

        assert project_id is not None
        assert document_id is not None
        assert version_id is not None
        assert project_id != document_id != version_id


class TestHelperFunctions:
    """Test helper functions."""

    def test_load_content_list(self):
        """Should load content_list.json from directory."""
        from mineru.coordinator import load_content_list

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            content_list = [
                {"type": "text", "text": "Hello"},
            ]
            (output_dir / "content_list.json").write_text(json.dumps(content_list))

            result = load_content_list(output_dir)

            assert result == content_list

    def test_load_content_list_missing(self):
        """Should return empty list if file doesn't exist."""
        from mineru.coordinator import load_content_list

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_content_list(Path(tmpdir))

            assert result == []
