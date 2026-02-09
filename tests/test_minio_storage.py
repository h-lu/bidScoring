"""Tests for MinIO storage module."""

import os
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

needs_minio = pytest.mark.skipif(
    not all([os.getenv("MINIO_ENDPOINT"), os.getenv("MINIO_ACCESS_KEY"), os.getenv("MINIO_SECRET_KEY")]),
    reason="MinIO credentials not set"
)


class TestMinIOStorageInit:
    """Test MinIOStorage initialization."""

    @patch("mineru.minio_storage.Minio")
    def test_init_with_defaults(self, mock_minio):
        """Should initialize with default secure=False."""
        from mineru.minio_storage import MinIOStorage

        storage = MinIOStorage(
            endpoint="localhost:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket"
        )

        mock_minio.assert_called_once()
        call_kwargs = mock_minio.call_args[1]
        assert call_kwargs["endpoint"] == "localhost:9000"
        assert call_kwargs["access_key"] == "key"
        assert call_kwargs["secret_key"] == "secret"
        assert call_kwargs["secure"] is False

    @patch("mineru.minio_storage.Minio")
    def test_init_with_secure(self, mock_minio):
        """Should initialize with secure=True when specified."""
        from mineru.minio_storage import MinIOStorage

        storage = MinIOStorage(
            endpoint="minio.example.com:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket",
            secure=True
        )

        call_kwargs = mock_minio.call_args[1]
        assert call_kwargs["secure"] is True


class TestMinIOStorageUpload:
    """Test MinIOStorage upload operations."""

    @patch("mineru.minio_storage.Minio")
    def test_upload_file_success(self, mock_minio_class):
        """Should upload file and return file record."""
        from mineru.minio_storage import MinIOStorage

        # Setup mocks
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_result = MagicMock(object_name="test.pdf", etag="abc123", size=12345)
        mock_client.fput_object.return_value = mock_result
        mock_client.bucket_exists.return_value = True

        storage = MinIOStorage(
            endpoint="localhost:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket"
        )

        # Create temp test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            result = storage.upload_file(
                local_path=temp_path,
                object_key="bids/test/version/test.pdf",
                metadata={"project_id": "test"}
            )

            assert result["object_key"] == "bids/test/version/test.pdf"
            assert result["etag"] == "abc123"
            assert result["size"] == 12345
            assert result["metadata"]["project_id"] == "test"

            # Verify fput_object was called
            mock_client.fput_object.assert_called_once()
        finally:
            temp_path.unlink()

    @patch("mineru.minio_storage.Minio")
    def test_upload_file_with_content_type(self, mock_minio_class):
        """Should upload file with specified content type."""
        from mineru.minio_storage import MinIOStorage

        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_result = MagicMock(object_name="test.pdf", etag="abc123", size=12345)
        mock_client.fput_object.return_value = mock_result
        mock_client.bucket_exists.return_value = True

        storage = MinIOStorage(
            endpoint="localhost:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket"
        )

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            temp_path = Path(f.name)

        try:
            storage.upload_file(
                local_path=temp_path,
                object_key="test.pdf",
                content_type="application/pdf"
            )

            call_args = mock_client.fput_object.call_args
            # Check kwargs for content_type
            assert call_args[1].get("content_type") == "application/pdf"
        finally:
            temp_path.unlink()


class TestMinIOStoragePresignedUrl:
    """Test MinIOStorage presigned URL generation."""

    @patch("mineru.minio_storage.Minio")
    def test_generate_presigned_url(self, mock_minio_class):
        """Should generate presigned URL for download."""
        from mineru.minio_storage import MinIOStorage
        from datetime import timedelta

        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        mock_client.presigned_get_object.return_value = "https://minio.example.com/test-bucket/test.pdf?expires=123"

        storage = MinIOStorage(
            endpoint="localhost:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket"
        )

        url = storage.generate_presigned_url(
            object_key="bids/test/version/test.pdf",
            expires=timedelta(hours=1)
        )

        assert url == "https://minio.example.com/test-bucket/test.pdf?expires=123"
        mock_client.presigned_get_object.assert_called_once()


class TestMinIOStoragePathHelpers:
    """Test MinIOStorage path helper methods."""

    @patch("mineru.minio_storage.Minio")
    def test_build_object_key(self, mock_minio_class):
        """Should build correct object key from components."""
        from mineru.minio_storage import MinIOStorage

        storage = MinIOStorage(
            endpoint="localhost:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket"
        )

        key = storage.build_object_key(
            project_id=str(uuid.uuid4()),
            version_id=str(uuid.uuid4()),
            file_type="original",
            file_name="test.pdf"
        )

        assert key.startswith("bids/")
        assert "/files/original/" in key
        assert key.endswith("test.pdf")

    @patch("mineru.minio_storage.Minio")
    def test_get_file_type_from_path(self, mock_minio_class):
        """Should detect file type from path."""
        from mineru.minio_storage import MinIOStorage

        storage = MinIOStorage(
            endpoint="localhost:9000",
            access_key="key",
            secret_key="secret",
            bucket="test-bucket"
        )

        # get_file_type expects full object key path
        assert storage.get_file_type("bids/proj/version/files/original/test.pdf") == "original"
        assert storage.get_file_type("bids/proj/version/files/parsed/full.md") == "parsed"
        assert storage.get_file_type("bids/proj/version/files/images/img_001.png") == "images"
        assert storage.get_file_type("invalid/path") == "unknown"
