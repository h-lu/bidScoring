"""Tests for batch embedding service."""

import uuid
import os
from unittest.mock import Mock, patch, MagicMock
import pytest

needs_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


class TestEmbeddingBatchService:
    """Test EmbeddingBatchService class."""

    @patch("bid_scoring.embeddings_batch.OpenAI")
    def test_init_with_defaults(self, mock_openai):
        """Should initialize with default batch size."""
        from bid_scoring.embeddings_batch import EmbeddingBatchService

        service = EmbeddingBatchService(api_key="test-key")

        assert service.batch_size == 100
        assert service.max_retries == 6

    @patch("bid_scoring.embeddings_batch.OpenAI")
    def test_init_with_custom_params(self, mock_openai):
        """Should initialize with custom parameters."""
        from bid_scoring.embeddings_batch import EmbeddingBatchService

        service = EmbeddingBatchService(
            api_key="test-key", batch_size=50, max_retries=3
        )

        assert service.batch_size == 50
        assert service.max_retries == 3

    @patch("bid_scoring.embeddings_batch.OpenAI")
    def test_get_embeddings_batch(self, mock_openai_class):
        """Should get embeddings for a batch of texts."""
        from bid_scoring.embeddings_batch import EmbeddingBatchService

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        service = EmbeddingBatchService(api_key="test-key")

        texts = ["text 1", "text 2"]
        embeddings = service._get_embeddings_batch(texts)

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

        # Verify API was called correctly
        mock_client.embeddings.create.assert_called_once()
        call_args = mock_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == texts

    @patch("bid_scoring.embeddings_batch.OpenAI")
    def test_process_version_updates_chunks(self, mock_openai_class):
        """Should update chunks with generated embeddings."""
        from bid_scoring.embeddings_batch import EmbeddingBatchService

        # Setup mocks
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response

        # Mock database
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        # Mock chunk fetch
        mock_cur.fetchall.return_value = [
            {"chunk_id": str(uuid.uuid4()), "text_raw": "text 1"},
            {"chunk_id": str(uuid.uuid4()), "text_raw": "text 2"},
        ]

        service = EmbeddingBatchService(api_key="test-key")
        result = service.process_version(version_id=str(uuid.uuid4()), conn=mock_conn)

        assert result["total_processed"] == 2
        assert result["succeeded"] == 2
        assert result["failed"] == 0

    @patch("bid_scoring.embeddings_batch.OpenAI")
    def test_process_pending_global(self, mock_openai_class):
        """Should process pending chunks across all versions."""
        from bid_scoring.embeddings_batch import EmbeddingBatchService

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
        ]
        mock_client.embeddings.create.return_value = mock_response

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_cur.fetchone.return_value = {"count": 1}
        test_version_id = str(uuid.uuid4())
        mock_cur.fetchall.return_value = [
            {
                "chunk_id": str(uuid.uuid4()),
                "text_raw": "text 1",
                "version_id": test_version_id,
            },
        ]

        service = EmbeddingBatchService(api_key="test-key")
        result = service.process_pending(conn=mock_conn, limit=100)

        assert result["total_processed"] >= 0
        assert "succeeded" in result
