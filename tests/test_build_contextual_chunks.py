"""Tests for build_contextual_chunks script."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from build_contextual_chunks import (
    get_stats,
    fetch_batch,
    _get_surrounding_chunks,
    process_batch,
    format_duration,
    DEFAULT_BATCH_SIZE,
)

# Also need to mock the imports in the module
import build_contextual_chunks as bcc_module


class TestGetStats:
    """Test statistics retrieval."""

    def test_get_stats_basic(self):
        """Should return correct statistics."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock query result: total_chunks, processed, to_process, empty_text
        mock_cur.fetchone.return_value = (100, 30, 65, 5)
        
        stats = get_stats(mock_conn)
        
        assert stats["total_chunks"] == 100
        assert stats["processed"] == 30
        assert stats["to_process"] == 65
        assert stats["empty_text"] == 5
        mock_cur.execute.assert_called_once()

    def test_get_stats_with_version_id(self):
        """Should filter by version_id when provided."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchone.return_value = (50, 20, 28, 2)
        
        stats = get_stats(mock_conn, version_id="test-version-id")
        
        # Check that version_id was used in query
        call_args = mock_cur.execute.call_args
        assert "test-version-id" in str(call_args)
        assert stats["total_chunks"] == 50


class TestFetchBatch:
    """Test batch fetching from database."""

    def test_fetch_batch_basic(self):
        """Should fetch chunks that need processing."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock two chunks to return
        mock_cur.fetchall.return_value = [
            ("chunk-1", "version-1", "Text content 1", 0, "Doc Title", 100),
            ("chunk-2", "version-1", "Text content 2", 1, "Doc Title", 100),
        ]
        mock_cur.fetchone.return_value = ("prev-chunk",)
        
        chunks = fetch_batch(mock_conn, batch_size=5)
        
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "chunk-1"
        assert chunks[0]["text_raw"] == "Text content 1"
        assert chunks[0]["document_title"] == "Doc Title"

    def test_fetch_batch_empty(self):
        """Should return empty list when no chunks to process."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchall.return_value = []
        
        chunks = fetch_batch(mock_conn)
        
        assert chunks == []

    def test_fetch_batch_respects_token_limit(self):
        """Should respect max_tokens limit."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Return many chunks, but token limit should restrict batch
        long_text = "x" * 1000  # ~500 tokens
        mock_cur.fetchall.return_value = [
            (f"chunk-{i}", "version-1", long_text, i, "Doc", 1000)
            for i in range(10)
        ]
        mock_cur.fetchone.return_value = ("prev-chunk",)
        
        # With max_tokens=1000, should only get 1-2 chunks
        chunks = fetch_batch(mock_conn, batch_size=10, max_tokens=1000)
        
        assert len(chunks) <= 2

    def test_fetch_batch_with_version_filter(self):
        """Should filter by version_id."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchall.return_value = [
            ("chunk-1", "version-abc", "Text", 0, "Doc", 50),
        ]
        mock_cur.fetchone.return_value = ("prev-chunk",)
        
        chunks = fetch_batch(mock_conn, version_id="version-abc")
        
        # Check version_id was used in query
        call_args = mock_cur.execute.call_args
        assert "version-abc" in str(call_args)


class TestGetSurroundingChunks:
    """Test fetching surrounding chunks."""

    def test_get_surrounding_chunks(self):
        """Should return surrounding chunk texts."""
        mock_cur = Mock()
        mock_cur.fetchall.return_value = [
            ("Previous chunk text",),
            ("Current chunk text",),
            ("Next chunk text",),
        ]
        
        result = _get_surrounding_chunks(mock_cur, "version-1", 5)
        
        assert len(result) == 3
        assert "Previous chunk text" in result
        assert "Next chunk text" in result

    def test_get_surrounding_chunks_empty(self):
        """Should handle no surrounding chunks."""
        mock_cur = Mock()
        mock_cur.fetchall.return_value = []
        
        result = _get_surrounding_chunks(mock_cur, "version-1", 0)
        
        assert result == []

    def test_get_surrounding_chunks_filters_null(self):
        """Should filter out NULL text."""
        mock_cur = Mock()
        mock_cur.fetchall.return_value = [
            ("Valid text",),
            (None,),
            ("",),
            ("Another valid",),
        ]
        
        result = _get_surrounding_chunks(mock_cur, "version-1", 5)
        
        assert len(result) == 2
        assert "Valid text" in result
        assert "Another valid" in result


class TestProcessBatch:
    """Test batch processing."""

    def test_process_batch_success(self):
        """Should successfully process a batch."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock context generator
        mock_generator = Mock()
        mock_generator.model = "gpt-4"
        mock_generator.generate_context_batch.return_value = [
            "Context for chunk 1",
            "Context for chunk 2",
        ]
        
        # Mock embeddings
        with patch.object(bcc_module, 'embed_texts') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            
            chunks = [
                {
                    "chunk_id": "chunk-1",
                    "version_id": "version-1",
                    "text_raw": "Text 1",
                    "document_title": "Doc",
                },
                {
                    "chunk_id": "chunk-2",
                    "version_id": "version-1",
                    "text_raw": "Text 2",
                    "document_title": "Doc",
                },
            ]
            
            success, fail = process_batch(
                mock_conn, chunks, mock_generator,
                embedding_client=Mock(),
                embedding_model="text-embedding-3-small",
            )
            
            assert success == 2
            assert fail == 0
            mock_conn.commit.assert_called_once()
            mock_cur.executemany.assert_called_once()

    def test_process_batch_empty(self):
        """Should handle empty batch."""
        mock_conn = Mock()
        mock_generator = Mock()
        
        success, fail = process_batch(mock_conn, [], mock_generator)
        
        assert success == 0
        assert fail == 0

    def test_process_batch_failure(self):
        """Should handle batch processing failure."""
        mock_conn = Mock()
        mock_generator = Mock()
        mock_generator.generate_context_batch.side_effect = Exception("API Error")
        
        chunks = [
            {"chunk_id": "chunk-1", "version_id": "v1", "text_raw": "Text", "document_title": "Doc"},
        ]
        
        success, fail = process_batch(mock_conn, chunks, mock_generator)
        
        assert success == 0
        assert fail == 1
        mock_conn.rollback.assert_called_once()

    def test_process_batch_contextualized_text_format(self):
        """Should format contextualized text correctly."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_generator = Mock()
        mock_generator.model = "gpt-4"
        mock_generator.generate_context_batch.return_value = ["Generated context"]
        
        with patch.object(bcc_module, 'embed_texts') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]
            
            chunks = [
                {
                    "chunk_id": "chunk-1",
                    "version_id": "version-1",
                    "text_raw": "Original text",
                    "document_title": "Doc",
                },
            ]
            
            process_batch(mock_conn, chunks, mock_generator)
            
            # Check that insert was called with correct contextualized text
            call_args = mock_cur.executemany.call_args
            insert_data = call_args[0][1]
            assert len(insert_data) == 1
            # contextualized_text should be: context + "\n\n" + original_text
            assert "Generated context" in insert_data[0][4]
            assert "Original text" in insert_data[0][4]


class TestFormatDuration:
    """Test duration formatting."""

    def test_format_duration_seconds(self):
        """Should format seconds."""
        assert format_duration(45.5) == "45.5秒"

    def test_format_duration_minutes(self):
        """Should format minutes."""
        assert format_duration(120) == "2.0分钟"
        assert format_duration(90) == "1.5分钟"

    def test_format_duration_hours(self):
        """Should format hours."""
        assert format_duration(3600) == "1.0小时"
        assert format_duration(7200) == "2.0小时"


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_batch_size(self):
        """Should have reasonable default batch size."""
        assert DEFAULT_BATCH_SIZE == 5
        assert DEFAULT_BATCH_SIZE <= 10  # Should be small for LLM calls


class TestIntegrationPatterns:
    """Integration-style tests for the full flow."""

    def test_batch_processing_flow(self):
        """Test complete batch processing flow."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock context generator
        mock_generator = Mock()
        mock_generator.model = "gpt-4"
        mock_generator.generate_context_batch.return_value = [
            "Context 1",
            "Context 2",
        ]
        
        # Mock embeddings
        with patch.object(bcc_module, 'embed_texts') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            
            chunks = [
                {
                    "chunk_id": f"chunk-{i}",
                    "version_id": "version-1",
                    "text_raw": f"Text {i}",
                    "document_title": "Test Document",
                    "section_title": "Section 1" if i == 0 else None,
                }
                for i in range(2)
            ]
            
            success, fail = process_batch(
                mock_conn, chunks, mock_generator,
                embedding_client=Mock(),
                embedding_model="text-embedding-3-small",
            )
            
            # Verify context generator was called with correct data
            call_args = mock_generator.generate_context_batch.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0]["chunk_text"] == "Text 0"
            assert call_args[0]["document_title"] == "Test Document"
            assert call_args[0]["section_title"] == "Section 1"
            
            # Verify embed_texts was called
            mock_embed.assert_called_once()
            embed_call_args = mock_embed.call_args[0][0]
            assert len(embed_call_args) == 2
            # Should be contextualized text
            assert "Context 1" in embed_call_args[0]
            assert "Text 0" in embed_call_args[0]

    def test_resume_capability(self):
        """Test that already processed chunks are skipped."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Simulate database returning only unprocessed chunks
        mock_cur.fetchall.return_value = [
            ("chunk-3", "version-1", "Text 3", 2, "Doc", 50),
        ]
        mock_cur.fetchone.return_value = ("prev-chunk",)
        
        chunks = fetch_batch(mock_conn)
        
        # Check that the fetch_batch query includes a LEFT JOIN to check for existing contextual_chunks
        # The first call should be the main fetch query
        first_call_args = mock_cur.execute.call_args_list[0][0][0]
        assert "LEFT JOIN contextual_chunks" in first_call_args
        assert "cc.chunk_id IS NULL" in first_call_args

    def test_error_recovery(self):
        """Test error handling and recovery."""
        mock_conn = Mock()
        mock_generator = Mock()
        
        # Simulate failure
        mock_generator.generate_context_batch.side_effect = Exception("LLM API Error")
        
        chunks = [
            {"chunk_id": "chunk-1", "version_id": "v1", "text_raw": "Text", "document_title": "Doc"},
        ]
        
        success, fail = process_batch(mock_conn, chunks, mock_generator)
        
        # Should rollback on failure
        mock_conn.rollback.assert_called_once()
        # Should mark all as failed
        assert fail == 1
        assert success == 0
