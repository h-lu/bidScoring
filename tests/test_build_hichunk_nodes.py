"""Tests for build_hichunk_nodes script."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from build_hichunk_nodes import (
    get_stats,
    fetch_pending_versions,
    get_chunk_mapping,
    insert_hierarchical_nodes,
    process_version,
    format_duration,
    reset_hierarchical_nodes,
    DEFAULT_BATCH_SIZE,
)

# Also need to mock the imports in the module
import build_hichunk_nodes as bhn_module


class TestResetHierarchicalNodes:
    """Test reset functionality."""

    def test_reset_all_versions_with_confirmation(self):
        """Should reset all records after confirmation."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock count query returns 5 records
        mock_cur.fetchone.return_value = (5,)
        
        with patch('builtins.input', return_value='yes'):
            result = reset_hierarchical_nodes(mock_conn, force=False)
        
        assert result is True
        mock_cur.execute.assert_any_call("SELECT COUNT(*) FROM hierarchical_nodes")
        mock_cur.execute.assert_any_call("DELETE FROM hierarchical_nodes")
        mock_conn.commit.assert_called_once()

    def test_reset_all_versions_cancelled(self):
        """Should not reset when user cancels."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchone.return_value = (5,)
        
        with patch('builtins.input', return_value='no'):
            result = reset_hierarchical_nodes(mock_conn, force=False)
        
        assert result is False
        # DELETE should not be called
        delete_calls = [call for call in mock_cur.execute.call_args_list if 'DELETE' in str(call)]
        assert len(delete_calls) == 0

    def test_reset_with_version_id(self):
        """Should reset only specified version."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchone.return_value = (3,)
        
        with patch('builtins.input', return_value='yes'):
            result = reset_hierarchical_nodes(mock_conn, version_id="test-version", force=False)
        
        assert result is True
        # Check version-specific count query
        count_calls = [call for call in mock_cur.execute.call_args_list 
                      if 'SELECT COUNT(*)' in str(call) and 'version_id' in str(call)]
        assert len(count_calls) > 0
        # Check version-specific delete
        delete_calls = [call for call in mock_cur.execute.call_args_list 
                       if 'DELETE' in str(call) and 'version_id' in str(call)]
        assert len(delete_calls) > 0

    def test_reset_force_skips_confirmation(self):
        """Should reset without confirmation when force=True."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchone.return_value = (10,)
        
        result = reset_hierarchical_nodes(mock_conn, force=True)
        
        assert result is True
        mock_cur.execute.assert_any_call("DELETE FROM hierarchical_nodes")
        mock_conn.commit.assert_called_once()

    def test_reset_empty_table(self):
        """Should handle empty table gracefully."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchone.return_value = (0,)
        
        result = reset_hierarchical_nodes(mock_conn, force=False)
        
        assert result is True
        # DELETE should not be called for empty table
        delete_calls = [call for call in mock_cur.execute.call_args_list if 'DELETE' in str(call)]
        assert len(delete_calls) == 0


class TestGetStats:
    """Test statistics retrieval."""

    def test_get_stats_basic(self):
        """Should return correct statistics."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock query results
        mock_cur.fetchone.side_effect = [
            (100, 30, 65),  # total_versions, processed, to_process
            (500,)  # total_nodes
        ]
        
        stats = get_stats(mock_conn)
        
        assert stats["total_versions"] == 100
        assert stats["processed"] == 30
        assert stats["to_process"] == 65
        assert stats["total_nodes"] == 500

    def test_get_stats_with_version_id(self):
        """Should filter by version_id when provided."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchone.side_effect = [
            (50, 20, 28),
            (200,)
        ]
        
        stats = get_stats(mock_conn, version_id="test-version-id")
        
        # Check that version_id was used in query
        call_args = mock_cur.execute.call_args
        assert "test-version-id" in str(call_args)
        assert stats["total_versions"] == 50


class TestFetchPendingVersions:
    """Test batch fetching from database."""

    def test_fetch_pending_versions_basic(self):
        """Should fetch versions that need processing."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Mock version data
        content_list = [{"type": "text", "text": "Test content", "page_idx": 1}]
        mock_cur.fetchall.return_value = [
            ("version-1", "doc-1", "Test Document", content_list),
            ("version-2", "doc-2", "Another Doc", content_list),
        ]
        
        versions = fetch_pending_versions(mock_conn, batch_size=5)
        
        assert len(versions) == 2
        assert versions[0]["version_id"] == "version-1"
        assert versions[0]["document_title"] == "Test Document"
        assert versions[0]["content_list"] == content_list

    def test_fetch_pending_versions_empty(self):
        """Should return empty list when no versions to process."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchall.return_value = []
        
        versions = fetch_pending_versions(mock_conn)
        
        assert versions == []

    def test_fetch_pending_versions_with_version_filter(self):
        """Should filter by version_id."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        content_list = [{"type": "text", "text": "Test", "page_idx": 1}]
        mock_cur.fetchall.return_value = [
            ("version-abc", "doc-abc", "Doc", content_list),
        ]
        
        versions = fetch_pending_versions(mock_conn, version_id="version-abc")
        
        # Check version_id was used in query
        call_args = mock_cur.execute.call_args
        assert "version-abc" in str(call_args)
        assert len(versions) == 1


class TestGetChunkMapping:
    """Test chunk mapping retrieval."""

    def test_get_chunk_mapping_basic(self):
        """Should return chunk index to ID mapping."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchall.return_value = [
            ("chunk-id-0", 0),
            ("chunk-id-1", 1),
            ("chunk-id-2", 2),
        ]
        
        mapping = get_chunk_mapping(mock_conn, "version-1")
        
        assert len(mapping) == 3
        assert mapping[0] == "chunk-id-0"
        assert mapping[1] == "chunk-id-1"
        assert mapping[2] == "chunk-id-2"

    def test_get_chunk_mapping_empty(self):
        """Should return empty dict when no chunks."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        mock_cur.fetchall.return_value = []
        
        mapping = get_chunk_mapping(mock_conn, "version-1")
        
        assert mapping == {}


class TestInsertHierarchicalNodes:
    """Test node insertion."""

    def test_insert_hierarchical_nodes_success(self):
        """Should successfully insert nodes."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Create mock nodes
        from bid_scoring.hichunk import HiChunkNode
        nodes = [
            HiChunkNode(level=3, node_type="document", content="Doc", node_id="doc-1"),
            HiChunkNode(level=2, node_type="section", content="Section", parent_id="doc-1", node_id="sec-1"),
        ]
        
        chunk_mapping = {}
        
        success, fail = insert_hierarchical_nodes(
            mock_conn, "version-1", nodes, chunk_mapping
        )
        
        assert success == 2
        assert fail == 0
        mock_conn.commit.assert_called_once()
        mock_cur.executemany.assert_called_once()

    def test_insert_hierarchical_nodes_empty(self):
        """Should handle empty node list."""
        mock_conn = Mock()
        
        success, fail = insert_hierarchical_nodes(mock_conn, "version-1", [], {})
        
        assert success == 0
        assert fail == 0

    def test_insert_hierarchical_nodes_with_chunk_mapping(self):
        """Should correctly map leaf nodes to chunks."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        from bid_scoring.hichunk import HiChunkNode
        
        # Create a leaf node with source_index
        leaf = HiChunkNode(
            level=0, 
            node_type="sentence", 
            content="Test sentence",
            node_id="leaf-1",
            metadata={"source_index": 0}
        )
        
        chunk_mapping = {0: "chunk-id-0"}
        
        success, fail = insert_hierarchical_nodes(
            mock_conn, "version-1", [leaf], chunk_mapping
        )
        
        assert success == 1
        assert fail == 0
        
        # Verify insert was called with correct chunk IDs
        call_args = mock_cur.executemany.call_args
        insert_data = call_args[0][1]
        assert len(insert_data) == 1
        # start_chunk_id and end_chunk_id should be set
        assert insert_data[0][7] == "chunk-id-0"  # start_chunk_id
        assert insert_data[0][8] == "chunk-id-0"  # end_chunk_id

    def test_insert_hierarchical_nodes_failure(self):
        """Should handle insertion failure."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Simulate failure
        mock_cur.executemany.side_effect = Exception("Database error")
        
        from bid_scoring.hichunk import HiChunkNode
        nodes = [HiChunkNode(level=3, node_type="document", content="Doc")]
        
        success, fail = insert_hierarchical_nodes(
            mock_conn, "version-1", nodes, {}
        )
        
        assert success == 0
        assert fail == 1
        mock_conn.rollback.assert_called_once()


class TestProcessVersion:
    """Test version processing."""

    @patch.object(bhn_module, 'get_chunk_mapping')
    @patch.object(bhn_module, 'insert_hierarchical_nodes')
    def test_process_version_success(self, mock_insert, mock_get_mapping):
        """Should successfully process a version."""
        mock_conn = Mock()
        mock_builder = Mock()
        
        # Mock builder output
        from bid_scoring.hichunk import HiChunkNode
        mock_nodes = [
            HiChunkNode(level=3, node_type="document", content="Test Doc"),
            HiChunkNode(level=2, node_type="section", content="Section 1"),
        ]
        mock_builder.build_hierarchy.return_value = mock_nodes
        
        mock_get_mapping.return_value = {0: "chunk-0", 1: "chunk-1"}
        mock_insert.return_value = (2, 0)
        
        version_data = {
            "version_id": "version-1",
            "doc_id": "doc-1",
            "document_title": "Test Document",
            "content_list": [{"type": "text", "text": "Content", "page_idx": 1}],
        }
        
        success, fail = process_version(mock_conn, version_data, mock_builder)
        
        assert success == 2
        assert fail == 0
        mock_builder.build_hierarchy.assert_called_once()
        mock_get_mapping.assert_called_once_with(mock_conn, "version-1")

    @patch.object(bhn_module, 'get_chunk_mapping')
    @patch.object(bhn_module, 'insert_hierarchical_nodes')
    def test_process_version_empty_content(self, mock_insert, mock_get_mapping):
        """Should handle empty content list."""
        mock_conn = Mock()
        mock_builder = Mock()
        
        # Return empty doc for empty content
        from bid_scoring.hichunk import HiChunkNode
        mock_builder.build_hierarchy.return_value = [
            HiChunkNode(level=3, node_type="document", content="Empty Doc")
        ]
        
        mock_get_mapping.return_value = {}
        mock_insert.return_value = (1, 0)
        
        version_data = {
            "version_id": "version-1",
            "doc_id": "doc-1",
            "document_title": "Empty",
            "content_list": [],
        }
        
        success, fail = process_version(mock_conn, version_data, mock_builder)
        
        assert success == 1  # Just the root node

    def test_process_version_failure(self):
        """Should handle processing failure gracefully."""
        mock_conn = Mock()
        mock_builder = Mock()
        
        # Simulate build failure
        mock_builder.build_hierarchy.side_effect = Exception("Build error")
        
        version_data = {
            "version_id": "version-1",
            "doc_id": "doc-1",
            "document_title": "Test",
            "content_list": [{"type": "text", "text": "Content"}],
        }
        
        success, fail = process_version(mock_conn, version_data, mock_builder)
        
        assert success == 0
        assert fail == 0  # No nodes to fail
        mock_conn.rollback.assert_called_once()


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
        assert DEFAULT_BATCH_SIZE == 10
        assert DEFAULT_BATCH_SIZE <= 50  # Should be reasonable for document processing


class TestIntegrationPatterns:
    """Integration-style tests for the full flow."""

    @patch.object(bhn_module, 'get_chunk_mapping')
    @patch.object(bhn_module, 'insert_hierarchical_nodes')
    def test_batch_processing_flow(self, mock_insert, mock_get_mapping):
        """Test complete batch processing flow."""
        mock_conn = Mock()
        mock_builder = Mock()
        
        from bid_scoring.hichunk import HiChunkNode
        
        # Create a simple hierarchy
        root = HiChunkNode(level=3, node_type="document", content="Doc", node_id="root")
        section = HiChunkNode(level=2, node_type="section", content="Section", node_id="sec", parent_id="root")
        para = HiChunkNode(level=1, node_type="paragraph", content="Paragraph", node_id="para", parent_id="sec")
        leaf = HiChunkNode(
            level=0, 
            node_type="sentence", 
            content="Sentence.", 
            node_id="leaf", 
            parent_id="para",
            metadata={"source_index": 0}
        )
        
        mock_nodes = [root, section, para, leaf]
        mock_builder.build_hierarchy.return_value = mock_nodes
        mock_get_mapping.return_value = {0: "chunk-0"}
        mock_insert.return_value = (4, 0)
        
        version_data = {
            "version_id": "v1",
            "doc_id": "d1",
            "document_title": "Test",
            "content_list": [{"type": "text", "text": "Sentence.", "page_idx": 1}],
        }
        
        success, fail = process_version(mock_conn, version_data, mock_builder, show_detail=False)
        
        # Verify builder was called with content_list
        call_args = mock_builder.build_hierarchy.call_args
        assert call_args[0][0] == version_data["content_list"]
        assert call_args[0][1] == "Test"
        
        # Verify insert was called
        mock_insert.assert_called_once()

    def test_resume_capability(self):
        """Test that already processed versions are skipped."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        # Simulate database returning only unprocessed versions
        content_list = [{"type": "text", "text": "Content"}]
        mock_cur.fetchall.return_value = [
            ("version-3", "doc-3", "Doc 3", content_list),
        ]
        
        versions = fetch_pending_versions(mock_conn)
        
        # Check that the fetch_pending_versions query includes a LEFT JOIN to check for existing nodes
        first_call_args = mock_cur.execute.call_args_list[0][0][0]
        assert "LEFT JOIN hierarchical_nodes" in first_call_args
        assert "hn.version_id IS NULL" in first_call_args

    @patch.object(bhn_module, 'insert_hierarchical_nodes')
    def test_error_recovery(self, mock_insert):
        """Test error handling and recovery."""
        mock_conn = Mock()
        mock_builder = Mock()
        
        # Simulate success but insert fails
        from bid_scoring.hichunk import HiChunkNode
        mock_builder.build_hierarchy.return_value = [
            HiChunkNode(level=3, node_type="document", content="Doc")
        ]
        mock_insert.return_value = (0, 1)  # 0 success, 1 fail
        
        version_data = {
            "version_id": "v1",
            "doc_id": "d1",
            "document_title": "Test",
            "content_list": [{"type": "text", "text": "Content"}],
        }
        
        with patch.object(bhn_module, 'get_chunk_mapping', return_value={}):
            success, fail = process_version(mock_conn, version_data, mock_builder)
        
        # Should report failure
        assert success == 0
        assert fail == 1


class TestNodeChunkMapping:
    """Test node to chunk mapping logic."""

    def test_leaf_node_chunk_mapping(self):
        """Test that leaf nodes are correctly mapped to chunks."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        from bid_scoring.hichunk import HiChunkNode
        
        # Create leaf nodes with different source_indices
        leaf1 = HiChunkNode(
            level=0,
            node_type="sentence",
            content="First sentence.",
            node_id="leaf-1",
            metadata={"source_index": 0}
        )
        leaf2 = HiChunkNode(
            level=0,
            node_type="sentence",
            content="Second sentence.",
            node_id="leaf-2",
            metadata={"source_index": 1}
        )
        
        chunk_mapping = {0: "chunk-0", 1: "chunk-1"}
        
        insert_hierarchical_nodes(mock_conn, "v1", [leaf1, leaf2], chunk_mapping)
        
        # Get the insert data
        call_args = mock_cur.executemany.call_args
        insert_data = call_args[0][1]
        
        # Verify each leaf is mapped to correct chunk
        assert len(insert_data) == 2
        # First leaf maps to chunk-0
        assert insert_data[0][7] == "chunk-0"  # start_chunk_id
        # Second leaf maps to chunk-1
        assert insert_data[1][7] == "chunk-1"

    def test_paragraph_node_chunk_mapping(self):
        """Test that paragraph nodes map to chunk ranges."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        
        from bid_scoring.hichunk import HiChunkNode
        
        # Create leaf nodes
        leaf1 = HiChunkNode(
            level=0, node_type="sentence", content="Sentence 1",
            node_id="leaf-1", metadata={"source_index": 0}
        )
        leaf2 = HiChunkNode(
            level=0, node_type="sentence", content="Sentence 2",
            node_id="leaf-2", metadata={"source_index": 1}
        )
        
        # Create paragraph containing both leaves
        para = HiChunkNode(
            level=1, node_type="paragraph", content="Sentence 1. Sentence 2.",
            node_id="para-1", children_ids=["leaf-1", "leaf-2"]
        )
        
        chunk_mapping = {0: "chunk-0", 1: "chunk-1"}
        
        insert_hierarchical_nodes(mock_conn, "v1", [leaf1, leaf2, para], chunk_mapping)
        
        call_args = mock_cur.executemany.call_args
        insert_data = call_args[0][1]
        
        # Find paragraph in insert data
        para_data = next((d for d in insert_data if d[4] == "paragraph"), None)
        assert para_data is not None
        
        # Paragraph should have start_chunk_id = chunk-0 and end_chunk_id = chunk-1
        assert para_data[7] == "chunk-0"  # start_chunk_id
        assert para_data[8] == "chunk-1"  # end_chunk_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
