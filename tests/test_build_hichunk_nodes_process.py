"""Tests for build_hichunk_nodes script."""

import os
import sys
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_hichunk_nodes import (  # noqa: E402
    get_chunk_mapping,
    insert_hierarchical_nodes,
    process_version,
)

# Also need to mock the imports in the module
import build_hichunk_nodes as bhn_module  # noqa: E402


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
            HiChunkNode(
                level=2,
                node_type="section",
                content="Section",
                parent_id="doc-1",
                node_id="sec-1",
            ),
        ]

        chunk_mapping = {}

        success, fail = insert_hierarchical_nodes(
            mock_conn, "version-1", nodes, chunk_mapping
        )

        assert success == 2
        assert fail == 0
        mock_conn.commit.assert_called_once()
        # 按层级插入会调用多次 executemany
        assert mock_cur.executemany.call_count >= 1

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
            metadata={"source_index": 0},
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
        assert insert_data[0][9].obj["covered_unit_range"] == {"start": 0, "end": 0}

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

        success, fail = insert_hierarchical_nodes(mock_conn, "version-1", nodes, {})

        assert success == 0
        assert fail == 1
        mock_conn.rollback.assert_called_once()


class TestProcessVersion:
    """Test version processing."""

    @patch.object(bhn_module, "get_chunk_mapping")
    @patch.object(bhn_module, "insert_hierarchical_nodes")
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

    @patch.object(bhn_module, "get_chunk_mapping")
    @patch.object(bhn_module, "insert_hierarchical_nodes")
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
