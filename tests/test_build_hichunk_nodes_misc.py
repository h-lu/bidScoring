"""Tests for build_hichunk_nodes script."""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_hichunk_nodes import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    fetch_pending_versions,
    format_duration,
    get_chunk_mapping,
    get_stats,
    insert_hierarchical_nodes,
    process_version,
    reset_hierarchical_nodes,
)

# Also need to mock the imports in the module
import build_hichunk_nodes as bhn_module  # noqa: E402
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

    @patch.object(bhn_module, "get_chunk_mapping")
    @patch.object(bhn_module, "insert_hierarchical_nodes")
    def test_batch_processing_flow(self, mock_insert, mock_get_mapping):
        """Test complete batch processing flow."""
        mock_conn = Mock()
        mock_builder = Mock()

        from bid_scoring.hichunk import HiChunkNode

        # Create a simple hierarchy
        root = HiChunkNode(level=3, node_type="document", content="Doc", node_id="root")
        section = HiChunkNode(
            level=2,
            node_type="section",
            content="Section",
            node_id="sec",
            parent_id="root",
        )
        para = HiChunkNode(
            level=1,
            node_type="paragraph",
            content="Paragraph",
            node_id="para",
            parent_id="sec",
        )
        leaf = HiChunkNode(
            level=0,
            node_type="sentence",
            content="Sentence.",
            node_id="leaf",
            parent_id="para",
            metadata={"source_index": 0},
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

        success, fail = process_version(
            mock_conn, version_data, mock_builder, show_detail=False
        )

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
        mock_cur.fetchall.side_effect = [
            [("version-3", "doc-3", "Doc 3")],  # 版本查询
            [
                (
                    "chunk-1",
                    0,
                    1,
                    None,
                    "text",
                    "Content",
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            ],
        ]

        _ = fetch_pending_versions(mock_conn)

        # Check that the fetch_pending_versions query includes a LEFT JOIN to check for existing nodes
        first_call_args = mock_cur.execute.call_args_list[0][0][0]
        assert "LEFT JOIN hierarchical_nodes" in first_call_args
        assert "hn.version_id IS NULL" in first_call_args

    @patch.object(bhn_module, "insert_hierarchical_nodes")
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

        with patch.object(bhn_module, "get_chunk_mapping", return_value={}):
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
            metadata={"source_index": 0},
        )
        leaf2 = HiChunkNode(
            level=0,
            node_type="sentence",
            content="Second sentence.",
            node_id="leaf-2",
            metadata={"source_index": 1},
        )

        chunk_mapping = {0: "chunk-0", 1: "chunk-1"}

        insert_hierarchical_nodes(mock_conn, "v1", [leaf1, leaf2], chunk_mapping)

        # Get all insert data from multiple calls
        all_insert_data = []
        for call in mock_cur.executemany.call_args_list:
            if len(call[0]) > 1:
                all_insert_data.extend(call[0][1])

        # Verify each leaf is mapped to correct chunk
        assert len(all_insert_data) == 2
        # First leaf maps to chunk-0
        assert all_insert_data[0][7] == "chunk-0"  # start_chunk_id
        # Second leaf maps to chunk-1
        assert all_insert_data[1][7] == "chunk-1"
        assert all_insert_data[0][9].obj["covered_unit_range"] == {"start": 0, "end": 0}
        assert all_insert_data[1][9].obj["covered_unit_range"] == {"start": 1, "end": 1}

    def test_paragraph_node_chunk_mapping(self):
        """Test that paragraph nodes map to chunk ranges."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        from bid_scoring.hichunk import HiChunkNode

        # Create leaf nodes
        leaf1 = HiChunkNode(
            level=0,
            node_type="sentence",
            content="Sentence 1",
            node_id="leaf-1",
            metadata={"source_index": 0},
        )
        leaf2 = HiChunkNode(
            level=0,
            node_type="sentence",
            content="Sentence 2",
            node_id="leaf-2",
            metadata={"source_index": 1},
        )

        # Create paragraph containing both leaves
        para = HiChunkNode(
            level=1,
            node_type="paragraph",
            content="Sentence 1. Sentence 2.",
            node_id="para-1",
            children_ids=["leaf-1", "leaf-2"],
        )

        chunk_mapping = {0: "chunk-0", 1: "chunk-1"}

        insert_hierarchical_nodes(mock_conn, "v1", [leaf1, leaf2, para], chunk_mapping)

        # 获取所有调用的数据（按层级分批插入）
        all_insert_data = []
        for call in mock_cur.executemany.call_args_list:
            if len(call[0]) > 1:
                all_insert_data.extend(call[0][1])

        # Find paragraph in insert data
        para_data = next((d for d in all_insert_data if d[4] == "paragraph"), None)
        assert para_data is not None

        # Paragraph should have start_chunk_id = chunk-0 and end_chunk_id = chunk-1
        assert para_data[7] == "chunk-0"  # start_chunk_id
        assert para_data[8] == "chunk-1"  # end_chunk_id
        assert para_data[9].obj["covered_unit_range"] == {"start": 0, "end": 1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
