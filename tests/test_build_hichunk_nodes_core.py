"""Tests for build_hichunk_nodes script."""

import os
import sys
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from build_hichunk_nodes import (
    fetch_pending_versions,
    get_stats,
    reset_hierarchical_nodes,
)


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

        with patch("builtins.input", return_value="yes"):
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

        with patch("builtins.input", return_value="no"):
            result = reset_hierarchical_nodes(mock_conn, force=False)

        assert result is False
        # DELETE should not be called
        delete_calls = [
            c for c in mock_cur.execute.call_args_list if "DELETE" in str(c)
        ]
        assert len(delete_calls) == 0

    def test_reset_with_version_id(self):
        """Should reset only specified version."""
        mock_conn = Mock()
        mock_cur = Mock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

        mock_cur.fetchone.return_value = (3,)

        with patch("builtins.input", return_value="yes"):
            result = reset_hierarchical_nodes(
                mock_conn, version_id="test-version", force=False
            )

        assert result is True
        # Check version-specific count query
        count_calls = [
            c
            for c in mock_cur.execute.call_args_list
            if "SELECT COUNT(*)" in str(c) and "version_id" in str(c)
        ]
        assert len(count_calls) > 0
        # Check version-specific delete
        delete_calls = [
            c
            for c in mock_cur.execute.call_args_list
            if "DELETE" in str(c) and "version_id" in str(c)
        ]
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
        delete_calls = [
            c for c in mock_cur.execute.call_args_list if "DELETE" in str(c)
        ]
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
            (500,),  # total_nodes
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

        mock_cur.fetchone.side_effect = [(50, 20, 28), (200,)]

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

        # Mock version data - 第一个查询返回版本信息
        mock_cur.fetchall.side_effect = [
            [("version-1", "doc-1", "Test Document")],  # 版本查询
            [  # chunks 查询 - 15列数据
                (
                    "chunk-1",
                    0,
                    1,
                    "[0,0,100,100]",
                    "text",
                    "Test content",
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        ]

        versions = fetch_pending_versions(mock_conn, batch_size=5)

        assert len(versions) == 1
        assert versions[0]["version_id"] == "version-1"
        assert versions[0]["document_title"] == "Test Document"
        assert len(versions[0]["content_list"]) == 1

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

        mock_cur.fetchall.side_effect = [
            [("version-abc", "doc-abc", "Doc")],  # 版本查询
            [  # chunks 查询
                (
                    "chunk-1",
                    0,
                    1,
                    None,
                    "text",
                    "Test",
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        ]

        versions = fetch_pending_versions(mock_conn, version_id="version-abc")

        # Check version_id was used in query
        call_args = mock_cur.execute.call_args
        assert "version-abc" in str(call_args)
        assert len(versions) == 1
