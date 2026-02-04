"""Tests for RAPTOR tree builder."""

import pytest
from unittest.mock import MagicMock, patch

from bid_scoring.raptor import RAPTORBuilder, RAPTORNode


class TestRAPTORNode:
    """Test RAPTORNode dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = RAPTORNode(
            level=0,
            node_type="leaf",
            content="Test content",
            metadata={"key": "value"},
        )
        assert node.level == 0
        assert node.node_type == "leaf"
        assert node.content == "Test content"
        assert node.metadata["key"] == "value"
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.embedding is None

    def test_node_id_generation(self):
        """Test that node IDs are auto-generated."""
        node1 = RAPTORNode()
        node2 = RAPTORNode()
        assert node1.node_id != node2.node_id
        assert len(node1.node_id) == 36  # UUID length

    def test_invalid_level(self):
        """Test that invalid level raises error."""
        with pytest.raises(ValueError, match="Level must be >= 0"):
            RAPTORNode(level=-1, node_type="leaf")

    def test_invalid_node_type(self):
        """Test that invalid node_type raises error."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            RAPTORNode(level=0, node_type="invalid")

    def test_valid_node_types(self):
        """Test valid node types."""
        leaf = RAPTORNode(node_type="leaf")
        summary = RAPTORNode(node_type="summary")
        assert leaf.node_type == "leaf"
        assert summary.node_type == "summary"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        node = RAPTORNode(
            level=1,
            node_type="summary",
            content="Summary text",
            parent_id="parent-uuid",
            children_ids=["child-uuid"],
            embedding=[0.1, 0.2, 0.3],
            metadata={"cluster_size": 3},
        )
        d = node.to_dict()
        assert d["level"] == 1
        assert d["node_type"] == "summary"
        assert d["content"] == "Summary text"
        assert d["parent_id"] == "parent-uuid"
        assert d["children_ids"] == ["child-uuid"]
        assert d["embedding"] == [0.1, 0.2, 0.3]
        assert d["metadata"] == {"cluster_size": 3}

    def test_add_child(self):
        """Test adding child nodes."""
        node = RAPTORNode()
        node.add_child("child1")
        node.add_child("child2")
        node.add_child("child1")  # Duplicate should not be added
        assert node.children_ids == ["child1", "child2"]

    def test_set_parent(self):
        """Test setting parent."""
        node = RAPTORNode()
        node.set_parent("parent-uuid")
        assert node.parent_id == "parent-uuid"


class TestRAPTORBuilderBasic:
    """Test basic RAPTORBuilder functionality."""

    def test_builder_initialization(self):
        """Test builder initialization with default parameters."""
        builder = RAPTORBuilder()
        assert builder.max_levels == 5
        assert builder.cluster_size == 10
        assert builder.min_cluster_size == 2
        assert builder.summary_max_tokens == 512
        assert builder.nodes == []

    def test_builder_initialization_with_params(self):
        """Test builder initialization with custom parameters."""
        builder = RAPTORBuilder(
            max_levels=3,
            cluster_size=5,
            min_cluster_size=3,
            summary_max_tokens=256,
        )
        assert builder.max_levels == 3
        assert builder.cluster_size == 5
        assert builder.min_cluster_size == 3
        assert builder.summary_max_tokens == 256

    def test_builder_with_mock_llm(self):
        """Test builder with mock LLM client."""
        mock_llm = MagicMock()
        builder = RAPTORBuilder(llm_client=mock_llm)
        assert builder._llm_client is mock_llm


class TestRAPTORBuilderValidation:
    """Test input validation for build_tree."""

    def test_empty_chunks(self):
        """Test that empty chunks raises error."""
        builder = RAPTORBuilder()
        with pytest.raises(ValueError, match="At least 2 chunks required"):
            builder.build_tree([])

    def test_single_chunk(self):
        """Test that single chunk raises error."""
        builder = RAPTORBuilder()
        with pytest.raises(ValueError, match="At least 2 chunks required"):
            builder.build_tree(["only one chunk"])

    def test_invalid_input_type(self):
        """Test that invalid input type raises error."""
        builder = RAPTORBuilder()
        with pytest.raises(ValueError, match="chunks must be a list"):
            builder.build_tree("not a list")

    def test_only_empty_chunks(self):
        """Test that only empty/whitespace chunks raises error."""
        builder = RAPTORBuilder()
        with pytest.raises(ValueError, match="At least 2 non-empty chunks required"):
            builder.build_tree(["", "   ", ""])

    def test_mixed_empty_and_valid_chunks(self):
        """Test that empty chunks are filtered out."""
        builder = RAPTORBuilder()
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            # Should work with 2 valid chunks after filtering
            nodes = builder.build_tree(["valid chunk 1", "", "valid chunk 2", "   "])
            
            # Check that only valid chunks were processed
            leaf_nodes = builder.get_leaf_nodes()
            assert len(leaf_nodes) == 2


class TestRAPTORBuilderTreeBuilding:
    """Test tree building functionality."""

    def test_build_tree_with_two_chunks(self):
        """Test building tree with minimum chunks."""
        builder = RAPTORBuilder()
        chunks = ["First chunk of text", "Second chunk of text"]
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
            
            nodes = builder.build_tree(chunks)
            
            # Should have at least 2 leaf nodes
            leaf_nodes = builder.get_leaf_nodes()
            assert len(leaf_nodes) == 2
            
            # Leaf nodes should have level 0
            for node in leaf_nodes:
                assert node.level == 0
                assert node.node_type == "leaf"

    def test_leaf_node_content(self):
        """Test that leaf nodes preserve original content."""
        builder = RAPTORBuilder()
        chunks = ["Content A", "Content B", "Content C"]
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1], [0.2], [0.3]]
            
            builder.build_tree(chunks)
            leaf_nodes = builder.get_leaf_nodes()
            
            contents = [n.content for n in leaf_nodes]
            assert "Content A" in contents
            assert "Content B" in contents
            assert "Content C" in contents

    def test_parent_child_relationships(self):
        """Test that parent-child relationships are correctly set."""
        builder = RAPTORBuilder()
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [
                [0.1, 0.1], [0.1, 0.2],  # Cluster 1
                [0.9, 0.9], [0.9, 0.8],  # Cluster 2
            ]
            
            nodes = builder.build_tree(chunks)
            leaf_nodes = builder.get_leaf_nodes()
            
            # Check that leaf nodes have parents
            parents = set()
            for leaf in leaf_nodes:
                if leaf.parent_id:
                    parents.add(leaf.parent_id)
            
            # Should have some parent nodes
            assert len(parents) > 0


class TestRAPTORBuilderClustering:
    """Test clustering functionality."""

    def test_calculate_num_clusters(self):
        """Test cluster number calculation."""
        builder = RAPTORBuilder(cluster_size=10)
        
        # Less than min_cluster_size
        assert builder._calculate_num_clusters(2) == 1
        
        # More than min_cluster_size
        assert builder._calculate_num_clusters(20) == 2  # 20 // 10 = 2
        assert builder._calculate_num_clusters(25) == 2  # 25 // 10 = 2
        assert builder._calculate_num_clusters(50) == 5  # 50 // 10 = 5
        
        # Should not exceed node count
        assert builder._calculate_num_clusters(5) <= 5

    def test_cluster_nodes_with_embeddings(self):
        """Test clustering nodes with embeddings."""
        builder = RAPTORBuilder(cluster_size=2)
        
        nodes = [
            RAPTORNode(content="A", embedding=[0.1, 0.1]),
            RAPTORNode(content="B", embedding=[0.1, 0.2]),
            RAPTORNode(content="C", embedding=[0.9, 0.9]),
            RAPTORNode(content="D", embedding=[0.9, 0.8]),
        ]
        
        clusters = builder._cluster_nodes(nodes)
        
        # Should create multiple clusters
        assert len(clusters) >= 1
        
        # All nodes should be in some cluster
        total_nodes = sum(len(c) for c in clusters)
        assert total_nodes == 4

    def test_cluster_nodes_without_embeddings(self):
        """Test clustering nodes without embeddings."""
        builder = RAPTORBuilder()
        
        nodes = [
            RAPTORNode(content="A"),
            RAPTORNode(content="B"),
        ]
        
        clusters = builder._cluster_nodes(nodes)
        
        # Should return single cluster containing all nodes
        assert len(clusters) == 1
        assert len(clusters[0]) == 2  # All nodes included

    def test_cluster_single_node(self):
        """Test clustering with single node."""
        builder = RAPTORBuilder()
        
        nodes = [RAPTORNode(content="Single", embedding=[0.1, 0.2])]
        
        clusters = builder._cluster_nodes(nodes)
        
        # Single node should be in one cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_cluster_empty_list(self):
        """Test clustering empty list."""
        builder = RAPTORBuilder()
        
        clusters = builder._cluster_nodes([])
        
        assert clusters == []


class TestRAPTORBuilderSummarization:
    """Test summarization functionality."""

    def test_summarize_cluster(self):
        """Test cluster summarization."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Generated summary"
        
        builder = RAPTORBuilder(llm_client=mock_llm)
        
        nodes = [
            RAPTORNode(content="Text about topic A"),
            RAPTORNode(content="Text about topic B"),
        ]
        
        summary = builder._summarize_cluster(nodes, level=1)
        
        assert summary == "Generated summary"
        mock_llm.complete.assert_called_once()

    def test_summarize_empty_cluster(self):
        """Test summarizing empty cluster."""
        builder = RAPTORBuilder()
        
        summary = builder._summarize_cluster([], level=1)
        
        assert summary == ""

    def test_summarize_with_empty_texts(self):
        """Test summarizing cluster with empty/whitespace texts."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Summary"
        
        builder = RAPTORBuilder(llm_client=mock_llm)
        
        nodes = [
            RAPTORNode(content="   "),
            RAPTORNode(content=""),
        ]
        
        summary = builder._summarize_cluster(nodes, level=1)
        
        assert summary == ""

    def test_summarize_with_llm_error(self):
        """Test summarization when LLM fails."""
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("LLM Error")
        
        builder = RAPTORBuilder(llm_client=mock_llm)
        
        nodes = [
            RAPTORNode(content="Long text that needs summarization..."),
        ]
        
        # Should not raise, returns fallback
        summary = builder._summarize_cluster(nodes, level=1)
        
        assert "Long text" in summary

    def test_create_summary_node(self):
        """Test creating summary node from cluster."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "Cluster summary"
        
        builder = RAPTORBuilder(llm_client=mock_llm)
        
        child_nodes = [
            RAPTORNode(content="Child 1"),
            RAPTORNode(content="Child 2"),
        ]
        
        summary_node = builder._create_summary_node(child_nodes, level=1)
        
        assert summary_node.level == 1
        assert summary_node.node_type == "summary"
        assert summary_node.content == "Cluster summary"
        assert len(summary_node.children_ids) == 2
        assert summary_node.metadata["cluster_size"] == 2
        
        # Check parent relationships
        for child in child_nodes:
            assert child.parent_id == summary_node.node_id


class TestRAPTORBuilderHelpers:
    """Test builder helper methods."""

    def test_get_nodes_by_level(self):
        """Test getting nodes by level."""
        builder = RAPTORBuilder()
        
        # Manually add nodes
        builder.nodes = [
            RAPTORNode(level=0),
            RAPTORNode(level=0),
            RAPTORNode(level=1),
            RAPTORNode(level=2),
        ]
        
        assert len(builder.get_nodes_by_level(0)) == 2
        assert len(builder.get_nodes_by_level(1)) == 1
        assert len(builder.get_nodes_by_level(2)) == 1
        assert len(builder.get_nodes_by_level(3)) == 0

    def test_get_leaf_nodes(self):
        """Test getting leaf nodes."""
        builder = RAPTORBuilder()
        
        builder.nodes = [
            RAPTORNode(node_type="leaf"),
            RAPTORNode(node_type="leaf"),
            RAPTORNode(node_type="summary"),
        ]
        
        leaves = builder.get_leaf_nodes()
        assert len(leaves) == 2

    def test_get_root_node(self):
        """Test getting root node."""
        builder = RAPTORBuilder()
        
        builder.nodes = [
            RAPTORNode(level=0),
            RAPTORNode(level=1),
            RAPTORNode(level=2),  # Highest level
        ]
        
        root = builder.get_root_node()
        assert root is not None
        assert root.level == 2

    def test_get_root_node_empty(self):
        """Test getting root node when no nodes."""
        builder = RAPTORBuilder()
        
        root = builder.get_root_node()
        assert root is None

    def test_get_tree_structure(self):
        """Test getting tree structure."""
        builder = RAPTORBuilder()
        
        root = RAPTORNode(level=1, node_type="summary", content="Root")
        child = RAPTORNode(level=0, node_type="leaf", content="Child")
        root.add_child(child.node_id)
        child.set_parent(root.node_id)
        
        builder.nodes = [root, child]
        builder._node_map = {root.node_id: root, child.node_id: child}
        
        tree = builder.get_tree_structure()
        
        assert tree["type"] == "summary"
        assert tree["level"] == 1
        assert tree["content"] == "Root"
        assert len(tree["children"]) == 1

    def test_get_tree_stats(self):
        """Test getting tree statistics."""
        builder = RAPTORBuilder()
        
        builder.nodes = [
            RAPTORNode(level=0, node_type="leaf"),
            RAPTORNode(level=0, node_type="leaf"),
            RAPTORNode(level=1, node_type="summary"),
        ]
        
        stats = builder.get_tree_stats()
        
        assert stats["total_nodes"] == 3
        assert stats["max_level"] == 1
        assert stats["leaf_count"] == 2
        assert stats["summary_count"] == 1
        assert stats["level_distribution"] == {0: 2, 1: 1}

    def test_get_tree_stats_empty(self):
        """Test getting stats for empty tree."""
        builder = RAPTORBuilder()
        
        stats = builder.get_tree_stats()
        
        assert stats["total_nodes"] == 0
        assert stats["max_level"] == 0
        assert stats["leaf_count"] == 0
        assert stats["summary_count"] == 0


class TestRAPTORBuilderEdgeCases:
    """Test edge cases and error handling."""

    def test_whitespace_only_chunks(self):
        """Test handling of whitespace-only chunks."""
        builder = RAPTORBuilder()
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [[0.1], [0.2]]
            
            chunks = ["Real content 1", "   ", "Real content 2", ""]
            nodes = builder.build_tree(chunks)
            
            # Only valid chunks should be included
            leaf_nodes = builder.get_leaf_nodes()
            contents = [n.content for n in leaf_nodes]
            assert "Real content 1" in contents
            assert "Real content 2" in contents
            assert "   " not in contents
            assert "" not in contents

    def test_many_chunks(self):
        """Test with many chunks."""
        builder = RAPTORBuilder(cluster_size=5)
        
        # Create 20 chunks
        chunks = [f"Chunk {i}" for i in range(20)]
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            # Create embeddings that should form clusters
            embeddings = [[i / 20.0, (20 - i) / 20.0] for i in range(20)]
            mock_embed.return_value = embeddings
            
            nodes = builder.build_tree(chunks)
            
            # Should have leaf nodes and some summary nodes
            leaf_nodes = builder.get_leaf_nodes()
            assert len(leaf_nodes) == 20
            
            # Total nodes should be more than just leaves
            assert len(nodes) > 20

    def test_embedding_generation_failure(self):
        """Test handling of embedding generation failure."""
        builder = RAPTORBuilder()
        
        with patch('bid_scoring.raptor.embed_texts') as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")
            
            chunks = ["Chunk 1", "Chunk 2"]
            nodes = builder.build_tree(chunks)
            
            # Should still create leaf nodes even without embeddings
            leaf_nodes = builder.get_leaf_nodes()
            assert len(leaf_nodes) == 2


class TestRAPTORIntegration:
    """Integration tests for RAPTOR builder."""

    def test_full_tree_construction(self):
        """Test complete tree construction flow."""
        builder = RAPTORBuilder(max_levels=3, cluster_size=3)
        
        chunks = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn canine leaps across a sleepy hound.",
            "Swift russet vixen vaults over the drowsy canine.",
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Neural networks are inspired by biological neurons.",
        ]
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            # First call for leaves (6 embeddings)
            # Second call for summaries
            def side_effect(texts, **kwargs):
                if len(texts) == 6:
                    # Leaves - create 2 distinct groups
                    return [
                        [0.1, 0.1], [0.1, 0.15], [0.12, 0.1],  # Group 1 (animals)
                        [0.9, 0.9], [0.85, 0.9], [0.9, 0.85],  # Group 2 (AI)
                    ]
                elif len(texts) == 2:
                    # Level 1 summaries
                    return [[0.15, 0.15], [0.88, 0.88]]
                else:
                    return [[0.5, 0.5]] * len(texts)
            
            mock_embed.side_effect = side_effect
            
            with patch.object(builder, '_summarize_cluster') as mock_summarize:
                mock_summarize.side_effect = [
                    "Summary about animals",
                    "Summary about AI/ML",
                    "Final document summary",
                ]
                
                nodes = builder.build_tree(chunks)
                
                # Verify structure
                leaf_nodes = builder.get_leaf_nodes()
                assert len(leaf_nodes) == 6
                
                # Check tree stats
                stats = builder.get_tree_stats()
                assert stats["total_nodes"] == len(nodes)
                assert stats["leaf_count"] == 6

    def test_tree_with_single_final_root(self):
        """Test that tree converges to single root."""
        builder = RAPTORBuilder(max_levels=5, cluster_size=2)
        
        # 4 chunks that should form a tree
        chunks = ["A", "B", "C", "D"]
        
        with patch.object(builder, '_generate_embeddings') as mock_embed:
            mock_embed.return_value = [
                [0.1, 0.1], [0.15, 0.15],
                [0.9, 0.9], [0.85, 0.85],
            ]
            
            nodes = builder.build_tree(chunks)
            
            root = builder.get_root_node()
            assert root is not None
            
            # Root should be at highest level
            max_level = max(n.level for n in nodes)
            assert root.level == max_level


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
