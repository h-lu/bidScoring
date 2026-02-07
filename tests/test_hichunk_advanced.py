"""Tests for HiChunk hierarchical chunking builder (advanced cases)."""

import pytest
from bid_scoring.hichunk import HiChunkBuilder


class TestHiChunkTextExtraction:
    """Test text extraction from various content types."""

    def test_extract_table_text(self):
        """Test text extraction from table."""
        builder = HiChunkBuilder()
        content_list = [
            {
                "type": "table",
                "table_caption": ["Table 1: Sample data"],
                "table_body": "<tr><td>Cell 1</td></tr>",
                "table_footnote": ["Source: Test"],
                "page_idx": 1,
            }
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf_nodes = [n for n in nodes if n.level == 0]
        assert len(leaf_nodes) == 1
        assert "Table 1" in leaf_nodes[0].content
        assert "Cell 1" in leaf_nodes[0].content
        assert "Source: Test" in leaf_nodes[0].content

    def test_extract_image_text(self):
        """Test text extraction from image."""
        builder = HiChunkBuilder()
        content_list = [
            {
                "type": "image",
                "image_caption": ["Figure 1: Sample image"],
                "image_footnote": ["Credit: Author"],
                "page_idx": 1,
            }
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf_nodes = [n for n in nodes if n.level == 0]
        assert len(leaf_nodes) == 1
        assert "Figure 1" in leaf_nodes[0].content
        assert "Credit: Author" in leaf_nodes[0].content

    def test_extract_list_text(self):
        """Test text extraction from list."""
        builder = HiChunkBuilder()
        content_list = [
            {
                "type": "list",
                "list_items": ["First item", "Second item", "Third item"],
                "page_idx": 1,
            }
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf_nodes = [n for n in nodes if n.level == 0]
        assert len(leaf_nodes) == 1
        assert "First item" in leaf_nodes[0].content
        assert "Second item" in leaf_nodes[0].content

    def test_skip_page_number(self):
        """Test that page_number items are skipped."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Content.", "page_idx": 1, "text_level": 0},
            {"type": "page_number", "text": "1", "page_idx": 1},
            {"type": "text", "text": "More content.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf_nodes = [n for n in nodes if n.level == 0]
        # Should only have 2 leaf nodes (page_number skipped)
        assert len(leaf_nodes) == 2


class TestHiChunkParentChildRelationships:
    """Test parent-child relationships in the hierarchy."""

    def test_parent_child_links(self):
        """Test that parent-child relationships are correctly set."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Sentence 1.", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Sentence 2.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        root = [n for n in nodes if n.level == 3][0]
        section = [n for n in nodes if n.level == 2][0]
        para_nodes = [n for n in nodes if n.level == 1]
        leaves = [n for n in nodes if n.level == 0]

        # Both sentences should be in same paragraph
        assert len(para_nodes) == 1
        para = para_nodes[0]

        # Check root -> section
        assert section.node_id in root.children_ids
        assert section.parent_id == root.node_id

        # Check section -> paragraph
        assert para.node_id in section.children_ids
        assert para.parent_id == section.node_id

        # Check paragraph -> leaves
        assert len(leaves) == 2
        for leaf in leaves:
            assert leaf.node_id in para.children_ids
            assert leaf.parent_id == para.node_id


class TestHiChunkMetadata:
    """Test metadata preservation."""

    def test_leaf_node_metadata(self):
        """Test that leaf nodes preserve metadata."""
        builder = HiChunkBuilder()
        content_list = [
            {
                "type": "text",
                "text": "Content.",
                "page_idx": 5,
                "text_level": 2,
                "bbox": [100, 200, 300, 400],
            }
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf = [n for n in nodes if n.level == 0][0]
        assert leaf.metadata["page_idx"] == 5
        assert leaf.metadata["text_level"] == 2
        assert leaf.metadata["bbox"] == [100, 200, 300, 400]
        assert leaf.metadata["element_type"] == "text"

    def test_paragraph_metadata(self):
        """Test paragraph node metadata."""
        builder = HiChunkBuilder()
        # Both sentences on same page - merged into one paragraph
        content_list = [
            {"type": "text", "text": "Sentence 1.", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Sentence 2.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        para = [n for n in nodes if n.level == 1][0]
        assert para.metadata["start_page"] == 1
        assert para.metadata["end_page"] == 1
        assert para.metadata["leaf_count"] == 2

    def test_paragraph_page_split(self):
        """Test that page changes split paragraphs."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Sentence 1.", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Sentence 2.", "page_idx": 2, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        para_nodes = [n for n in nodes if n.level == 1]
        # Page change creates 2 separate paragraphs
        assert len(para_nodes) == 2
        assert para_nodes[0].metadata["start_page"] == 1
        assert para_nodes[0].metadata["end_page"] == 1
        assert para_nodes[1].metadata["start_page"] == 2
        assert para_nodes[1].metadata["end_page"] == 2

    def test_root_metadata(self):
        """Test root node metadata."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Chapter 1", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "Content.", "page_idx": 2, "text_level": 0},
            {"type": "text", "text": "Chapter 2", "page_idx": 5, "text_level": 1},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        root = [n for n in nodes if n.level == 3][0]
        assert root.metadata["document_title"] == "Test Doc"
        assert root.metadata["section_count"] == 2


class TestHiChunkBuilderHelpers:
    """Test builder helper methods."""

    def test_get_nodes_by_level(self):
        """Test getting nodes by level."""
        builder = HiChunkBuilder()
        # Title (heading) and Content are separate paragraphs
        content_list = [
            {"type": "text", "text": "Title", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "Content.", "page_idx": 1, "text_level": 0},
        ]
        builder.build_hierarchy(content_list, "Test Doc")

        # 2 leaf nodes (title and content)
        assert len(builder.get_nodes_by_level(0)) == 2
        # 2 paragraph nodes (title is separate paragraph, content is separate)
        assert len(builder.get_nodes_by_level(1)) == 2
        # 1 section node
        assert len(builder.get_nodes_by_level(2)) == 1
        # 1 root node
        assert len(builder.get_nodes_by_level(3)) == 1

    def test_get_root_node(self):
        """Test getting root node."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Content.", "page_idx": 1, "text_level": 0},
        ]
        builder.build_hierarchy(content_list, "Test Doc")

        root = builder.get_root_node()
        assert root is not None
        assert root.level == 3
        assert root.node_type == "document"

    def test_get_tree_structure(self):
        """Test getting tree structure."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Title", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "Content.", "page_idx": 1, "text_level": 0},
        ]
        builder.build_hierarchy(content_list, "Test Doc")

        tree = builder.get_tree_structure()
        assert tree["type"] == "document"
        assert tree["level"] == 3
        assert len(tree["children"]) == 1


class TestHiChunkEdgeCases:
    """Test edge cases and error handling."""

    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "   ", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Real content.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        # Whitespace-only items should be filtered out
        leaf_nodes = [n for n in nodes if n.level == 0]
        assert len(leaf_nodes) == 1
        assert leaf_nodes[0].content == "Real content."

    def test_empty_text_items(self):
        """Test handling of empty text items."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Content.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf_nodes = [n for n in nodes if n.level == 0]
        assert len(leaf_nodes) == 1

    def test_multiple_headings_no_content(self):
        """Test document with only headings."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Chapter 1", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "Chapter 2", "page_idx": 2, "text_level": 1},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        # Each heading is its own leaf and paragraph
        leaf_nodes = [n for n in nodes if n.level == 0]
        para_nodes = [n for n in nodes if n.level == 1]
        section_nodes = [n for n in nodes if n.level == 2]

        assert len(leaf_nodes) == 2
        assert len(para_nodes) == 2
        assert len(section_nodes) == 2

    def test_real_world_sample(self):
        """Test with a realistic sample similar to actual MineRU output."""
        builder = HiChunkBuilder()
        content_list = [
            {
                "type": "text",
                "text": "1. 项目概述",
                "page_idx": 1,
                "text_level": 1,
                "bbox": [100, 100, 300, 140],
            },
            {
                "type": "text",
                "text": "本项目旨在建设一个智能招投标评分系统。",
                "page_idx": 1,
                "text_level": 0,
                "bbox": [100, 150, 500, 190],
            },
            {
                "type": "text",
                "text": "系统需要支持多种评分标准和自动化处理。",
                "page_idx": 1,
                "text_level": 0,
                "bbox": [100, 200, 500, 240],
            },
            {
                "type": "text",
                "text": "2. 技术要求",
                "page_idx": 2,
                "text_level": 1,
                "bbox": [100, 100, 300, 140],
            },
            {
                "type": "list",
                "list_items": ["支持PDF解析", "支持表格识别", "支持OCR"],
                "page_idx": 2,
                "sub_type": "ordered",
            },
            {
                "type": "text",
                "text": "3. 交付要求",
                "page_idx": 3,
                "text_level": 1,
                "bbox": [100, 100, 300, 140],
            },
            {
                "type": "text",
                "text": "交付周期：6个月",
                "page_idx": 3,
                "text_level": 0,
                "bbox": [100, 150, 300, 190],
            },
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        # Verify structure
        leaf_nodes = [n for n in nodes if n.level == 0]
        para_nodes = [n for n in nodes if n.level == 1]
        section_nodes = [n for n in nodes if n.level == 2]
        root_nodes = [n for n in nodes if n.level == 3]

        assert len(root_nodes) == 1
        assert len(section_nodes) == 3  # 三个章节标题
        assert len(para_nodes) >= 3  # 多个段落
        assert len(leaf_nodes) >= 5  # 多个叶子节点

        # Verify section titles
        section_titles = [s.content for s in section_nodes]
        assert "1. 项目概述" in section_titles
        assert "2. 技术要求" in section_titles
        assert "3. 交付要求" in section_titles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
