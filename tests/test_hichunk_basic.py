"""Tests for HiChunk hierarchical chunking builder."""

import pytest
from bid_scoring.hichunk import HiChunkBuilder, HiChunkNode


class TestHiChunkNode:
    """Test HiChunkNode dataclass."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = HiChunkNode(
            level=0,
            node_type="sentence",
            content="Test content",
            metadata={"page_idx": 1},
        )
        assert node.level == 0
        assert node.node_type == "sentence"
        assert node.content == "Test content"
        assert node.metadata["page_idx"] == 1
        assert node.parent_id is None
        assert node.children_ids == []

    def test_node_id_generation(self):
        """Test that node IDs are auto-generated."""
        node1 = HiChunkNode()
        node2 = HiChunkNode()
        assert node1.node_id != node2.node_id
        assert len(node1.node_id) == 36  # UUID length

    def test_invalid_level(self):
        """Test that invalid level raises error."""
        with pytest.raises(ValueError, match="Level must be 0-3"):
            HiChunkNode(level=4, node_type="sentence")

        with pytest.raises(ValueError, match="Level must be 0-3"):
            HiChunkNode(level=-1, node_type="sentence")

    def test_invalid_node_type(self):
        """Test that invalid node_type raises error."""
        with pytest.raises(ValueError, match="Invalid node_type"):
            HiChunkNode(level=0, node_type="invalid")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        node = HiChunkNode(
            level=1,
            node_type="paragraph",
            content="Paragraph text",
            parent_id="parent-uuid",
            children_ids=["child-uuid"],
            metadata={"key": "value"},
        )
        d = node.to_dict()
        assert d["level"] == 1
        assert d["node_type"] == "paragraph"
        assert d["content"] == "Paragraph text"
        assert d["parent_id"] == "parent-uuid"
        assert d["children_ids"] == ["child-uuid"]
        assert d["metadata"] == {"key": "value"}

    def test_add_child(self):
        """Test adding child nodes."""
        node = HiChunkNode()
        node.add_child("child1")
        node.add_child("child2")
        node.add_child("child1")  # Duplicate should not be added
        assert node.children_ids == ["child1", "child2"]

    def test_set_parent(self):
        """Test setting parent."""
        node = HiChunkNode()
        node.set_parent("parent-uuid")
        assert node.parent_id == "parent-uuid"


class TestHiChunkBuilderBasic:
    """Test basic HiChunkBuilder functionality."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = HiChunkBuilder()
        assert builder.nodes == []

    def test_empty_content_list(self):
        """Test building hierarchy with empty content list."""
        builder = HiChunkBuilder()
        nodes = builder.build_hierarchy([], "Empty Doc")

        assert len(nodes) == 1
        assert nodes[0].level == 3
        assert nodes[0].node_type == "document"
        assert nodes[0].content == "Empty Doc"
        assert nodes[0].metadata.get("empty") is True

    def test_invalid_content_list_type(self):
        """Test that invalid content_list raises error."""
        builder = HiChunkBuilder()
        with pytest.raises(ValueError, match="content_list must be a list"):
            builder.build_hierarchy("not a list", "Test")

    def test_single_text_item(self):
        """Test with a single text item."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Hello world", "page_idx": 1, "text_level": 0}
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        # Should have: 1 leaf + 1 para + 1 section + 1 root = 4 nodes
        assert len(nodes) == 4

        leaf_nodes = [n for n in nodes if n.level == 0]
        para_nodes = [n for n in nodes if n.level == 1]
        section_nodes = [n for n in nodes if n.level == 2]
        root_nodes = [n for n in nodes if n.level == 3]

        assert len(leaf_nodes) == 1
        assert len(para_nodes) == 1
        assert len(section_nodes) == 1
        assert len(root_nodes) == 1


class TestHiChunkParagraphMerging:
    """Test paragraph merging logic."""

    def test_merge_adjacent_sentences(self):
        """Test that adjacent text items are merged into a paragraph."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "First sentence.", "page_idx": 1, "text_level": 0},
            {
                "type": "text",
                "text": "Second sentence.",
                "page_idx": 1,
                "text_level": 0,
            },
            {"type": "text", "text": "Third sentence.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        leaf_nodes = [n for n in nodes if n.level == 0]
        para_nodes = [n for n in nodes if n.level == 1]

        # 3 leaf nodes
        assert len(leaf_nodes) == 3
        # Should be merged into 1 paragraph
        assert len(para_nodes) == 1
        assert len(para_nodes[0].children_ids) == 3
        assert "First sentence." in para_nodes[0].content

    def test_heading_starts_new_paragraph(self):
        """Test that heading starts a new paragraph."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Normal text.", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Chapter 1", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "After heading.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        para_nodes = [n for n in nodes if n.level == 1]

        # Should have 3 paragraphs: normal text, heading, after heading
        assert len(para_nodes) == 3

    def test_page_change_starts_new_paragraph(self):
        """Test that page change starts a new paragraph."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Page 1 text.", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Page 2 text.", "page_idx": 2, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        para_nodes = [n for n in nodes if n.level == 1]

        # Should be 2 paragraphs
        assert len(para_nodes) == 2

    def test_table_separates_paragraph(self):
        """Test that table separates paragraphs."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Before table.", "page_idx": 1, "text_level": 0},
            {
                "type": "table",
                "table_body": "Table content",
                "page_idx": 1,
                "text_level": 0,
            },
            {"type": "text", "text": "After table.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        para_nodes = [n for n in nodes if n.level == 1]

        # Should be 3 paragraphs: text, table, text
        assert len(para_nodes) == 3

    def test_element_type_change_separates_paragraph(self):
        """Test that element type change starts new paragraph."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Some text.", "page_idx": 1, "text_level": 0},
            {
                "type": "list",
                "list_items": ["Item 1", "Item 2"],
                "page_idx": 1,
                "text_level": 0,
            },
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        para_nodes = [n for n in nodes if n.level == 1]

        # Should be 2 paragraphs
        assert len(para_nodes) == 2


class TestHiChunkSectionDetection:
    """Test section detection logic."""

    def test_single_section_no_heading(self):
        """Test that document without headings has single section."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Paragraph 1.", "page_idx": 1, "text_level": 0},
            {"type": "text", "text": "Paragraph 2.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        section_nodes = [n for n in nodes if n.level == 2]

        # Should have 1 default section
        assert len(section_nodes) == 1
        assert section_nodes[0].content == "默认章节"

    def test_headings_create_sections(self):
        """Test that headings create sections."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Chapter 1", "page_idx": 1, "text_level": 1},
            {
                "type": "text",
                "text": "Content in chapter 1.",
                "page_idx": 1,
                "text_level": 0,
            },
            {"type": "text", "text": "Chapter 2", "page_idx": 2, "text_level": 1},
            {
                "type": "text",
                "text": "Content in chapter 2.",
                "page_idx": 2,
                "text_level": 0,
            },
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        section_nodes = [n for n in nodes if n.level == 2]

        # Should have 2 sections
        assert len(section_nodes) == 2
        assert section_nodes[0].content == "Chapter 1"
        assert section_nodes[1].content == "Chapter 2"

    def test_nested_headings(self):
        """Test handling of nested headings."""
        builder = HiChunkBuilder()
        content_list = [
            {"type": "text", "text": "Title", "page_idx": 1, "text_level": 1},
            {"type": "text", "text": "Subtitle", "page_idx": 1, "text_level": 2},
            {"type": "text", "text": "Content.", "page_idx": 1, "text_level": 0},
        ]
        nodes = builder.build_hierarchy(content_list, "Test Doc")

        section_nodes = [n for n in nodes if n.level == 2]

        # Both level 1 and level 2 headings create sections
        assert len(section_nodes) == 2


