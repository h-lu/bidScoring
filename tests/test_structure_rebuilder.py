"""Tests for structure rebuilder module."""


def test_merge_short_chunks_into_paragraph():
    """测试将短句合并为段落"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "细胞和组织", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "本身会发出荧光", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "这种自体荧光会干扰观察", "text_level": None, "page_idx": 1, "chunk_index": 2},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 1
    assert "细胞和组织" in paragraphs[0]["content"]
    assert "本身会发出荧光" in paragraphs[0]["content"]
    assert paragraphs[0]["merged_count"] == 3


def test_heading_stops_merging():
    """测试遇到标题时停止合并: [短句1, 短句2, 标题, 短句3] → [para1, heading, para2]"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "这是一个短句", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "这是另一个短句", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "第一章", "text_level": 1, "page_idx": 1, "chunk_index": 2},
        {"chunk_id": "4", "text_raw": "短句继续", "text_level": None, "page_idx": 1, "chunk_index": 3},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 3
    # First paragraph: merged chunks 1 and 2
    assert paragraphs[0]["merged_count"] == 2
    assert "这是一个短句" in paragraphs[0]["content"]
    # Second paragraph: the heading
    assert paragraphs[1].get("is_heading") is True
    assert paragraphs[1]["content"] == "第一章"
    # Third paragraph: chunk 4 alone
    assert paragraphs[2]["merged_count"] == 1
    assert paragraphs[2]["content"] == "短句继续"


def test_page_change_stops_merging():
    """测试页面变化时停止合并: [page1短句, page2短句] → [para1, para2]"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "第一页的短句", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "第一页继续", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "第二页的短句", "text_level": None, "page_idx": 2, "chunk_index": 0},
        {"chunk_id": "4", "text_raw": "第二页继续", "text_level": None, "page_idx": 2, "chunk_index": 1},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 2
    # First paragraph: chunks from page 1
    assert paragraphs[0]["page_idx"] == 1
    assert paragraphs[0]["merged_count"] == 2
    assert "第一页的短句" in paragraphs[0]["content"]
    # Second paragraph: chunks from page 2
    assert paragraphs[1]["page_idx"] == 2
    assert paragraphs[1]["merged_count"] == 2
    assert "第二页的短句" in paragraphs[1]["content"]


def test_long_text_stays_independent():
    """测试长文本保持独立: [长文本(>80char)] → [para1]"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    long_text = (
        "这是一个很长的文本段落，它的长度已经远远地超过了最小段落长度阈值八十字符的限制要求，"
        "因此它应该保持独立而不与其他短句合并成为一个新的段落单元，用于测试长文本独立功能。"
    )
    assert len(long_text) > 80, f"Text length is {len(long_text)}"
    
    chunks = [
        {"chunk_id": "1", "text_raw": long_text, "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "短句", "text_level": None, "page_idx": 1, "chunk_index": 1},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 2
    # First paragraph: the long text alone
    assert paragraphs[0]["merged_count"] == 1
    assert paragraphs[0]["content"] == long_text
    # Second paragraph: the short chunk alone
    assert paragraphs[1]["merged_count"] == 1
    assert paragraphs[1]["content"] == "短句"


def test_sentence_punctuation_stops_merging():
    """测试句子结束标点停止合并"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "这是第一句。", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "这是第二句！", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "这是第三句？", "text_level": None, "page_idx": 1, "chunk_index": 2},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    # Each chunk ends with sentence punctuation, so each becomes its own paragraph
    assert len(paragraphs) == 3
    assert paragraphs[0]["content"] == "这是第一句。"
    assert paragraphs[1]["content"] == "这是第二句！"
    assert paragraphs[2]["content"] == "这是第三句？"


def test_empty_chunks_list():
    """测试空输入返回空列表"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge([])
    
    assert paragraphs == []


def test_all_headings():
    """测试所有块都是标题的情况"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "第一章", "text_level": 1, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "第二章", "text_level": 1, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "第三章", "text_level": 1, "page_idx": 1, "chunk_index": 2},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 3
    for i, para in enumerate(paragraphs):
        assert para.get("is_heading") is True
        assert para["merged_count"] == 1


def test_element_type_handling():
    """测试表格/图片等特殊元素不被合并"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "短句1", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "表格内容", "text_level": None, "page_idx": 1, "chunk_index": 1, "element_type": "table"},
        {"chunk_id": "3", "text_raw": "短句2", "text_level": None, "page_idx": 1, "chunk_index": 2},
        {"chunk_id": "4", "text_raw": "图片说明", "text_level": None, "page_idx": 1, "chunk_index": 3, "element_type": "image"},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    # Should have 4 paragraphs: [para1], [table], [para2], [image]
    assert len(paragraphs) == 4
    assert paragraphs[0]["content"] == "短句1"
    assert paragraphs[1]["content"] == "表格内容"
    assert paragraphs[2]["content"] == "短句2"
    assert paragraphs[3]["content"] == "图片说明"


def test_single_chunk():
    """测试单一块输入"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "只有一个短句", "text_level": None, "page_idx": 1, "chunk_index": 0},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    assert len(paragraphs) == 1
    assert paragraphs[0]["content"] == "只有一个短句"
    assert paragraphs[0]["merged_count"] == 1
    assert paragraphs[0]["source_chunk_ids"] == ["1"]


def test_build_section_tree_from_headings():
    """测试从标题构建章节树"""
    from bid_scoring.structure_rebuilder import TreeBuilder, RebuiltNode
    
    paragraphs = [
        {"type": "heading", "content": "一、技术规格", "level": 1, "page_idx": 1, "is_heading": True, "source_chunks": ["1"]},
        {"type": "paragraph", "content": "激光共聚焦显微镜参数如下", "level": 0, "page_idx": 1, "is_heading": False, "source_chunks": ["2"]},
        {"type": "paragraph", "content": "分辨率: 0.5微米", "level": 0, "page_idx": 1, "is_heading": False, "source_chunks": ["3"]},
        {"type": "heading", "content": "二、商务条款", "level": 1, "page_idx": 2, "is_heading": True, "source_chunks": ["4"]},
        {"type": "paragraph", "content": "质保期5年", "level": 0, "page_idx": 2, "is_heading": False, "source_chunks": ["5"]},
    ]
    
    builder = TreeBuilder()
    sections = builder.build_sections(paragraphs)
    
    assert len(sections) == 2
    assert sections[0].heading == "一、技术规格"
    assert sections[0].node_type == "section"
    assert len(sections[0].children) == 2  # 两个段落
    assert sections[1].heading == "二、商务条款"
    assert len(sections[1].children) == 1


def test_build_document_tree():
    """测试构建完整文档树"""
    from bid_scoring.structure_rebuilder import TreeBuilder, RebuiltNode
    
    # Create sections manually
    section1 = RebuiltNode(
        node_type="section",
        level=1,
        heading="技术规格",
        content="技术规格",
        page_range=(1, 1),
        children=[
            RebuiltNode(node_type="paragraph", level=0, content="参数1"),
            RebuiltNode(node_type="paragraph", level=0, content="参数2"),
        ]
    )
    section2 = RebuiltNode(
        node_type="section",
        level=1,
        heading="商务条款",
        content="商务条款",
        page_range=(2, 2),
        children=[RebuiltNode(node_type="paragraph", level=0, content="质保")]
    )
    
    builder = TreeBuilder()
    doc_root = builder.build_document_tree([section1, section2], "测试文档")
    
    assert doc_root.node_type == "document"
    assert doc_root.heading == "测试文档"
    assert doc_root.level == 2
    assert len(doc_root.children) == 2


def test_skip_short_content_for_llm():
    """测试跳过短内容的LLM调用"""
    from bid_scoring.structure_rebuilder import HierarchicalContextGenerator, RebuiltNode
    
    # Mock LLM to track calls - matches llm_client.chat.completions.create interface
    class MockCompletions:
        def __init__(self):
            self.call_count = 0
        def create(self, *args, **kwargs):
            self.call_count += 1
            return type('Response', (), {'choices': [type('Choice', (), {'message': type('Message', (), {'content': 'mocked'})()})()]})()
    
    class MockChat:
        def __init__(self):
            self.completions = MockCompletions()
        @property
        def call_count(self):
            return self.completions.call_count
    
    class MockLLM:
        def __init__(self):
            self.chat = MockChat()
        @property
        def call_count(self):
            return self.chat.call_count
    
    mock_llm = MockLLM()
    generator = HierarchicalContextGenerator(llm_client=mock_llm, document_title="测试文档")
    
    # 短内容 (<50字符) - 不应该调用LLM
    short_node = RebuiltNode(
        node_type="paragraph",
        level=0,
        content="细胞",  # 2 chars
        heading="技术规格"
    )
    context = generator.generate_for_node(short_node, "技术规格")
    assert mock_llm.call_count == 0
    assert "测试文档" in context
    assert "技术规格" in context
    
    # 中等内容 (50-500字符) - 应该调用LLM
    medium_node = RebuiltNode(
        node_type="paragraph",
        level=0,
        content="细胞和组织本身会发出荧光，这种自体荧光会干扰观察。共聚焦显微镜可以有效解决这个问题，提供更清晰的成像效果。",  # >50 chars
        heading="技术规格"
    )
    context = generator.generate_for_node(medium_node, "技术规格")
    assert mock_llm.call_count == 1
    assert context == "mocked"
    
    # 验证统计
    stats = generator.get_stats()
    assert stats['short_skipped'] == 1
    assert stats['medium_processed'] == 1


def test_rule_based_context_generation():
    """测试基于规则的上下文生成"""
    from bid_scoring.structure_rebuilder import HierarchicalContextGenerator
    
    generator = HierarchicalContextGenerator(document_title="显微镜文档")
    
    # 有章节标题
    context = generator._generate_rule_based_context("技术规格")
    assert "显微镜文档" in context
    assert "技术规格" in context
    
    # 无章节标题
    context = generator._generate_rule_based_context(None)
    assert "显微镜文档" in context
