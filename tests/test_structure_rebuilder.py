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
    """测试页面变化时停止合并: [page1短句, page2短句] → [para1, para2]
    
    ParagraphMerger 只负责基本的 chunks → paragraphs 转换，
    不合并短内容。短内容会在 TreeBuilder 阶段按 Section 合并。
    """
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "第一页的短句", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "第一页继续", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "第二页的短句", "text_level": None, "page_idx": 2, "chunk_index": 0},
        {"chunk_id": "4", "text_raw": "第二页继续", "text_level": None, "page_idx": 2, "chunk_index": 1},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    # ParagraphMerger 不合并短内容，所以应该有 2 个 paragraphs（按 page 分组）
    assert len(paragraphs) == 2
    assert "第一页的短句" in paragraphs[0]["content"]
    assert "第二页的短句" in paragraphs[1]["content"]


def test_long_text_with_short_following():
    """测试长文本后的短内容处理: [长文本, 短句] → [长文本 para], [短句 para]
    
    ParagraphMerger 不合并短内容，保持独立 paragraphs。
    短内容会在 TreeBuilder 阶段按 Section 合并。
    """
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
    
    # ParagraphMerger 不合并短内容，所以应该有 2 个 paragraphs
    assert len(paragraphs) == 2
    assert paragraphs[0]["content"] == long_text
    assert paragraphs[1]["content"] == "短句"


def test_sentence_punctuation_keeps_separate():
    """测试句子结束标点保持独立 paragraphs
    
    ParagraphMerger 不合并短内容，每个 chunk 成为一个独立的 paragraph。
    """
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "这是第一句。", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "这是第二句！", "text_level": None, "page_idx": 1, "chunk_index": 1},
        {"chunk_id": "3", "text_raw": "这是第三句？", "text_level": None, "page_idx": 1, "chunk_index": 2},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    # ParagraphMerger 不合并短内容，所以应该有 3 个 paragraphs
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


def test_element_type_handling_keeps_separate():
    """测试表格/图片等特殊元素保持独立
    
    ParagraphMerger 不合并短内容，特殊元素保持独立的 paragraphs。
    """
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    chunks = [
        {"chunk_id": "1", "text_raw": "短句1", "text_level": None, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "表格内容", "text_level": None, "page_idx": 1, "chunk_index": 1, "element_type": "table"},
        {"chunk_id": "3", "text_raw": "短句2", "text_level": None, "page_idx": 1, "chunk_index": 2},
        {"chunk_id": "4", "text_raw": "图片说明", "text_level": None, "page_idx": 1, "chunk_index": 3, "element_type": "image"},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500)
    paragraphs = merger.merge(chunks)
    
    # ParagraphMerger 不合并短内容，应该有 4 个 paragraphs
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
    """测试从标题构建章节树 - 每个 section 合并为一个 paragraph"""
    from bid_scoring.structure_rebuilder import TreeBuilder, RebuiltNode
    
    paragraphs = [
        {"type": "heading", "content": "一、技术规格", "level": 1, "page_idx": 1, "is_heading": True, "source_chunk_ids": ["1"]},
        {"type": "paragraph", "content": "激光共聚焦显微镜参数如下", "level": 0, "page_idx": 1, "is_heading": False, "source_chunk_ids": ["2"]},
        {"type": "paragraph", "content": "分辨率: 0.5微米", "level": 0, "page_idx": 1, "is_heading": False, "source_chunk_ids": ["3"]},
        {"type": "heading", "content": "二、商务条款", "level": 1, "page_idx": 2, "is_heading": True, "source_chunk_ids": ["4"]},
        {"type": "paragraph", "content": "质保期5年", "level": 0, "page_idx": 2, "is_heading": False, "source_chunk_ids": ["5"]},
    ]
    
    builder = TreeBuilder()
    sections = builder.build_sections(paragraphs)
    
    assert len(sections) == 2
    
    # First section
    assert sections[0].heading == "一、技术规格"
    assert sections[0].node_type == "section"
    assert len(sections[0].children) == 1  # 合并为一个 paragraph
    # 验证内容合并
    assert "激光共聚焦显微镜参数如下" in sections[0].children[0].content
    assert "分辨率: 0.5微米" in sections[0].children[0].content
    # 验证 source_chunks 合并（包括 heading 的 chunk）
    assert set(sections[0].children[0].source_chunks) == {"1", "2", "3"}
    
    # Second section
    assert sections[1].heading == "二、商务条款"
    assert len(sections[1].children) == 1
    assert sections[1].children[0].content == "质保期5年"
    assert sections[1].children[0].source_chunks == ["4", "5"]


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


# =============================================================================
# Tests for _merge_short_paragraphs_forward
# =============================================================================


def test_merge_short_paragraphs_forward_basic():
    """测试基本的向前合并功能: [短句, 长句] → [合并后的长句]"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "短句", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "这是一个比较长的段落内容，用于测试向前合并功能", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert len(result) == 1
    assert "短句" in result[0]["content"]
    assert "这是一个比较长的段落" in result[0]["content"]
    assert result[0]["merged_count"] == 2
    assert set(result[0]["source_chunk_ids"]) == {"1", "2"}


def test_merge_short_paragraphs_empty_filtered():
    """测试空内容被过滤"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "有效内容", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert len(result) == 1
    assert result[0]["content"] == "有效内容"


def test_merge_short_paragraphs_heading_not_merged():
    """测试标题不会被合并"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "短句1", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1, "is_heading": False},
        {"content": "标题", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1, "is_heading": True},
        {"content": "短句2", "source_chunk_ids": ["3"], "merged_count": 1, "page_idx": 1, "is_heading": False},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert len(result) == 3
    # 短句1 无法向前合并到标题，只能保留
    assert result[0]["content"] == "短句1"
    assert result[1]["content"] == "标题"
    assert result[1].get("is_heading") is True
    # 短句2 可以向后合并
    assert "短句2" in result[2]["content"]


def test_merge_short_paragraphs_consecutive_short():
    """测试连续短段落合并: [短1, 短2, 长] → [短1+短2+长]"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "短A", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "短B", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
        {"content": "这是一个很长的段落内容用于测试合并", "source_chunk_ids": ["3"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert len(result) == 1
    assert "短A" in result[0]["content"]
    assert "短B" in result[0]["content"]
    assert "这是一个很长的段落" in result[0]["content"]
    assert result[0]["merged_count"] == 3


def test_merge_short_paragraphs_last_short_backward():
    """测试最后一个短段落向后合并: [长, 短] → [长+短]"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "这是一个很长的段落内容用于测试", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "短句", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert len(result) == 1
    assert "这是一个很长的段落" in result[0]["content"]
    assert "短句" in result[0]["content"]


def test_merge_short_paragraphs_last_short_after_heading():
    """测试标题后的最后一个短段落: [heading, 短] → [heading, 短]（无法合并）"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "标题", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1, "is_heading": True},
        {"content": "短句", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1, "is_heading": False},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    # 短句无法向前合并到标题，也无法向后合并（因为是最后一个），只能保留
    assert len(result) == 2
    assert result[0]["content"] == "标题"
    assert result[1]["content"] == "短句"


def test_merge_short_paragraphs_whitespace_stripped():
    """测试空白字符被正确处理"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "  短句  ", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "长段落内容用于测试空白字符处理", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert len(result) == 1
    # 空白应该被 strip 处理
    assert "  短句  " not in result[0]["content"]
    assert "短句" in result[0]["content"]


def test_merge_short_paragraphs_normal_length_preserved():
    """测试正常长度的段落保持不变（内容长度 >= short_threshold）"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    # 使用长度 >= 20 的段落，应该保持不变
    paragraphs = [
        {"content": "这是一个长度超过二十字符的段落内容用于测试", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "这是另一个长度超过二十字符的段落内容用于测试", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    # 正常长度的段落（>=20字符）应该保持不变
    assert len(result) == 2
    assert "这是一个长度超过二十字符" in result[0]["content"]
    assert "这是另一个长度超过二十字符" in result[1]["content"]


def test_merge_short_paragraphs_empty_input():
    """测试空输入"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    result = merger._merge_short_paragraphs_forward([])
    
    assert result == []


def test_merge_short_paragraphs_all_empty():
    """测试全空内容"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1},
        {"content": "", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    assert result == []


def test_merge_short_paragraphs_integration_with_merge():
    """测试与主 merge 方法集成的端到端场景"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    # 模拟真实场景：标题 + 短正文 + 长正文
    chunks = [
        {"chunk_id": "1", "text_raw": "一、投标函", "text_level": 1, "page_idx": 1, "chunk_index": 0},
        {"chunk_id": "2", "text_raw": "有限公司", "text_level": None, "page_idx": 1, "chunk_index": 1},  # 短内容
        {"chunk_id": "3", "text_raw": "这是一个很长的段落内容用于测试向前合并功能，包含了详细的投标信息", "text_level": None, "page_idx": 1, "chunk_index": 2},
    ]
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = merger.merge(chunks)
    
    # 应该有 2 个段落：heading + merged paragraph
    assert len(paragraphs) == 2
    assert paragraphs[0].get("is_heading") is True
    assert paragraphs[0]["content"] == "一、投标函"
    # 短内容 "有限公司" 应该被合并到后面的长段落
    assert "有限公司" in paragraphs[1]["content"]
    assert "这是一个很长的段落" in paragraphs[1]["content"]


def test_merge_short_paragraphs_complex_scenario():
    """测试复杂场景：多个短内容和长内容混合"""
    from bid_scoring.structure_rebuilder import ParagraphMerger
    
    merger = ParagraphMerger(min_length=80, max_length=500, short_threshold=20)
    paragraphs = [
        {"content": "标题A", "source_chunk_ids": ["1"], "merged_count": 1, "page_idx": 1, "is_heading": True},
        {"content": "短1", "source_chunk_ids": ["2"], "merged_count": 1, "page_idx": 1},  # 短，无法向前合并到标题
        {"content": "这是长段落A的内容用于测试", "source_chunk_ids": ["3"], "merged_count": 1, "page_idx": 1},  # 长，会合并短1
        {"content": "标题B", "source_chunk_ids": ["4"], "merged_count": 1, "page_idx": 1, "is_heading": True},
        {"content": "短2", "source_chunk_ids": ["5"], "merged_count": 1, "page_idx": 1},  # 短，无法向前合并到标题
        {"content": "短3", "source_chunk_ids": ["6"], "merged_count": 1, "page_idx": 1},  # 短，会合并到短2
        {"content": "这是长段落B的内容", "source_chunk_ids": ["7"], "merged_count": 1, "page_idx": 1},  # 长，会合并短2+短3
    ]
    
    result = merger._merge_short_paragraphs_forward(paragraphs)
    
    # 预期结果: [标题A, 长段落A(含短1), 标题B, 长段落B(含短2+短3)]
    assert len(result) == 4
    assert result[0]["content"] == "标题A"
    assert result[0].get("is_heading") is True
    assert "短1" in result[1]["content"]
    assert result[2]["content"] == "标题B"
    assert result[2].get("is_heading") is True
    assert "短2" in result[3]["content"]
    assert "短3" in result[3]["content"]
