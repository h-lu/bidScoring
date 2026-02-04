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
