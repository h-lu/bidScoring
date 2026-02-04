"""Structure rebuilder module for merging chunks into natural paragraphs."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RebuiltNode:
    """重建后的文档树节点"""
    node_type: str  # 'document', 'section', 'paragraph'
    level: int = 0
    heading: str = ""
    content: str = ""
    page_range: Tuple[int, int] = (0, 0)
    source_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["RebuiltNode"] = field(default_factory=list)


class TreeBuilder:
    """树构建器 - 从段落构建章节层次结构"""
    
    def build_sections(self, paragraphs: List[Dict]) -> List[RebuiltNode]:
        """构建章节树，每个 section 合并为一个 paragraph
        
        策略：
        1. 每个 section 只包含一个 paragraph
        2. 该 paragraph 包含 section 下所有内容的合并文本
        3. 保留所有 source_chunk_ids 用于精准溯源
        4. 适合大上下文窗口的 LLM 和现代 RAG 系统
        """
        sections = []
        current_section = None
        current_content_parts = []
        current_source_chunks = []
        current_page_range = [0, 0]
        
        def finalize_section():
            """Finalize current section with merged content"""
            if current_section is None:
                return
            
            if current_content_parts:
                # Merge all content into a single paragraph
                merged_content = "\n\n".join(current_content_parts)
                
                paragraph_node = RebuiltNode(
                    node_type='paragraph',
                    level=0,
                    content=merged_content,
                    page_range=(current_page_range[0], current_page_range[1]),
                    source_chunks=current_source_chunks.copy(),
                    metadata={'merged_count': len(current_source_chunks)}
                )
                current_section.children = [paragraph_node]
                
                # Update section content to include all text
                current_section.content = merged_content
                current_section.page_range = (current_page_range[0], current_page_range[1])
            
            sections.append(current_section)
        
        for para in paragraphs:
            if para.get('is_heading'):
                # Finalize previous section
                finalize_section()
                
                # Start new section
                current_section = RebuiltNode(
                    node_type='section',
                    level=1,
                    heading=para['content'],
                    content=para['content'],
                    page_range=(para['page_idx'], para['page_idx']),
                    metadata={'heading_level': para.get('level', 1)}
                )
                current_content_parts = []
                current_source_chunks = para.get('source_chunk_ids', []) or para.get('source_chunks', [])
                current_page_range = [para['page_idx'], para['page_idx']]
            else:
                # Regular paragraph - accumulate content
                if current_section is None:
                    # Create default section for content before first heading
                    current_section = RebuiltNode(
                        node_type='section',
                        level=1,
                        heading='文档开头',
                        content='',
                        page_range=(para.get('page_idx', 0), para.get('page_idx', 0)),
                        metadata={'is_default': True}
                    )
                    current_content_parts = []
                    current_source_chunks = []
                    current_page_range = [para.get('page_idx', 0), para.get('page_idx', 0)]
                
                # Accumulate content
                if para.get('content'):
                    current_content_parts.append(para['content'])
                
                # Accumulate source chunks
                para_sources = para.get('source_chunk_ids', []) or para.get('source_chunks', [])
                current_source_chunks.extend(para_sources)
                
                # Update page range
                page_idx = para.get('page_idx', 0)
                current_page_range[0] = min(current_page_range[0], page_idx) if current_page_range[0] else page_idx
                current_page_range[1] = max(current_page_range[1], page_idx)
        
        # Finalize last section
        finalize_section()
        
        # Filter out empty sections (those with no content or only headings)
        # Empty sections occur when:
        # 1. A heading is followed immediately by another heading
        # 2. A heading is at the end of document with no following content
        # 3. Duplicate headings in marketing materials
        filtered_sections = [s for s in sections if s.children]
        
        logger.info(f"Built {len(filtered_sections)} sections ({len(sections) - len(filtered_sections)} empty sections filtered)")
        
        return filtered_sections
    
    def _create_paragraph_nodes(self, paragraphs: List[Dict]) -> List[RebuiltNode]:
        """Convert paragraph dicts to RebuiltNode objects
        
        Filters out empty paragraphs and attempts to merge very short ones
        with adjacent content.
        """
        nodes = []
        pending_short = None  # For merging very short paragraphs
        
        for i, p in enumerate(paragraphs):
            content = p.get('content', '').strip()
            
            # Skip empty paragraphs (from image/table captions)
            if not content:
                continue
            
            # Handle very short content (<= 10 chars) - try to merge with next
            if len(content) <= 10 and i < len(paragraphs) - 1:
                # Store for potential merge
                if pending_short is None:
                    pending_short = p
                    continue
                else:
                    # Merge with previous short paragraph
                    prev_content = pending_short.get('content', '')
                    merged_content = prev_content + ' ' + content if prev_content else content
                    
                    # Get combined source chunks
                    prev_sources = pending_short.get('source_chunk_ids') or pending_short.get('source_chunks', [])
                    curr_sources = p.get('source_chunk_ids') or p.get('source_chunks', [])
                    
                    p['content'] = merged_content
                    p['source_chunk_ids'] = prev_sources + curr_sources
                    p['merged_count'] = pending_short.get('merged_count', 1) + p.get('merged_count', 1)
                    pending_short = None
            
            # If we have a pending short paragraph and current is normal, save both
            if pending_short is not None:
                # Save the pending short paragraph as-is
                prev_content = pending_short.get('content', '')
                if len(prev_content) > 0:  # Only save if not empty
                    prev_source_ids = pending_short.get('source_chunk_ids') or pending_short.get('source_chunks', [])
                    nodes.append(RebuiltNode(
                        node_type='paragraph',
                        level=0,
                        content=prev_content,
                        page_range=pending_short.get('page_range', (pending_short.get('page_idx', 0), pending_short.get('page_idx', 0))),
                        source_chunks=prev_source_ids,
                        metadata={'merged_count': pending_short.get('merged_count', 1)}
                    ))
                pending_short = None
            
            # Create node for current paragraph
            source_ids = p.get('source_chunk_ids') or p.get('source_chunks', [])
            nodes.append(RebuiltNode(
                node_type='paragraph',
                level=0,
                content=content,
                page_range=p.get('page_range', (p.get('page_idx', 0), p.get('page_idx', 0))),
                source_chunks=source_ids,
                metadata={'merged_count': p.get('merged_count', 1)}
            ))
        
        # Handle any remaining pending short paragraph
        if pending_short is not None:
            prev_content = pending_short.get('content', '')
            if len(prev_content) > 0:
                prev_source_ids = pending_short.get('source_chunk_ids') or pending_short.get('source_chunks', [])
                nodes.append(RebuiltNode(
                    node_type='paragraph',
                    level=0,
                    content=prev_content,
                    page_range=pending_short.get('page_range', (pending_short.get('page_idx', 0), pending_short.get('page_idx', 0))),
                    source_chunks=prev_source_ids,
                    metadata={'merged_count': pending_short.get('merged_count', 1)}
                ))
        
        return nodes
    
    def build_document_tree(self, sections: List[RebuiltNode], document_title: str) -> RebuiltNode:
        """Build complete document tree"""
        all_pages = []
        for section in sections:
            all_pages.extend([section.page_range[0], section.page_range[1]])
        
        return RebuiltNode(
            node_type='document',
            level=2,
            heading=document_title,
            content=document_title,
            page_range=(min(all_pages) if all_pages else 0, max(all_pages) if all_pages else 0),
            children=sections,
            metadata={'section_count': len(sections)}
        )


class ParagraphMerger:
    """Merge short chunks into natural paragraphs based on length and context rules."""
    
    def __init__(self, min_length: int = 80, max_length: int = 500, short_threshold: int = 20):
        """Initialize the merger with length thresholds.
        
        Args:
            min_length: Minimum paragraph length before merging (default: 80)
            max_length: Maximum paragraph length to stop merging (default: 500)
            short_threshold: Content length below this is considered "short" and will be merged forward (default: 20)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.short_threshold = short_threshold
        self._sentence_end_pattern = re.compile(r'[.!?。！？]$')
    
    def _is_valid_heading(self, chunk: dict[str, Any]) -> bool:
        """检查 chunk 是否是有效的标题。
        
        有效标题的条件：
        - text_level = 1
        - 内容非空
        - element_type 不为 'table', 'image' 等特殊类型
        """
        if chunk.get("text_level") != 1:
            return False
        
        text = chunk.get("text_raw", "").strip()
        if not text:
            return False
        
        # 过滤掉特殊 element_type（如图片、表格标记）
        # 但允许 element_type 为 None 或 'text' 的情况
        element_type = chunk.get("element_type")
        if element_type in {"table", "image", "header", "footer"}:
            return False
        
        return True

    def merge(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge short chunks into paragraphs.
        
        Args:
            chunks: List of chunk dictionaries with keys:
                - chunk_id: Unique identifier
                - text_raw: Raw text content
                - text_level: Heading level (1-6) or None for body text
                - page_idx: Page number
                - chunk_index: Index within the page
                
        Returns:
            List of paragraph dictionaries with keys:
                - content: Merged text content
                - source_chunk_ids: List of original chunk IDs
                - merged_count: Number of chunks merged
                - page_idx: Page index (from first chunk)
        """
        if not chunks:
            return []
        
        # Sort chunks by page_idx and chunk_index
        sorted_chunks = sorted(chunks, key=lambda c: (c.get("page_idx", 0), c.get("chunk_index", 0)))
        
        paragraphs: list[dict[str, Any]] = []
        buffer: list[dict[str, Any]] = []
        
        for chunk in sorted_chunks:
            # Heading stops merging - flush buffer first
            if self._is_valid_heading(chunk):
                if buffer:
                    paragraphs.append(self._create_paragraph(buffer))
                    buffer = []
                paragraphs.append(self._create_heading_paragraph(chunk))
                continue
            
            # Page change stops merging
            if buffer and chunk.get("page_idx") != buffer[-1].get("page_idx"):
                paragraphs.append(self._create_paragraph(buffer))
                buffer = []
            
            # Check for special element types that should not be merged
            element_type = chunk.get("element_type")
            if element_type in {"table", "image", "header", "footer"}:
                if buffer:
                    paragraphs.append(self._create_paragraph(buffer))
                    buffer = []
                paragraphs.append(self._create_paragraph([chunk]))
                continue
            
            # Current buffer content length
            current_length = sum(len(c.get("text_raw", "")) for c in buffer)
            chunk_text = chunk.get("text_raw", "")
            
            # Check if current text ends with sentence punctuation - flush before adding new chunk
            if buffer and self._sentence_end_pattern.search(buffer[-1].get("text_raw", "")):
                paragraphs.append(self._create_paragraph(buffer))
                buffer = []
                current_length = 0
            
            # If buffer is empty or chunk is short and buffer won't exceed max, add to buffer
            if not buffer:
                buffer.append(chunk)
            elif len(chunk_text) < self.min_length and (current_length + len(chunk_text)) <= self.max_length:
                buffer.append(chunk)
            else:
                # Flush current buffer and start new one
                paragraphs.append(self._create_paragraph(buffer))
                buffer = [chunk]
            
            # Check if buffer reached max length - flush if so
            current_length = sum(len(c.get("text_raw", "")) for c in buffer)
            if buffer and current_length >= self.max_length:
                paragraphs.append(self._create_paragraph(buffer))
                buffer = []
        
        # Flush remaining buffer
        if buffer:
            paragraphs.append(self._create_paragraph(buffer))
        
        return paragraphs
    
    def _create_paragraph(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Create a paragraph from a list of chunks."""
        content = "".join(c.get("text_raw", "") for c in chunks)
        source_ids = [c.get("chunk_id") for c in chunks]
        first_chunk = chunks[0]
        
        return {
            "content": content,
            "source_chunk_ids": source_ids,
            "merged_count": len(chunks),
            "page_idx": first_chunk.get("page_idx"),
        }
    
    def _create_heading_paragraph(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Create a paragraph representing a heading."""
        return {
            "content": chunk.get("text_raw", ""),
            "source_chunk_ids": [chunk.get("chunk_id")],
            "merged_count": 1,
            "page_idx": chunk.get("page_idx"),
            "is_heading": True,
            "text_level": chunk.get("text_level"),
        }
    
    def _merge_short_paragraphs_forward(self, paragraphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge short paragraphs forward into adjacent paragraphs.
        
        Strategy:
        1. Headings are never merged - they always remain independent
        2. For short body paragraphs (content < short_threshold), merge with NEXT paragraph
        3. If next paragraph doesn't exist or is a heading, merge with PREVIOUS paragraph
        4. Empty paragraphs are filtered out entirely
        5. Preserve source_chunk_ids tracking for all merges
        
        Args:
            paragraphs: List of paragraph dictionaries from merge()
            
        Returns:
            List of paragraphs with short content merged forward
        """
        if not paragraphs:
            return []
        
        result: list[dict[str, Any]] = []
        
        for i, para in enumerate(paragraphs):
            content = para.get("content", "").strip()
            is_heading = para.get("is_heading", False)
            
            # Skip empty paragraphs entirely
            if not content:
                continue
            
            # Headings are never merged - always keep them independent
            if is_heading:
                result.append(para)
                continue
            
            # Check if this is a short paragraph (at or below threshold)
            is_short = len(content) <= self.short_threshold
            
            if not is_short:
                # Normal length paragraph - keep as-is
                result.append(para)
                continue
            
            # This is a short paragraph - try to merge with adjacent paragraphs
            # Priority 1: Try to merge FORWARD with next paragraph
            if i < len(paragraphs) - 1:
                next_para = paragraphs[i + 1]
                next_is_heading = next_para.get("is_heading", False)
                
                if not next_is_heading:
                    # Merge forward into next paragraph
                    next_content = next_para.get("content", "").strip()
                    merged_content = content + " " + next_content if next_content else content
                    
                    current_sources = para.get("source_chunk_ids", [])
                    next_sources = next_para.get("source_chunk_ids", [])
                    
                    next_para["content"] = merged_content
                    next_para["source_chunk_ids"] = current_sources + next_sources
                    next_para["merged_count"] = para.get("merged_count", 1) + next_para.get("merged_count", 1)
                    # Skip adding this short para to result - it's merged into next
                    continue
            
            # Priority 2: Try to merge BACKWARD with previous paragraph
            if result:
                prev_para = result[-1]
                prev_is_heading = prev_para.get("is_heading", False)
                
                if not prev_is_heading:
                    # Merge backward into previous paragraph
                    prev_content = prev_para.get("content", "").strip()
                    merged_content = prev_content + " " + content if prev_content else content
                    
                    prev_sources = prev_para.get("source_chunk_ids", [])
                    current_sources = para.get("source_chunk_ids", [])
                    
                    prev_para["content"] = merged_content
                    prev_para["source_chunk_ids"] = prev_sources + current_sources
                    prev_para["merged_count"] = prev_para.get("merged_count", 1) + para.get("merged_count", 1)
                    # Skip adding this short para to result - it's merged into prev
                    continue
            
            # Can't merge anywhere - keep the short paragraph as-is
            result.append(para)
        
        return result


class HierarchicalContextGenerator:
    """分层上下文生成器 - 根据节点类型和长度采用不同策略"""
    
    SHORT_THRESHOLD = 50
    MEDIUM_THRESHOLD = 500
    
    def __init__(self, llm_client=None, document_title: str = ""):
        self.llm_client = llm_client
        self.document_title = document_title
        self.stats = {'short_skipped': 0, 'medium_processed': 0, 'long_processed': 0}
    
    def generate_for_tree(self, root: RebuiltNode) -> None:
        """为整棵树生成上下文"""
        for section in root.children:
            self._generate_for_section(section)
    
    def _generate_for_section(self, section: RebuiltNode) -> None:
        """为章节及其段落生成上下文"""
        for para in section.children:
            para.context = self.generate_for_node(para, section.heading)
        
        # 为章节生成摘要
        if section.children:
            combined = ' '.join([p.content for p in section.children[:3]])
            section.context = self._generate_summary(combined[:400], section.heading)
    
    def generate_for_node(self, node: RebuiltNode, section_title: str) -> str:
        """为单个节点生成上下文"""
        content = node.content
        content_len = len(content)
        
        # 策略1: 超短文本 - 跳过LLM
        if content_len < self.SHORT_THRESHOLD:
            self.stats['short_skipped'] += 1
            return self._generate_rule_based_context(section_title)
        
        # 策略2: 中等文本 - 使用LLM
        if content_len <= self.MEDIUM_THRESHOLD:
            if self.llm_client:
                self.stats['medium_processed'] += 1
                return self._call_llm_for_context(content, section_title)
            return self._generate_rule_based_context(section_title)
        
        # 策略3: 长文本 - 摘要模式
        self.stats['long_processed'] += 1
        return self._generate_summary(content[:600], section_title)
    
    def _generate_rule_based_context(self, section_title: str) -> str:
        """基于规则的上下文（无LLM）"""
        if section_title and section_title != "文档开头":
            return f"此内容来自《{self.document_title}》的「{section_title}」部分。"
        return f"此内容来自《{self.document_title}》。"
    
    def _call_llm_for_context(self, content: str, section_title: str) -> str:
        """调用LLM生成上下文"""
        if not self.llm_client:
            return self._generate_rule_based_context(section_title)
        
        prompt = f"""为以下文档片段生成简洁上下文（1-2句话）：
文档：《{self.document_title}》
章节：{section_title}
内容：{content[:300]}
要求：说明核心信息和在文档中的作用。"""
        
        try:
            # Note: This is a simplified version. In production, use proper OpenAI API
            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "你是专业的文档分析助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=100
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM调用失败: {e}")
        
        return self._generate_rule_based_context(section_title)
    
    def _generate_summary(self, content: str, section_title: str) -> str:
        """生成摘要"""
        preview = content[:150] + "..." if len(content) > 150 else content
        return f"来自《{self.document_title}》「{section_title}」：{preview}"
    
    def get_stats(self) -> dict:
        """获取统计"""
        total = sum(self.stats.values())
        return {
            **self.stats,
            'total': total,
            'llm_savings_percent': (self.stats['short_skipped'] / total * 100) if total > 0 else 0
        }
