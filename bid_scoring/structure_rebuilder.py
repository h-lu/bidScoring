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
        """构建章节树"""
        sections = []
        current_section = None
        current_paragraphs = []
        
        for para in paragraphs:
            if para.get('is_heading'):
                # Save previous section
                if current_section and current_paragraphs:
                    current_section.children = self._create_paragraph_nodes(current_paragraphs)
                    sections.append(current_section)
                    current_paragraphs = []
                
                # Create new section
                current_section = RebuiltNode(
                    node_type='section',
                    level=1,
                    heading=para['content'],
                    content=para['content'],
                    page_range=(para['page_idx'], para['page_idx']),
                    metadata={'heading_level': para.get('level', 1)}
                )
            else:
                # Regular paragraph
                if current_section is None:
                    # Create default section for content before first heading
                    current_section = RebuiltNode(
                        node_type='section',
                        level=1,
                        heading='文档开头',
                        content='文档开头内容',
                        page_range=(para.get('page_idx', 0), para.get('page_idx', 0)),
                        metadata={'is_default': True}
                    )
                current_paragraphs.append(para)
        
        # Handle last section
        if current_section:
            if current_paragraphs:
                current_section.children = self._create_paragraph_nodes(current_paragraphs)
            sections.append(current_section)
        
        return sections
    
    def _create_paragraph_nodes(self, paragraphs: List[Dict]) -> List[RebuiltNode]:
        """Convert paragraph dicts to RebuiltNode objects"""
        return [
            RebuiltNode(
                node_type='paragraph',
                level=0,
                content=p['content'],
                page_range=p.get('page_range', (p['page_idx'], p['page_idx'])),
                source_chunks=p.get('source_chunks', []),
                metadata={'merged_count': p.get('merged_count', 1)}
            )
            for p in paragraphs
        ]
    
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
    
    def __init__(self, min_length: int = 80, max_length: int = 500):
        """Initialize the merger with length thresholds.
        
        Args:
            min_length: Minimum paragraph length before merging (default: 80)
            max_length: Maximum paragraph length to stop merging (default: 500)
        """
        self.min_length = min_length
        self.max_length = max_length
        self._sentence_end_pattern = re.compile(r'[.!?。！？]$')
    
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
            if chunk.get("text_level") == 1:
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
