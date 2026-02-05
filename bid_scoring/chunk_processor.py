"""Chunk Processor for Small-to-Big Retrieval Strategy.

This module implements intelligent chunk merging for optimal RAG performance:
- Small chunks (< MIN_CHUNK_SIZE) are merged to avoid fragmentation
- Large chunks (> MAX_CHUNK_SIZE) are kept intact (assuming MinerU did good splitting)
- Section-level content is preserved for LLM context
- Chunk-level content is optimized for embedding and search

Best Practices (2026):
- Chunk size: 200-800 chars (~150-600 tokens) for optimal embedding quality
- Parent-child relationship enables small-to-big retrieval
- Dual content storage: chunks for search, sections for generation
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Chunk size thresholds
MIN_CHUNK_SIZE = 200  # Merge chunks smaller than this
MAX_CHUNK_SIZE = 800  # Optimal max for embedding quality
MAX_EMBEDDING_TOKENS = 8191  # OpenAI embedding limit


@dataclass
class ProcessedChunk:
    """Processed chunk ready for embedding and storage."""
    content: str  # Full content for reference
    content_for_embedding: str  # Processed content for embedding
    char_count: int
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_start: int = 0
    page_end: int = 0


@dataclass
class SectionWithChunks:
    """Section with its processed chunks for small-to-big retrieval."""
    heading: str
    level: int
    content: str  # Full merged content for LLM
    page_range: Tuple[int, int]
    chunks: List[ProcessedChunk]
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_chunk_ids: List[str] = field(default_factory=list)


class SmartChunkMerger:
    """Intelligently merge chunks for optimal embedding quality.
    
    Strategy:
    1. Merge consecutive short chunks (< MIN_CHUNK_SIZE) to avoid fragmentation
    2. Keep medium chunks (MIN_CHUNK_SIZE to MAX_CHUNK_SIZE) as-is
    3. Preserve large chunks (> MAX_CHUNK_SIZE) assuming semantic boundaries
    4. Never merge across different pages or sections
    """
    
    def __init__(
        self,
        min_size: int = MIN_CHUNK_SIZE,
        max_size: int = MAX_CHUNK_SIZE,
        max_tokens: int = MAX_EMBEDDING_TOKENS
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.max_tokens = max_tokens
    
    def merge_chunks(
        self,
        raw_chunks: List[Dict[str, Any]],
        section_heading: str = ""
    ) -> List[ProcessedChunk]:
        """Merge raw chunks intelligently.
        
        Args:
            raw_chunks: List of raw chunk dicts with keys:
                - chunk_id: str
                - text_raw or content: str
                - page_idx: int
                - chunk_index: int
            section_heading: Section heading for context
            
        Returns:
            List of processed chunks ready for embedding
        """
        if not raw_chunks:
            return []
        
        # Sort by page and index to ensure correct order
        sorted_chunks = sorted(
            raw_chunks,
            key=lambda c: (c.get("page_idx", 0), c.get("chunk_index", 0))
        )
        
        processed_chunks: List[ProcessedChunk] = []
        buffer: List[Dict[str, Any]] = []
        buffer_size = 0
        
        for chunk in sorted_chunks:
            # Support both 'text_raw' (MinerU) and 'content' (ParagraphMerger) keys
            chunk_text = chunk.get("text_raw", chunk.get("content", "")).strip()
            if not chunk_text:
                continue
            
            chunk_len = len(chunk_text)
            chunk_page = chunk.get("page_idx", 0)
            
            # Check if we need to flush buffer before this chunk
            if buffer:
                last_chunk_page = buffer[-1].get("page_idx", 0)
                
                # Never merge across pages
                if chunk_page != last_chunk_page:
                    processed_chunks.append(self._flush_buffer(buffer))
                    buffer = []
                    buffer_size = 0
                
                # If buffer is large enough, flush it
                elif buffer_size >= self.min_size:
                    processed_chunks.append(self._flush_buffer(buffer))
                    buffer = []
                    buffer_size = 0
            
            # Add current chunk to buffer
            buffer.append(chunk)
            buffer_size += chunk_len
            
            # If buffer exceeds max size, flush it
            if buffer_size >= self.max_size:
                processed_chunks.append(self._flush_buffer(buffer))
                buffer = []
                buffer_size = 0
        
        # Flush remaining buffer
        if buffer:
            processed_chunks.append(self._flush_buffer(buffer))
        
        # Post-process: merge any remaining very small chunks forward
        processed_chunks = self._merge_small_chunks_forward(processed_chunks)
        
        return processed_chunks
    
    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> ProcessedChunk:
        """Create a processed chunk from buffer."""
        # Support both 'text_raw' (MinerU) and 'content' (ParagraphMerger) keys
        content = "\n".join(c.get("text_raw", c.get("content", "")).strip() for c in buffer)
        source_ids = [c.get("chunk_id") for c in buffer if c.get("chunk_id")]
        
        pages = [c.get("page_idx", 0) for c in buffer]
        page_start = min(pages) if pages else 0
        page_end = max(pages) if pages else 0
        
        # For embedding, truncate if too long (safety check)
        content_for_embedding = content[:self.max_size * 2]  # Generous limit
        
        return ProcessedChunk(
            content=content,
            content_for_embedding=content_for_embedding,
            char_count=len(content_for_embedding),
            source_chunk_ids=source_ids,
            page_start=page_start,
            page_end=page_end,
            metadata={
                "merged_count": len(buffer),
                "original_char_count": len(content)
            }
        )
    
    def _merge_small_chunks_forward(
        self,
        chunks: List[ProcessedChunk]
    ) -> List[ProcessedChunk]:
        """Merge very small chunks forward into the next chunk."""
        if not chunks:
            return chunks
        
        result: List[ProcessedChunk] = []
        pending_small: Optional[ProcessedChunk] = None
        
        for chunk in chunks:
            # If chunk is large enough, add it directly
            if chunk.char_count >= self.min_size:
                if pending_small:
                    # Merge pending small chunk forward
                    merged_content = pending_small.content + "\n" + chunk.content
                    merged_embedding_content = (
                        pending_small.content_for_embedding + "\n" + chunk.content_for_embedding
                    )
                    chunk = ProcessedChunk(
                        content=merged_content,
                        content_for_embedding=merged_embedding_content[:self.max_size * 2],
                        char_count=min(len(merged_embedding_content), self.max_size * 2),
                        source_chunk_ids=pending_small.source_chunk_ids + chunk.source_chunk_ids,
                        page_start=min(pending_small.page_start, chunk.page_start),
                        page_end=max(pending_small.page_end, chunk.page_end),
                        metadata={
                            "merged_count": (
                                pending_small.metadata.get("merged_count", 1) + 
                                chunk.metadata.get("merged_count", 1)
                            ),
                            "forward_merged": True
                        }
                    )
                    pending_small = None
                result.append(chunk)
            else:
                # Small chunk - try to merge forward
                if pending_small:
                    # We have two small chunks in a row, merge them
                    merged = ProcessedChunk(
                        content=pending_small.content + "\n" + chunk.content,
                        content_for_embedding=(
                            pending_small.content_for_embedding + "\n" + chunk.content_for_embedding
                        )[:self.max_size * 2],
                        char_count=min(
                            len(pending_small.content_for_embedding + chunk.content_for_embedding),
                            self.max_size * 2
                        ),
                        source_chunk_ids=pending_small.source_chunk_ids + chunk.source_chunk_ids,
                        page_start=min(pending_small.page_start, chunk.page_start),
                        page_end=max(pending_small.page_end, chunk.page_end),
                        metadata={
                            "merged_count": (
                                pending_small.metadata.get("merged_count", 1) + 
                                chunk.metadata.get("merged_count", 1)
                            )
                        }
                    )
                    pending_small = merged
                else:
                    pending_small = chunk
        
        # If we still have a pending small chunk, add it
        if pending_small:
            if result:
                # Merge backward into last chunk
                last = result[-1]
                merged_content = last.content + "\n" + pending_small.content
                merged_embedding_content = (
                    last.content_for_embedding + "\n" + pending_small.content_for_embedding
                )[:self.max_size * 2]
                result[-1] = ProcessedChunk(
                    content=merged_content,
                    content_for_embedding=merged_embedding_content,
                    char_count=len(merged_embedding_content),
                    source_chunk_ids=last.source_chunk_ids + pending_small.source_chunk_ids,
                    page_start=min(last.page_start, pending_small.page_start),
                    page_end=max(last.page_end, pending_small.page_end),
                    metadata={
                        "merged_count": (
                            last.metadata.get("merged_count", 1) + 
                            pending_small.metadata.get("merged_count", 1)
                        ),
                        "backward_merged": True
                    }
                )
            else:
                result.append(pending_small)
        
        return result


class SectionChunkBuilder:
    """Build sections with smart chunking for small-to-big retrieval."""
    
    def __init__(self, merger: Optional[SmartChunkMerger] = None):
        self.merger = merger or SmartChunkMerger()
    
    def build_sections_with_chunks(
        self,
        paragraphs: List[Dict[str, Any]],
        document_title: str = ""
    ) -> List[SectionWithChunks]:
        """Build sections with smart chunking from paragraphs.
        
        Args:
            paragraphs: List of paragraph dicts from ParagraphMerger
            document_title: Document title for context
            
        Returns:
            List of sections, each containing processed chunks
        """
        sections: List[SectionWithChunks] = []
        
        # Use a mutable container to allow modification in nested function
        state = {
            'current_section': None,
            'current_chunks_raw': [],
            'current_all_sources': []
        }
        
        def finalize_section():
            """Finalize current section with smart chunking."""
            if state['current_section'] is None or not state['current_chunks_raw']:
                return
            
            # Process chunks with smart merging
            processed_chunks = self.merger.merge_chunks(
                state['current_chunks_raw'],
                state['current_section'].heading
            )
            
            # Merge all content for the section (for LLM context)
            full_content = "\n\n".join(c.content for c in processed_chunks)
            
            # Get page range
            all_pages = []
            for c in state['current_chunks_raw']:
                page = c.get("page_idx", 0)
                all_pages.append(page)
            
            page_range = (min(all_pages), max(all_pages)) if all_pages else (0, 0)
            
            state['current_section'].content = full_content
            state['current_section'].chunks = processed_chunks
            state['current_section'].page_range = page_range
            state['current_section'].source_chunk_ids = state['current_all_sources']
            
            sections.append(state['current_section'])
        
        for para in paragraphs:
            if para.get("is_heading"):
                # Finalize previous section
                finalize_section()
                
                # Start new section
                state['current_section'] = SectionWithChunks(
                    heading=para["content"],
                    level=para.get("text_level", 1),
                    content="",
                    page_range=(0, 0),
                    chunks=[],
                    metadata={"heading_level": para.get("text_level", 1)}
                )
                state['current_chunks_raw'] = []
                state['current_all_sources'] = para.get("source_chunk_ids", []) or para.get("source_chunks", [])
            else:
                # Regular paragraph
                if state['current_section'] is None:
                    # Create default section for content before first heading
                    state['current_section'] = SectionWithChunks(
                        heading="文档开头",
                        level=1,
                        content="",
                        page_range=(0, 0),
                        chunks=[],
                        metadata={"is_default": True}
                    )
                
                # Add to raw chunks
                state['current_chunks_raw'].append(para)
                
                # Accumulate sources
                para_sources = para.get("source_chunk_ids", []) or para.get("source_chunks", [])
                state['current_all_sources'].extend(para_sources)
        
        # Finalize last section
        finalize_section()
        
        # Filter out empty sections
        filtered = [s for s in sections if s.chunks]
        
        logger.info(
            f"Built {len(filtered)} sections with {sum(len(s.chunks) for s in filtered)} chunks "
            f"({len(sections) - len(filtered)} empty sections filtered)"
        )
        
        return filtered
    
    def get_stats(self, sections: List[SectionWithChunks]) -> Dict[str, Any]:
        """Get statistics about processed sections and chunks."""
        total_chunks = sum(len(s.chunks) for s in sections)
        
        if total_chunks == 0:
            return {"error": "No chunks found"}
        
        char_counts = [c.char_count for s in sections for c in s.chunks]
        
        return {
            "section_count": len(sections),
            "total_chunks": total_chunks,
            "avg_chunks_per_section": total_chunks / len(sections) if sections else 0,
            "char_count_stats": {
                "min": min(char_counts),
                "max": max(char_counts),
                "avg": sum(char_counts) / len(char_counts),
                "median": sorted(char_counts)[len(char_counts) // 2]
            },
            "chunks_by_size": {
                "small_<200": sum(1 for c in char_counts if c < 200),
                "medium_200_800": sum(1 for c in char_counts if 200 <= c <= 800),
                "large_>800": sum(1 for c in char_counts if c > 800)
            }
        }


class ParagraphMerger:
    """Merge short chunks into natural paragraphs based on length and context rules.
    
    Migrated from structure_rebuilder.py to consolidate chunk processing logic.
    """
    
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
        """检查 chunk 是否是有效的标题。"""
        if chunk.get("text_level") != 1:
            return False
        
        text = chunk.get("text_raw", "").strip()
        if not text:
            return False
        
        element_type = chunk.get("element_type")
        if element_type in {"table", "image", "header", "footer"}:
            return False
        
        return True

    def merge(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge short chunks into paragraphs."""
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


def create_small_to_big_sections(
    raw_chunks: List[Dict[str, Any]],
    document_title: str = "",
    min_chunk_size: int = MIN_CHUNK_SIZE,
    max_chunk_size: int = MAX_CHUNK_SIZE
) -> List[SectionWithChunks]:
    """Convenience function to create sections with smart chunking.
    
    This is the main entry point for the small-to-big strategy.
    
    Args:
        raw_chunks: Raw chunks from MinerU (uses 'text' key) or converted format (uses 'text_raw' key)
        document_title: Document title
        min_chunk_size: Minimum chunk size for merging
        max_chunk_size: Maximum chunk size
        
    Returns:
        Sections with processed chunks ready for storage
    """
    # Convert MinerU format to ParagraphMerger format if needed
    # MinerU uses 'text', ParagraphMerger expects 'text_raw'
    converted_chunks = []
    for i, item in enumerate(raw_chunks):
        # Check if already converted
        if 'text_raw' in item:
            converted_chunks.append(item)
        else:
            converted = {
                'chunk_id': item.get('chunk_id', f'chunk_{i}'),
                'text_raw': item.get('text', ''),
                'text_level': item.get('text_level'),
                'page_idx': item.get('page_idx', 0),
                'chunk_index': i,
                'element_type': item.get('type'),
            }
            converted_chunks.append(converted)
    
    # Step 1: Merge raw chunks into paragraphs using existing logic
    paragraph_merger = ParagraphMerger(
        min_length=min_chunk_size,
        max_length=max_chunk_size
    )
    paragraphs = paragraph_merger.merge(converted_chunks)
    
    # Step 2: Build sections with smart chunking
    builder = SectionChunkBuilder(
        merger=SmartChunkMerger(min_size=min_chunk_size, max_size=max_chunk_size)
    )
    sections = builder.build_sections_with_chunks(paragraphs, document_title)
    
    # Log stats
    stats = builder.get_stats(sections)
    logger.info(f"Small-to-big processing stats: {stats}")
    
    return sections
