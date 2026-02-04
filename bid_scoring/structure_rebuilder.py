"""Structure rebuilder module for merging chunks into natural paragraphs."""

import re
from typing import Any


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
            
            # Current buffer content length
            current_length = sum(len(c.get("text_raw", "")) for c in buffer)
            chunk_text = chunk.get("text_raw", "")
            
            # If buffer is empty or chunk is short and buffer won't exceed max, add to buffer
            if not buffer:
                buffer.append(chunk)
            elif len(chunk_text) < self.min_length and (current_length + len(chunk_text)) <= self.max_length:
                buffer.append(chunk)
            else:
                # Flush current buffer and start new one
                paragraphs.append(self._create_paragraph(buffer))
                buffer = [chunk]
            
            # Check if current text ends with sentence punctuation - flush if so
            if buffer and self._sentence_end_pattern.search(buffer[-1].get("text_raw", "")):
                paragraphs.append(self._create_paragraph(buffer))
                buffer = []
            
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
