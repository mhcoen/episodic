"""
Format-preserving document chunker for RAG system.

This module provides intelligent chunking that preserves document formatting
while creating appropriate chunks for vector search.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ChunkType(Enum):
    """Types of content chunks for special handling."""
    HEADER = "header"
    CODE_BLOCK = "code_block"
    LIST = "list"
    TABLE = "table"
    PARAGRAPH = "paragraph"
    MIXED = "mixed"


@dataclass
class DocumentChunk:
    """A chunk of document with preserved formatting."""
    chunk_id: str
    original_text: str  # Original formatting preserved
    clean_text: str     # For embeddings/search
    start_idx: int      # Character position in original
    end_idx: int        # Character position in original
    chunk_type: ChunkType
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'chunk_id': self.chunk_id,
            'original_text': self.original_text,
            'clean_text': self.clean_text,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'chunk_type': self.chunk_type.value,
            'metadata': self.metadata
        }


class FormatPreservingChunker:
    """Chunks documents while preserving formatting and structure."""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 100,
                 min_chunk_size: int = 100):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in words
            chunk_overlap: Overlap between chunks in words
            min_chunk_size: Minimum chunk size to avoid tiny chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
    def chunk_document(self, 
                      content: str, 
                      doc_id: str,
                      doc_type: str = "markdown") -> List[DocumentChunk]:
        """
        Chunk a document preserving its formatting.
        
        Args:
            content: Document content
            doc_id: Document identifier for chunk IDs
            doc_type: Type of document (markdown, text, etc.)
            
        Returns:
            List of DocumentChunk objects
        """
        if doc_type == "markdown":
            return self._chunk_markdown(content, doc_id)
        else:
            return self._chunk_plaintext(content, doc_id)
    
    def _chunk_markdown(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk markdown content respecting its structure."""
        chunks = []
        lines = content.split('\n')
        
        current_chunk_lines = []
        current_chunk_start = 0
        current_word_count = 0
        in_code_block = False
        chunk_index = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_words = len(line.split())
            
            # Check for code block boundaries
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    # If we have content, save current chunk
                    if current_chunk_lines and current_word_count >= self.min_chunk_size:
                        chunk = self._create_chunk(
                            current_chunk_lines, 
                            doc_id, 
                            chunk_index,
                            current_chunk_start
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        current_chunk_lines = []
                        current_chunk_start = i
                        current_word_count = 0
                else:
                    # Ending a code block
                    in_code_block = False
                    # Include the closing ``` in the code block chunk
                    current_chunk_lines.append(line)
                    # Save the code block as its own chunk
                    chunk = self._create_chunk(
                        current_chunk_lines,
                        doc_id,
                        chunk_index,
                        current_chunk_start,
                        chunk_type=ChunkType.CODE_BLOCK
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_lines = []
                    current_chunk_start = i + 1
                    current_word_count = 0
                    i += 1
                    continue
            
            # Don't split within code blocks
            if in_code_block:
                current_chunk_lines.append(line)
                current_word_count += line_words
                i += 1
                continue
            
            # Check if this is a header
            is_header = line.strip().startswith('#')
            
            # Check for natural break points
            should_break = False
            if current_word_count > self.chunk_size:
                # Look for good break points
                if is_header:
                    should_break = True
                elif i > 0 and lines[i-1].strip() == '':
                    # Empty line before current line
                    should_break = True
                elif line.strip() == '' and i < len(lines) - 1:
                    # Current line is empty
                    should_break = True
            
            if should_break and current_chunk_lines:
                # Save current chunk
                chunk = self._create_chunk(
                    current_chunk_lines,
                    doc_id,
                    chunk_index,
                    current_chunk_start
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines
                current_chunk_start = i - len(overlap_lines)
                current_word_count = sum(len(line.split()) for line in overlap_lines)
            
            # Add current line to chunk
            current_chunk_lines.append(line)
            current_word_count += line_words
            i += 1
        
        # Don't forget the last chunk
        if current_chunk_lines:
            chunk = self._create_chunk(
                current_chunk_lines,
                doc_id,
                chunk_index,
                current_chunk_start
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_plaintext(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk plain text content."""
        # For plain text, we'll split on paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        chunk_index = 0
        char_offset = 0
        
        for para in paragraphs:
            para_words = para.split()
            
            if len(para_words) <= self.chunk_size:
                # Paragraph fits in one chunk
                chunk = DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    original_text=para,
                    clean_text=' '.join(para_words),
                    start_idx=char_offset,
                    end_idx=char_offset + len(para),
                    chunk_type=ChunkType.PARAGRAPH,
                    metadata={'paragraph_index': chunk_index}
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Need to split paragraph
                for i in range(0, len(para_words), self.chunk_size - self.chunk_overlap):
                    chunk_words = para_words[i:i + self.chunk_size]
                    # Preserve original formatting by finding the text span
                    chunk_text = ' '.join(chunk_words)
                    
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_index}",
                        original_text=chunk_text,  # TODO: preserve original spacing
                        clean_text=chunk_text,
                        start_idx=char_offset + i,
                        end_idx=char_offset + i + len(chunk_text),
                        chunk_type=ChunkType.PARAGRAPH,
                        metadata={'paragraph_index': chunk_index}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            
            char_offset += len(para) + 2  # Account for double newline
        
        return chunks
    
    def _create_chunk(self,
                     lines: List[str],
                     doc_id: str,
                     chunk_index: int,
                     start_line: int,
                     chunk_type: Optional[ChunkType] = None) -> DocumentChunk:
        """Create a chunk from lines of text."""
        original_text = '\n'.join(lines)
        
        # Create clean text for embeddings
        # Remove extra whitespace but preserve some structure
        clean_lines = []
        for line in lines:
            # Keep some structure indicators
            if line.strip().startswith('#'):
                clean_lines.append(line.strip())
            elif line.strip().startswith('-') or line.strip().startswith('*'):
                clean_lines.append(line.strip())
            else:
                clean_lines.append(' '.join(line.split()))
        
        clean_text = ' '.join(clean_lines)
        
        # Detect chunk type if not specified
        if chunk_type is None:
            chunk_type = self._detect_chunk_type(lines)
        
        # Calculate metadata
        metadata = {
            'line_count': len(lines),
            'word_count': len(clean_text.split()),
            'has_code': '```' in original_text,
            'has_list': any(line.strip().startswith(('-', '*', '+')) for line in lines),
            'start_line': start_line,
            'chunk_version': 2  # Version 2 = format-preserving
        }
        
        # Detect indentation level
        indentations = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentations.append(indent)
        
        if indentations:
            metadata['min_indent'] = min(indentations)
            metadata['max_indent'] = max(indentations)
        
        return DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{chunk_index}",
            original_text=original_text,
            clean_text=clean_text,
            start_idx=start_line,  # Using line numbers for now
            end_idx=start_line + len(lines),
            chunk_type=chunk_type,
            metadata=metadata
        )
    
    def _detect_chunk_type(self, lines: List[str]) -> ChunkType:
        """Detect the primary type of content in the chunk."""
        has_header = any(line.strip().startswith('#') for line in lines)
        has_code = any(line.strip().startswith('```') for line in lines)
        has_list = any(line.strip().startswith(('-', '*', '+', '1.')) for line in lines)
        has_table = any('|' in line and line.count('|') > 1 for line in lines)
        
        # Priority order
        if has_code:
            return ChunkType.CODE_BLOCK
        elif has_table:
            return ChunkType.TABLE
        elif has_header:
            return ChunkType.HEADER
        elif has_list:
            return ChunkType.LIST
        elif sum([has_header, has_code, has_list, has_table]) > 1:
            return ChunkType.MIXED
        else:
            return ChunkType.PARAGRAPH
    
    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get lines for overlap from the end of current chunk."""
        if not lines:
            return []
        
        # Calculate overlap in words
        overlap_words = 0
        overlap_lines = []
        
        # Work backwards from the end
        for line in reversed(lines):
            line_words = len(line.split())
            if overlap_words + line_words <= self.chunk_overlap:
                overlap_lines.insert(0, line)
                overlap_words += line_words
            else:
                break
        
        return overlap_lines