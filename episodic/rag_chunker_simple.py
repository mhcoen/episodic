"""
Document chunking functionality for RAG system.
"""

import re
from typing import List, Optional, Tuple, Dict
from io import StringIO

import typer

from episodic.config import config
from episodic.configuration import get_text_color


def chunk_document(content: str, chunk_size: int = None, 
                  overlap: int = None) -> List[Tuple[str, Dict[str, int]]]:
    """
    Split a document into overlapping chunks for better retrieval.
    
    Args:
        content: The document content to chunk
        chunk_size: Size of each chunk in characters (default from config)
        overlap: Overlap between chunks in characters (default from config)
        
    Returns:
        List of tuples (chunk_text, metadata) where metadata contains start/end positions
    """
    if chunk_size is None:
        chunk_size = config.get("rag_chunk_size", 1000)
    if overlap is None:
        overlap = config.get("rag_chunk_overlap", 200)
    
    # Ensure overlap is less than chunk size
    overlap = min(overlap, chunk_size // 2)
    
    chunks = []
    
    # Clean up the content - normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    if len(content) <= chunk_size:
        # If content is smaller than chunk size, return as single chunk
        chunks.append((content, {"start": 0, "end": len(content)}))
        return chunks
    
    # Split into sentences for better chunk boundaries
    sentences = _split_into_sentences(content)
    
    current_chunk = StringIO()
    current_start = 0
    current_pos = 0
    
    for i, sentence in enumerate(sentences):
        sentence_with_space = sentence + " "
        
        # Check if adding this sentence would exceed chunk size
        if current_chunk.tell() + len(sentence_with_space) > chunk_size and current_chunk.tell() > 0:
            # Save current chunk
            chunk_text = current_chunk.getvalue().strip()
            if chunk_text:
                chunks.append((chunk_text, {
                    "start": current_start,
                    "end": current_pos
                }))
            
            # Start new chunk with overlap
            # Find sentences to include in overlap
            overlap_start = max(0, current_pos - overlap)
            overlap_text = _get_overlap_text(content, overlap_start, current_pos)
            
            current_chunk = StringIO()
            current_chunk.write(overlap_text)
            current_start = overlap_start
        
        # Add sentence to current chunk
        current_chunk.write(sentence_with_space)
        current_pos += len(sentence_with_space)
    
    # Don't forget the last chunk
    chunk_text = current_chunk.getvalue().strip()
    if chunk_text:
        chunks.append((chunk_text, {
            "start": current_start,
            "end": current_pos
        }))
    
    # Show debug info
    if config.get("debug"):
        typer.echo(f"Chunked document into {len(chunks)} chunks", fg=get_text_color())
        for i, (chunk, meta) in enumerate(chunks[:3]):  # Show first 3 chunks
            preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            typer.echo(f"  Chunk {i+1}: {len(chunk)} chars, pos {meta['start']}-{meta['end']}", 
                      fg=get_text_color())
            typer.echo(f"    Preview: {preview}", fg=get_text_color(), dim=True)
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting - can be improved with NLTK if needed
    # Split on period, question mark, or exclamation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def _get_overlap_text(content: str, start: int, end: int) -> str:
    """
    Get text for overlap region, trying to start at sentence boundary.
    
    Args:
        content: Full content
        start: Start position
        end: End position
        
    Returns:
        Text for overlap region
    """
    # Try to find a sentence boundary near the start
    search_start = max(0, start - 50)
    search_text = content[search_start:end]
    
    # Look for sentence boundaries
    for pattern in ['. ', '! ', '? ', '\n\n']:
        pos = search_text.find(pattern)
        if pos != -1:
            actual_start = search_start + pos + len(pattern)
            return content[actual_start:end]
    
    # No sentence boundary found, use the original start
    return content[start:end]