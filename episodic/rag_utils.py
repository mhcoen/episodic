"""
Utilities for RAG functionality.

This module consolidates common patterns and utilities used across RAG components.
"""

import sys
import io
from contextlib import contextmanager
from functools import wraps
from io import StringIO
from typing import Optional, List, Dict, Callable

import typer

from episodic.config import config
from episodic.configuration import get_text_color


@contextmanager
def suppress_chromadb_telemetry():
    """
    Context manager to suppress ChromaDB telemetry errors.
    
    ChromaDB has a known issue with telemetry causing errors.
    This suppresses those specific errors while allowing other
    errors to propagate.
    """
    import os
    import warnings
    
    old_stderr = sys.stderr
    captured_stderr = StringIO()
    
    # Save original settings
    original_filters = warnings.filters[:]
    
    # Initialize variables that need to be cleaned up
    null_fd = None
    stderr_backup = None
    stderr_fd = None
    
    try:
        # Create a proper file descriptor for stderr redirect
        # This handles C-level stderr writes from ChromaDB
        null_fd = os.open(os.devnull, os.O_WRONLY)
        
        # Try to get stderr file descriptor
        try:
            stderr_fd = sys.stderr.fileno()
            stderr_backup = os.dup(stderr_fd)
        except (AttributeError, OSError, io.UnsupportedOperation):
            # stderr doesn't have a file descriptor (e.g., StringIO or special environment)
            # Just skip file descriptor manipulation
            stderr_fd = None
            stderr_backup = None
        
        # Redirect both Python and C-level stderr
        if stderr_fd is not None:
            os.dup2(null_fd, stderr_fd)
        sys.stderr = captured_stderr
        
        # Add warning filters
        warnings.filterwarnings("ignore", message=".*telemetry.*")
        warnings.filterwarnings("ignore", message=".*Failed to send telemetry.*")
        warnings.filterwarnings("ignore", category=UserWarning)
        
        yield
        
    finally:
        # Restore original stderr
        if stderr_backup is not None:
            try:
                # stderr_fd should be set if stderr_backup exists
                if stderr_fd is not None:
                    os.dup2(stderr_backup, stderr_fd)
                os.close(stderr_backup)
            except:
                pass  # Ignore errors during cleanup
        if null_fd is not None:
            try:
                os.close(null_fd)
            except:
                pass  # Ignore errors during cleanup
        sys.stderr = old_stderr
        warnings.filters[:] = original_filters
        
        # Only show non-telemetry errors
        try:
            error_output = captured_stderr.getvalue()
            if error_output:
                # Filter out specific telemetry error patterns
                filtered_lines = []
                for line in error_output.splitlines():
                    line_lower = line.lower()
                    if ("telemetry" not in line_lower and 
                        "capture() takes" not in line and
                        "ClientStartEvent" not in line and
                        "CollectionGetEvent" not in line and
                        "CollectionQueryEvent" not in line and
                        "CollectionAddEvent" not in line and
                        "Failed to send telemetry event" not in line):
                        filtered_lines.append(line)
                
                filtered_output = '\n'.join(filtered_lines).strip()
                if filtered_output:
                    sys.stderr.write(filtered_output + '\n')
        except:
            pass  # Ignore errors when writing error output


def requires_rag(func: Callable) -> Callable:
    """
    Decorator that ensures RAG is enabled and initialized.
    
    Use this decorator on any function that requires RAG functionality.
    It will check if RAG is enabled and properly initialized before
    executing the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not config.get('rag_enabled', False):
            typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
            return
        
        from episodic.rag import ensure_rag_initialized
        if not ensure_rag_initialized():
            return
        
        return func(*args, **kwargs)
    
    return wrapper


def find_document_by_partial_id(doc_id: str, docs: List[Dict]) -> Optional[Dict]:
    """
    Find a document by partial ID match.
    
    Args:
        doc_id: Partial or full document ID
        docs: List of document dictionaries with 'id' field
        
    Returns:
        Matching document or None if not found or multiple matches
    """
    matching_docs = [d for d in docs if d['id'].startswith(doc_id)]
    
    if not matching_docs:
        typer.secho(f"No document found with ID starting with: {doc_id}", fg="red")
        return None
    
    if len(matching_docs) > 1:
        typer.secho(f"Multiple documents found matching '{doc_id}':", fg="yellow")
        for doc in matching_docs:
            source = doc.get('source', 'Unknown')
            typer.secho(f"  {doc['id']} - {source}", fg=get_text_color())
        return None
    
    return matching_docs[0]


def validate_chunk_params(chunk_size: int, overlap: int) -> bool:
    """
    Validate document chunking parameters.
    
    Args:
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        True if valid, False otherwise
    """
    if chunk_size <= 0:
        typer.secho("Chunk size must be positive", fg="red")
        return False
    
    if overlap < 0:
        typer.secho("Overlap cannot be negative", fg="red")
        return False
    
    if overlap >= chunk_size:
        typer.secho("Overlap must be less than chunk size", fg="red")
        return False
    
    return True


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def validate_file_for_indexing(filepath: str, check_size: bool = True) -> bool:
    """
    Validate a file before indexing.
    
    Args:
        filepath: Path to the file to validate
        check_size: Whether to check file size limits
        
    Returns:
        True if file is valid for indexing, False otherwise
    """
    import os
    
    if not os.path.exists(filepath):
        typer.secho(f"File not found: {filepath}", fg="red")
        return False
    
    if not os.path.isfile(filepath):
        typer.secho(f"Not a file: {filepath}", fg="red")
        return False
    
    # Check file extension
    file_ext = os.path.splitext(filepath)[1].lower()
    allowed_types = config.get('rag_allowed_file_types', ['.txt', '.md', '.pdf', '.rst'])
    if file_ext not in allowed_types:
        typer.secho(f"Unsupported file type: {file_ext}", fg="red")
        typer.secho(f"Allowed types: {', '.join(allowed_types)}", fg="yellow")
        return False
    
    # Check file size if requested
    if check_size:
        file_size = os.path.getsize(filepath)
        max_size = config.get('rag_max_file_size', 10 * 1024 * 1024)
        if file_size > max_size:
            typer.secho(
                f"File too large: {format_file_size(file_size)} "
                f"(max: {format_file_size(max_size)})", 
                fg="red"
            )
            return False
    
    return True