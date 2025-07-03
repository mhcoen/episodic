"""RAG-related commands for Episodic."""

import typer
from typing import Optional, List
from datetime import datetime

from episodic.config import config
from episodic.configuration import get_text_color, get_system_color, get_heading_color
from episodic.rag import get_rag_system, ensure_rag_initialized


def search(query: str, limit: Optional[int] = None):
    """Search the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    rag = get_rag_system()
    n_results = limit or config.get('rag_max_results', 5)
    threshold = config.get('rag_search_threshold', 0.7)
    
    results = rag.search(query, n_results=n_results, threshold=threshold)
    
    if not results['documents']:
        typer.secho("No relevant results found.", fg=get_text_color())
        return
    
    typer.secho(f"\n🔍 Search Results for: '{query}'", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    for i, (doc, metadata, distance, doc_id) in enumerate(zip(
        results['documents'], 
        results['metadatas'], 
        results['distances'],
        results['ids']
    )):
        relevance = 1 - distance  # Convert distance to similarity
        typer.secho(f"\n[{i+1}] ", nl=False, fg=get_system_color(), bold=True)
        typer.secho(f"Relevance: {relevance:.2%}", fg=get_system_color())
        typer.secho(f"Source: {metadata.get('source', 'Unknown')}", fg=get_text_color())
        typer.secho(f"ID: {doc_id[:8]}...", fg=get_text_color())
        
        # Show snippet
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        typer.secho(f"{snippet}", fg=get_text_color())


def index_text(content: str, source: Optional[str] = None):
    """Index text content into the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    rag = get_rag_system()
    source = source or "manual_input"
    
    doc_ids = rag.add_document(content, source)
    
    if len(doc_ids) == 1:
        typer.secho(f"✅ Document indexed successfully", fg=get_system_color())
        typer.secho(f"   ID: {doc_ids[0][:8]}...", fg=get_text_color())
    else:
        typer.secho(f"✅ Document indexed in {len(doc_ids)} chunks", fg=get_system_color())
        typer.secho(f"   Parent ID: {doc_ids[0].split('-')[0][:8]}...", fg=get_text_color())
    
    typer.secho(f"   Source: {source}", fg=get_text_color())
    typer.secho(f"   Words: {len(content.split())}", fg=get_text_color())


def index_file(filepath: str):
    """Index a file into the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    import os
    if not os.path.exists(filepath):
        typer.secho(f"File not found: {filepath}", fg="red")
        return
    
    # Check file size
    file_size = os.path.getsize(filepath)
    max_size = config.get('rag_max_file_size', 10 * 1024 * 1024)
    if file_size > max_size:
        typer.secho(f"File too large: {file_size / 1024 / 1024:.1f}MB (max: {max_size / 1024 / 1024:.1f}MB)", fg="red")
        return
    
    # Check allowed file types
    file_ext = os.path.splitext(filepath)[1].lower()
    allowed_types = config.get('rag_allowed_file_types', ['.txt', '.md', '.pdf', '.rst'])
    if file_ext not in allowed_types:
        typer.secho(f"Unsupported file type: {file_ext}", fg="red")
        typer.secho(f"Allowed types: {', '.join(allowed_types)}", fg="yellow")
        return
    
    # Determine file type and load accordingly
    if filepath.endswith('.pdf'):
        # Try to use existing PDF loading logic if available
        try:
            from episodic.commands.documents import extract_pdf_content
            content = extract_pdf_content(filepath)
            source = os.path.basename(filepath)
        except ImportError:
            typer.secho("PDF support not available. Install required dependencies.", fg="red")
            return
    else:
        # Read text file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            source = os.path.basename(filepath)
        except Exception as e:
            typer.secho(f"Error reading file: {e}", fg="red")
            return
    
    # Index the content
    index_text(content, source=source)


def rag_toggle(enable: Optional[bool] = None):
    """Enable or disable RAG functionality."""
    if enable is None:
        # Toggle current state
        current = config.get('rag_enabled', False)
        enable = not current
    
    config.set('rag_enabled', enable)
    
    status = "enabled" if enable else "disabled"
    typer.secho(f"RAG {status}", fg=get_system_color())
    
    if enable:
        # Initialize RAG system
        if ensure_rag_initialized():
            rag = get_rag_system()
            if rag:
                stats = rag.get_stats()
                typer.secho(f"Knowledge base: {stats['total_documents']} documents", fg=get_text_color())


def rag_stats():
    """Show RAG system statistics."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    rag = get_rag_system()
    stats = rag.get_stats()
    
    typer.secho("\n📊 RAG System Statistics", fg=get_heading_color(), bold=True)
    typer.secho("─" * 40, fg=get_heading_color())
    
    typer.secho("Documents indexed: ", nl=False, fg=get_text_color())
    typer.secho(f"{stats['total_documents']}", fg=get_system_color())
    
    typer.secho("Embedding model: ", nl=False, fg=get_text_color())
    typer.secho(f"{stats['embedding_model']}", fg=get_system_color())
    
    typer.secho("Auto-search: ", nl=False, fg=get_text_color())
    typer.secho(f"{config.get('rag_auto_search', True)}", fg=get_system_color())
    
    typer.secho("Search threshold: ", nl=False, fg=get_text_color())
    typer.secho(f"{config.get('rag_search_threshold', 0.7)}", fg=get_system_color())
    
    typer.secho("Max results: ", nl=False, fg=get_text_color())
    typer.secho(f"{config.get('rag_max_results', 5)}", fg=get_system_color())
    
    # Show source distribution
    sources = rag.get_source_distribution()
    if sources:
        typer.secho("\nDocument Sources:", fg=get_heading_color())
        for source, count in sources.items():
            typer.secho(f"  {source}: ", nl=False, fg=get_text_color())
            typer.secho(f"{count} documents", fg=get_system_color())


def docs_list(limit: Optional[int] = None, source: Optional[str] = None):
    """List documents in the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    rag = get_rag_system()
    docs = rag.list_documents(limit=limit, source_filter=source)
    
    if not docs:
        typer.secho("No documents found.", fg=get_text_color())
        return
    
    typer.secho(f"\n📄 Documents in Knowledge Base", fg=get_heading_color(), bold=True)
    if source:
        typer.secho(f"   Filtered by source: {source}", fg=get_text_color())
    typer.secho("─" * 60, fg=get_heading_color())
    
    for doc in docs:
        typer.secho(f"\nID: ", nl=False, fg=get_text_color())
        typer.secho(f"{doc['id'][:8]}...", fg=get_system_color())
        
        typer.secho("Source: ", nl=False, fg=get_text_color())
        typer.secho(f"{doc['source']}", fg=get_system_color())
        
        typer.secho("Words: ", nl=False, fg=get_text_color())
        typer.secho(f"{doc['word_count']}", fg=get_system_color())
        
        typer.secho("Indexed: ", nl=False, fg=get_text_color())
        typer.secho(f"{doc['indexed_at']}", fg=get_system_color())
        
        if doc['content']:
            typer.secho("Preview: ", nl=False, fg=get_text_color())
            typer.secho(f"{doc['content']}", fg=get_text_color())


def docs_show(doc_id: str):
    """Show full content of a specific document."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    rag = get_rag_system()
    
    # Support partial ID matching
    docs = rag.list_documents()
    matching_docs = [d for d in docs if d['id'].startswith(doc_id)]
    
    if not matching_docs:
        typer.secho(f"No document found with ID starting with: {doc_id}", fg="red")
        return
    
    if len(matching_docs) > 1:
        typer.secho(f"Multiple documents found matching '{doc_id}':", fg="yellow")
        for doc in matching_docs:
            typer.secho(f"  {doc['id']} - {doc['source']}", fg=get_text_color())
        return
    
    # Get full document
    doc = rag.get_document(matching_docs[0]['id'])
    if not doc:
        typer.secho(f"Failed to retrieve document.", fg="red")
        return
    
    typer.secho(f"\n📄 Document Details", fg=get_heading_color(), bold=True)
    typer.secho("─" * 60, fg=get_heading_color())
    
    typer.secho("ID: ", nl=False, fg=get_text_color())
    typer.secho(f"{doc['id']}", fg=get_system_color())
    
    metadata = doc['metadata']
    typer.secho("Source: ", nl=False, fg=get_text_color())
    typer.secho(f"{metadata.get('source', 'Unknown')}", fg=get_system_color())
    
    typer.secho("Words: ", nl=False, fg=get_text_color())
    typer.secho(f"{metadata.get('word_count', 'Unknown')}", fg=get_system_color())
    
    typer.secho("Indexed: ", nl=False, fg=get_text_color())
    typer.secho(f"{metadata.get('indexed_at', 'Unknown')}", fg=get_system_color())
    
    if metadata.get('chunk_index') is not None:
        typer.secho("Chunk: ", nl=False, fg=get_text_color())
        typer.secho(f"{metadata['chunk_index'] + 1} of {metadata.get('total_chunks', '?')}", fg=get_system_color())
    
    typer.secho("\nContent:", fg=get_heading_color())
    typer.secho("─" * 60, fg=get_heading_color())
    typer.secho(doc['content'], fg=get_text_color())


def docs_remove(doc_id: str):
    """Remove a document from the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    rag = get_rag_system()
    
    # Support partial ID matching
    docs = rag.list_documents()
    matching_docs = [d for d in docs if d['id'].startswith(doc_id)]
    
    if not matching_docs:
        typer.secho(f"No document found with ID starting with: {doc_id}", fg="red")
        return
    
    if len(matching_docs) > 1:
        typer.secho(f"Multiple documents found matching '{doc_id}':", fg="yellow")
        for doc in matching_docs:
            typer.secho(f"  {doc['id']} - {doc['source']}", fg=get_text_color())
        return
    
    doc_to_remove = matching_docs[0]
    
    # Confirm removal
    typer.secho(f"Remove document: {doc_to_remove['source']} ({doc_to_remove['id'][:8]}...)?", 
                fg="yellow")
    if not typer.confirm("Are you sure?"):
        typer.secho("Cancelled.", fg=get_text_color())
        return
    
    # Remove all chunks if this is a chunked document
    parent_id = doc_to_remove['id'].split('-')[0]
    removed_count = 0
    
    for doc in docs:
        if doc['id'].startswith(parent_id):
            if rag.remove_document(doc['id']):
                removed_count += 1
    
    if removed_count > 0:
        typer.secho(f"✅ Removed {removed_count} document chunk(s)", fg=get_system_color())
    else:
        typer.secho("Failed to remove document.", fg="red")


def docs_clear(source: Optional[str] = None):
    """Clear documents from the knowledge base."""
    if not config.get('rag_enabled', False):
        typer.secho("RAG is not enabled. Use '/rag on' to enable.", fg="yellow")
        return
    
    if not ensure_rag_initialized():
        return
    
    # Confirm clearing
    if source:
        typer.secho(f"Clear all documents from source '{source}'?", fg="yellow")
    else:
        typer.secho("Clear ALL documents from the knowledge base?", fg="yellow")
    
    if not typer.confirm("Are you sure?"):
        typer.secho("Cancelled.", fg=get_text_color())
        return
    
    rag = get_rag_system()
    count = rag.clear_documents(source_filter=source)
    
    if count > 0:
        typer.secho(f"✅ Cleared {count} documents", fg=get_system_color())
    else:
        typer.secho("No documents to clear.", fg=get_text_color())


def docs_command(action: Optional[str] = None, *args):
    """Main docs command handler."""
    if not action:
        # Default to list
        docs_list()
        return
    
    action = action.lower()
    
    if action == "list":
        # Parse list arguments
        limit = None
        source = None
        
        # Simple argument parsing
        for i, arg in enumerate(args):
            if arg == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
            elif arg == "--source" and i + 1 < len(args):
                source = args[i + 1]
        
        docs_list(limit=limit, source=source)
    
    elif action == "show":
        if not args:
            typer.secho("Usage: /docs show <doc_id>", fg="red")
            return
        docs_show(args[0])
    
    elif action == "remove" or action == "rm":
        if not args:
            typer.secho("Usage: /docs remove <doc_id>", fg="red")
            return
        docs_remove(args[0])
    
    elif action == "clear":
        source = args[0] if args else None
        docs_clear(source=source)
    
    else:
        typer.secho(f"Unknown action: {action}", fg="red")
        typer.secho("Available actions: list, show, remove, clear", fg=get_text_color())