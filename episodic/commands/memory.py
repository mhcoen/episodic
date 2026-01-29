"""
Memory management commands for viewing, searching, and managing memory entries.

This module provides commands for interacting with the RAG-based memory system:
- /memory - View and search memory entries
- /forget - Remove memory entries
- /memory-stats - Show memory system statistics
"""

import typer
from typing import Optional, List
from datetime import datetime

from episodic.config import config
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color,
    get_error_color, get_warning_color, get_success_color
)


def memory_command(action: Optional[str] = None, *args):
    """
    View and manage memory entries.

    Usage:
        /memory                    # Show recent memory entries
        /memory search <query>     # Search memory entries
        /memory show <id>          # Show specific memory entry
        /memory list [limit]       # List memories with optional limit
        /memory index [limit]      # Index unindexed conversations
        /memory index --failed     # Retry failed indexes only
        /memory stats              # Show indexing statistics
        /memory clean              # Remove poisoned recall-query entries
        /memory clean <pattern>    # Remove entries matching pattern
    """
    if not action:
        # Show recent memories
        list_memories(limit=10)
    elif action == "search":
        if args:
            query = " ".join(args)
            search_memories(query)
        else:
            typer.secho("Usage: /memory search <query>", fg=get_error_color())
    elif action == "show":
        if args:
            show_memory(args[0])
        else:
            typer.secho("Usage: /memory show <id>", fg=get_error_color())
    elif action == "list":
        limit = int(args[0]) if args else 20
        list_memories(limit=limit)
    elif action == "index":
        # Parse args for --failed flag and limit
        failed_only = '--failed' in args
        # Filter out --failed from args to get limit
        limit_args = [a for a in args if a != '--failed']
        limit = int(limit_args[0]) if limit_args else 100
        index_recent_conversations(limit=limit, failed_only=failed_only)
    elif action == "stats":
        show_indexing_stats()
    elif action == "clean":
        if args:
            pattern = " ".join(args)
        else:
            pattern = None
        clean_poisoned_memories(pattern)
    else:
        typer.secho(f"Unknown memory action: {action}", fg=get_error_color())
        typer.secho("Available: search, show, list, index, stats, clean", fg=get_text_color())


def list_memories(limit: int = 20):
    """List recent memory entries."""
    # Get RAG system (always available for viewing)
    from episodic.rag import get_rag_system
    rag = get_rag_system()
    if not rag:
        typer.secho("❌ Failed to initialize memory system", fg=get_error_color())
        return
    
    typer.secho("\n📚 Memory Entries", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    # Get recent documents (conversation memories only)
    docs = rag.list_documents(limit=limit, source_filter='conversation')
    
    if not docs:
        typer.secho("\nNo memories stored yet.", fg=get_text_color())
        typer.secho("Memories are created automatically from conversations.", fg=get_text_color(), dim=True)
        return
    
    for i, doc in enumerate(docs):
        # Format timestamp
        indexed_at = doc.get('indexed_at', '')
        if indexed_at:
            try:
                dt = datetime.fromisoformat(indexed_at)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = indexed_at[:16]
        else:
            time_str = "Unknown"
        
        # Get source type
        source = doc.get('source', 'unknown')
        source_icon = {
            'conversation': '💬',
            'file': '📄',
            'text': '📝',
            'web': '🌐'
        }.get(source, '📌')
        
        # Display entry
        typer.secho(f"\n{source_icon} [{doc['doc_id'][:8]}] ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(f"{time_str}", fg=get_text_color(), dim=True)
        
        # Show preview of content (first 100 chars)
        if doc.get('preview'):
            preview = doc['preview'][:100].strip()
            if len(doc['preview']) > 100:
                preview += "..."
            typer.secho(f"   {preview}", fg=get_text_color())
        
        # Show metadata
        metadata = doc.get('metadata', {})
        if metadata.get('topic'):
            typer.secho(f"   Topic: {metadata['topic']}", fg=get_text_color(), dim=True)
        if metadata.get('filename'):
            typer.secho(f"   File: {metadata['filename']}", fg=get_text_color(), dim=True)
        
        # Show usage stats
        if doc.get('retrieval_count', 0) > 0:
            typer.secho(f"   Retrieved: {doc['retrieval_count']} times", fg=get_success_color(), dim=True)
    
    typer.secho(f"\nShowing {len(docs)} of {len(docs)} memories", fg=get_text_color(), dim=True)
    typer.secho("Use '/memory show <id>' to see full content", fg=get_text_color(), dim=True)
    
    # Show RAG status hint
    if not config.get('rag_enabled', False):
        typer.secho("\n💡 Auto-context is disabled. Enable with '/set rag on'", fg=get_text_color(), dim=True)


def search_memories(query: str):
    """Search memory entries."""
    # Get RAG system (always available for searching)
    from episodic.rag import get_rag_system
    rag = get_rag_system()
    if not rag:
        typer.secho("❌ Failed to initialize memory system", fg=get_error_color())
        return
    
    typer.secho(f"\n🔍 Searching memories for: {query}", fg=get_heading_color())
    
    # Search (conversation memories only)
    results = rag.search(query, n_results=10, source_filter='conversation')
    
    if not results['results']:
        typer.secho("\nNo matching memories found.", fg=get_text_color())
        return
    
    # Filter results by relevance threshold
    relevance_threshold = config.get('memory_relevance_threshold', 0.3)
    filtered_results = [
        r for r in results['results']
        if (r.get('relevance_score') or 0) >= relevance_threshold
    ]
    
    if not filtered_results:
        typer.secho(f"\nNo memories found with relevance >= {relevance_threshold}.", fg=get_text_color())
        typer.secho("Try a different search term or adjust the threshold with '/set memory_relevance_threshold'", fg=get_text_color(), dim=True)
        return
    
    typer.secho(f"\nFound {len(filtered_results)} relevant matches:", fg=get_text_color())
    typer.secho("─" * 50, fg=get_heading_color())
    
    for i, result in enumerate(filtered_results):
        metadata = result.get('metadata', {})
        doc_id = metadata.get('doc_id', 'unknown')
        source = metadata.get('source', 'unknown')
        score = result.get('relevance_score') or 0
        
        # Source icon
        source_icon = {
            'conversation': '💬',
            'file': '📄', 
            'text': '📝',
            'web': '🌐'
        }.get(source, '📌')
        
        # Display result
        typer.secho(f"\n{i+1}. {source_icon} [{doc_id[:8]}] ", fg=get_system_color(), bold=True, nl=False)
        typer.secho(f"(relevance: {score:.2f})", fg=get_text_color(), dim=True)
        
        # Show content
        content = result['content'][:200].strip()
        if len(result['content']) > 200:
            content += "..."
        typer.secho(f"   {content}", fg=get_text_color())
        
        # Show metadata
        if metadata.get('topic'):
            typer.secho(f"   Topic: {metadata['topic']}", fg=get_text_color(), dim=True)
        if metadata.get('filename'):
            typer.secho(f"   File: {metadata['filename']}", fg=get_text_color(), dim=True)


def show_memory(doc_id: str):
    """Show full content of a specific memory entry."""
    # Get RAG system (always available for viewing)
    from episodic.rag import get_rag_system
    rag = get_rag_system()
    if not rag:
        typer.secho("❌ Failed to initialize memory system", fg=get_error_color())
        return
    
    # Handle partial IDs (first 8 chars)
    if len(doc_id) == 8:
        # Search for matching document
        docs = rag.list_documents()
        matches = [d for d in docs if d['doc_id'].startswith(doc_id)]
        if not matches:
            typer.secho(f"\n❌ No memory found with ID starting with: {doc_id}", fg=get_error_color())
            return
        elif len(matches) > 1:
            typer.secho(f"\n⚠️  Multiple memories found starting with: {doc_id}", fg=get_warning_color())
            for doc in matches:
                typer.secho(f"  • {doc['doc_id']}", fg=get_text_color())
            return
        else:
            doc_id = matches[0]['doc_id']
    
    # Get document
    doc = rag.get_document(doc_id)
    if not doc:
        typer.secho(f"\n❌ Memory not found: {doc_id}", fg=get_error_color())
        return
    
    # Display document
    typer.secho(f"\n📚 Memory Entry: {doc_id[:8]}", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    # Metadata
    typer.secho("\nMetadata:", fg=get_system_color(), bold=True)
    typer.secho(f"  Source: {doc.get('source', 'unknown')}", fg=get_text_color())
    typer.secho(f"  Indexed: {doc.get('indexed_at', 'unknown')}", fg=get_text_color())
    typer.secho(f"  Chunks: {doc.get('chunk_count', 0)}", fg=get_text_color())
    typer.secho(f"  Retrieved: {doc.get('retrieval_count', 0)} times", fg=get_text_color())
    
    metadata = doc.get('metadata', {})
    if metadata:
        for key, value in metadata.items():
            if key not in ['source', 'indexed_at']:
                typer.secho(f"  {key.title()}: {value}", fg=get_text_color())
    
    # Content preview (we can't show full content without accessing chunks)
    typer.secho("\nContent Preview:", fg=get_system_color(), bold=True)
    if doc.get('preview'):
        typer.secho(doc['preview'], fg=get_text_color())
    else:
        typer.secho("(Full content stored in chunks)", fg=get_text_color(), dim=True)
    
    # Retrieval history
    if doc.get('last_retrieved'):
        typer.secho(f"\nLast retrieved: {doc['last_retrieved']}", fg=get_text_color(), dim=True)


def forget_command(target: Optional[str] = None, *args):
    """
    Remove memory entries.
    
    Usage:
        /forget <id>               # Forget specific memory
        /forget --contains <text>  # Forget memories containing text
        /forget --source <source>  # Forget memories from source
        /forget --all              # Clear all memories (with confirmation)
    """
    if not target:
        typer.secho("Usage: /forget <id> or /forget --contains <text>", fg=get_error_color())
        return
    
    # Get RAG system (always available for management)
    from episodic.rag import get_rag_system
    rag = get_rag_system()
    if not rag:
        typer.secho("❌ Failed to initialize memory system", fg=get_error_color())
        return
    
    if target == "--all":
        # Clear all memories
        if not typer.confirm("\n⚠️  Delete ALL memories? This cannot be undone."):
            typer.secho("Cancelled.", fg=get_text_color())
            return
        
        # Clear only conversation memories (not user documents)
        count = rag.clear_documents(source_filter='conversation')
        typer.secho(f"\n✅ Removed {count} conversation memories", fg=get_success_color())
        
    elif target == "--contains":
        # Forget memories containing text
        if not args:
            typer.secho("Usage: /forget --contains <text>", fg=get_error_color())
            return
        
        search_text = " ".join(args)
        typer.secho(f"\nSearching for memories containing: {search_text}", fg=get_text_color())
        
        # Search for matching documents
        results = rag.search(search_text, n_results=50)
        if not results['results']:
            typer.secho("No matching memories found.", fg=get_text_color())
            return
        
        # Get unique document IDs
        doc_ids = set()
        for result in results['results']:
            if doc_id := result.get('metadata', {}).get('doc_id'):
                doc_ids.add(doc_id)
        
        typer.secho(f"Found {len(doc_ids)} matching memories.", fg=get_text_color())
        if not typer.confirm("Delete these memories?"):
            typer.secho("Cancelled.", fg=get_text_color())
            return
        
        # Remove documents
        removed = 0
        for doc_id in doc_ids:
            if rag.remove_document(doc_id):
                removed += 1
        
        typer.secho(f"\n✅ Removed {removed} memories", fg=get_success_color())
        
    elif target == "--source":
        # Forget memories from source
        if not args:
            typer.secho("Usage: /forget --source <source>", fg=get_error_color())
            return
        
        source = args[0]
        count = rag.clear_documents(source_filter=source)
        typer.secho(f"\n✅ Removed {count} memories from source: {source}", fg=get_success_color())
        
    else:
        # Forget specific memory by ID
        doc_id = target
        
        # Handle partial IDs
        if len(doc_id) == 8:
            docs = rag.list_documents()
            matches = [d for d in docs if d['doc_id'].startswith(doc_id)]
            if not matches:
                typer.secho(f"\n❌ No memory found with ID starting with: {doc_id}", fg=get_error_color())
                return
            elif len(matches) > 1:
                typer.secho(f"\n⚠️  Multiple memories found starting with: {doc_id}", fg=get_warning_color())
                for doc in matches:
                    typer.secho(f"  • {doc['doc_id']}", fg=get_text_color())
                return
            else:
                doc_id = matches[0]['doc_id']
        
        # Remove document
        if rag.remove_document(doc_id):
            typer.secho(f"\n✅ Removed memory: {doc_id[:8]}", fg=get_success_color())
        else:
            typer.secho(f"\n❌ Memory not found: {doc_id}", fg=get_error_color())


def memory_stats_command():
    """Show memory system statistics."""
    # Get RAG system (always available for stats)
    from episodic.rag import get_rag_system
    rag = get_rag_system()
    if not rag:
        typer.secho("❌ Failed to initialize memory system", fg=get_error_color())
        return
    
    typer.secho("\n📊 Memory System Statistics", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())
    
    # Get statistics
    stats = rag.get_stats()
    
    # General stats
    typer.secho("\nGeneral:", fg=get_system_color(), bold=True)
    typer.secho(f"  Total documents: {stats.get('total_documents', 0)}", fg=get_text_color())
    typer.secho(f"  Total chunks: {stats.get('collection_count', 0)}", fg=get_text_color())
    typer.secho(f"  Avg chunks/doc: {stats.get('avg_chunks_per_doc', 0):.1f}", fg=get_text_color())
    typer.secho(f"  Total retrievals: {stats.get('total_retrievals', 0)}", fg=get_text_color())
    
    # Source distribution
    source_dist = stats.get('source_distribution', {})
    if source_dist:
        typer.secho("\nDocuments by Source:", fg=get_system_color(), bold=True)
        for source, count in source_dist.items():
            icon = {
                'conversation': '💬',
                'file': '📄',
                'text': '📝',
                'web': '🌐'
            }.get(source, '📌')
            typer.secho(f"  {icon} {source}: {count}", fg=get_text_color())
    
    # Storage info
    typer.secho("\nStorage:", fg=get_system_color(), bold=True)
    if 'db_size' in stats:
        size_mb = stats['db_size'] / (1024 * 1024)
        typer.secho(f"  Database size: {size_mb:.1f} MB", fg=get_text_color())
    typer.secho(f"  Embedding model: {stats.get('embedding_model', 'unknown')}", fg=get_text_color())
    
    # Recent activity
    if stats.get('recent_additions'):
        typer.secho("\nRecent Additions:", fg=get_system_color(), bold=True)
        for doc in stats['recent_additions'][:5]:
            time_str = doc.get('indexed_at', 'unknown')[:16]
            source = doc.get('source', 'unknown')
            typer.secho(f"  • {time_str} - {source}", fg=get_text_color())
    
    # Configuration
    typer.secho("\nConfiguration:", fg=get_system_color(), bold=True)
    rag_enabled = config.get('rag_enabled', False)
    typer.secho(f"  Auto-context: {'✓ Active' if rag_enabled else '✗ Disabled'}", 
                fg=get_success_color() if rag_enabled else get_warning_color())
    typer.secho(f"  Auto-enhance: {config.get('rag_auto_enhance', True)}", fg=get_text_color())
    typer.secho(f"  Chunk size: {config.get('rag_chunk_size', 500)}", fg=get_text_color())
    typer.secho(f"  Search results: {config.get('rag_max_results', 5)}", fg=get_text_color())
    
    if not rag_enabled:
        typer.secho("\n💡 Tip: Enable auto-context with '/set rag on'", fg=get_text_color(), dim=True)
        typer.secho("   This will automatically use memories to enhance responses", fg=get_text_color(), dim=True)


def index_recent_conversations(limit: int = 100, failed_only: bool = False):
    """Index conversations into memory for semantic search.

    Uses durable status tracking to determine what needs indexing.

    Args:
        limit: Maximum number of exchanges to index
        failed_only: If True, only retry previously failed indexes
    """
    import asyncio
    from episodic.db import get_unindexed_nodes, get_failed_nodes, get_indexing_stats, get_node
    from episodic.rag_memory_sqlite import memory_rag

    # Get nodes to index based on durable status
    if failed_only:
        node_rows = get_failed_nodes(index_type='conversation', limit=limit)
        node_ids = [r['node_id'] for r in node_rows]
        typer.secho(f"\n🔄 Retrying {len(node_ids)} failed indexes...", fg=get_system_color())
    else:
        node_ids = get_unindexed_nodes(index_type='conversation', limit=limit)
        typer.secho(f"\n🔄 Indexing {len(node_ids)} unindexed exchanges...", fg=get_system_color())

    if not node_ids:
        typer.secho("✅ Nothing to index", fg=get_success_color())
        # Show stats
        stats = get_indexing_stats('conversation')
        if stats:
            typer.secho(f"\nTotal indexed: {stats.get('ok', 0)}, Failed: {stats.get('failed', 0)}",
                        fg=get_text_color(), dim=True)
        return

    # Index each exchange
    ok_count = 0
    fail_count = 0
    first_error = None

    for node_id in node_ids:
        # Get the user node
        user_node = get_node(node_id)
        if not user_node or user_node.get('role') != 'user':
            continue

        # Find the assistant response (child node)
        from episodic.db import get_children
        children = get_children(node_id)
        assistant_node = None
        for child in children:
            if child.get('role') == 'assistant':
                assistant_node = child
                break

        if not assistant_node:
            continue

        try:
            # Direct synchronous call - ChromaDB and SQLite are both sync
            memory_rag.index_exchange(user_node, assistant_node)
            ok_count += 1
        except Exception as e:
            fail_count += 1
            if first_error is None:
                first_error = str(e)
            if config.get("debug"):
                typer.secho(f"  Failed: {node_id[:8]}: {e}", fg=get_error_color(), dim=True)

    # Show first error if any failures
    if first_error:
        typer.secho(f"  First error: {first_error}", fg=get_error_color())

    # Report results
    typer.secho(f"\n✅ Indexed: {ok_count}", fg=get_success_color())
    if fail_count:
        typer.secho(f"❌ Failed: {fail_count}", fg=get_error_color())

    # Show overall stats
    stats = get_indexing_stats('conversation')
    typer.secho(f"\nTotal indexed: {stats.get('ok', 0)}, Failed: {stats.get('failed', 0)}",
                fg=get_text_color(), dim=True)


def clean_poisoned_memories(pattern: Optional[str] = None):
    """Remove poisoned memory entries (recall queries that stored hallucinations).

    By default, removes entries containing recall-query patterns like
    "what did we discuss" which pollute memory with hallucinated responses.

    Args:
        pattern: Optional custom pattern to match. If None, uses default
                 recall-query patterns.
    """
    from episodic.rag_collections import get_multi_collection_rag, CollectionType

    rag = get_multi_collection_rag()

    # Default patterns to clean: recall/meta-queries that pollute memory
    default_patterns = [
        "what did we discuss",
        "what did we talk about",
        "what was that about",
        "remind me what",
        "remember when we",
    ]

    patterns_to_clean = [pattern] if pattern else default_patterns
    total_deleted = 0

    typer.secho("\n🧹 Cleaning poisoned memory entries...", fg=get_heading_color())

    for p in patterns_to_clean:
        deleted = rag.delete_by_content_pattern(p, CollectionType.CONVERSATION)
        if deleted > 0:
            typer.secho(f"  Removed {deleted} entries matching: '{p}'", fg=get_text_color())
            total_deleted += deleted

    if total_deleted > 0:
        typer.secho(f"\n✅ Removed {total_deleted} poisoned memory entries", fg=get_success_color())
        typer.secho("Future recall queries will not be indexed.", fg=get_text_color(), dim=True)
    else:
        typer.secho("\n✅ No poisoned entries found", fg=get_success_color())


def show_indexing_stats():
    """Show detailed indexing statistics."""
    from episodic.db import get_indexing_stats, get_failed_nodes, get_unindexed_nodes

    typer.secho("\n📊 Memory Indexing Status", fg=get_heading_color(), bold=True)
    typer.secho("─" * 50, fg=get_heading_color())

    # Get stats
    stats = get_indexing_stats('conversation')
    ok_count = stats.get('ok', 0)
    failed_count = stats.get('failed', 0)
    pending_count = stats.get('pending', 0)

    # Count unindexed (no status record at all)
    unindexed = get_unindexed_nodes(limit=10000)
    unindexed_count = len(unindexed)

    typer.secho("\nStatus:", fg=get_system_color(), bold=True)
    typer.secho(f"  ✅ Successfully indexed: {ok_count}", fg=get_success_color())
    typer.secho(f"  ❌ Failed: {failed_count}", fg=get_error_color() if failed_count else get_text_color())
    typer.secho(f"  ⏳ Never attempted: {unindexed_count - failed_count}", fg=get_text_color())

    # Show recent failures if any
    if failed_count > 0:
        typer.secho("\nRecent Failures:", fg=get_system_color(), bold=True)
        failed_nodes = get_failed_nodes(limit=5)
        for node in failed_nodes:
            error_preview = node['last_error'][:60] if node['last_error'] else 'Unknown error'
            if node['last_error'] and len(node['last_error']) > 60:
                error_preview += '...'
            typer.secho(f"  • {node['node_id'][:8]}: {error_preview}", fg=get_text_color())
            typer.secho(f"    Attempts: {node['attempts']}, Failed: {node['failed_at'][:16] if node['failed_at'] else 'unknown'}",
                        fg=get_text_color(), dim=True)

    # Hint for retry
    if failed_count > 0:
        typer.secho(f"\n💡 Retry failures with: /memory index --failed", fg=get_text_color(), dim=True)
    elif unindexed_count > failed_count:
        typer.secho(f"\n💡 Index pending with: /memory index", fg=get_text_color(), dim=True)
