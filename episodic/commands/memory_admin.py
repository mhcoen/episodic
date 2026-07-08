"""Memory admin operations: indexing, cleaning, indexing stats.

Split out of commands/memory.py to keep it under the size limit. Re-imported
there so `episodic.commands.memory.<fn>` (dispatch + test patch targets) still
resolves.
"""

from typing import Optional

import typer

from episodic.config import config
from episodic.configuration import (
    get_heading_color, get_text_color, get_system_color,
    get_error_color, get_warning_color, get_success_color,
)


def index_recent_conversations(limit: int = 100, failed_only: bool = False):
    """Index conversations into memory for semantic search.

    Uses durable status tracking to determine what needs indexing.
    Processes ALL unindexed nodes in batches until complete.

    Args:
        limit: Batch size for processing (default 100)
        failed_only: If True, only retry previously failed indexes
    """
    from episodic.db import get_unindexed_nodes, get_failed_nodes, get_indexing_stats, get_node
    from episodic.db import get_children, update_indexing_status
    from episodic.rag_memory_sqlite import memory_rag

    # Track totals across all batches
    total_ok = 0
    total_fail = 0
    total_skip = 0
    first_error = None

    # Process in batches until nothing left
    batch_num = 0
    while True:
        batch_num += 1

        # Get next batch of nodes to index
        if failed_only:
            node_rows = get_failed_nodes(index_type='conversation', limit=limit)
            node_ids = [r['node_id'] for r in node_rows]
        else:
            node_ids = get_unindexed_nodes(index_type='conversation', limit=limit)

        if not node_ids:
            break  # All done

        if batch_num == 1:
            # Show initial message
            mode = "failed indexes" if failed_only else "unindexed exchanges"
            typer.secho(f"\n🔄 Indexing {mode}...", fg=get_system_color())

        # Index each exchange in this batch
        for node_id in node_ids:
            # Get the user node
            user_node = get_node(node_id)
            if not user_node or user_node.get('role') != 'user':
                # Mark as skipped so it doesn't keep appearing in unindexed list
                update_indexing_status(node_id, 'conversation', status='skipped', error='Not a user node')
                total_skip += 1
                continue

            # Find the assistant response (child node)
            children = get_children(node_id)
            assistant_node = None
            for child in children:
                if child.get('role') == 'assistant':
                    assistant_node = child
                    break

            if not assistant_node:
                # Mark as skipped - user message without assistant response
                update_indexing_status(node_id, 'conversation', status='skipped', error='No assistant response')
                total_skip += 1
                continue

            try:
                # Direct synchronous call - ChromaDB and SQLite are both sync
                memory_rag.index_exchange(user_node, assistant_node)
                total_ok += 1
            except Exception as e:
                error_str = str(e)
                # If content already indexed (duplicate hash), treat as success
                if 'UNIQUE constraint failed' in error_str and 'content_hash' in error_str:
                    # Content is already in index under different node_id - mark as ok
                    update_indexing_status(node_id, 'conversation', status='ok')
                    total_ok += 1
                else:
                    total_fail += 1
                    if first_error is None:
                        first_error = error_str
                    if config.get("debug"):
                        typer.secho(f"  Failed: {node_id[:8]}: {e}", fg=get_error_color(), dim=True)

    # After all batches complete
    if batch_num == 1 and total_ok == 0 and total_skip == 0 and total_fail == 0:
        typer.secho("✅ Nothing to index", fg=get_success_color())
        stats = get_indexing_stats('conversation')
        if stats:
            typer.secho(f"\nTotal indexed: {stats.get('ok', 0)}, Failed: {stats.get('failed', 0)}",
                        fg=get_text_color(), dim=True)
        return

    # Show first error if any failures
    if first_error:
        typer.secho(f"  First error: {first_error}", fg=get_error_color())

    # Report results
    typer.secho(f"\n✅ Indexed: {total_ok}", fg=get_success_color())
    if total_skip:
        typer.secho(f"⏭️  Skipped: {total_skip} (no assistant response)", fg=get_text_color(), dim=True)
    if total_fail:
        typer.secho(f"❌ Failed: {total_fail}", fg=get_error_color())

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
    from episodic.db_connection import get_connection

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

    # Delete from both ChromaDB and SQLite for each pattern
    for p in patterns_to_clean:
        # Delete from ChromaDB
        chroma_deleted = rag.delete_by_content_pattern(p, CollectionType.CONVERSATION)

        # Delete from SQLite rag_documents table (match on preview column)
        with get_connection() as conn:
            cursor = conn.cursor()
            # First get doc_ids to delete from ChromaDB by ID too
            cursor.execute(
                "SELECT doc_id FROM rag_documents WHERE source = 'conversation' AND preview LIKE ?",
                (f'%{p}%',)
            )
            doc_ids = [row[0] for row in cursor.fetchall()]

            # Delete matching docs from ChromaDB by ID (in case content search missed them)
            for doc_id in doc_ids:
                try:
                    collection = rag.get_collection(CollectionType.CONVERSATION)
                    collection.delete(ids=[doc_id])
                except Exception:
                    pass  # May already be deleted

            # Delete from SQLite
            cursor.execute(
                "DELETE FROM rag_documents WHERE source = 'conversation' AND preview LIKE ?",
                (f'%{p}%',)
            )
            sqlite_deleted = cursor.rowcount
            conn.commit()

        deleted = max(chroma_deleted, sqlite_deleted)  # Report the higher count
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
    skipped_count = stats.get('skipped', 0)

    # Count unindexed (no status record at all, or failed)
    unindexed = get_unindexed_nodes(limit=10000)
    unindexed_count = len(unindexed)
    never_attempted = unindexed_count - failed_count

    typer.secho("\nStatus:", fg=get_system_color(), bold=True)
    typer.secho(f"  ✅ Successfully indexed: {ok_count}", fg=get_success_color())
    typer.secho(f"  ❌ Failed: {failed_count}", fg=get_error_color() if failed_count else get_text_color())
    if skipped_count > 0:
        typer.secho(f"  ⏭️  Skipped (no response): {skipped_count}", fg=get_text_color(), dim=True)
    if never_attempted > 0:
        typer.secho(f"  ⏳ Never attempted: {never_attempted}", fg=get_text_color())

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
    elif never_attempted > 0:
        typer.secho(f"\n💡 Index pending with: /memory index", fg=get_text_color(), dim=True)
