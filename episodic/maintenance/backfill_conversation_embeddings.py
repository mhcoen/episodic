"""
Backfill missing conversation embeddings in ChromaDB.

This is needed when:
1. enable_topic_reactivation is turned on for an existing database
2. Conversations were stored before memory_rag was enabled
3. ChromaDB was corrupted or reset

Usage:
    from episodic.maintenance.backfill_conversation_embeddings import backfill_embeddings
    report = backfill_embeddings()
    print(report)

For incremental indexing (O(1) on enable):
    from episodic.maintenance.backfill_conversation_embeddings import backfill_embeddings_incremental
    report = backfill_embeddings_incremental()
    print(report)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

from episodic.db_connection import get_connection
from episodic.db_checkpoint import (
    get_embedding_checkpoint,
    set_embedding_checkpoint,
    get_nodes_after_checkpoint,
    get_max_node_rowid,
    ensure_configuration_table
)
from episodic.rag_collections import get_multi_collection_rag, CollectionType

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBackfillReport:
    """Report of embedding backfill operation."""

    total_nodes: int = 0
    already_indexed: int = 0
    newly_indexed: int = 0
    errors: int = 0

    def __str__(self) -> str:
        return (
            f"=== Embedding Backfill Report ===\n"
            f"Total nodes in database: {self.total_nodes}\n"
            f"Already indexed in Chroma: {self.already_indexed}\n"
            f"Newly indexed: {self.newly_indexed}\n"
            f"Errors: {self.errors}\n"
            f"Coverage: {(self.already_indexed + self.newly_indexed) / max(self.total_nodes, 1) * 100:.1f}%"
        )


def check_embedding_coverage() -> Tuple[int, int, List[str]]:
    """
    Check how many nodes have embeddings in ChromaDB.

    Returns:
        Tuple of (total_nodes, indexed_count, missing_node_ids)
    """
    rag = get_multi_collection_rag()
    collection = rag.get_collection(CollectionType.CONVERSATION)

    missing_ids = []
    indexed_count = 0

    with get_connection() as conn:
        cursor = conn.execute("""
            SELECT id FROM nodes
            WHERE role IN ('user', 'assistant')
            AND content IS NOT NULL AND content != ''
        """)

        all_ids = [row[0] for row in cursor.fetchall()]
        total_nodes = len(all_ids)

        # Check in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i:i+batch_size]
            result = collection.get(ids=batch)
            found_ids = set(result.get('ids', []))

            for node_id in batch:
                if node_id in found_ids:
                    indexed_count += 1
                else:
                    missing_ids.append(node_id)

    return total_nodes, indexed_count, missing_ids


def backfill_embeddings(dry_run: bool = False) -> EmbeddingBackfillReport:
    """
    Backfill missing conversation embeddings.

    Args:
        dry_run: If True, only report what would be done without making changes

    Returns:
        EmbeddingBackfillReport with details of the operation
    """
    report = EmbeddingBackfillReport()

    rag = get_multi_collection_rag()
    collection = rag.get_collection(CollectionType.CONVERSATION)

    with get_connection() as conn:
        # Get all conversation nodes
        cursor = conn.execute("""
            SELECT id, role, content, rowid
            FROM nodes
            WHERE role IN ('user', 'assistant')
            AND content IS NOT NULL AND content != ''
            ORDER BY rowid
        """)

        nodes_to_index = []

        for row in cursor.fetchall():
            node_id, role, content, rowid = row
            report.total_nodes += 1

            # Check if already indexed
            result = collection.get(ids=[node_id])
            if len(result.get('ids', [])) > 0:
                report.already_indexed += 1
            else:
                nodes_to_index.append({
                    'id': node_id,
                    'role': role,
                    'content': content,
                    'rowid': rowid
                })

        if dry_run:
            report.newly_indexed = len(nodes_to_index)
            logger.info(f"Dry run: would index {len(nodes_to_index)} nodes")
            return report

        # Batch add to Chroma
        batch_size = 50
        for i in range(0, len(nodes_to_index), batch_size):
            batch = nodes_to_index[i:i+batch_size]

            ids = [n['id'] for n in batch]
            documents = [n['content'] for n in batch]
            metadatas = [
                {"role": n['role'], "source": "conversation", "rowid": n['rowid']}
                for n in batch
            ]

            try:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                report.newly_indexed += len(batch)
                logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch)} nodes")
            except Exception as e:
                report.errors += len(batch)
                logger.error(f"Error indexing batch: {e}")

    return report


def needs_backfill() -> Tuple[bool, int]:
    """
    Check if the database needs embedding backfill.

    Returns:
        Tuple of (needs_backfill, missing_count)
    """
    total, indexed, missing = check_embedding_coverage()
    return len(missing) > 0, len(missing)


def backfill_embeddings_incremental(
    batch_size: int = 50,
    progress_callback: Optional[callable] = None
) -> EmbeddingBackfillReport:
    """
    Incrementally backfill conversation embeddings using checkpoint tracking.

    This is O(new_nodes) instead of O(total_nodes) - it only processes
    nodes added since the last checkpoint. On first run, it indexes everything
    once and sets the checkpoint.

    Args:
        batch_size: Number of nodes to index per batch
        progress_callback: Optional callback(indexed, total) for progress updates

    Returns:
        EmbeddingBackfillReport with details of the operation
    """
    report = EmbeddingBackfillReport()

    # Ensure configuration table exists
    ensure_configuration_table()

    # Get current checkpoint
    checkpoint = get_embedding_checkpoint()
    max_rowid = get_max_node_rowid()

    logger.debug(f"Incremental backfill: checkpoint={checkpoint}, max_rowid={max_rowid}")

    # If checkpoint >= max_rowid, nothing to do
    if checkpoint >= max_rowid:
        logger.info("No new nodes to index (checkpoint is current)")
        return report

    # Get nodes after checkpoint
    new_nodes = get_nodes_after_checkpoint(checkpoint)

    if not new_nodes:
        # Update checkpoint to current max anyway
        set_embedding_checkpoint(max_rowid)
        logger.info("No indexable nodes after checkpoint")
        return report

    report.total_nodes = len(new_nodes)
    logger.info(f"Found {len(new_nodes)} new nodes to index")

    # Get the RAG collection
    rag = get_multi_collection_rag()
    collection = rag.get_collection(CollectionType.CONVERSATION)

    # Batch add to Chroma
    highest_rowid = checkpoint
    for i in range(0, len(new_nodes), batch_size):
        batch = new_nodes[i:i + batch_size]

        ids = [n['id'] for n in batch]
        documents = [n['content'] for n in batch]
        metadatas = [
            {"role": n['role'], "source": "conversation", "rowid": n['rowid']}
            for n in batch
        ]

        try:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            report.newly_indexed += len(batch)

            # Track highest rowid in this batch
            batch_max_rowid = max(n['rowid'] for n in batch)
            highest_rowid = max(highest_rowid, batch_max_rowid)

            logger.debug(f"Indexed batch {i // batch_size + 1}: {len(batch)} nodes")

            if progress_callback:
                progress_callback(report.newly_indexed, report.total_nodes)

        except Exception as e:
            report.errors += len(batch)
            logger.error(f"Error indexing batch: {e}")

    # Update checkpoint to the highest indexed rowid
    set_embedding_checkpoint(highest_rowid)
    logger.info(f"Updated embedding checkpoint to {highest_rowid}")

    return report


def needs_incremental_backfill() -> Tuple[bool, int]:
    """
    Check if there are new nodes to index using checkpoint tracking.

    This is O(1) - just compares checkpoint vs max rowid.

    Returns:
        Tuple of (needs_backfill, estimated_new_count)
    """
    ensure_configuration_table()

    checkpoint = get_embedding_checkpoint()
    max_rowid = get_max_node_rowid()

    if checkpoint >= max_rowid:
        return False, 0

    # Estimate count (could be fewer if some nodes don't have content)
    estimated = max_rowid - checkpoint
    return True, estimated
