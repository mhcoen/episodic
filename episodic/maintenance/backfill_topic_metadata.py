"""
Backfill topic_start_node_id metadata for existing Chroma documents.

SQLite topic_nodes is source of truth. Includes reconciliation reporting.

Run this once after adding topic_start_node_id to conversation indexing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from episodic.db_connection import get_connection
from episodic.db_topic_nodes import get_node_topic

logger = logging.getLogger(__name__)


@dataclass
class BackfillReport:
    """Detailed backfill reconciliation report."""

    total_scanned: int = 0
    already_has_metadata: int = 0
    updated: int = 0
    missing_in_topic_nodes: int = 0  # No topic assignment in SQLite
    conflicts_resolved: int = 0  # Chroma had different value than SQLite
    errors: int = 0
    still_missing_after: int = 0  # Should be 0 if successful


def format_backfill_report(report: BackfillReport) -> str:
    """Human-readable report."""
    lines = [
        "=== Chroma Topic Metadata Backfill Report ===",
        f"Total documents scanned: {report.total_scanned}",
        f"Already had correct metadata: {report.already_has_metadata}",
        f"Updated: {report.updated}",
        f"Conflicts resolved (SQLite wins): {report.conflicts_resolved}",
        f"Missing in topic_nodes (skipped): {report.missing_in_topic_nodes}",
        f"Errors: {report.errors}",
        f"Still missing after backfill: {report.still_missing_after}",
        "",
        "Status: "
        + ("COMPLETE" if report.still_missing_after == 0 else "INCOMPLETE"),
    ]
    return "\n".join(lines)


def backfill_topic_metadata_with_report(dry_run: bool = False) -> BackfillReport:
    """
    Add topic_start_node_id to Chroma documents.

    Source of truth: SQLite topic_nodes table.
    Idempotent and safe to re-run.

    Args:
        dry_run: If True, only report what would be done without making changes.

    Returns:
        BackfillReport with detailed reconciliation information.
    """
    from episodic.rag_collections import CollectionType, get_multi_collection_rag

    report = BackfillReport()

    try:
        rag = get_multi_collection_rag()
        collection = rag.get_collection(CollectionType.CONVERSATION)
    except Exception as e:
        logger.error(f"Could not get Chroma collection: {e}")
        report.errors = 1
        return report

    # Get all documents
    all_docs = collection.get(include=["metadatas", "documents", "embeddings"])

    if not all_docs["ids"]:
        return report

    report.total_scanned = len(all_docs["ids"])

    updates_to_apply = []

    with get_connection() as conn:
        for i, doc_id in enumerate(all_docs["ids"]):
            metadata = all_docs["metadatas"][i] if all_docs.get("metadatas") else {}

            # Get user_id from metadata to look up topic
            user_node_id = metadata.get("user_id", doc_id)

            # Look up topic from SQLite (source of truth)
            sqlite_topic_id = get_node_topic(user_node_id, conn=conn)
            chroma_topic_id = metadata.get("topic_start_node_id")

            # Case 1: Already correct
            if chroma_topic_id and chroma_topic_id == sqlite_topic_id:
                report.already_has_metadata += 1
                continue

            # Case 2: No topic in SQLite
            if not sqlite_topic_id:
                report.missing_in_topic_nodes += 1
                continue

            # Case 3: Conflict - Chroma has different value
            if chroma_topic_id and chroma_topic_id != sqlite_topic_id:
                logger.warning(
                    f"Conflict for {doc_id}: Chroma={chroma_topic_id[:8]}, "
                    f"SQLite={sqlite_topic_id[:8]}. Using SQLite."
                )
                report.conflicts_resolved += 1

            # Queue update
            updates_to_apply.append(
                {
                    "id": doc_id,
                    "index": i,
                    "metadata": {**metadata, "topic_start_node_id": sqlite_topic_id},
                }
            )

    # Apply updates
    if not dry_run and updates_to_apply:
        for update in updates_to_apply:
            try:
                idx = update["index"]
                document = all_docs["documents"][idx] if all_docs.get("documents") else ""

                # Get embedding if available
                embedding = None
                if all_docs.get("embeddings") is not None:
                    emb_data = all_docs["embeddings"][idx]
                    if emb_data is not None:
                        embedding = list(emb_data)

                # Delete and re-add with updated metadata
                collection.delete(ids=[update["id"]])

                if embedding is not None:
                    collection.add(
                        ids=[update["id"]],
                        documents=[document],
                        embeddings=[embedding],
                        metadatas=[update["metadata"]],
                    )
                else:
                    collection.add(
                        ids=[update["id"]],
                        documents=[document],
                        metadatas=[update["metadata"]],
                    )

                report.updated += 1
            except Exception as e:
                logger.error(f"Failed to update {update['id']}: {e}")
                report.errors += 1
    elif dry_run:
        report.updated = len(updates_to_apply)

    # Verify completion (re-scan)
    if not dry_run:
        all_docs_after = collection.get(include=["metadatas"])
        with get_connection() as conn:
            for i, doc_id in enumerate(all_docs_after["ids"]):
                metadata = (
                    all_docs_after["metadatas"][i]
                    if all_docs_after.get("metadatas")
                    else {}
                )
                user_node_id = metadata.get("user_id", doc_id)
                sqlite_topic_id = get_node_topic(user_node_id, conn=conn)
                if sqlite_topic_id and not metadata.get("topic_start_node_id"):
                    report.still_missing_after += 1

    return report


def backfill_topic_metadata() -> Dict[str, Any]:
    """
    Add topic_start_node_id to existing conversation documents in Chroma.

    Returns:
        Dict with counts of updated, skipped, and failed documents

    Note: This is the legacy interface. Use backfill_topic_metadata_with_report()
    for the full reconciliation report.
    """
    report = backfill_topic_metadata_with_report(dry_run=False)

    return {
        "total": report.total_scanned,
        "updated": report.updated,
        "skipped": report.already_has_metadata,
        "failed": report.errors,
        "no_topic": report.missing_in_topic_nodes,
    }


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("Backfilling topic_start_node_id metadata (DRY RUN)...")
    else:
        print("Backfilling topic_start_node_id metadata...")

    report = backfill_topic_metadata_with_report(dry_run=dry_run)
    print()
    print(format_backfill_report(report))

    if report.still_missing_after == 0 and not dry_run:
        print("\nBackfill complete!")
    elif dry_run:
        print(f"\nDry run complete. Would update {report.updated} documents.")
    else:
        print("\nBackfill incomplete - some documents still missing metadata.")
