"""Rollback command: delete nodes after a given point and reset state."""

import sqlite3
import typer

from episodic.configuration import (
    get_error_color, get_success_color, get_text_color,
    get_warning_color, get_heading_color,
)


def rollback_command(ref: str = None):
    """Roll back conversation to a specific node.

    Deletes all nodes after the target node, cleans up topic
    boundaries and KG data for deleted nodes, and resets the
    conversation head and KG high-water mark.

    Usage:
        /rollback <short_id>   Roll back to node with short_id
    """
    if not ref:
        typer.secho("Usage: /rollback <short_id>", fg=get_text_color())
        typer.secho(
            "  Deletes all nodes after the specified node and resets state.",
            fg=get_text_color(), dim=True,
        )
        return

    from episodic.db_nodes import get_node, set_head, resolve_node_ref
    from episodic.db_connection import get_connection
    from episodic.conversation import conversation_manager

    # Resolve the reference to a UUID
    node_uuid = resolve_node_ref(ref)
    if not node_uuid:
        typer.secho(f"Node not found: {ref}", fg=get_error_color())
        return

    # Get full node info to find its rowid
    node = get_node(node_uuid)
    if not node:
        typer.secho(f"Node not found: {ref}", fg=get_error_color())
        return

    short_id = node.get('short_id', ref)

    with get_connection() as conn:
        # Get the rowid for this node
        row = conn.execute(
            "SELECT rowid FROM nodes WHERE id = ?", (node_uuid,)
        ).fetchone()
        if not row:
            typer.secho(f"Cannot resolve rowid for node {ref}", fg=get_error_color())
            return
        target_rowid = row[0]

        # Count nodes that will be deleted
        count_row = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE rowid > ?", (target_rowid,)
        ).fetchone()
        delete_count = count_row[0] if count_row else 0

        if delete_count == 0:
            typer.secho("No nodes to delete (already at latest).", fg=get_text_color())
            return

        typer.secho(
            f"Rolling back to node {short_id} (rowid {target_rowid})...",
            fg=get_heading_color(),
        )
        typer.secho(
            f"  Deleting {delete_count} nodes after rowid {target_rowid}",
            fg=get_warning_color(),
        )

        # Collect UUIDs of nodes to delete (for topic/KG cleanup)
        rows = conn.execute(
            "SELECT id, rowid FROM nodes WHERE rowid > ?", (target_rowid,)
        ).fetchall()
        deleted_uuids = {r[0] for r in rows}
        deleted_rowids = {r[1] for r in rows}

        # 1. Delete the nodes themselves
        conn.execute("DELETE FROM nodes WHERE rowid > ?", (target_rowid,))

        # 2. Clean up topic boundaries referencing deleted nodes
        _cleanup_topics(conn, deleted_uuids)

        # 3. Clean up KG data for deleted nodes
        kg_cleaned = _cleanup_kg(conn, deleted_rowids, target_rowid)

        # 4. Reset conversation head
        set_head(node_uuid)
        conversation_manager.set_current_node_id(node_uuid)

        conn.commit()

        typer.secho(
            f"Rolled back to node {short_id}:",
            fg=get_success_color(), bold=True,
        )
        typer.secho(f"  {delete_count} nodes deleted", fg=get_text_color())
        if kg_cleaned:
            typer.secho(f"  KG data cleaned, HWM reset to {target_rowid}", fg=get_text_color())


def _cleanup_topics(conn, deleted_uuids: set):
    """Remove topic data referencing deleted nodes."""
    if not deleted_uuids:
        return

    # Delete from topic_nodes where node_id is a deleted UUID
    try:
        placeholders = ','.join('?' * len(deleted_uuids))
        conn.execute(
            f"DELETE FROM topic_nodes WHERE node_id IN ({placeholders})",
            list(deleted_uuids),
        )
    except sqlite3.OperationalError:
        pass

    # Delete topics whose start_node_id is a deleted UUID
    try:
        placeholders = ','.join('?' * len(deleted_uuids))
        conn.execute(
            f"DELETE FROM topics WHERE start_node_id IN ({placeholders})",
            list(deleted_uuids),
        )
    except sqlite3.OperationalError:
        pass

    # Update topics whose end_node_id references a deleted node
    # (set end_node_id to NULL so the topic stays open)
    try:
        placeholders = ','.join('?' * len(deleted_uuids))
        conn.execute(
            f"UPDATE topics SET end_node_id = NULL "
            f"WHERE end_node_id IN ({placeholders})",
            list(deleted_uuids),
        )
    except sqlite3.OperationalError:
        pass


def _cleanup_kg(conn, deleted_rowids: set, target_rowid: int) -> bool:
    """Remove KG data for deleted nodes and reset HWM.

    Returns True if KG tables existed and were cleaned.
    """
    if not deleted_rowids:
        return False

    # Check if KG tables exist
    try:
        conn.execute("SELECT 1 FROM kg_entities LIMIT 1")
    except sqlite3.OperationalError:
        return False

    placeholders = ','.join('?' * len(deleted_rowids))
    rowid_list = list(deleted_rowids)

    # Delete patches for deleted nodes
    try:
        conn.execute(
            f"DELETE FROM kg_patches WHERE node_id IN ({placeholders})",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Delete assertions for deleted nodes
    try:
        conn.execute(
            f"DELETE FROM kg_assertions WHERE node_id IN ({placeholders})",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Delete mentions for deleted nodes
    try:
        conn.execute(
            f"DELETE FROM kg_mentions WHERE node_id IN ({placeholders})",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Delete edges created by deleted nodes
    try:
        conn.execute(
            f"DELETE FROM kg_edges WHERE created_node_id IN ({placeholders})",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Delete entity aliases created by deleted nodes
    try:
        conn.execute(
            f"DELETE FROM kg_entity_aliases "
            f"WHERE created_node_id IN ({placeholders})",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Delete entities created by deleted nodes (except user:self)
    try:
        conn.execute(
            f"DELETE FROM kg_entities "
            f"WHERE created_node_id IN ({placeholders}) "
            f"AND canonical_key != 'user:self'",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Remove deleted nodes from skiplist
    try:
        conn.execute(
            f"DELETE FROM kg_skiplist WHERE node_id IN ({placeholders})",
            rowid_list,
        )
    except sqlite3.OperationalError:
        pass

    # Reset HWM to target_rowid
    try:
        conn.execute(
            "UPDATE kg_state SET value = ? WHERE key = 'high_water_mark'",
            (str(target_rowid),),
        )
    except sqlite3.OperationalError:
        pass

    return True
