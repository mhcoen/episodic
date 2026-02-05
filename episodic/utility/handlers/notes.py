"""
Notes Handler.

Handles note-taking utility commands:
- note_add: Create a new note
- note_list: List recent notes
- note_search: Search notes by text
- note_delete: Delete a note
"""

import uuid
import sqlite3
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

from ..types import UtilityQuery, UtilityResult


def _persist_note(conn: sqlite3.Connection, note_id: str, text: str) -> None:
    """Insert note into database."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO notes (id, text, created_at)
        VALUES (?, ?, ?)
    """, (
        note_id,
        text,
        int(time.time()),
    ))
    conn.commit()


def _get_notes(
    conn: sqlite3.Connection,
    limit: int = 10,
    search_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get notes from database."""
    cursor = conn.cursor()

    if search_text:
        cursor.execute("""
            SELECT id, text, created_at
            FROM notes
            WHERE text LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{search_text}%", limit))
    else:
        cursor.execute("""
            SELECT id, text, created_at
            FROM notes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

    notes = []
    for row in cursor.fetchall():
        notes.append({
            "id": row[0],
            "text": row[1],
            "created_at": row[2],
        })

    return notes


def _get_note_by_id(conn: sqlite3.Connection, note_id: str) -> Optional[Dict[str, Any]]:
    """Get note by ID."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, text, created_at
        FROM notes
        WHERE id = ?
    """, (note_id,))

    row = cursor.fetchone()
    if row is None:
        return None

    return {
        "id": row[0],
        "text": row[1],
        "created_at": row[2],
    }


def _delete_note(conn: sqlite3.Connection, note_id: str) -> bool:
    """Delete note by ID."""
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    return cursor.rowcount > 0


def _format_note_time(timestamp: int, user_tz: str = "America/Chicago") -> str:
    """Format note timestamp for display."""
    tz = ZoneInfo(user_tz)
    dt = datetime.fromtimestamp(timestamp, tz=tz)
    now = datetime.now(tz)

    if dt.date() == now.date():
        return dt.strftime("%I:%M %p").lstrip("0")
    elif (now - dt).days < 7:
        return dt.strftime("%A %I:%M %p").lstrip("0")
    else:
        return dt.strftime("%b %d")


def handle_note_add(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle note_add command.

    Args in query:
        text: The note content
    """
    text = query.args.get("text", "").strip()

    if not text:
        return UtilityResult.error("missing_text", "No note text provided")

    if conn is None:
        return UtilityResult.error("no_database", "Notes require database connection")

    # Generate ID and save
    note_id = str(uuid.uuid4())
    _persist_note(conn, note_id, text)

    # Truncate for display if long
    display_text = text if len(text) <= 50 else text[:47] + "..."

    return UtilityResult.ok(
        display=f"Note saved: {display_text}",
        speech="Note saved",
        _command="note_add",
        note_id=note_id,
        text=text,
    )


def handle_note_list(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle note_list command.

    Args in query:
        limit: Maximum number of notes to return (default 10)
    """
    if conn is None:
        return UtilityResult.error("no_database", "Notes require database connection")

    limit = query.args.get("limit", 10)
    notes = _get_notes(conn, limit=limit)

    if not notes:
        return UtilityResult.ok(
            display="No notes",
            speech="You have no notes",
            notes=[],
        )

    # Build display
    lines = []
    for i, note in enumerate(notes, 1):
        text = note["text"]
        if len(text) > 60:
            text = text[:57] + "..."
        time_str = _format_note_time(note["created_at"], user_tz)
        lines.append(f"  {i}. {text} ({time_str})")

    display = "Notes:\n" + "\n".join(lines)

    if len(notes) == 1:
        speech = f"You have one note: {notes[0]['text'][:50]}"
    else:
        speech = f"You have {len(notes)} notes"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        notes=[{
            "id": n["id"],
            "text": n["text"],
            "created_at": n["created_at"],
        } for n in notes],
    )


def handle_note_search(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """
    Handle note_search command.

    Args in query:
        query_text: Text to search for
        limit: Maximum results (default 10)
    """
    if conn is None:
        return UtilityResult.error("no_database", "Notes require database connection")

    search_text = query.args.get("query_text", "").strip()

    if not search_text:
        return UtilityResult.error("missing_query", "No search text provided")

    limit = query.args.get("limit", 10)
    notes = _get_notes(conn, limit=limit, search_text=search_text)

    if not notes:
        return UtilityResult.ok(
            display=f"No notes matching '{search_text}'",
            speech=f"No notes found for {search_text}",
            notes=[],
            query_text=search_text,
        )

    # Build display
    lines = []
    for i, note in enumerate(notes, 1):
        text = note["text"]
        if len(text) > 60:
            text = text[:57] + "..."
        time_str = _format_note_time(note["created_at"], user_tz)
        lines.append(f"  {i}. {text} ({time_str})")

    display = f"Notes matching '{search_text}':\n" + "\n".join(lines)

    if len(notes) == 1:
        speech = f"Found one note: {notes[0]['text'][:50]}"
    else:
        speech = f"Found {len(notes)} notes"

    return UtilityResult.ok(
        display=display,
        speech=speech,
        notes=[{
            "id": n["id"],
            "text": n["text"],
            "created_at": n["created_at"],
        } for n in notes],
        query_text=search_text,
    )


def handle_note_delete(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
) -> UtilityResult:
    """
    Handle note_delete command.

    Args in query:
        note_id: ID of note to delete
        index: Index from recent list (1-based)
    """
    if conn is None:
        return UtilityResult.error("no_database", "Notes require database connection")

    note_id = query.args.get("note_id")
    index = query.args.get("index")

    if note_id:
        note = _get_note_by_id(conn, note_id)
    elif index:
        # Get note by index in recent list
        notes = _get_notes(conn, limit=index)
        if len(notes) >= index:
            note = notes[index - 1]
        else:
            note = None
    else:
        # Delete most recent
        notes = _get_notes(conn, limit=1)
        note = notes[0] if notes else None

    if note is None:
        return UtilityResult.error("note_not_found", "Note not found")

    deleted = _delete_note(conn, note["id"])

    if not deleted:
        return UtilityResult.error("delete_failed", "Could not delete note")

    text_preview = note["text"][:30] + "..." if len(note["text"]) > 30 else note["text"]

    return UtilityResult.ok(
        display=f"Deleted note: {text_preview}",
        speech="Note deleted",
        note_id=note["id"],
    )


# Command routing for note category
NOTE_HANDLERS = {
    "note_add": handle_note_add,
    "note_list": handle_note_list,
    "note_search": handle_note_search,
    "note_delete": handle_note_delete,
}


def dispatch_note_command(
    query: UtilityQuery,
    conn: Optional[sqlite3.Connection] = None,
    user_tz: str = "America/Chicago",
) -> UtilityResult:
    """Dispatch a note category command to the appropriate handler."""
    handler = NOTE_HANDLERS.get(query.command)

    if handler is None:
        return UtilityResult.error(
            "unknown_command",
            f"Unknown note command: {query.command}"
        )

    if query.command in ("note_list", "note_search"):
        return handler(query, conn, user_tz)

    return handler(query, conn)
