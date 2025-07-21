"""Add preview field to rag_documents table."""

import sqlite3
from typing import Any


def up(conn: sqlite3.Connection) -> None:
    """Add preview column to rag_documents table."""
    cursor = conn.cursor()
    
    # Add preview column
    cursor.execute('''
        ALTER TABLE rag_documents ADD COLUMN preview TEXT
    ''')
    
    conn.commit()


def down(conn: sqlite3.Connection) -> None:
    """Remove preview column from rag_documents table."""
    # SQLite doesn't support DROP COLUMN directly
    # Would need to recreate table without the column
    pass